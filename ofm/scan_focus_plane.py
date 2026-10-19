import logging
import math
import statistics
import threading
import time
from typing import List, Optional, Tuple

import labthings_fastapi as lt
from pydantic import Field

from openflexure_microscope_server.things.camera import BaseCamera
from openflexure_microscope_server.things.smooth_af_core import (
    CurvePoint,
    fit_log_gaussian,
)
from openflexure_microscope_server.things.smooth_autofocus import (
    NoFocusFoundError,
    SmoothAutofocusParams,
    StreamSharpnessMonitor,
    SmoothAutofocusThing,
    _EncoderBurst,
    _probe_regime,
)
from openflexure_microscope_server.things.stage import BaseStage

LOGGER = logging.getLogger(__name__)

NEAR_SPAN = 30.0
MAX_DEPTH = 90.0
SWEEP_VELOCITY = 80.0
TRANSIT_VELOCITY = 9000.0


def _sweep_velocity() -> float:
    """Engine-calibrated near-sweep speed (12 u/s per u of FWHM)."""
    from openflexure_microscope_server.things.smooth_autofocus import (
        SmoothAutofocusThing,
    )

    return SmoothAutofocusThing._cal_near_velocity(
        object.__new__(SmoothAutofocusThing)
    )


GRID_STEP = 350.0
GRID = 5
MAX_PEAK_PTP = 1280.0
MAX_NEIGHBOUR_JUMP = 375.0
MIN_FITS = 4
STATIC_SHARP_FRAC = 0.6
DENSIFY_JUMP = 20.0
DENSIFY_MAX = 2
ENVELOPE_XY = 700.0
MAX_RADIUS = 700.0
COAST_GRID = 32
COAST_CELL = 2.0 * MAX_RADIUS / COAST_GRID
EDGE_WALK_STEP = 5.0
EDGE_WALK_MAX_V = 200.0
EDGE_WALK_POOL = 40.0
MESH_CELLS = 128
MESH_CELL = 2.0 * MAX_RADIUS / MESH_CELLS
Z_ANCHOR_EXT = 200.0
POINTS_PER_LEG_MIN = 6
FIT_WINDOW = 0.7
STATIC_FRAMES = 8
MARKER_STALL_LIMIT = 3
MARKER_WAIT_S = 5.0
MAX_Z_EXCURSION = 200.0
CANYON_REACHES = (60.0, 120.0, 170.0)
CANYON_REACH_MAX = 170.0


class SampleNotFoundError(RuntimeError):
    """No usable focus surface could be measured."""


class ScanFocusPlaneThing(lt.Thing):
    """Scanning-side focus-surface pass and SAMPLE_MESH handoff.

    Separate from the interactive autofocus thing on purpose: this is the
    entry point scanning will launch. It runs the proven continuous-sweep
    mechanics in miniature: a slow +-30 z-unit sweep per point (for use
    when the stage is already in focus), the same log-gaussian shape gate,
    and telemetry-labelled legs. Every move is relative; nothing here
    assumes a coordinate frame.
    """

    _class_settings = {"validate_properties_on_set": True}
    _cam: BaseCamera = lt.thing_slot()
    _stage: BaseStage = lt.thing_slot()
    _af: SmoothAutofocusThing = lt.thing_slot("smooth_autofocus")
    _PASS_BUSY: bool = False
    _PASS_COOL_TIMER: Optional[threading.Timer] = None
    _PASS_COOL_DELAY: float = 3.0

    def _cancel_pass_cooler(self) -> None:
        t = type(self)._PASS_COOL_TIMER
        if t is not None:
            t.cancel()
            type(self)._PASS_COOL_TIMER = None

    def _arm_pass_cooler(self) -> None:
        self._cancel_pass_cooler()
        t = threading.Timer(self._PASS_COOL_DELAY, self._quiet_pass_camera)
        t.daemon = True
        type(self)._PASS_COOL_TIMER = t
        t.start()

    def _quiet_pass_camera(self) -> None:
        """10s after the last pass: sensor to 20fps. Encoders were already
        restarted by the burst exit; fps is deliberately left high across
        back-to-back runs to avoid the ~0.4s sensor re-clock."""
        type(self)._PASS_COOL_TIMER = None
        if getattr(type(self), "_PASS_BUSY", False):
            return
        try:
            with self._cam._streaming_picamera() as picam:
                picam.set_controls({"FrameRate": 20.0})
        except Exception:
            LOGGER.warning("pass cooler failed", exc_info=True)

    def _hw_steps_param(self, delta_hw: float) -> int:
        inv = getattr(self._stage, "axis_inverted", {}) or {}
        return (
            -int(round(delta_hw)) if inv.get("z", False) else int(round(delta_hw))
        )

    def _stage_z(self) -> float:
        try:
            v = self._last_hw
            return float(v)
        except Exception:
            return float("nan")

    def _send_gcode(self, script: str) -> dict:
        sender = getattr(self._stage, "_send_gcode_script", None)
        if sender is not None:
            return sender(script=script)
        import httpx

        resp = httpx.post(
            "http://127.0.0.1:7125/printer/gcode/script",
            timeout=10.0,
            json={"script": script},
        )
        resp.raise_for_status()
        return resp.json()

    def _still_rising(self, pts: list) -> bool:
        """Claw condition: bottom of the curve still climbing toward focus.

        True when the last quarter of the curve keeps rising overall and
        the final frames are not below the segment median — i.e. the peak
        lies deeper than the current sweep floor.
        """
        seg = pts[max(0, len(pts) - len(pts) // 4):]
        if len(seg) < 4:
            return False
        rising = seg[-1][2] > seg[0][2] * 1.08
        not_falling = seg[-1][2] >= statistics.median(
            [p[2] for p in seg]
        ) * 0.97
        return rising and not_falling

    def _still_rising_top(self, pts: list) -> bool:
        """Peak-above-ceiling test: the curve's maximum sharpness sits in
        the top quarter and the topmost samples hold at/near it, i.e. the
        focus peak lies at or beyond the sweep's upper edge. A rising
        approach ramp below the peak must not trigger extension - on a
        downward sweep that ramp always 'rises' toward the peak."""
        if len(pts) < 12:
            return False
        s_max = max(p[2] for p in pts)
        if s_max <= 0:
            return False
        top_q = sorted(pts, key=lambda p: -p[0])[
            : max(4, len(pts) // 4)
        ]
        if max(p[2] for p in top_q) < s_max * 0.97:
            return False
        edge = statistics.median(
            [p[2] for p in sorted(top_q, key=lambda p: -p[0])[:3]]
        )
        return edge >= s_max * 0.9

    @staticmethod
    def _sharp_at(monitor, mark: int, t_lo: float, t_hi: float) -> Optional[float]:
        """Median sharpness over a time window from the live monitor."""
        with monitor._hook_lock:
            vals = [
                s for t, _c, s, _p, _b in zip(
                    monitor.times[mark:],
                    monitor.chroma[mark:],
                    monitor.sharpness[mark:],
                    monitor.pooled[mark:],
                    monitor.brenner[mark:],
                ) if t_lo <= t <= t_hi
            ]
        if not vals:
            return None
        return float(statistics.median(vals))

    def _static_verify(
            self,
            monitor: StreamSharpnessMonitor,
            z_start: float,
            ref_s: float,
    ) -> Tuple[Optional[float], str]:
        """Cheap predicted-z check: settle, take frames, gate on pooled +
        sharpness floor vs the pass reference. No motion, no sweep.

        Returns (z_start, "static_ok") on pass, (None, reason) on fail.
        """
        mark = len(monitor.times)
        monitor.wait_frames(STATIC_FRAMES, mark, timeout=2.0)
        pre = self._frames_since5(monitor, mark)
        if len(pre) < STATIC_FRAMES:
            return None, "static_noframes"
        pooled = statistics.median([f[3] for f in pre])
        sharp = statistics.median([f[2] for f in pre])
        self._last_point_diag = {
            "static": True,
            "pooled": pooled,
            "sharp": sharp,
            "ref_s": ref_s,
        }
        if pooled <= 0:
            return None, "static_empty"
        if sharp < STATIC_SHARP_FRAC * ref_s:
            return None, f"static_lowsharp:{sharp:.0f}<{STATIC_SHARP_FRAC:.2f}x{ref_s:.0f}"
        return float(z_start), "static_ok"

    BG_FRAMES = 8
    BG_SHARP_MED = 4.0
    BG_POOLED_MED = 2.0
    BG_POOLED_SPREAD = 2.0

    def _sample_present(self, monitor: StreamSharpnessMonitor, mark: int) -> bool:
        """Background gate from the live AF metric stream - no camera grabs
        (a grab inside the monitor's stream context deadlocks the pass).

        Blank glass reads sharpness ~1.4-1.8 with pooled flat at zero;
        any sample evidence (structure in pooled16 or sharpness level)
        counts as present. No frames in time or any error -> present
        (fail-open: the point gets measured and normal gates apply).
        """
        try:
            if not monitor.wait_frames(1, mark, timeout=0.5):
                return True
            frames = monitor.frames_since(mark)[-3:]
            sharp = [f[2] for f in frames]
            pooled = [f[3] for f in frames]
            s_med = statistics.median(sharp)
            p_med = statistics.median(pooled)
            p_spread = max(pooled) - min(pooled)
            bg = (
                    s_med < self.BG_SHARP_MED
                    and p_med <= self.BG_POOLED_MED
                    and p_spread <= self.BG_POOLED_SPREAD
            )
            if bg:
                LOGGER.info(
                    "PLANE: point skipped as background "
                    "(s_med=%.1f p_med=%.1f p_spread=%.1f)",
                    s_med, p_med, p_spread,
                )
            return not bg
        except Exception:
            LOGGER.warning("background gate failed - measuring anyway", exc_info=True)
            return True

    def _clamp_anchor(self, z: float) -> float:
        """Predicted anchors may shape the sweep center but never leave
        the approved z envelope around the pass start (Z_ANCHOR_MAX;
        extended to Z_ANCHOR_EXT only when the measured surface gradient
        actually predicts focus out there)."""
        lo = self._pass_z_start - MAX_DEPTH
        hi = self._pass_z_start + MAX_DEPTH
        grad = abs(getattr(self, "_grad_reach", 0.0))
        if grad > 0:
            lo = self._pass_z_start - Z_ANCHOR_EXT
            hi = self._pass_z_start + Z_ANCHOR_EXT
        return max(lo, min(hi, z))

    def _predict_z(self, row: int, col: int) -> Optional[float]:
        """Surface prediction for a lattice point: least-squares plane
        through measured controls when stable, else mean of already-
        measured 8-neighbours. None before any control exists."""
        plane = self._surface_plane()
        if plane is not None:
            a, b, c = plane
            x = self._px + (col - (self._grid_n - 1) / 2.0) * self._step_u
            y = self._py + (row - (self._grid_n - 1) / 2.0) * self._step_u
            return self._clamp_anchor(a * x + b * y + c)
        pts = self._pts_xy
        vals = [
            pts[(row + dr, col + dc)]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1))
            if (row + dr, col + dc) in pts
        ]
        return sum(vals) / len(vals) if vals else None

    _CTRL_XY: dict = {}

    _SUSPECTS: dict = {}

    def _note_control(self, x: float, y: float, z: float) -> None:
        """Record a successful control (stage XY, z offset) for prediction.
        A point deviating strongly from the current surface plane is stored
        but withheld from the fit until a second nearby measurement agrees
        (one false peak must not tilt the surface)."""
        key = (round(x, 3), round(y, 3))
        plane = self._surface_plane()
        if plane is not None:
            a, b, c = plane
            pred = a * x + b * y + c
            if abs(z - pred) > 100.0:
                suspects = type(self)._SUSPECTS
                prev = suspects.pop(key, None)
                if prev is None or abs(prev[2] - z) > 60.0:
                    suspects[key] = (x, y, z)
                    LOGGER.info(
                        "PLANE: control (%.0f,%.0f) z=%+.1f deviates "
                        "%.0f from plane (pred %+.1f) - withheld as suspect",
                        x, y, z, abs(z - pred), pred,
                    )
                    return
                LOGGER.info(
                    "PLANE: suspect (%.0f,%.0f) confirmed by second "
                    "measurement - promoted to control", x, y,
                )
        type(self)._CTRL_XY[key] = float(z)

    def _surface_plane(self) -> Optional[Tuple[float, float, float]]:
        """Least-squares plane z = a*x + b*y + c through the measured
        controls. Falls back to None with < 3 controls or a singular fit."""
        pts = list(type(self)._CTRL_XY.items())
        if len(pts) < 3:
            return None
        xs = [x for (x, _y), _z in pts]
        ys = [y for (_x, y), _z in pts]
        if max(xs) - min(xs) < self._step_u * 0.5:
            return None
        if max(ys) - min(ys) < self._step_u * 0.5:
            return None
        sx = sy = sz = sxx = syy = sxy = sxz = syz = 0.0
        n = float(len(pts))
        for (x, y), z in pts:
            sx += x
            sy += y
            sz += z
            sxx += x * x
            syy += y * y
            sxy += x * y
            sxz += x * z
            syz += y * z
        det = (
                n * (sxx * syy - sxy * sxy)
                - sx * (sx * syy - sxy * sy)
                + sy * (sx * sxy - sxx * sy)
        )
        if abs(det) < 1e-6:
            return None
        a = b = c = 0.0
        try:
            import numpy as np

            m = np.array(
                [[n, sx, sy], [sx, sxx, sxy], [sy, sxy, syy]], dtype=float
            )
            v = np.array([sz, sxz, syz], dtype=float)
            a, b, c = np.linalg.solve(m, v)
        except Exception:
            return None
        if abs(a) > 0.5 or abs(b) > 0.5:
            return None
        self._grad_reach = max(
            abs(a) * ENVELOPE_XY, abs(b) * ENVELOPE_XY
        )
        return (float(a), float(b), float(c))

    def _predict_xy(self, x: float, y: float) -> Optional[float]:
        """Surface estimate at arbitrary XY: least-squares plane through
        measured controls when available, inverse-distance mean fallback."""
        plane = self._surface_plane()
        if plane is not None:
            a, b, c = plane
            return self._clamp_anchor(a * x + b * y + c)
        pts = getattr(self, "_pts_xy", {})
        px_py = self._offsets_all()
        num = den = 0.0
        for (row, col), z in pts.items():
            ox, oy = px_py[(row, col)]
            d2 = (ox - x) ** 2 + (oy - y) ** 2
            if d2 < 1e-9:
                return float(z)
            w = 1.0 / d2
            num += w * z
            den += w
        return num / den if den > 0 else None

    def _offsets_all(self) -> dict:
        """Stage-XY of every coarse-lattice key (grid/step from pass)."""
        grid = self._grid_n
        step = self._step_u
        half = (grid - 1) / 2.0
        return {
            (r, c): (
                self._px + (c - half) * step,
                self._py + (r - half) * step,
            )
            for r in range(grid)
            for c in range(grid)
        }

    DENSE_LATTICE = 128

    def _dense_cell(self) -> float:
        """Dense output-lattice cell size across the +-MAX_RADIUS span."""
        return (2.0 * MAX_RADIUS) / self.DENSE_LATTICE

    def _find_nearest_sample(
            self,
            monitor: StreamSharpnessMonitor,
            gx: float,
            gy: float,
            done: List[Tuple[float, float]],
    ) -> Optional[Tuple[float, float]]:
        """Edge locator for a missed lattice point: binary-search the
        segment toward the nearest measured control point until the
        sample boundary is bracketed to dense-cell size (~11u), then
        return the on-sample end. Moves land in done for the return
        replay; ~6 probe stops per miss."""
        cell = self._dense_cell()
        refs = getattr(self, "_pts_xy", {})
        if not refs:
            return None
        ox = self._offsets_all()
        in_disk = [
            p for p in ox.values()
            if math.hypot(p[0] - self._px, p[1] - self._py) <= MAX_RADIUS
        ]
        if not in_disk:
            return None
        mx, my = min(
            in_disk,
            key=lambda p: (p[0] - gx) ** 2 + (p[1] - gy) ** 2,
        )
        lo_x, lo_y = gx, gy
        hi_x, hi_y = mx, my
        while math.hypot(hi_x - lo_x, hi_y - lo_y) > cell:
            t_x, t_y = (lo_x + hi_x) / 2.0, (lo_y + hi_y) / 2.0
            cur_x, cur_y = self._xy_tracked()
            dx, dy = t_x - cur_x, t_y - cur_y
            if dx or dy:
                self._relative_move_xy(dx, dy)
                done.append((dx, dy))
            z_pred = self._predict_xy(t_x, t_y)
            if z_pred is not None:
                self._sweep_leg(
                    monitor, z_pred, TRANSIT_VELOCITY, move_only=True
                )
            if self._sample_present(monitor, len(monitor.times)):
                hi_x, hi_y = t_x, t_y
            else:
                lo_x, lo_y = t_x, t_y
        cur_x, cur_y = self._xy_tracked()
        dx, dy = hi_x - cur_x, hi_y - cur_y
        if dx or dy:
            self._relative_move_xy(dx, dy)
            done.append((dx, dy))
        return (hi_x, hi_y)

    def _sweep_halfspan(self) -> float:
        """Half-window of the z sweep: +-30 once the surface plane is
        stable (>=3 controls), +-60 while the predictor is weak."""
        return NEAR_SPAN if self._surface_plane() is not None else 2 * NEAR_SPAN

    def _claw_cap(self) -> float:
        """How far the claw may extend per direction. Base +-MAX_DEPTH
        (90). Widens toward Z_ANCHOR_EXT (200) only when the measured
        surface gradient predicts focus that far from the reference -
        the extension is available, never sought."""
        cap = MAX_DEPTH
        grad = abs(getattr(self, "_grad_reach", 0.0))
        if grad > MAX_DEPTH:
            cap = min(Z_ANCHOR_EXT, grad)
        return cap

    def _measure_point(
            self, monitor: StreamSharpnessMonitor, anchor: float
    ) -> Tuple[Optional[float], str]:
        """Sweep a symmetric window around this point's predicted anchor
        and shape-gate a fit on real labels. Extensions continue the same
        glide downward or upward while the curve keeps climbing, to
        +-MAX_DEPTH either way.

        Returns (fitted peak z in OFM coordinates, reason). reason is "ok"
        on success, otherwise why the point produced no usable peak.
        """
        if getattr(self, "_static_fast", False) and getattr(self, "_ref_s", None):
            peak, reason = self._static_verify(monitor, anchor, self._ref_s)
            if peak is not None:
                LOGGER.info(
                    "PLANE: static shortcut accepted (%s) ref=%.0f",
                    reason, self._ref_s,
                )
                return peak, "static_ok"
            LOGGER.info(
                "PLANE: static shortcut declined (%s), sweeping",
                reason,
            )
        self._sweep_leg(
            monitor, anchor + self._sweep_halfspan(),
            TRANSIT_VELOCITY, move_only=True
        )
        mark = len(monitor.times)
        static_med = self._sharp_at(
            monitor, mark,
            time.monotonic() - 0.02, time.monotonic() + 0.5,
        )
        if static_med is None:
            static_med = 0.0

        traj, frames = self._sweep_leg(
            monitor, anchor - self._sweep_halfspan(),
            max(SWEEP_VELOCITY, _sweep_velocity())
        )
        if not traj and getattr(self, "_last_leg_status", "") == "telestall":
            LOGGER.info("PLANE: telestall retry once")
            traj, frames = self._sweep_leg(
                monitor, anchor - self._sweep_halfspan(),
                max(SWEEP_VELOCITY, _sweep_velocity())
            )
        if not traj:
            return None, f"sweep:{getattr(self, '_last_leg_status', 'notraj')}"
        pts = self._label_leg(frames, traj)
        depth = self._sweep_halfspan()
        claw_cap = self._claw_cap()
        while (
                len(pts) >= 12 and depth < claw_cap
                and self._still_rising(pts)
        ):
            traj2, frames2 = self._sweep_leg(
                monitor, anchor - depth - NEAR_SPAN,
                max(SWEEP_VELOCITY, _sweep_velocity())
            )
            if not traj2:
                break
            pts = pts + self._label_leg(frames2, traj2)
            depth += NEAR_SPAN
        depth_up = 0.0
        while (
                len(pts) >= 12 and depth_up < claw_cap
                and self._still_rising_top(pts)
        ):
            top_z = max(p[0] for p in pts)
            traj2, frames2 = self._sweep_leg(
                monitor, top_z + NEAR_SPAN,
                max(SWEEP_VELOCITY, _sweep_velocity())
            )
            if not traj2:
                break
            pts = pts + self._label_leg(frames2, traj2)
            depth_up += NEAR_SPAN
        self._last_point_diag = {
            "n_frames": len(frames),
            "n_pts": len(pts),
            "depth": depth,
            "depth_up": depth_up,
            "z_span": [min((p[0] for p in pts), default=None),
                       max((p[0] for p in pts), default=None)],
            "peak_s": max((p[2] for p in pts), default=None),
            "curve": [[round(p[0], 2), round(p[2], 1)] for p in pts],
        }
        if len(pts) < POINTS_PER_LEG_MIN:
            LOGGER.info("PLANE: sparse leg (%d pts), skipping", len(pts))
            self._sweep_leg(
                monitor, anchor, TRANSIT_VELOCITY, move_only=True
            )
            return None, f"sparse:{len(pts)}"
        static_pooled = 0.0
        with monitor._hook_lock:
            recent_p = monitor.pooled[-3:]
        if recent_p:
            static_pooled = statistics.median(recent_p)
        fit = fit_log_gaussian(
            [CurvePoint(z=p[0], sharpness=p[2], brightness=p[1]) for p in pts],
            window=FIT_WINDOW,
        )
        fit_b = fit_log_gaussian(
            [CurvePoint(z=p[0], sharpness=p[4], brightness=p[1]) for p in pts],
            window=FIT_WINDOW,
        )
        if not fit.ok or fit.peak is None:
            agree = (
                    fit_b.ok
                    and fit_b.peak is not None
                    and static_pooled > 0
            )
            if agree:
                self._sweep_leg(
                    monitor, fit_b.peak, TRANSIT_VELOCITY, move_only=True
                )
                self._last_point_diag["brenner_rescue"] = True
                LOGGER.info(
                    "PLANE: sharp flat but brenner fit peak=%.1f pooled=%.1f",
                    fit_b.peak, static_pooled,
                )
                return float(fit_b.peak), "ok"
            if static_pooled > 0:
                self._sweep_leg(
                    monitor, anchor, TRANSIT_VELOCITY, move_only=True
                )
                LOGGER.info(
                    "PLANE: flat, sample present (pooled=%.1f), anchored at start",
                    static_pooled,
                )
                self._last_point_diag["anchored"] = True
                return float(anchor), "anchored"
            LOGGER.info("PLANE: fit rejected (%s), no sample", fit.reason)
            self._sweep_leg(
                monitor, anchor, TRANSIT_VELOCITY, move_only=True
            )
            return None, f"fit:{fit.reason}"
        if (
                fit_b.ok
                and fit_b.peak is not None
                and abs(fit_b.peak - fit.peak)
                > max(4.0, 0.15 * (fit.fwhm or 30.0))
        ):
            LOGGER.info(
                "PLANE: metric disagreement sharp=%.1f brenner=%.1f",
                fit.peak, fit_b.peak,
            )
            self._sweep_leg(
                monitor, anchor, TRANSIT_VELOCITY, move_only=True
            )
            return None, "disagree"
        if not (
                anchor - MAX_DEPTH <= fit.peak <= anchor + MAX_DEPTH
        ):
            LOGGER.info(
                "PLANE: peak %.1f outside swept window [%.1f..%.1f]",
                fit.peak, anchor - MAX_DEPTH, anchor + MAX_DEPTH,
            )
            self._sweep_leg(
                monitor, anchor, TRANSIT_VELOCITY, move_only=True
            )
            return None, f"peak_outside:{fit.peak:.1f}"
        self._sweep_leg(monitor, fit.peak, TRANSIT_VELOCITY, move_only=True)
        LOGGER.info(
            "PLANE: point fit peak=%.1f r2=%.3f snr=%.1f static=%.0f",
            fit.peak, fit.r2, fit.snr, static_med,
        )
        return float(fit.peak), "ok"

    def _sweep_leg(
            self,
            monitor: StreamSharpnessMonitor,
            target: float,
            velocity: float,
            move_only: bool = False,
    ) -> Tuple[list, list]:
        """One z leg: mark frames, queue the move, wait for motion_report
        to certify arrival, then return the real trajectory and the frames
        inside the move window. The window is bounded below by the previous
        leg's end time so stale deque samples never leak in. With
        move_only=True this is a bare reposition leg (return/park): no
        frame mark, no trajectory collection. All waits are happens-before
        on telemetry, never timed."""
        stage = self._stage
        cur = self._last_hw
        if cur != cur:
            self._last_leg_status = "noz"
            return [], []
        z_span = self._pass_z_start - MAX_Z_EXCURSION
        z_ceiling = self._pass_z_start + MAX_Z_EXCURSION
        clamped = max(z_span, min(z_ceiling, target))
        if clamped != target:
            LOGGER.info(
                "PLANE: leg target %.1f clamped to %.1f "
                "(+/- %d envelope)", target, clamped, MAX_Z_EXCURSION,
            )
            target = clamped
        delta = target - cur
        if abs(delta) < 0.5:
            self._last_leg_status = "skip"
            return [], []
        z_span = self._pass_z_start - MAX_Z_EXCURSION
        z_ceiling = self._pass_z_start + MAX_Z_EXCURSION
        if not (z_span <= target <= z_ceiling):
            self._last_leg_status = "insane_delta"
            LOGGER.error(
                "PLANE: commanded z target %.1f outside sane span "
                "[%.1f, %.1f] - aborting pass",
                target, z_span, z_ceiling,
            )
            raise SampleNotFoundError(
                f"z motion sanity: target {target:.1f} outside sane span"
            )
        seconds = abs(delta) / velocity
        motion_since = getattr(stage, "motion_since", None)
        motion_wait = getattr(stage, "motion_wait", None)
        if motion_since is None or motion_wait is None:
            LOGGER.info("PLANE: no motion telemetry on stage, cannot sweep")
            self._last_leg_status = "notelemetry"
            return [], []
        mark = len(monitor.times)
        ztol = max(2.0, 0.05 * abs(delta))
        frame_off = float(getattr(self, "_frame_offset", 0.0))

        af_thing = getattr(self, "_af", None)
        learned = getattr(af_thing, "_frame_offset", None) if af_thing is not None else None
        self._frame_offset = float(learned) if learned is not None else 0.0
        frame_off = self._frame_offset

        clear = getattr(stage, "clear_motion_history", None)
        if clear is not None:
            clear()
        stage.move_relative_z_timed(self._hw_steps_param(delta), seconds)
        self._last_hw = target
        leg_start = time.monotonic()

        def _motion_started():
            return any(
                len(sm_p) > 2 and abs(sm_p[2]) > 0.5
                for sm_p in motion_since(0)
            )

        def _marker_seen():
            for s in reversed(motion_since(0)):
                if (
                        len(s) > 4
                        and s[4] is not None
                        and s[1] is not None
                        and abs(s[2]) < 0.5
                        and abs(s[1] - (target + frame_off)) <= ztol
                ):
                    return s
            return None

        def _transit_drained():
            drained = getattr(stage, "motion_drained", None)
            return drained is not None and drained(0.0)

        if move_only:
            deadline = leg_start + max(seconds, 0.05) + 1.0
            quiet_after = leg_start + seconds + 0.2
            moved = False
            while time.monotonic() < deadline:
                motion_wait(timeout=0.1)
                if _transit_drained():
                    self._last_leg_status = "ok"
                    return [], []
                now = time.monotonic()
                if not moved and any(
                    len(sm_p) > 2 and abs(sm_p[2]) > 0.5
                    for sm_p in motion_since(0)
                ):
                    moved = True
                if now >= quiet_after and not moved:
                    self._last_leg_status = "ok"
                    return [], []
            LOGGER.info(
                "PLANE: transit timeout target=%.1f waited=%.2fs "
                "samples=%d",
                target, time.monotonic() - leg_start,
                len(motion_since(0)),
            )
            self._last_leg_status = "telestall"
            return [], []

        t_wait0 = leg_start
        start_deadline = t_wait0 + 10.0
        while not _motion_started():
            if time.monotonic() > start_deadline:
                break
            motion_wait(timeout=0.25)
        marker = None
        last_z = None
        motion_seen = False
        z_change_at = None
        hard_end = t_wait0 + 15.0
        while marker is None:
            now = time.monotonic()
            if now > hard_end:
                break
            seq = motion_since(0)
            if seq:
                newest_z = seq[-1][1]
                if newest_z is not None and newest_z != last_z:
                    if last_z is not None and not motion_seen:
                        motion_seen = True
                    if motion_seen:
                        z_change_at = now
                    last_z = newest_z
            if motion_seen and z_change_at is not None:
                if now - z_change_at > 1.5:
                    break
            elif now - t_wait0 > 6.0:
                break
            monitor.wait_frames(4, mark, timeout=0.5)
            motion_wait(timeout=0.5)
            marker = _marker_seen()
        _t_wait = time.monotonic() - t_wait0
        if marker is None:
            recent = motion_since(0)[-6:]
            LOGGER.info(
                "PLANE: TELESTALL target=%.1f waited=%.2fs samples=%d "
                "frame=%.2f recent=%s",
                target, _t_wait, len(motion_since(0)), frame_off,
                [
                    (round(sm_p[1], 2), round(sm_p[2], 2))
                    for sm_p in recent
                ],
            )
            self._last_leg_status = "telestall"
            return [], []
        m_et = marker[4]
        raw = [
            s for s in motion_since(0)
            if len(s) > 4 and s[4] is not None and s[1] is not None
               and s[4] <= m_et
        ]
        raw.sort(key=lambda s: s[4])
        if len(raw) < 2:
            return [], []
        hoff = getattr(stage, "handshake_offset", lambda: None)()
        if hoff is None:
            traj = [(s[4], s[1]) for s in raw]
        else:
            traj = [(s[4] - hoff, s[1]) for s in raw]
        t_lo, t_hi = traj[0][0], traj[-1][0]
        frames = self._frames_since5(monitor, mark, t_lo, t_hi)
        return traj, frames

    @staticmethod
    def _frames_since5(monitor, mark, t_lo=None, t_hi=None) -> list:
        """frames_since equivalent that includes the brenner channel."""
        with monitor._hook_lock:
            rows = list(zip(
                monitor.times[mark:],
                monitor.chroma[mark:],
                monitor.sharpness[mark:],
                monitor.pooled[mark:],
                monitor.brenner[mark:],
            ))
        if t_lo is None:
            return rows
        return [r for r in rows if t_lo <= r[0] <= t_hi]

    def _label_leg(self, frames, traj) -> List[tuple]:
        """Interpolate real telemetry z at each frame timestamp."""
        if len(frames) < POINTS_PER_LEG_MIN or len(traj) < 2:
            return []
        t0, t1 = traj[0][0], traj[-1][0]
        if t1 <= t0:
            return []
        pts = []
        for (t, cbr, s, p, b) in frames:
            if t < t0 or t > t1:
                continue
            zq = traj[0][1]
            for i in range(1, len(traj)):
                if traj[i - 1][0] <= t <= traj[i][0]:
                    dt = traj[i][0] - traj[i - 1][0]
                    frac = 0.0 if dt <= 0 else (t - traj[i - 1][0]) / dt
                    zq = traj[i - 1][1] + frac * (traj[i][1] - traj[i - 1][1])
                    break
            pts.append((zq, cbr, s, p, b))
        return pts

    def _xy_tracked(self) -> Tuple[float, float]:
        """Software-tracked XY of the running pass (never the stale cache).
        Falls back to the stage read once at pass start."""
        pos = getattr(self, "_xy_pos", None)
        if pos is None:
            pos = (self._px, self._py)
            self._xy_pos = pos
        return pos

    def _assert_envelope(self, x: float, y: float) -> None:
        """Hard envelope: every commanded XY position must sit within
        MAX_RADIUS of the pass origin (radial) and ENVELOPE_XY per axis.
        Checked in software-tracked coordinates before any dispatch;
        violations abort the pass.
        """
        if not hasattr(self, "_px"):
            return
        if (
                abs(x - self._px) > ENVELOPE_XY
                or abs(y - self._py) > ENVELOPE_XY
        ):
            raise SampleNotFoundError(
                f"envelope violation: ({x:.0f},{y:.0f}) exceeds "
                f"+/-{ENVELOPE_XY:.0f} of origin ({self._px:.0f},{self._py:.0f})"
            )
        dr = ((x - self._px) ** 2 + (y - self._py) ** 2) ** 0.5
        if dr > MAX_RADIUS:
            raise SampleNotFoundError(
                f"envelope violation: ({x:.0f},{y:.0f}) is {dr:.0f} "
                f"from origin, over MAX_RADIUS {MAX_RADIUS:.0f}"
            )

    def _relative_move_xy(self, dx: float, dy: float) -> None:
        cur_x, cur_y = self._xy_tracked()
        nx, ny = cur_x + dx, cur_y + dy
        self._assert_envelope(nx, ny)
        self._xy_pos = (nx, ny)
        mover = getattr(self._stage, "move_gcode", None)
        noblock = getattr(self._stage, "move_relative_xy_noblock", None)
        if noblock is not None:
            noblock(int(round(dx)), int(round(dy)))
        elif mover is not None:
            mover(
                self._stage.MovementType.RELATIVE,
                False,
                [int(round(dx)), int(round(dy)), 0],
                planned=True,
            )
        else:
            self._stage.move_relative(x=int(round(dx)), y=int(round(dy)))

    @staticmethod
    def _point_in_polys(x: float, y: float, polys) -> bool:
        """Ray-casting test against the edge-walk polylines. Points on a
        boundary count as sample (fail toward measuring)."""
        inside = False
        for poly in polys:
            n = len(poly)
            for k in range(n):
                x1, y1 = poly[k]
                x2, y2 = poly[(k + 1) % n]
                if (y1 > y) != (y2 > y):
                    xint = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                    if x < xint:
                        inside = not inside
        return inside

    def _build_masked_grid(
            self, polys,
    ) -> Tuple[List[float], List[float], List[List[bool]]]:
        """Full-square MESH_CELLS lattice over the whole pass envelope;
        mask[j][i] = True when the cell centre is on sample. The mask is
        evaluated on a coarse stride (COAST_CELL) and refined by the
        edge-walk polygons, so a 129x129 grid costs one ray-cast per
        coarse cell plus boundary cells only."""
        half = MAX_RADIUS
        n = MESH_CELLS + 1
        xs = [self._px - half + i * MESH_CELL for i in range(n)]
        ys = [self._py - half + j * MESH_CELL for j in range(n)]
        coarse = int(round(COAST_CELL / MESH_CELL)) or 1
        mask: List[List[bool]] = [[False] * n for _ in range(n)]
        for j in range(0, n, coarse):
            for i in range(0, n, coarse):
                on = self._point_in_polys(xs[i], ys[j], polys)
                for dj in range(coarse):
                    jj = j + dj
                    if jj >= n:
                        break
                    row = mask[jj]
                    for di in range(coarse):
                        ii = i + di
                        if ii < n:
                            row[ii] = on
        # boundary cells: exact ray-cast where the coarse fill meets its
        # opposite (keeps edges at polygon precision, not COAST_CELL)
        for j in range(n):
            for i in range(n):
                m = mask[j][i]
                nb = (
                    mask[j + 1][i] if j + 1 < n else m,
                    mask[j - 1][i] if j > 0 else m,
                    mask[j][i + 1] if i + 1 < n else m,
                    mask[j][i - 1] if i > 0 else m,
                )
                if any(b != m for b in nb):
                    mask[j][i] = self._point_in_polys(
                        xs[i], ys[j], polys
                    )
        return xs, ys, mask

    @staticmethod
    def _near_boundary(mask, ri: int, ci: int) -> bool:
        rows, cols = len(mask), len(mask[0])
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = ri + dr, ci + dc
                if 0 <= r < rows and 0 <= c < cols and not mask[r][c]:
                    return True
        return False

    def _install_from_fits(
            self,
            fits: List[dict],
            px: float,
            py: float,
            step: float,
            z_start: float = 0.0,
            polys=None,
    ) -> dict:
        measured = [
            (f["x"], f["y"], f["z"])
            for f in fits
            if f["z"] is not None
               and str(f.get("reason", "")).split(":")[0] in (
                   "ok", "anchored", "static_ok", "edge", "late",
                   "canyon", "holeedge", "coast",
               )
        ]
        if len(measured) < 4:
            raise SampleNotFoundError(
                f"only {len(measured)} usable control points - need >= 4"
            )
        if polys:
            xs, ys, mask = self._build_masked_grid(polys)
        else:
            xs, ys, mask = self._build_masked_grid([])
        cols, rows = len(xs), len(ys)

        def _idw(lx: float, ly: float) -> float:
            ranked = sorted(
                measured,
                key=lambda p: (p[0] - lx) ** 2 + (p[1] - ly) ** 2,
            )
            near = ranked[:4]
            d0 = math.hypot(near[0][0] - lx, near[0][1] - ly)
            if d0 < 1e-6:
                return near[0][2]
            num = den = 0.0
            for cx_, cy_, cz_ in near:
                w = 1.0 / ((cx_ - lx) ** 2 + (cy_ - ly) ** 2)
                num += w * cz_
                den += w
            return num / den

        sample_cells = sum(1 for row in mask for m in row if m)
        matrix = [[0.0] * cols for _ in range(rows)]
        direct = 0
        boundary_z = 0
        for ri in range(rows):
            for ci in range(cols):
                if not mask[ri][ci]:
                    continue
                lx, ly = xs[ci], ys[ri]
                ranked = sorted(
                    measured,
                    key=lambda p: (p[0] - lx) ** 2 + (p[1] - ly) ** 2,
                )
                d0 = math.hypot(ranked[0][0] - lx, ranked[0][1] - ly)
                if d0 < 1e-6:
                    direct += 1
                    matrix[ri][ci] = ranked[0][2]
                    continue
                if self._near_boundary(mask, ri, ci) and d0 <= 2 * MESH_CELL:
                    matrix[ri][ci] = ranked[0][2]
                    boundary_z += 1
                    continue
                matrix[ri][ci] = _idw(lx, ly)
        if sample_cells < 4:
            raise SampleNotFoundError(
                f"edge-walk mask leaves only {sample_cells} sample cells"
            )
        LOGGER.info(
            "PLANE: %d controls, %d/%d cells on sample over full "
            "%dx%d square (cell %.1f), %d cells masked as glass, "
            "%d boundary cells pinned",
            len(measured), direct, sample_cells, cols, rows,
            MESH_CELL, cols * rows - sample_cells, boundary_z,
        )
        flat = [
            z
            for row, row_m in zip(matrix, mask)
            for z, m in zip(row, row_m)
            if m
        ]
        if max(flat) - min(flat) > MAX_PEAK_PTP:
            raise SampleNotFoundError(
                f"focus range {max(flat) - min(flat):.1f} exceeds "
                f"{MAX_PEAK_PTP:.0f} - refusing to install"
            )
        def _m(r, c):
            return mask[r][c]

        for r in range(len(matrix)):
            for c in range(len(xs) - 1):
                if _m(r, c) and _m(r, c + 1) and abs(
                        matrix[r][c + 1] - matrix[r][c]
                ) > MAX_NEIGHBOUR_JUMP:
                    raise SampleNotFoundError(
                        f"x jump at row {r} col {c} exceeds "
                        f"{MAX_NEIGHBOUR_JUMP:.0f} - refusing to install"
                    )
        for r in range(len(matrix) - 1):
            for c in range(len(xs)):
                if _m(r, c) and _m(r + 1, c) and abs(
                        matrix[r + 1][c] - matrix[r][c]
                ) > MAX_NEIGHBOUR_JUMP:
                    raise SampleNotFoundError(
                        f"y jump at row {r} col {c} exceeds "
                        f"{MAX_NEIGHBOUR_JUMP:.0f} - refusing to install"
                    )
        grid_s = " ".join(f"{z:.3f}" for z in flat)
        mask_s = "".join(
            "1" if m else "0" for row in mask for m in row
        )
        script = (
            f"SAMPLE_MESH COLS={len(xs)} ROWS={len(matrix)} "
            f"MIN_X={xs[0]:.3f} MAX_X={xs[-1]:.3f} "
            f"MIN_Y={ys[0]:.3f} MAX_Y={ys[-1]:.3f} "
            f"ZREF={z_start:.3f} "
            f"MASK=\"{mask_s}\" "
            f"GRID=\"{grid_s}\""
        )
        LOGGER.info(
            "PLANE: installing %dx%d mesh x=[%.0f..%.0f] y=[%.0f..%.0f] "
            "z=[%.1f..%.1f]",
            len(xs), len(matrix), xs[0], xs[-1], ys[0], ys[-1],
            min(flat), max(flat),
        )
        return self._send_gcode(script)

    @lt.action
    def measure_pass(
            self,
            grid: int = Field(GRID, gt=1),
            step: float = Field(GRID_STEP, gt=0),
            install: bool = Field(True),
    ) -> dict:
        """Grid the sample with relative moves, sweep +-30 at each point,
        fit each curve, and hand the surface to SAMPLE_MESH.

        Camera runs the proven AF lifecycle: crop990 at 120fps with the
        encoder burst, per-frame metrics from the pre_callback hook.
        Points sit on a (grid x grid) lattice spaced `step` apart, visited
        center-out (nearest first) so interior fits seed the outer windows;
        points beyond MAX_RADIUS of the start position are never visited.
        Each point is checked against the camera's background detector and
        skipped cheaply when it images bare background. Mesh coordinates
        are anchored to the stage's reported position at pass start
        (klippy frame). Points without a shape-gated peak inside the swept
        window are skipped. Installs nothing unless at least MIN_FITS
        points fit and every install guard passes.
        """
        if getattr(type(self), "_PASS_BUSY", False):
            raise SampleNotFoundError("another pass is already running")
        try:
            st = self._stage._get_status(payload={
                "objects": {"webhooks": None}
            })
            state_msg = str(
                st["result"]["status"]["webhooks"]["state_message"]
            )
            if "jog mode" in state_msg.lower():
                raise SampleNotFoundError(
                    "klippy is in jog mode - exit jog (JOG VALUE=0) "
                    "before running a focus pass"
                )
        except SampleNotFoundError:
            raise
        except Exception:
            LOGGER.warning(
                "could not read klippy state for jog interlock", exc_info=True
            )
        type(self)._PASS_BUSY = True
        self._cancel_pass_cooler()
        try:
            sm_st = self._stage._get_status(payload={
                "objects": {"sample_mesh": None}
            })
            sm = (
                (sm_st.get("result") or {}).get("status") or {}
            ).get("sample_mesh") or {}
            if sm.get("active"):
                self._send_gcode("SAMPLE_MESH_SUSPEND")
                type(self)._PASS_SUSPENDED_MESH = True
                LOGGER.info(
                    "PLANE: mesh transform suspended for pass"
                )
        except Exception:
            type(self)._PASS_SUSPENDED_MESH = False
            LOGGER.warning(
                "mesh suspend for pass failed", exc_info=True
            )
        prev_exp = prev_gain = None
        try:
            prev_exp = self._cam.exposure_time
            prev_gain = self._cam.analogue_gain
        except Exception:
            prev_exp = prev_gain = None
        self._pass_base = (prev_gain, prev_exp)
        if (prev_gain or 0.0) >= 3.5:
            raise SampleNotFoundError(
                "camera left in probe regime (gain %.1f) by a dead flow - "
                "restore calibration before running a pass" % prev_gain
            )
        try:
            if hasattr(self._stage, "set_homed"):
                self._stage.set_homed()
        except Exception:
            pass
        try:
            if hasattr(self._stage, "update_position"):
                self._stage.update_position()
        except Exception:
            pass
        try:
            self._last_hw = float(self._stage.position["z"])
        except Exception:
            self._last_hw = float("nan")

        fits: List[dict] = []
        try:
            if self._cam.streaming_mode != "crop990":
                self._cam._start_streaming("crop990")
            with StreamSharpnessMonitor(self._cam, plane="main") as monitor:
                with self._cam._streaming_picamera() as picam:
                    picam.set_controls({"FrameRate": 120.0})
                monitor.wait_frames(4, 0, timeout=2.0)
                with _EncoderBurst(self._cam):
                    _g, _e = _probe_regime(prev_gain, prev_exp)
                    self._cam.set_probe_controls_raw(_g, _e)
                    self._pass_inner(monitor, grid, step, fits)
                    if install:
                        self._densify(
                            monitor, fits, grid, step, self._pass_z_start
                        )
        finally:
            try:
                self._cam.restore_calibration_raw()
            except Exception:
                LOGGER.warning("pass gain/exposure restore failed", exc_info=True)
            try:
                if hasattr(self._stage, "update_position"):
                    self._stage.update_position()
            except Exception:
                LOGGER.warning("Stage position sync failed", exc_info=True)
            try:
                if getattr(type(self), "_PASS_SUSPENDED_MESH", False):
                    self._send_gcode("SAMPLE_MESH_RESUME")
                    type(self)._PASS_SUSPENDED_MESH = False
            except Exception:
                LOGGER.warning(
                    "mesh resume after pass failed", exc_info=True
                )
            type(self)._PASS_BUSY = False
            self._arm_pass_cooler()

        measured = [f for f in fits if f["z"] is not None]
        reasons = {}
        for f in fits:
            key = str(f.get("reason", "unknown"))
            reasons[key] = reasons.get(key, 0) + 1
        ok = len(measured) >= MIN_FITS
        result = {
            "ok": ok,
            "reasons": reasons,
            "start": {"x": None, "y": None, "z": None},
            "fits": fits,
            "n_measured": len(measured),
            "installed": False,
            "install_response": None,
        }
        if ok and install:
            try:
                live_x = float(self._stage.position["x"])
                live_y = float(self._stage.position["y"])
            except Exception:
                live_x = live_y = None
            trk_x, trk_y = self._xy_tracked()
            drift = None if live_x is None else max(
                abs(live_x - trk_x), abs(live_y - trk_y))
            if drift is not None and drift > 2.0:
                result["reasons"]["install_refused"] = (
                    f"frame drifted {drift:.0f} units during pass; "
                    "re-zero or re-home invalidated the frame - "
                    "mesh not installed"
                )
                result["ok"] = False
                return result
            try:
                result["install_response"] = self._install_from_fits(
                    fits, self._px, self._py, step,
                    z_start=self._pass_z_start,
                    polys=getattr(self, "_coast_polys", None) or None,
                )
                result["installed"] = True
                self._run_post_pass_af(result)
            except SampleNotFoundError as exc:
                result["ok"] = False
                result["reasons"]["install_refused"] = str(exc)
                result["fits"] = fits
            return result
        return result

    def _pass_inner(
            self,
            monitor: StreamSharpnessMonitor,
            grid: int,
            step: float,
            fits: List[dict],
    ) -> None:
        """Serpentine the grid, measure each point, return home on exit.

        Points classified measured / anchored / skipped carry their z in
        `self._pts_xy` for the adaptive densify stage.
        """
        self._px = float(self._stage.position["x"])
        self._py = float(self._stage.position["y"])
        z_start = self._stage_z()
        if z_start != z_start:
            self._last_hw = float(self._stage.position["z"])
            z_start = self._stage_z()
        if z_start != z_start:
            raise SampleNotFoundError("no stage z reading at pass start")
        self._pass_z_start = z_start
        self._ref_s = None
        self._static_fast = False
        self._pts_xy = {}
        self._xy_pos = (self._px, self._py)
        self._grid_n = grid
        self._step_u = step
        type(self)._CTRL_XY = {}
        self._coast_polys = []
        half = (grid - 1) / 2.0

        plan: List[Tuple[int, int, float, float, bool]] = []
        for row in range(grid):
            for col in range(grid):
                x = self._px + (col - half) * step
                y = self._py + (row - half) * step
                r = ((x - self._px) ** 2 + (y - self._py) ** 2) ** 0.5
                plan.append((row, col, x, y, r <= MAX_RADIUS))
        center_first = sorted(
            range(len(plan)),
            key=lambda i: (
                (plan[i][2] - self._px) ** 2 + (plan[i][3] - self._py) ** 2,
                plan[i][0],
                plan[i][1],
            ),
        )
        plan = [plan[i] for i in center_first]

        missed: List[Tuple[int, int, float, float]] = []
        done: List[Tuple[float, float]] = []
        cur_x, cur_y = self._xy_tracked()
        try:
            occ = self._occupancy_map(monitor, done)
            coasts = self._trace_coastlines(occ)
            self._coast_polys = self._refine_coastlines(
                monitor, coasts, done
            )
            coasts = self._trace_coastlines(occ)
            cur_x, cur_y = self._xy_tracked()
            LOGGER.info(
                "PLANE: coastline map %d/%d cells present, %d coastlines",
                sum(1 for v in occ.values() if v), len(occ), len(coasts),
            )
            in_sample = {
                key for key, present in occ.items() if present
            }

            def coast_aware(r, c, gx, gy):
                n = COAST_GRID
                ci = int(round(
                    (gy - self._py) / COAST_CELL + (n - 1) / 2.0
                ))
                cj = int(round(
                    (gx - self._px) / COAST_CELL + (n - 1) / 2.0
                ))
                return (ci, cj) in in_sample or (
                    (ci, cj) not in occ
                )
            for row, col, gx, gy, inside in plan:
                if not inside:
                    fits.append({
                        "x": gx,
                        "y": gy,
                        "z": None,
                        "reason": "skipped:outside_radius",
                        "diag": None,
                    })
                    continue
                if not coast_aware(row, col, gx, gy):
                    fits.append({
                        "x": gx,
                        "y": gy,
                        "z": None,
                        "reason": "skipped:coastline",
                        "diag": None,
                    })
                    continue
                dx, dy = gx - cur_x, gy - cur_y
                if dx or dy:
                    self._relative_move_xy(dx, dy)
                    done.append((dx, dy))
                    cur_x, cur_y = gx, gy
                z_pred = self._predict_z(row, col)
                anchor = z_start if z_pred is None else z_pred
                if z_pred is not None:
                    self._sweep_leg(
                        monitor, anchor, TRANSIT_VELOCITY, move_only=True
                    )
                if not self._sample_present(
                        monitor, len(monitor.times)
                ):
                    fits.append({
                        "x": gx,
                        "y": gy,
                        "z": None,
                        "reason": "skipped:background",
                        "diag": None,
                    })
                    missed.append((row, col, gx, gy))
                    cur_x, cur_y = self._xy_tracked()
                    continue
                peak, reason = self._measure_point(monitor, anchor)
                fits.append({
                    "x": gx,
                    "y": gy,
                    "z": None if peak is None else peak - z_start,
                    "reason": reason,
                    "diag": getattr(self, "_last_point_diag", None),
                })
                if peak is not None:
                    self._pts_xy[(row, col)] = peak - z_start
                    self._note_control(gx, gy, peak - z_start)
                    if reason == "ok":
                        self._ref_s = max(
                            self._ref_s or 0.0,
                            float(getattr(self, "_last_point_diag", {}).get("peak_s") or 0.0),
                        )
                    elif reason in ("anchored", "static_ok"):
                        self._static_fast = True
            for row, col, gx, gy in missed:
                cur_x, cur_y = self._xy_tracked()
                dx, dy = gx - cur_x, gy - cur_y
                if dx or dy:
                    self._relative_move_xy(dx, dy)
                    done.append((dx, dy))
                    cur_x, cur_y = gx, gy
                z_pred = self._predict_z(row, col)
                anchor = z_start if z_pred is None else z_pred
                if z_pred is not None:
                    self._sweep_leg(
                        monitor, anchor, TRANSIT_VELOCITY, move_only=True
                    )
                if self._sample_present(monitor, len(monitor.times)):
                    peak, reason = self._measure_point(monitor, anchor)
                    fits.append({
                        "x": gx,
                        "y": gy,
                        "z": None if peak is None else peak - z_start,
                        "reason": f"late:{reason}",
                        "diag": getattr(self, "_last_point_diag", None),
                    })
                    if peak is not None:
                        self._pts_xy[(row, col)] = peak - z_start
                        self._note_control(gx, gy, peak - z_start)
                    continue
                canyon_hit = False
                for reach in CANYON_REACHES:
                    self._sweep_leg(
                        monitor, anchor - reach, TRANSIT_VELOCITY,
                        move_only=True,
                    )
                    if not self._sample_present(
                            monitor, len(monitor.times)
                    ):
                        continue
                    peak, reason = self._measure_point(
                        monitor, anchor - reach
                    )
                    fits.append({
                        "x": gx,
                        "y": gy,
                        "z": None if peak is None else peak - z_start,
                        "reason": f"canyon:{reason}",
                        "diag": getattr(self, "_last_point_diag", None),
                    })
                    if peak is not None:
                        self._pts_xy[(row, col)] = peak - z_start
                        self._note_control(gx, gy, peak - z_start)
                    canyon_hit = True
                    break
                if canyon_hit:
                    cur_x, cur_y = self._xy_tracked()
                    continue
                if math.hypot(gx - self._px, gy - self._py) > MAX_RADIUS:
                    continue
                edge = self._find_nearest_sample(
                    monitor, gx, gy, done
                )
                if edge is None:
                    continue
                ex, ey = edge
                if math.hypot(ex - self._px, ey - self._py) > MAX_RADIUS:
                    LOGGER.info(
                        "PLANE: edge find (%.0f,%.0f) landed outside the "
                        "disk - discarding", ex, ey,
                    )
                    continue
                dup = any(
                        (cx - ex) ** 2 + (cy - ey) ** 2 < 30.0 ** 2
                        for cx, cy in type(self)._CTRL_XY
                )
                if dup:
                    LOGGER.info(
                        "PLANE: edge find (%.0f,%.0f) duplicates an "
                        "existing control - skipping", ex, ey,
                    )
                    continue
                z_pred_e = self._predict_xy(ex, ey)
                e_anchor = z_start if z_pred_e is None else z_pred_e
                self._sweep_leg(
                    monitor, e_anchor, TRANSIT_VELOCITY, move_only=True
                )
                z_hit, r_hit = self._measure_point(monitor, e_anchor)
                fits.append({
                    "x": ex,
                    "y": ey,
                    "z": None if z_hit is None else z_hit - z_start,
                    "reason": f"edge:{r_hit}",
                    "diag": self._last_point_diag,
                })
                if z_hit is not None:
                    self._note_control(ex, ey, z_hit - z_start)
                cur_x, cur_y = self._xy_tracked()
            if type(self)._CTRL_XY:
                deep = [
                    (x, y)
                    for (x, y), z in type(self)._CTRL_XY.items()
                    if abs(y - self._py) > self._step_u * 0.5
                    and z < -60.0
                ]
                mapped = set()
                for dx_c, dy_c in deep:
                    if math.hypot(
                            dx_c - self._px, dy_c - self._py
                    ) > MAX_RADIUS:
                        continue
                    col = round(dx_c / self._step_u)
                    if col in mapped:
                        continue
                    mapped.add(col)
                    cx0 = dx_c
                    cy0 = self._py
                    ex, ey2 = self._xy_tracked()
                    ddx, ddy = cx0 - ex, cy0 - ey2
                    if ddx or ddy:
                        self._relative_move_xy(ddx, ddy)
                        done.append((ddx, ddy))
                    self._map_hole_edge(
                        monitor, cx0, cy0, dy_c, done
                    )
                    if len(mapped) >= self.HOLE_EDGE_SEARCHES:
                        break
                cur_x, cur_y = self._xy_tracked()
        finally:
            for dx, dy in reversed(done):
                if dx or dy:
                    self._relative_move_xy(-dx, -dy)
            try:
                self._sweep_leg(
                    monitor, z_start, TRANSIT_VELOCITY, move_only=True
                )
            except Exception:
                LOGGER.warning(
                    "PLANE: z restore after abort failed", exc_info=True
                )
            fits.extend(getattr(self, "_pass_fits_local", []))
            self._pass_fits_local = []
        self._sweep_leg(
            monitor, anchor, TRANSIT_VELOCITY, move_only=True
        )

    def _run_post_pass_af(self, result: dict) -> None:
        self._cancel_pass_cooler()
        self._af._forced_restore_base = getattr(self, "_pass_base", None)
        try:
            af = self._af.smooth_focus(SmoothAutofocusParams())
            result["post_af"] = {"peak_z": af.peak_z, "mode": af.mode}
            LOGGER.info(
                "PLANE: post-pass AF parked at %.1f (%s)",
                af.peak_z, af.mode,
            )
        except NoFocusFoundError as exc:
            result["post_af"] = {"refused": str(exc)}
            LOGGER.info("PLANE: post-pass AF refused: %s", exc)
        except Exception as exc:
            result["post_af"] = {"error": repr(exc)}
            LOGGER.warning("post-pass AF failed", exc_info=True)
        finally:
            self._af._forced_restore_base = None
            self._arm_pass_cooler()

    def _offsets(self, key, grid: int, step: float) -> Tuple[float, float]:
        """Absolute stage offsets of a (possibly fractional) grid key."""
        r, c = key
        return (
            self._px + (c - (grid - 1) / 2.0) * step,
            self._py + (r - (grid - 1) / 2.0) * step,
        )

    HOLE_EDGE_SEARCHES = 3
    HOLE_EDGE_STEPS = 5

    def _map_hole_edge(
            self, monitor, x_start, y_start, y_target, done,
    ) -> None:
        """Binary-search the sample->hole transition between a measured
        plateau point and a measured far point, then drop a control just
        inside the plateau side so the mesh pins the edge instead of
        interpolating blindly across the gap. The walk never leaves the
        mapped disk: mids beyond MAX_RADIUS count as absent without
        moving (the envelope law outranks the search)."""
        lo_y, hi_y = y_start, y_target
        if not self._sample_present(monitor, len(monitor.times)):
            lo_y, hi_y = y_target, y_start
        for _ in range(self.HOLE_EDGE_STEPS):
            mid = (lo_y + hi_y) / 2.0
            if math.hypot(x_start - self._px, mid - self._py) > MAX_RADIUS:
                hi_y = mid
                continue
            dx, dy = 0.0, mid - self._xy_tracked()[1]
            if dx or dy:
                self._relative_move_xy(dx, dy)
                done.append((dx, dy))
            if self._sample_present(monitor, len(monitor.times)):
                lo_y = mid
            else:
                hi_y = mid
        if math.hypot(x_start - self._px, lo_y - self._py) > MAX_RADIUS:
            LOGGER.info(
                "PLANE: hole edge at (%.0f,%.0f) outside the mapped "
                "disk - skipping", x_start, lo_y,
            )
            return
        ex = self._stage_x_now()
        ey = lo_y
        z_pred = self._predict_xy(ex, ey)
        anchor = self._pass_z_start if z_pred is None else z_pred
        self._sweep_leg(
            monitor, anchor, TRANSIT_VELOCITY, move_only=True
        )
        if not self._sample_present(monitor, len(monitor.times)):
            LOGGER.info(
                "PLANE: hole edge at (%.0f,%.0f) reads background at "
                "plateau side - skipping", ex, ey,
            )
            return
        z_hit, r_hit = self._measure_point(monitor, anchor)
        self._pass_fits_local = getattr(self, "_pass_fits_local", [])
        self._pass_fits_local.append({
            "x": ex,
            "y": ey,
            "z": None if z_hit is None else z_hit - self._pass_z_start,
            "reason": f"holeedge:{r_hit}",
            "diag": getattr(self, "_last_point_diag", None),
        })
        if z_hit is not None:
            self._note_control(ex, ey, z_hit - self._pass_z_start)

    OCC_SWEEP_VELOCITY = 1700.0

    def _occupancy_map(
            self, monitor, done,
    ) -> dict:
        """Gate-only occupancy over the +/-MAX_RADIUS disk on a
        COAST_GRID x COAST_GRID lattice, in ONE continuous serpentine
        sweep: a single move per row, frame stamp mapped to cells along
        the way. Records moves in done for the finally replay. Returns
        {(i, j): bool} present map. Gate logic is unchanged (_sample_
        present thresholds); fail-open per cell when starved."""
        n = COAST_GRID
        cell = COAST_CELL
        occ = {}
        cur_x, cur_y = self._xy_tracked()
        col = list(range(n))
        for i in range(n):
            gy = self._py + (i - (n - 1) / 2.0) * cell
            if i % 2:
                col = col[::-1]
            in_cells = [
                j for j in col
                if math.hypot(
                    (self._px + (j - (n - 1) / 2.0) * cell) - self._px,
                    gy - self._py,
                ) <= MAX_RADIUS
            ]
            if not in_cells:
                for j in col:
                    occ[(i, j)] = False
                continue
            j0, j1 = in_cells[0], in_cells[-1]
            x0 = self._px + (j0 - (n - 1) / 2.0) * cell
            x1 = self._px + (j1 - (n - 1) / 2.0) * cell
            self._assert_envelope(x0, gy)
            self._assert_envelope(x1, gy)
            dx = x0 - cur_x
            dy = gy - cur_y
            if dx or dy:
                self._relative_move_xy(dx, dy)
                done.append((dx, dy))
                cur_x, cur_y = x0, gy
            span = abs(x1 - x0)
            if span < 1.0:
                occ[(i, j0)] = self._sample_present(
                    monitor, len(monitor.times)
                )
                for j in col:
                    if j != j0:
                        occ[(i, j)] = False
                continue
            mark = len(monitor.times)
            monitor.wait_frames(1, mark, timeout=0.5)
            self._relative_move_xy(x1 - x0, 0.0)
            done.append((x1 - x0, 0.0))
            cur_x = x1
            t0 = time.monotonic()
            travel = span / self.OCC_SWEEP_VELOCITY
            for j in in_cells:
                gx = self._px + (j - (n - 1) / 2.0) * cell
                frac = abs(gx - x0) / span
                t_gate = t0 + frac * travel + 0.5 / self.OCC_SWEEP_VELOCITY
                monitor.wait_until(t_gate, timeout=max(0.05, travel))
                frames = monitor.frames_since(mark)
                if frames:
                    picked = min(
                        frames,
                        key=lambda f: abs(f[0] - (t_gate - t0)),
                    )
                    sharp = [picked[2]]
                    pooled = [picked[3]]
                    bg = (
                        sharp[0] < self.BG_SHARP_MED
                        and pooled[0] <= self.BG_POOLED_MED
                    )
                    occ[(i, j)] = not bg
                else:
                    occ[(i, j)] = True
        return occ

    def _trace_coastlines(self, occ: dict) -> list:
        """Moore-neighbour boundary walk over the occupancy map.
        Returns a list of coastlines; each is an ordered list of cells
        (i, j) on a present/absent boundary. Recursion-free."""
        n = COAST_GRID

        def at(i, j):
            return 0 <= i < n and 0 <= j < n and occ.get((i, j), False)

        visited = set()
        coasts = []
        starts = [
            (i, j) for i in range(n) for j in range(n)
            if at(i, j) and not at(i, j - 1)
        ]
        for s in sorted(starts):
            if s in visited:
                continue
            coast = [s]
            visited.add(s)
            prev_dir = 6
            cur = s
            for _ in range(n * n * 8):
                nbrs = [
                    (cur[0] + di, cur[1] + dj)
                    for di, dj in (
                        (-1, -1), (-1, 0), (-1, 1), (0, 1),
                        (1, 1), (1, 0), (1, -1), (0, -1),
                    )
                ]
                k = nbrs.index(
                    nbrs[(prev_dir - 2) % 8]
                ) if False else (prev_dir - 2) % 8
                found = None
                for step in range(8):
                    idx = (k + step) % 8
                    if at(*nbrs[idx]):
                        found = idx
                        break
                if found is None:
                    break
                nxt = nbrs[found]
                prev_dir = found
                if nxt == s:
                    break
                if nxt in visited:
                    break
                visited.add(nxt)
                coast.append(nxt)
                cur = nxt
            coasts.append(coast)
        return coasts

    def _refine_coastlines(
            self, monitor, coasts, done,
    ) -> List[List[Tuple[float, float]]]:
        """Precision edge walk: follow each traced coastline as a chain of
        short continuous hops (EDGE_WALK_STEP apart), classifying each hop
        from the live stream - no stops, no sweeps. Emits ordered
        polylines of boundary points (+/- half a hop). Hops never leave
        the mapped disk; the envelope law outranks the walk."""

        def cell_center(i, j):
            n = COAST_GRID
            return (
                self._px + (i - (n - 1) / 2.0) * COAST_CELL,
                self._py + (j - (n - 1) / 2.0) * COAST_CELL,
            )

        def classify(x: float, y: float) -> bool:
            mark = len(monitor.times)
            if not monitor.wait_frames(1, mark, timeout=0.25):
                return True
            frames = monitor.frames_since(mark)[-2:]
            if not frames:
                return True
            sharp = statistics.median([f[2] for f in frames])
            pooled = statistics.median([f[3] for f in frames])
            return not (
                sharp < self.BG_SHARP_MED
                and pooled <= self.BG_POOLED_MED
            )

        polys: List[List[Tuple[float, float]]] = []
        for coast in coasts:
            if len(coast) < 2:
                continue
            poly: List[Tuple[float, float]] = []
            pos = None
            for i, j in coast:
                cx, cy = cell_center(i, j)
                if pos is None:
                    if (math.hypot(cx - self._px, cy - self._py) > MAX_RADIUS
                            or abs(cx - self._px) > ENVELOPE_XY
                            or abs(cy - self._py) > ENVELOPE_XY):
                        continue
                    cur_x, cur_y = self._xy_tracked()
                    dx, dy = cx - cur_x, cy - cur_y
                    if dx or dy:
                        self._relative_move_xy(dx, dy)
                        done.append((dx, dy))
                    pos = (cx, cy)
                    if classify(cx, cy):
                        poly.append(pos)
                    continue
                seg = math.hypot(cx - pos[0], cy - pos[1])
                if seg < 1e-9:
                    continue
                nsub = max(1, int(round(seg / EDGE_WALK_STEP)))
                pool = max(1, int(EDGE_WALK_POOL / EDGE_WALK_STEP))
                hops = [
                    (
                        pos[0] + (cx - pos[0]) * k / nsub,
                        pos[1] + (cy - pos[1]) * k / nsub,
                    )
                    for k in range(1, nsub + 1)
                ]
                hops = [
                    (wx, wy) for wx, wy in hops
                    if math.hypot(wx - self._px, wy - self._py) <= MAX_RADIUS
                    and abs(wx - self._px) <= ENVELOPE_XY
                    and abs(wy - self._py) <= ENVELOPE_XY
                ]
                for b0 in range(0, len(hops), pool):
                    batch = hops[b0:b0 + pool]
                    if not batch:
                        continue
                    self._assert_envelope(batch[-1][0], batch[-1][1])
                    dx = batch[-1][0] - pos[0]
                    dy = batch[-1][1] - pos[1]
                    if dx or dy:
                        mark = len(monitor.times)
                        self._relative_move_xy(dx, dy)
                        done.append((dx, dy))
                        travel = math.hypot(dx, dy) / EDGE_WALK_MAX_V
                        t0 = time.monotonic()
                        for hi, (wx, wy) in enumerate(batch):
                            frac = (hi + 1) / len(batch)
                            t_gate = t0 + frac * travel
                            monitor.wait_until(
                                t_gate, timeout=max(0.05, travel) + 0.05
                            )
                            if classify(wx, wy):
                                poly.append((wx, wy))
                        pos = batch[-1]
                    else:
                        for wx, wy in batch:
                            if classify(wx, wy):
                                poly.append((wx, wy))
                        pos = batch[-1]
                pos = (cx, cy)
            if len(poly) >= 2:
                polys.append(poly)
        LOGGER.info(
            "PLANE: edge walk traced %d coastline polylines, "
            "%d boundary points total",
            len(polys), sum(len(p) for p in polys),
        )
        return polys

    def _stage_x_now(self) -> float:
        return float(self._stage.position["x"])

    def _stage_y_now(self) -> float:
        return float(self._stage.position["y"])

    def _move_xy_abs(self, x: float, y: float) -> None:
        cur_x, cur_y = self._xy_tracked()
        self._relative_move_xy(x - cur_x, y - cur_y)

    def _densify(
            self,
            monitor: StreamSharpnessMonitor,
            fits: List[dict],
            grid: int,
            step: float,
            z_start: float,
    ) -> None:
        """Adaptive density: level 1 measures every lattice midpoint
        (halving effective spacing), deeper levels only where neighbors
        disagree beyond DENSIFY_JUMP. Midpoints resolve cheaply - static
        verification against the interpolated prediction first, full
        sweep only when that gate fails. Levels capped at DENSIFY_MAX.
        """
        if DENSIFY_MAX <= 0:
            return
        if getattr(self, "_ref_s", None) is None:
            return
        pts = getattr(self, "_pts_xy", {})
        if len(pts) < 2:
            return
        for level in range(1, DENSIFY_MAX + 1):
            spacing = step / (2 ** (level - 1))
            keys = list(pts.keys())
            pending: List[Tuple[Tuple[float, float], float]] = []
            seen = set()
            for i, k1 in enumerate(keys):
                x1, y1 = self._offsets(k1, grid, step)
                for k2 in keys[i + 1:]:
                    x2, y2 = self._offsets(k2, grid, step)
                    if abs(x2 - x1) + abs(y2 - y1) > spacing * 1.01:
                        continue
                    pair = tuple(sorted([k1, k2]))
                    if pair in seen:
                        continue
                    seen.add(pair)
                    z1, z2 = pts[k1], pts[k2]
                    if level > 1 and abs(z2 - z1) <= DENSIFY_JUMP:
                        continue
                    pending.append(
                        (((k1[0] + k2[0]) / 2, (k1[1] + k2[1]) / 2),
                         (z1 + z2) / 2)
                    )
            if not pending:
                return
            for mk, zm in pending:
                mx, my = self._offsets(mk, grid, step)
                self._move_xy_abs(mx, my)
                peak, reason = self._static_verify(
                    monitor, z_start + zm, self._ref_s
                )
                if peak is None:
                    peak, reason = self._measure_point(monitor, z_start)
                fits.append({
                    "x": mx,
                    "y": my,
                    "z": None if peak is None else peak - z_start,
                    "reason": reason,
                    "diag": getattr(self, "_last_point_diag", None),
                    "densify_level": level,
                })
                if peak is not None:
                    pts[mk] = peak - z_start
            self._move_xy_abs(self._px, self._py)
