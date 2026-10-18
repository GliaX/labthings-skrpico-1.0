import logging
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
    StreamSharpnessMonitor,
    _EncoderBurst,
    _probe_regime,
)
from openflexure_microscope_server.things.stage import BaseStage

LOGGER = logging.getLogger(__name__)

NEAR_SPAN = 30.0
MAX_DEPTH = 90.0
SWEEP_VELOCITY = 80.0
RETURN_VELOCITY = 500.0


def _sweep_velocity() -> float:
    """Engine-calibrated near-sweep speed (12 u/s per u of FWHM)."""
    from openflexure_microscope_server.things.smooth_autofocus import (
        SmoothAutofocusThing,
    )

    return SmoothAutofocusThing._cal_near_velocity(
        object.__new__(SmoothAutofocusThing)
    )


GRID_STEP = 150.0
GRID = 3
MAX_PEAK_PTP = 512.0
MAX_NEIGHBOUR_JUMP = 150.0
MIN_FITS = 4
STATIC_SHARP_FRAC = 0.6
DENSIFY_JUMP = 20.0
DENSIFY_MAX = 2
ENVELOPE_XY = 310.0
POINTS_PER_LEG_MIN = 6
FIT_WINDOW = 0.7
STATIC_FRAMES = 8
MARKER_STALL_LIMIT = 3
MARKER_WAIT_S = 5.0


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
    _seg_floor_et: Optional[float] = None
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

    def _measure_point(
            self, monitor: StreamSharpnessMonitor, z_start: float
    ) -> Tuple[Optional[float], str]:
        """Sweep the full +-NEAR_SPAN window around the pass z anchor and
        shape-gate a fit on real labels.

        All z targets are absolute offsets from z_start (captured once per
        pass), so commanded motion stays inside the window regardless of
        fit outcomes or stalls.

        Returns (fitted peak z in OFM coordinates, reason). reason is "ok"
        on success, otherwise why the point produced no usable peak.
        """
        if getattr(self, "_static_fast", False) and getattr(self, "_ref_s", None):
            peak, reason = self._static_verify(monitor, z_start, self._ref_s)
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
            monitor, z_start + NEAR_SPAN, RETURN_VELOCITY, move_only=True
        )
        mark = len(monitor.times)
        monitor.wait_frames(STATIC_FRAMES, mark, timeout=2.0)
        pre = monitor.frames_since(mark)[-STATIC_FRAMES:]
        static_med = (
            statistics.median([f[2] for f in pre]) if len(pre) >= 3 else 0.0
        )
        if static_med <= 0:
            LOGGER.info("PLANE: no frames at point, skipping")
            return None, "noframes"

        traj, frames = self._sweep_leg(
            monitor, z_start - NEAR_SPAN, max(SWEEP_VELOCITY, _sweep_velocity())
        )
        if not traj:
            return None, f"sweep:{getattr(self, '_last_leg_status', 'notraj')}"
        pts = self._label_leg(frames, traj)
        depth = NEAR_SPAN
        while len(pts) >= 12 and depth < MAX_DEPTH and self._still_rising(pts):
            traj2, frames2 = self._sweep_leg(
                monitor, z_start - depth - NEAR_SPAN,
                max(SWEEP_VELOCITY, _sweep_velocity())
            )
            if not traj2:
                break
            pts = pts + self._label_leg(frames2, traj2)
            depth += NEAR_SPAN
        self._last_point_diag = {
            "n_frames": len(frames),
            "n_pts": len(pts),
            "depth": depth,
            "z_span": [min((p[0] for p in pts), default=None),
                       max((p[0] for p in pts), default=None)],
            "peak_s": max((p[2] for p in pts), default=None),
            "curve": [[round(p[0], 2), round(p[2], 1)] for p in pts],
        }
        if len(pts) < POINTS_PER_LEG_MIN:
            LOGGER.info("PLANE: sparse leg (%d pts), skipping", len(pts))
            self._sweep_leg(
                monitor, z_start, RETURN_VELOCITY, move_only=True
            )
            return None, f"sparse:{len(pts)}"
        static_pooled = (
            statistics.median([f[3] for f in pre]) if pre else 0.0
        )
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
                    monitor, fit_b.peak, RETURN_VELOCITY, move_only=True
                )
                self._last_point_diag["brenner_rescue"] = True
                LOGGER.info(
                    "PLANE: sharp flat but brenner fit peak=%.1f pooled=%.1f",
                    fit_b.peak, static_pooled,
                )
                return float(fit_b.peak), "ok"
            if static_pooled > 0:
                self._sweep_leg(
                    monitor, z_start, RETURN_VELOCITY, move_only=True
                )
                LOGGER.info(
                    "PLANE: flat, sample present (pooled=%.1f), anchored at start",
                    static_pooled,
                )
                self._last_point_diag["anchored"] = True
                return float(z_start), "anchored"
            LOGGER.info("PLANE: fit rejected (%s), no sample", fit.reason)
            self._sweep_leg(
                monitor, z_start, RETURN_VELOCITY, move_only=True
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
                monitor, z_start, RETURN_VELOCITY, move_only=True
            )
            return None, "disagree"
        if not (z_start - MAX_DEPTH <= fit.peak <= z_start + NEAR_SPAN):
            LOGGER.info(
                "PLANE: peak %.1f outside swept window [%.1f..%.1f]",
                fit.peak, z_start - MAX_DEPTH, z_start + NEAR_SPAN,
            )
            self._sweep_leg(
                monitor, z_start, RETURN_VELOCITY, move_only=True
            )
            return None, f"peak_outside:{fit.peak:.1f}"
        self._sweep_leg(monitor, fit.peak, RETURN_VELOCITY, move_only=True)
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
        delta = target - cur
        if abs(delta) < 0.5:
            self._last_leg_status = "skip"
            return [], []
        seconds = abs(delta) / velocity
        motion_since = getattr(stage, "motion_since", None)
        motion_wait = getattr(stage, "motion_wait", None)
        if motion_since is None or motion_wait is None:
            LOGGER.info("PLANE: no motion telemetry on stage, cannot sweep")
            self._last_leg_status = "notelemetry"
            return [], []
        seg_floor = self._seg_floor_et
        mark = len(monitor.times)
        ztol = max(2.0, 0.05 * abs(delta))

        def _marker_seen():
            for s in reversed(motion_since(0)):
                if (
                        len(s) > 4
                        and s[4] is not None
                        and s[1] is not None
                        and (seg_floor is None or s[4] > seg_floor)
                        and abs(s[2]) < 0.5
                        and not s[3]
                        and abs(s[1] - target) <= ztol
                ):
                    return s
            return None

        stage.move_relative_z_timed(self._hw_steps_param(delta), seconds)
        self._last_hw = target
        marker = None
        stall = 0
        prev_n = None
        t_wait0 = time.monotonic()
        while marker is None and stall < MARKER_STALL_LIMIT:
            if not move_only:
                monitor.wait_frames(4, mark, timeout=0.5)
            motion_wait(timeout=0.5)
            n = len(motion_since(0))
            if prev_n is not None and n == prev_n:
                stall += 1
            else:
                stall = 0
            prev_n = n
            marker = _marker_seen()
        _t_wait = time.monotonic() - t_wait0
        if marker is None:
            LOGGER.info(
                "PLANE: TELESTALL target=%.1f waited=%.2fs samples=%d",
                target, _t_wait, len(motion_since(0)),
            )
            self._last_leg_status = "telestall"
            return [], []
        m_et = marker[4]
        self._seg_floor_et = m_et
        if move_only:
            return [], []
        raw = [
            s for s in motion_since(0)
            if len(s) > 4 and s[4] is not None and s[1] is not None
               and (seg_floor is None or s[4] > seg_floor)
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
        ENVELOPE_XY of the pass origin. Checked in software-tracked
        coordinates before any dispatch; violations abort the pass.
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

    def _install_from_fits(self, fits: List[dict]) -> dict:
        pts = [(f["x"], f["y"], f["z"]) for f in fits]
        xs = sorted(set(p[0] for p in pts))
        ys = sorted(set(p[1] for p in pts))
        if len(xs) < 2 or len(ys) < 2:
            raise SampleNotFoundError(
                "need at least a 2x2 measured extent to build a mesh"
            )
        lookup = {(p[0], p[1]): p[2] for p in pts}
        step = xs[1] - xs[0] if len(xs) > 1 else (
            ys[1] - ys[0] if len(ys) > 1 else 0.0)
        synthesized: List[tuple] = []
        filled = set(lookup)
        while True:
            pending = [
                (x, y) for y in ys for x in xs
                if (x, y) not in filled
            ]
            if not pending:
                break
            progressed = False
            for x, y in pending:
                nb = [
                    lookup[(nx, ny)]
                    for (nx, ny) in ((x - step, y), (x + step, y),
                                     (x, y - step), (x, y + step))
                    if (nx, ny) in filled and lookup[(nx, ny)] is not None
                ]
                if len(nb) >= 2 or (len(nb) == 1 and y == ys[0]):
                    lookup[(x, y)] = sum(nb) / len(nb)
                    filled.add((x, y))
                    synthesized.append((x, y, lookup[(x, y)]))
                    progressed = True
            if not progressed:
                break
        remaining = [
            (x, y) for y in ys for x in xs
            if lookup.get((x, y)) is None
        ]
        if remaining:
            raise SampleNotFoundError(
                f"incomplete grid, unfillable holes at {remaining}"
            )
        if synthesized:
            LOGGER.info(
                "PLANE: synthesized %d edge cells from neighbor mean: %s",
                len(synthesized), synthesized,
            )
        matrix = [[lookup[(x, y)] for x in xs] for y in ys]
        flat = [z for row in matrix for z in row]
        if max(flat) - min(flat) > MAX_PEAK_PTP:
            raise SampleNotFoundError(
                f"focus range {max(flat) - min(flat):.1f} exceeds "
                f"{MAX_PEAK_PTP:.0f} - refusing to install"
            )
        for r in range(len(matrix)):
            for c in range(len(xs) - 1):
                if abs(matrix[r][c + 1] - matrix[r][c]) > MAX_NEIGHBOUR_JUMP:
                    raise SampleNotFoundError(
                        f"x jump at row {r} col {c} exceeds "
                        f"{MAX_NEIGHBOUR_JUMP:.0f} - refusing to install"
                    )
        for r in range(len(matrix) - 1):
            for c in range(len(xs)):
                if abs(matrix[r + 1][c] - matrix[r][c]) > MAX_NEIGHBOUR_JUMP:
                    raise SampleNotFoundError(
                        f"y jump at row {r} col {c} exceeds "
                        f"{MAX_NEIGHBOUR_JUMP:.0f} - refusing to install"
                    )
        grid_s = " ".join(f"{z:.3f}" for z in flat)
        script = (
            f"SAMPLE_MESH COLS={len(xs)} ROWS={len(matrix)} "
            f"MIN_X={xs[0]:.3f} MAX_X={xs[-1]:.3f} "
            f"MIN_Y={ys[0]:.3f} MAX_Y={ys[-1]:.3f} "
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
        Serpentine over (grid x grid) points spaced `step` apart, starting
        and ending at the current position. Mesh coordinates are anchored
        to the stage's reported position at pass start (klippy frame).
        Points without a shape-gated peak inside the swept window are
        skipped. Installs nothing unless the grid is complete, every guard
        passes, and at least MIN_FITS points fit.
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
        prev_exp = prev_gain = None
        try:
            prev_exp = self._cam.exposure_time
            prev_gain = self._cam.analogue_gain
        except Exception:
            prev_exp = prev_gain = None
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
                    self._cam.analogue_gain = _g
                    self._cam.exposure_time = _e
                    self._pass_inner(monitor, grid, step, fits)
                    if install:
                        self._densify(
                            monitor, fits, grid, step, self._pass_z_start
                        )
        finally:
            try:
                if prev_exp is not None:
                    self._cam.exposure_time = int(prev_exp)
                if prev_gain is not None:
                    self._cam.analogue_gain = prev_gain
            except Exception:
                LOGGER.warning("pass gain/exposure restore failed", exc_info=True)
            try:
                if hasattr(self._stage, "update_position"):
                    self._stage.update_position()
            except Exception:
                LOGGER.warning("Stage position sync failed", exc_info=True)
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
                coarse = [
                    f for f in fits
                    if "densify_level" not in f and f["z"] is not None
                ]
                result["install_response"] = self._install_from_fits(coarse)
                result["installed"] = True
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
        self._seg_floor_et = None
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
        half = (grid - 1) / 2.0

        order: List[Tuple[int, int]] = []
        for row in range(grid):
            cols = range(grid) if row % 2 == 0 else reversed(range(grid))
            for col in cols:
                order.append((row, col))
        moves: List[Tuple[float, float]] = [(-half * step, -half * step)]
        prev = order[0]
        for row, col in order[1:]:
            moves.append(((col - prev[1]) * step, (row - prev[0]) * step))
            prev = (row, col)

        self._pts_xy = {}
        done: List[Tuple[float, float]] = []
        try:
            for (row, col), (dx, dy) in zip(order, moves):
                if dx or dy:
                    self._relative_move_xy(dx, dy)
                    done.append((dx, dy))
                peak, reason = self._measure_point(monitor, z_start)
                x = self._px + (col - half) * step
                y = self._py + (row - half) * step
                fits.append({
                    "x": x,
                    "y": y,
                    "z": None if peak is None else peak - z_start,
                    "reason": reason,
                    "diag": getattr(self, "_last_point_diag", None),
                })
                if peak is not None:
                    self._pts_xy[(row, col)] = peak - z_start
                    if reason == "ok":
                        self._ref_s = max(
                            self._ref_s or 0.0,
                            float(getattr(self, "_last_point_diag", {}).get("peak_s") or 0.0),
                        )
                    elif reason in ("anchored", "static_ok"):
                        self._static_fast = True
        finally:
            for dx, dy in reversed(done):
                if dx or dy:
                    self._relative_move_xy(-dx, -dy)
        self._sweep_leg(
            monitor, z_start, RETURN_VELOCITY, move_only=True
        )

    def _offsets(self, key, grid: int, step: float) -> Tuple[float, float]:
        """Absolute stage offsets of a (possibly fractional) grid key."""
        r, c = key
        return (
            self._px + (c - (grid - 1) / 2.0) * step,
            self._py + (r - (grid - 1) / 2.0) * step,
        )

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
        """Sparse-first adaptive density: only where neighbors disagree
        beyond DENSIFY_JUMP, insert midpoints and resolve them cheaply -
        static verification against the interpolated prediction first,
        full sweep only when that gate fails. Levels capped at DENSIFY_MAX.
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
                    if abs(z2 - z1) <= DENSIFY_JUMP:
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
