import json
import logging
import statistics
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Self

import numpy as np
from picamera2.encoders import MJPEGEncoder
from openflexure_microscope_server.things.camera.picamera import (
    PicameraStreamOutput,
)
from pydantic import BaseModel, Field

import labthings_fastapi as lt

from .autofocus import (
    AutofocusThing,
    NoFocusFoundError as ScanNoFocusFoundError,
    SharpnessDataArrays,
)
from .camera import BaseCamera
from .stage import BaseStage
from .smooth_af_core import (
    CurvePoint,
    FitOutcome,
    fit_log_gaussian,
    median_filter_curve,
)

LOGGER = logging.getLogger(__name__)

_FAR_SPAN = 1024.0
_FAR_VELOCITY = 9000.0
_FAR_MEDIAN_BAR = 1.25
_NEAR_SPAN = 110.0
_NEAR_VELOCITY = 250.0
_TINY_SPAN = 180.0
_TINY_VELOCITY = 700.0
_TINY_MIN_V = 100.0
_PARK_VELOCITY = 9000.0
_CAL_V_ACCEL = 6000.0
_MEDIAN_BAR = 1.6
_EDGE_MARGIN = 0.12
FPS = 120.0


class NoFocusFoundError(RuntimeError):
    """No confident focus was found."""


class SmoothAutofocusParams(BaseModel):
    search_span_steps: int = Field(default=32, gt=0)
    min_r2: float = Field(default=0.5, ge=0, le=1)
    min_snr: float = Field(default=1.4, ge=1)





@dataclass
class SmoothFocusResult:
    peak_z: float
    mode: str
    fit: FitOutcome
    curve: List[CurvePoint] = field(default_factory=list)
    elapsed_s: float = 0.0


@dataclass
class StreamSharpnessMonitor:
    """Per-frame metrics from the camera's raw YUV pre_callback hook."""

    def __init__(self, camera: BaseCamera) -> None:
        self.camera = camera
        self.running = False
        self.times: List[float] = []
        self.sharpness: List[float] = []
        self.brightness: List[float] = []
        self.chroma: List[float] = []
        self.pooled: List[float] = []
        self.decoded = 0
        self._hook_lock = threading.Lock()
        self.frame_event = threading.Event()

    def _hook_cb(self, ts, sharp, bright, chroma=0.0, pooled=0.0) -> None:
        with self._hook_lock:
            self.times.append(ts)
            self.sharpness.append(sharp)
            self.brightness.append(bright)
            self.chroma.append(chroma)
            self.pooled.append(pooled)
            self.decoded += 1
        self.frame_event.set()

    def frames_since(self, mark: int):
        return list(zip(
            self.times[mark:],
            self.chroma[mark:],
            self.sharpness[mark:],
            self.pooled[mark:],
        ))

    def wait_frames(self, n: int, mark: int = 0, timeout: float = 2.0) -> bool:
        """Happens-before: returns when n new frames are ingested."""
        deadline = time.monotonic() + timeout
        while len(self.times) - mark < n:
            self.frame_event.clear()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self.frame_event.wait(remaining)
        return True

    def __enter__(self) -> Self:
        self.running = True
        set_hook = getattr(self.camera, "set_af_metric_hook", None)
        if set_hook is None:
            raise RuntimeError("AF requires a YUV metric hook camera")
        set_hook(self._hook_cb)
        return self

    def __exit__(self, *_exc) -> None:
        self.running = False
        try:
            self.camera.set_af_metric_hook(None)
        except Exception:
            pass


class _EncoderBurst:
    """Throttle sensor fps for a high-fps metric burst, restore after.

    The camera lives in crop990 with the encoders running at preview
    fps; the burst raises the sensor to the metric rate and stops the
    encoders (nobody consumes the UI stream mid-AF). Restore restarts
    the encoders and drops back to preview fps — both live controls,
    no reconfigure.
    """

    def __init__(
        self, camera, fps: float, restore_fps: float = 20.0
    ) -> None:
        self._cam = camera
        self.fps = float(fps)
        self.restore_fps = float(restore_fps)
        self._stopped = False

    def __enter__(self) -> "_EncoderBurst":
        try:
            with self._cam._streaming_picamera() as picam:
                if picam.started:
                    try:
                        picam.stop_encoder()
                        _log("BURST stop_encoder t=%.3f" % (time.monotonic(),))
                    except Exception:
                        return self
            self._stopped = True
        except Exception:
            self._stopped = False
        return self

    def __exit__(self, *exc) -> bool:
        try:
            with self._cam._streaming_picamera() as picam:
                mode_info = getattr(self._cam, "streaming_modes", {}).get(
                    getattr(self._cam, "streaming_mode", "default")
                )
                use_lores = (
                    mode_info is not None
                    and getattr(mode_info, "use_lores_as_preview", False)
                )
                stream_name = "lores" if use_lores else "main"
                if not picam.started:
                    try:
                        picam.start()
                    except Exception as e:
                        LOGGER.error("AF burst: camera start failed: %s", e)
                bitrate = getattr(self._cam, "mjpeg_bitrate", 10000000)
                enc1 = MJPEGEncoder(bitrate)
                try:
                    picam.start_recording(
                        enc1,
                        PicameraStreamOutput(
                            self._cam.mjpeg_stream, enc1
                        ),
                        name=stream_name,
                    )
                except Exception as e:
                    LOGGER.error("AF burst: main encoder restore failed: %s", e)
                enc2 = MJPEGEncoder(100000000)
                try:
                    picam.start_encoder(
                        enc2,
                        PicameraStreamOutput(
                            self._cam.lores_mjpeg_stream, enc2
                        ),
                        name="lores",
                    )
                except Exception as e:
                    LOGGER.error("AF burst: lores encoder restore failed: %s", e)
                try:
                    picam.set_controls({"FrameRate": float(self.restore_fps)})
                except Exception:
                    pass
        except Exception as e:
            LOGGER.error("AF burst: camera restore failed: %s", e)
        return False


def _log(line: str) -> None:
    try:
        Path(__file__).resolve().parents[3].joinpath(
            "smooth_af_moves.log"
        ).open("a").write(line + "\n")
    except Exception:
        pass


def _probe_regime(prev_gain, prev_exp):
    """Probe imaging regime from the pre-AF state: gain doubled, exposure
    halved (less motion blur), clamped to keep 120fps and sane frames."""
    g = 2.0 * prev_gain if prev_gain else 2.0
    g = max(1.0, min(16.0, g))
    e = (prev_exp / 2.0) if prev_exp else 1000.0
    e = max(100.0, min(8000.0, e))
    return g, int(e)


def _dump_json(name: str, payload: dict) -> None:
    try:
        with open(
            f"/home/admin/openflexure-microscope-server/{name}", "w"
        ) as f:
            json.dump(payload, f)
    except Exception:
        pass


class SmoothAutofocusThing(lt.Thing):
    """Autofocus from a continuous sharpness curve sampled while the stage sweeps.

    Flow (AF_GOALS.md): static reference frames (calibration, never parks) ->
    near sweep +-110u -> far sweep 2048u -> claw (sharp, pooled16 ballpark) ->
    tiny sweep +-180u -> park. Every wait is a happens-before: an
    M400-blocked move return or a frame-count Event predicate. No timed
    sleeps in the flow.
    """

    _class_settings = {"validate_properties_on_set": True}
    _cam: BaseCamera = lt.thing_slot()
    _stage: BaseStage = lt.thing_slot()
    _SESS_Z: Optional[float] = None
    _SESS_STATIC: float = 0.0
    _SESS_TS: float = 0.0
    _FAR_DIR: int = 1
    _CAL_FWHM: Optional[float] = None
    _CAL_CONTRAST: Optional[float] = None
    _CAL_BACKLASH: Optional[float] = None
    _CAL_PEAK_SHARP: Optional[float] = None
    _CAL_ACCEL: Optional[float] = None
    _CAL_LOADED: bool = False
    _seg_floor_et: Optional[float] = None
    _last_leg_samples: list = []

    def _hw_steps_param(self, delta_hw):
        inv = getattr(self._stage, "axis_inverted", {}) or {}
        return -delta_hw if inv.get("z", False) else delta_hw

    def _stage_z(self) -> float:
        try:
            return float(self._last_hw)
        except Exception:
            try:
                return float(self._stage.position["z"])
            except Exception:
                return float("nan")

    def _sync_move(
        self,
        monitor: StreamSharpnessMonitor,
        target: float,
        velocity: float,
        mark: int,
        labeled: bool = True,
    ):
        """One leg: queue the move (non-blocking POST), then wait for the
        motion_report websocket telemetry to show the motion start and
        drain. Labels come from real reported positions, not a modeled
        window. The OFM cache is re-synced after every exit unless the
        falling-edge push certified it or an unlabeled leg completed on
        geometry (commanded target is then the truth)."""
        try:
            return self._sync_move_inner(
                monitor, target, velocity, mark, labeled
            )
        finally:
            if not labeled:
                self._last_hw = target

    def _sync_move_inner(
        self,
        monitor: StreamSharpnessMonitor,
        target: float,
        velocity: float,
        mark: int,
        labeled: bool = True,
    ):
        cur = self._last_hw
        delta = target - cur
        if abs(delta) < 0.5:
            return None, []
        seconds = abs(delta) / velocity
        _log(
            f"move target={target:.2f} cur={cur:.2f} delta={delta:.2f} "
            f"v={velocity:.0f} t={time.time():.3f}"
        )
        motion_since = getattr(self._stage, "motion_since", None)
        motion_wait = getattr(self._stage, "motion_wait", None)
        if motion_since is None:
            self._stage.move_relative_z_timed(
                self._hw_steps_param(int(round(delta))), seconds
            )
            self._last_hw = target
            return None, []
        hoff = self._stage.handshake_offset()
        seg_floor = self._seg_floor_et
        ztol = max(2.0, 0.005 * abs(delta))

        def _marker_seen():
            for s in motion_since(0):
                if (
                    len(s) > 4
                    and s[4] is not None
                    and (seg_floor is None or s[4] > seg_floor)
                    and abs(s[2]) < 0.5
                    and not s[3]
                    and abs(s[1] - target) <= ztol
                ):
                    return s
            return None

        self._stage.move_relative_z_timed(
            self._hw_steps_param(int(round(delta))), seconds
        )
        self._last_hw = target
        marker = _marker_seen()
        prev_n = None
        stall = 0
        while marker is None and stall < 3:
            if motion_wait is None:
                break
            motion_wait(timeout=5.0)
            n = len(motion_since(0))
            if prev_n is not None and n == prev_n:
                stall += 1
            else:
                stall = 0
            prev_n = n
            marker = _marker_seen()
        if marker is None:
            _log(
                f"LEG TELESTALL target={target:.1f} "
                f"t={time.time():.3f}"
            )
            return None, []
        m_et = marker[4]
        self._seg_floor_et = m_et
        raw = [
            s for s in motion_since(0)
            if len(s) > 4 and s[4] is not None
            and (seg_floor is None or s[4] > seg_floor)
            and s[4] <= m_et
        ]
        raw.sort(key=lambda s: s[4])
        self._last_leg_samples = raw
        _log(
            f"LEG MARKER n={len(raw)} z={marker[1]:.1f} "
            f"et={m_et:.3f} t={time.time():.3f}"
        )
        if not labeled:
            return None, []
        if hoff is None:
            traj = [(s[0], s[1]) for s in raw]
            frames = list(monitor.frames_since(mark))
        else:
            traj = [(s[4] - hoff, s[1]) for s in raw]
            t_lo = traj[0][0]
            t_hi = traj[-1][0]
            frames = [
                f for f in monitor.frames_since(mark)
                if t_lo <= f[0] <= t_hi
            ]
        try:
            _dump_json("smooth_af_legdump.json", {
                "leg": [cur, target, velocity, seconds],
                "samples": [
                    [s[0], s[1], s[2], s[3],
                     s[4] if len(s) > 4 else None] for s in raw
                ],
                "hoff": hoff,
                "traj": traj,
                "n_frames": len(frames),
                "t0f": frames[0][0] if frames else None,
                "tNf": frames[-1][0] if frames else None,
            })
        except Exception:
            pass
        return traj, frames

    def _label_leg(self, frames, z_a: float, z_b: float, traj,
                   velocity: float = 0.0):
        """Label frames by interpolating real motion_report positions at
        each frame's timestamp. traj is the telemetry sample list."""
        if len(frames) < 6 or not traj:
            return []
        zs = [(t, z) for (t, z) in traj]
        t0, t1 = zs[0][0], zs[-1][0]
        if t1 <= t0:
            return []
        pts = []
        for (t, cbr, s, p) in frames:
            if t < t0 or t > t1:
                continue
            zq = zs[0][1]
            for i in range(1, len(zs)):
                if zs[i - 1][0] <= t <= zs[i][0]:
                    dt = zs[i][0] - zs[i - 1][0]
                    f = 0.0 if dt <= 0 else (t - zs[i - 1][0]) / dt
                    zq = zs[i - 1][1] + f * (zs[i][1] - zs[i - 1][1])
                    break
            pts.append((zq, cbr, s, p))
        pts.sort()
        b = type(self)._CAL_BACKLASH
        if b:
            shift = b if z_b > z_a else -b
            pts = [(q[0] + shift,) + tuple(q[1:]) for q in pts]
        if velocity <= 1000.0:
            _cp = median_filter_curve(
                [CurvePoint(z=q[0], sharpness=q[2]) for q in pts], window=3
            )
            return [
                (q[0], q[1], _cp[i].sharpness, q[3])
                for i, q in enumerate(pts)
            ]
        return pts

    def _claw(self, pts, tag: str):
        """Pooled16 ballpark first (step-2 detector), sharp cluster fallback.
        Returns (best_z, tag_str)."""
        if len(pts) < 6:
            return None, "|pooledflat" + tag
        p_vals = [p for (_z, _c, _s, p) in pts]
        p_med = statistics.median(p_vals)
        p_max = max(p_vals)
        thr = p_med + 0.35 * max(p_max - p_med, 1e-9)
        pp = [(z, p) for (z, _c, _s, p) in pts if p >= thr]
        if len(pp) >= 3:
            zp = sorted(z for z, _p in pp)
            if zp[-1] - zp[0] <= 260.0:
                hit = sum(z * p for z, p in pp) / sum(p for z, p in pp)
                return hit, f"|claw_pool{hit:.0f}(dp{p_max - p_med:.2f}){tag}"
        s_vals = [s for (_z, _c, s, _p) in pts]
        s_med = statistics.median(s_vals)
        contrast = type(self)._CAL_CONTRAST
        bar = ((contrast * 0.85) if contrast else _MEDIAN_BAR) * max(
            s_med, 1e-9
        )
        above = [i for i, q in enumerate(pts) if q[2] >= bar]
        runs = []
        run = []
        for i in above:
            if run and i == run[-1] + 1:
                run.append(i)
            else:
                if len(run) >= 2:
                    runs.append(run)
                run = [i]
        if len(run) >= 2:
            runs.append(run)
        for cl in runs:
            zs = [pts[i][0] for i in cl]
            if zs[-1] - zs[0] <= 120.0:
                hit = sum(pts[i][0] * pts[i][2] for i in cl) / sum(
                    pts[i][2] for i in cl)
                return hit, f"|claw_sharp{hit:.0f}{tag}"
        far_bar = (
            (contrast * 0.45) if contrast else _FAR_MEDIAN_BAR
        ) * max(s_med, 1e-9)
        singles = [
            pts[i] for i in range(len(pts)) if pts[i][2] >= far_bar
        ]
        if singles:
            s_best = max(q[2] for q in singles)
            z_best = next(q[0] for q in singles if q[2] == s_best)
            return z_best, f"|claw_far{s_best:.0f}@{z_best:.0f}{tag}"
        return None, "|pooledflat" + tag

    def _refine_hit(self, pts, claw_z: float) -> float:
        """Re-center a claw hit on the sharpest frame within +-60u."""
        if not pts:
            return claw_z
        local = [q for q in pts if abs(q[0] - claw_z) <= 60.0]
        if not local:
            return claw_z
        return max(local, key=lambda q: q[2])[0]

    def _cal_near_velocity(self) -> float:
        """Near sweep speed at the measured dynamic-hysteresis knee.

        Benchmarked on the 20.7u-FWHM peak: full contrast holds to
        250 u/s and collapses by 413 u/s -> 12 u/s per u of FWHM.
        """
        fwhm = type(self)._CAL_FWHM
        if not fwhm or not type(self)._CAL_CONTRAST:
            return 250.0
        return max(150.0, min(1000.0, 12.0 * fwhm))

    def _fit_gate(self, pts, static_ref: float = 0.0) -> Optional[tuple]:
        """Curve-relative acceptance: gaussian fit + median bar + edge margin,
        plus an absolute bar against the static reference."""
        if len(pts) < 6:
            return None
        cp = [CurvePoint(z=q[0], sharpness=q[2]) for q in pts]
        fit = fit_log_gaussian(cp)
        if not fit.ok:
            return None
        s_vals = [q[2] for q in pts]
        s_sorted = sorted(s_vals)
        s_floor = s_sorted[max(0, len(s_sorted) * 3 // 10 - 1)]
        best = max(s_vals)
        contrast = type(self)._CAL_CONTRAST
        bar = ((contrast * 0.85) if contrast else 1.5) * max(s_floor, 1e-9)
        if best < bar:
            return None
        cal_peak = type(self)._CAL_PEAK_SHARP
        if cal_peak and static_ref > 0:
            cal_peak = min(cal_peak, 1.9 * static_ref)
        if cal_peak and best < 0.85 * cal_peak:
            return None
        z_lo, z_hi = pts[0][0], pts[-1][0]
        arg = max(pts, key=lambda q: q[2])[0]
        span = z_hi - z_lo
        if span > 0 and not (
            z_lo + _EDGE_MARGIN * span <= arg <= z_hi - _EDGE_MARGIN * span
        ):
            return None
        if fit.fwhm is not None and fit.fwhm > 60.0 and best >= 1.6 * max(
            s_floor, 1e-9
        ):
            top = sorted(pts, key=lambda q: q[2], reverse=True)
            top_n = top[: max(2, len(top) // 5)]
            wsum = sum(q[2] for q in top_n)
            if wsum > 0:
                z_fit = (
                    sum(q[0] * q[2] for q in top_n) / wsum
                )
                return (
                    FitOutcome(
                        ok=True, peak=z_fit, fwhm=None, r2=fit.r2,
                        snr=fit.snr
                    ),
                    best,
                )
        if (fit.r2 is None or fit.r2 < 0.5) or (
            fit.snr is None or fit.snr < 1.2
        ):
            if best < 1.6 * max(s_floor, 1e-9):
                return None
            top = sorted(pts, key=lambda q: q[2], reverse=True)
            top_n = top[: max(2, len(top) // 5)]
            wsum = sum(q[2] for q in top_n)
            z_fit = (
                sum(q[0] * q[2] for q in top_n) / wsum if wsum > 0 else arg
            )
            return (
                FitOutcome(
                    ok=True, peak=z_fit, fwhm=None, r2=fit.r2, snr=fit.snr
                ),
                best,
            )
        return fit, best

    def _park(self, monitor: StreamSharpnessMonitor, target: float, mark: int):
        """Blocking park; settled means the move returned (trailing sync)."""
        self._sync_move(monitor, target, _PARK_VELOCITY, mark, labeled=False)
        return self._stage_z()

    def _load_cal(self) -> None:
        try:
            p = "/home/admin/openflexure-microscope-server/"
            p += "smooth_af_cal.json"
            import json as _json
            with open(p) as f:
                d = _json.load(f)
            a = float(d.get("accel") or 0.0)
            if 15000.0 < a < 120000.0:
                type(self)._CAL_ACCEL = a
                _log(f"CALACCEL loaded {a:.0f}")
            fw = d.get("fwhm_u")
            if fw is not None:
                fw = float(fw)
                if 2.0 < fw < 400.0:
                    type(self)._CAL_FWHM = fw
                    _log(f"CALFWHM loaded {fw:.1f}")
            ct = d.get("contrast")
            if ct is not None:
                type(self)._CAL_CONTRAST = float(ct)
        except Exception:
            pass
        finally:
            type(self)._CAL_LOADED = True

    def _accel_from_telemetry(self, seg_a, seg_b):
        """Measure acceleration from the emitted motion report series.

        Per leg: plateau = max |v|, rise ends at the first sample past
        90% plateau, a = v_p^2 / (2 d_rise). Triangle legs (no plateau)
        are skipped; plausible estimates are averaged.
        """
        ests = []
        for seg in (seg_a, seg_b):
            if not seg or len(seg) < 6:
                continue
            vs = [abs(s[2]) for s in seg]
            vp = max(vs)
            if vp < 500.0:
                continue
            i90 = next(
                (i for i, v in enumerate(vs) if v >= 0.9 * vp), None
            )
            if not i90:
                continue
            d = abs(seg[i90][1] - seg[0][1])
            if d < 10.0:
                continue
            ests.append(vp * vp / (2.0 * d))
        if not ests:
            _log("ACCELTEL no-est")
            return None
        a = sum(ests) / len(ests)
        if not (15000.0 < a < 120000.0):
            _log(f"ACCELTEL out-of-range a={a:.0f} ests={ests}")
            return None
        return a

    def _save_cal(self, key: str, value: float) -> None:
        try:
            p = "/home/admin/openflexure-microscope-server/"
            p += "smooth_af_cal.json"
            import json as _json
            try:
                with open(p) as f:
                    d = _json.load(f)
            except Exception:
                d = {}
            d[key] = value
            with open(p, "w") as f:
                _json.dump(d, f)
        except Exception:
            pass

    def _save_last_focus(
        self, monitor: "StreamSharpnessMonitor", mark: int, peak: float
    ) -> None:
        mark2 = len(monitor.times)
        monitor.wait_frames(4, mark2, timeout=0.5)
        post = monitor.frames_since(mark2)[-4:]
        focus_static = (
            statistics.median([f[2] for f in post]) if len(post) >= 3 else 0.0
        )
        _log(
            f"SAVE anchor={peak:.1f} focus_static={focus_static:.0f} "
            f"pre_static={type(self)._SESS_STATIC:.0f}"
        )
        type(self)._SESS_Z = float(peak)
        type(self)._SESS_TS = time.time()
        if focus_static > 0 and focus_static > type(self)._SESS_STATIC:
            type(self)._SESS_STATIC = float(focus_static)

    @contextmanager
    def _fast_stream(self, restore: bool = True, mode_name: str = "fast_preview"):
        prev_mode = None
        try:
            prev_mode = self._cam.streaming_mode
        except Exception:
            prev_mode = None
        switched = False
        try:
            if (
                prev_mode is not None
                and mode_name in self._cam.streaming_modes
                and prev_mode != mode_name
            ):
                self._cam._start_streaming(mode_name)
                switched = True
        except Exception:
            LOGGER.warning("Could not switch to fast_preview", exc_info=True)
            switched = False
        try:
            yield not switched
        finally:
            if switched and restore:
                try:
                    self._cam._start_streaming(mode_name)
                except Exception:
                    LOGGER.warning(
                        "Could not restore streaming mode", exc_info=True
                    )

    def _run_flow(self, monitor: StreamSharpnessMonitor) -> SmoothFocusResult:
        t0 = time.monotonic()
        mode = ""
        z0 = self._last_hw

        pre = monitor.frames_since(0)[-4:]
        static_med = (
            statistics.median([f[2] for f in pre]) if len(pre) >= 3 else 0.0
        )
        mode += f"|static{static_med:.0f}"

        sess_ref = type(self)._SESS_STATIC
        if not (
            static_med > 0 and sess_ref > 0 and static_med >= 0.7 * sess_ref
        ):
            mode += "|near_skip"
        else:
            d_near = 1 if type(self)._FAR_DIR == 1 else -1
            near_v = self._cal_near_velocity()
            near_mark = len(monitor.times)
            traj_a, frames_a = self._sync_move(
                monitor, z0 + d_near * _NEAR_SPAN, near_v, near_mark
            )
            pts_a = self._label_leg(
                frames_a, z0, z0 + d_near * _NEAR_SPAN, traj_a, near_v
            )
            near_mark2 = len(monitor.times)
            traj, frames = self._sync_move(
                monitor, z0 - d_near * _NEAR_SPAN, near_v, near_mark2
            )
            near_pts = self._label_leg(
                frames, z0 + d_near * _NEAR_SPAN, z0 - d_near * _NEAR_SPAN,
                traj, near_v
            )
            _dump_json("smooth_af_seekpts_near.json", {
                "z": [q[0] for q in near_pts],
                "s": [q[2] for q in near_pts],
                "c": [q[1] for q in near_pts],
                "p": [q[3] for q in near_pts],
            })
            gate = self._fit_gate(pts_a, static_med)
            if gate is not None:
                fit, best = gate
                mode += f"|near_up{fit.peak:.1f}"
                park_z = self._park(monitor, fit.peak, len(monitor.times))
                mode += f"|park{park_z:.1f}"
                self._save_last_focus(monitor, len(monitor.times), fit.peak)
                return SmoothFocusResult(
                    peak_z=park_z, mode=mode, fit=fit,
                    curve=[CurvePoint(z=q[0], sharpness=q[2]) for q in pts_a],
                    elapsed_s=time.monotonic() - t0,
                )
            near_pts = pts_a + near_pts
            near_pts.sort()
            gate = self._fit_gate(near_pts, static_med)
            if gate is not None:
                fit, best = gate
                mode += f"|near_fit{fit.peak:.1f}"
                park_z = self._park(monitor, fit.peak, len(monitor.times))
                mode += f"|park{park_z:.1f}"
                self._save_last_focus(monitor, len(monitor.times), fit.peak)
                return SmoothFocusResult(
                    peak_z=park_z, mode=mode, fit=fit,
                    curve=[CurvePoint(z=q[0], sharpness=q[2]) for q in near_pts],
                    elapsed_s=time.monotonic() - t0,
                )
            mode += "|near_miss"

        cur = self._stage_z()
        if not (cur == cur):
            cur = self._last_hw
        leg_a = cur + _FAR_SPAN
        leg_b = cur - _FAR_SPAN
        if not type(self)._CAL_LOADED:
            self._load_cal()
        if type(self)._CAL_ACCEL is None:
            far_v = _CAL_V_ACCEL
            mode += "|accelcal"
        else:
            far_v = _FAR_VELOCITY
        _log(
            f"FARREGIME_V v={far_v:.0f} cal={type(self)._CAL_ACCEL}"
        )

        segs = {}
        best_hit = None
        best_pts = None
        leg_hits = []
        far_exp_prev = None
        far_gain_prev = None
        try:
            far_exp_prev = self._cam.exposure_time
            far_gain_prev = self._cam.analogue_gain
        except Exception:
            far_exp_prev = far_gain_prev = None
        far_exp = 1000
        far_gain = 2.0
        try:
            self._cam.analogue_gain = far_gain
            self._cam.exposure_time = far_exp
            _log(f"FARREGIME exp={far_exp} gain={far_gain}")
        except Exception:
            LOGGER.warning("Far exposure regime failed", exc_info=True)
        for tag, target, prev_end in (("far", leg_a, cur), ("far2", leg_b, leg_a)):
            far_mark = len(monitor.times)
            traj, frames = self._sync_move(
                monitor, target, far_v, far_mark
            )
            leg_pts = self._label_leg(frames, prev_end, target, traj,
                                      far_v)
            segs[tag] = list(self._last_leg_samples or [])
            _dump_json(f"smooth_af_seekpts_{tag}.json", {
                "z": [q[0] for q in leg_pts],
                "s": [q[2] for q in leg_pts],
                "c": [q[1] for q in leg_pts],
                "p": [q[3] for q in leg_pts],
            })
            hit, claw_tag = self._claw(leg_pts, tag)
            if hit is not None:
                hit = self._refine_hit(leg_pts, hit)
            mode += claw_tag
            if hit is not None:
                leg_hits.append((tag, hit))
                if best_hit is None or max(
                    q[2] for q in leg_pts
                ) > best_pts[0]:
                    best_hit = hit
                    best_pts = (max(q[2] for q in leg_pts), leg_pts, tag)
            else:
                mode += "|nohit"

        try:
            if far_exp_prev is not None:
                self._cam.exposure_time = int(far_exp_prev)
            if far_gain_prev is not None:
                self._cam.analogue_gain = far_gain_prev
        except Exception:
            LOGGER.warning("Far exposure restore failed", exc_info=True)

        a_meas = self._accel_from_telemetry(
            segs.get("far"), segs.get("far2")
        )
        if a_meas is not None:
            _log(
                f"ACCELTEL a={a_meas:.0f} prev={type(self)._CAL_ACCEL} "
                f"v={far_v:.0f} SAVED"
            )
            type(self)._CAL_ACCEL = a_meas
            self._save_cal("accel", a_meas)

        if best_hit is not None:
            claw_z = best_hit
            fwhm = (
                type(self)._CAL_FWHM if type(self)._CAL_CONTRAST else None
            )
            span_h = max(
                60.0, min(_TINY_SPAN, (fwhm * 1.25) if fwhm else 60.0)
            )
            tiny_lo = max(claw_z - span_h, min(leg_a, leg_b))
            tiny_hi = min(claw_z + span_h, max(leg_a, leg_b))
            f_ref = fwhm if fwhm else 20.0
            tiny_v = max(
                _TINY_MIN_V,
                min(_TINY_VELOCITY, FPS * f_ref / 5.0),
            )
            tiny_mark = len(monitor.times)
            self._sync_move(
                monitor, tiny_lo, _PARK_VELOCITY, tiny_mark, labeled=False
            )
            tiny_mark2 = len(monitor.times)
            traj3, frames3 = self._sync_move(
                monitor, tiny_hi, tiny_v, tiny_mark2
            )
            tiny_pts = self._label_leg(frames3, tiny_lo, tiny_hi, traj3,
                                       tiny_v)
            _dump_json("smooth_af_seekpts_tiny.json", {
                "z": [q[0] for q in tiny_pts],
                "s": [q[2] for q in tiny_pts],
                "c": [q[1] for q in tiny_pts],
                "p": [q[3] for q in tiny_pts],
            })
            gate = self._fit_gate(tiny_pts)
            if gate is None and tiny_pts:
                s_floor_t = sorted(q[2] for q in tiny_pts)[
                    max(0, len(tiny_pts) * 3 // 10 - 1)
                ]
                t_arg = max(tiny_pts, key=lambda q: q[2])
                if (
                    t_arg[2] >= 1.25 * max(s_floor_t, 1e-9)
                    and abs(t_arg[0] - claw_z) <= 60.0
                ):
                    top_t = sorted(
                        tiny_pts, key=lambda q: q[2], reverse=True
                    )[: max(2, len(tiny_pts) // 5)]
                    wsum_t = sum(q[2] for q in top_t)
                    z_fit = (
                        sum(q[0] * q[2] for q in top_t) / wsum_t
                        if wsum_t > 0
                        else t_arg[0]
                    )
                    mode += "|tiny_near"
                    gate = (
                        FitOutcome(ok=True, peak=z_fit, fwhm=None),
                        t_arg[2],
                    )
            if gate is not None:
                fit, best = gate
                mode += f"|tiny_fit{fit.peak:.1f}"
                if fit.fwhm is not None:
                    type(self)._CAL_FWHM = fit.fwhm
                    self._save_cal("fwhm_u", fit.fwhm)
                    mode += f"|fwhm{fit.fwhm:.1f}"
                park_z = self._park(monitor, fit.peak, len(monitor.times))
                mode += f"|park{park_z:.1f}"
                self._save_last_focus(monitor, len(monitor.times), fit.peak)
                return SmoothFocusResult(
                    peak_z=park_z, mode=mode, fit=fit,
                    curve=[
                        CurvePoint(z=q[0], sharpness=q[2]) for q in tiny_pts
                    ],
                    elapsed_s=time.monotonic() - t0,
                )
            mode += "|tiny_fail"
            _dump_json("smooth_af_finefail.json", {
                "z": [q[0] for q in tiny_pts],
                "s": [q[2] for q in tiny_pts],
                "b": [q[1] for q in tiny_pts],
            })

        mode += "|no_anchor_refuse"
        ret_mark = len(monitor.times)
        self._sync_move(monitor, z0, _PARK_VELOCITY, ret_mark, labeled=False)
        mode += f"|return{self._stage_z():.1f}"
        raise NoFocusFoundError(mode)

    @lt.action
    def smooth_focus(
        self, params: SmoothAutofocusParams
    ) -> SmoothFocusResult:
        """Sweep autofocus: near sweep, far sweep, claw, tiny sweep, park.

        Raises NoFocusFoundError (and returns the stage to its start) rather
        than parking on an unverified peak.
        """
        t0 = time.monotonic()
        burst = 120.0
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
        self._last_hw = self._stage_z()
        try:
            monitor = StreamSharpnessMonitor(self._cam)
            _t_switch = time.monotonic()
            with self._fast_stream(
                restore=True, mode_name="crop990"
            ) as _fast, monitor:
                with self._cam._streaming_picamera() as picam:
                    picam.set_controls({"FrameRate": float(burst)})
                _t_ramp = time.monotonic()
                monitor.wait_frames(4, 0, timeout=2.0)
                _log(
                    f"SETUP ramp120 t={time.monotonic() - _t_ramp:.3f} "
                    f"n={len(monitor.times)}"
                )
                if not _fast:
                    _log(
                        f"SETUP mode_ready t={time.monotonic() - _t_switch:.3f}"
                    )
                else:
                    _log("SETUP mode_ready t=0.000 (already crop990)")
                _t_burst = time.monotonic()
                with _EncoderBurst(self._cam, burst):
                    _log(
                        f"SETUP burst_ready t={time.monotonic() - _t_burst:.3f}"
                    )
                    try:
                        _g, _e = _probe_regime(prev_gain, prev_exp)
                        self._cam.analogue_gain = _g
                        self._cam.exposure_time = _e
                        try:
                            result = self._run_flow(monitor)
                        except NoFocusFoundError as exc:
                            _log(
                                f"RUN total={time.monotonic() - t0:.2f} "
                                f"mode={exc} RETRY1"
                            )
                            try:
                                if hasattr(self._stage, "set_homed"):
                                    self._stage.set_homed()
                            except Exception:
                                pass
                            result = self._run_flow(monitor)
                        LOGGER.info(
                            "AF total=%.2fs mode=%s",
                            time.monotonic() - t0, result.mode,
                        )
                        _log(
                            f"RUN total={time.monotonic() - t0:.2f} "
                            f"mode={result.mode}"
                        )
                        return result
                    except NoFocusFoundError as exc:
                        _log(
                            f"RUN total={time.monotonic() - t0:.2f} "
                            f"mode={exc}"
                        )
                        raise
        finally:
            try:
                if prev_exp is not None:
                    self._cam.exposure_time = int(prev_exp)
                if prev_gain is not None:
                    self._cam.analogue_gain = prev_gain
            except Exception:
                LOGGER.warning("Exposure/gain restore failed", exc_info=True)
            try:
                if hasattr(self._stage, "update_position"):
                    self._stage.update_position()
            except Exception:
                LOGGER.warning("Stage position sync failed", exc_info=True)

class SmoothCompatAutofocus(SmoothAutofocusThing, AutofocusThing):
    """Drop-in AutofocusThing backed by the smooth autofocus engine.

    Keeps the smooth module's own units and gates: dz is interpreted as the
    total search span in stage units (not the stock stage's step semantics),
    and NoFocusFoundError from the engine surfaces as the scanner's
    ScanNoFocusFoundError.
    """

    _af_params = SmoothAutofocusParams()

    def _run_smooth(self, span: int) -> SharpnessDataArrays:
        params = self._af_params.model_copy(update={"search_span_steps": span})
        result = self.smooth_focus(params)
        z = [pt.z for pt in result.curve]
        s = [pt.sharpness for pt in result.curve]
        n = len(z)
        return SharpnessDataArrays(
            jpeg_times=np.array(z, dtype=float),
            jpeg_sizes=np.array(s, dtype=float),
            focus_foms=np.array(s, dtype=float),
            stage_times=np.zeros(n, dtype=float),
            stage_positions=[
                {"x": 0, "y": 0, "z": int(round(zi))} for zi in z
            ],
        )

    @lt.action
    def fast_autofocus(
        self,
        dz: int = 2000,
        start: Literal["centre", "base"] = "centre",
        sharpness_metric=None,
        record=None,
    ) -> SharpnessDataArrays:
        """Autofocus with the smooth engine; dz = total search span (stage units)."""
        span = max(8, min(int(dz), 192))
        try:
            return self._run_smooth(span)
        except NoFocusFoundError as exc:
            raise ScanNoFocusFoundError(str(exc)) from exc

    @lt.action
    def looping_autofocus(
        self,
        dz: int = 2000,
        start: Literal["centre", "base"] = "centre",
        sharpness_metric=None,
        record=None,
    ) -> tuple[list[float], list[float]]:
        """Run the smooth engine once; it already re-verifies and re-anchors."""
        try:
            self._run_smooth(max(8, min(int(dz), 192)))
        except NoFocusFoundError as exc:
            raise ScanNoFocusFoundError(str(exc)) from exc
        peaks = [float(self._stage.position["z"])]
        return peaks, []
