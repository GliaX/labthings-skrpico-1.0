import json
import logging
import statistics
import subprocess
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
    _frame_stash_start,
    _frame_stash_dump,
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
_FAR_VELOCITY = 2000.0
_NEAR_SPAN = 110.0
_NEAR_VELOCITY = 900.0
_TINY_SPAN = 180.0
_TINY_VELOCITY = 700.0
_PARK_VELOCITY = 3000.0
_MEDIAN_BAR = 1.6
_EDGE_MARGIN = 0.12


class NoFocusFoundError(RuntimeError):
    """No confident focus was found."""


class SmoothAutofocusParams(BaseModel):
    search_span_steps: int = Field(default=32, gt=0)
    wide_span_steps: int = Field(default=96, gt=0)
    wide_velocity: float = Field(default=600.0, gt=0)
    tip_velocity: float = Field(default=100.0, gt=0)
    tip_span_steps: int = Field(default=24, gt=0)
    coarse_velocity: float = Field(default=300.0, gt=0)
    fine_velocity: float = Field(default=150.0, gt=0)
    fine_span_steps: int = Field(default=3, gt=0)
    chunk_steps: int = Field(default=1, gt=0)
    min_r2: float = Field(default=0.5, ge=0, le=1)
    min_snr: float = Field(default=1.4, ge=1)
    min_abs_sharpness: float = Field(default=2.0, ge=0)
    min_abs_sharpness_fast: float = Field(default=2.0, ge=0)
    fwhm_min_steps: float = Field(default=0.4, gt=0)
    fwhm_max_steps: float = Field(default=24.0, gt=0)


_FOLLOW_LOG_PATH = (
    Path(__file__).resolve().parents[3] / "smooth_af_follow_log.jsonl"
)


@dataclass
class SmoothFocusResult:
    peak_z: float
    mode: str
    fit: FitOutcome
    curve: List[CurvePoint] = field(default_factory=list)
    elapsed_s: float = 0.0


@dataclass
class MetricProbeResult:
    sharpness: float
    brightness: float
    n_frames: int
    fps: float
    streaming_mode: str
    chroma: float = 0.0
    pooled: float = 0.0


class FollowParams(BaseModel):
    interval_s: float = Field(default=60.0, gt=0)
    iterations: int = Field(default=0, ge=0)
    search_span_steps: int = Field(default=16, gt=0)


class FollowEntry(BaseModel):
    index: int
    ts: float
    peak_z: Optional[float]
    elapsed_s: float
    mode: str
    error: Optional[str] = None


class FollowResult(BaseModel):
    stops: int
    peaks: List[float]
    entries: List[FollowEntry]


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
    """Stop both MJPEG encoders for a high-fps metric burst, restore after.

    The encoders (VideoCore) are the dominant heat source at high fps; during
    the burst nobody consumes the UI stream, so they are stopped. Must be
    entered INSIDE _fast_stream (a mode switch restarts the encoders). The
    restore mirrors _start_streaming's encoder half: the main preview
    recording plus the lores encoder, then the run's own fps cap.
    """

    def __init__(self, camera, fps: float, restore_fps: float = 120.0) -> None:
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
                    except Exception:
                        return self
                _frame_stash_start()
            self._stopped = True
        except Exception:
            self._stopped = False
        return self

    def __exit__(self, *exc) -> bool:
        _frame_stash_dump(
            "/home/admin/openflexure-microscope-server/smooth_af_frames.npz"
        )
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
    _MOVE_LAG: float = 0.45
    _fps_cap: float = 90.0

    def _hw_steps_param(self, delta_hw):
        inv = getattr(self._stage, "axis_inverted", {}) or {}
        return -delta_hw if inv.get("z", False) else delta_hw

    def _stage_z(self) -> float:
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
    ):
        """One blocking leg. Returns (traj, frames) with the window fully
        closed — the move's own trailing-sync return is the happens-before."""
        cur = self._last_hw
        delta = target - cur
        if abs(delta) < 0.5:
            return None, []
        seconds = max(0.15, abs(delta) / velocity)
        _log(
            f"move target={target:.2f} cur={cur:.2f} delta={delta:.2f} "
            f"v={velocity:.0f} t={time.time():.3f} "
            f"lag={type(self)._MOVE_LAG:.3f}"
        )
        t_post = time.monotonic()
        self._stage.move_relative_z_timed(
            self._hw_steps_param(int(round(delta))), seconds,
            trailing_sync=True,
        )
        t_ret = time.monotonic()
        lag_sample = t_ret - t_post - seconds
        if 0.0 < lag_sample < 2.0:
            type(self)._MOVE_LAG = min(
                1.0, max(0.05, 0.7 * type(self)._MOVE_LAG + 0.3 * lag_sample)
            )
        t_end = t_post + seconds + lag_sample
        t_start = t_end - seconds
        z_end = cur + delta
        self._last_hw = z_end
        traj = [(t_start, float(cur)), (t_end, float(z_end))]
        frames = [
            f for f in monitor.frames_since(mark)
            if t_start + 0.05 * seconds <= f[0] <= t_end - 0.02
        ]
        return traj, frames

    def _label_leg(self, frames, z_a: float, z_b: float, traj,
                   velocity: float = 0.0):
        """Dead-reckoning labels over the leg span; median-filtered below 1000 u/s."""
        if len(frames) < 6 or traj is None:
            return []
        t0, t1 = traj[0][0], traj[-1][0]
        if t1 <= t0:
            return []
        pts = [
            (
                z_a + (z_b - z_a) * (t - t0) / (t1 - t0),
                cbr, s, p,
            )
            for (t, cbr, s, p) in frames
        ]
        pts.sort()
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
        if len(pp) >= 6:
            zp = sorted(z for z, _p in pp)
            if zp[-1] - zp[0] <= 260.0:
                hit = sum(z * p for z, p in pp) / sum(p for z, p in pp)
                return hit, f"|claw_pool{hit:.0f}(dp{p_max - p_med:.2f}){tag}"
        s_vals = [s for (_z, _c, s, _p) in pts]
        s_med = statistics.median(s_vals)
        bar = _MEDIAN_BAR * max(s_med, 1e-9)
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
        return None, "|pooledflat" + tag

    def _fit_gate(self, pts) -> Optional[tuple]:
        """Curve-relative acceptance: gaussian fit + median bar + edge margin."""
        if len(pts) < 6:
            return None
        cp = [CurvePoint(z=q[0], sharpness=q[2]) for q in pts]
        fit = fit_log_gaussian(cp)
        if not fit.ok:
            return None
        s_vals = [q[2] for q in pts]
        s_med = statistics.median(s_vals)
        best = max(s_vals)
        if best < _MEDIAN_BAR * max(s_med, 1e-9):
            return None
        z_lo, z_hi = pts[0][0], pts[-1][0]
        arg = max(pts, key=lambda q: q[2])[0]
        span = z_hi - z_lo
        if span > 0 and not (
            z_lo + _EDGE_MARGIN * span <= arg <= z_hi - _EDGE_MARGIN * span
        ):
            return None
        if (fit.r2 is None or fit.r2 < 0.5) or (
            fit.snr is None or fit.snr < 1.2
        ):
            return None
        return fit, best

    def _park(self, monitor: StreamSharpnessMonitor, target: float, mark: int):
        """Blocking park; settled means the move returned (trailing sync)."""
        self._sync_move(monitor, target, _PARK_VELOCITY, mark)
        return self._stage_z()

    def _save_last_focus(
        self, peak: float, static: float = 0.0, curve_top: float = 0.0
    ) -> None:
        _log(f"SAVE anchor={peak:.1f} static={static:.0f} top={curve_top:.0f}")
        type(self)._SESS_Z = float(peak)
        type(self)._SESS_TS = time.time()
        cur_ref = type(self)._SESS_STATIC
        if curve_top > 0:
            if static >= 0.6 * curve_top and static >= cur_ref:
                type(self)._SESS_STATIC = float(static)
        elif static > 0 and static >= cur_ref:
            type(self)._SESS_STATIC = float(static)

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
                    self._cam._start_streaming(prev_mode or "default")
                except Exception:
                    LOGGER.warning(
                        "Could not restore streaming mode", exc_info=True
                    )

    def _run_flow(self, monitor: StreamSharpnessMonitor) -> SmoothFocusResult:
        t0 = time.monotonic()
        mode = ""
        z0 = self._last_hw

        mark0 = len(monitor.times)
        monitor.wait_frames(8, mark0, timeout=2.0)
        pre = monitor.frames_since(mark0)
        static_med = (
            statistics.median([f[2] for f in pre]) if len(pre) >= 4 else 0.0
        )
        mode += f"|static{static_med:.0f}"

        d_near = 1 if type(self)._FAR_DIR == 1 else -1
        near_mark = len(monitor.times)
        self._sync_move(
            monitor, z0 + d_near * _NEAR_SPAN, _NEAR_VELOCITY, near_mark
        )
        near_mark2 = len(monitor.times)
        traj, frames = self._sync_move(
            monitor, z0 - d_near * _NEAR_SPAN, _NEAR_VELOCITY, near_mark2
        )
        near_pts = self._label_leg(
            frames, z0 + d_near * _NEAR_SPAN, z0 - d_near * _NEAR_SPAN,
            traj, _NEAR_VELOCITY
        )
        _dump_json("smooth_af_seekpts_near.json", {
            "z": [q[0] for q in near_pts],
            "s": [q[2] for q in near_pts],
            "c": [q[1] for q in near_pts],
            "p": [q[3] for q in near_pts],
        })
        gate = self._fit_gate(near_pts)
        if gate is not None:
            fit, best = gate
            mode += f"|near_fit{fit.peak:.1f}"
            park_z = self._park(monitor, fit.peak, len(monitor.times))
            mode += f"|park{park_z:.1f}"
            self._save_last_focus(fit.peak, static_med, best)
            return SmoothFocusResult(
                peak_z=park_z, mode=mode, fit=fit,
                curve=[CurvePoint(z=q[0], sharpness=q[2]) for q in near_pts],
                elapsed_s=time.monotonic() - t0,
            )
        mode += "|near_miss"

        anchor = type(self)._SESS_Z
        cur = self._last_hw
        if anchor is not None and abs(anchor - cur) > 8.0:
            dirn = 1 if anchor > cur else -1
        else:
            type(self)._FAR_DIR = -type(self)._FAR_DIR
            dirn = type(self)._FAR_DIR
        leg_a = cur + dirn * _FAR_SPAN
        leg_b = cur - dirn * _FAR_SPAN

        far_mark = len(monitor.times)
        traj, frames = self._sync_move(monitor, leg_a, _FAR_VELOCITY, far_mark)
        leg1_pts = self._label_leg(frames, cur, leg_a, traj, _FAR_VELOCITY)
        _dump_json("smooth_af_seekpts_far1.json", {
            "z": [q[0] for q in leg1_pts],
            "s": [q[2] for q in leg1_pts],
            "c": [q[1] for q in leg1_pts],
            "p": [q[3] for q in leg1_pts],
        })
        claw_z, claw_tag = self._claw(leg1_pts, "far")
        mode += claw_tag
        if claw_z is None:
            far_mark2 = len(monitor.times)
            traj2, frames2 = self._sync_move(
                monitor, leg_b, _FAR_VELOCITY, far_mark2
            )
            leg2_pts = self._label_leg(frames2, leg_a, leg_b, traj2,
                                       _FAR_VELOCITY)
            _dump_json("smooth_af_seekpts_far2.json", {
                "z": [q[0] for q in leg2_pts],
                "s": [q[2] for q in leg2_pts],
                "c": [q[1] for q in leg2_pts],
                "p": [q[3] for q in leg2_pts],
            })
            claw_z, claw_tag2 = self._claw(leg2_pts, "far2")
            mode += claw_tag2

        if claw_z is not None:
            tiny_lo = max(claw_z - _TINY_SPAN, min(leg_a, leg_b))
            tiny_hi = min(claw_z + _TINY_SPAN, max(leg_a, leg_b))
            tiny_a = tiny_lo if dirn > 0 else tiny_hi
            tiny_b = tiny_hi if dirn > 0 else tiny_lo
            tiny_mark = len(monitor.times)
            self._sync_move(monitor, tiny_a, _PARK_VELOCITY, tiny_mark)
            tiny_mark2 = len(monitor.times)
            traj3, frames3 = self._sync_move(
                monitor, tiny_b, _TINY_VELOCITY, tiny_mark2
            )
            tiny_pts = self._label_leg(frames3, tiny_a, tiny_b, traj3,
                                       _TINY_VELOCITY)
            _dump_json("smooth_af_seekpts_tiny.json", {
                "z": [q[0] for q in tiny_pts],
                "s": [q[2] for q in tiny_pts],
                "c": [q[1] for q in tiny_pts],
                "p": [q[3] for q in tiny_pts],
            })
            gate = self._fit_gate(tiny_pts)
            if gate is not None:
                fit, best = gate
                mode += f"|tiny_fit{fit.peak:.1f}"
                park_z = self._park(monitor, fit.peak, len(monitor.times))
                mode += f"|park{park_z:.1f}"
                self._save_last_focus(fit.peak, static_med, best)
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
        self._sync_move(monitor, z0, _PARK_VELOCITY, ret_mark)
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
            try:
                self._last_hw = float(self._stage.position["z"])
            except Exception:
                self._last_hw = 0.0
            monitor = StreamSharpnessMonitor(self._cam)
            with self._fast_stream(
                restore=True, mode_name="crop990"
            ) as _fast, monitor:
                with _EncoderBurst(self._cam, burst):
                    try:
                        self._cam.analogue_gain = 2.0
                        cur_exp = self._cam.exposure_time
                        if cur_exp is None or cur_exp > 1000:
                            self._cam.exposure_time = 1000
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
                    except NoFocusFoundError:
                        _log(
                            f"RUN total={time.monotonic() - t0:.2f} "
                            f"mode=REFUSED"
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

    @lt.action
    def metric_probe(
        self,
        dwell_s: float = 0.8,
        burst_fps: float = 0.0,
    ) -> MetricProbeResult:
        """Read-only: median raw-Y sharpness at the current position.

        Calibration instrument — collects frames until the requested frame
        budget (dwell_s at ~100fps) is met, without moving the stage.
        """
        burst = float(burst_fps or 120.0)
        monitor = StreamSharpnessMonitor(self._cam)
        try:
            prev_gain = self._cam.analogue_gain
        except Exception:
            prev_gain = None
        try:
            prev_exp = self._cam.exposure_time
        except Exception:
            prev_exp = None
        with self._fast_stream(restore=True, mode_name="crop990"):
            with _EncoderBurst(self._cam, burst), monitor:
                try:
                    self._cam.analogue_gain = 2.0
                    if prev_exp is None or prev_exp > 1000:
                        self._cam.exposure_time = 1000
                    monitor.wait_frames(
                        max(8, int(dwell_s * 100)), 0, timeout=dwell_s * 2.5
                    )
                finally:
                    try:
                        if prev_exp is not None:
                            self._cam.exposure_time = int(prev_exp)
                        if prev_gain is not None:
                            self._cam.analogue_gain = prev_gain
                    except Exception:
                        LOGGER.warning(
                            "Exposure/gain restore failed", exc_info=True
                        )
        n = len(monitor.sharpness)
        med = statistics.median(monitor.sharpness) if monitor.sharpness else 0.0
        bright = (
            statistics.median(monitor.brightness) if monitor.brightness else 0.0
        )
        chroma = (
            statistics.median(monitor.chroma) if monitor.chroma else 0.0
        )
        pooled = (
            statistics.median(monitor.pooled) if monitor.pooled else 0.0
        )
        eff_fps = 0.0
        if n >= 3:
            ts_sorted = sorted(monitor.times)
            diffs = [b - a for a, b in zip(ts_sorted, ts_sorted[1:])]
            med_dt = statistics.median(diffs)
            eff_fps = 1.0 / med_dt if med_dt > 0 else 0.0
        try:
            mode = self._cam.streaming_mode
        except Exception:
            mode = "?"
        return MetricProbeResult(
            sharpness=med,
            brightness=bright,
            n_frames=n,
            fps=round(eff_fps, 1),
            streaming_mode=str(mode),
            chroma=chroma,
            pooled=pooled,
        )

    @lt.action
    def follow_focus(self, params: FollowParams) -> FollowResult:
        """Re-find focus every ``interval_s`` seconds, ``iterations`` times.

        Each cycle runs the normal ``smooth_focus`` search around the previous
        peak and appends the outcome to a JSONL log next to the server config.
        ``iterations=0`` runs until the action is cancelled. Peaks land in
        ``smooth_af_follow_log.jsonl`` — one JSON object per line.
        """
        t_start = time.time()
        stops = 0
        peaks: List[float] = []
        entries: List[FollowEntry] = []
        cycle_params = SmoothAutofocusParams(
            search_span_steps=params.search_span_steps
        )
        while True:
            idx = len(entries)
            t_cycle = time.time()
            error = None
            peak = None
            mode = ""
            elapsed = 0.0
            try:
                r = self.smooth_focus(cycle_params)
                peak = r.peak_z
                mode = r.mode
                elapsed = r.elapsed_s
                peaks.append(peak)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                mode = "error"
            entries.append(FollowEntry(
                index=idx,
                ts=round(t_cycle - t_start, 3),
                peak_z=peak,
                elapsed_s=round(elapsed, 3),
                mode=mode,
                error=error,
            ))
            try:
                with _FOLLOW_LOG_PATH.open("a") as f:
                    f.write(json.dumps(entries[-1].model_dump()) + "\n")
            except Exception:
                LOGGER.warning("Could not append to follow log", exc_info=True)
            stops += 1
            if params.iterations and stops >= params.iterations:
                break
            wait = params.interval_s - (time.time() - t_cycle)
            if wait > 0:
                time.sleep(wait)
        return FollowResult(stops=stops, peaks=peaks, entries=entries)


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
