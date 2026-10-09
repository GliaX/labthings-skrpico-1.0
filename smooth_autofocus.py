import io
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
from PIL import Image
from pydantic import BaseModel, Field

import labthings_fastapi as lt

from .autofocus import (
    AutofocusThing,
    NoFocusFoundError as ScanNoFocusFoundError,
    SharpnessDataArrays,
)
from .camera import BaseCamera
from .stage import BacklashCompensation, BaseStage
from .smooth_af_core import (
    CurvePoint,
    FitOutcome,
    cluster_curve,
    fit_log_gaussian,
    median_filter_curve,
)

LOGGER = logging.getLogger(__name__)


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
    min_abs_sharpness: float = Field(default=15.0, ge=0)
    min_abs_sharpness_fast: float = Field(default=15.0, ge=0)
    fwhm_min_steps: float = Field(default=0.4, gt=0)
    fwhm_max_steps: float = Field(default=24.0, gt=0)


_LAST_FOCUS_PATH = (
    Path(__file__).resolve().parents[3] / "smooth_af_last_focus.json"
)
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


@dataclass
class _Attempt:
    peak: Optional[float]
    z_guess: float
    confident: bool
    best_sharp: float
    mode: str
    outcome: FitOutcome
    combined: List[CurvePoint]


class StreamSharpnessMonitor:
    """Per-frame sharpness polled from the camera's lores MJPEG ringbuffer."""

    def __init__(self, camera: BaseCamera) -> None:
        self.camera = camera
        self.running = False
        self.times: List[float] = []
        self.sharpness: List[float] = []
        self.brightness: List[float] = []
        self._thread: Optional[threading.Thread] = None
        self._last_index = -1

    def _poll(self) -> None:
        import datetime as _dt

        min_ts = _dt.datetime.min
        while self.running:
            t0 = time.monotonic()
            snapshot = []
            try:
                rb = self.camera.lores_mjpeg_stream._ringbuffer
                snapshot = [(e.index, e.timestamp, e.frame) for e in rb]
            except Exception:
                snapshot = []
            new = sorted(
                (
                    s
                    for s in snapshot
                    if s[0] is not None and s[0] > self._last_index and s[2]
                ),
                key=lambda s: s[0],
            )
            for idx, ts, frame in new:
                if not ts or ts == min_ts:
                    continue
                try:
                    img = np.asarray(
                        Image.open(io.BytesIO(frame)).convert("L"),
                        dtype=np.uint8,
                    )
                    a = img.astype(np.int16)
                    dx = a[:, 1:] - a[:, :-1]
                    dy = a[1:, :] - a[:-1, :]
                    sharp = float(
                        (dx * dx).mean() + (dy * dy).mean()
                    )
                    bright = float(img.mean())
                except Exception:
                    continue
                self._last_index = idx
                self.times.append(ts.timestamp())
                self.sharpness.append(sharp)
                self.brightness.append(bright)
            time.sleep(max(0.005, 0.025 - (time.monotonic() - t0)))

    def __enter__(self) -> Self:
        self.running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        time.sleep(0.1)


class SmoothAutofocusThing(lt.Thing):
    """Autofocus from a continuous sharpness curve sampled while the stage sweeps."""

    _class_settings = {"validate_properties_on_set": True}
    _cam: BaseCamera = lt.thing_slot()
    _stage: BaseStage = lt.thing_slot()

    def _track_move(self, seconds: float, poll: float = 0.04) -> List[tuple]:
        traj: List[tuple] = []
        deadline = time.monotonic() + max(3.0, seconds * 2.5)
        time.sleep(0.05)

        def probe() -> tuple:
            vel = 0.0
            try:
                vel = self._stage.z_live_velocity()
            except Exception:
                pass
            try:
                pos = self._stage.z_live_position()
            except Exception:
                pos = traj[-1][1] if traj else 0.0
            if pos == 0.0 and traj and abs(traj[-1][1]) > 1.0:
                pos = traj[-1][1]
            return time.time(), pos, vel

        while time.monotonic() < deadline:
            t, pos, vel = probe()
            traj.append((t, pos))
            if vel > 1e-4:
                break
            time.sleep(poll)
        settled = 0
        while time.monotonic() < deadline:
            t, pos, vel = probe()
            traj.append((t, pos))
            if vel < 1e-4:
                settled += 1
                if settled >= 3:
                    break
            else:
                settled = 0
            time.sleep(poll)
        return traj

    def _hw_steps_param(self, delta_hw: int) -> int:
        inv = getattr(self._stage, "axis_inverted", {}) or {}
        return -delta_hw if inv.get("z", False) else delta_hw

    def _wait_idle(self, timeout: float = 1.0) -> None:
        try:
            self._stage.move_gcode_sync()
        except Exception:
            pass

    def _timed_move_to(
        self,
        monitor: StreamSharpnessMonitor,
        target_hw: float,
        velocity: float,
        keep_frames: bool = False,
        poll: float = 0.02,
        sync: bool = False,
        min_seconds: float = 0.15,
    ) -> List[tuple]:
        cur = self._last_hw
        delta_hw = int(round(target_hw - cur))
        try:
            Path(__file__).resolve().parents[3].joinpath(
                "smooth_af_moves.log"
            ).open("a").write(
                f"move target={target_hw} cur={cur} delta={delta_hw} "
                f"sync={sync} t={time.time():.3f}\n"
            )
        except Exception:
            pass
        self._last_move_t = [time.time(), None]
        if delta_hw == 0:
            return []
        seconds = max(min_seconds, abs(delta_hw) / velocity)
        mark = len(monitor.times)
        z_start = cur
        if sync:
            self._stage.move_relative_z_timed(
                self._hw_steps_param(delta_hw), seconds, trailing_sync=True
            )
            t_end = time.time()
        else:
            self._stage.move_relative_z_timed(
                self._hw_steps_param(delta_hw), seconds, trailing_sync=False
            )
            time.sleep(seconds + 0.05)
            t_end = time.time()
        t_start = t_end - seconds
        z_end = cur + delta_hw
        traj = [(t_start, float(z_start)), (t_end, float(z_end))]
        self._last_hw = z_end
        self._last_move_t[1] = time.time()
        if not keep_frames:
            del monitor.times[mark:]
            del monitor.sharpness[mark:]
            del monitor.brightness[mark:]
        return traj

    def _points_from_traj(
        self,
        monitor: StreamSharpnessMonitor,
        mark: int,
        traj: List[tuple],
    ) -> List[CurvePoint]:
        times = monitor.times[mark:]
        points: List[CurvePoint] = []
        if len(times) < 2 or len(traj) < 2:
            return points
        tt = np.array([t for t, _ in traj])
        zz = np.array([z for _, z in traj])
        for t, s, b in zip(times, monitor.sharpness[mark:], monitor.brightness[mark:]):
            if t < tt[0] or t > tt[-1]:
                continue
            points.append(CurvePoint(
                z=float(np.interp(t, tt, zz)), sharpness=s, brightness=b))
        return points

    def _sweep_timed(
        self,
        monitor: StreamSharpnessMonitor,
        z_start: int,
        z_end: int,
        velocity: float,
    ) -> List[CurvePoint]:
        cur = getattr(self, "_last_hw", None)
        lo = min(z_start, z_end)
        hi = max(z_start, z_end)
        if cur is None or not (lo - 2.0 <= cur <= hi + 2.0):
            self._timed_move_to(monitor, z_start, velocity, keep_frames=False)
            self._dwell(0.12)
        elif cur is not None and abs(z_end - cur) < 4.0:
            z_end = int(round(z_end + (hi - lo) * (1.0 if z_end >= z_start else -1.0)))
        act_fps = 0.0
        if len(monitor.times) >= 2:
            recent = [
                t for t in monitor.times if t >= time.time() - 1.5
            ]
            if len(recent) >= 2:
                span = recent[-1] - recent[0]
                if span > 0:
                    act_fps = (len(recent) - 1) / span
        span_abs = abs(z_end - z_start)
        natural = max(0.15, span_abs / velocity)
        seconds = natural
        if act_fps > 1.0:
            need = 12.0 / act_fps
            if need > seconds and span_abs > 8:
                seconds = need
        mark = len(monitor.times)
        traj = self._timed_move_to(
            monitor, z_end, velocity, keep_frames=True, min_seconds=seconds
        )
        times = monitor.times[mark:]
        points: List[CurvePoint] = []
        if len(times) < 2 or len(traj) < 2:
            return points
        tt = np.array([t for t, _ in traj])
        zz = np.array([z for _, z in traj])
        for t, s, b in zip(times, monitor.sharpness[mark:], monitor.brightness[mark:]):
            if t < tt[0] or t > tt[-1]:
                continue
            points.append(CurvePoint(
                z=float(np.interp(t, tt, zz)), sharpness=s, brightness=b))
        return points

    def _sweep_chunked(
        self,
        monitor: StreamSharpnessMonitor,
        z_start: int,
        z_end: int,
        chunk: int,
    ) -> List[CurvePoint]:
        direction = 1 if z_end >= z_start else -1
        n_chunks = max(1, int(abs(z_end - z_start) // chunk))
        bounds: List[tuple] = []
        for _ in range(n_chunks):
            t0 = time.time()
            z0 = float(self._stage.position["z"])
            self._stage.move_relative(
                z=direction * chunk, backlash_compensation=BacklashCompensation.Z_ONLY
            )
            z1 = float(self._stage.position["z"])
            t1 = time.time()
            bounds.append((t0, t1, z0, z1))
        times = np.array(monitor.times)
        points: List[CurvePoint] = []
        for k, (t0, t1, z0, z1) in enumerate(bounds):
            if k == 0 or k == len(bounds) - 1:
                continue
            i0 = int(np.searchsorted(times, t0))
            i1 = int(np.searchsorted(times, t1))
            for i in range(i0, i1):
                frac = (times[i] - t0) / (t1 - t0) if t1 > t0 else 0.0
                points.append(
                    CurvePoint(
                        z=z0 + (z1 - z0) * frac,
                        sharpness=monitor.sharpness[i],
                        brightness=monitor.brightness[i],
                    )
                )
        return points

    def _run_search(
        self,
        monitor: StreamSharpnessMonitor,
        anchor: int,
        params: SmoothAutofocusParams,
        timed: bool,
        tip: bool = True,
    ) -> _Attempt:
        bottom = anchor - params.search_span_steps // 2
        top = bottom + params.search_span_steps
        if timed:
            coarse = self._sweep_timed(
                monitor, bottom, top, params.coarse_velocity
            )
        else:
            coarse = self._sweep_chunked(
                monitor, bottom, top, params.chunk_steps
            )
        if not coarse:
            try:
                Path(__file__).resolve().parents[3].joinpath(
                    "smooth_af_last_curve.json"
                ).write_text(json.dumps({
                    "mode": "empty_coarse",
                    "monitor_times": len(monitor.times),
                    "last_t": monitor.times[-5:] if monitor.times else [],
                    "now": time.time(),
                }))
            except Exception:
                pass
            raise NoFocusFoundError(
                "No frames captured during the coarse sweep. "
                "Is the camera streaming?"
            )
        fit_c = fit_log_gaussian(coarse)
        if fit_c.ok:
            z_guess = fit_c.peak
            mode = "coarse_fit"
        else:
            z_guess = max(coarse, key=lambda p: p.sharpness).z
            mode = "argmax_fallback"
        if not tip:
            best_sharp = max((p.sharpness for p in coarse), default=0.0)
            peak = fit_c.peak if fit_c.ok else z_guess
            return _Attempt(
                peak, z_guess, fit_c.ok, best_sharp, mode, fit_c, coarse
            )
        f_span = max(params.fine_span_steps, 3 * (fit_c.fwhm or 0))
        f_span = min(max(f_span, 2.0), 10.0)
        fine_lo = int(round(z_guess - f_span))
        fine_hi = int(round(z_guess + f_span))
        fine: List[CurvePoint] = []
        if fine_hi - fine_lo >= 2:
            if timed:
                fine = self._sweep_timed(
                    monitor, fine_lo, fine_hi, params.fine_velocity
                )
            else:
                self._stage.move_relative(
                    z=fine_lo - top,
                    backlash_compensation=BacklashCompensation.Z_ONLY,
                )
                fine = self._sweep_chunked(
                    monitor, fine_lo, fine_hi, params.chunk_steps
                )
        combined = cluster_curve(median_filter_curve(coarse + fine))
        fit_f = fit_log_gaussian(combined)
        confident = (
            fit_f.ok
            and fit_f.r2 >= params.min_r2
            and fit_f.snr >= params.min_snr
        )
        if not confident:
            mode += "_fit_failed" if not fit_f.ok else "_gates_failed"
        best_sharp = max((p.sharpness for p in combined), default=0.0)
        peak = fit_f.peak if confident else z_guess
        return _Attempt(peak, z_guess, confident, best_sharp, mode, fit_f, combined)

    def _save_last_focus(self, peak: float) -> None:
        try:
            _LAST_FOCUS_PATH.write_text(
                json.dumps({"z": float(peak), "ts": time.time()})
            )
        except Exception:
            LOGGER.warning("Could not save last focus position", exc_info=True)

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

    @lt.action
    def dash_focus(
        self,
        span: int = 32,
        velocity: float = 120.0,
        passes: int = 4,
        camera_mode: str = "fast_preview",
    ) -> SmoothFocusResult:
        """Shuttle-dash autofocus: sweep back and forth, arrive at the fitted peak.

        Runs up to `passes` continuous sweeps (alternating direction, no park
        settle between legs), fitting after each pass. When a pass fits with
        r2 >= 0.7, the stage shuts directly onto the fitted peak as the final
        move — arriving at focus, not parking after it. Camera mode selectable
        (fast_preview / crop990) and left active on exit. Last-focus is saved
        only on a gated fit.
        """
        t0 = time.monotonic()
        timed = hasattr(self._stage, "move_relative_z_timed")
        if not timed:
            raise NoFocusFoundError("dash_focus needs move_relative_z_timed")
        fps = self._thermal_framerate()
        self._fps_cap = fps
        if fps < 90.0:
            try:
                self._cam.set_stream_framerate(fps)
            except Exception:
                LOGGER.warning("fps cap failed", exc_info=True)
        if hasattr(self._stage, "set_homed"):
            self._stage.set_homed()
        monitor = StreamSharpnessMonitor(self._cam)
        with self._fast_stream(restore=False, mode_name=camera_mode) as fast:
            time.sleep(0.1 if fast else 1.0)
            with monitor:
                self._last_hw = self._stage.z_live_position()
                z0 = self._last_hw
                anchor = z0
                mode = "shuttle"
                try:
                    anchor = float(json.loads(_LAST_FOCUS_PATH.read_text())["z"])
                    mode = "shuttle|lf"
                except Exception:
                    pass
                direction = 1.0 if anchor >= z0 else -1.0
                points: List[CurvePoint] = []
                fit_d = FitOutcome(ok=False, reason="no passes run")
                peak = z0
                t_sweep = 0.0
                t_fit = 0.0
                t_park = 0.0
                for p in range(passes):
                    if p == 0:
                        a, b = z0, anchor + direction * span
                    else:
                        a, b = self._last_hw, self._last_hw - direction * span
                    mark = len(monitor.times)
                    tp0 = time.monotonic()
                    traj = self._dash_sweep(monitor, float(a), float(b), velocity)
                    t_sweep += time.monotonic() - tp0
                    raw = self._points_from_traj(monitor, mark, traj)
                    tp0 = time.monotonic()
                    pts = cluster_curve(median_filter_curve(raw))
                    f = fit_log_gaussian(pts)
                    t_fit += time.monotonic() - tp0
                    if f.ok and (not fit_d.ok or f.r2 >= fit_d.r2):
                        points = pts
                        fit_d = f
                        peak = f.peak
                    if f.ok and f.r2 >= 0.7:
                        self._last_hw = self._stage.z_live_position()
                        tp0 = time.monotonic()
                        self._dash_move(
                            self._hw_steps_param(int(round(peak - self._last_hw))),
                            velocity,
                        )
                        t_park = time.monotonic() - tp0
                        mode += f"|fit@{p + 1}"
                        break
                else:
                    if fit_d.ok:
                        mode += "|fit|nopark"
                    else:
                        peak = (
                            max(points, key=lambda p: p.sharpness).z
                            if points
                            else z0
                        )
                        mode += "|argmax|nopark"
                if hasattr(self._stage, "update_position"):
                    self._stage.update_position()
                mode += (
                    f"|sw{t_sweep:.2f}fi{t_fit:.2f}pk{t_park:.2f}"
                    f"tot{time.monotonic() - t0:.2f}"
                )
        if fit_d.ok:
            self._save_last_focus(peak)
        return SmoothFocusResult(
            peak_z=float(peak),
            mode=mode,
            fit=FitOutcome(
                ok=fit_d.ok,
                reason=fit_d.reason,
                peak=fit_d.peak,
                fwhm=fit_d.fwhm,
                r2=fit_d.r2,
                snr=fit_d.snr,
            ),
            curve=sorted(points, key=lambda p: p.z),
            elapsed_s=time.monotonic() - t0,
        )

    def _dash_sweep(
        self,
        monitor: StreamSharpnessMonitor,
        z_start: float,
        z_end: float,
        velocity: float,
    ) -> List[tuple]:
        delta = int(round(z_end - z_start))
        if delta == 0:
            return []
        seconds = max(0.15, abs(delta) / velocity)
        mark = len(monitor.times)
        self._stage.move_relative_z_timed(self._hw_steps_param(delta), seconds)
        traj = self._dash_track(seconds, abs(delta))
        self._last_hw = traj[-1][1] if traj else z_end
        return traj

    def _dash_move(self, steps: int, velocity: float) -> None:
        if steps == 0:
            return
        seconds = max(0.1, abs(steps) / velocity)
        self._stage.move_relative_z_timed(steps, seconds)
        deadline = time.monotonic() + max(0.5, seconds + 0.5)
        prev = self._stage.z_live_position()
        stable = 0
        while time.monotonic() < deadline:
            time.sleep(0.02)
            cur = self._stage.z_live_position()
            if abs(cur - prev) < 0.05:
                stable += 1
                if stable >= 2:
                    break
            else:
                stable = 0
            prev = cur
        self._last_hw = cur

    def _dash_track(self, seconds: float, span_units: int) -> List[tuple]:
        traj: List[tuple] = []
        deadline = time.monotonic() + seconds + 1.0
        t_target = time.monotonic() + seconds
        prev = None
        while time.monotonic() < deadline:
            t = time.time()
            try:
                pos = self._stage.z_live_position()
            except Exception:
                pos = prev if prev is not None else 0.0
            if pos == 0.0 and prev is not None and abs(prev) > 1.0:
                pos = prev
            traj.append((t, pos))
            prev = pos
            if time.monotonic() > t_target and pos != 0.0:
                if prev is not None and abs(pos - prev) < 0.02:
                    stable = getattr(self, "_dash_stable", 0) + 1
                    self._dash_stable = stable
                    if stable >= 2:
                        break
                else:
                    self._dash_stable = 0
            time.sleep(0.012)
        self._dash_stable = 0
        return traj

    def _soc_temp(self) -> Optional[float]:
        try:
            out = subprocess.run(
                ["vcgencmd", "measure_temp"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            return float(out.split("=")[1].split("'")[0])
        except Exception:
            try:
                raw = Path("/sys/class/thermal/thermal_zone0/temp").read_text()
                return float(raw) / 1000.0
            except Exception:
                return None

    _fps_cap: float = 90.0

    def _dwell(self, base: float) -> None:
        """Sleep scaled to the active fps cap so dwells collect the same
        number of frames regardless of thermal throttling."""
        time.sleep(base * 90.0 / self._fps_cap)

    def _thermal_framerate(self, high: float = 78.0) -> float:
        """Pick a sustainable stream fps from the SoC temperature.

        Non-blocking thermal management: instead of pausing until the chip
        cools, run the sweeps at a frame rate the thermals can sustain. The
        adaptive sweep duration compensates automatically, so runs stay honest
        (>=12 frames per sweep) just slower.
        """
        temp = self._soc_temp()
        if temp is None:
            return 90.0
        if temp < 78.0:
            return 90.0
        if temp < 81.0:
            return 60.0
        if temp < 84.0:
            return 45.0
        return 30.0

    @lt.action
    def smooth_focus(self, params: SmoothAutofocusParams) -> SmoothFocusResult:
        """Continuous coarse sweep, then a continuous fine sweep; parks at the peak.

        Searches around the last successful focus if the stage has moved away from
        it. If the curve found is weak or the scene is too low-contrast, retries
        once with a doubled window; raises NoFocusFoundError rather than parking
        on a wrong plane.
        """
        t0 = time.monotonic()
        fps = self._thermal_framerate()
        self._fps_cap = fps
        monitor = StreamSharpnessMonitor(self._cam)
        session = (time.time() - getattr(self, "_last_focus_ts", 0.0)) < 60.0
        self._last_focus_ts = time.time()
        with self._fast_stream(
            restore=not session, mode_name="crop990"
        ) as fast, monitor:
            if fast and fps < 90.0:
                try:
                    self._cam.set_stream_framerate(fps)
                    LOGGER.info("Thermal fps cap %.0f", fps)
                except Exception:
                    LOGGER.warning("fps cap failed", exc_info=True)
            if not fast:
                time.sleep(0.3)
            t_settle = time.monotonic()
            LOGGER.info(
                "AF timing: settle=%.2fs", t_settle - t0
            )
            try:
                result = self._focus_once(
                    monitor,
                    params,
                    t0,
                    min_abs=params.min_abs_sharpness
                )
                LOGGER.info(
                    "AF timing: total=%.2fs mode=%s",
                    time.monotonic() - t0,
                    result.mode,
                )
                try:
                    Path(__file__).resolve().parents[3].joinpath(
                        "smooth_af_moves.log"
                    ).open("a").write(
                        f"RUN total={time.monotonic() - t0:.2f} "
                        f"mode={result.mode}\n"
                    )
                except Exception:
                    pass
                return result
            finally:
                LOGGER.info(
                    "AF timing: focus exited at %.2fs total",
                    time.monotonic() - t0,
                )

    def _focus_once(
        self,
        monitor: StreamSharpnessMonitor,
        params: SmoothAutofocusParams,
        t0: float,
        min_abs: float,
    ) -> SmoothFocusResult:
        timed = hasattr(self._stage, "move_relative_z_timed")
        if hasattr(self._stage, "set_homed"):
            self._stage.set_homed()
        if not getattr(self, "_af_exposure_sticky", False):
            try:
                old_exp = self._cam.exposure_time
                if old_exp is not None and old_exp > 2000:
                    self._cam.exposure_time = 2000
                    self._af_exposure_sticky = True
                    self._dwell(0.15)
            except Exception:
                LOGGER.warning("Exposure cap failed; running uncapped", exc_info=True)
        return self._focus_inner(monitor, params, t0, min_abs, timed)

    def _focus_inner(
        self,
        monitor: StreamSharpnessMonitor,
        params: SmoothAutofocusParams,
        t0: float,
        min_abs: float,
        timed: bool,
    ) -> SmoothFocusResult:
        if timed:
            self._dwell(0.12)
            z0 = int(round(self._stage.z_live_position()))
            self._last_hw = float(z0)
        else:
            z0 = int(round(float(self._stage.position["z"])))
        anchor = z0
        mode = ""
        try:
            last_focus = json.loads(_LAST_FOCUS_PATH.read_text())["z"]
            if abs(last_focus - z0) > 2:
                anchor = int(round(last_focus))
                mode = "last_focus_anchor|"
        except Exception:
            pass
        if timed:
            self._wait_idle()
        _t = {"start": time.time()}
        wide_params = params.model_copy(update={
            "search_span_steps": params.wide_span_steps,
            "coarse_velocity": params.wide_velocity,
        })
        cur_anchor = anchor
        tip_anchor: Optional[int] = None
        wide = None
        anchor_fresh = False
        try:
            _lf = json.loads(_LAST_FOCUS_PATH.read_text())
            anchor_fresh = (time.time() - float(_lf.get("ts", 0))) < 120.0
        except Exception:
            anchor_fresh = False
        if timed and abs(anchor - z0) <= 24 and anchor_fresh:
            small = self._run_search(
                monitor,
                anchor - 4,
                params.model_copy(update={
                    "search_span_steps": 24,
                    "coarse_velocity": params.tip_velocity,
                }),
                timed,
            )
            mode += "small150|" + small.mode + "|"
            small_ok = small.confident and small.best_sharp >= min_abs
            if not small_ok and small.best_sharp >= 2.0 * min_abs:
                small_ok = True
                mode += "small_argmax|"
            if small_ok:
                _t["wide_done"] = _t["tip_start"] = _t["tip_done"] = time.time()
                attempt = small
                combined_pts = small.combined
                peak_f = float(small.peak)
                self._last_hw = self._stage.z_live_position()
                self._timed_move_to(
                    monitor, peak_f, params.tip_velocity, sync=True
                )
                self._dwell(0.12)
                ver = monitor.sharpness[-20:]
                v_sm = statistics.median(ver) if ver else 0.0
                mode += f"smallpark{v_sm:.0f}"
                if v_sm >= min_abs:
                    self._save_last_focus(peak_f)
                    return SmoothFocusResult(
                        peak_z=peak_f,
                        mode=mode,
                        fit=FitOutcome(
                            ok=small.outcome.ok,
                            reason=small.outcome.reason,
                            peak=small.outcome.peak,
                            fwhm=small.outcome.fwhm,
                            r2=small.outcome.r2,
                            snr=small.outcome.snr,
                        ),
                        curve=sorted(combined_pts, key=lambda p: p.z),
                        elapsed_s=time.monotonic() - t0,
                    )
                mode += "|small_failed"
        hot_min = 0.6 * params.min_abs_sharpness_fast
        for slide in range(1):
            wide = self._run_search(
                monitor, cur_anchor, wide_params, timed, tip=False
            )
            if slide == 0:
                mode += "wide600|"
            n_w = len(wide.combined)
            q_w = max(3, n_w // 4)
            w_lo = statistics.fmean(p.sharpness for p in wide.combined[:q_w])
            w_hi = statistics.fmean(p.sharpness for p in wide.combined[-q_w:])
            if max(w_lo, w_hi) < hot_min:
                tip_anchor = int(round(wide.z_guess))
                break
            if max(w_lo, w_hi) < 3.0 * max(min(w_lo, w_hi), 1e-9):
                tip_anchor = int(round(wide.z_guess))
                break
            direction = 1.0 if w_hi > w_lo else -1.0
            mode += f"slide{slide + 1}|"
            cur_anchor = int(round(wide.z_guess)) + int(
                direction * params.wide_span_steps * 0.75
            )
        if wide is None:
            wide = self._run_search(monitor, anchor, wide_params, timed, tip=False)
        _t["wide_done"] = time.time()
        try:
            Path(__file__).resolve().parents[3].joinpath(
                "smooth_af_wide_debug.json"
            ).write_text(json.dumps({
                "n": len(wide.combined),
                "t_span": [
                    round(min((p.z for p in wide.combined), default=0), 1),
                    round(max((p.z for p in wide.combined), default=0), 1),
                ],
                "s_span": [
                    round(min((p.sharpness for p in wide.combined), default=0), 1),
                    round(max((p.sharpness for p in wide.combined), default=0), 1),
                ],
                "z_guess": round(wide.z_guess, 1),
                "monitor_n": len(monitor.times),
                "last_frames": [
                    round(t, 3) for t in monitor.times[-15:]
                ],
                "now": round(time.time(), 3),
            }))
        except Exception:
            pass
        tip_anchor = int(round(wide.z_guess))
        attempt = wide
        if timed:
            fine_lo = int(round(wide.z_guess)) - 24
            fine_hi = int(round(wide.z_guess)) + 8
            _t["tip_start"] = time.time()
            fine_pts = self._sweep_timed(
                monitor,
                fine_lo,
                fine_hi,
                params.tip_velocity,
            )
            _t["tip_done"] = time.time()
            if len(fine_pts) < 6:
                mode += f"|n{len(fine_pts)}"
            fit_f = fit_log_gaussian(fine_pts)
            best = max((p.sharpness for p in fine_pts), default=0.0)
            if fit_f.ok and fit_f.r2 >= params.min_r2 and fit_f.snr >= params.min_snr and best >= min_abs:
                attempt = _Attempt(
                    fit_f.peak, fit_f.peak, True, best, "fine_fit", fit_f, fine_pts
                )
                mode += f"fine{params.tip_velocity:.0f}|"
            else:
                attempt = _Attempt(
                    max(fine_pts, key=lambda p: p.sharpness).z if fine_pts else wide.z_guess,
                    wide.z_guess, False, best,
                    "fine_gates_failed", fit_f, fine_pts,
                )
                mode += f"fine{params.tip_velocity:.0f}|" + attempt.mode
            if not (attempt.confident and attempt.best_sharp >= min_abs):
                c_all = sorted(attempt.combined, key=lambda p: p.z)
                n_all = len(c_all)
                if n_all >= 8:
                    q = max(3, n_all // 4)
                    s_top = statistics.fmean(p.sharpness for p in c_all[-q:])
                    s_bot = statistics.fmean(p.sharpness for p in c_all[:q])
                    shelf = [p for p in c_all if p.sharpness >= 0.8 * s_top]
                    shelf_span = (
                        shelf[-1].z - shelf[0].z if len(shelf) >= 3 else 0.0
                    )
                    window_span = c_all[-1].z - c_all[0].z
                    if (
                        s_top >= min_abs
                        and s_top >= 3.0 * max(s_bot, 1e-9)
                        and window_span > 0
                        and shelf_span >= 0.25 * window_span
                    ):
                        peak_pl = sum(
                            p.z * p.sharpness for p in shelf
                        ) / sum(p.sharpness for p in shelf)
                        mode += f"|plateau_accept(top{s_top:.0f})"
                        attempt = _Attempt(
                            peak_pl, attempt.z_guess, True,
                            attempt.best_sharp, attempt.mode,
                            attempt.outcome, attempt.combined,
                        )
        else:
            attempt = self._run_search(monitor, anchor, params, timed)
            mode += attempt.mode
        if not (attempt.confident and attempt.best_sharp >= min_abs):
            c_all = sorted(attempt.combined, key=lambda p: p.z)
            n_all = len(c_all)
            plateau = False
            if n_all >= 8:
                q = max(3, n_all // 4)
                s_top = statistics.fmean(p.sharpness for p in c_all[-q:])
                s_bot = statistics.fmean(p.sharpness for p in c_all[:q])
                shelf = [p for p in c_all if p.sharpness >= 0.7 * s_top]
                shelf_span = shelf[-1].z - shelf[0].z if len(shelf) >= 3 else 0.0
                window_span = c_all[-1].z - c_all[0].z
                plateau = (
                    s_top >= min_abs
                    and s_top >= 3.0 * max(s_bot, 1e-9)
                    and window_span > 0
                    and shelf_span >= 0.25 * window_span
                )
                if plateau:
                    peak = sum(p.z * p.sharpness for p in shelf) / sum(
                        p.sharpness for p in shelf
                    )
                    mode += f"|plateau_accept(top{s_top:.0f})"
                    attempt = _Attempt(
                        peak, attempt.z_guess, True, attempt.best_sharp,
                        attempt.mode, attempt.outcome, attempt.combined,
                    )
            if not attempt.confident and wide is not None and wide.combined:
                c_w = sorted(wide.combined, key=lambda p: p.z)
                q_w2 = max(3, len(c_w) // 4)
                w_top = statistics.fmean(p.sharpness for p in c_w[-q_w2:])
                cur_pos = self._last_hw
                resc = int(round(max(c_w, key=lambda p: p.sharpness).z))
                if w_top >= min_abs and abs(resc - cur_pos) <= params.wide_span_steps:
                    mode += f"|wide_rescue{resc}"
                    self._timed_move_to(monitor, float(resc), params.tip_velocity)
                    self._dwell(0.25)
                    _vm = len(monitor.sharpness)
                    self._dwell(0.15)
                    vres = monitor.sharpness[_vm:]
                    v_med_r = statistics.median(vres) if vres else 0.0
                    mode += f"|rescue_v{v_med_r:.0f}(min{min_abs:.0f})"
                    if v_med_r >= min_abs:
                        attempt = _Attempt(
                            float(resc), float(resc), True, float(v_med_r),
                            attempt.mode, attempt.outcome, attempt.combined,
                        )
        if not (attempt.confident and attempt.best_sharp >= min_abs):
            zs = [p.z for p in attempt.combined]
            ss = [p.sharpness for p in attempt.combined]
            import math
            span = (
                math.log(max(ss)) - math.log(max(min(ss), 1e-9))
                if ss and max(ss) > 0
                else 0.0
            )
            try:
                Path(__file__).resolve().parents[3].joinpath(
                    "smooth_af_last_curve.json"
                ).write_text(json.dumps({
                    "mode": mode,
                    "curve": [
                        {"z": round(p.z, 3), "s": round(p.sharpness, 2)}
                        for p in sorted(attempt.combined, key=lambda p: p.z)
                    ],
                }))
            except Exception:
                pass
            if timed and attempt.combined:
                try:
                    home = json.loads(_LAST_FOCUS_PATH.read_text())["z"]
                    if abs(self._last_hw - home) > 8.0:
                        mode += f"|return{home:.0f}"
                        self._timed_move_to(
                            monitor, home, params.tip_velocity, sync=True
                        )
                except Exception:
                    pass
            raise NoFocusFoundError(
                f"No confident focus: best sharpness {attempt.best_sharp:.1f} "
                f"(threshold {min_abs}), mode {mode}, fit {attempt.outcome}; "
                f"curve n={len(attempt.combined)} z=[{min(zs):.1f},{max(zs):.1f}] "
                f"sharp=[{min(ss):.1f},{max(ss):.1f}] log-span={span:.2f}"
                if ss
                else f"No confident focus: empty curve, mode {mode}"
            )
        peak = float(attempt.peak)
        outcome = attempt.outcome
        combined = attempt.combined
        try:
            Path(__file__).resolve().parents[3].joinpath(
                "smooth_af_last_curve.json"
            ).write_text(json.dumps({
                "mode": mode,
                "curve": [
                    {"z": round(p.z, 3), "s": round(p.sharpness, 2)}
                    for p in sorted(combined, key=lambda p: p.z)
                ],
            }))
        except Exception:
            pass
        self._save_last_focus(peak)
        if timed:
            self._timed_move_to(
                monitor, peak, params.tip_velocity, sync=True
            )
            actual = self._last_hw
            park_err = abs(actual - peak)
            if park_err > 1.0:
                self._last_hw = actual
                self._timed_move_to(
                    monitor, peak, params.coarse_velocity
                )
                actual = self._last_hw
                park_err = abs(actual - peak)
            mode += f"|park_err{park_err:.1f}"
            self._dwell(0.12)
            verify = monitor.sharpness[-20:]
            v_med = statistics.median(verify) if verify else 0.0
            near = [p for p in combined if abs(p.z - actual) < 4.0]
            sweep_best = max((p.sharpness for p in near), default=v_med)
            if v_med < 0.7 * sweep_best and sweep_best > min_abs:
                mode += f"|park_verify{v_med:.0f}v{sweep_best:.0f}"
                arg = max(combined, key=lambda p: p.sharpness)
                mode += f"|argz{arg.z:.1f}s{arg.sharpness:.0f}"
                if abs(arg.z - actual) > 2.0 and arg.sharpness >= min_abs:
                    mode += f"|reseek{arg.z:.1f}"
                    re_params = params.model_copy(
                        update={"search_span_steps": min(params.search_span_steps, 16)}
                    )
                    re_anchor = int(round(arg.z))
                    attempt3 = self._run_search(monitor, re_anchor, re_params, timed)
                    mode += "|" + attempt3.mode
                    if attempt3.confident and attempt3.best_sharp >= min_abs:
                        attempt = attempt3
                        peak = float(attempt.peak)
                        combined = attempt.combined
                        self._save_last_focus(peak)
                    else:
                        combined = attempt3.combined
                    self._timed_move_to(monitor, peak - 1.0, params.coarse_velocity)
                    self._timed_move_to(monitor, peak, params.coarse_velocity)
                    actual = self._last_hw
                    self._dwell(0.12)
                    verify = monitor.sharpness[-20:]
                    v_med = statistics.median(verify) if verify else 0.0
                    mode += f"|reverify{v_med:.0f}"
            if v_med >= min_abs and not attempt.confident:
                self._save_last_focus(actual)
                mode += f"|dwellsave{v_med:.0f}"
            if hasattr(self._stage, "update_position"):
                self._stage.update_position()
        else:
            self._stage.move_absolute(
                z=int(round(peak)), backlash_compensation=BacklashCompensation.Z_ONLY
            )
        return SmoothFocusResult(
            peak_z=peak,
            mode=mode,
            fit=FitOutcome(
                ok=outcome.ok,
                reason=outcome.reason,
                peak=outcome.peak,
                fwhm=outcome.fwhm,
                r2=outcome.r2,
                snr=outcome.snr,
            ),
            curve=sorted(combined, key=lambda p: p.z),
            elapsed_s=time.monotonic() - t0,
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


class SmoothCompatAutofocus(AutofocusThing):
    """Drop-in AutofocusThing backed by the smooth autofocus engine.

    Keeps the smooth module's own units and gates: dz is interpreted as the
    total search span in stage units (not the stock stage's step semantics),
    sweeps run at the smooth module's velocities, and acceptance is the
    dwell-verified small-first path. Intended to be registered in place of
    ``AutofocusThing`` so scan workflows call the smooth engine directly.
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
        """Run the smooth engine once; it already re-verifies and re-anchors.

        The stock version loops up to 10 times until the peak lands mid-window;
        the smooth engine's own small-first/wide/verify chain does that job, so
        a single pass is issued here.
        """
        try:
            self._run_smooth(max(8, min(int(dz), 192)))
        except NoFocusFoundError as exc:
            raise ScanNoFocusFoundError(str(exc)) from exc
        peaks = [float(self._stage.position["z"])]
        return peaks, []
