import json
import logging
import queue
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Self

import labthings_fastapi as lt
import numpy as np
from picamera2.encoders import MJPEGEncoder
from pydantic import BaseModel, Field

from openflexure_microscope_server.things.camera.picamera import (
    PicameraStreamOutput,
)
from .autofocus import (
    AutofocusThing,
    NoFocusFoundError as ScanNoFocusFoundError,
    SharpnessDataArrays,
)
from .camera import BaseCamera
from .smooth_af_core import (
    CurvePoint,
    FitOutcome,
    fit_log_gaussian,
    median_filter_curve,
)
from .stage import BaseStage

sys.setswitchinterval(0.001)

LOGGER = logging.getLogger(__name__)

_FAR_SPAN = 1024.0
_FAR_VELOCITY = 2800.0
_FAR_MEDIAN_BAR = 1.25
_TINY_SPAN = 180.0
_TINY_VELOCITY = 700.0
_TINY_MIN_V = 100.0
_SCAN_SPAN = 30.0
_SCAN_LOCAL = threading.local()
_PARK_VELOCITY = 9000.0
_PARK_CREEP = 300.0
_CAL_V_ACCEL = 4900.0


def _SCAN_MODE() -> bool:
    return bool(getattr(_SCAN_LOCAL, "mode", False))


_MEDIAN_BAR = 1.6
_POOL_MIN_DP = 15.0
_EDGE_MARGIN = 0.12
FPS = 140.0


def _fps_safe_exp(cam) -> int:
    try:
        c = int(cam.exposure_range_max())
        if 100 <= c <= 50000:
            return c
    except Exception:
        pass
    return 6013


class NoFocusFoundError(RuntimeError):
    """No confident focus was found."""


class SmoothAutofocusParams(BaseModel):
    search_span_steps: int = Field(default=0, ge=0)


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

    def __init__(self, camera: BaseCamera, plane: str = "lores") -> None:
        self.camera = camera
        self.plane = plane
        self.times: List[float] = []
        self.sharpness: List[float] = []
        self.brightness: List[float] = []
        self.chroma: List[float] = []
        self.pooled: List[float] = []
        self.brenner: List[float] = []
        self._hook_lock = threading.Lock()
        self.frame_event = threading.Event()

    def _hook_cb(
            self, ts, sharp, bright, chroma=0.0, pooled=0.0, brenner=0.0
    ) -> None:
        with self._hook_lock:
            self.times.append(ts)
            self.sharpness.append(sharp)
            self.brightness.append(bright)
            self.chroma.append(chroma)
            self.pooled.append(pooled)
            self.brenner.append(brenner)
        self.frame_event.set()

    def frames_since(self, mark: int):
        return list(zip(
            self.times[mark:],
            self.chroma[mark:],
            self.sharpness[mark:],
            self.pooled[mark:],
        ))

    def wait_until(self, t_mono: float, timeout: float = 2.0) -> bool:
        """Block until the monitor's wall clock reaches t_mono
        (time.monotonic domain) or timeout elapses."""
        remain = t_mono - time.monotonic()
        if remain <= 0:
            return True
        time.sleep(min(remain, timeout))
        return True

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
        set_hook = getattr(self.camera, "set_af_metric_hook", None)
        if set_hook is None:
            raise RuntimeError("AF requires a YUV metric hook camera")
        set_hook(self._hook_cb, plane=self.plane)
        return self

    def __exit__(self, *_exc) -> None:
        try:
            self.camera.set_af_metric_hook(None)
        except Exception:
            pass


class _EncoderBurst:
    """Throttle sensor fps for a high-fps metric burst, restore after.

    The camera lives in crop990; the burst stops the encoders (nobody
    consumes the UI stream mid-AF) and the sensor stays at the metric
    rate across runs — restoring a lower FrameRate forces a sensor
    re-clock (~0.4s) on the next run, so it is deliberately not done.
    Restore restarts the encoders only.
    """

    def __init__(self, camera) -> None:
        self._cam = camera

    def __enter__(self) -> "_EncoderBurst":
        try:
            with self._cam._streaming_picamera() as picam:
                if picam.started:
                    try:
                        picam.stop_encoder()
                        _log("BURST stop_encoder t=%.3f" % (time.monotonic(),))
                    except Exception:
                        return self
        except Exception:
            pass
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
        except Exception as e:
            LOGGER.error("AF burst: camera restore failed: %s", e)
        return False


_LOGQ: "queue.Queue[str]" = queue.Queue()
_DUMPQ: "queue.Queue[tuple]" = queue.Queue()


def _writer_loop() -> None:
    """Background flusher for the AF hot-path writes.

    Synchronous SD writes between sweep legs stall the event chain
    (proven by the capture throttle incident); every hot-path write
    goes through these queues instead and is flushed off-thread.
    """
    while True:
        try:
            line = _LOGQ.get(timeout=0.5)
        except queue.Empty:
            line = None
        batch = []
        if line is not None:
            batch.append(line)
        try:
            while True:
                batch.append(_LOGQ.get_nowait())
        except queue.Empty:
            pass
        if batch:
            try:
                Path(__file__).resolve().parents[3].joinpath(
                    "smooth_af_moves.log"
                ).open("a").write("\n".join(batch) + "\n")
            except Exception:
                pass
        dumps = []
        try:
            while True:
                dumps.append(_DUMPQ.get_nowait())
        except queue.Empty:
            pass
        for name, payload in dumps:
            try:
                with open(
                        f"/home/admin/openflexure-microscope-server/{name}", "w"
                ) as f:
                    json.dump(payload, f)
            except Exception:
                pass


WRITER_THREAD = threading.Thread(
    target=_writer_loop, daemon=True, name="smooth-af-writer"
)


def _writer_kick() -> None:
    if not WRITER_THREAD.is_alive():
        WRITER_THREAD.start()


def _log(line: str) -> None:
    try:
        _LOGQ.put(line)
        _writer_kick()
    except Exception:
        pass


def _probe_regime(prev_gain, prev_exp):
    """Probe imaging regime from the pre-AF state.

    Pre-strobe this halved exposure to cut motion blur; the LED flash is
    the shutter now, so halving buys nothing and only pushed the frame
    out of its fps-safe window. State passes through unchanged and the
    camera clamps exposure to the mode's fps ceiling."""
    g_in = float(prev_gain or 0.0)
    if g_in >= 3.5:
        raise ValueError(
            "camera already in probe regime (gain %.1f) - a previous "
            "flow died before restore; refusing to stack halvings" % g_in
        )
    if not prev_exp:
        raise ValueError(
            "no live exposure reading - refusing to invent an AF regime"
        )
    return float(prev_gain) if prev_gain else 1.0, int(prev_exp)


def _dump_json(name: str, payload: dict) -> None:
    try:
        _DUMPQ.put((name, payload))
        _writer_kick()
    except Exception:
        pass


class SmoothAutofocusThing(lt.Thing):
    """Autofocus from a continuous sharpness curve sampled while the stage sweeps.

    Flow: static reference -> far sweep (both legs, 2800; early exit
    after leg 1 on a strong hit) ->
    claw -> tiny sweep -> two-stage park. Every wait is a happens-before
    on telemetry events; no timed waits, no M400.
    """

    _class_settings = {"validate_properties_on_set": True}
    _cam: BaseCamera = lt.thing_slot()
    _stage: BaseStage = lt.thing_slot()
    _SESS_STATIC: float = 0.0
    _CAL_FWHM: Optional[float] = None
    _CAL_CONTRAST: Optional[float] = None
    _CAL_ACCEL: Optional[float] = None
    _COOL_TIMER: Optional[threading.Timer] = None
    _COOLING: bool = False
    _COOL_DONE: Optional[threading.Event] = None
    _AF_BUSY: bool = False
    _LAST_REGIME: Optional[tuple] = None
    _COOL_DELAY: float = 3.0
    _CAL_LOADED: bool = False
    _TAU_FAR: Optional[float] = None
    _TAU_TINY: Optional[float] = None
    _REFINE_DZ: Optional[float] = None
    _REFINE_HIST: list = []
    _last_leg_samples: list = []
    _CORR: dict = {}
    _SCAN_LAST_Z: Optional[float] = None
    _SCAN_FAILS: int = 0
    _LAST_SCAN_TS: float = 0.0

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
        dz0 = float(getattr(self, "_frame_offset", 0.0))
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
        ztol = max(2.0, 0.005 * abs(delta))

        def _marker_seen():
            for s in motion_since(0):
                if (
                        len(s) > 4
                        and s[4] is not None
                        and abs(s[2]) < 0.5
                        and not s[3]
                        and abs(s[1] - (target + dz0)) <= ztol
                ):
                    return s
            return None

        clear = getattr(self._stage, "clear_motion_history", None)
        if clear is not None:
            clear()

        poll_start = getattr(self._stage, "_leg_poller_start", None)
        poll_stop = getattr(self._stage, "_leg_poller_stop_fn", None)
        if poll_start is not None:
            poll_start()
        try:
            self._stage.move_relative_z_timed(
                self._hw_steps_param(int(round(delta))), seconds
            )
        finally:
            if poll_stop is not None:
                poll_stop()
        self._last_hw = target
        # Certify at data arrival: klippy pushes motion updates every ~8ms;
        # if the terminal push's z missed ztol (one tick of travel) or got
        # skipped, certify from a one-shot query the moment we see v==0
        # data instead of burning the grace cap.
        grace_end = time.monotonic() + seconds * 2.0 + 0.2
        marker = _marker_seen()
        query_state = getattr(self._stage, "query_motion_state", None)
        while marker is None and time.monotonic() < grace_end:
            if motion_wait is not None:
                motion_wait(timeout=max(0.02, grace_end - time.monotonic()))
            marker = _marker_seen()
            if marker is None and query_state is not None:
                try:
                    st = query_state()
                    lp = st.get("live_position")
                    v_raw = st.get("live_velocity")
                    if lp is not None:
                        z_now = float(lp[2])
                        v_now = (
                            float(v_raw) if v_raw is not None else None
                        )
                        if (
                                (v_now is None or abs(v_now) < 0.5)
                                and abs(z_now - (target + dz0)) <= ztol
                        ):
                            et = time.monotonic() + getattr(
                                self._stage, "_et_handshake", 0.0
                            )
                            marker = (
                                et, z_now,
                                v_now if v_now is not None else 0.0,
                                0.0 if v_now is not None else 1.0,
                                et,
                            )
                            _log(
                                f"LEG FAST-CERTIFY target={target:.1f} "
                                f"z={z_now:.2f} t={time.time():.3f}"
                            )
                except Exception:
                    LOGGER.warning(
                        "fast-certify motion query failed", exc_info=True
                    )
        if marker is None:
            try:
                st = query_state() if query_state is not None else {}
                lp = st.get("live_position")
                v_raw = st.get("live_velocity")
                if lp is not None:
                    z_now = float(lp[2])
                    v_now = (
                        float(v_raw) if v_raw is not None else None
                    )
                    settle_end = time.monotonic() + 2.0
                    while v_now is not None and abs(v_now) >= 0.5:
                        if time.monotonic() >= settle_end:
                            break
                        if motion_wait is not None:
                            motion_wait(timeout=0.1)
                        st = query_state()
                        lp = st.get("live_position")
                        v_raw = st.get("live_velocity")
                        if lp is None:
                            break
                        z_now = float(lp[2])
                        v_now = (
                            float(v_raw)
                            if v_raw is not None
                            else None
                        )
                    z_ok = abs(z_now - (target + dz0)) <= ztol
                    v_ok = v_now is None or abs(v_now) < 0.5
                    if v_ok and not z_ok:
                        self._frame_offset = z_now - target
                        dz0 = self._frame_offset
                        z_ok = abs(z_now - (target + dz0)) <= ztol
                        _log(
                            f"FRAME LEARN offset={dz0:.2f} "
                            f"(live {z_now:.2f} vs target {target:.1f})"
                        )
                    if z_ok and v_ok:
                        et = (
                                time.monotonic()
                                + getattr(
                            self._stage, "_et_handshake", 0.0
                        )
                        )
                        marker = (
                            et, z_now,
                            v_now if v_now is not None else 0.0,
                            0.0 if v_now is not None else 1.0,
                            et,
                        )
                        _log(
                            f"LEG QUIET-CERTIFY target={target:.1f} "
                            f"z={z_now:.2f} v={v_now} "
                            f"t={time.time():.3f}"
                        )
            except Exception:
                LOGGER.warning(
                    "quiet-end motion query failed", exc_info=True
                )
        if marker is None:
            _log(
                f"LEG TELESTALL target={target:.1f} "
                f"t={time.time():.3f}"
            )
            self._telestall_run = True
            return None, []
        m_et = marker[4]
        raw = [
            s for s in motion_since(0)
            if len(s) > 4 and s[4] is not None
               and s[4] <= m_et
        ]
        raw.sort(key=lambda s: s[4])
        self._last_leg_samples = raw
        if not raw:
            _log(
                f"LEG EMPTY-TRAJ target={target:.1f} "
                f"t={time.time():.3f}"
            )
            self._telestall_run = True
            return None, []
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

    def _label_leg(self, frames, z_a: float, z_b: float, traj, velocity: float):
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
        if len(pp) >= 3 and p_max - p_med >= _POOL_MIN_DP:
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

    def _cancel_cooler(self) -> None:
        t = type(self)._COOL_TIMER
        if t is not None:
            t.cancel()
            type(self)._COOL_TIMER = None

    def _arm_cooler(self) -> None:
        self._cancel_cooler()
        type(self)._COOL_DONE = threading.Event()
        t = threading.Timer(self._COOL_DELAY, self._quiet_camera)
        t.daemon = True
        type(self)._COOL_TIMER = t
        t.start()

    def _quiet_camera(self) -> None:
        """10s after the last AF: sensor to 20fps, encoders ensured
        running, gain/exposure re-applied with readback. Deferred
        housekeeping outside the sweep chain; a new AF run cancels it."""
        type(self)._COOLING = True
        try:
            if getattr(self, "_AF_BUSY", False):
                return
            if _SCAN_MODE() or (
                    time.monotonic() - type(self)._LAST_SCAN_TS < 60.0
            ):
                _log("COOL skip: scan session active/recent")
                return
            held = getattr(self._cam, "stream_mode_held", False)
            mode = getattr(self._cam, "streaming_mode", "default")
            if mode != "default" and not held:
                for attempt in range(4):
                    if getattr(self, "_AF_BUSY", False):
                        break
                    try:
                        self._cam._stop_streaming(stop_web_stream=False)
                        self._cam._start_streaming("default")
                        _log("COOL mode=default (restored)")
                        break
                    except Exception as e:
                        if attempt == 3:
                            LOGGER.error(
                                "AF cooler: mode restore failed: %s", e
                            )
                        else:
                            time.sleep(1.5)
            try:
                with self._cam._streaming_picamera() as picam:
                    if not picam.started:
                        try:
                            picam.start()
                        except Exception as e:
                            LOGGER.error(
                                "AF cooler: camera start failed: %s", e)
                    if not getattr(picam, "is_recording", False):
                        bitrate = getattr(
                            self._cam, "mjpeg_bitrate", 10000000
                        )
                        mode_info = getattr(
                            self._cam, "streaming_modes", {}
                        ).get(getattr(self._cam, "streaming_mode", "default"))
                        use_lores = (
                                mode_info is not None
                                and getattr(
                            mode_info, "use_lores_as_preview", False
                        )
                        )
                        stream_name = "lores" if use_lores else "main"
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
                            LOGGER.error(
                                "AF cooler: main encoder start failed: %s", e)
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
                            LOGGER.error(
                                "AF cooler: lores encoder start failed: %s", e)
                    try:
                        picam.set_controls({"FrameRate": 20.0})
                        _log("COOL fps=20 enc=on")
                    except Exception as e:
                        LOGGER.error(
                            "AF cooler: fps restore failed: %s", e)
            except Exception as e:
                LOGGER.error("AF cooler failed: %s", e)
            if getattr(self, "_AF_BUSY", False):
                _log("COOL regime abort: AF started mid-cool")
                return
            reg = type(self)._LAST_REGIME
            if not reg:
                _log("COOL regime skip: no calibrated base known")
                return
            try:
                rb = self._cam.restore_calibration_raw()
                g2 = float(rb["gain"])
                e2 = int(rb["exposure"])
                ok = abs(g2 - reg[0]) < 0.05 and abs(
                    e2 - int(reg[1])
                ) <= 2
                _log(
                    f"COOL regime g={reg[0]:.2f} e={reg[1]} "
                    f"readback=({g2:.2f},{e2}) "
                    f"verify={'ok' if ok else 'MISMATCH'}"
                )
            except Exception:
                LOGGER.warning(
                    "AF cooler: regime restore failed", exc_info=True
                )
        finally:
            type(self)._COOLING = False
            _ev = type(self)._COOL_DONE
            if _ev is not None:
                _ev.set()

    def _park(self, monitor: StreamSharpnessMonitor, target: float, mark: int):
        """Two-stage park: fast to 30u short, then creep the last 30u at
        low speed approaching from one side — no overshoot, no backlash
        flip, park lands where tiny fit said. One submission, one
        certification for the pair."""
        cur = self._stage_z()
        sgn = 1.0 if target >= cur else -1.0
        near = target - sgn * 30.0
        if abs(near - cur) > 1.0:
            self._sync_move_pair(
                monitor, near, _PARK_VELOCITY, target, _PARK_CREEP
            )
        else:
            self._sync_move(
                monitor, target, _PARK_CREEP, len(monitor.times),
                labeled=False,
            )
        return self._stage_z()

    def _sync_move_pair(
            self, monitor: StreamSharpnessMonitor,
            z_mid: float, v_mid: float, z_end: float, v_end: float,
    ):
        """Two moves, one gcode submission, one certification at the pair
        end: the terminal sample of the SECOND move certifies the chain.
        Leg windows are still sliced per move by telemetry timestamps, so
        labels stay per-leg."""
        st = self._stage
        if getattr(st, "motion_since", None) is None:
            self._sync_move(monitor, z_mid, v_mid, len(monitor.times),
                            labeled=False)
            return self._sync_move(
                monitor, z_end, v_end, len(monitor.times)
            )
        cur = self._last_hw
        d1 = z_mid - cur
        d2 = z_end - z_mid
        if abs(d1) < 0.5 and abs(d2) < 0.5:
            return None, []
        mark = len(monitor.times)
        clear = getattr(st, "clear_motion_history", None)
        if clear is not None:
            clear()
        poll_start = getattr(st, "_leg_poller_start", None)
        poll_stop = getattr(st, "_leg_poller_stop_fn", None)
        if poll_start is not None:
            poll_start()
        try:
            s1 = abs(d1) / v_mid
            s2 = abs(d2) / v_end
            if getattr(st, "axis_inverted", {}).get("z", False):
                d1, d2 = -d1, -d2
            sp1 = abs(d1) / s1 if s1 > 0 else 0.0
            sp2 = abs(d2) / s2 if s2 > 0 else 0.0
            script = "G91\n"
            if abs(d1) >= 0.5:
                script += (
                    f"G1 Z{d1:.3f} S{sp1:.6f} F{sp1 * 60.0:.3f}\n"
                )
            if abs(d2) >= 0.5:
                script += (
                    f"G1 Z{d2:.3f} S{sp2:.6f} F{sp2 * 60.0:.3f}\n"
                )
            t0 = time.monotonic()
            st._send_gcode_script(st._client, script)
            total = s1 + s2
            grace_end = time.monotonic() + total * 2.0 + 0.2
            motion_wait = getattr(st, "motion_wait", None)
            motion_since = st.motion_since
            ztol = max(2.0, 0.005 * abs(d2))
            dz0 = float(getattr(self, "_frame_offset", 0.0))
            tgt2 = z_end + dz0
            marker = None
            query_state = getattr(st, "query_motion_state", None)
            while marker is None and time.monotonic() < grace_end:
                if motion_wait is not None:
                    motion_wait(timeout=max(
                        0.02, grace_end - time.monotonic()
                    ))
                for s in motion_since(0):
                    if (
                            len(s) > 4
                            and s[4] is not None
                            and abs(s[2]) < 0.5
                            and not s[3]
                            and abs(s[1] - tgt2) <= ztol
                    ):
                        marker = s
                        break
                if marker is None and query_state is not None:
                    try:
                        q = query_state()
                        lp = q.get("live_position")
                        v_raw = q.get("live_velocity")
                        if lp is not None:
                            z_now = float(lp[2])
                            v_now = (
                                float(v_raw)
                                if v_raw is not None else None
                            )
                            if (
                                    v_now is None or abs(v_now) < 0.5
                            ) and abs(z_now - tgt2) <= ztol:
                                et = time.monotonic() + getattr(
                                    st, "_et_handshake", 0.0
                                )
                                marker = (
                                    et, z_now,
                                    v_now if v_now is not None else 0.0,
                                    0.0 if v_now is not None else 1.0,
                                    et,
                                )
                    except Exception:
                        pass
            self._last_hw = z_end
            if marker is None:
                _log(f"PAIR TELESTALL z_end={z_end:.1f} "
                     f"t={time.time():.3f}")
                self._telestall_run = True
                return None, []
            m_et = marker[4]
            raw = [
                s for s in motion_since(0)
                if len(s) > 4 and s[4] is not None and s[4] <= m_et
            ]
            raw.sort(key=lambda s: s[4])
            # split the trajectory at the velocity boundary between the
            # two segments: last |v| <= v_mid*1.3 before the final ramp
            split_i = 0
            for i, s in enumerate(raw):
                if abs(s[2]) <= v_mid * 1.3 and s[4] < m_et - s2 * 0.5:
                    split_i = i
            hoff_fn = getattr(st, "handshake_offset", None)
            hoff = hoff_fn() if hoff_fn is not None else None
            if hoff is not None:
                traj1 = [(s[4] - hoff, s[1]) for s in raw[:split_i + 1]]
                traj2 = [(s[4] - hoff, s[1]) for s in raw[split_i:]]
            else:
                traj1 = [(s[0], s[1]) for s in raw[:split_i + 1]]
                traj2 = [(s[0], s[1]) for s in raw[split_i:]]
            frames = list(monitor.frames_since(mark))
            return (traj1, traj2), frames
        finally:
            if poll_stop is not None:
                poll_stop()

    def _rest_static(
            self, monitor: StreamSharpnessMonitor, cap_s: float | None = None
    ) -> float:
        """Settled rest sharpness: median of a 12-frame window, repeated
        until two consecutive windows agree within 4%% (plateau) or the
        cap expires. Motion-arrival gated, never a fixed sleep. Cap
        defaults to 2.0 s (button flow) or 0.2 s in scan flow."""
        if cap_s is None:
            cap_s = 0.2 if _SCAN_MODE() else 2.0
        m = len(monitor.times)
        prev = None
        t_end = time.monotonic() + cap_s
        while True:
            monitor.wait_frames(12, m, timeout=max(0.15, t_end - time.monotonic()))
            win = monitor.frames_since(m)[-12:]
            if len(win) < 8:
                break
            med = statistics.median([f[2] for f in win])
            if (
                    prev is not None and max(med, prev, 1.0) < 1.05 * min(med, prev)
            ) or time.monotonic() >= t_end:
                return med
            prev = med
            m = len(monitor.times)
        return prev if prev is not None else 0.0

    def _refine_landing(
            self, monitor: StreamSharpnessMonitor, park_z: float
    ) -> tuple:
        """Rest-truth arbiter after the park: banked direct jump when
        known, else +-12u probes with one +-24u extension; 16-frame
        medians, +8% to accept a step. Skips entirely when the landing
        already reads >=200 (clearly focused in scan regime)."""
        s0 = self._rest_static(monitor)
        if s0 >= 200.0:
            _log("REFINE skip: landing sharp (%.0f)" % s0)
            return park_z, s0
        best_z, best_s = park_z, s0
        probes = 0
        dz = type(self)._REFINE_DZ
        if dz is not None and abs(dz) >= 6.0:
            self._sync_move(
                monitor, park_z + dz, _PARK_CREEP,
                len(monitor.times), labeled=False,
            )
            zr = self._stage_z()
            s1 = self._rest_static(monitor)
            probes = 1
            _log(
                "REFINE banked dz=%.1f z=%.1f s=%.0f prev=%.0f"
                % (dz, zr, s1, best_s)
            )
            if s1 > best_s * 1.08:
                best_z, best_s = zr, s1
                type(self)._REFINE_HIST.append(dz)
                del type(self)._REFINE_HIST[:-5]
            else:
                self._sync_move(
                    monitor, best_z, _PARK_CREEP,
                    len(monitor.times), labeled=False,
                )
                best_z = self._stage_z()
                vdz = -24.0 if dz > 0 else 24.0
                self._sync_move(
                    monitor, best_z + vdz, _PARK_CREEP,
                    len(monitor.times), labeled=False,
                )
                zr = self._stage_z()
                s1 = self._rest_static(monitor)
                _log(
                    "REFINE scout %+.0f z=%.1f s=%.0f prev=%.0f"
                    % (vdz, zr, s1, best_s)
                )
                if s1 > best_s * 1.08:
                    best_z, best_s = zr, s1
                    type(self)._REFINE_HIST.append(vdz)
                    del type(self)._REFINE_HIST[:-5]
                    self._sync_move(
                        monitor, best_z + (12.0 if vdz > 0 else -12.0),
                        _PARK_CREEP, len(monitor.times), labeled=False,
                    )
                    zr = self._stage_z()
                    s1 = self._rest_static(monitor)
                    _log(
                        "REFINE climb z=%.1f s=%.0f prev=%.0f"
                        % (zr, s1, best_s)
                    )
                    if s1 > best_s * 1.08:
                        best_z, best_s = zr, s1
                        type(self)._REFINE_HIST.append(
                            12.0 if vdz > 0 else -12.0
                        )
                        del type(self)._REFINE_HIST[:-5]
                    else:
                        self._sync_move(
                            monitor, best_z, _PARK_CREEP,
                            len(monitor.times), labeled=False,
                        )
                        best_z = self._stage_z()
        if probes == 0:
            vdz = 24.0
            self._sync_move(
                monitor, best_z + vdz, _PARK_CREEP,
                len(monitor.times), labeled=False,
            )
            zr = self._stage_z()
            s1 = self._rest_static(monitor)
            probes = 1
            _log(
                "REFINE k=1 z=%.1f s=%.0f prev=%.0f"
                % (zr, s1, best_s)
            )
            if s1 > best_s * 1.08:
                best_z, best_s = zr, s1
                type(self)._REFINE_HIST.append(vdz)
                del type(self)._REFINE_HIST[:-5]
            else:
                self._sync_move(
                    monitor, best_z, _PARK_CREEP,
                    len(monitor.times), labeled=False,
                )
                best_z = self._stage_z()
        if abs(best_z - park_z) > 0.5:
            bank = sorted(type(self)._REFINE_HIST)
            if bank:
                med = bank[len(bank) // 2]
                type(self)._REFINE_DZ = med
                self._save_cal("refine_dz", med)
                _log("REFINE bank dz=%.1f n=%d" % (med, len(bank)))
            self._sync_move(
                monitor, best_z, _PARK_CREEP,
                len(monitor.times), labeled=False,
            )
            best_z = self._stage_z()
            best_s = self._rest_static(monitor)
        return best_z, best_s

    def _revert_if_worse(
            self, monitor: StreamSharpnessMonitor, z0: float,
            start_s: float, land_z: float,
    ):
        land_s = self._rest_static(monitor)
        if start_s < 30.0 or land_s >= 0.7 * start_s:
            _log(
                "LAND start=%.0f land=%.0f verdict=keep"
                % (start_s, land_s)
            )
            return None
        _log(
            "REGRESS start=%.0f land=%.0f verdict=revert"
            % (start_s, land_s)
        )
        self._sync_move(
            monitor, z0, _PARK_VELOCITY, len(monitor.times), labeled=False,
        )
        rev_s = self._rest_static(monitor)
        zr = self._stage_z()
        if rev_s > land_s:
            _log(
                "REVERTED z=%.2f static=%.0f (start=%.0f land=%.0f)"
                % (zr, rev_s, start_s, land_s)
            )
            return zr
        _log(
            "REVERT-REFUSED z0=%.0f <= land=%.0f, returning to landing"
            % (rev_s, land_s)
        )
        self._sync_move(
            monitor, land_z, _PARK_VELOCITY, len(monitor.times),
            labeled=False,
        )
        return self._stage_z()

    def _load_cal(self) -> None:
        try:
            p = "/home/admin/openflexure-microscope-server/"
            p += "smooth_af_cal.json"
            with open(p) as f:
                d = json.load(f)
            corr = d.get("label_corr")
            if isinstance(corr, dict) and corr.get("v") and corr.get("du"):
                try:
                    v = float(corr["v"])
                    du = float(corr["du"])
                    if (
                            200.0 <= v <= 20000.0
                            and abs(du) <= 2000.0
                            and abs(du / v) <= 0.5
                    ):
                        type(self)._CORR = {"v": v, "du": du}
                        _log(f"CORR loaded v={v:.0f} du={du:.1f}")
                except (TypeError, ValueError):
                    pass
            a = float(d.get("accel") or 0.0)
            tf = d.get("tau_far")
            if tf is not None:
                try:
                    tf = float(tf)
                    if 0.0 < tf < 0.5:
                        type(self)._TAU_FAR = tf
                except (TypeError, ValueError):
                    pass
            tt = d.get("tau_tiny")
            if tt is not None:
                try:
                    tt = float(tt)
                    if 0.0 < tt < 0.5:
                        type(self)._TAU_TINY = tt
                except (TypeError, ValueError):
                    pass
            rd = d.get("refine_dz")
            if rd is not None:
                try:
                    rd = float(rd)
                    if 0.0 < abs(rd) <= 100.0:
                        type(self)._REFINE_DZ = rd
                except (TypeError, ValueError):
                    pass
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

    def _save_cal(self, key: str, value) -> None:
        try:
            p = "/home/admin/openflexure-microscope-server/"
            p += "smooth_af_cal.json"
            try:
                with open(p) as f:
                    d = json.load(f)
            except Exception:
                d = {}
            d[key] = value
            with open(p, "w") as f:
                json.dump(d, f)
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
        if focus_static > 0 and focus_static > type(self)._SESS_STATIC:
            type(self)._SESS_STATIC = float(focus_static)
        try:
            sender = getattr(self._stage, "_send_gcode_script", None)
            pos = self._stage.position if sender is not None else None
            if (
                    sender is not None and pos is not None
                    and not getattr(self, "_far_regime_active", False)
            ):
                sender(script="SAMPLE_MESH_RESUME")
                st = self._stage._get_status(payload={
                    "objects": {"sample_mesh": None}
                })
                sm = (
                             (st.get("result") or {}).get("status") or {}
                     ).get("sample_mesh") or {}
                surface = sm.get("surface_here")
                zref = sm.get("zref")
                if surface is not None:
                    delta = float(pos.get("z")) - float(surface)
                    sender(script=f"SAMPLE_MESH_SHIFT DELTA={delta:.3f}")
                    self._mesh_suspended_by_af = False
                    _log(
                        f"MESHSHIFT delta={delta:.1f} "
                        f"surface_here={surface:.1f}"
                    )
                elif zref is not None:
                    delta = float(pos.get("z")) - float(zref)
                    sender(script=f"SAMPLE_MESH_SHIFT DELTA={delta:.3f}")
                    self._mesh_suspended_by_af = False
                    _log(f"MESHSHIFT delta={delta:.1f} (legacy zref)")
        except Exception:
            LOGGER.warning("mesh resume/shift after AF failed", exc_info=True)

    def _run_flow(self, monitor: StreamSharpnessMonitor) -> SmoothFocusResult:
        t0 = time.monotonic()
        mode = ""
        span = float(getattr(self, "_span_hint", 0.0) or _FAR_SPAN)
        half_span = min(span / 2.0, _FAR_SPAN)
        try:
            st = self._stage._get_status(payload={
                "objects": {"gcode_move": None}
            })
            self._last_hw = float(
                st["result"]["status"]["gcode_move"]["position"][2]
            )
        except Exception:
            LOGGER.warning(
                "AF logical-z resync failed, using cached", exc_info=True
            )
        self._frame_offset = 0.0
        z0 = self._last_hw

        self._mesh_suspended_by_af = False
        try:
            st = self._stage._get_status(payload={
                "objects": {"sample_mesh": None}
            })
            sm = (
                         (st.get("result") or {}).get("status") or {}
                 ).get("sample_mesh") or {}
            if sm.get("active"):
                sender = getattr(self._stage, "_send_gcode_script", None)
                if sender is not None:
                    sender(script="SAMPLE_MESH_SUSPEND")
                    self._mesh_suspended_by_af = True
                    _log("MESH SUSPEND for AF (transform deflection)")
        except Exception:
            LOGGER.warning(
                "mesh suspend for AF failed", exc_info=True
            )

        try:
            if _SCAN_MODE():
                _log("AFREGIME scan: held by wrapper from saved base")
            else:
                prev_gain_af = self._cam.analogue_gain
                prev_exp_af = self._cam.exposure_time
                if prev_exp_af:
                    _g, _e = _probe_regime(prev_gain_af, prev_exp_af)
                    _e = max(100, min(_fps_safe_exp(self._cam), _e))
                    self._cam.set_probe_controls_raw(_g, _e)
                    _log(f"AFREGIME g={_g} e={_e} (from {prev_gain_af}/{prev_exp_af})")
                else:
                    _log("AFREGIME skipped: no live exposure reading")
        except Exception:
            LOGGER.warning("AF regime set failed", exc_info=True)
            _log("AFREGIME SETFAIL")

        m_settle = len(monitor.times)
        monitor.wait_frames(16, m_settle, timeout=0.25)
        pre = monitor.frames_since(0)[-4:]
        static_med = (
            statistics.median([f[2] for f in pre]) if len(pre) >= 3 else 0.0
        )
        mode += f"|static{static_med:.0f}"

        if not type(self)._CAL_LOADED:
            self._load_cal()

        cur = self._stage_z()
        if not (cur == cur):
            cur = self._last_hw
        leg_a = cur + half_span
        leg_b = cur - half_span
        if not type(self)._CAL_LOADED:
            self._load_cal()
        if _SCAN_MODE():
            return self._run_flow_scan(monitor, z0, static_med)

        if type(self)._CAL_ACCEL is None:
            far_v = _CAL_V_ACCEL
            mode += "|accelcal"
        else:
            far_v = _FAR_VELOCITY
        _log(
            f"FARREGIME_V v={far_v:.0f} cal={type(self)._CAL_ACCEL}"
        )

        segs = {}
        self._telestall_run = False
        best_hit = None
        best_sharp = 0.0
        far_exp_prev = None
        far_gain_prev = None
        reg = type(self)._LAST_REGIME
        if reg is not None:
            far_exp_prev = reg[1]
            far_gain_prev = reg[0]
        if far_exp_prev is None:
            try:
                far_exp_prev = self._cam.exposure_time
            except Exception:
                far_exp_prev = None
        if far_gain_prev is None:
            try:
                far_gain_prev = self._cam.analogue_gain
            except Exception:
                far_gain_prev = None
        if far_exp_prev:
            far_gain, far_exp = _probe_regime(
                type(self)._LAST_REGIME[0] if type(self)._LAST_REGIME else 2.0,
                type(self)._LAST_REGIME[1] if type(self)._LAST_REGIME else far_exp_prev,
            )
            far_exp = max(100, min(_fps_safe_exp(self._cam), far_exp))
        else:
            raise ValueError(
                "no live exposure reading - refusing to invent far regime"
            )
        try:
            self._cam.set_probe_controls_raw(far_gain, far_exp)
        except Exception as e:
            _log(f"FARREGIME SETFAIL {e!r}")
        _log(f"FARREGIME exp={far_exp} gain={far_gain} "
             f"prev={far_exp_prev}")
        for tag, target, prev_end in (("far", leg_a, cur), ("far2", leg_b, leg_a)):
            if self._telestall_run:
                mode += f"|{tag}_abort"
                break
            far_mark = len(monitor.times)
            traj, frames = self._sync_move(
                monitor, target, far_v, far_mark
            )
            corr = type(self)._CORR or {}
            tau_c = 0.0
            if type(self)._TAU_FAR is not None:
                tau_c = type(self)._TAU_FAR * far_v
            elif corr.get("v"):
                tau_c = float(corr["du"]) / float(corr["v"]) * far_v
            if tau_c:
                mode += f"|{tag}corr{tau_c:.0f}"
            leg_pts = self._label_leg(
                frames,
                prev_end + tau_c if target > prev_end else prev_end - tau_c,
                target + tau_c if target > prev_end else target - tau_c,
                traj,
                far_v,
            )
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
                leg_best = max(q[2] for q in leg_pts)
                if best_hit is None or leg_best > best_sharp:
                    best_hit = hit
                    best_sharp = leg_best
            else:
                mode += "|nohit"
            if (tag == "far" and hit is not None and leg_pts):
                s_med1 = statistics.median(
                    [q[2] for q in leg_pts]) or 1e-9
                if (
                    max(q[2] for q in leg_pts) >= 3.0 * max(s_med1, 1e-9)
                    and max(q[3] for q in leg_pts)
                    - statistics.median([q[3] for q in leg_pts])
                    >= 15.0
                ):
                    mode += "|far1_skip"
                    break

        try:
            self._cam.set_probe_controls_raw(
                far_gain_prev or 1.0,
                max(100, min(_fps_safe_exp(self._cam),
                             far_exp if far_exp else 500)),
            )
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
            result = self._tiny_and_park(
                monitor, claw_z, mode, z0, static_med, t0
            )
            if result is not None:
                return result

        mode += "|no_anchor_refuse"
        self._span_hint = 0.0
        ret_mark = len(monitor.times)
        self._sync_move(monitor, z0, _PARK_VELOCITY, ret_mark, labeled=False)
        mode += f"|return{self._stage_z():.1f}"
        raise NoFocusFoundError(mode)

    def _tiny_and_park(
            self,
            monitor: StreamSharpnessMonitor,
            anchor: float,
            mode: str,
            z0: float,
            static_med: float,
            t0: float,
            span: float | None = None,
    ) -> Optional[SmoothFocusResult]:
        """Generic tiny sweep around `anchor`: span ± around it at
        fine_sweep_velocity, gaussian-gated fit, two-stage park, settled
        verify, soft revert, mesh shift. Shared by the button flow (anchor
        = claw hit, span = fwhm-derived) and the scan-only flow (anchor =
        last focused z, span = caller-set)."""
        fwhm = (
            type(self)._CAL_FWHM if type(self)._CAL_CONTRAST else None
        )
        if span:
            span_h = float(span)
        else:
            span_h = max(
                95.0, min(_TINY_SPAN, (fwhm * 1.25) if fwhm else 95.0)
            )
        tiny_lo = anchor - span_h
        tiny_hi = anchor + span_h
        tiny_v = self.fine_sweep_velocity(fwhm)
        self._sync_move_pair(
            monitor,
            tiny_hi + 60.0, _PARK_VELOCITY, tiny_hi, _PARK_CREEP,
        )
        tiny_mark2 = len(monitor.times)
        traj3, frames3 = self._sync_move(
            monitor, tiny_lo, tiny_v, tiny_mark2
        )
        corr = type(self)._CORR or {}
        tau_t = 0.0
        if type(self)._TAU_TINY is not None:
            tau_t = type(self)._TAU_TINY * tiny_v
        elif corr.get("v"):
            tau_t = float(corr["du"]) / float(corr["v"]) * tiny_v
        if tau_t:
            mode += f"|tinycorr{tau_t:.0f}"
        tiny_pts = self._label_leg(
            frames3,
            tiny_lo + tau_t,
            tiny_hi + tau_t,
            traj3,
            tiny_v,
        )
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
                    and abs(t_arg[0] - anchor) <= span_h
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
            park_z = self._park(
                monitor,
                fit.peak + (type(self)._REFINE_DZ or 0.0),
                len(monitor.times),
            )
            mode += f"|park{park_z:.1f}"
            park_z, _rs = self._refine_landing(monitor, park_z)
            mode += f"|refine{park_z:.1f}"
            rev = self._revert_if_worse(
                monitor, z0, static_med, park_z
            )
            if rev is not None:
                park_z = rev
                mode += "|reverted"
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
        return None

    def _run_flow_scan(
            self,
            monitor: StreamSharpnessMonitor,
            z0: float,
            static_med: float,
    ) -> SmoothFocusResult:
        """Scan-only flow: anchor-stanced tiny sweep -> park -> settled
        verify. Anchor = last focused z; on 2 consecutive gated misses it
        falls back to one full-chain run to re-seed the anchor."""
        mode = "scan"
        t0 = time.monotonic()
        span = float(
            getattr(_SCAN_LOCAL, "span", 0.0) or _SCAN_SPAN
        )
        if type(self)._SCAN_LAST_Z is None:
            type(self)._SCAN_LAST_Z = self._stage_z()
        anchor = float(type(self)._SCAN_LAST_Z)
        result = self._tiny_and_park(
            monitor, anchor, mode, z0, static_med,
            time.monotonic(), span,
        )
        if result is not None:
            type(self)._SCAN_LAST_Z = float(result.peak_z)
            type(self)._SCAN_FAILS = 0
            return result
        type(self)._SCAN_FAILS += 1
        mode += f"|miss{type(self)._SCAN_FAILS}"
        if type(self)._SCAN_FAILS < 2:
            mode += "|held"
            self._sync_move(
                monitor, anchor, _PARK_VELOCITY,
                len(monitor.times), labeled=False,
            )
            _log(f"SCANFLOW {mode}")
            return SmoothFocusResult(
                peak_z=self._stage_z(),
                mode=mode,
                fit=FitOutcome(ok=False, peak=anchor, fwhm=None),
                curve=[],
                elapsed_s=time.monotonic() - t0,
            )
        mode += "|reseed_full"
        type(self)._SCAN_FAILS = 0
        type(self)._SCAN_LAST_Z = None
        self._span_hint = 0.0
        ret_mark = len(monitor.times)
        self._sync_move(monitor, z0, _PARK_VELOCITY, ret_mark, labeled=False)
        try:
            self._cam.restore_calibration_raw()
            _log("SCANFLOW reseed: base regime restored for full chain")
        except Exception:
            LOGGER.warning("reseed base restore failed", exc_info=True)
        _log(f"SCANFLOW {mode}")
        try:
            return self._run_flow_full_once(monitor)
        except NoFocusFoundError as exc:
            _log("SCANFLOW reseed whiff - retrying full chain once")
            try:
                return self._run_flow_full_once(monitor)
            except NoFocusFoundError as exc2:
                mode += f"|reseed_fail({exc2})"
                _log(f"SCANFLOW {mode}")
                raise NoFocusFoundError(mode) from exc2

    def _run_flow_full_once(
            self, monitor: StreamSharpnessMonitor
    ) -> SmoothFocusResult:
        """One full-chain run with scan mode suppressed."""
        scan = getattr(_SCAN_LOCAL, "mode", False)
        _SCAN_LOCAL.mode = False
        try:
            return self._run_flow(monitor)
        finally:
            _SCAN_LOCAL.mode = scan

    @lt.action
    def scan_autofocus(self, span: float = 0.0) -> SmoothFocusResult:
        """Scan-only focus: anchor tiny sweep -> park -> settled verify.
        span sets the sweep half-span in stage units (0 = default)."""
        _SCAN_LOCAL.span = float(span)
        _SCAN_LOCAL.mode = True
        try:
            return self.smooth_focus(self._scan_params())
        except NoFocusFoundError as exc:
            if "reseed_fail" in str(exc):
                raise
            raise NoFocusFoundError(
                str(exc).replace("scan|", "", 1)
            ) from exc
        finally:
            _SCAN_LOCAL.mode = False
            _SCAN_LOCAL.span = 0.0

    @lt.action
    def scan_autofocus_full(self) -> SmoothFocusResult:
        """Scan-only focus, forced full chain (no anchor tiny sweep)."""
        _SCAN_LOCAL.span = 0.0
        _SCAN_LOCAL.mode = True
        try:
            return self.smooth_focus(self._scan_params())
        finally:
            _SCAN_LOCAL.mode = False

    def _scan_params(self) -> SmoothAutofocusParams:
        return SmoothAutofocusParams(search_span_steps=0)

    @classmethod
    def fine_sweep_velocity(cls, fwhm: float | None) -> float:
        """Shared fine-sweep speed law: FPS x FWHM / 5, clamped.

        Single source of truth for the engine's tiny sweep and the
        scanner's per-point fine sweeps."""
        f_ref = fwhm if fwhm else 20.0
        return max(_TINY_MIN_V, min(_TINY_VELOCITY, FPS * f_ref / 5.0))

    @lt.action
    def smooth_focus(
            self, params: SmoothAutofocusParams
    ) -> SmoothFocusResult:
        """Sweep autofocus: far sweep, claw, tiny sweep, park.

        Raises NoFocusFoundError (and returns the stage to its start) rather
        than parking on an unverified peak.
        """
        t0 = time.monotonic()
        burst = 140.0
        hint = float(params.search_span_steps)
        self._span_hint = (
            max(200.0, min(hint, 1024.0)) if hint > 0 else 0.0
        )
        prev_exp = prev_gain = None
        forced_base = getattr(self, "_forced_restore_base", None)
        try:
            if forced_base is not None:
                prev_gain, prev_exp = forced_base
                type(self)._LAST_REGIME = None
            elif _SCAN_MODE():
                if not type(self)._CAL_LOADED:
                    self._load_cal()
                prev_gain = float(self._cam._analogue_gain)
                prev_exp = int(self._cam._exposure_time)
                type(self)._LAST_REGIME = (prev_gain, prev_exp)
            else:
                prev_exp = self._cam.exposure_time
                prev_gain = self._cam.analogue_gain
                if prev_gain is not None and prev_exp is not None:
                    if float(prev_gain) < 3.5:
                        type(self)._LAST_REGIME = (
                            float(prev_gain), int(prev_exp)
                        )
                    else:
                        type(self)._LAST_REGIME = None
        except Exception:
            prev_exp = prev_gain = None
        type(self)._AF_BUSY = True
        if _SCAN_MODE():
            type(self)._LAST_SCAN_TS = time.monotonic()
        self._cam._raw_controls_held = True
        _cool_ev = type(self)._COOL_DONE
        if _cool_ev is not None and type(self)._COOLING:
            _cool_ev.wait(timeout=5.0)
        self._cancel_cooler()
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
            _fast = True
            try:
                if self._cam.streaming_mode != "crop990":
                    self._cam._start_streaming("crop990")
                    _fast = False
            except Exception:
                LOGGER.warning("Could not switch to crop990", exc_info=True)
            try:
                Path(
                    "/home/admin/openflexure-microscope-server/af_ts_probe.log"
                ).unlink(missing_ok=True)
            except OSError:
                pass
            _probe_mod = sys.modules.get(
                "openflexure_microscope_server.things.camera.picamera"
            )
            if _probe_mod is not None and hasattr(
                    _probe_mod, "_AF_TS_PROBE"
            ):
                _probe_mod._AF_TS_PROBE["n"] = 0
                _probe_mod._AF_TS_PROBE["t"] = time.time()
            with monitor:
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
                with _EncoderBurst(self._cam):
                    _log(
                        f"SETUP burst_ready t={time.monotonic() - _t_burst:.3f}"
                    )
                    try:
                        _g, _e = _probe_regime(prev_gain, prev_exp)
                        _e = max(100, min(_fps_safe_exp(self._cam), _e))
                        self._cam.set_probe_controls_raw(_g, _e)
                        try:
                            result = self._run_flow(monitor)
                        except NoFocusFoundError as exc:
                            if "scan|" in str(exc) and "reseed_full" in str(exc):
                                _log(
                                    f"RUN total={time.monotonic() - t0:.2f} "
                                    f"mode={exc} NORETRY"
                                )
                                raise
                            if not (
                                "tiny_fail" in str(exc)
                                or (
                                    "no_anchor_refuse" in str(exc)
                                    and not _SCAN_MODE()
                                )
                            ):
                                _log(
                                    f"RUN total={time.monotonic() - t0:.2f} "
                                    f"mode={exc} NORETRY"
                                )
                                raise
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
                self._cam.restore_calibration_raw()
            except Exception:
                LOGGER.warning("Exposure/gain restore failed", exc_info=True)
            finally:
                self._cam._raw_controls_held = False
            try:
                if hasattr(self._stage, "update_position"):
                    self._stage.update_position()
            except Exception:
                LOGGER.warning("Stage position sync failed", exc_info=True)
            try:
                if getattr(self, "_mesh_suspended_by_af", False):
                    sender = getattr(self._stage, "_send_gcode_script", None)
                    if sender is not None:
                        sender(script="SAMPLE_MESH_RESUME")
                    self._mesh_suspended_by_af = False
                    _log("MESH RESUME after AF abort")
            except Exception:
                LOGGER.warning(
                    "mesh resume after AF failed", exc_info=True
                )
            type(self)._AF_BUSY = False
            self._arm_cooler()


class SmoothCompatAutofocus(SmoothAutofocusThing, AutofocusThing):
    """Drop-in AutofocusThing backed by the smooth autofocus engine.

    Keeps the smooth module's own units and gates: dz is interpreted as the
    total search span in stage units (not the stock stage's step semantics),
    and NoFocusFoundError from the engine surfaces as the scanner's
    ScanNoFocusFoundError.
    """

    _af_params = SmoothAutofocusParams()

    def _run_smooth(self, span: int) -> SharpnessDataArrays:
        params = self._af_params.model_copy(update={"search_span_steps": 0})
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
    def scan_autofocus(self, span: float = 0.0) -> SmoothFocusResult:
        """Scan-only focus: anchor tiny sweep -> park -> settled verify.
        Falls back to the full chain after 2 consecutive gated misses."""
        try:
            return super().scan_autofocus(span=span)
        except NoFocusFoundError as exc:
            raise ScanNoFocusFoundError(str(exc)) from exc

    @lt.action
    def scan_autofocus_full(self) -> SmoothFocusResult:
        """Scan-only focus, forced full chain (no anchor tiny sweep)."""
        try:
            return super().scan_autofocus_full()
        except NoFocusFoundError as exc:
            raise ScanNoFocusFoundError(str(exc)) from exc

    @lt.action
    def fast_autofocus(
            self,
            dz: int = 2000,
            start: Literal["centre", "base"] = "centre",
            sharpness_metric=None,
            record=None,
    ) -> SharpnessDataArrays:
        """Autofocus with the smooth engine; dz is accepted for API
        compatibility only - the engine always runs its full-span chain."""
        try:
            return self._run_smooth(0)
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
            self._run_smooth(0)
        except NoFocusFoundError as exc:
            raise ScanNoFocusFoundError(str(exc)) from exc
        peaks = [float(self._stage.position["z"])]
        return peaks, []
