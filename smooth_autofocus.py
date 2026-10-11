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
_FAR_VELOCITY = 6000.0
_FAR_MEDIAN_BAR = 1.25
_NEAR_SPAN = 110.0
_NEAR_VELOCITY = 250.0
_TINY_SPAN = 180.0
_TINY_VELOCITY = 700.0
_TINY_MIN_V = 100.0
_PARK_VELOCITY = 6000.0
_MEDIAN_BAR = 1.6
_EDGE_MARGIN = 0.12
FPS = 120.0


class NoFocusFoundError(RuntimeError):
    """No confident focus was found."""


class SmoothAutofocusParams(BaseModel):
    search_span_steps: int = Field(default=32, gt=0)
    min_r2: float = Field(default=0.5, ge=0, le=1)
    min_snr: float = Field(default=1.4, ge=1)


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


@dataclass
class GateCalibration:
    fwhm_u: float
    contrast: float
    peak_sharpness: float
    floor_sharpness: float
    peak_z: float
    sweep_error_u: float
    backlash_u: float
    near_velocity: float
    tiny_velocity: float
    tiny_span: float
    mode: str


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
                _frame_stash_start()
                _log("BURST stash_start t=%.3f" % (time.monotonic(),))
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
    _cached_accel: Optional[float] = None
    _leg_certified: bool = False
    _FAR_DIR: int = 1
    _CAL_FWHM: Optional[float] = None
    _CAL_CONTRAST: Optional[float] = None
    _CAL_BACKLASH: Optional[float] = None
    _CAL_PEAK_SHARP: Optional[float] = None

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
        self._leg_certified = False
        try:
            return self._sync_move_inner(
                monitor, target, velocity, mark, labeled
            )
        finally:
            if not self._leg_certified:
                if not labeled:
                    self._last_hw = target
                else:
                    try:
                        if hasattr(self._stage, "update_position"):
                            self._stage.update_position()
                    except Exception:
                        pass

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
        drained = getattr(self._stage, "motion_drained", None)
        t_post = time.monotonic()
        self._stage.move_relative_z_timed(
            self._hw_steps_param(int(round(delta))), seconds
        )
        self._last_hw = target
        if motion_since is None or drained is None:
            return None, []
        motion_wait = getattr(self._stage, "motion_wait", None)
        samples = []
        moving_seen = False
        t_open = t_post + 0.5 * seconds
        # Motion end is known from geometry: commanded time + accel corners.
        # Labeled legs give the v=0 push one status tick (~0.27s) to land
        # for end-position certification; unlabeled legs (approach, park,
        # return) need no push and break right at motion end.
        accel = type(self)._cached_accel
        if accel is None:
            _ag = getattr(self._stage, "printer_max_accel", None)
            if callable(_ag):
                try:
                    _a = _ag()
                    if _a and 0.0 < _a < 1e7:
                        type(self)._cached_accel = float(_a)
                        accel = float(_a)
                except Exception:
                    accel = None
        corner = 0.0
        if accel and seconds > 0.0:
            t_acc = velocity / accel
            if velocity * t_acc < abs(delta):
                corner = 2.0 * t_acc
        slack = 0.27 if labeled else 0.03
        expected_end = t_post + seconds + corner
        deadline = min(
            t_post + seconds + 2.0,
            expected_end + (0.9 if labeled else slack + 0.6),
        )
        while time.monotonic() < deadline:
            new = motion_since(t_open)
            if len(new) == len(samples):
                if moving_seen and time.monotonic() > expected_end + slack:
                    break
                if motion_wait is None:
                    continue
                motion_wait(
                    timeout=min(
                        0.5,
                        max(
                            0.01,
                            min(
                                deadline - time.monotonic(),
                                expected_end + slack - time.monotonic(),
                            ),
                        ),
                    )
                )
                new = motion_since(t_open)
                if not new:
                    continue
            samples = new
            if abs(samples[-1][2]) > 0.5:
                moving_seen = True
            elif moving_seen:
                break
            if not labeled and time.monotonic() > expected_end + slack:
                break
        _log(
            f"LEG EXIT labeled={labeled} n_samples={len(samples)} "
            f"moving_seen={moving_seen} "
            f"last_v={samples[-1][2] if samples else None} "
            f"waited={time.monotonic() - t_post:.3f} cmd={seconds:.3f} "
            f"t={time.time():.3f}"
        )
        falling = None
        mids = []
        if moving_seen and samples and abs(samples[-1][2]) < 0.5:
            falling = samples[-1]
        elif not moving_seen and len(samples) == 1 and abs(
            samples[-1][1] - cur
        ) > 0.5 * abs(delta):
            falling = samples[-1]
        elif moving_seen and samples and abs(samples[-1][2]) >= 0.5:
            _log(
                f"LEG GEOMFALL cmd={seconds:.3f} "
                f"last_v={samples[-1][2]:.1f} t={time.time():.3f}"
            )
            falling = (expected_end, cur + delta, 0.0, 0.0, None)
        if falling is None:
            return None, []
        t_fall, z_end = falling[0], falling[1]
        if abs(z_end - (cur + delta)) > max(2.0, 0.05 * abs(delta)):
            _log(
                f"LEG MISMATCH end={z_end:.2f} want={cur + delta:.2f}"
                f" delta={delta:.2f} t={time.time():.3f}"
            )
            return None, []
        margin = 1.0 / FPS
        clock_off = getattr(self._stage, "clock_offset", None)
        clock_spread = getattr(self._stage, "clock_spread", None)
        off = clock_off() if callable(clock_off) else None
        spr = clock_spread() if callable(clock_spread) else None
        if off is not None and (spr is None or spr > 0.05):
            _log(
                f"LEG OFFUNSTABLE off={off:.4f} spr={spr if spr is not None else -1:.4f}"
                f" t={time.time():.3f}"
            )
            off = None

        def _s5(s):
            if len(s) > 4:
                return s
            return (s[0], s[1], s[2], s[3], None)

        slope_cmd = delta / seconds
        _ho = getattr(self._stage, "handshake_offset", None)
        hoff = _ho() if callable(_ho) else None
        if hoff is not None:
            remapped = []
            for s in samples:
                if len(s) > 4 and s[4] is not None:
                    remapped.append((s[4] - hoff,) + tuple(s[1:]))
                else:
                    remapped.append(s)
            samples = remapped
        mids = [
            s for s in (_s5(q) for q in samples[:-1])
            if abs(s[2]) > 0.5
            and not s[3]
            and abs(s[1] - cur) > min(2.0, 0.02 * abs(delta))
            and abs(abs(s[2]) - abs(slope_cmd)) <= 0.05 * abs(slope_cmd)
        ]
        if not mids and samples:
            _sps = [
                _s5(s) for s in samples
                if abs(s[2]) > 0.5
                and not _s5(s)[3]
                and abs(_s5(s)[1] - cur) > min(2.0, 0.02 * abs(delta))
            ]
            if _sps:
                mids = [_sps[-1]]
        accel = type(self)._cached_accel

        def _build(ts_, te_, v_abs):
            total = z_end - cur
            sgn = 1.0 if total >= 0 else -1.0
            if accel:
                t_acc = v_abs / accel
                d_acc = 0.5 * v_abs * t_acc
                if 2.0 * d_acc >= abs(total):
                    t_acc = (abs(total) / accel) ** 0.5
                    pts = [
                        (ts_, cur),
                        (ts_ + t_acc, cur + sgn * 0.5 * accel * t_acc * t_acc),
                        (te_, z_end),
                    ]
                else:
                    z_mc = z_end - sgn * d_acc
                    pts = [
                        (ts_, cur),
                        (ts_ + t_acc, cur + sgn * d_acc),
                        (
                            ts_ + t_acc + (abs(total) - 2.0 * d_acc) / v_abs,
                            z_mc,
                        ),
                        (te_, z_end),
                    ]
                ta_ = t_acc
            else:
                pts = [(ts_, cur), (te_, z_end)]
                ta_ = 0.0
            frames = [
                f for f in monitor.frames_since(mark)
                if ts_ + margin <= f[0] <= te_ - margin
            ]
            return pts, frames

        if mids:
            ta, za = mids[-1][0], mids[-1][1]
            if accel:
                t_acc = min(velocity / accel, (abs(delta) / accel) ** 0.5)
                d_acc = 0.5 * accel * t_acc * t_acc
            else:
                t_acc, d_acc = 0.0, 0.0
            sgn = 1.0 if delta >= 0 else -1.0
            z_rel = min(max((za - cur) * sgn, 0.0), abs(delta))
            t_acc2 = t_acc * t_acc
            if z_rel <= d_acc:
                tau = (2.0 * z_rel / accel) ** 0.5 if accel else 0.0
            elif z_rel >= abs(delta) - d_acc:
                z_left = abs(delta) - z_rel
                tau = (
                    t_acc + (abs(delta) - 2.0 * d_acc) / velocity
                    + t_acc - (2.0 * z_left / accel) ** 0.5
                ) if accel else seconds
                tau = min(tau, seconds + 2.0 * t_acc)
            else:
                tau = t_acc + (z_rel - d_acc) / velocity
            T_total = seconds + (2.0 * t_acc if accel else 0.0)
            t_start = min(max(ta - tau, ta - T_total), ta)
            t_fall = t_start + T_total
            _log(
                f"LEG TRAPANCHOR tau={tau:.3f} T={T_total:.3f} "
                f"t={time.time():.3f}"
            )
            return _build(t_start, t_fall, velocity)
        elif off is not None:
            _log(
                f"LEG off={off:.4f} spr={spr:.4f} mids=0 t={time.time():.3f}"
            )
        t_start = None
        fall_et = falling[4] if len(falling) > 4 else None
        if fall_et is not None and hoff is not None:
            t_fall = fall_et - hoff
            t_acc_extra = 2.0 * (abs(delta) / 2.0) / velocity if False else 0.0
            if accel:
                t_acc = velocity / accel
                d_acc = 0.5 * velocity * t_acc
                if 2.0 * d_acc < abs(delta):
                    t_acc_extra = 2.0 * t_acc
            t_start = t_fall - seconds - t_acc_extra
            _log(
                f"LEG ETANCHOR dt={falling[0] - t_fall:.3f} "
                f"acc={t_acc_extra:.3f} t={time.time():.3f}"
            )
        if t_start is None:
            t_start = t_fall - seconds
        traj = [(t_start, cur), (t_fall, z_end)]
        frames = [
            f for f in monitor.frames_since(mark)
            if t_start + margin <= f[0] <= t_fall - margin
        ]
        try:
            _dump_json("smooth_af_legdump.json", {
                "leg": [cur, target, velocity, seconds],
                "mids": [[s[0], s[1], s[2], s[3], s[4]] for s in mids],
                "samples": [
                    [s[0], s[1], s[2], s[3],
                     s[4] if len(s) > 4 else None] for s in samples
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
        if len(pp) >= 6:
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

        anchor = type(self)._SESS_Z
        cur = self._stage_z()
        if not (cur == cur):
            cur = self._last_hw
        if anchor is not None and abs(anchor - cur) > 8.0:
            dirn = 1 if anchor > cur else -1
        else:
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
        if claw_z is not None:
            claw_z = self._refine_hit(leg1_pts, claw_z)
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
            if claw_z is not None:
                claw_z = self._refine_hit(leg2_pts, claw_z)
            mode += claw_tag2

        if claw_z is not None:
            fwhm = (
                type(self)._CAL_FWHM if type(self)._CAL_CONTRAST else None
            )
            span_h = max(15.0, min(_TINY_SPAN, (fwhm * 1.25) if fwhm else 50.0))
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
                    and abs(t_arg[0] - claw_z) <= 45.0
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
                    fit, best = gate
            if gate is not None:
                fit, best = gate
                mode += f"|tiny_fit{fit.peak:.1f}"
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
            if "far2" not in mode:
                mode += "|retryfar2"
                far_mark2 = len(monitor.times)
                traj2, frames2 = self._sync_move(
                    monitor, leg_b, _FAR_VELOCITY, far_mark2
                )
                leg2_pts = self._label_leg(frames2, claw_z, leg_b, traj2,
                                           _FAR_VELOCITY)
                _dump_json("smooth_af_seekpts_far2.json", {
                    "z": [q[0] for q in leg2_pts],
                    "s": [q[2] for q in leg2_pts],
                    "c": [q[1] for q in leg2_pts],
                    "p": [q[3] for q in leg2_pts],
                })
                claw_z2, claw_tag2 = self._claw(leg2_pts, "far2")
                if claw_z2 is not None:
                    claw_z2 = self._refine_hit(leg2_pts, claw_z2)
                mode += claw_tag2
                if claw_z2 is not None:
                    span_h2 = span_h
                    t_lo2 = max(claw_z2 - span_h2, min(leg_a, leg_b))
                    t_hi2 = min(claw_z2 + span_h2, max(leg_a, leg_b))
                    v2 = tiny_v
                    tm2 = len(monitor.times)
                    self._sync_move(
                        monitor, t_lo2, _PARK_VELOCITY, tm2, labeled=False
                    )
                    tm2b = len(monitor.times)
                    traj4, frames4 = self._sync_move(
                        monitor, t_hi2, v2, tm2b
                    )
                    t_pts2 = self._label_leg(frames4, t_lo2, t_hi2, traj4, v2)
                    _dump_json("smooth_af_seekpts_tiny.json", {
                        "z": [q[0] for q in t_pts2],
                        "s": [q[2] for q in t_pts2],
                        "c": [q[1] for q in t_pts2],
                        "p": [q[3] for q in t_pts2],
                    })
                    gate2 = self._fit_gate(t_pts2)
                    if gate2 is None and t_pts2:
                        fl2 = sorted(q[2] for q in t_pts2)[
                            max(0, len(t_pts2) * 3 // 10 - 1)
                        ]
                        a2 = max(t_pts2, key=lambda q: q[2])
                        if (
                            a2[2] >= 1.25 * max(fl2, 1e-9)
                            and abs(a2[0] - claw_z2) <= 45.0
                        ):
                            tp2 = sorted(
                                t_pts2, key=lambda q: q[2], reverse=True
                            )[: max(2, len(t_pts2) // 5)]
                            ws2 = sum(q[2] for q in tp2)
                            zf2 = (
                                sum(q[0] * q[2] for q in tp2) / ws2
                                if ws2 > 0 else a2[0]
                            )
                            mode += "|tiny_near"
                            gate2 = (
                                FitOutcome(ok=True, peak=zf2, fwhm=None),
                                a2[2],
                            )
                    if gate2 is not None:
                        fit, best = gate2
                        mode += f"|tiny_fit{fit.peak:.1f}"
                        park_z = self._park(
                            monitor, fit.peak, len(monitor.times)
                        )
                        mode += f"|park{park_z:.1f}"
                        self._save_last_focus(
                            monitor, len(monitor.times), fit.peak
                        )
                        return SmoothFocusResult(
                            peak_z=park_z, mode=mode, fit=fit,
                            curve=[
                                CurvePoint(z=q[0], sharpness=q[2])
                                for q in t_pts2
                            ],
                            elapsed_s=time.monotonic() - t0,
                        )
                    mode += "|tiny_fail2"

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

    @lt.action
    def metric_probe(
        self,
        n_frames: int = 100,
        burst_fps: float = 0.0,
    ) -> MetricProbeResult:
        """Diagnostic: median raw-Y sharpness at the current position.

        Collects exactly ``n_frames`` frames without moving the stage.
        """
        burst = float(burst_fps or FPS)
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
            _log("PROBE mode_switched t=%.3f" % (time.monotonic(),))
            with _EncoderBurst(self._cam, burst), monitor:
                try:
                    _g, _e = _probe_regime(prev_gain, prev_exp)
                    self._cam.analogue_gain = _g
                    self._cam.exposure_time = _e
                    monitor.wait_frames(
                        max(8, int(n_frames)), 0,
                        timeout=max(2.0, 4.0 * max(8, int(n_frames)) / FPS),
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
    def calibrate_gates(
        self, sweep_velocity: float = 80.0, sweep_span: float = 40.0
    ) -> GateCalibration:
        """Measure focus-curve width and contrast on the current view.

        Focus the sample by eye, zero the stage there, then run this: it
        sweeps +-40u through focus at 80 u/s inside a 120fps burst, folds
        the frames into stage coordinates, and stores FWHM/contrast as
        gate settings for near/tiny velocity and claw bars.
        """
        t0 = time.monotonic()
        monitor = StreamSharpnessMonitor(self._cam)
        try:
            prev_exp = self._cam.exposure_time
            prev_gain = self._cam.analogue_gain
        except Exception:
            prev_exp = prev_gain = None
        mode = ""
        pts = []
        try:
            try:
                if hasattr(self._stage, "set_homed"):
                    self._stage.set_homed()
                z0 = self._stage_z()
            except Exception:
                z0 = self._last_hw if hasattr(self, "_last_hw") else 0.0
            self._last_hw = z0
            d_near = 1 if type(self)._FAR_DIR == 1 else -1
            with self._fast_stream(
                restore=True, mode_name="crop990"
            ) as _fast, monitor:
                with _EncoderBurst(self._cam, 120.0):
                    try:
                        _g, _e = _probe_regime(prev_gain, prev_exp)
                        self._cam.analogue_gain = _g
                        self._cam.exposure_time = _e
                        m1 = len(monitor.times)
                        self._sync_move(
                            monitor, z0 + d_near * sweep_span, sweep_velocity, m1
                        )
                        m2 = len(monitor.times)
                        traj_b, frames_b = self._sync_move(
                            monitor, z0 - d_near * sweep_span, sweep_velocity, m2
                        )
                        pts_b = self._label_leg(
                            frames_b,
                            z0 + d_near * sweep_span, z0 - d_near * sweep_span,
                            traj_b, sweep_velocity,
                        )
                        m3 = len(monitor.times)
                        traj_c, frames_c = self._sync_move(
                            monitor, z0, sweep_velocity, m3
                        )
                        pts_c = self._label_leg(
                            frames_c,
                            z0 - d_near * sweep_span, z0,
                            traj_c, sweep_velocity,
                        )
                        mode = f"|cal nb={len(pts_b)} nc={len(pts_c)}"
                    except NoFocusFoundError:
                        raise
                    except Exception:
                        try:
                            mrec = len(monitor.times)
                            self._sync_move(monitor, z0, 300.0, mrec)
                            _log(
                                f"RECOVER to {z0:.1f} t={time.time():.3f}"
                            )
                        except Exception:
                            _log(
                                f"RECOVER FAILED t={time.time():.3f}"
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
        if len(pts_b) < 8 or len(pts_c) < 8:
            raise NoFocusFoundError("cal: too few frames" + mode)

        def _leg_centroid(points):
            zs = [q[0] for q in points]
            lo, hi = min(zs), max(zs)
            span = hi - lo
            inner = [
                q for q in points
                if lo + 0.1 * span <= q[0] <= hi - 0.1 * span
            ]
            if len(inner) < 4:
                return None
            ss = sorted((q[2] for q in inner), reverse=True)
            bar = ss[max(0, len(ss) // 4)]
            top = [q for q in inner if q[2] >= bar]
            wsum = sum(q[2] for q in top)
            if wsum <= 0:
                return None
            return sum(q[0] * q[2] for q in top) / wsum

        cb = _leg_centroid(pts_b)
        cc = _leg_centroid(pts_c)
        if cb is None or cc is None:
            raise NoFocusFoundError("cal: no inner frames" + mode)
        sweep_raw = float(cb) - float(cc)
        b = abs(sweep_raw) / 2.0
        d_near = 1 if type(self)._FAR_DIR == 1 else -1
        if b > 0.25:
            pts_b = (
                [(q[0] - b,) + tuple(q[1:]) for q in pts_b]
                if d_near == 1
                else [(q[0],) + tuple(q[1:]) for q in pts_b]
            )
            pts_c = (
                [(q[0],) + tuple(q[1:]) for q in pts_c]
                if d_near == 1
                else [(q[0] + b,) + tuple(q[1:]) for q in pts_c]
            )
        cb = _leg_centroid(pts_b)
        cc = _leg_centroid(pts_c)
        zb, zc = float(cb), float(cc)
        sweep_err = zb - zc
        zpk = (zb + zc) / 2.0
        merged = (
            [(q[0] - sweep_err / 2.0, q[2]) for q in pts_b]
            + [(q[0] + sweep_err / 2.0, q[2]) for q in pts_c]
        )
        merged.sort()
        az = np.array([q[0] for q in merged])
        as_ = np.array([q[1] for q in merged])
        pk = float(as_.max())
        far_mask = np.abs(az - zpk) > 30.0
        floor = (
            float(np.median(as_[far_mask]))
            if int(far_mask.sum()) >= 6
            else float(np.median(as_))
        )
        contrast = pk / max(floor, 1e-9)
        half = floor + (pk - floor) / 2.0
        above = az[as_ >= half]
        fwhm = float(above.max() - above.min()) if len(above) >= 2 else 0.0
        type(self)._CAL_FWHM = fwhm if fwhm > 0 else None
        type(self)._CAL_CONTRAST = contrast if contrast > 1.05 else None
        type(self)._CAL_BACKLASH = b if contrast > 1.05 else None
        type(self)._CAL_PEAK_SHARP = pk if contrast > 1.05 else None
        try:
            save = getattr(self._stage, "_save_cal", None)
            if callable(save):
                save({
                    "fwhm_u": type(self)._CAL_FWHM,
                    "contrast": type(self)._CAL_CONTRAST,
                    "backlash_u": type(self)._CAL_BACKLASH,
                    "peak_sharpness": type(self)._CAL_PEAK_SHARP,
                })
        except Exception:
            pass
        near_v = self._cal_near_velocity()
        span_h = max(15.0, min(_TINY_SPAN, (fwhm * 1.25) if fwhm else _TINY_SPAN))
        tiny_v = max(_TINY_MIN_V, min(_TINY_VELOCITY, FPS * fwhm / 5.0))
        mode += f"|fwhm{fwhm:.1f}|c{contrast:.2f}|err{sweep_err:+.1f}|b{b:.1f}"
        _log(
            f"CAL fwhm={fwhm:.1f} contrast={contrast:.2f} zpk={zpk:.1f} "
            f"err={sweep_err:+.1f} b={b:.1f} near_v={near_v:.0f} "
            f"tiny_span={2 * span_h:.0f} tiny_v={tiny_v:.0f}"
        )
        return GateCalibration(
            fwhm_u=round(fwhm, 1),
            contrast=round(contrast, 3),
            peak_sharpness=round(pk, 1),
            floor_sharpness=round(floor, 1),
            peak_z=round(zpk, 1),
            sweep_error_u=round(sweep_err, 1),
            backlash_u=round(b, 1),
            near_velocity=round(near_v, 0),
            tiny_velocity=round(tiny_v, 0),
            tiny_span=round(2 * span_h, 0),
            mode=mode,
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
