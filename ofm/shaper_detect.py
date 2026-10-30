
import logging
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, List, Literal, Optional

import httpx
import labthings_fastapi as lt
import numpy as np

from openflexure_microscope_server.things import OFMThing
from .camera import BaseCamera

try:
    from .strobe_control import StrobeControlThing
except Exception:
    StrobeControlThing = None

LOGGER = logging.getLogger(__name__)

DATA_FOLDER = Path("/home/admin/ofm/openflexure/data/shaper_detect")
TRACK_SIZE = 64
TEXTURE_MIN_VAR = 30.0
TRACK_HALF = TRACK_SIZE // 2
FRAME_W = 1332
FRAME_H = 990
EXCITATION_VELOCITY = 90.0
EXCITATION_VELOCITY_FACTOR = 1.0
EXCITATION_ACCEL_PER_HZ = 75.0
OCTAVES_PER_PAIR = 4.0
DEFAULT_MIN_FREQ = 20.0
DEFAULT_MAX_FREQ = 65.0
DEFAULT_SECONDS = 16.0
MIN_CORRELATION = 0.5
SPAN = TRACK_HALF - 2
SHAPER_FAMILIES = ["zv", "mzv", "zvd", "ei", "2hump_ei", "3hump_ei"]
ROLL_COLS_HALF = 80
ROLL_BLOCK_ROWS = 8
ROLL_LINE_TIME_US = 6.6865
ROLL_FRAME_PERIOD_S = 0.007143
ROLL_DEFAULT_EXPOSURE_US = 200
ROLL_MAX_FRAMES = 1600
ROLL_HALF = 14
ROLL_SPAN = 60
ROLL_MAX_STEP = 8.0
ROLL_MIN_EFFECTIVE_FS = 2000.0
ROLL_MIN_LIVE_BLOCKS = 24
PRINTER_CFG_REMOTE = "config/printer.cfg"
APPLY_TIMEOUT_S = 60.0


def _make_ref(frame):
    half = 200
    H, W = frame.shape
    best = None
    best_var = TEXTURE_MIN_VAR
    for cy in range(H // 2 - half, H // 2 + half - TRACK_SIZE, 16):
        for cx in range(W // 2 - half, W // 2 + half - TRACK_SIZE, 16):
            patch = frame[
                cy : cy + TRACK_SIZE, cx : cx + TRACK_SIZE
            ].astype(np.float32)
            v = float(patch.var())
            if v > best_var:
                best_var = v
                best = (cy, cx)
    if best is None:
        return None
    cy, cx = best
    ref = frame[cy : cy + TRACK_SIZE, cx : cx + TRACK_SIZE].astype(np.float32)
    ref -= ref.mean()
    norm = math.sqrt(float((ref * ref).sum()))
    if norm < 1e-6:
        return None
    ref /= norm
    region = TRACK_SIZE + 2 * SPAN
    pad = np.zeros((region, region), dtype=np.float32)
    pad[:TRACK_SIZE, :TRACK_SIZE] = ref
    ref_f = np.fft.fft2(pad)
    return (ref, ref_f, cy, cx)


def _track(frame, refp):
    ref, ref_f, cy, cx = refp
    region = TRACK_SIZE + 2 * SPAN
    y0 = cy - SPAN
    x0 = cx - SPAN
    if y0 < 0 or x0 < 0 or y0 + region > frame.shape[0] or x0 + region > frame.shape[1]:
        return None
    reg = frame[y0 : y0 + region, x0 : x0 + region].astype(np.float32)
    corr = np.fft.ifft2(np.fft.fft2(reg) * np.conj(ref_f)).real
    valid = corr[: 2 * SPAN + 1, : 2 * SPAN + 1]
    f64 = reg.astype(np.float64)
    sat = np.zeros((region + 1, region + 1))
    sat[1:, 1:] = np.cumsum(np.cumsum(f64, 0), 1)
    sat2 = np.zeros((region + 1, region + 1))
    sat2[1:, 1:] = np.cumsum(np.cumsum(f64 * f64, 0), 1)
    s = 2 * SPAN + 1
    tt = float(TRACK_SIZE * TRACK_SIZE)
    box1 = (
        sat[TRACK_SIZE:, TRACK_SIZE:]
        - sat[:-TRACK_SIZE, TRACK_SIZE:]
        - sat[TRACK_SIZE:, :-TRACK_SIZE]
        + sat[:-TRACK_SIZE, :-TRACK_SIZE]
    )
    box2 = (
        sat2[TRACK_SIZE:, TRACK_SIZE:]
        - sat2[:-TRACK_SIZE, TRACK_SIZE:]
        - sat2[TRACK_SIZE:, :-TRACK_SIZE]
        + sat2[:-TRACK_SIZE, :-TRACK_SIZE]
    )
    var = box2 - box1 * box1 / tt
    norm = np.sqrt(np.maximum(var, 0.0))
    scores = valid / np.maximum(norm, 1e-6)
    scores = np.where(norm > 1e-6, scores, -2.0)
    j, i = np.unravel_index(int(np.argmax(scores)), scores.shape)
    if scores[j, i] < MIN_CORRELATION:
        return None
    sx, sy = 0.0, 0.0
    if 0 < i < s - 1:
        row = scores[j]
        denom = row[i - 1] - 2.0 * row[i] + row[i + 1]
        if denom < -1e-9:
            sx = 0.5 * (row[i - 1] - row[i + 1]) / denom
    if 0 < j < s - 1:
        col = scores[:, i]
        denom = col[j - 1] - 2.0 * col[j] + col[j + 1]
        if denom < -1e-9:
            sy = 0.5 * (col[j - 1] - col[j + 1]) / denom
    return float(i + sx - SPAN), float(j + sy - SPAN), float(scores[j, i])


class ShaperDetectThing(OFMThing):
    """Detects stage resonance frequencies with the camera and fits Kalico
    input shapers. One action call sweeps one axis, records frames with the
    fast camera mode, template-tracks them into a displacement trace and
    feeds the excited axis through the stock shaper_calibrate PSD and
    scoring. Each axis is calibrated from its own run."""

    _cam: BaseCamera = lt.thing_slot()
    _strobe: "StrobeControlThing" = lt.thing_slot()

    def __init__(
            self,
            thing_server_interface: lt.ThingServerInterface,
            **kwargs: Any,
    ) -> None:
        self.port = kwargs.get("moonrakerport", 7125)
        self.baseurl = kwargs.get("baseurl", "http://127.0.0.1")
        self._client = httpx.Client(
            base_url=f"{self.baseurl}:{self.port}",
            timeout=httpx.Timeout(10.0, read=180.0),
        )
        self._lock = threading.Lock()
        self._sink: Optional[list] = None
        self._ref = None
        super().__init__(thing_server_interface)

    accel_per_hz: float = lt.setting(default=EXCITATION_ACCEL_PER_HZ)
    excitation_velocity: float = lt.setting(default=EXCITATION_VELOCITY)
    excitation_velocity_factor: float = lt.setting(default=EXCITATION_VELOCITY_FACTOR)
    rolling_auto_apply: bool = lt.setting(default=False)

    def _gcode(self, script: str) -> None:
        r = self._client.post(
            "/printer/gcode/script", json={"script": script}
        )
        r.raise_for_status()

    def _gcode_chunked(self, script: str, chunk_chars: int = 1400,
                       gap_s: float = 0.12) -> None:
        lines = script.splitlines()
        batch: list = []
        size = 0
        for line in lines:
            batch.append(line)
            size += len(line) + 1
            if size >= chunk_chars:
                self._gcode("\n".join(batch))
                batch = []
                size = 0
                time.sleep(gap_s)
        if batch:
            self._gcode("\n".join(batch))


    def _saved_shaper_from_cfg(self) -> dict:
        try:
            text = self._printer_cfg_text()
        except Exception:
            return {}
        vals = {}
        in_sec = False
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("["):
                in_sec = s == "[input_shaper]"
                continue
            if not in_sec or "=" not in s or s.startswith("#"):
                continue
            k, _, v = s.partition("=")
            vals[k.strip()] = v.strip()
        return vals

    def _query(self, objects: List[str]) -> dict:
        r = self._client.get(
            "/printer/objects/query",
            params={k: "" for k in objects},
        )
        r.raise_for_status()
        return r.json().get("result", r.json()).get("status", {})

    @staticmethod
    def _upsert_input_shaper_lines(
        text: str, axis: str, name: str, freq: float
    ) -> str:
        in_section = False
        tkey = "shaper_type_%s" % axis
        fkey = "shaper_freq_%s" % axis
        lines = text.splitlines(keepends=True)
        for idx, line in enumerate(lines):
            s = line.strip()
            if s.startswith("["):
                if in_section:
                    break
                in_section = s == "[input_shaper]"
                if in_section:
                    section_span = (idx, idx)
                continue
            if in_section:
                if not s or s.startswith("#"):
                    continue
                if "=" in s and s.split("=", 1)[0].strip() in (tkey, fkey):
                    section_span = (section_span[0], idx + 1)
        if section_span is not None and section_span[1] > section_span[0]:
            s0, s1 = section_span
            sect = []
            replaced = {"t": False, "f": False}
            for line in lines[s0 + 1 : s1]:
                s = line.strip()
                if s.startswith(tkey) and "=" in s:
                    sect.append("%s = %s\n" % (tkey, name))
                    replaced["t"] = True
                elif s.startswith(fkey) and "=" in s:
                    sect.append("%s = %.3f\n" % (fkey, freq))
                    replaced["f"] = True
                else:
                    sect.append(line)
            if not replaced["t"]:
                sect.append("%s = %s\n" % (tkey, name))
            if not replaced["f"]:
                sect.append("%s = %.3f\n" % (fkey, freq))
            return "".join(lines[: s0 + 1] + sect + lines[s1:])
        section = [
            "[input_shaper]\n",
            "%s = %s\n" % (tkey, name),
            "%s = %.3f\n" % (fkey, freq),
            "\n",
        ]
        out = []
        inserted = False
        for line in lines:
            if not inserted and line.lstrip().startswith("#*# <"):
                out.extend(section)
                inserted = True
            out.append(line)
        if not inserted:
            while out and not out[-1].strip():
                out.pop()
            out.append("\n")
            out.extend(section)
        return "".join(out)


    def _printer_cfg_text(self) -> str:
        r = self._client.get(
            "/server/files/config/printer.cfg", timeout=APPLY_TIMEOUT_S
        )
        r.raise_for_status()
        return r.text

    def _write_printer_cfg(self, text: str) -> None:
        r = self._client.post(
            "/server/files/upload",
            data={"root": "config", "filename": "printer.cfg"},
            files={
                "file": (
                    "printer.cfg",
                    text.encode("utf-8"),
                    "text/plain",
                )
            },
            timeout=APPLY_TIMEOUT_S,
        )
        r.raise_for_status()

    def _restart_klippy(self) -> None:
        r = self._client.post("/printer/restart", timeout=APPLY_TIMEOUT_S)
        r.raise_for_status()
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            try:
                info = self._client.get(
                    "/printer/info", timeout=5.0
                ).json()
                if info.get("result", info).get("state") == "ready":
                    return
            except Exception:
                pass
            time.sleep(0.5)
        raise RuntimeError("klippy did not return to ready within 30s")

    def _apply_shaper_cfg(self, axis: str, name: str, freq: float) -> dict:
        text = self._printer_cfg_text()
        new_text = self._upsert_input_shaper_lines(text, axis, name, freq)
        if new_text == text:
            return {"written": False, "restart": False,
                    "note": "printer.cfg already carries these values"}
        self._write_printer_cfg(new_text)
        self._restart_klippy()
        check = self._printer_cfg_text()
        import re as _re
        ok = _re.search(
            r"(?mi)^\s*shaper_freq_%s\s*=\s*%.3f\s*$" % (axis, freq),
            check,
        )
        return {"written": True, "restart": True,
                "verified": bool(ok)}

    def _maybe_apply(self, report: dict, axis: str) -> None:
        res = report.get(axis)
        if not isinstance(res, dict) or "recommended" not in res:
            report["applied"] = None
            return
        rec = res["recommended"]
        try:
            report["applied"] = self._apply_shaper_cfg(
                axis, str(rec["name"]), float(rec["freq"])
            )
        except Exception as exc:
            LOGGER.exception("auto-apply of shaper failed")
            report["applied"] = {"written": False, "error": str(exc)}

    def _motion_queue_drained(self) -> bool:
        status = self._query(["toolhead"])["toolhead"]
        return (
            float(status["print_time"]) - float(status["estimated_print_time"])
            <= 0.05
        )

    def _camera_hook(self, request) -> None:
        self._hook_calls += 1
        sink = self._sink
        if sink is None:
            return
        try:
            ts = float(request.get_metadata()["SensorTimestamp"]) / 1e9
            frame = request.make_array("main")[:FRAME_H, :FRAME_W]
            ref = self._ref
            if ref is None:
                self._ref = _make_ref(frame)
                if self._ref is None:
                    self._no_texture = True
                    self._hook_errors += 1
                    return
                result = (0.0, 0.0, 1.0)
            else:
                result = _track(frame, ref)
        except Exception:
            self._hook_errors += 1
            return
        if result is None:
            self._hook_errors += 1
            return
        dx, dy, score = result
        sink.append((ts, dx, dy, score))

    def _sweep_pairs(self, freq_min: float, freq_max: float) -> List[float]:
        pairs = []
        f = freq_min
        while f <= freq_max + 1e-9:
            pairs.append(f)
            f *= 2.0 ** (1.0 / OCTAVES_PER_PAIR)
        return pairs

    def _excitation_script(self, axis, freq_min, freq_max, seconds):
        pairs = self._sweep_pairs(freq_min, freq_max)
        one_pass = sum(1.0 / f for f in pairs)
        repeats = max(1, int(round(seconds / one_pass)))
        lines = []
        v = self.excitation_velocity * self.excitation_velocity_factor
        vf = v
        for _ in range(repeats):
            for f in pairs:
                d = v / (2.0 * f)
                lines.append(f"G1 {axis}{d:.6f} F{vf * 60.0:.1f} S{vf:.1f}")
                lines.append(f"G1 {axis}-{d:.6f} F{vf * 60.0:.1f} S{vf:.1f}")
        return "\n".join(lines), len(pairs) * repeats

    def _wait_frames(self, count: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            sink = self._sink
            if isinstance(sink, tuple):
                if sink[1][0] >= count:
                    return True
            elif isinstance(sink, list) and len(sink) >= count:
                return True
            time.sleep(0.05)
        return False

    def _wait_motion_done(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        time.sleep(0.2)
        while time.monotonic() < deadline:
            if self._motion_queue_drained():
                return True
            time.sleep(0.1)
        return False

    @lt.action
    def calibrate_input_shaper(
            self,
            axis: Literal["x", "y"],
            freq_min: float = DEFAULT_MIN_FREQ,
            freq_max: float = DEFAULT_MAX_FREQ,
            seconds: float = DEFAULT_SECONDS,
            shapers: Optional[List[str]] = None,
            save_trace: bool = True,
    ) -> dict:
        with self._lock:
            return self._run(axis, freq_min, freq_max, seconds, shapers,
                             save_trace)

    def _run(self, axis, freq_min, freq_max, seconds, shapers, save_trace):
        if axis not in ("x", "y"):
            raise ValueError("axis must be x or y")
        if freq_max <= freq_min or freq_min < 5.0 or freq_max > 200.0:
            raise ValueError("require 5 <= freq_min < freq_max <= 200")
        st_state = dict(self._strobe.state)
        width = st_state.get("width_us")
        if width is not None and width > 100000:
            self._strobe.exit_continuous_light()
        status = self._query(["input_shaper", "toolhead"])
        saved_shaper = status.get("input_shaper") or self._saved_shaper_from_cfg()
        saved_accel = float(status["toolhead"]["max_accel"])
        cam = self._cam
        prev_mode = getattr(cam, "streaming_mode", None)
        self._ref = None
        self._no_texture = False
        self._hook_calls = 0
        self._hook_errors = 0
        self._hook_last_err = None
        self._hook_cpu_ns = 0
        sink: list = []
        script, n_segments = self._excitation_script(
            axis, freq_min, freq_max, seconds
        )
        shaper_off = True
        try:
            self._gcode(
                "SET_INPUT_SHAPER SHAPER_FREQ_X=0 SHAPER_FREQ_Y=0"
            )
        except Exception:
            LOGGER.exception("shaper disable failed; continuing")
        enc_stopped = False
        try:
            cam.change_streaming_mode("crop990")
            picam = cam._picamera
            orig_pre = picam.pre_callback

            def _wrapped(request, _orig=orig_pre):
                self._camera_hook(request)

            picam.pre_callback = _wrapped
            try:
                picam.stop_encoder()
                enc_stopped = True
            except Exception:
                LOGGER.exception("MJPEG encoder stop failed; capturing with encoders active")
            self._sink = sink
            if not self._wait_frames(12, 5.0):
                hint = (
                    "; no textured feature in the central region - "
                    "move a detailed part of the sample into view"
                    if getattr(self, "_no_texture", False)
                    else ""
                )
                raise RuntimeError(
                    f"camera produced no tracked frames "
                    f"(hook calls={self._hook_calls}, "
                    f"hook errors={self._hook_errors}){hint}"
                )
            self._gcode(
                "SET_KINEMATIC_POSITION SET_HOMED=XYZ\n"
                + script + "\nSYNC_PULSE_STATUS STROBE=camera"
            )
            if not self._wait_motion_done(seconds * 2.0 + 30.0):
                raise RuntimeError("excitation did not finish in time")
            cap_end = time.monotonic()
            self._sink = None
            tail = time.monotonic() + 0.4
            while time.monotonic() < tail:
                time.sleep(0.05)
        finally:
            self._sink = None
            if enc_stopped:
                try:
                    cam._stop_streaming(stop_web_stream=False)
                except Exception:
                    LOGGER.exception("post-capture stream stop failed")
                try:
                    cam._start_streaming(getattr(cam, "streaming_mode", "crop990"))
                except Exception:
                    LOGGER.exception("post-capture stream restart failed")
            try:
                self._gcode(f"M204 S{saved_accel:.0f}")
                if shaper_off:
                    tx = saved_shaper.get("shaper_type_x")
                    ty = saved_shaper.get("shaper_type_y")
                    fx = saved_shaper.get("shaper_freq_x")
                    fy = saved_shaper.get("shaper_freq_y")
                    restore = []
                    if tx:
                        restore.append(f"SHAPER_TYPE_X={tx}")
                    if ty:
                        restore.append(f"SHAPER_TYPE_Y={ty}")
                    if fx is not None:
                        restore.append(f"SHAPER_FREQ_X={float(fx):.3f}")
                    if fy is not None:
                        restore.append(f"SHAPER_FREQ_Y={float(fy):.3f}")
                    if restore:
                        self._gcode(
                            "SET_INPUT_SHAPER " + " ".join(restore)
                        )
            except Exception:
                LOGGER.exception("shaper state restore failed")
            if prev_mode:
                try:
                    cam.change_streaming_mode(prev_mode)
                except Exception:
                    LOGGER.exception("streaming mode restore failed")
        if len(sink) < 140:
            raise RuntimeError(
                f"only {len(sink)} usable frames; need more for analysis"
            )
        arr = np.array(sink, dtype=np.float64)
        t = arr[:, 0] - arr[0, 0]
        fs = 1.0 / float(np.median(np.diff(t)))
        raw = np.column_stack([t, arr[:, 1], arr[:, 2], arr[:, 3]])
        trace_path = None
        if save_trace:
            DATA_FOLDER.mkdir(parents=True, exist_ok=True)
            trace_path = DATA_FOLDER / (
                f"{time.strftime('%Y%m%d_%H%M%S')}_{axis}.npz"
            )
            np.savez_compressed(
                trace_path,
                t=t, dx=arr[:, 1], dy=arr[:, 2], score=arr[:, 3], fs=fs,
            )
        if freq_max > 0.47 * fs:
            raise RuntimeError(
                f"freq_max {freq_max} exceeds the camera Nyquist limit "
                f"for fs={fs:.1f}; resonances above fs/2 alias into the PSD. "
                f"Trace saved to {trace_path}"
            )
        col = 1 if axis == "x" else 2
        other_col = 2 if axis == "x" else 1
        report = {
            "axis": axis,
            "frames": int(arr.shape[0]),
            "fs_measured": fs,
            "duration_s": float(t[-1]),
            "segments": n_segments,
            "trace_file": str(trace_path) if trace_path else None,
            "track_score_median": float(np.median(arr[:, 3])),
            "track_score_p10": float(np.percentile(arr[:, 3], 10)),
        }
        calibrator = _load_kalico_calibrator()
        try:
            report[axis] = _analyze_axis(calibrator, raw, col, shapers)
        except Exception as exc:
            LOGGER.exception("axis %s analysis failed", axis)
            report[axis] = {"error": str(exc)}
        try:
            cross = _analyze_axis(calibrator, raw, other_col, shapers=None)
            report[f"cross_axis_{ 'y' if axis == 'x' else 'x' }"] = {
                "dominant_peak_hz": cross["dominant_peak_hz"],
                "note": "informational only; calibrate this axis "
                        "with its own run",
            }
        except Exception:
            pass
        self._maybe_apply(report, axis)
        return report


    def _rolling_camera_hook(self, request) -> None:
        _t0 = time.perf_counter_ns()
        sink = self._sink
        if sink is None:
            return
        try:
            buf, wpos, ts_arr, geo = sink
            k = wpos[0]
            if k >= ROLL_MAX_FRAMES:
                return
            frame = request.make_array("main")
            if geo[0] is None:
                cols = frame.shape[1]
                rows = min(buf.shape[1], cols)   # YUV420: cols == luma height
                if rows < ROLL_BLOCK_ROWS * 8:
                    raise ValueError(
                        f"stream luma rows {rows} too small"
                    )
                cx = cols // 2
                geo[0] = (
                    slice(0, rows),
                    slice(cx - ROLL_COLS_HALF, cx + ROLL_COLS_HALF),
                )
            ts_arr[k] = request.get_metadata()["SensorTimestamp"] / 1e9
            np.copyto(buf[k], frame[geo[0]])
        except Exception as exc:
            self._hook_errors += 1
            self._hook_calls += 1
            self._hook_last_err = repr(exc)[:200]
            return
        wpos[0] = k + 1
        self._hook_calls += 1
        self._hook_cpu_ns += time.perf_counter_ns() - _t0

    @lt.action
    def calibrate_input_shaper_rolling(
            self,
            axis: Literal["x", "y"],
            freq_min: float = DEFAULT_MIN_FREQ,
            freq_max: float = DEFAULT_MAX_FREQ,
            seconds: float = DEFAULT_SECONDS,
            exposure_us: int = ROLL_DEFAULT_EXPOSURE_US,
            shapers: Optional[List[str]] = None,
            save_trace: bool = True,
    ) -> dict:
        with self._lock:
            return self._run_rolling(
                axis, freq_min, freq_max, seconds, exposure_us, shapers,
                save_trace,
            )

    def _run_rolling(self, axis, freq_min, freq_max, seconds, exposure_us,
                     shapers, save_trace):
        if axis not in ("x", "y"):
            raise ValueError("axis must be x or y")
        if freq_max <= freq_min or freq_min < 5.0 or freq_max > 200.0:
            raise ValueError("require 5 <= freq_min < freq_max <= 200")
        cont = getattr(self._strobe, "_continuous", None)
        if cont is None:
            st_state = dict(self._strobe.state)
            width = st_state.get("width_us")
            cont = width is not None and width > 100000
        held_light = bool(cont)
        if not held_light:
            self._strobe.enter_continuous_light()
        status = self._query(["input_shaper", "toolhead"])
        saved_shaper = status.get("input_shaper") or self._saved_shaper_from_cfg()
        saved_accel = float(status["toolhead"]["max_accel"])
        cam = self._cam
        prev_mode = getattr(cam, "streaming_mode", None)
        self._hook_calls = 0
        self._hook_errors = 0
        self._hook_last_err = None
        self._hook_cpu_ns = 0
        strips_buf = np.zeros(
            (ROLL_MAX_FRAMES, 700, 2 * ROLL_COLS_HALF), dtype=np.uint8
        )
        ts_buf = np.zeros(ROLL_MAX_FRAMES, dtype=np.float64)
        wpos = [0]
        script, n_segments = self._excitation_script(
            axis, freq_min, freq_max, seconds
        )
        shaper_off = True
        try:
            self._gcode(
                "SET_INPUT_SHAPER SHAPER_FREQ_X=0 SHAPER_FREQ_Y=0"
            )
        except Exception:
            LOGGER.exception("shaper disable failed; continuing")
        enc_stopped = False
        try:
            cam.change_streaming_mode("crop990")
            prev_exposure = getattr(cam, "exposure_time", None)
            prev_gain = getattr(cam, "analogue_gain", None)
            cam.exposure_time = int(exposure_us)
            cam.analogue_gain = 1.0
            picam = cam._picamera
            try:
                picam.stop_encoder()
                enc_stopped = True
            except Exception:
                LOGGER.exception("MJPEG encoder stop failed; capturing with encoders active")
            sink = (strips_buf, wpos, ts_buf, [None])
            orig_pre = picam.pre_callback
            self._sink = sink

            def _wrapped(request, _orig=orig_pre):
                self._rolling_camera_hook(request)

            picam.pre_callback = _wrapped

            cap_start = time.monotonic()
            deadline = time.monotonic() + 5.0
            while wpos[0] < 12 and time.monotonic() < deadline:
                time.sleep(0.05)
            if wpos[0] < 12:
                raise RuntimeError(
                    f"camera delivered only {wpos[0]} frames in 5s "
                    f"(hook errors={self._hook_errors}, last={getattr(self, '_hook_last_err', '?')}); scene too dark at "
                    f"{exposure_us}us?"
                )
            self._gcode(
                "SET_KINEMATIC_POSITION SET_HOMED=XYZ\n"
                + script + "\nSYNC_PULSE_STATUS STROBE=camera"
            )
            if not self._wait_motion_done(seconds * 2.0 + 30.0):
                raise RuntimeError("excitation did not finish in time")
            cap_end = time.monotonic()
            self._sink = None
        finally:
            self._sink = None
            try:
                picam.pre_callback = orig_pre
            except Exception:
                LOGGER.exception("pre_callback restore failed")
            if enc_stopped:
                try:
                    cam._stop_streaming(stop_web_stream=False)
                except Exception:
                    LOGGER.exception("post-capture stream stop failed")
                try:
                    cam._start_streaming(getattr(cam, "streaming_mode", "crop990"))
                except Exception:
                    LOGGER.exception("post-capture stream restart failed")
            try:
                self._gcode(f"M204 S{saved_accel:.0f}")
                if shaper_off:
                    tx = saved_shaper.get("shaper_type_x")
                    ty = saved_shaper.get("shaper_type_y")
                    fx = saved_shaper.get("shaper_freq_x")
                    fy = saved_shaper.get("shaper_freq_y")
                    restore = []
                    if tx:
                        restore.append(f"SHAPER_TYPE_X={tx}")
                    if ty:
                        restore.append(f"SHAPER_TYPE_Y={ty}")
                    if fx is not None:
                        restore.append(f"SHAPER_FREQ_X={float(fx):.3f}")
                    if fy is not None:
                        restore.append(f"SHAPER_FREQ_Y={float(fy):.3f}")
                    if restore:
                        self._gcode(
                            "SET_INPUT_SHAPER " + " ".join(restore)
                        )
                if prev_exposure is not None:
                    cam.exposure_time = prev_exposure
                if prev_gain is not None:
                    cam.analogue_gain = prev_gain
                if not held_light:
                    self._strobe.exit_continuous_light()
                if prev_mode is not None and prev_mode != "crop990":
                    cam.change_streaming_mode(prev_mode)
            except Exception:
                LOGGER.exception("rolling-run restore failed")
        n_captured = wpos[0]
        if n_captured < 100:
            raise RuntimeError(f"only {n_captured} usable frames captured")
        raw, texture_blocks = _rolling_series(
            strips_buf[:n_captured], ts_buf[:n_captured], axis
        )
        smear_hz = 0.05 / (exposure_us * 1e-6)
        eff_fs = 1.0 / float(np.median(np.diff(raw[:, 0])))
        trace_path = None
        if save_trace:
            out = Path("/home/admin/ofm/openflexure/data/shaper_detect")
            out.mkdir(parents=True, exist_ok=True)
            trace_path = out / f"rolling_{axis}_{int(time.time())}.npz"
            sample_idx = np.linspace(
                0, n_captured - 1, 5, dtype=int
            )
            np.savez_compressed(
                trace_path,
                raw=raw,
                sample_strips=strips_buf[sample_idx],
                sample_ts=ts_buf[sample_idx],
            )
        delivered_fps = n_captured / max(
            cap_end - cap_start, 1e-6
        )
        live_blocks = int(texture_blocks)
        if eff_fs < ROLL_MIN_EFFECTIVE_FS:
            raise RuntimeError(
                f"effective sample rate only {eff_fs:.0f} Hz "
                f"(need >= {ROLL_MIN_EFFECTIVE_FS:.0f} for the rolling "
                f"band) - rolling analysis invalid; trace saved to "
                f"{trace_path}"
            )
        if live_blocks < ROLL_MIN_LIVE_BLOCKS:
            raise RuntimeError(
                f"only {live_blocks} of {texture_blocks} textured strip "
                f"blocks tracked real motion (rest locked onto static "
                f"pattern: reposition the sample so detailed structure "
                f"fills the central strip, or raise exposure_us); trace "
                f"saved to {trace_path}"
            )
        report = {
            "mode": "rolling",
            "axis": axis,
            "exposure_us": int(exposure_us),
            "frames": int(n_captured),
            "hook_cpu_ms": round(self._hook_cpu_ns / 1e6, 1),
            "hook_cpu_per_frame_ms": round(
                self._hook_cpu_ns / max(n_captured, 1) / 1e6, 3
            ),
            "block_sample_rate_hz": float(
                1.0 / (ROLL_BLOCK_ROWS * ROLL_LINE_TIME_US * 1e-6)
            ),
            "effective_sample_rate_hz": float(eff_fs),
            "delivered_fps": float(delivered_fps),
            "live_blocks": live_blocks,
            "dead_blocks": 123 - live_blocks,
            "smear_ceiling_hz": float(smear_hz),
            "segments": int(n_segments),
            "trace": str(trace_path) if trace_path else None,
        }
        if freq_max > smear_hz:
            report["warning"] = (
                f"freq_max {freq_max} above per-row smear ceiling "
                f"{smear_hz:.0f} Hz: peaks stay detectable but their "
                "amplitudes sag"
            )
        dec_m = max(1, int(eff_fs / 500.0))
        raw_a = _decimate(raw, dec_m)
        report["analysis_decimation"] = dec_m
        report["analysis_fs_hz"] = float(
            1.0 / float(np.median(np.diff(raw_a[:, 0])))
        )
        calibrator = _load_kalico_calibrator()
        res = None
        try:
            res = _analyze_axis(calibrator, raw_a, 1, shapers)
            report[axis] = res
        except Exception as exc:
            LOGGER.exception("rolling axis %s analysis failed", axis)
            report[axis] = {"error": str(exc)}
        report["applied"] = None
        report["rolling_auto_apply"] = self.rolling_auto_apply
        rec = (res or {}).get("recommended")
        peak = (res or {}).get("dominant_peak_hz")
        sane = (
            res is not None
            and isinstance(peak, (int, float))
            and 10.0 <= float(peak) <= 200.0
        )
        report["peak_plausible"] = sane
        if res is not None and not sane:
            report["gate"] = (
                f"dominant peak {float(peak):.0f} Hz outside plausible "
                "structural range [10, 200] Hz - treated as tracking "
                "noise; result advisory only"
            )
        if sane and self.rolling_auto_apply:
            self._maybe_apply(report, axis)
        return report


def _decimate(raw, m):
    if m <= 1:
        return raw
    kernel = np.ones(m, dtype=np.float64) / m
    cols = [
        np.convolve(raw[:, c], kernel, mode="same")[::m]
        for c in range(raw.shape[1])
    ]
    n = min(len(c) for c in cols)
    return np.column_stack([c[:n] for c in cols])


def _rolling_series(strips_buf, ts_buf, axis):
    strips = []
    for k in range(strips_buf.shape[0]):
        s = strips_buf[k].astype(np.float32)
        if float((s * s).sum()) < 1e-6:
            continue
        if axis == "y":
            s = np.ascontiguousarray(np.rot90(s, k=-1))
        strips.append((float(ts_buf[k]), s))
    if len(strips) < 2:
        raise RuntimeError("no usable strips")
    h0 = strips[0][1]
    n_blocks = h0.shape[0] // ROLL_BLOCK_ROWS
    if n_blocks < 8:
        raise RuntimeError("strip too short for block tracking")
    width = h0.shape[1]
    refs = []
    for b in range(n_blocks):
        y0 = b * ROLL_BLOCK_ROWS
        patch = h0[y0 : y0 + ROLL_BLOCK_ROWS,
                   width // 2 - ROLL_HALF : width // 2 + ROLL_HALF]
        if float(patch.var()) < TEXTURE_MIN_VAR:
            refs.append(None)
            continue
        patch = patch - patch.mean()
        n = math.sqrt(float((patch * patch).sum()))
        refs.append(patch / n if n > 1e-6 else None)
    live = sum(1 for r in refs if r is not None)
    if live < 8:
        raise RuntimeError(
            f"only {live} of {n_blocks} rolling blocks have texture "
            f"(variance >= {TEXTURE_MIN_VAR:.0f}) - reposition the sample "
            "or raise exposure_us"
        )
    xs = np.arange(-ROLL_SPAN, ROLL_SPAN + 1)
    W = 2 * ROLL_HALF
    shifts = np.zeros((len(strips), n_blocks), dtype=np.float32)
    shifts[:] = np.nan
    for fi, (base_t, s) in enumerate(strips):
        x0s = np.clip(width // 2 - ROLL_HALF + xs, 0, width - W)
        windows = np.stack(
            [
                s[b * ROLL_BLOCK_ROWS : (b + 1) * ROLL_BLOCK_ROWS, x0 : x0 + W]
                for b in range(n_blocks) for x0 in x0s
            ]
        ).astype(np.float32)
        m = windows - windows.mean(axis=(1, 2), keepdims=True)
        norms = np.sqrt((m * m).sum(axis=(1, 2)))
        ref_ok = [refs[b] is not None for b in range(n_blocks)]
        zref = np.zeros((ROLL_BLOCK_ROWS, W), dtype=np.float32)
        refv = np.stack(
            [refs[b] if refs[b] is not None else zref
             for b in range(n_blocks) for _ in xs]
        )
        corr = (m * refv).sum(axis=(1, 2)).reshape(n_blocks, len(xs)) / np.maximum(
            norms, 1e-6
        ).reshape(n_blocks, len(xs))
        corr = np.where(norms.reshape(n_blocks, len(xs)) > 1e-6, corr, -2.0)
        j = np.argmax(corr, axis=1)
        peaks = corr[np.arange(n_blocks), j]
        ok = peaks >= MIN_CORRELATION
        ok &= np.array(ref_ok)
        sub = np.zeros(n_blocks, dtype=np.float32)
        rows = np.nonzero(ok)[0]
        for b in rows:
            jj = j[b]
            if 0 < jj < len(xs) - 1:
                den = corr[b, jj - 1] - 2.0 * corr[b, jj] + corr[b, jj + 1]
                if den < -1e-9:
                    sub[b] = 0.5 * (corr[b, jj - 1] - corr[b, jj + 1]) / den
        shifts[fi, ok] = (j[ok] + sub[ok]) - ROLL_SPAN
    ts_out, d_out = [], []
    for fi, (base_t, _s) in enumerate(strips):
        for b in range(n_blocks):
            v = shifts[fi, b]
            if np.isnan(v):
                continue
            t_b = base_t + (b + 0.5) * ROLL_BLOCK_ROWS * ROLL_LINE_TIME_US * 1e-6
            ts_out.append(t_b)
            d_out.append(float(v))
    t = np.asarray(ts_out, dtype=np.float64)
    d = np.asarray(d_out, dtype=np.float64)
    order = np.argsort(t)
    t, d = t[order], d[order]
    z = np.zeros_like(d)
    texture_blocks = sum(1 for r in refs if r is not None)
    return np.column_stack([t, d, z, z]), texture_blocks


def _load_kalico_calibrator():
    import importlib.util

    candidates = [
        Path("/home/admin/klipper/klippy"),
        Path("C:/glia/EU_dotacje/mikroskop/kallicofork/kalico/klippy"),
    ]
    for base in candidates:
        if not base.is_dir():
            continue
        if not (base / "extras" / "__init__.py").is_file():
            continue
        if str(base) not in sys.path:
            sys.path.insert(0, str(base))
        mod = importlib.import_module("extras.shaper_calibrate")
        return mod.ShaperCalibrate(None)
    raise RuntimeError("kalico klippy extras not found for shaper analysis")


def _analyze_axis(calibrator, raw, col, shapers):
    data = np.column_stack(
        [raw[:, 0], raw[:, col], raw[:, col], raw[:, col]]
    )
    calibration = calibrator.calc_freq_response(data)
    if calibration is None:
        raise RuntimeError("not enough frames for a 0.5s PSD window")
    calibration.set_numpy(calibrator.numpy)
    w = (2.0 * np.pi * calibration.freq_bins) ** 4
    for attr in ("psd_sum", "psd_x", "psd_y", "psd_z"):
        setattr(calibration, attr, getattr(calibration, attr) * w)
    calibration.normalize_to_frequencies()
    best, _all = calibrator.find_best_shaper(
        calibration,
        shapers=shapers,
        max_smoothing=0.5,
        scv=5.0,
    )
    if best is None:
        raise RuntimeError("no shaper could be fitted")
    psd_axis = calibration.psd_x if col == 1 else calibration.psd_y
    lo = calibration.freq_bins >= 10.0
    if not lo.any():
        raise RuntimeError("frequency axis empty above 10 Hz")
    band = np.where(lo, psd_axis, 0.0)
    prominence = float(band.max() / max(np.median(psd_axis[lo]), 1e-12))
    if prominence < 4.0:
        raise RuntimeError(
            "no dominant resonance in the swept band "
            f"(peak prominence {prominence:.1f}x median; need >= 4x). "
            "Raise excitation_velocity, widen the band, or check "
            "that the sample feature is tracked."
        )
    peak_idx = int(np.argmax(band))
    table = []
    families = shapers or SHAPER_FAMILIES
    for name in families:
        try:
            res, _all = calibrator.find_best_shaper(
                calibration, shapers=[name], max_smoothing=0.5, scv=5.0,
            )
            if res is None:
                continue
        except Exception:
            continue
        table.append(
            {
                "name": res.name,
                "freq": float(res.freq),
                "vibrs_pct": float(res.vibrs) * 100.0,
                "smoothing": float(res.smoothing),
                "score": float(res.score),
            }
        )
    table.sort(key=lambda r: r["score"])
    return {
        "recommended": {
            "name": best.name,
            "freq": float(best.freq),
            "vibrs_pct": float(best.vibrs) * 100.0,
            "smoothing": float(best.smoothing),
            "score": float(best.score),
        },
        "dominant_peak_hz": float(calibration.freq_bins[peak_idx]),
        "shaper_table": table,
    }
