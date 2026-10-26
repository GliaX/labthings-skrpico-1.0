"""Hardware-synced illumination strobe control.

The strobe is the shutter: camera exposure is pinned high so it always
spans most of the frame, and image brightness is steered by the LED flash
width, re-gated by the firmware on every frame's blanking edge. The
calibrated quantity is ``flash_duty`` (flash width / frame period), which
is optics- and scene-dependent but frame-rate-independent, so it transfers
to any streaming mode by measuring that mode's frame rate live from the
firmware pulse counter.
"""

import time
from typing import Any, Mapping, Optional

import httpx
import labthings_fastapi as lt
import numpy as np

from openflexure_microscope_server.things import OFMThing
from .camera import BaseCamera

DEFAULT_PERIOD_US = 45475
COUNTER_TIMEOUT_S = 2.5


class StrobeControlThing(OFMThing):
    """Controls the MCU-timed illumination strobe and holds its calibration.

    Exposure is pinned high (AEC off) and image brightness is steered with
    the LED flash width. ``flash_duty`` is the invariant produced by
    calibration; the working flash width is always ``flash_duty`` times the
    measured frame period of the current streaming mode.
    """

    _cam: BaseCamera = lt.thing_slot()

    def __init__(
            self,
            thing_server_interface: lt.ThingServerInterface,
            **kwargs: Any,
    ) -> None:
        self.port = kwargs.get("moonrakerport", 7125)
        self.baseurl = kwargs.get("baseurl", "http://127.0.0.1")
        self.timeout = httpx.Timeout(10.0)
        self._client = httpx.Client(
            base_url=f"{self.baseurl}:{self.port}", timeout=self.timeout
        )
        self._last_width_us: Optional[int] = None
        self._last_error: Optional[str] = None
        super().__init__(thing_server_interface)

    enabled: bool = lt.setting(default=True)
    flash_duty: float = lt.setting(default=0.011)
    mode_duties: dict = lt.setting(default_factory=dict)
    exposure_us: int = lt.setting(default=0)
    analogue_gain: float = lt.setting(default=1.0)

    def __enter__(self) -> "StrobeControlThing":
        """Pin exposure high and arm the strobe when the server starts."""
        super().__enter__()
        if not self.enabled:
            return self
        try:
            period = self._period_us()
            self._pin_exposure(period)
            duty = self.mode_duties.get(
                getattr(self._cam, "streaming_mode", "default"), self.flash_duty
            )
            width = max(20, min(int(duty * period), period - 1000))
            self._send(
                f"HWSYNC_STROBE STROBE=camera ENABLE=1 WIDTH={width / 1e6:.6f}"
            )
            self._send("SYNC_PULSE_STATUS STROBE=camera")
            self._last_width_us = width
            self.logger.info(
                "Strobe armed at %d us (period %d us, duty %.5f)",
                width,
                period,
                self.flash_duty,
            )
        except Exception as exc:
            self._last_error = str(exc)
            self.logger.warning("Strobe not armed: %s", exc)
        return self

    @lt.property
    def sync_calibration(self) -> Mapping[str, Mapping[str, Any]]:
        """Reference sync timings measured on this camera."""
        return {
            "default": {"fps": 22.0, "blanking_us": 225, "frame_period_us": 45475},
            "full_resolution": {
                "fps": 20.8,
                "blanking_us": 225,
                "frame_period_us": 45475,
            },
            "crop990": {
                "fps": 102.4,
                "blanking_us": 2930,
                "frame_period_us": 9755,
            },
        }

    @lt.property
    def state(self) -> Mapping[str, Any]:
        """Last known firmware state: armed, active, count, width_us."""
        state: dict[str, Any] = {
            "armed": None,
            "active": None,
            "count": None,
            "width_us": self._last_width_us,
            "error": self._last_error,
        }
        try:
            response = self._client.get(
                "/server/gcode_store", params={"count": 100}
            )
            response.raise_for_status()
            store = response.json()["result"]["gcode_store"]
            for entry in reversed(store):
                message = entry.get("message", "")
                if message.startswith("// sync_pulse:"):
                    for field in message.split("// sync_pulse:")[1].split():
                        key, _, value = field.partition("=")
                        if key in ("armed", "active"):
                            state[key] = value == "1"
                        elif key == "count":
                            state["count"] = int(value)
                        elif key == "width":
                            state["width_us"] = round(int(value) / 12.0)
                    break
        except Exception:
            pass
        return state

    @lt.action
    def strobe_on(self, width_us: Optional[int] = None) -> None:
        """Arm the strobe at the calibrated duty for the current frame
        rate, or at an explicit width if given."""
        if width_us is None:
            period = self._period_us()
            duty = self.mode_duties.get(self._current_mode(), self.flash_duty)
            width_us = max(20, min(int(duty * period), period - 1000))
        self._send(
            f"HWSYNC_STROBE STROBE=camera ENABLE=1 WIDTH={width_us / 1e6:.6f}"
        )
        self._send("SYNC_PULSE_STATUS STROBE=camera")
        self._last_width_us = width_us
        self._last_error = None

    @lt.action
    def strobe_off(self) -> None:
        """Disarm the strobe."""
        self._send("HWSYNC_STROBE STROBE=camera ENABLE=0")

    @lt.action
    def set_pulse_width(self, microseconds: int) -> None:
        """Set the flash width directly (manual override; also updates
        ``flash_duty`` from the measured frame period)."""
        period = self._period_us()
        if microseconds >= period:
            raise ValueError(
                f"{microseconds} us exceeds the {period} us frame period"
            )
        if microseconds < 20:
            raise ValueError("flash width must be at least 20 us")
        self._send(
            f"HWSYNC_STROBE STROBE=camera ENABLE=1 "
            f"WIDTH={microseconds / 1e6:.6f}"
        )
        self._send("SYNC_PULSE_STATUS STROBE=camera")
        self.flash_duty = microseconds / period
        self._last_width_us = microseconds

    @lt.action
    def calibrate_flash(
            self,
            target_p99: float = 230.0,
            min_width_us: int = 20,
            max_width_us: Optional[int] = None,
            settle_s: float = 0.6,
            iterations: int = 12,
    ) -> Mapping[str, Any]:
        """Bisect the flash width against a target bright-tail percentile.

        Measures the current frame rate from the firmware pulse counter,
        pins exposure high for that rate, then bisects width until the
        99th percentile of a grabbed frame reaches ``target_p99``. For
        histopathology the goal is maximum photon budget with an unclipped
        background: tissue-free glass sits just below saturation. Stores
        the result as ``flash_duty`` so any other frame rate can be
        recovered as ``duty * period``.
        """
        period = self._period_us()
        self._pin_exposure(period)
        hi = max_width_us if max_width_us is not None else int(period * 0.9)
        lo = max(min_width_us, 20)
        if hi <= lo:
            raise ValueError(
                f"search range [{lo}, {hi}] us is empty for period {period} us"
            )
        best: Optional[tuple[int, float]] = None
        converged = False
        result_p99: Optional[float] = None
        result_mean: Optional[float] = None
        chosen = lo
        for i in range(iterations):
            chosen = (lo + hi) // 2
            self._send(
                f"HWSYNC_STROBE STROBE=camera ENABLE=1 "
                f"WIDTH={chosen / 1e6:.6f}"
            )
            time.sleep(max(settle_s, 3.0 * (period / 1e6)))
            frame = self._cam.grab_as_array(stream_name="lores")
            result_p99 = float(np.percentile(frame, 99))
            result_mean = float(np.mean(frame))
            best = (chosen, result_p99)
            self.logger.info(
                "calibrate_flash %d/%d: width=%d us p99=%.1f mean=%.1f",
                i + 1,
                iterations,
                chosen,
                result_p99,
                result_mean,
            )
            if abs(result_p99 - target_p99) <= 2.0:
                converged = True
                break
            if result_p99 < target_p99:
                lo = chosen
            else:
                hi = chosen
        if best is None or result_mean is None:
            raise RuntimeError(
                "calibrate_flash produced no measurement; is the preview "
                "stream running?"
            )
        duty = chosen / period
        self.flash_duty = duty
        if converged:
            self.mode_duties = {
                **self.mode_duties,
                self._current_mode(): duty,
            }
        self._send("SYNC_PULSE_STATUS STROBE=camera")
        report = {
            "converged": converged,
            "mode": self._current_mode(),
            "target_p99": target_p99,
            "p99": result_p99,
            "mean": result_mean,
            "period_us": period,
            "width_us": chosen,
            "flash_duty": duty,
        }
        self.logger.info(
            "calibrate_flash done: duty=%.5f width=%d us p99=%.1f "
            "converged=%s",
            duty,
            chosen,
            result_p99,
            converged,
        )
        return report

    @lt.action
    def calibrate_all_modes(self, target_p99: float = 230.0) -> Mapping[str, Any]:
        """Calibrate the flash in every legal streaming mode, then restore.

        Switches the camera through each of its streaming modes, calibrates
        the flash in each, persists a duty per mode, and re-arms in the
        original mode.
        """
        original_mode = self._current_mode()
        original_duty = self.mode_duties.get(original_mode, self.flash_duty)
        results = []
        try:
            for mode in self._cam.streaming_modes:
                if mode != self._current_mode():
                    self._cam.change_streaming_mode(mode=mode)
                    time.sleep(3.0)
                try:
                    results.append(
                        dict(self.calibrate_flash(target_p99=target_p99))
                    )
                except Exception as exc:
                    self.logger.warning(
                        "calibration failed in mode %s: %s", mode, exc
                    )
                    results.append({"mode": mode, "error": str(exc)})
        finally:
            if self._current_mode() != original_mode:
                self._cam.change_streaming_mode(mode=original_mode)
            original_result = next(
                (r for r in results if r.get("mode") == original_mode),
                None,
            )
            if original_result is None or not original_result.get(
                "converged"
            ):
                self.mode_duties = {
                    **self.mode_duties,
                    original_mode: original_duty,
                }
            self._arm_with_duty(
                self.mode_duties.get(original_mode, self.flash_duty)
            )
        self.logger.info(
            "calibrate_all_modes done: %d modes, %d converged",
            len(results),
            sum(1 for r in results if r.get("converged")),
        )
        return {"original_mode": original_mode, "results": results}

    @lt.action
    def apply_flash_for_current_mode(self) -> Mapping[str, Any]:
        """Apply ``flash_duty`` scaled to the measured frame period."""
        period = self._period_us()
        width = self._width_for_period(period)
        self._send(
            f"HWSYNC_STROBE STROBE=camera ENABLE=1 WIDTH={width / 1e6:.6f}"
        )
        self._send("SYNC_PULSE_STATUS STROBE=camera")
        self._last_width_us = width
        return {
            "period_us": period,
            "width_us": width,
            "flash_duty": self.flash_duty,
        }

    def _period_us(self) -> int:
        """Frame period in microseconds from the camera sync line.

        pigpio counts GPIO18 edges over 2 s; the sync line toggles once per
        frame in every streaming mode. Falls back to the firmware pulse
        counter, which is only valid when the flash outlasts the sync HIGH
        time.
        """
        for attempt in range(3):
            try:
                import pigpio
                pi = pigpio.pi()
                if not pi.connected:
                    raise RuntimeError("pigpiod unreachable")
                count = 0

                def _tick(_gpio, _level, _tick):
                    nonlocal count
                    count += 1

                cb = pi.callback(18, pigpio.RISING_EDGE, _tick)
                time.sleep(2.0)
                cb.cancel()
                pi.stop()
                fps = count / 2.0
                if 2.0 <= fps <= 500.0:
                    return round(1e6 / fps)
                self.logger.warning(
                    "sync-edge attempt %d gave %.1f fps; retrying",
                    attempt + 1,
                    fps,
                )
            except ImportError:
                break
            except Exception as exc:
                self.logger.warning(
                    "sync-edge attempt %d failed: %s", attempt + 1, exc
                )
            time.sleep(1.0)
        period = self._period_us_fw()
        if 1000 <= period <= 500000:
            return period
        raise RuntimeError(
            "frame rate measurement out of bounds; is the camera streaming?"
        )

    def _period_us_fw(self) -> int:
        """Frame period from the firmware pulse counter (fallback)."""
        self._send("SYNC_PULSE_STATUS STROBE=camera")
        first = self.state.get("count")
        if first is None:
            return DEFAULT_PERIOD_US
        t0 = time.monotonic()
        deadline = t0 + COUNTER_TIMEOUT_S
        while time.monotonic() < deadline:
            time.sleep(0.5)
            self._send("SYNC_PULSE_STATUS STROBE=camera")
            second = self.state.get("count")
            if second is None:
                continue
            if second > first:
                elapsed = time.monotonic() - t0
                if elapsed <= 0:
                    continue
                fps = (second - first) / elapsed
                if fps > 0:
                    return round(1e6 / fps)
        return DEFAULT_PERIOD_US

    def _width_for_period(self, period_us: int) -> int:
        width = int(self.flash_duty * period_us)
        return max(20, min(width, period_us - 1000))

    def _current_mode(self) -> str:
        try:
            return str(getattr(self._cam, "streaming_mode", "default"))
        except Exception:
            return "default"

    def _arm_with_duty(self, duty: float) -> None:
        period = self._period_us()
        self._pin_exposure(period)
        width = max(20, min(int(duty * period), period - 1000))
        self._send(
            f"HWSYNC_STROBE STROBE=camera ENABLE=1 WIDTH={width / 1e6:.6f}"
        )
        self._send("SYNC_PULSE_STATUS STROBE=camera")
        self._last_width_us = width

    def _pin_exposure(self, period_us: Optional[int] = None) -> None:
        """Disable AEC and pin exposure high so it spans most of the frame."""
        try:
            self._cam.set_camera_control("AeEnable", False)
        except AttributeError:
            self.logger.warning(
                "Camera does not support set_camera_control; exposure not pinned"
            )
            return
        if period_us is None:
            period_us = self._period_us()
        exposure = self.exposure_us if self.exposure_us > 0 else period_us - 1000
        self._cam.exposure_time = exposure
        self._cam.analogue_gain = self.analogue_gain

    def _send(self, script: str) -> None:
        response = self._client.post(
            "/printer/gcode/script", json={"script": script}
        )
        response.raise_for_status()
        data = response.json()
        if data.get("error"):
            raise IOError(f"Moonraker rejected script {script!r}: {data['error']}")
