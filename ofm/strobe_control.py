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
        self._width_by_mode: dict[str, int] = {}
        self._period_by_mode: dict[str, int] = {}
        self._continuous = False
        super().__init__(thing_server_interface)

    enabled: bool = lt.setting(default=True)
    flash_duty: float = lt.setting(default=0.011)
    mode_duties: dict = lt.setting(default_factory=dict)
    mode_periods: dict = lt.setting(default_factory=dict)
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
            boot_mode = self._current_mode()
            self._period_by_mode[boot_mode] = period
            self._width_by_mode[boot_mode] = width
            self.mode_periods = {**self.mode_periods, boot_mode: period}
            self._send(
                f"HWSYNC_STROBE STROBE=camera ENABLE=1 WIDTH={width / 1e6:.6f}"
            )
            self._send("SYNC_PULSE_STATUS STROBE=camera")
            self._last_width_us = width
            self.logger.info(
                "Strobe armed at %d us (period %d us, duty %.5f, mode %s)",
                width,
                period,
                duty,
                boot_mode,
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
            # full_resolution is an alias of default in the HQ camera class;
            # timings are identical, key kept for that mode name.
            "full_resolution": {
                "fps": 22.0,
                "blanking_us": 225,
                "frame_period_us": 45475,
            },
            "crop990": {
                "fps": 140.0,
                "blanking_us": 523,
                "frame_period_us": 7143,
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
            mode = self._current_mode()
            period = self._period_by_mode.get(mode, DEFAULT_PERIOD_US)
            duty = self.mode_duties.get(mode, self.flash_duty)
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
        period = self._period_by_mode.get(self._current_mode(), DEFAULT_PERIOD_US)
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
        mode = self._current_mode()
        self.mode_duties = {**self.mode_duties, mode: microseconds / period}
        self._width_by_mode[mode] = microseconds
        self._last_width_us = microseconds

    def capture_operating_point(self) -> None:
        """Store the camera's calibrated gain as the pinned operating
        point used during the continuous-light phase."""
        self.exposure_us = int(self._cam.exposure_time)
        self.analogue_gain = float(self._cam.analogue_gain)
        self.logger.info(
            "operating point captured: exposure %d us, gain %.2f",
            self.exposure_us,
            self.analogue_gain,
        )

    def enter_continuous_light(self) -> None:
        """Light the LED continuously for measurements that need
        continuous illumination (WIDTH=1 s holds it HIGH across
        many frames). Suppresses mode-change re-arms until
        ``exit_continuous_light``."""
        self._continuous = True
        self._send("HWSYNC_STROBE STROBE=camera ENABLE=1 WIDTH=1")
        self._send("SYNC_PULSE_STATUS STROBE=camera")

    def exit_continuous_light(self) -> None:
        """Re-arm the calibrated flash width and re-pin the exposure.

        Entering the strobe phase drops the continuous-light exposure:
        zeroing ``exposure_us`` makes every re-arm derive the full-frame
        fps-safe pin for the active mode."""
        self._continuous = False
        self.exposure_us = 0
        self.strobe_on()
        self._pin_exposure()

    @lt.action
    def calibrate_flash(
            self,
            target_p99: float = 245.0,
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
        self._pin_exposure(None)
        period = self._period_us()
        sensor_period: Optional[int]
        try:
            sensor_period = self._cam.sensor_frame_period_us()
        except Exception as exc:
            self.logger.warning("sensor cadence cross-check failed: %s", exc)
            sensor_period = None
        if sensor_period is not None:
            if abs(sensor_period - period) > max(500, period // 8):
                self.logger.warning(
                    "sync line says %d us but sensor timestamps say %d us; "
                    "trusting the sensor",
                    period,
                    sensor_period,
                )
                period = sensor_period
        self.mode_periods = {**self.mode_periods, self._current_mode(): period}
        ceiling = self.exposure_ceiling_us()
        if ceiling is not None and period > ceiling + 2000:
            raise RuntimeError(
                "measured period %d us exceeds the fps-safe ceiling %d us; "
                "refusing to calibrate a stretched frame" % (period, ceiling)
            )
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
        mode = self._current_mode()
        self.mode_duties = {**self.mode_duties, mode: duty}
        self._width_by_mode[mode] = chosen
        self._period_by_mode[mode] = period
        if converged:
            self.flash_duty = duty
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
    def calibrate_all_modes(self, target_p99: float = 245.0) -> Mapping[str, Any]:
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
                bounds = (
                    {
                        "min_width_us": 50,
                        "max_width_us": 1000,
                        "target_p99": 225.0,
                    }
                    if mode == "crop990"
                    else {"target_p99": target_p99}
                )
                try:
                    results.append(
                        dict(self.calibrate_flash(**bounds))
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
        mode = self._current_mode()
        self._arm_with_duty(
            self.mode_duties.get(mode, self.flash_duty), mode=mode
        )
        return {
            "period_us": self._period_by_mode.get(mode),
            "width_us": self._width_by_mode.get(mode),
            "flash_duty": self.mode_duties.get(mode, self.flash_duty),
        }

    def _period_us(self) -> int:
        """Frame period in microseconds from the camera sync line.

        The sync line carries two rising edges per frame: consecutive
        gaps alternate between two populations (one anchored on the
        exposure, one spanning it plus blanking twice), so any
        single-gap statistic - median, mode - reports one population
        and not the frame rate. The mean over all gaps is the true
        frame period. Falls back to the firmware pulse counter, which
        has proven unreliable and is a last resort only."""
        for attempt in range(3):
            try:
                import pigpio
                pi = pigpio.pi()
                if not pi.connected:
                    raise RuntimeError("pigpiod unreachable")
                deltas: list[int] = []
                last: list = [None]

                def _tick(_gpio, _level, tick):
                    if last[0] is not None:
                        deltas.append(tick - last[0])
                    last[0] = tick

                cb = pi.callback(18, pigpio.RISING_EDGE, _tick)
                time.sleep(2.0)
                cb.cancel()
                pi.stop()
                if len(deltas) < 20:
                    raise RuntimeError(
                        "only %d sync edges in 2s; is the camera streaming?"
                        % (len(deltas) + 1)
                    )
                period = round(sum(deltas) / len(deltas))
                if 1000 <= period <= 200000:
                    return period
                self.logger.warning(
                    "sync-edge attempt %d gave period %d us; retrying",
                    attempt + 1,
                    period,
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

    def on_mode_changed(self) -> None:
        """Re-arm with this mode's calibrated duty after a stream restart.

        Called by the camera thing whenever it (re)configures the sensor,
        so the flash width always matches the mode that is streaming.
        """
        if self._continuous:
            return
        self._rearm_logged()

    def pre_mode_change(self, mode: str) -> None:
        """Arm the new mode's width BEFORE the sensor reconfigures.

        Uses only cached calibration (never measures here - that would
        stall the stream restart), so the very first frame of the new mode
        already has the correct flash.
        """
        if self._continuous:
            return
        if not self.enabled:
            return
        if self._width_by_mode.get(mode) is None:
            return
        try:
            self._arm_with_duty(
                self.mode_duties.get(mode, self.flash_duty), mode=mode
            )
        except Exception as exc:
            self._last_error = str(exc)
            self.logger.warning("pre-arm for %s failed: %s", mode, exc)

    def _rearm_logged(self) -> None:
        if not self.enabled:
            return
        try:
            self._arm_with_duty(
                self.mode_duties.get(self._current_mode(), self.flash_duty)
            )
        except Exception as exc:
            self._last_error = str(exc)
            self.logger.warning("re-arm after mode change failed: %s", exc)

    def _arm_with_duty(self, duty: float, mode: Optional[str] = None) -> None:
        """Arm with the calibrated width for ``mode``.

        Instant when the mode's period is already known (persisted from a
        previous calibration or boot); on a true first encounter it measures
        once, persists the period, then arms.
        """
        mode = mode or self._current_mode()
        width = self._width_by_mode.get(mode)
        if width is None:
            period = self.mode_periods.get(mode)
            if period is None:
                period = self._period_us()
                self.mode_periods = {**self.mode_periods, mode: period}
            width = max(20, min(int(duty * period), period - 1000))
            self._width_by_mode[mode] = width
        self._pin_exposure(self.mode_periods.get(mode))
        self._send(
            f"HWSYNC_STROBE STROBE=camera ENABLE=1 WIDTH={width / 1e6:.6f}"
        )
        self._send("SYNC_PULSE_STATUS STROBE=camera")
        self._last_width_us = width

    def exposure_ceiling_us(self, mode: Optional[str] = None) -> Optional[int]:
        """Highest exposure that keeps the mode at full frame rate.

        The camera's own exposure-range maximum is the authority: it is
        derived from the sensor's real frame geometry (frame length
        minus integration offset), so it already encodes the true
        blanking. Falls back to the sync spec only when the camera
        cannot be asked."""
        try:
            ceiling = int(self._cam.exposure_range_max())
            if ceiling > 0:
                return ceiling
        except Exception:
            pass
        spec = self.sync_calibration.get(mode or self._current_mode())
        if not spec:
            return None
        return int(spec["frame_period_us"] - spec["blanking_us"] - 100)

    def _pin_exposure(self, period_us: Optional[int] = None) -> None:
        """Disable AEC and pin the shutter to the full frame period.

        Every row then integrates for the whole period, so the flash is
        captured wherever it falls inside the frame. The raw control
        write accepts the full-period value (as with the 45 ms pin in
        the 22 fps mode) even where the reported control range would
        clamp it; frame rate is set by the mode's framing limits, not
        by ExposureTime."""
        try:
            self._cam.set_camera_control("AeEnable", False)
        except AttributeError:
            self.logger.warning(
                "Camera does not support set_camera_control; exposure not pinned"
            )
            return
        if period_us is None:
            period_us = self._period_us()
        exposure = (
            self.exposure_us if self.exposure_us > 0 else int(period_us)
        )
        try:
            self._cam.set_probe_controls_raw(self.analogue_gain, exposure)
        except Exception as exc:
            self.logger.warning(
                "raw exposure pin failed (%s); falling back to property",
                exc,
            )
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
