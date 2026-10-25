import threading
import time
from typing import Optional

import labthings_fastapi as lt

from .camera import BaseCamera
from .stage import BaseStage

_TARGET_US = 45475.0
_HOLD_US = 45474.0
_PARK_ROW = 6.0


class FreezeMotionThing(lt.Thing):
    """PWM-carrier strobe with a timestamp phase-locked loop.

    pigpio free-runs a fixed-width LED pulse train on the Pi header
    (BCM 24, physical pin 18) so illumination is uniform and immune to
    CPU load. A correction loop reads the camera's sensor timestamps
    from the pre-callback, measures the true frame period, and trims
    the PWM period in microsecond steps so the carrier stays locked to
    the sensor and the readout straddle stays parked at the top of the
    frame. Period swaps happen at wave boundaries, so no glitch frames
    are produced.
    """

    _cam: BaseCamera = lt.thing_slot()
    _stage: BaseStage = lt.thing_slot()

    _lock = threading.Lock()
    _pi = None

    _firing = False
    _pulse_thread = None
    _viewing = True

    _tau = 0.0003
    _period_us = _TARGET_US
    _ts_hist = []
    _anchor_ts = None
    _anchor_mono = None
    _band_row = None
    _band_ts = None
    _meds = []
    _med_ref = None
    _hist = []
    _last_corr = 0.0
    _locked = False
    _dbg = 0
    _saved_ae = None
    _pin_us = None
    _phase_us = 0.0
    _dither_hi = False
    _dither_t = 0.0
    _parking_until = None
    _rev_burst_until = None
    _prev_none = False
    _upburst_until = None
    _hi_frac = 0.5
    _row_hist = []

    @lt.action
    def pin_period(self, us: Optional[float] = None) -> None:
        """Pin the PWM period (us); None releases the pin."""
        self._pin_us = us
        if us is not None and self._firing:
            pi = self._pi
            if pi is not None:
                self._apply(pi, float(us))

    @lt.action
    def strobe_on(self, tau: Optional[float] = 0.0003) -> None:
        cam = self._cam
        self._saved_ae = (
            int(cam.exposure_time),
            float(cam.analogue_gain),
        )
        cam.exposure_time = 45100
        cam.analogue_gain = 1.0
        try:
            cam.set_camera_control("AeEnable", False)
        except Exception:
            pass
        self._tau = float(tau)
        self._gpio_wave_start()
        self._firing = True
        with self._lock:
            self._cam.set_band_hook(self._on_band)
        t = self._pulse_thread
        if t is None or not t.is_alive():
            self._pulse_thread = threading.Thread(
                target=self._pll_loop, daemon=True
            )
            self._pulse_thread.start()

    @lt.action
    def strobe_off(self) -> None:
        self._firing = False
        t = self._pulse_thread
        if t is not None:
            t.join(timeout=3.0)
        with self._lock:
            self._cam.set_band_hook(None)
        self._gpio_wave_stop()
        saved = self._saved_ae
        cam = self._cam
        if saved is not None:
            cam.exposure_time = int(saved[0])
            cam.analogue_gain = float(saved[1])
            self._saved_ae = None
        try:
            cam.set_camera_control("AeEnable", True)
        except Exception:
            pass

    @lt.action
    def set_viewing(self, on: bool) -> None:
        self._viewing = bool(on)
        pi = self._pi
        if pi is not None and not self._firing:
            pi.write(24, 1 if self._viewing else 0)

    @lt.action
    def start_observing(self) -> None:
        with self._lock:
            self._cam.set_band_hook(self._on_band)

    @lt.action
    def stop_observing(self) -> None:
        with self._lock:
            self._cam.set_band_hook(None)

    @lt.action
    def band_samples(self, seconds: float = 12.0) -> list:
        with self._lock:
            self._hist = []
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            time.sleep(0.5)
        with self._lock:
            return [[t, r] for (t, r, a) in self._hist]

    @lt.property
    def firing(self) -> bool:
        return self._firing

    @lt.property
    def viewing(self) -> bool:
        return self._viewing

    @lt.property
    def tau_ms(self) -> float:
        return self._tau * 1000.0

    @lt.property
    def period_us(self) -> float:
        return self._period_us

    @lt.property
    def measured_period_ms(self):
        with self._lock:
            h = list(self._ts_hist)
        if len(h) < 5:
            return None
        ds = sorted(b - a for a, b in zip(h[-17:-1], h[-16:]))
        return ds[len(ds) // 2] * 1000.0

    @lt.property
    def last_band_row(self):
        return self._band_row

    @lt.property
    def last_correction_us(self) -> float:
        return self._last_corr

    @lt.property
    def hi_frac(self) -> float:
        return self._hi_frac

    @lt.property
    def locked(self) -> bool:
        return self._locked

    def _pigpio(self):
        if self._pi is None:
            import pigpio

            pi = pigpio.pi()
            if not pi.connected:
                raise RuntimeError("cannot reach pigpiod on localhost:8888")
            self._pi = pi
        return self._pi

    def _make_wave(self, pi, period_us: float, phase_us: float = 0.0):
        import pigpio

        tau_us = max(5, int(round(self._tau * 1e6)))
        per = max(1, int(round(period_us)))
        phase = max(0, int(round(phase_us))) % per
        rest = max(1, per - phase - tau_us)
        pi.wave_add_generic(
            [
                pigpio.pulse(0, 1 << 24, phase),
                pigpio.pulse(1 << 24, 0, tau_us),
                pigpio.pulse(0, 1 << 24, rest),
            ]
        )
        return pi.wave_create()

    def _gpio_wave_start(self):
        import pigpio

        pi = self._pigpio()
        pi.set_mode(24, pigpio.OUTPUT)
        wid = self._make_wave(pi, self._period_us, self._phase_us)
        pi.wave_send_using_mode(wid, pigpio.WAVE_MODE_REPEAT_SYNC)
        self._waves = [wid]

    def _gpio_wave_stop(self):
        pi = self._pi
        if pi is None:
            return
        try:
            pi.wave_tx_stop()
            for w in list(getattr(self, "_waves", [])):
                try:
                    pi.wave_delete(w)
                except Exception:
                    pass
            self._waves = []
            pi.write(24, 1 if self._viewing else 0)
        except Exception:
            pass

    def _retune(self, pi, new_period_us: float, phase_us: Optional[float] = None) -> None:
        import pigpio

        if phase_us is not None:
            self._phase_us = float(phase_us)
        wid = self._make_wave(pi, new_period_us, self._phase_us)
        pi.wave_send_using_mode(wid, pigpio.WAVE_MODE_REPEAT_SYNC)
        self._period_us = new_period_us
        time.sleep(0.05)
        for w in self._waves:
            try:
                pi.wave_delete(w)
            except Exception:
                pass
        self._waves = [wid]

    @lt.action
    def set_phase(self, phase_us: float) -> None:
        """Set the pulse phase within the frame period (microseconds)."""
        pi = self._pigpio()
        self._apply(pi, self._period_us, float(phase_us))

    @lt.property
    def phase_us(self) -> float:
        return self._phase_us

    def _on_band(self, ts, row, med=None):
        now = time.monotonic()
        self._dbg += 1
        if self._dbg == 1 or self._dbg % 200 == 0:
            try:
                with open("/tmp/fm_dbg.log", "a") as f:
                    f.write("cb %d ts=%s row=%s\n" % (self._dbg, str(ts)[:14], str(row)))
            except Exception:
                pass
        if ts is not None:
            with self._lock:
                self._ts_hist.append(ts)
                self._ts_hist = self._ts_hist[-40:]
                if self._anchor_ts is None:
                    self._anchor_ts = ts
                    self._anchor_mono = now
                else:
                    pred = self._anchor_mono + (ts - self._anchor_ts)
                    err = now - pred
                    if abs(err) < 0.005:
                        self._anchor_mono = pred + err * 0.15
                    else:
                        self._anchor_ts = ts
                        self._anchor_mono = now
        if med is not None:
            if len(self._meds) < 10:
                self._meds.append(med)
                if len(self._meds) == 10:
                    self._med_ref = sorted(self._meds)[5]
            elif self._med_ref is not None and abs(med - self._med_ref) > 20:
                return
        self._band_ts = ts
        self._band_row = row
        if row is not None and ts is not None:
            with self._lock:
                self._hist.append((ts, row, now))
                cutoff = now - 60.0
                self._hist = [h for h in self._hist if h[2] >= cutoff][-300:]

    def _frame_period(self):
        with self._lock:
            h = list(self._ts_hist)
        if len(h) < 10:
            return None
        ds = sorted(b - a for a, b in zip(h[-17:-1], h[-16:]))
        return ds[len(ds) // 2]

    def _band_drift(self):
        with self._lock:
            h = [(t, r, a) for (t, r, a) in self._hist]
        now = time.monotonic()
        s = [(t, r) for t, r, a in h if now - a <= 8.0 and r is not None]
        if len(s) < 8 or s[-1][0] - s[0][0] < 3.0:
            return None, None
        row_med = sorted(r for _, r in s)[len(s) // 2]
        v = (s[-1][1] - s[0][1]) / (s[-1][0] - s[0][0])
        return row_med, v

    def _pll_loop(self):
        time.sleep(2.0)
        warm = 0
        hidden_since = None
        ae_tick = 0.0
        while self._firing:
            pi = self._pi
            if pi is None:
                time.sleep(1.0)
                continue
            if time.monotonic() - ae_tick > 3.0:
                ae_tick = time.monotonic()
                try:
                    self._cam.set_camera_control("AeEnable", False)
                    self._cam.exposure_time = 45100
                    self._cam.analogue_gain = 1.0
                except Exception:
                    pass
            warm += 1
            if warm < 8:
                time.sleep(1.5)
                continue
            if self._pin_us is not None:
                self._apply(pi, float(self._pin_us))
                self._locked = True
                time.sleep(1.0)
                continue
            row = self._band_row
            now = time.monotonic()
            if row is not None:
                self._row_hist = (self._row_hist + [(now, row)])[-40:]
            if row is not None and row < 18.0:
                self._locked = False
                self._apply(pi, 45400.0)
                self._upburst_until = time.monotonic() + 1.5
                time.sleep(0.15)
                continue
            if row is None:
                if self._upburst_until is not None:
                    if now < self._upburst_until:
                        self._apply(pi, 45400.0)
                        time.sleep(0.12)
                        continue
                    self._upburst_until = None
                    self._apply(pi, 45474.0)
                    self._locked = True
                    self._prev_none = True
                    time.sleep(0.4)
                    continue
                if now - self._dither_t >= 2.0 * self._hi_frac:
                    self._dither_hi = not self._dither_hi
                    self._dither_t = now
                    self._hi_frac = max(0.2, min(0.8, self._hi_frac))
                    if len(self._row_hist) >= 8:
                        (t0, r0), (t1, r1) = self._row_hist[0], self._row_hist[-1]
                        if t1 > t0:
                            v = (r1 - r0) / (t1 - t0)
                            self._hi_frac = max(0.2, min(0.8, 0.5 - v * 6.0))
                            self._row_hist = self._row_hist[-2:]
                target = 45475.0 if self._dither_hi else 45474.0
                self._apply(pi, target)
                self._locked = True
                self._prev_none = True
                time.sleep(0.4)
                continue
            if self._prev_none:
                self._prev_none = False
            if row > 175.0:
                self._locked = False
                self._apply(pi, 45535.0)
                time.sleep(0.15)
                continue
            self._locked = False
            self._apply(pi, 45400.0)
            time.sleep(1.0)

    def _apply(self, pi, period_us, phase_us=None):
        per_changed = abs(self._period_us - period_us) >= 0.5
        ph_changed = phase_us is not None and abs(self._phase_us - phase_us) >= 0.5
        if not per_changed and not ph_changed:
            return
        if period_us != self._period_us:
            self._period_us = period_us
        try:
            self._retune(pi, self._period_us, phase_us)
        except Exception:
            pass
