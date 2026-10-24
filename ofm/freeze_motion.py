import threading
import time
from typing import Optional

import labthings_fastapi as lt

from .camera import BaseCamera
from .stage import BaseStage


class FreezeMotionThing(lt.Thing):
    """Strobed freeze-motion capture with LED phase steering.

    The LED pulses once per frame period, free-run on the MCU clock. The
    loop first matches the LED period to the camera's true frame period
    (nulling band drift), then catches the drifting straddle band as it
    sweeps past the parking row and snaps back to the matched period.
    Each cycle-time reprogram emits one over-bright glitch frame, so
    reprograms are kept rare and self-terminating.
    """

    _cam: BaseCamera = lt.thing_slot()
    _stage: BaseStage = lt.thing_slot()

    _lock = threading.Lock()
    _steering = False
    _thread = None
    _strobe_on = False
    _band_ts = None
    _band_row = None
    _band_arrival = 0.0
    _med_ref = None
    _meds = []
    _hist = []
    _last_reprog = 0.0
    _last_err = None
    _last_v = None
    _matched = 0.0455404
    _cycle = 0.0455404
    _tau = 0.0005
    _target = 12.0
    _cmin = 0.0420
    _cmax = 0.0490
    _slope = 160000.0
    _parked = False
    _reprogs = 0

    @lt.action
    def strobe_on(self, tau: Optional[float] = 0.0012, cycle_time: Optional[float] = None) -> None:
        if not hasattr(self._stage, "strobe_set"):
            raise RuntimeError("stage has no strobe_set capability")
        self._tau = tau
        if cycle_time is not None:
            self._cycle = cycle_time
            self._matched = cycle_time
        self._strobe_on = True
        self._reprogram()

    @lt.action
    def strobe_off(self) -> None:
        self.stop_steering()
        self._strobe_on = False
        self._stage.strobe_set(0.0)

    @lt.action
    def start_steering(self, target_row: float = 13.0) -> None:
        with self._lock:
            if self._steering:
                return
            self._target = target_row
            self._hist = []
            self._parked = False
            self._cam.set_band_hook(self._on_band)
            self._steering = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    @lt.action
    def stop_steering(self) -> None:
        with self._lock:
            self._steering = False
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)
            self._thread = None
        self._cam.set_band_hook(None)
        self._hist = []

    @lt.property
    def steering(self) -> bool:
        return self._steering

    @lt.property
    def steering_cycle_time(self) -> float:
        return self._cycle

    @lt.property
    def matched_cycle_time(self) -> float:
        return self._matched

    @lt.property
    def last_band_row(self):
        return self._band_row

    @lt.property
    def steering_velocity(self):
        return self._last_v

    @lt.property
    def parked(self) -> bool:
        return self._parked

    @lt.property
    def refreshing(self) -> bool:
        return self._steering and not self._parked

    @lt.property
    def reprogram_count(self) -> int:
        return self._reprogs

    @lt.action
    def start_observing(self) -> None:
        self._cam.set_band_hook(self._on_band)

    @lt.action
    def stop_observing(self) -> None:
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

    def _clamp(self, c):
        return min(self._cmax, max(self._cmin, c))

    def _reprogram(self):
        try:
            self._stage.strobe_set(1.0 - self._tau / self._cycle, self._cycle)
            self._last_reprog = time.monotonic()
            self._reprogs += 1
            self._hist = []
            self._last_err = None
        except Exception as e:
            self._last_err = repr(e)
            time.sleep(2.0)

    def _on_band(self, ts, row, med=None):
        now = time.monotonic()
        if med is not None:
            if len(self._meds) < 10:
                self._meds.append(med)
                if len(self._meds) == 10:
                    self._med_ref = sorted(self._meds)[5]
            elif self._med_ref is not None and abs(med - self._med_ref) > 20:
                return
        if now - self._last_reprog < 0.8:
            return
        self._band_ts = ts
        self._band_row = row
        self._band_arrival = now
        if row is not None and ts is not None:
            self._hist.append((ts, row, now))
            cutoff = now - 45.0
            self._hist = [h for h in self._hist if h[2] >= cutoff][-300:]

    def _velocity(self, window=6.0):
        now = time.monotonic()
        s = [(t, r) for t, r, a in self._hist if now - a <= window]
        if len(s) < 5 or s[-1][0] - s[0][0] < 2.5:
            return None
        return (s[-1][1] - s[0][1]) / (s[-1][0] - s[0][0])

    def _loop(self):
        time.sleep(2.0)
        while self._steering:
            if time.monotonic() - self._band_arrival > 4.0:
                time.sleep(1.5)
                continue
            v = self._velocity()
            if v is None:
                time.sleep(1.0)
                continue
            self._last_v = v
            if abs(v) > 0.6:
                self._matched = self._clamp(self._matched - v / self._slope)
                self._cycle = self._matched
                self._reprogram()
                self._parked = False
                time.sleep(7.0)
                continue
            self._parked = True
            time.sleep(5.0)
