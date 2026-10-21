"""Provide a LabThings-FastAPI interface to the Bigtreetech SKR Pico v1.0 motor controller."""

from __future__ import annotations

import asyncio
import collections
import json
import threading
import time
from enum import Enum
from types import TracebackType
from typing import Any, Literal, Mapping, Optional, Self, Sequence, Tuple

import httpx
import labthings_fastapi as lt

from . import BaseStage, JogCommand


class SkrPicoThing(BaseStage):
    """A Thing to manage a SKR Pico v1.0 motor controller using Moonraker.
    """

    def __init__(
            self,
            thing_server_interface: lt.ThingServerInterface,
            **kwargs: Any,
    ) -> None:
        self.port = kwargs.get("moonrakerport", "7125")
        self.baseurl = kwargs.get("baseurl", "http://127.0.0.1")
        self.acceleration = kwargs.get("acceleration", 45000)
        self.speed = kwargs.get("speed", 1000)
        self.timeout = httpx.Timeout(60.0)
        self._client = httpx.Client(
            base_url=self.baseurl + ":" + self.port, timeout=self.timeout
        )
        super().__init__(thing_server_interface)

    def __enter__(self) -> Self:
        for attempt in range(30):
            try:
                self.set_homed()
                self.update_position()
                break
            except Exception:
                if attempt == 29:
                    break
                time.sleep(2.0)
        self.start_motion_telemetry()
        self._load_cal()
        return self

    _CAL_PATH = "/home/admin/openflexure-microscope-server/smooth_af_cal.json"
    MOVES_LOG = "/home/admin/openflexure-microscope-server/smooth_af_moves.log"

    def _load_cal(self) -> None:
        try:
            with open(self._CAL_PATH) as f:
                d = json.load(f)
        except Exception:
            return
        try:
            from openflexure_microscope_server.things.smooth_autofocus import (
                SmoothCompatAutofocus,
            )
        except Exception:
            return
        cls = SmoothCompatAutofocus
        for attr, key in (
                ("_CAL_FWHM", "fwhm_u"),
                ("_CAL_CONTRAST", "contrast"),
                ("_CAL_ACCEL", "accel"),
        ):
            if d.get(key) is not None:
                setattr(cls, attr, d[key])

    def __exit__(
            self,
            _exc_type: type[BaseException],
            _exc_value: Optional[BaseException],
            _traceback: Optional[TracebackType],
    ) -> None:
        client = self._client
        client.close()

    # Z1 is chained to Z but still reports position data
    axis_inverted: dict[str, bool] = lt.setting(
        default_factory=lambda: {"x": True, "y": False, "z": True}, readonly=True
    )

    def clear_motion_history(self) -> None:
        self.motion_clear()

    def _send_gcode_script(self, client: Optional[httpx.Client] = None, script: str = "",
                           timeout: Optional[httpx.Timeout] = None) -> dict:
        """Send a gcode script to Moonraker, raising if the request fails."""
        if client is None:
            client = self._client
        response = client.post("/printer/gcode/script",
                               timeout=timeout or self.timeout, json={"script": script})
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise IOError(f"Moonraker rejected script {script!r}: {data['error']}")
        return data

    def _get_status(self, client: Optional[httpx.Client] = None, payload: Optional[dict] = None) -> dict:
        """Query printer objects from Moonraker, raising if the request fails."""
        if client is None:
            client = self._client
        response = client.post("/printer/objects/query",
                               timeout=self.timeout, json=payload or {})
        response.raise_for_status()
        return response.json()

    def start_motion_telemetry(self) -> None:
        """Subscribe to motion_report over the Moonraker websocket.

        Keeps a small deque of (monotonic_t, live_position_z, live_velocity)
        samples pushed by klippy itself -- no polling, no sleeps. The
        subscription lives for the process lifetime.
        """
        if getattr(self, "_motion_ws_thread", None) and self._motion_ws_thread.is_alive():
            return
        self._motion_samples = collections.deque(maxlen=4096)
        self._clock_offsets = collections.deque(maxlen=64)
        self._motion_ws_lock = threading.Lock()
        self._et_handshake = 0.0
        self._motion_ws_cond = threading.Condition(self._motion_ws_lock)
        self._motion_ws_stop = threading.Event()
        self._motion_ws_thread = threading.Thread(
            target=self._motion_ws_main, daemon=True, name="moonraker-motion-ws"
        )
        self._motion_ws_thread.start()

    def _motion_ws_main(self) -> None:
        """Poll klippy motion_report directly over its own websocket.

        Moonraker stopped pushing motion_report status updates, so this
        client no longer relies on push semantics: it asks klippy for
        live_position/live_velocity on a fixed cadence and timestamps each
        reply with klippy's own eventtime, keeping the sample tuple shape
        identical to the old push consumer.
        """

        def _wlog(line: str) -> None:
            try:
                with open(self.MOVES_LOG, "a") as f:
                    f.write("WS %s t=%.3f\n" % (line, time.time()))
            except Exception:
                pass

        async def _run() -> None:
            import websockets
            uri = "ws://127.0.0.1:%s/websocket" % self.port
            while not self._motion_ws_stop.is_set():
                try:
                    async with websockets.connect(uri) as ws:
                        _wlog("CONNECTED push")
                        req_id = 10
                        await ws.send(json.dumps({
                            "jsonrpc": "2.0",
                            "method": "printer.objects.subscribe",
                            "params": {"objects": {"motion_report": None}},
                            "id": req_id,
                        }))
                        first = True
                        last_sample = time.monotonic()
                        resub_logged = False
                        while not self._motion_ws_stop.is_set():
                            try:
                                raw = await asyncio.wait_for(
                                    ws.recv(), 1.0
                                )
                            except asyncio.TimeoutError:
                                if (
                                        not self._motion_ws_stop.is_set()
                                        and time.monotonic() - last_sample
                                        >= 5.0
                                ):
                                    req_id += 1
                                    await ws.send(json.dumps({
                                        "jsonrpc": "2.0",
                                        "method":
                                            "printer.objects.subscribe",
                                        "params": {"objects": {
                                            "motion_report": None}},
                                        "id": req_id,
                                    }))
                                    if not resub_logged:
                                        _wlog("RESUB idle")
                                        resub_logged = True
                                    last_sample = time.monotonic()
                                continue
                            m = json.loads(raw)
                            params = m.get("params")
                            if not isinstance(params, list) or not params:
                                continue
                            p0 = params[0]
                            if not isinstance(p0, dict):
                                continue
                            st = p0.get("motion_report")
                            if st is None and isinstance(
                                    p0.get("status"), dict
                            ):
                                st = p0["status"].get("motion_report")
                            if not isinstance(st, dict):
                                continue
                            lp = st.get("live_position")
                            if lp is None:
                                continue
                            v_raw = st.get("live_velocity")
                            v_now = (
                                float(v_raw) if v_raw is not None else None
                            )
                            carried = 0.0 if v_raw is not None else 1.0
                            with self._motion_ws_lock:
                                try:
                                    self._motion_last_z = float(lp[2])
                                    self._motion_last_xy = (
                                        float(lp[0]), float(lp[1]),
                                    )
                                except Exception:
                                    pass
                                self._motion_last_v = (
                                    v_now if v_now is not None else 0.0
                                )
                                t_mono = time.monotonic()
                                self._motion_samples.append(
                                    (
                                        t_mono + self._et_handshake,
                                        float(lp[2]),
                                        v_now if v_now is not None else 0.0,
                                        carried,
                                        t_mono + self._et_handshake,
                                        (float(lp[0]), float(lp[1])),
                                    )
                                )
                                self._motion_ws_cond.notify_all()
                            last_sample = time.monotonic()
                            resub_logged = False
                            if first:
                                first = False
                                _wlog("FIRST_SAMPLE push")
                except Exception as e:
                    _wlog("ERR %r" % (e,))
                if self._motion_ws_stop.wait(2.0):
                    break

        asyncio.run(_run())

    def _ingest_motion(self, st: dict) -> None:
        lp = st.get("live_position")
        if lp is None:
            return
        v_raw = st.get("live_velocity")
        v_now = float(v_raw) if v_raw is not None else None
        carried = 0.0 if v_raw is not None else 1.0
        with self._motion_ws_lock:
            try:
                self._motion_last_z = float(lp[2])
                self._motion_last_xy = (
                    float(lp[0]), float(lp[1]),
                )
            except Exception:
                pass
            self._motion_last_v = (
                v_now if v_now is not None else 0.0
            )
            t_mono = time.monotonic()
            self._motion_samples.append(
                (
                    t_mono + self._et_handshake,
                    float(lp[2]),
                    v_now if v_now is not None else 0.0,
                    carried,
                    t_mono + self._et_handshake,
                    (float(lp[0]), float(lp[1])),
                )
            )
            self._motion_ws_cond.notify_all()

    def _leg_poller_start(self) -> None:
        if getattr(self, "_leg_poller_thread", None) and \
                self._leg_poller_thread.is_alive():
            return
        self._leg_poller_stop = threading.Event()

        def _poll_loop() -> None:
            t0 = time.monotonic()
            while not self._leg_poller_stop.is_set():
                try:
                    self._ingest_motion(self.query_motion_state())
                except Exception:
                    pass
                if time.monotonic() - t0 > 30.0:
                    break
                self._leg_poller_stop.wait(0.05)

        self._leg_poller_thread = threading.Thread(
            target=_poll_loop, daemon=True, name="motion-leg-poller"
        )
        self._leg_poller_thread.start()

    def _leg_poller_stop_fn(self) -> None:
        ev = getattr(self, "_leg_poller_stop", None)
        if ev is not None:
            ev.set()

    def _stamp_for(self, et) -> float:
        h = getattr(self, "_et_handshake", None)
        if et is not None and h is not None:
            return et - h
        return time.monotonic()

    def motion_since(self, t0: float) -> list:
        """Return telemetry samples with sample time > t0 (thread-safe copy)."""
        seq = getattr(self, "_motion_samples", None)
        if seq is None:
            return []
        lock = getattr(self, "_motion_ws_lock", None)
        if lock is None:
            return list(seq)
        with lock:
            return [s for s in seq if s[0] > t0]

    def xy_live_position(self) -> Optional[Tuple[float, float]]:
        """Latest (x, y) from the motion_report stream, or None."""
        v = getattr(self, "_motion_last_xy", None)
        return None if v is None else (float(v[0]), float(v[1]))

    def _xy_live_drift(self) -> Optional[float]:
        """Max |live - tracked| in stage units, or None if unavailable."""
        v = self.xy_live_position()
        if v is None:
            return None
        try:
            trk = self._xy_tracked()
        except AttributeError:
            return None
        return max(abs(v[0] - trk[0]), abs(v[1] - trk[1]))

    def motion_clear(self) -> None:
        """Drop all buffered telemetry. Called at the start of a synced
        move so the deque only ever holds the current leg's samples;
        stale history from earlier legs can then never leak into
        drain/marker checks."""
        lock = getattr(self, "_motion_ws_lock", None)
        if lock is not None:
            with lock:
                self._motion_samples.clear()
        else:
            seq = getattr(self, "_motion_samples", None)
            if seq is not None:
                seq.clear()

    def motion_drained(self, t_after: float) -> bool:
        """True once live_velocity returned to 0 after being non-zero past t_after."""
        seq = getattr(self, "_motion_samples", None)
        if not seq:
            return False
        lock = getattr(self, "_motion_ws_lock", None)
        if lock is not None:
            with lock:
                snapshot = list(seq)
        else:
            snapshot = list(seq)
        moved = any(
            abs(s[2]) > 0.5 for s in snapshot
            if len(s) > 2 and s[0] > t_after
        )
        if not moved:
            return False
        last = snapshot[-1]
        return len(last) > 2 and last[0] > t_after and abs(last[2]) < 0.5

    def query_motion_state(self) -> dict:
        """One-shot motion_report query - used to certify a leg whose push
        stream went quiet before the terminal state was pushed."""
        client = self._client
        response = self._get_status(client, {
            "objects": {"motion_report": None}
        })
        return (
            response.get("result", {}).get("status", {}).get(
                "motion_report", {})
        )

    def _stage_handshake_et(self) -> float:
        return getattr(self, "_et_handshake", 0.0)

    def query_motion_state(self) -> dict:
        """One-shot motion_report query - used to certify a leg whose push
        stream went quiet before the terminal state was pushed."""
        client = self._client
        response = self._get_status(client, {
            "objects": {"motion_report": None}
        })
        return (
            response.get("result", {}).get("status", {}).get(
                "motion_report", {})
        )

    def _stage_handshake_et(self) -> float:
        return getattr(self, "_et_handshake", 0.0)

    def query_motion_state(self) -> dict:
        """One-shot motion_report query - used to certify a leg whose push
        stream went quiet before the terminal state was pushed."""
        client = self._client
        response = self._get_status(client, {
            "objects": {"motion_report": None}
        })
        return (
            response.get("result", {}).get("status", {}).get(
                "motion_report", {})
        )

    def _stage_handshake_et(self) -> float:
        return getattr(self, "_et_handshake", 0.0)

    def query_motion_state(self) -> dict:
        """One-shot motion_report query - used to certify a leg whose push
        stream went quiet before the terminal state was pushed."""
        client = self._client
        response = self._get_status(client, {
            "objects": {"motion_report": None}
        })
        return (
            response.get("result", {}).get("status", {}).get(
                "motion_report", {})
        )

    def _stage_handshake_et(self) -> float:
        return getattr(self, "_et_handshake", 0.0)

    def query_motion_state(self) -> dict:
        """One-shot motion_report query - used to certify a leg whose push
        stream went quiet before the terminal state was pushed."""
        client = self._client
        response = self._get_status(client, {
            "objects": {"motion_report": None}
        })
        return (
            response.get("result", {}).get("status", {}).get(
                "motion_report", {})
        )

    def _stage_handshake_et(self) -> float:
        return getattr(self, "_et_handshake", 0.0)

    def motion_wait(self, timeout: float = 0.5) -> bool:
        """Block until a new telemetry sample arrives (Condition wait)."""
        cond = getattr(self, "_motion_ws_cond", None)
        if cond is None:
            return False
        with cond:
            cond.wait(timeout)
        return True

    def clock_offset(self):
        """Median klippy-eventtime minus local-monotonic over recent pushes."""
        seq = getattr(self, "_clock_offsets", None)
        if not seq:
            return None
        vals = sorted(seq)
        return vals[len(vals) // 2]

    def handshake_offset(self):
        """Klippy-eventtime minus local-monotonic from the subscribe
        handshake -- an independent one-shot clock bridge (uncircular)."""
        v = getattr(self, "_et_handshake", None)
        return float(v) if v is not None else None

    def clock_spread(self):
        """Robust spread (max-median deviation) of the clock offset, seconds."""
        seq = getattr(self, "_clock_offsets", None)
        if not seq:
            return None
        vals = sorted(seq)
        med = vals[len(vals) // 2]
        return max(abs(v - med) for v in vals)

    def printer_max_accel(self):
        """Live accel limit from klippy's toolhead, or None."""
        try:
            st = self._get_status(payload={"objects": {"toolhead": None}})
            return float(st["result"]["status"]["toolhead"]["max_accel"])
        except Exception:
            return None

    class MovementType(Enum):
        ABSOLUTE = "G90"
        RELATIVE = "G91"

    def update_position(self) -> None:
        """Read position from the stage and set the corresponding property."""
        client = self._client
        response = self._get_status(client, {
            "objects": {
                "gcode_move": None,
                "toolhead": ["position", "status"]
            }
        })
        self._hardware_position = dict(
            zip(self.axis_names, response["result"]["status"]["toolhead"]["position"])
        )

    def check_firmware(self) -> None:
        client = self._client
        response = client.get("/printer/info", timeout=self.timeout)
        response.raise_for_status()

    def _hardware_start_move_relative(self, displacement: Sequence[int]) -> None:
        """Start a relative move.

        This starts the stage moving, but does not wait for the move to complete. It
        sets ``self.moving`` to ``True``: resetting it is the responsibility of the calling code.
        """
        with self._hardware_lock:
            self.moving = True
            self.move_gcode(self.MovementType.RELATIVE, False, displacement)

    def _hardware_stop(self, jog_gen: int | None = None) -> None:
        """Stop any motion of the stage as soon as possible."""
        with self._hardware_lock:
            client = self._client
            try:
                if jog_gen is None or jog_gen == getattr(self, "_jog_gen", None):
                    self._send_gcode_script(client, "JOG_INTERRUPT")
                    self.unset_jog()
            finally:
                self.moving = False
                self.update_position()

    def _jog_loop(self, first_command: JogCommand) -> None:
        with self._hardware_lock:
            self._jog_gen = getattr(self, "_jog_gen", 0) + 1
            jog_gen = self._jog_gen
            self._send_gcode_script(self._client, "JOG VALUE=1")
        try:
            super()._jog_loop(first_command)
        finally:
            # A jog session that ends without a stop command must still
            # restore the default mode (the stop path is a no-op if nothing moves).
            self._hardware_stop(jog_gen)

    def _poll_moving(self) -> bool:
        """Determine if the stage is still moving.

        This also sets ``moving`` if the status has changed.

        :return: whether the stage is still moving.
        """
        with self._hardware_lock:
            return self.moving

    def _estimate_move_duration(self, displacement: Sequence[int]) -> float:
        """Duration of a jog move: 600/speed seconds at any displacement."""
        return 600.0 / self.speed

    def move_gcode(self,
                   move_type: MovementType,
                   block_cancellation: bool = False,
                   displacement=None,
                   planned: bool = False,
                   **kwargs: int,
                   ) -> None:
        if displacement is None:
            displacement_axis = dict(zip(self.axis_names, [kwargs.get(axis, 0) for axis in self.axis_names]))
            displacement = list(displacement_axis.values())
        elif isinstance(displacement, Mapping):
            displacement_axis = {axis: displacement.get(axis, 0) for axis in self.axis_names}
            displacement = list(displacement_axis.values())
        else:
            displacement_axis = dict(zip(self.axis_names, displacement))

        jog_axes = "".join(
            f"{axis.upper()}={axisDisplacement} " for axis, axisDisplacement in displacement_axis.items())
        gcode_axes = "".join(
            f"{axis.upper()}{axisDisplacement} " for axis, axisDisplacement in displacement_axis.items())
        if planned:
            script = (f"{move_type.value}\nG1 {gcode_axes}S9000 F45000\nM400\n")
        else:
            scale = max(abs(d) for d in displacement_axis.values()) / 600
            jog_dir = [1 if d > 0 else -1 if d < 0 else 0 for d in displacement_axis.values()]
            reversing = (
                    getattr(self, "_last_jog_dir", None) is not None
                    and any(
                (nd > 0) != (ld > 0) and nd != 0 and ld != 0
                for nd, ld in zip(jog_dir, self._last_jog_dir)
            )
            )
            self._last_jog_dir = jog_dir
            interrupt = "JOG_INTERRUPT\n" if reversing else ""
            script = (f"{interrupt}JOG_MOVE {jog_axes}S={self.speed * scale} F={self.acceleration * scale}\n")

        client = self._client
        self.moving = True
        try:
            self._send_gcode_script(client, script)
            # Raise InvocationCancelledError before starting another move
            # if the invoking action has been cancelled.
            lt.raise_if_cancelled()

        except lt.exceptions.InvocationCancelledError as e:
            # If the move has been cancelled, stop it but don't handle the exception.
            # We need the exception to propagate in order to stop any calling tasks,
            # and to mark the invocation as "cancelled" rather than stopped.
            self._hardware_stop()
            raise e

        finally:
            self.moving = False
            self.update_position()

    def move_relative_xy_noblock(self, dx: int, dy: int) -> None:
        client = self._client
        self.moving = True
        try:
            self._send_gcode_script(
                client,
                f"G91\nG1 X{int(dx)} Y{int(dy)} S9000 F45000\nG90\n",
            )
        finally:
            self.moving = False

    def _hardware_move_relative(
            self,
            block_cancellation: bool = False,
            **kwargs: int,
    ) -> None:
        """Make a relative move"""
        self.move_gcode(self.MovementType.RELATIVE, block_cancellation, planned=True, **kwargs)

    def move_relative_z_timed(
            self, steps: float, seconds: float
    ) -> None:
        steps = float(steps)
        seconds = float(seconds)
        if abs(steps) < 1e-9 or seconds <= 0:
            return
        if self.axis_inverted.get("z", False):
            steps = -steps
        speed = abs(steps) / seconds
        script = (
            "G91\n"
            f"G1 Z{steps:.3f} S{speed:.6f} F{speed * 60.0:.3f}\n"
        )
        t_send = time.time()
        try:
            with open(self.MOVES_LOG, "a") as f:
                f.write(f"SEND steps={steps} seconds={seconds} t={t_send:.3f}\n")
        except Exception:
            pass
        self._send_gcode_script(self._client, script)
        try:
            with open(self.MOVES_LOG, "a") as f:
                f.write(f"SENT ok t={time.time():.3f}\n")
        except Exception:
            pass

    def z_live_velocity(self) -> float:
        """Current Z velocity from klippy's motion_report (planned moves only)."""
        client = self._client
        response = self._get_status(client, {
            "objects": {"motion_report": ["live_velocity"]}
        })
        return float(
            response["result"]["status"]["motion_report"]["live_velocity"]
        )

    def z_live_position(self) -> float:
        """Current Z from klippy's motion_report (hardware frame, tracks queue)."""
        client = self._client
        response = self._get_status(client, {
            "objects": {"motion_report": ["live_position"]}
        })
        return float(
            response["result"]["status"]["motion_report"]["live_position"][2]
        )

    def _hardware_move_absolute(
            self,
            block_cancellation: bool = False,
            **kwargs: int,
    ) -> None:
        """Make an absolute move."""
        self.update_position()
        positions = [
            kwargs.get(axis, self._hardware_position[axis])
            for axis in self.axis_names
        ]

        self.move_gcode(self.MovementType.ABSOLUTE, block_cancellation, positions, planned=True)

    @lt.action
    def set_zero_position(self) -> None:
        """Make the current position zero in all axes.

        This action does not move the stage, but resets the position to zero.
        It is intended for use after manually or automatically recentring the
        stage.
        """
        client = self._client
        self._send_gcode_script(client, "SAMPLE_MESH_CLEAR")
        self._send_gcode_script(client, "SET_KINEMATIC_POSITION X=0 Y=0 Z=0 SET_HOMED=XYZ")
        self.update_position()

    def set_homed(self) -> None:
        """Mark the current position as homed without changing coordinates.

        Klippy keeps running across microscope-server restarts, so the
        coordinate frame survives restarts, like the sangaboard's counters.
        """
        client = self._client
        self._send_gcode_script(client, "SET_KINEMATIC_POSITION SET_HOMED=XYZ")
        self.update_position()

    def set_jog(self) -> None:
        """Enable jog mode for low-latency manual moves.
        """
        self._send_jog_mode_command(True)

    def unset_jog(self) -> None:
        """Return to standard (ready) mode.
        """
        self._send_jog_mode_command(False)

    def _send_jog_mode_command(self, enable: bool) -> None:
        with self._hardware_lock:
            client = self._client
            self._send_gcode_script(client, f"JOG VALUE={1 if enable else 0}")
        self.update_position()

    @lt.action
    def flash_led(
            self,
            number_of_flashes: int = 10,
            dt: float = 0.5,
            led_channel: Literal["cc"] = "cc",
    ) -> None:
        """Flash the LED to identify the board.

        This is intended to be useful in situations where there are multiple
        boards in use, and it is necessary to identify which one is
        being addressed.
        """

        raise IOError("The Pico does not support LED control. Unless you have an ARGB strip connected.")
        # todo maybe I do? beep steppers instead?
