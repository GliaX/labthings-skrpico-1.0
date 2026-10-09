"""Provide a LabThings-FastAPI interface to the Bigtreetech SKR Pico v1.0 motor controller."""

from __future__ import annotations

import httpx
import time
from types import TracebackType
from typing import Any, Literal, Mapping, Optional, Self, Sequence
from enum import Enum

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
        self._step_time = 0.000001
        self._client = httpx.Client(
            base_url=self.baseurl + ":" + self.port, timeout=self.timeout
        )
        super().__init__(thing_server_interface)

    def __enter__(self) -> Self:
        self.set_homed()
        return self
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

    def _send_gcode_script(self, client: Optional[httpx.Client] = None, script: str = "", timeout: Optional[httpx.Timeout] = None) -> dict:
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


    def _hardware_stop(self) -> None:
        """Stop any motion of the stage as soon as possible."""
        with self._hardware_lock:
            client = self._client
            try:
                self._send_gcode_script(client, "JOG_INTERRUPT")
            finally:
                self.unset_jog()
                self.moving = False
                self.update_position()

    def _jog_loop(self, first_command: JogCommand) -> None:
        try:
            self.set_jog()
            super()._jog_loop(first_command)
        finally:
            # A jog session that ends without a stop command must still
            # restore the default mode (the stop path is a no-op if nothing moves).
            self._hardware_stop()

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

        jog_axes = "".join(f"{axis.upper()}={axisDisplacement} " for axis, axisDisplacement in displacement_axis.items())
        gcode_axes = "".join(f"{axis.upper()}{axisDisplacement} " for axis, axisDisplacement in displacement_axis.items())
        if planned:
            script = (f"{move_type.value}\nG1 {gcode_axes}S9000 F45000\nM400\n")
        else:
            scale = max(abs(d) for d in displacement_axis.values()) / 600
            script = (f"JOG VALUE=1\nJOG_MOVE {jog_axes}S={self.speed * scale} F={self.acceleration * scale}\n")

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

    def _hardware_move_relative(
        self,
        block_cancellation: bool = False,
        **kwargs: int,
    ) -> None:
        """Make a relative move"""
        self.move_gcode(self.MovementType.RELATIVE, block_cancellation, planned=True, **kwargs)

    def move_relative_z_timed(
        self, steps: int, seconds: float, trailing_sync: bool = False
    ) -> None:
        steps = int(steps)
        seconds = float(seconds)
        if steps == 0 or seconds <= 0:
            return
        if self.axis_inverted.get("z", False):
            steps = -steps
        speed = abs(steps) / seconds
        script = (
            "G91\n"
            f"G1 Z{steps} S{speed:.6f} F{speed * 60.0:.3f}\n"
            + ("\nM400\n" if trailing_sync else "")
        )
        t_send = time.time()
        try:
            with open("/home/admin/openflexure-microscope-server/smooth_af_moves.log", "a") as f:
                f.write(f"SEND steps={steps} seconds={seconds} sync={trailing_sync} t={t_send:.3f}\n")
        except Exception:
            pass
        client = self._client
        self._send_gcode_script(
            client, script, timeout=httpx.Timeout(30.0 + seconds)
        )
        try:
            with open("/home/admin/openflexure-microscope-server/smooth_af_moves.log", "a") as f:
                f.write(f"SENT ok t={time.time():.3f}\n")
        except Exception:
            pass
        self.move_started_at = t_send + 0.02
        self.move_ended_at = t_send + 0.02 + seconds

    move_started_at: float = 0.0
    move_ended_at: float = 0.0

    def move_gcode_sync(self) -> None:
        """Block until the printer's motion queue is fully drained (M400)."""
        client = self._client
        self._send_gcode_script(client, "M400")

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
        #todo maybe I do? beep steppers instead?

