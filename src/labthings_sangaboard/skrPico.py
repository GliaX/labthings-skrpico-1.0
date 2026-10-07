"""Provide a LabThings-FastAPI interface to the Bigtreetech SKR Pico v1.0 motor controller."""

from __future__ import annotations

import httpx
from types import TracebackType
from typing import Any, Literal, Mapping, Optional, Self
from enum import Enum

import labthings_fastapi as lt

from . import BaseStage

class SkrPicoThing(BaseStage):
    """A Thing to manage a SKR Pico v1.0 motor controller using Moonraker.
    """
    def __init__(
        self,
        thing_server_interface: lt.ThingServerInterface,
        **kwargs: Any,
    ) -> None:
        self.port = kwargs["moonrakerport"] if "moonrakerport" in kwargs else "7125"
        self.baseurl = kwargs["baseurl"] if "baseurl" in kwargs else "http://127.0.0.1"
        self.acceleration = kwargs["acceleration"] if "acceleration" in kwargs else 45000
        self.speed = kwargs["speed"] if "speed" in kwargs else 1000
        self.timeout = httpx.Timeout(60.0)
        self._step_time = 0.000001
        super().__init__(thing_server_interface, **kwargs)

    def __enter__(self) -> Self:
        self.set_zero_position()
        with httpx.Client() as client:
            r = client.get(self.baseurl + ":" + self.port + "/printer/info",timeout=self.timeout)
            # todo check http status throw error?
    def __exit__(
            self,
            _exc_type: type[BaseException],
            _exc_value: Optional[BaseException],
            _traceback: Optional[TracebackType],
    ) -> None:
        with httpx.Client() as client:
            client.close()

    # Z1 is chained to Z but still reports position data
    axis_inverted: dict[str, bool] = lt.setting(
        default={"x": True, "y": False, "z": True}, readonly=True
    )
    class MovementType(Enum):
        ABSOLUTE = "G90"
        RELATIVE = "G91"

    def update_position(self) -> None:
        """Read position from the stage and set the corresponding property."""
        with (httpx.Client() as client):
            response = client.post(self.baseurl + ":" + self.port + "/printer/objects/query", timeout=self.timeout, json = {
                "objects": {
                    "gcode_move": None,
                    "toolhead": ["position", "status"]
                }
            }).json()
            # todo check http status

            self._hardware_position = dict(
                zip(self.axis_names, response["result"]["status"]["toolhead"]["position"])
            )


    def check_firmware(self) -> None:
        httpx.get(self.baseurl + "/printer/info", timeout=self.timeout) # todo check http status

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
            with (httpx.Client() as client):
                try:
                    response = client.post(self.baseurl + ":" + self.port + "/printer/gcode/script",
                                           timeout=self.timeout, json={
                            "script": "JOG_INTERRUPT"
                        }).json()
                # todo check http status
                # todo implement api key / security
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

        with (httpx.Client() as client):
            self.moving = True
            try:
                response = client.post(self.baseurl + ":" + self.port + "/printer/gcode/script", timeout=self.timeout, json={
                    "script": script
                }).json()
                # Raise InvocationCancelledError before starting another move
                # if the invoking action has been cancelled.
                lt.raise_if_cancelled()

            except lt.exceptions.InvocationCancelledError as e:
                # If the move has been cancelled, stop it but don't handle the exception.
                # We need the exception to propagate in order to stop any calling tasks,
                # and to mark the invocation as "cancelled" rather than stopped.
                self._hardware_stop()
                raise e

            # todo check http status
            # todo implement api key / security
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
        with httpx.Client() as client:
            response = client.post(self.baseurl + ":" + self.port + "/printer/gcode/script", timeout=self.timeout, json={
                "script": "SET_KINEMATIC_POSITION X=0 Y=0 Z=0 SET_HOMED=XYZ"

        }).json()
        self.update_position()
        # todo check http status

    def set_jog(self) -> None:
        """Enable jog mode for low-latency manual moves.
        """
        self._send_jog_mode_command(True)

    def unset_jog(self) -> None:
        """Return to standard (ready) mode.
        """
        self._send_jog_mode_command(False)

    def _send_jog_mode_command(self, enable: bool) -> None:
        with httpx.Client() as client:
            response = client.post(self.baseurl + ":" + self.port + "/printer/gcode/script", timeout=self.timeout,
                                   json={
                                       "script": f"JOG VALUE={1 if enable else 0}"
                                   }).json()
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

