"""Provide a LabThings-FastAPI interface to the Bigtreetech SKR Pico v1.0 motor controller."""

from __future__ import annotations

import httpx
from types import TracebackType
from typing import Any, Literal, Optional, Self
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
        self.acceleration = kwargs["acceleration"] if "acceleration" in kwargs else 15000
        self.speed = kwargs["speed"] if "speed" in kwargs else 1000
        super().__init__(thing_server_interface, **kwargs)

    def __enter__(self) -> Self:
        self.set_zero_position()
        with httpx.Client() as client:
            r = client.get(self.baseurl + ":" + self.port + "/printer/info")
            # todo check http status throw error?
    def __exit__(
            self,
            _exc_type: type[BaseException],
            _exc_value: Optional[BaseException],
            _traceback: Optional[TracebackType],
    ) -> None:
        """Close the sangaboard connection when the Thing context manager is closed."""
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
            response = client.post(self.baseurl + ":" + self.port + "/printer/objects/query", json = {
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
        httpx.get(self.baseurl + "/printer/info") # todo check http status

    def move_gcode(self,
        move_type: MovementType,
        block_cancellation: bool = False, ## todo later, implement cancels
        **kwargs: int,
                   ) -> None:

        displacement = dict(zip(self.axis_names, [kwargs.get(axis, 0) for axis in self.axis_names]))
        with (httpx.Client() as client):
            self.moving = True
            try:
                response = client.post(self.baseurl + ":" + self.port + "/printer/gcode/script", json={
                    "script": f"{move_type.value} \n" +
                              "G1 "+ "".join(f"{axis.upper()}{axisDisplacement} " for axis, axisDisplacement in displacement.items()) +
                              f"S{self.speed} F{self.acceleration}\n"
                }).json()
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
        """Make a relative move using G91"""
        self.move_gcode(self.MovementType.RELATIVE, block_cancellation, **kwargs)


    def _hardware_move_absolute(
        self,
        block_cancellation: bool = False,
        **kwargs: int,
    ) -> None:
        """Make an absolute move."""
        self.move_gcode(self.MovementType.ABSOLUTE, block_cancellation, **kwargs)

    @lt.action
    def set_zero_position(self) -> None:
        """Make the current position zero in all axes.

        This action does not move the stage, but resets the position to zero.
        It is intended for use after manually or automatically recentring the
        stage.
        """
        with httpx.Client() as client:
            response = client.post(self.baseurl + ":" + self.port + "/printer/gcode/script", json={
                "script": "SET_KINEMATIC_POSITION X=0 Y=0 Z=0 SET_HOMED=XYZ"

        }).json()
        self.update_position()
        # todo check http status

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

