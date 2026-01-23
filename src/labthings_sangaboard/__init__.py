"""A package for stage control Things.

`BaseStage` is the base class that provides core stage functionality, but
no hardware interface control. To create a stage Thing to control a specific
piece of hardware the BaseStage should be subclassed, and any method raising
a NotImplementedError should be created.

As the object will be used as a context manager create the hardware connection in
``__enter__`` (not in ``__init__``), and close the connection with ``__exit__``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, overload

import labthings_fastapi as lt


class RedefinedBaseMovementError(RuntimeError):
    """The subclass of BaseStage has overridden ``move_relative`` or ``move_absolute``.

    Overriding ``move_relative`` or ``move_absolute`` can be problematic as these use the
    external position not the hardware position. It is recommended to override
    ``_hardware_move_relative`` and ``_hardware_move_absolute`` instead.

    The BaseStage will raise this on ``__init__``, it is the last thing ``__init__``
    does. As such, this exception can be captured by ``try`` if a stage needs to
    override these for a specific reason.
    """


class BaseStage(lt.Thing):
    """A base stage class for OpenFlexure translation stages.

    This can't be used directly but should reduce boilerplate code when
    implementing new stages.

    Note that the coordinate system used for the microscope may need to have different
    axis direction as those used by the underlying stage controller.

    A minimal working stage must implement ``_hardware_move_relative``
    and ``_hardware_move_absolute`` actions, which update the ``_hardware_position``
    attribute on completion, and also should implement ``set_zero_position``.
    """

    _axis_names = ("x", "y", "z")

    def __init__(self, thing_server_interface: lt.ThingServerInterface) -> None:
        """Initialise the stage.

        :raises RedefinedBaseMovementError: if ``move_relative`` and/or
            ``move_absolute`` are overridden. It is recommended to override
            ``_hardware_move_relative`` and/or ``_hardware_move_absolute`` instead so
            that all code in the child class uses the hardware reference frame.
        """
        super().__init__(thing_server_interface)
        self._hardware_position = dict.fromkeys(self._axis_names, 0)

        # This must be the last thing the function does in case it is caught in a try.
        if (
            self.__class__.move_relative.func is not BaseStage.move_relative.func
            or self.__class__.move_absolute.func is not BaseStage.move_absolute.func
        ):
            raise RedefinedBaseMovementError(
                "move_relative and/or move_absolute has been overridden. This may "
                "cause issues as the base methods implement converting from program "
                "coordinates to hardware coordinates. Consider overriding "
                "_hardware_move_relative and/or _hardware_move_absolute instead."
            )

    @lt.property
    def axis_names(self) -> Sequence[str]:
        """The names of the stage's axes, in order."""
        return self._axis_names

    @lt.property
    def position(self) -> Mapping[str, int]:
        """Current position of the stage."""
        return self._apply_axis_direction(self._hardware_position)

    moving: bool = lt.property(default=False, readonly=True)
    """Whether the stage is in motion."""

    axis_inverted: dict[str, bool] = lt.setting(
        default={"x": False, "y": False, "z": False}, readonly=True
    )
    """Used to convert coordinates between the program frame and the hardware frame."""

    @overload
    def _apply_axis_direction(self, position: list[int] | tuple[int]) -> list[int]: ...

    @overload
    def _apply_axis_direction(
        self, position: Mapping[str, int]
    ) -> Mapping[str, int]: ...

    def _apply_axis_direction(
        self, position: list[int] | tuple[int] | Mapping[str, int]
    ) -> list[int] | Mapping[str, int]:
        if isinstance(position, (list, tuple)):
            return [
                -int(pos) if inverted else int(pos)
                for pos, inverted in zip(
                    position, self.axis_inverted.values(), strict=True
                )
            ]
        if isinstance(position, Mapping):
            try:
                return {
                    ax: -int(position[ax])
                    if self.axis_inverted[ax]
                    else int(position[ax])
                    for ax in position
                }
            except KeyError as e:
                raise KeyError(
                    f"One or more axis in {position.keys()} is not defined."
                ) from e
        raise TypeError(
            "Position must be a sequence of positions or a mapping from axis to position."
        )

    @property
    def thing_state(self) -> Mapping[str, Any]:
        """Summary metadata describing the current state of the stage."""
        return {"position": self.position}

    @lt.action
    def invert_axis_direction(self, axis: Literal["x", "y", "z"]) -> None:
        """Invert the direction setting of the given axis.

        :param axis: The axis name (x, y or z) to invert.
        """
        # Not mutating in place so that setting is saved on change.
        direction = self.axis_inverted
        try:
            direction[axis] = not direction[axis]
        except KeyError as e:
            raise KeyError(f"The axis {axis} is not defined.") from e
        self.axis_inverted = direction

    @lt.action
    def move_relative(self, block_cancellation: bool = False, **kwargs: int) -> None:
        """Make a relative move. Keyword arguments should be axis names."""
        self._hardware_move_relative(
            block_cancellation=block_cancellation,
            **self._apply_axis_direction(kwargs),
        )

    def _hardware_move_relative(
        self, block_cancellation: bool = False, **kwargs: int
    ) -> None:
        """Make a relative move in the coordinate system used by the physical hardware.

        Make sure to use and update ``self._hardware_position`` not ``self.position``.
        """
        raise NotImplementedError(
            "StageThings must define their own _hardware_move_relative method"
        )

    @lt.action
    def move_absolute(self, block_cancellation: bool = False, **kwargs: int) -> None:
        """Make an absolute move. Keyword arguments should be axis names."""
        self._hardware_move_absolute(
            block_cancellation=block_cancellation,
            **self._apply_axis_direction(kwargs),
        )

    def _hardware_move_absolute(
        self,
        block_cancellation: bool = False,
        **kwargs: int,
    ) -> None:
        """Make a absolute move in the coordinate system used by the physical hardware.

        Make sure to use and update ``self._hardware_position`` not ``self.position``.
        """
        raise NotImplementedError(
            "StageThings must define their own move_absolute method"
        )

    @lt.action
    def set_zero_position(self) -> None:
        """Make the current position zero in all axes.

        This action does not move the stage, but resets the position to zero.
        It is intended for use after manually or automatically recentring the
        stage.
        """
        raise NotImplementedError(
            "StageThings must define their own set_zero_position method"
        )

    @lt.action
    def get_xyz_position(self) -> tuple[int, int, int]:
        """Return a tuple containing (x, y, z) position.

        :raises KeyError: if this stage does not have axes named "x", "y", and "z".

        This method provides the interface expected by the camera_stage_mapping.
        """
        position_dict = self.position
        return (position_dict["x"], position_dict["y"], position_dict["z"])

    @lt.action
    def move_to_xyz_position(self, xyz_pos: tuple[int, int, int]) -> None:
        """Move to the location specified by an (x, y, z) tuple.

        :param xyz_pos: The (x, y, z) position to move to.

        :raises KeyError: if this stage does not have axes named "x", "y", and "z".

        This method provides the interface expected by the camera_stage_mapping.
        """
        self.move_absolute(x=xyz_pos[0], y=xyz_pos[1], z=xyz_pos[2])
