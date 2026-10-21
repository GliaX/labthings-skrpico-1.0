"""OpenFlexure Microscope API extension for stage calibration.

This file contains the HTTP API for camera/stage calibration. It
includes calibration functions that measure the relationship between
stage coordinates and camera coordinates, as well as functions that
move by a specified displacement in pixels, perform closed-loop moves,
and return the calibration data.

This module is only intended to be called from the OpenFlexure Microscope
server, and depends on that server and its underlying LabThings library.
"""

import json
import time
from typing import Any, List, Mapping, NamedTuple, Optional, Tuple, cast

import labthings_fastapi as lt
import numpy as np
from camera_stage_mapping.camera_stage_calibration_1d import (
    calibrate_backlash_1d,
    image_to_stage_displacement_from_1d,
)
from camera_stage_mapping.camera_stage_tracker import Tracker
from camera_stage_mapping.exceptions import MappingError
from labthings_fastapi.types.numpy import DenumpifyingDict

from .camera import BaseCamera
from .stage import BaseStage
from .view_alignment import ViewAlignmentMixin
from .smooth_autofocus import _EncoderBurst


class MoveHistory(NamedTuple):
    """A named tuple containing the position over time for a single move.

    This a named tuple with elements:

    * ``times``
    * ``stage_positions``
    """

    times: List[float]
    stage_positions: List[tuple[int, int, int]]


def _array_to_stage_tuple(pos: np.ndarray) -> tuple[int, int, int]:
    """Convert a numpy array into a tuple of ints.

    :param pos: Input position array must be 3 elements long.
    :return: a tuple of 3 integers
    :raises ValueError: If the array is not of length 3.
    """
    pos_tuple = tuple(int(i) for i in pos)
    if len(pos_tuple) == 3:
        return cast(tuple[int, int, int], pos_tuple)
    raise ValueError("Input array was not 3 elements long.")


def _serialise_numpy_in_dict(dict_with_numpy: dict) -> dict:
    serialised = json.loads(DenumpifyingDict(dict_with_numpy).model_dump_json())
    if not isinstance(serialised, dict):
        raise TypeError(f"Expecting a dictionary to serialise not a {type(serialised)}")
    return serialised


class RecordedMove:
    """Call stage movement and maintain a record of position and time.

    This class is callable, the callable wraps stage.move_to_xyz_position.

    The class records a list of all moves made and how long they took. This is useful
    for calibrating the stage as it allows measuring how long moves take.
    """

    def __init__(self, stage: BaseStage) -> None:
        """Set the stage client used for for movement.

        :param stage: the stage client to be used. ``stage.move_to_xyz_position`` will
            be called whenever the instance is called.
        """
        self._stage = stage
        self._current_position: Optional[tuple[int, int, int]] = None
        self._history: List[Tuple[float, tuple[int, int, int]]] = []

    def __call__(self, new_position: np.ndarray) -> None:
        """Move to a new position, and record it."""
        new_stage_pos = _array_to_stage_tuple(new_position)
        starting_pos = self._current_position
        if starting_pos is not None:
            self._history.append((time.time(), starting_pos))
        self._stage.move_to_xyz_position(xyz_pos=new_stage_pos)
        self._current_position = new_stage_pos
        self._history.append((time.time(), new_stage_pos))

    @property
    def history(self) -> MoveHistory:
        """The history, as a numpy array of times and another of positions."""
        times: List[float] = [t for t, p in self._history]
        positions: List[tuple[int, int, int]] = [p for t, p in self._history]
        return MoveHistory(times, positions)

    def clear_history(self) -> None:
        """Reset our history to be an empty list."""
        self._history = []


class CSMUncalibratedError(lt.exceptions.InvocationError):
    """An Exception raised if camera stage mapping data is needed but unavailable.

    Camera Stage Mapping data is needed to convert from distances specified in fractions
    of the field of view to distances in motor steps. This is used when clicking on the
    live preview to move, or when performing a scan.
    """


class CameraStageMapper(ViewAlignmentMixin, lt.Thing):
    """A Thing to manage mapping between image and stage coordinates.

    To use this Thing, the stage must have axes named "x", "y", and "z", or must
    override the ``get_xyz_position()`` and ``move_to_xyz_position()`` methods.
    """

    _class_settings = {"validate_properties_on_set": True}

    _cam: BaseCamera = lt.thing_slot()
    _stage: BaseStage = lt.thing_slot()

    def _use_fast_cal(self) -> bool:
        """True when the stage is an skrPico-class stage: calibration runs
        in the camera's high-fps crop990 mode because the stage is fast
        enough to outrun the full-res frame rate. Any other stage keeps
        the stock full-resolution behaviour."""
        return hasattr(self._stage, "query_motion_state")

    def _calibrate_1d_fast(self, direction: Tuple[int, int, int]) -> dict:
        """1D calibration in the camera's high-fps streaming mode.

        The stage is fast enough to outrun full-resolution frames, so the
        tracker runs against the live crop990 stream instead. The result is
        scaled to full-resolution pixel units so downstream consumers see
        the same matrix as a full-res calibration would produce."""
        cam = self._cam
        held_before = cam.stream_mode_held
        if not held_before:
            cam.hold_stream_mode("csm_cal")
        prev_mode = cam.streaming_mode
        try:
            if prev_mode != "crop990":
                cam._stop_streaming(stop_web_stream=False)
                cam._start_streaming("crop990")
            with _EncoderBurst(cam):
                recorded_move = RecordedMove(self._stage)
                tracker = Tracker(
                    cam.capture_downsampled_array,
                    self._stage.get_xyz_position,
                    settle=cam.settle,
                )
                direction_array: np.ndarray = np.array(direction)
                starting_position = self._stage.position
                try:
                    result: dict = calibrate_backlash_1d(
                        tracker, recorded_move, direction_array, logger=self.logger
                    )
                except lt.exceptions.InvocationCancelledError:
                    self.logger.info("User cancelled the CSM calibration")
                    self._stage.move_absolute(
                        **starting_position, block_cancellation=True,
                        backlash_compensation=None,
                    )
                    raise
                except MappingError:
                    self.logger.info("Returning to starting position due to failed calibration")
                    self._stage.move_absolute(
                        **starting_position, block_cancellation=True,
                        backlash_compensation=None,
                    )
                    raise
                result["move_history"] = recorded_move.history
                result["image_resolution"] = cam.capture_downsampled_array().shape[:2]
                result["calibration_stream_mode"] = "crop990"
                return result
        finally:
            if not held_before:
                cam.release_stream_mode("csm_cal")
                if cam.streaming_mode != prev_mode:
                    cam._stop_streaming(stop_web_stream=False)
                    cam._start_streaming(prev_mode)


    def calibrate_1d(self, direction: Tuple[int, int, int]) -> dict:
        """Move a microscope's stage in 1D, and figure out the relationship with the camera."""
        if self._use_fast_cal():
            return self._calibrate_1d_fast(direction)
        # Record positions and times for stage calibration
        recorded_move = RecordedMove(self._stage)
        tracker = Tracker(
            self._cam.capture_downsampled_array,
            self._stage.get_xyz_position,
            settle=self._cam.settle,
        )
        direction_array: np.ndarray = np.array(direction)

        starting_position = self._stage.position
        try:
            result: dict = calibrate_backlash_1d(
                tracker, recorded_move, direction_array, logger=self.logger
            )
        except lt.exceptions.InvocationCancelledError as e:
            self.logger.info("User cancelled the camera stage mapping calibration")
            self.logger.info("Returning to starting position")
            self._stage.move_absolute(
                **starting_position, block_cancellation=True, backlash_compensation=None
            )
            raise e
        except MappingError as e:
            self.logger.info("Returning to starting position due to failed calibration")
            self._stage.move_absolute(
                **starting_position, block_cancellation=True, backlash_compensation=None
            )
            raise e
        result["move_history"] = recorded_move.history
        result["image_resolution"] = self._cam.capture_downsampled_array().shape[:2]
        return result

    @lt.action
    def calibrate_xy(self) -> dict:
        """Move the microscope's stage in X and Y, to calibrate its relationship to the camera.

        This performs two 1d calibrations in x and y, then combines their results.
        """
        downsampling_factor = self._cam.downsampled_array_factor
        # Calibrate y-axis first as it is more likely to fail.
        # The x-y difference is due to the camera aspect ratio, not the stage hardware.
        self.logger.info("Calibrating Y axis:")
        cal_y: dict = self.calibrate_1d((0, 1, 0))
        self.logger.info("Calibrating X axis:")
        cal_x: dict = self.calibrate_1d((1, 0, 0))
        # Fast-mode calibrations ran against the crop990 stream: scale their
        # matrices to full-resolution pixel units so the combined result
        # matches what a full-res calibration would have produced.
        if cal_x.get("calibration_stream_mode") == "crop990":
            full_res = self._cam.streaming_modes["default"].main_resolution
            scale = full_res[0] / cal_x["image_resolution"][0]
            self.logger.info(
                "Scaling crop990 calibration x%.2f to full-res units", scale
            )
            for cal in (cal_x, cal_y):
                cal["image_to_stage_displacement"] = (
                    cal["image_to_stage_displacement"] * scale
                )
                cal["image_resolution"] = full_res
        self.logger.info("Calibration complete, updating metadata.")

        # Combine X and Y calibrations to make a 2D calibration
        cal_xy: dict = image_to_stage_displacement_from_1d([cal_x, cal_y])
        # Correct the result for downsampling performed in the hardware interface
        # (this may be to speed up correlation, or to avoid debayering artifacts)
        cal_xy["image_to_stage_displacement"] /= downsampling_factor
        corrected_resolution = tuple(
            r * downsampling_factor for r in cal_x["image_resolution"]
        )

        csm_matrix = cal_xy["image_to_stage_displacement"]
        csm_as_string = (
            f"[{round(csm_matrix[0][0], 2)}, {round(csm_matrix[0][1], 2)},],"
            f"[{round(csm_matrix[1][0], 2)}, {round(csm_matrix[1][1], 2)}]"
        )
        self.logger.info(f"CSM matrix is {csm_as_string}.")

        data = {
            "camera_stage_mapping_calibration": cal_xy,
            "linear_calibration_x": cal_x,
            "linear_calibration_y": cal_y,
            "downsampled_image_resolution": cal_x["image_resolution"],
            "image_resolution": corrected_resolution,
            "downsampling": downsampling_factor,
        }

        data = _serialise_numpy_in_dict(data)
        self.last_calibration = data

        return data

    @lt.property
    def image_to_stage_displacement_matrix(self) -> Optional[List[List[float]]]:
        """A 2x2 matrix that converts displacement in image coordinates to stage coordinates.

        Note that this matrix is defined using "matrix coordinates", i.e. image coordinates
        may be (y,x). This is an artifact of the way numpy, opencv, etc. define images. If
        you are making use of this matrix in your own code, you will need to take care of
        that conversion.

        It is often helpful to give a concrete example: to make a move in image coordinates
        (``dy``, ``dx``), where ``dx`` is horizontal, i.e. the longer dimension of the image, you
        should move the stage by:

        .. code-block:: python

            stage_disp = np.dot(
                np.array([dy,dx]),
                np.array(image_to_stage_displacement_matrix),
            )

        """
        if self.last_calibration is None:
            return None
        displacement_matrix = self.last_calibration["camera_stage_mapping_calibration"][
            "image_to_stage_displacement"
        ]
        return np.array(displacement_matrix).tolist()

    last_calibration: Optional[dict] = lt.setting(default=None, readonly=True)
    """The most recent CSM calibration."""

    @lt.property
    def image_resolution(self) -> Optional[Tuple[float, float]]:
        """The image size used to calibrate the image_to_stage_displacement_matrix."""
        if self.last_calibration is None:
            return None
        return self.last_calibration["image_resolution"]

    @lt.property
    def calibration_required(self) -> bool:
        """Whether the camera stage mapper needs calibrating."""
        return self.image_to_stage_displacement_matrix is None

    def assert_calibration(self) -> List[List[float]]:
        """Return image_to_stage_displacement matrix or raise error if it's not set."""
        if self.image_to_stage_displacement_matrix is None:
            raise CSMUncalibratedError(
                "The camera_stage_mapping calibration is not yet available. "
                "This probably means you need to run the calibration routine."
            )
        return self.image_to_stage_displacement_matrix

    @lt.property
    def can_calibrate(self) -> bool:
        """Whether the Thing permits calibration."""
        return True

    @lt.action
    def move_in_image_coordinates(self, x: float, y: float) -> None:
        """Move by a given number of pixels on the camera.

        NB x and y here refer to what is usually understood to be the horizontal and
        vertical axes of the image. In many toolkits, "matrix indices" are used, which
        swap the order of these coordinates. This includes opencv and PIL. So, don't be
        surprised if you find it necessary to swap x and y around.

        As a general rule, ``x`` usually corresponds to the longer dimension of the image,
        and ``y`` to the shorter one. Checking what shape your chosen toolkit reports for
        an image usually helps resolve any ambiguity.
        """
        self._stage.move_relative(
            **self.convert_image_to_stage_coordinates(x=x, y=y),
            block_cancellation=False,
            backlash_compensation=None,
        )

    @lt.action
    def convert_image_to_stage_coordinates(
            self, x: float, y: float, **_kwargs: float
    ) -> Mapping[str, int]:
        """Convert image coordinates to stage coordinates. Only x and y are returned."""
        csm_matrix = self.assert_calibration()
        return csm_img_to_stage(csm_matrix, x=x, y=y)

    @lt.action
    def convert_stage_to_image_coordinates(
            self, x: int, y: int, **_kwargs: int
    ) -> Mapping[str, float]:
        """Convert stage coordinates to image coordinates. Only x and y are returned."""
        csm_matrix = self.assert_calibration()
        return csm_stage_to_img(csm_matrix, x=x, y=y)

    @lt.property
    def thing_state(self) -> Mapping[str, Any]:
        """Summary metadata describing the current state of the Thing."""
        return {
            k: getattr(self, k)
            for k in ["image_to_stage_displacement_matrix", "image_resolution"]
        }


def csm_img_to_stage(
        matrix: np.ndarray | list[list[float]],
        *,
        x: float | int,
        y: float | int,
        **_kwargs: int,
) -> Mapping[str, int]:
    """Apply any CSM matrix to image coordinates.

    x and y must be kwargs and extra kwargs are ignored, allowing:
    ``csm_img_to_stage(matrix, **position)`` to run for a mapping position.

    Note that x and y are the actual (x, y) of the image, not the (m, n) indices used
    by numpy

    :param matrix: The matrix to use in the calculation
    :param x: the x image coordinate (keyword only)
    :param y: the y image coordinate (keyword only)
    :return: The resulting stage coordinates as a mapping.
    """
    # Note this is (y,x) not (x,y) to put it in numpy image indices
    relative_move: np.ndarray = np.dot(np.array([y, x]), np.array(matrix))
    return {"x": round(relative_move[0]), "y": round(relative_move[1])}


def csm_stage_to_img(
        matrix: np.ndarray | list[list[float]],
        *,
        x: float | int,
        y: float | int,
        **_kwargs: int,
) -> Mapping[str, float]:
    """Apply any CSM matrix to stage coordinates to get image coordinates.

    x and y must be kwargs and extra kwargs are ignored, allowing:
    ``csm_img_to_stage(matrix, **position)`` to run for a mapping position.

    Note that x and y are the actual (x, y) of the image, not the (m, n) indices used
    numpy

    :param matrix: The matrix to use in the calculation
    :param x: the x stage coordinate (keyword only)
    :param y: the y stage coordinate (keyword only)
    :return: The resulting img coordinates as a mapping.
    """
    inverse_matrix = np.linalg.inv(np.array(matrix))
    relative_move = np.dot(np.array([x, y]), inverse_matrix)
    # Note that the relative move is (y, x) as it is from numpy and is in matrix coords.
    return {"x": float(relative_move[1]), "y": float(relative_move[0])}
