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
import os
import time
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal, Optional

import numpy as np

import labthings_fastapi as lt
from openflexure_stitching.correlation import displacement_between_images

from openflexure_microscope_server.things import OFMThing

from .camera import BaseCamera
from .stage import BaseStage

SETTLING_TIME = 0.1


class CSMUncalibratedError(lt.exceptions.InvocationError):
    """An Exception raised if camera stage mapping data is needed but unavailable.

    Camera Stage Mapping data is needed to convert from distances specified in fractions
    of the field of view to distances in motor steps. This is used when clicking on the
    live preview to move, or when performing a scan.
    """


class MappingError(RuntimeError):
    """Failure during the Camera Stage Mapping procedure."""


def _csm_matrix_to_string(mat: np.ndarray) -> str:
    return (
        f"[[{round(mat[0, 0], 2)}, {round(mat[0, 1], 2)},],"
        f"[{round(mat[1, 0], 2)}, {round(mat[1, 1], 2)}]]"
    )


class CameraStageMapper(OFMThing):
    """A Thing to manage mapping between image and stage coordinates.

    To use this Thing, the stage must have axes named "x", "y", and "z", or must
    override the ``get_xyz_position()`` and ``move_to_xyz_position()`` methods.
    """

    _class_settings = {"validate_properties_on_set": True}

    _cam: BaseCamera = lt.thing_slot()
    _stage: BaseStage = lt.thing_slot()

    @lt.action
    def calibrate_xy(self) -> None:
        """Measure the relationship between camera and stage coordinates.

        Note: this is the method used by camera_stage_mapping to measure the CSM matrix,
        calling this directly will perform a measurement but not set the measurement,
        for use.
        """
        # The tracker must look through an isotropic stream. crop990's
        # main stream is an anisotropic 1332x990 sensor window squeezed
        # into 700x700, which bends the y scale by 990/1332 (measured
        # 2026-11-01: y=4.43 px/u against the true 5.83). Hold default
        # (square sensor, isotropic) for the measurement.
        cam = self._cam
        held_before = cam.stream_mode_held
        prev_mode = cam.streaming_mode
        if not held_before:
            cam.hold_stream_mode("csm_cal")
        if prev_mode != "default":
            cam._stop_streaming(stop_web_stream=False)
            cam._start_streaming("default")
        if cam.streaming_mode != "default":
            self.logger.warning(
                "CSM cal on non-default stream %s; y scale will be wrong",
                cam.streaming_mode,
            )
        try:
            self._calibrate_xy_body()
        finally:
            if not held_before:
                cam.release_stream_mode("csm_cal")
                if cam.streaming_mode != prev_mode:
                    cam._stop_streaming(stop_web_stream=False)
                    cam._start_streaming(prev_mode)

    def _calibrate_xy_body(self) -> None:
        cal_data: dict[str, Any] = {}
        starting_position = self._stage.position
        image_resolution = self._cam.grab_as_array().shape[:2]
        # Set the target motion in pixels to one 10th of the smallest camera axis
        target_movement_px = min(image_resolution) // 10

        # Strictly typed definitions to iterate over for MyPys sake.
        axes: tuple[Literal["x", "y"], ...] = ("y", "x")
        for axis in axes:
            cal_data |= self._measure_single_axis(
                axis,
                target_movement_px=target_movement_px,
                starting_position=starting_position,
            )

        # If the data was collected always save it even if there is an error in fitting
        try:
            # Fit all the results
            self.logger.info("Fitting all data")
            csm_matrix_np, backlash = self._fit_all_csm_runs(cal_data)

            self.logger.info(f"Estimated backlash {backlash} steps")
            self.logger.info(f"CSM matrix is {_csm_matrix_to_string(csm_matrix_np)}.")

            csm_matrix = csm_matrix_np.tolist()
            cal_data["csm_matrix"] = csm_matrix
            cal_data["backlash"] = backlash
        finally:
            timestamp = self._write_csm_data_to_disk(cal_data)

        # Save settings if successfully fitted.
        self.image_to_stage_displacement_matrix = csm_matrix
        self.image_resolution = image_resolution
        # Data is saved on an error, but the timestamp isn't updated.
        self.last_cal_timestamp = timestamp

    def _measure_single_axis(
        self,
        axis: Literal["x", "y"],
        *,
        target_movement_px: int,
        starting_position: Mapping[str, int],
    ) -> dict[str, Any]:
        """Measure a single axis, and return the measurement data in a dictionary."""
        # Note: `cal_data` keys must contain the axis name, as this dictionary
        # is merged with the same dictionary from the other axis.
        cal_data: dict[str, Any] = {}
        try:
            self.logger.info(f"Measuring {axis} axis")
            # First using increasing steps to find motion
            self.logger.info("Moving until motion is detected")
            stage_positions, cam_offsets = self._detect_motion_on_axis(
                axis=axis, target_movement_px=target_movement_px
            )
            # Calculate step size
            index = 0 if axis == "x" else 1
            step_size = stage_positions[-1][index] - stage_positions[0][index]
            cal_data[f"{axis}_detection"] = {
                "stage_positions": stage_positions,
                "cam_offsets": cam_offsets,
            }

            # Run motion backwards and forwards to measure CSM
            cal_data[f"{axis}_run"] = []
            total_moves = 4
            for i in range(total_moves):
                sign = 1 if i % 2 == 0 else -1
                self.logger.info(
                    f"Performing {axis} axis measurement {i + 1} of {total_moves}"
                )
                stage_positions, cam_offsets = self._measure_axis_displacement(
                    axis=axis, step_size=sign * step_size, n_moves=10
                )
                cal_data[f"{axis}_run"].append(
                    {
                        "stage_positions": stage_positions,
                        "cam_offsets": cam_offsets,
                    }
                )
        finally:
            self._stage.move_absolute(
                **starting_position, block_cancellation=True, backlash_compensation=None
            )
        return cal_data

    def _write_csm_data_to_disk(self, data: dict[str, Any]) -> str:
        """Write the CSM data to disk and return the timestamp."""
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        datafile_path = os.path.join(self.data_dir, f"csm_data_{timestamp}.json")
        with open(datafile_path, "w", encoding="utf-8") as file_obj:
            json.dump(data, file_obj, indent=4)
        return timestamp

    def _detect_motion_on_axis(
        self, axis: Literal["x", "y"], target_movement_px: int
    ) -> tuple[list[tuple[int, int, int]], list[tuple[float, float]]]:
        """Move the stage in increasing steps until motion is detected.

        The stage is moved in one axis. This will complete if motion is detected in
        any direction in the camera coordinates.

        :return: Two lists, One of absolute x positions and camera .
        """
        is_sample, _bg_message = self._cam.image_is_sample()
        if not is_sample:
            raise MappingError(
                "No sample detected. Camera stage mapping requires a sample."
            )
        stage_pos = [self._stage.get_xyz_position()]
        cam_pos = [(0.0, 0.0)]

        initial_image = self._cam.grab_as_array()

        motion_detected = False
        for step_size in [2**i for i in range(15)]:
            ## Explicitly define steps as MyPy gets condused by the kwarg dict
            x_step, y_step = (step_size, 0) if axis == "x" else (0, step_size)
            # Move and measure displacement
            self._stage.move_relative(x=x_step, y=y_step)
            self._cam.settle()
            new_image = self._cam.grab_as_array()
            offset = displacement_between_images(initial_image, new_image)

            # Save positions
            stage_pos.append(self._stage.get_xyz_position())
            cam_pos.append((offset[0], offset[1]))

            largest = np.max(np.abs(offset))
            if largest > target_movement_px:
                motion_detected = True
                break
        if not motion_detected:
            raise MappingError(
                "No motion detected. Check a sample is present and motors are working."
            )
        return stage_pos, cam_pos

    def _measure_axis_displacement(
        self, axis: Literal["x", "y"], step_size: int, n_moves: int = 10
    ) -> tuple[list[tuple[int, int, int]], list[tuple[float, float]]]:
        """Measure camera vs stage displacement in 1 direction on one axis."""
        stage_pos = [self._stage.get_xyz_position()]
        cam_pos = [(0.0, 0.0)]

        previous_image = self._cam.grab_as_array()

        ## Explicitly define steps as MyPy gets condused by the kwarg dict
        x_step, y_step = (step_size, 0) if axis == "x" else (0, step_size)

        for _i in range(n_moves):
            self._stage.move_relative(x=x_step, y=y_step)
            self._cam.settle()
            new_image = self._cam.grab_as_array()
            offset = displacement_between_images(previous_image, new_image)
            previous_image = new_image

            # Save positions
            stage_pos.append(self._stage.get_xyz_position())
            # Note: Cam position is cumulative
            cam_pos.append((cam_pos[-1][0] + offset[0], cam_pos[-1][1] + offset[1]))
        return stage_pos, cam_pos

    def _fit_all_csm_runs(self, csm_data: dict[str, Any]) -> tuple[np.ndarray, float]:
        """Fit all runs from the camera stage.

        :param csm_data: The dictionary created by ``calibrate_xy``

        :return: The camera stage mapping matrix and the magnitude of the backlash.
        """
        all_stage_displacements = []
        all_cam_displacements = []
        backlashes = []

        # Strictly typed definitions to iterate over for MyPys sake.
        axes: tuple[Literal["x", "y"], ...] = ("x", "y")
        for axis in axes:
            axis_runs = csm_data[f"{axis}_run"]

            for i, run_data in enumerate(axis_runs):
                stage = np.array(run_data["stage_positions"])
                cam = np.array(run_data["cam_offsets"])

                if i > 0:
                    backlash, points_to_remove = self._fit_backlash(axis, stage, cam)
                    backlashes.append(backlash)
                else:
                    backlash = 0
                    points_to_remove = 0

                stage_displacements = stage[1:, :] - stage[:-1, :]
                cam_displacements = cam[1:, :] - cam[:-1, :]

                all_stage_displacements.append(stage_displacements[points_to_remove:])
                all_cam_displacements.append(cam_displacements[points_to_remove:])

        csm_matrix = self._fit_csm_matrix(
            all_stage_displacements, all_cam_displacements
        )
        return csm_matrix, float(np.mean(backlashes))

    def _fit_backlash(
        self, axis: Literal["x", "y"], stage_pos: np.ndarray, cam_pos: np.ndarray
    ) -> tuple[float, int]:
        """Fit the backlash in this run.

        Start with the full fit and remove the initial points until the standard error
        minimises. Then use the displacement between the first point and the
        extrapolated fit to calculate the backlash.

        :param stage_pos: A nx3 matrix of stage positions.
        :param cam_pos: A nx2 matrix of camera positions.

        :return: The estimated backlash, and the number of points to exclude due to
            backlash.
        """
        stage_pos = stage_pos[:, 0] if axis == "x" else stage_pos[:, 1]
        # Use last position to get camera index as the first cam_pos for a run is 0,0
        cam_is_x = abs(cam_pos[-1, 0]) > abs(cam_pos[-1, 1])
        cam_pos = cam_pos[:, 0] if cam_is_x else cam_pos[:, 1]

        n_points = len(stage_pos)
        # Allow removing up to 1/2 of the points
        max_cut = n_points // 2

        # standard errors of each fit
        errs = []
        # backlashes for each fit
        backlashes = []
        for n_cut in range(max_cut + 1):
            stage_pos_cut = stage_pos[n_cut:]
            cam_pos_cut = cam_pos[n_cut:]
            m, c = np.polyfit(stage_pos_cut, cam_pos_cut, 1)
            residuals = cam_pos_cut - (stage_pos_cut * m + c)
            errs.append(np.std(residuals, ddof=2) / np.sqrt(len(stage_pos_cut)))
            # Use first element to calculate backlash from extrapolated line
            backlashes.append(stage_pos[0] - (cam_pos[0] - c) / m)
        # The backlash for the lowest standard error fit.
        backlash = abs(backlashes[np.argmin(errs)])
        points_to_remove = int(np.ceil(abs(backlash / (stage_pos[1] - stage_pos[0]))))
        # If the standard error of the residuals is above 5 pixels raise an error.
        if np.min(errs) > 5:
            raise MappingError(
                "Unable to track stage motion. Ensure the sample is in focus and has "
                "clear features."
            )
        return float(backlash), points_to_remove

    def _fit_csm_matrix(
        self,
        all_stage_displacements: list[np.ndarray],
        all_cam_displacements: list[np.ndarray],
    ) -> np.ndarray:
        """Fit the camera stage mapping matrix from all the displacements recorded.

        :param all_stage_displacements: A list of nx3 matrixies with the camera
            displacements for each run.
        :param all_cam_displacements: A list of nx3 matrixies with the stage
            displacements for each run.

        :return: The camera stage mapping matrix.
        """
        # Create a matrix of the positions
        stage_mat = np.concatenate(all_stage_displacements)[:, :2]
        cam_mat = np.concatenate(all_cam_displacements)

        csm_matrix, _res, _rank, _s = np.linalg.lstsq(cam_mat, stage_mat)

        residuals = np.linalg.norm(np.dot(cam_mat, csm_matrix) - stage_mat)
        steps = np.linalg.norm(stage_mat, axis=1)
        fractional_error = residuals / np.sqrt(len(stage_mat)) / np.mean(steps)
        if fractional_error > 0.1:
            # Add a warning that is followed instantly by the error. This should
            # never show in the summary log view but will show to more technical
            # users expanding the full log.
            self.logger.warning(
                "The relative RMS error of fitting the camera stage matrix was "
                f"{fractional_error:.4f}"
            )
            raise MappingError(
                "Stage motion was not as expected, check the stage is moving freely."
            )
        return csm_matrix

    image_to_stage_displacement_matrix: Optional[list[list[float]]] = lt.setting(
        default=None, readonly=True
    )
    """A 2x2 matrix that converts displacement in image coordinates to stage coordinates.

    Note that this matrix is defined using "matrix coordinates", i.e. image coordinates
    are (y,x). To make a move in image coordinates (``dy``, ``dx``), where ``dx`` is
    horizontal, you should move the stage by:

    .. code-block:: python

        stage_disp = np.dot(
            np.array([dy,dx]),
            np.array(image_to_stage_displacement_matrix),
        )

    """

    image_resolution: Optional[tuple[float, float]] = lt.setting(
        default=None, readonly=True
    )
    """The image size used to calibrate the image_to_stage_displacement_matrix."""

    last_cal_timestamp: Optional[str] = lt.setting(default=None, readonly=True)
    """The timestamp of the last calibration. This can be used to get the data file."""

    @lt.property
    def last_calibration(self) -> Optional[dict]:
        """The most recent CSM calibration."""
        if self.last_cal_timestamp is None:
            return None

        filename = f"csm_data_{self.last_cal_timestamp}.json"
        datafile_path = os.path.join(self.data_dir, filename)
        if not os.path.isfile(datafile_path):
            return None

        with open(datafile_path, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)

    @lt.property
    def calibration_required(self) -> bool:
        """Whether the camera stage mapper needs calibrating."""
        return self.image_to_stage_displacement_matrix is None

    def assert_calibration(self) -> list[list[float]]:
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
