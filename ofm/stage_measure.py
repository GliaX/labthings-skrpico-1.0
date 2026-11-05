"""File contains functions used to measure the stage against the camera.

This includes:

* Finding the centre of motion
* Measuring the range of motion
"""

import json
import labthings_fastapi as lt
import math
import numpy as np
import os
import time
from copy import copy
from datetime import datetime
from threading import Lock
from typing import Any, Literal, Mapping, Optional, overload

from camera_stage_mapping import fft_image_tracking
# Things
from openflexure_microscope_server.things import OFMThing
from .autofocus import AutofocusThing
from .camera import BaseCamera
from .camera_stage_mapping import CameraStageMapper
from .stage import BaseStage

## Size of movement in percentage of field of view
SMALL_STEP = 10
MEDIUM_STEP = 25
BIG_STEP = 50

PARASITIC_MOTION_TOL = 0.1
DETECT_MOTION_TOL = 0.65
PROBE_SETTLE_S = 0.4

STOCK_LEG_LENGTH_MM = 60.0
TURNING_POINT_MIN_Z_SPAN = 25.0


class CamStageTracker:
    """Class for tracking camera and stage movement together.

    The attributes storing data are:

    * ``stage_coords`` A list of the the stage coordinates at each measurement location
    * ``displacements`` A list of the displacement calculated after each movement
    """

    def __init__(self) -> None:
        """Define useful data tracked throughout test."""
        self.stage_coords: list[Mapping[str, int]] = []
        self.offsets: list[Mapping[str, float]] = []

    def record_movement(
            self, current_pos: Mapping[str, int], offset: Mapping[str, float]
    ) -> None:
        """Record the current position and the measured offset of the last move."""
        self.stage_coords.append(current_pos)
        self.offsets.append(offset)

    @property
    def final_position(self) -> Mapping[str, int]:
        """The last stage coordinate recorded."""
        return copy(self.stage_coords[-1])

    def fit_axis(self, axis: Literal["x", "y"]) -> np.poly1d:
        """Quadratic fit stage z against x or y and return poly1d of fit.

        Note that this considers only one of x and y, and ignores the other coordinate. If
        this function is used on moves where both x and y are changing, it will give misleading
        results.
        """
        lateral_positions = [i[axis] for i in self.stage_coords]
        z_positions = [i["z"] for i in self.stage_coords]
        fit_params = np.polyfit(lateral_positions, z_positions, 2)
        return np.poly1d(fit_params)

    def predict_z_displacement(
            self,
            axis: Literal["x", "y"],
            stage_movement: Mapping[str, int],
            stage_position: Mapping[str, int],
    ) -> int:
        """Predict the z-displacement needed for a given movement.

        :param axis: The axis which is being measured. This must be 'x' or 'y'.
        :param stage_movement: The movement to be performed in stage coordinates.
        :param stage_position: The current stage position in stage coordinates.
        :return: The predicted relative z displacement needed to stay in focus.
        """
        fit_func = self.fit_axis(axis)
        z_dest = fit_func(stage_position[axis] + stage_movement[axis])

        return int(z_dest - stage_position["z"])

    def find_turning_point(self, axis: Literal["x", "y"]) -> Mapping[str, int]:
        """Find the turing point from the recorded coordinates."""
        fit_func = self.fit_axis(axis)
        turning_loc = fit_func.deriv().roots[0]
        turning_z = fit_func(turning_loc)

        # As we only move in 1 direction, pull the other axis from the coords.
        other_axis = "x" if axis == "y" else "y"
        other_coord = self.stage_coords[-1][other_axis]
        return {axis: int(turning_loc), other_axis: other_coord, "z": int(turning_z)}


class ParasiticMotionError(lt.exceptions.InvocationError):
    """Custom exception raised when parasitic motion is detected.

    Parasitic motion is when motion in the direction not being measured
    is too high.
    """


class StageMeasurementThing(OFMThing):
    """A class used to measure the range of motion of the stage in X and Y."""

    _class_settings = {"validate_properties_on_set": True}

    _autofocus: AutofocusThing = lt.thing_slot()
    _cam: BaseCamera = lt.thing_slot()
    _csm: CameraStageMapper = lt.thing_slot()
    _stage: BaseStage = lt.thing_slot()

    calibrated_range: Optional[tuple[int, int]] = lt.setting(
        default=None, readonly=True
    )

    leg_length_mm: Optional[float] = lt.setting(default=None)

    def __init__(self, thing_server_interface: lt.ThingServerInterface) -> None:
        """Initialise and create the lock."""
        super().__init__(thing_server_interface)
        self._lock = Lock()
        self._stream_resolution: Optional[tuple[int, int]] = None
        self._cam_stage_data = CamStageTracker()
        self._probe_moves = 5

    def _leg_scale(self) -> float:
        """Probe-span scale factor from the configured leg length.

        The parasitic z drop scales as h/(2x^2)-equivalent: doubling the
        leg length halves the curvature signal per unit travel, so probe
        spans are scaled by sqrt(leg/60) to restore the stock per-probe
        z signal. Stock (None or 60mm) returns exactly 1.0."""
        leg = self.leg_length_mm
        if leg is None:
            return 1.0
        try:
            s = math.sqrt(float(leg) / STOCK_LEG_LENGTH_MM)
        except (TypeError, ValueError):
            return 1.0
        return max(1.0, min(s, 3.0))

    @lt.action
    def perform_rom_test(self) -> Mapping[str, Any]:
        """Measures the range of motion of the stage across the x and y axes.

        The range of motion is measured by first taking 5 'medium' sized steps which
        gives enough positions to predict future z positions. Next, one 'big' step is
        taken followed by 3 'small' steps to test that the stage is still moving as
        expected and has not reached the edge. Once the edge has been found and to
        account for the possibility that the edge was reached during a "big" step, the
        stage is moved in a sequence of steps in the opposite direction until motion is
        detected. This position is taken as the final position.

        Throughout the test, parasitic motion (motion in the axis not being measured)
        is tracked and an error is raised if it exceeds a minimum amount. Currently
        this is 10% of the expected motion in the measured axis.

        :return: Results dictionary separated into keys of each axis and direction.
        """
        got_lock = self._lock.acquire(blocking=False)
        if not got_lock:
            raise RuntimeError("Trying to run ROM test when a test is already running.")

        try:
            self.logger.info(
                "Using the stage to measure the Range of Motion. "
                "Please ensure you are using a sample that covers the whole range of "
                "motion. This should be approximately 12 x 12 mm."
            )
            start_time = time.time()

            self._set_stream_resolution()

            # The total range for the two axes under test
            step_range = []

            # Strictly typed definitions to iterate over for MyPys sake.
            axes: tuple[Literal["x"], Literal["y"]] = ("x", "y")
            directions: tuple[Literal[1], Literal[-1]] = (1, -1)

            stage_dic = {}
            cor_dic = {}

            for axis in axes:
                # The final position of the axis under test in the given direction
                axis_limits = []
                for axis_dir in directions:
                    # Create a new tracker at start of measurement.
                    self._cam_stage_data = CamStageTracker()
                    self._move_until_edge(axis=axis, direction=axis_dir)
                    # Append final position from tracker
                    axis_limits.append(self._cam_stage_data.final_position[axis])
                    axis_string = "positive" if axis_dir == 1 else "negative"
                    stage_dic[f"{axis}_{axis_string}"] = (
                        self._cam_stage_data.stage_coords
                    )
                    cor_dic[f"{axis}_{axis_string}"] = self._cam_stage_data.offsets
                # Calculate step range.
                step_range.append(abs(axis_limits[0] - axis_limits[1]))

            total_time = time.time() - start_time
            self.logger.info(
                f"Range of motion is {step_range[0]} x {step_range[1]} steps"
            )
            data = {
                "calibrated_range": [step_range[0], step_range[1]],
                "stage_coords": stage_dic,
                "correlations": cor_dic,
                "csm_matrix": self._csm.image_to_stage_displacement_matrix,
                "time": total_time,
            }

            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
            datafile_path = os.path.join(self.data_dir, f"rom_data_{timestamp}.json")
            with open(datafile_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

        finally:
            self._lock.release()
        return {
            "Time": total_time,
            "CSM Matrix": self._csm.image_to_stage_displacement_matrix,
            "Step Range": step_range,
        }

    @lt.action
    def perform_recentre(self) -> None:
        """Measures the curvature of the motion to centre x and y axes."""
        got_lock = self._lock.acquire(blocking=False)
        if not got_lock:
            raise RuntimeError("Trying to run recentre when a test is already running.")

        try:
            self.logger.info("Recentring the stage.")
            # Suppress the AF cool-down while recentre runs: it fires ~3s
            # after the last autofocus and stops/reconfigures the camera
            # mid-walk, deadlocking subsequent grabs (2026-11-01 y-axis).
            self._autofocus._cancel_cooler()
            type(self._autofocus)._COOL_SUPPRESS = True

            self._set_stream_resolution()

            # Strictly typed definitions to iterate over for MyPys sake.
            axes: tuple[Literal["x"], Literal["y"]] = ("x", "y")

            try:
                for axis in axes:
                    self._recentre_axis(axis)
            finally:
                type(self._autofocus)._COOL_SUPPRESS = False

            centre = self._stage.position
            centre_str = f"({centre['x']}, {centre['y']})"
            self.logger.info(f"Centre is estimated at {centre_str}.")
            # Set the central position to (0,0,0)
            self._stage.set_zero_position()
            self.logger.info("Position reset to (0, 0, 0).")

        finally:
            self._lock.release()

    def _set_stream_resolution(self) -> None:
        """Set the self._stream_resolution attribute by reading camera."""
        stream_shape = self._cam.grab_as_array().shape
        # Swap axes as numpy is [y, x]
        self._stream_resolution = (stream_shape[1], stream_shape[0])

    def _move_until_edge(
            self, axis: Literal["x", "y"], direction: Literal[1, -1]
    ) -> None:
        """Move in one direction until movement per step decreases significantly.

        This should move until the edge of the stage. Once the edge is reached there
        will be some movement as there is no hard stop, but it will reduce
        significantly.

        :param axis: The axis which is being measured. This must be 'x' or 'y'.
        :param direction: The direction which is being measured. This must be 1 or -1.
        :return: Results dictionary containing stage positions,
           correlations and the final position.
        """
        # Start by performing an autofocus and recording starting position
        self._autofocus.looping_autofocus(dz=1000)
        starting_position = self._stage.position
        self._cam_stage_data.stage_coords.append(starting_position)

        try:
            dir_str = "positive" if direction == 1 else "negative"
            self.logger.info(f"Beginning the {axis}-axis in the {dir_str} direction")

            self.logger.info("Moving the stage in 5 medium sized steps.")
            self._moves_for_z_prediction(axis=axis, direction=direction)

            still_moving = True
            # Loop taking 1 big step followed by 3 small steps to check if the
            # stage is moving as expected or has reached end of range of motion.
            # Stop once end of range of motion is detected.
            while still_moving:
                self._big_z_corrected_movement(axis=axis, direction=direction)
                # Autofocus and record position
                self._autofocus.looping_autofocus(dz=800)
                self._cam_stage_data.stage_coords.append(self._stage.position)

                # Perform small moves to check stage is still moving.
                still_moving = self._stage_still_moves(axis=axis, direction=direction)

            # Stage may have crashed during a large move. Move stage back until motion
            # is detected
            self.logger.info("Moving stage back until motion is detected")
            self._move_back_until_motion_detected(axis=axis, direction=direction)

            # Replace final position.
            self._cam_stage_data.stage_coords[-1] = self._stage.position

        finally:
            self._stage.move_absolute(
                **starting_position, block_cancellation=True, backlash_compensation=None
            )

    def _recentre_axis(self, axis: Literal["x", "y"]) -> None:
        """Recentre a single axis.

        :param axis: The axis to recentre
        """
        self.logger.info(f"Finding centre in {axis}.")
        # A new tracker for this axis.
        self._cam_stage_data = CamStageTracker()
        # Direction is in image coords, initial assumption is that centre is at (0,0).
        direction = self._img_dir_from_stage_coords({"x": 0, "y": 0, "z": 0}, axis)
        direction = self._verify_probe_direction(axis, direction)

        i = 0
        while True:
            i += 1
            s = self._leg_scale()
            self._probe_moves = (
                5 if s == 1.0 else max(5, min(9, int(round(5 * s))))
            )
            # Find z then make a number of moves (autofocussing and logging position)
            # 5 moves initially (2 subsequently).
            self._autofocus.looping_autofocus(dz=1000)
            self._cam_stage_data.stage_coords.append(self._stage.position)
            n_moves = self._probe_moves if i == 1 else max(
                2, int(round(self._probe_moves / 2))
            )
            self._moves_for_z_prediction(
                axis=axis,
                direction=direction,
                n_moves=n_moves,
            )

            centred, direction = self._recentre_decision(axis)
            if centred:
                break

            # If still going at iteration 9 exit
            if i > 9:
                raise RuntimeError(f"Couldn't find centre of {axis}-axis")

            # Make a big z-corrected move towards estimate of centre.
            self._big_z_corrected_movement(axis, direction)

    def _verify_probe_direction(
            self, axis: Literal["x", "y"], direction: Literal[1, -1]
    ) -> Literal[1, -1]:
        """Check the initial walk direction against live camera evidence.

        Stage-coordinate beliefs about where the centre lies go stale across
        restarts and power cycles. Take one probe step and compare the
        measured feature shift with the shift the commanded direction should
        produce (same frame conventions as the parasitic check); if the
        feature moved against the commanded direction it is receding from
        the centre, so flip."""
        movement = self._movement_in_img_coords(
            fov_perc=MEDIUM_STEP * self._leg_scale(),
            axis=axis,
            direction=direction,
        )
        expected = self._measured_shift_for(movement)
        time.sleep(PROBE_SETTLE_S)
        self._cam.discard_frames()
        self.logger.info("Direction check: grabbing before frame.")
        before = self._cam.grab_as_array()
        self.logger.info("Direction check: probing a step.")
        self._stage_move_from_img(movement)
        time.sleep(PROBE_SETTLE_S)
        self._cam.discard_frames()
        self.logger.info("Direction check: measuring offset.")
        try:
            measured = self._offset_from(before)
        except RuntimeError:
            self.logger.info(
                "Direction check: no sample detected; keeping direction %d.",
                direction,
            )
            self._stage_move_from_img({k: -v for k, v in movement.items()})
            return direction
        agreement = expected["x"] * measured["x"] + expected["y"] * measured["y"]
        magnitude = (measured["x"] ** 2 + measured["y"] ** 2) ** 0.5
        self.logger.info(
            "Direction check: expected %s, measured %s (|%.0f|px, dot %.0f).",
            {k: round(v) for k, v in expected.items()},
            {k: round(v) for k, v in measured.items()},
            magnitude,
            agreement,
        )
        flipped = -1 if direction == 1 else 1
        # Undo the probe step either way; walk in the verified direction.
        self._stage_move_from_img({k: -v for k, v in movement.items()})
        if magnitude < 10.0:
            self.logger.info(
                "Feature barely moved; keeping stage-coordinate direction %d.",
                direction,
            )
            return direction
        if agreement < 0:
            self.logger.info(
                "Feature moved against commanded direction; flipping to %d.",
                flipped,
            )
            return flipped
        return direction

    def _measured_shift_for(
            self, movement: Mapping[str, float]
    ) -> Mapping[str, float]:
        """Predicted camera-frame feature shift for a preview-space command.

        Applies the same chain as _img_to_stage_rel (click convention) and
        maps the stage move back to image pixels through the matrix
        transpose, in the screen frame used by the offset dictionaries."""
        matrix = np.array(self._csm.image_to_stage_displacement_matrix)
        x2, y2 = -movement["y"], movement["x"]
        stage = np.dot(np.array([y2, x2]), matrix)
        inv = np.linalg.inv(matrix)
        cam = inv @ np.array([stage[0], stage[1]])
        return {"x": float(cam[1]), "y": -float(cam[0])}

    def _recentre_decision(
            self, axis: Literal["x", "y"]
    ) -> tuple[bool, Literal[1, -1]]:
        """Decide what to do next during recentreing.

        The algorithm here:

        * Estimate turning point location
        * Check if distance to point is further than a "BIG_STEP"
        * If smaller, move to estimated centre, and return that it is centred
        * If larger, return that it is not centred, and the direction to move in.

        :param axis: The axis to recentre
        :return: A tuple. The first value is True for "the stage is now centred" and
            False for "the stage is not centred". The second value is the direction to
            move in, this only has meaning if the first value is False.
        """
        estimate = self._cam_stage_data.find_turning_point(axis=axis)
        img_perc = self._distance_in_img_percentage(estimate, axis)

        if self.leg_length_mm is not None:
            zs = [c["z"] for c in self._cam_stage_data.stage_coords]
            z_span = abs(max(zs) - min(zs))
            # Gate scales with the probe span: stock probes are 50% of the
            # frame and the gate constant was tuned for that span.
            gate = TURNING_POINT_MIN_Z_SPAN * (
                MEDIUM_STEP * self._leg_scale() / 50.0
            )
            if z_span < gate:
                self.logger.info(
                    "Turning-point estimate rejected: probe z-span "
                    "%.0f < %.0f (weak curvature signal).",
                    z_span,
                    gate,
                )
                # The curvature fit is unusable, so the stage-coordinate
                # estimate is worthless. Decide from the recorded feature
                # offsets instead: if the feature drifted far from the frame
                # centre, the centre lies back the way we came.
                offs = self._cam_stage_data.offsets
                if offs:
                    cum = sum(o[axis] for o in offs if axis in o)
                    flipped = -1 if cum > 0 else 1
                    self.logger.info(
                        "Weak-curvature direction from offsets: cum %.0f -> "
                        "direction %d.",
                        cum,
                        flipped,
                    )
                    return False, flipped
                return False, 1

        # if the distance is less than 1 big step away then move to it and exit
        if abs(img_perc) < BIG_STEP:
            self.logger.info(f"Estimated centre of {axis}-axis is {estimate[axis]}")
            self._stage.move_absolute(
                **estimate, block_cancellation=False, backlash_compensation=None
            )
            # Note the second return, the direction, is meaningless here.
            return True, 1
        self.logger.info(
            f"Estimated centre {abs(img_perc):.0f}% of a field of view away. "
            "Moving towards centre."
        )
        # Else calculate the desired direction in image coordinates.
        direction = self._img_dir_from_stage_coords(estimate, axis)
        return False, direction

    def _img_percentage_to_img_coords(
            self, fov_perc: int, axis: Literal["x", "y"]
    ) -> float:
        """For a given image percentage and axis return the distance in img coords.

        :param fov_perc: The percentage of field of view the stage should move by.
        :param axis: The axis which is being measured. This must be 'x' or 'y'.
        :return: Distance in image coordinates (pixels)
        """
        if self._stream_resolution is None:
            raise RuntimeError(
                "Stream resolution must be set before generating movement in coords"
            )

        img_index = 0 if axis == "x" else 1
        return (fov_perc / 100) * self._stream_resolution[img_index]

    def _img_dir_from_stage_coords(
            self, target: Mapping[str, int], axis: Literal["x", "y"]
    ) -> Literal[1, -1]:
        """For a target location in stage coords, return the direction in image coordinates.

        :param target: The target poisiton in stage coordinates
        :param axis: The axis which is being measured. This must be 'x' or 'y'.
        :return: Direction to move in image coordinates.
        """
        target_im_coords = self._csm.convert_stage_to_image_coordinates(**target)
        current_loc = self._stage.position
        current_loc_im_coords = self._csm.convert_stage_to_image_coordinates(
            **current_loc
        )
        return -1 if target_im_coords[axis] < current_loc_im_coords[axis] else 1

    def _distance_in_img_percentage(
            self, target: Mapping[str, int], axis: Literal["x", "y"]
    ) -> float:
        """For a target location in stage coords return the distance in percentage of FOV.

        :param target: The target poisiton in stage coordinates
        :param axis: The axis which is being measured. This must be 'x' or 'y'.
        :return: Percentage of field of view the stage should move by.
        """
        if self._stream_resolution is None:
            raise RuntimeError(
                "Stream resolution must be set before converting coords to percentage"
            )

        current_loc = self._stage.position
        move_stage = {key: target[key] - current_loc[key] for key in target}
        move_img = self._csm.convert_stage_to_image_coordinates(**move_stage)

        img_index = 0 if axis == "x" else 1
        return (move_img[axis] / self._stream_resolution[img_index]) * 100

    def _movement_in_img_coords(
            self,
            fov_perc: int,
            axis: Literal["x", "y"],
            direction: Literal[1, -1],
    ) -> Mapping[str, float]:
        """Return the dictionary for a move in image coordinates.

        This dictionary can be passed directly to csm.move_in_image_coordinates

        :param fov_perc: The percentage of field of view the stage should move by.
        :param axis: The resolution of the stream from the camera.
        :param direction: The direction the stage moves.

        :return: The movement size in image coordinates
        """
        distance = self._img_percentage_to_img_coords(fov_perc=fov_perc, axis=axis)
        if axis == "x":
            return {"x": distance * direction, "y": 0}
        return {"x": 0, "y": distance * direction}

    def _img_to_stage_rel(self, movement: Mapping[str, float]) -> Mapping[str, int]:
        """Convert an image-px movement to stage units without moving.

        Uses the user-verified click convention: the movement dict is a
        preview-space offset, converted to camera frame as (dy, -dx),
        then applied through the matrix row product exactly like
        csm_img_to_stage. A +x command therefore drives the same stage
        motion as clicking right of centre in the webapp."""
        matrix = np.array(self._csm.image_to_stage_displacement_matrix)
        x2, y2 = -movement["y"], movement["x"]
        rel = np.dot(np.array([y2, x2]), matrix)
        return {"x": int(round(rel[0])), "y": int(round(rel[1]))}

    def _stage_move_from_img(self, movement: Mapping[str, float]) -> Mapping[str, int]:
        """Convert an image-px movement to stage units and execute it."""
        stage_move = self._img_to_stage_rel(movement)
        self._stage.move_relative(
            **stage_move, block_cancellation=False, backlash_compensation=None
        )
        return stage_move

    def _moves_for_z_prediction(
            self, axis: Literal["x", "y"], direction: Literal[1, -1], n_moves: int = 5
    ) -> None:
        """Perform medium sized moves with autofocus for z feed-forward.

        z-feed forward allows prediction of the z-position as the stage moves. For the
        feed forward calculation to work an initial number of measurements must be
        taken.

        :param direction: The direction the stage moves.
        :param axis: The axis which is being measured. This must be 'x' or 'y'.
        :param n_moves: Number of moves to make. Default is 5 which is enough for an
            initial z estimate.
        """
        movement = self._movement_in_img_coords(
            fov_perc=MEDIUM_STEP * self._leg_scale(),
            axis=axis,
            direction=direction,
        )
        for _loop in range(n_moves):
            offset = self._move_and_measure(movement=movement)

            self._cam_stage_data.record_movement(self._stage.position, offset)
            if _parasitic_motion_detected(movement, offset):
                self.logger.info(
                    "Parasitic probe %d: commanded %s, measured %s",
                    _loop + 1,
                    movement,
                    offset,
                )
                raise ParasiticMotionError(
                    "Parasitic motion detected during images to calculate z-curvature. "
                    "This may indicate you have started at the end of the range of "
                    "travel, or that camera stage mapping is poorly calibrated."
                )

    def _big_z_corrected_movement(
            self, axis: Literal["x", "y"], direction: Literal[1, -1]
    ) -> None:
        """Take one big move with feed-forward z-correction.

        The size is defined by the BIG_STEP constant.

        :param axis: The axis to move in.
        :param direction: The direction to move in.
        """
        movement = self._movement_in_img_coords(
            fov_perc=BIG_STEP * self._leg_scale(),
            axis=axis,
            direction=direction,
        )
        # Convert to stage coordinates (true convention) for z prediction
        stage_movement = self._img_to_stage_rel(movement)

        z_disp = self._cam_stage_data.predict_z_displacement(
            axis=axis,
            stage_movement=stage_movement,
            stage_position=self._stage.position,
        )
        self._stage.move_relative(z=z_disp)
        self._stage_move_from_img(movement)

    def _stage_still_moves(
            self,
            axis: Literal["x", "y"],
            direction: Literal[1, -1],
    ) -> bool:
        """Carry out 3 small moves in a given direction and axis.

        An image is taken before and after each move to check the stage has moved as
        far as it should. If the offset (calculated by cross correlation) is less than
        expected, 3 attempts are made to refocus the image to ensure that image quality
        is not causing the offset to mistakenly be reported as a low value. If the
        calculated offset is still too low then this is taken as an indication that the
        edge has been found.

        :param axis: The axis which is being measured. This must be 'x' or 'y'.
        :param direction: The direction the stage moves.
        """
        movement = self._movement_in_img_coords(
            fov_perc=SMALL_STEP, axis=axis, direction=direction
        )
        abs_min_offset = abs(movement[axis] * DETECT_MOTION_TOL)

        for _loop in range(3):
            offset = self._move_and_measure(
                movement=movement,
                # Don't autofocus initially
                perform_autofocus=False,
                # But retry focus 3 times if detected motion is too small or parasitic
                max_autofocus_repeats=3,
                abs_min_offset=abs_min_offset,
            )
            parasitic_motion = _parasitic_motion_detected(movement, offset)
            self.logger.info(f"Offset measured as {int(offset[axis])}")
            self._cam_stage_data.record_movement(self._stage.position, offset)

            if abs(offset[axis]) < abs_min_offset or parasitic_motion:
                # If the offset is below the abs min offset, or significant
                # parasitic motion is detected then the edge has been found.
                self.logger.info("Edge has been found.")
                return False
        # If we reached here the stage is still moving fine
        return True

    def _move_back_until_motion_detected(
            self,
            axis: Literal["x", "y"],
            direction: Literal[1, -1],
    ) -> None:
        """Move the stage against the direction of test unil motion is detected.

        In the case that the stage has reached the end of the motion, this method moves
        the motor in the opposite direction until reliable motion is detected.

        :param axis: The axis in which the stage is moving. This must be 'x' or 'y'.
        :param direction: The direction in which the stage was moving during the test,
            this method will move in the opposite direction.
        """
        movement = self._movement_in_img_coords(
            fov_perc=SMALL_STEP, axis=axis, direction=-direction
        )

        max_moves = int(1.5 * BIG_STEP / SMALL_STEP)
        # minimum number of pixels for motion to be detected as reliable
        motion_minimum = abs(movement[axis] * DETECT_MOTION_TOL)

        for i in range(max_moves):
            # Increment movement
            self.logger.info(f"Testing with step size {int(movement[axis])}")

            # Run an autofocus every 3 steps
            run_autofocus = i % 3 == 2
            offset = self._move_and_measure(
                movement=movement, perform_autofocus=run_autofocus
            )

            self.logger.info(f"Offset measured as {int(abs(offset[axis]))}")
            if abs(offset[axis]) > motion_minimum:
                self.logger.info("Motion detected.")
                return
        # If this point is reached then motion is never detected
        raise RuntimeError("Cannot detect motion again after reaching end of range.")

    def _move_and_measure(
            self,
            movement: Mapping[str, float],
            perform_autofocus: bool = True,
            max_autofocus_repeats: int = 0,
            abs_min_offset: float = 0.0,
    ) -> Mapping[str, float]:
        """Move the stage and measure the offset between the two positions.

        :param movement: A dictionary containing the distance to move in image coords.
        :param perform_autofocus: Set to False to disable autofocus after move.
            Default is  True
        :param max_autofocus_repeats: The number of times to repeat the focus if the
            detected (on-axis) offset is below ``abs_min_offset``. This will only work
            if ``abs_min_offset`` is also set
        :param abs_min_offset: The absolute minimum (on-axis) offset, under which the
            autofocus is repeated.
        :return: The calculated offset from cross correlation.
        """
        if max_autofocus_repeats > 0 and abs_min_offset <= 0:
            raise ValueError(
                "abs_min_offset must be positive if max_autofocus_repeats > 0."
            )

        # Take image before move
        time.sleep(PROBE_SETTLE_S)
        self._cam.discard_frames()
        before_img = self._cam.grab_as_array()
        # Move and autofocus if required
        # Move in the intended physical axis (stage-space conversion)
        stage_cmd = self._img_to_stage_rel(movement)
        self._stage.move_relative(
            **stage_cmd, block_cancellation=False, backlash_compensation=None
        )
        if perform_autofocus:
            self._autofocus.looping_autofocus(dz=800)
        # Take image and calculate offset
        time.sleep(PROBE_SETTLE_S)
        self._cam.discard_frames()
        offset = self._offset_from(before_img)

        if max_autofocus_repeats > 0:
            axis = _axis_from_movement_dict(movement)
            af_repeats = 0
            parasitic_motion = _parasitic_motion_detected(movement, offset)
            while abs(offset[axis]) < abs_min_offset or parasitic_motion:
                af_repeats += 1
                self.logger.info(
                    f"Motion not detected. Refocusing to check. Attempt {af_repeats}/3."
                )
                self._autofocus.looping_autofocus(dz=800)
                # Re-take image and calculate offset
                offset = self._offset_from(before_img)
                parasitic_motion = _parasitic_motion_detected(movement, offset)
                if af_repeats >= max_autofocus_repeats:
                    # break if the maximum number of tries are exceeded.
                    break

        return offset

    def _offset_from(self, before_img: np.ndarray) -> Mapping[str, float]:
        """Take an image and calculate the offset from an input image.

        :param before_img: The image the offset should be calculated with respect to.
        :return: The calculated offset as a dictionary in pixels
        """
        self.logger.info("Offset: checking sample detection.")
        is_sample, _bg_message = self._cam.image_is_sample()
        if not is_sample:
            raise RuntimeError(
                "No sample detected. Sample must be densely featured and cover the "
                "whole range of motion."
            )
        self.logger.info("Offset: grabbing after frame.")
        after_img = self._cam.grab_as_array()
        self.logger.info("Offset: correlating.")
        offset = fft_image_tracking.displacement_between_images(
            image_0=before_img,
            image_1=after_img,
            sigma=10,
            fractional_threshold=0.1,
            pad=True,
        )
        row, col = offset[0], offset[1]
        return {"x": float(row), "y": float(-col)}


# The number of return arguments depends on if return_other is True or False
# let MyPy know with overloads typed to Literal[False] and Literal[True].
@overload
def _axis_from_movement_dict(
        movement: Mapping[str, float], return_other: Literal[False]
) -> str: ...


@overload
def _axis_from_movement_dict(
        movement: Mapping[str, float], return_other: Literal[True]
) -> tuple[str, str]: ...


# Overload the case where return_other is not specified, this is needed
# because MyPy doesn't see that return other's default is `False` and use
# the `Literal[False]` overload.
@overload
def _axis_from_movement_dict(movement: Mapping[str, float]) -> str: ...


# Finally The function.
def _axis_from_movement_dict(
        movement: Mapping[str, float], return_other: bool = False
) -> str | tuple[str, str]:
    """Return the axis that a given movement dictionary moves in.

    For example: ``_axis_from_movement_dict({"x": 10, "y":0})`` will return ``x``.

    :param movement: The movement dictionary.
    :param return_other: If True return a tuple with the first value being the movement
        axis and the second being the other axis. Default is False
    :return: The movement axis (and optionally the other axis) as strings.
    """
    # Not exclusive or, this errors if both axes are zero or neither axes are zero.
    if not ((movement["x"] == 0) ^ (movement["y"] == 0)):
        raise ValueError("Either x or y movement should be zero, but not both.")

    if return_other:
        return ("y", "x") if movement["x"] == 0 else ("x", "y")
    return "y" if movement["x"] == 0 else "x"


def _parasitic_motion_detected(
        movement: Mapping[str, float], offset: Mapping[str, float]
) -> bool:
    """Compare desired movement to measured offset, report if parasitic motion was detected.

    :param movement: The movement dictionary in image coordinates.
    :param offset: The offset calculated from correlation.
    :return: True if too much parasitic motion is detects. False if movement is as
        expected.
    """
    movement_axis, other_axis = _axis_from_movement_dict(movement, return_other=True)
    parasitic_motion = abs(offset[other_axis])
    max_parasitic_motion = abs(movement[movement_axis] * PARASITIC_MOTION_TOL)
    return parasitic_motion > max_parasitic_motion
