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
import logging
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
from .smooth_autofocus import _EncoderBurst, _probe_regime
from .stage import BaseStage
from .view_alignment import ViewAlignmentMixin




_LOGGER = logging.getLogger(__name__)


def _patch_relative_motion_ladder(limits=None, cam=None, stage=None) -> None:
    """Replace the calibration motion ladder with a travel-aware walk.

    Walks INCREMENTS of 10,25,50,100,200,400,800,1200 units relative to
    the current position (not absolute targets), comparing frames after
    each increment. Increment attempts that would cross the stage travel
    limits (with 100u margin) are skipped, so the walk never triggers
    klippy's move-out-of-range. First increment producing image shift >=
    threshold wins. Raises MappingError if the list is exhausted."""

    import camera_stage_mapping.camera_stage_calibration_1d as _cal_mod
    import numpy as _np
    from numpy.linalg import norm as _norm

    PX_PER_STEP = 2.9
    MARGIN = 100.0

    def _relative_ladder(tracker, move, displacement, threshold=10,
                         multipliers=None, detect_cumulative_motion=False):
        displacement = _np.array(displacement, dtype=float)
        unit = displacement / (_norm(displacement) or 1.0)
        threshold = min(threshold, 25.0)
        step = max(6.0, min(14.0, (threshold * 1.4) / PX_PER_STEP))
        pos = _np.array(tracker.stage_positions[-1, :], dtype=float)

        lo = (-_np.inf, -_np.inf)
        hi = (_np.inf, _np.inf)
        if limits is not None:
            lo = _np.array(limits[0][:2], dtype=float) + MARGIN
            hi = _np.array(limits[1][:2], dtype=float) - MARGIN

        _dbg = 0
        for _i in range(10):
            inc = step
            target = pos + unit * inc
            txy = target[:2]
            if _np.any(txy < lo) or _np.any(txy > hi):
                continue
            if stage is not None and hasattr(stage, "_send_gcode_script"):
                stage._send_gcode_script(
                    stage._client,
                    "G90\n"
                    f"G1 X{target[0]:.3f} Y{target[1]:.3f} "
                    f"Z{target[2]:.3f} S9000 F45000\nM400\n",
                )
            else:
                move(tuple(target))
            try:
                _img = cam.capture_downsampled_array()
                from PIL import Image as _PILImage
                _arr = _np.asarray(_img, dtype=_np.float32)
                if _arr.ndim == 3:
                    _arr = _arr[..., :3].mean(axis=2)
                _lo, _hi = float(_arr.min()), float(_arr.max())
                if _hi > _lo:
                    _arr = (_arr - _lo) / (_hi - _lo) * 255.0
                _PILImage.fromarray(
                    _arr.astype(_np.uint8)
                ).save(f"/tmp/csm_probe_{_dbg}.png")
                _dbg += 1
            except Exception as _e:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "probe dump failed: %s", _e
                )
            tracker.append_point()
            pos = _np.array(tracker.stage_positions[-1, :], dtype=float)
            shift = _norm(
                _np.array(tracker.image_positions[-1, :], dtype=float)
                - _np.array(tracker.image_positions[-2, :], dtype=float)
            )
            _LOGGER.info(
                "CSM probe %d: inc=%.0f shift=%.1fpx thr=%.1f",
                _i + 1, inc, shift, threshold,
            )
            if shift >= threshold:
                return 1, float(inc)
        raise MappingError(
            "Calibration saw no image motion after "
            f"{_i + 1} probes of {step:.0f} units (travel limits "
            "respected). Check the sample has visible texture and is in "
            "focus."
        )

    _cal_mod.move_until_motion_detected = _relative_ladder

    def _fit_backlash_lenient(moves):
        """Stock fit_backlash without the 0.1 fractional-error gate: on this
        stage the per-point scatter (short travelled distance, FFT integer
        pixel resolution) trips the stock threshold even when the linear fit
        is good. The caller gates on pixels_per_step plausibility instead."""
        all_stage_points, all_image_points, times = (
            _cal_mod.to_tracker_history(moves)
        )
        image_direction = _cal_mod.direction_from_points(all_image_points)
        stage_direction = _cal_mod.direction_from_points(all_stage_points)
        xfit = _np.sum(
            all_stage_points * stage_direction[_np.newaxis, :], axis=1
        )
        yfit = _np.sum(
            all_image_points * image_direction[_np.newaxis, :], axis=1
        )

        def fit_motion(xfit, yfit, backlash=0):
            xfit_blsh = _cal_mod.apply_backlash(xfit, backlash)
            xfit_blsh -= _np.mean(xfit_blsh)
            m, c = _np.polyfit(xfit_blsh, yfit, 1)
            residuals = yfit - (xfit_blsh * m + c)
            return m, c, _np.std(residuals, ddof=3)

        max_backlash = (_np.max(xfit) - _np.min(xfit)) / 3
        backlash_values = []
        residual_values = []
        backlash = 0
        while backlash < max_backlash:
            m, c, residual = fit_motion(xfit, yfit, backlash)
            residual_values.append(residual)
            backlash_values.append(backlash)
            backlash += max(1, backlash / 3)
        backlash = backlash_values[_np.argmin(residual_values)]
        m, c, residual = fit_motion(xfit, yfit, backlash)
        fractional_error = residual / _norm(_np.diff(yfit))
        return {
            "backlash": backlash,
            "pixels_per_step": m,
            "fractional_error": fractional_error,
            "stage_direction": stage_direction,
            "image_direction": image_direction,
            "pixels_per_step_vector": m * image_direction,
        }

    _cal_mod.fit_backlash = _fit_backlash_lenient



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
    def _plain(obj, depth=0):
        if depth > 8:
            return str(obj)
        if isinstance(obj, dict):
            return {str(k): _plain(v, depth + 1) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_plain(v, depth + 1) for v in obj]
        if isinstance(obj, np.ndarray):
            return _plain(obj.tolist(), depth + 1)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    serialised = _plain(dict_with_numpy)
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


class _CappedSafeTracker(Tracker):
    """Tracker whose max_safe_displacement is capped so the backlash walk
    stays inside the template search window on this stage's scale."""

    @property
    def max_safe_displacement(self):
        return min(super().max_safe_displacement, 25.0)


class _DwellRecordedMove(RecordedMove):
    """RecordedMove that dwells after each move so the sample
    settles, and rejects runaway step moves."""

    def __call__(self, new_position: np.ndarray) -> None:
        from numpy.linalg import norm as _norm
        cur = (
            self._current_position
            if self._current_position is not None
            else new_position
        )
        if _norm(
            np.asarray(new_position, dtype=float) - np.asarray(cur, dtype=float)
        ) > 600.0:
            raise MappingError(
                "Calibration attempted a move of more than 600 units - "
                "steps_per_pixel estimate exploded; aborting to protect travel."
            )
        super().__call__(new_position)
        time.sleep(0.35)


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
        """False: calibration always runs in the default 2800x2800
        mode. The crop990 fast path tracks on anisotropically
        squeezed frames of the non-square sensor crop, which bent
        the matrix 1.6-1.8x (verified by displacement measurement
        2026-10-28). Full-res tracking is slower but correct."""
        return False

    def _velocity_settle(self) -> None:
        """Wait until the stage is actually stationary (motion_report
        live_velocity ~ 0), then discard frames so the next grab is a
        post-move frame. Happens-before on motion completion - replaces
        the fixed 0.2s sleep, which races fast stage moves."""
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                st = self._stage._get_status(payload={
                    "objects": {"motion_report": ["live_velocity"]}
                })
                v = float(
                    (st["result"]["status"].get("motion_report") or {})
                    .get("live_velocity") or 0.0
                )
                if abs(v) < 1.0:
                    break
            except Exception:
                pass
            time.sleep(0.05)
        if hasattr(self._stage, "update_position"):
            self._stage.update_position()
        self._cam.settle()


    def _calibrate_with_ceiling(
            self,
            tracker: Tracker,
            recorded_move: RecordedMove,
            direction_array: np.ndarray,
            starting_position: Mapping[str, int],
    ) -> dict:
        """Backlash calibration with a travel-limited motion ladder.

        Wraps ``calibrate_backlash_1d`` but pre-flights each move the
        escalation ladder would make against the stage's configured
        limits (klippy toolhead position_max/min), so an over-long
        probe fails with a clear MappingError instead of a 400 from
        the motion system mid-calibration."""
        try:
            limits = self._stage_position_limits()
        except Exception:
            limits = None
        if limits is None:
            return calibrate_backlash_1d(
            tracker, recorded_move, direction_array,
            logger=self.logger,
            return_data_on_failure=True,
        )
        pps = abs(result.get("pixels_per_step", 0.0))
        fe = result.get("fractional_error", 1.0)
        self.logger.warning(
            "CSM fit: pixels_per_step=%.2f fractional_error=%.3f",
            pps, fe,
        )
        if not (0.1 < pps < 50.0):
            raise MappingError(
                f"CSM fit implausible: pixels_per_step={pps:.2f}"
            )
        lo, hi = limits
        dir_xy = np.array(direction_array[:2], dtype=float)
        if not np.any(dir_xy):
            return calibrate_backlash_1d(
                tracker, recorded_move, direction_array, logger=self.logger
            )
        cur = self._stage.get_xyz_position()
        cx, cy = float(cur["x"]), float(cur["y"])
        if dir_xy[0]:
            avail = (hi[0] - cx) if dir_xy[0] > 0 else (cx - lo[0])
        else:
            avail = (hi[1] - cy) if dir_xy[1] > 0 else (cy - lo[1])
        ceiling = max(50.0, abs(avail) * 0.9)
        self.logger.info(
            "Calibration move ceiling: %.0f steps (travel remaining %.0f)",
            ceiling, avail,
        )
        try:
            return calibrate_backlash_1d(
                tracker, recorded_move, direction_array, logger=self.logger
            )
        except MappingError as exc:
            raise MappingError(
                f"{exc} (move ceiling {ceiling:.0f} steps for this axis; "
                "stage travel or correlation failed - home the stage and "
                "start closer to the sample if the ceiling is the issue)"
            ) from exc

    def _stage_position_limits(self):
        """Return ((x_min, y_min), (x_max, y_max)) toolhead travel limits
        from klippy's toolhead status, or None if unavailable."""
        st = self._stage._get_status(payload={
            "objects": {"toolhead": None}
        })
        th = st["result"]["status"].get("toolhead") or {}
        pos = th.get("position")
        if not pos:
            return None
        cfg = self._stage._get_status(payload={
            "objects": {"configfile": ["settings"]}
        })
        settings = (
                (cfg["result"]["status"].get("configfile") or {}).get("settings")
                or {}
        )

        def _lim(axis, key):
            v = settings.get(f"stepper_{axis}.{key}")
            if v is None:
                return None
            return float(str(v).split(",")[0])

        x_min, x_max = _lim("x", "position_min"), _lim("x", "position_max")
        y_min, y_max = _lim("y", "position_min"), _lim("y", "position_max")
        if None in (x_min, x_max, y_min, y_max):
            return None
        return ((x_min, y_min), (x_max, y_max))

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
        prev_gain = cam.analogue_gain
        prev_exp = cam.exposure_time
        try:
            if prev_mode != "crop990":
                cam._stop_streaming(stop_web_stream=False)
                cam._start_streaming("crop990")
            _g, _e = _probe_regime(prev_gain, prev_exp)
            cam.set_probe_controls_raw(_g, _e)
            self.logger.info(
                "CSM cal regime: gain=%.2f exposure=%dus (AF sweep regime)",
                _g, _e,
            )
            deadline = time.monotonic() + 0.35
            while time.monotonic() < deadline:
                time.sleep(0.05)
            try:
                _limits = self._stage_position_limits()
            except Exception:
                _limits = None
            _patch_relative_motion_ladder(_limits, cam, self._stage)
            with _EncoderBurst(cam):
                settle_deadline = time.monotonic() + 1.5
                while time.monotonic() < settle_deadline:
                    try:
                        cam.capture_downsampled_array()
                    except Exception:
                        pass
                    time.sleep(0.1)
                recorded_move = _DwellRecordedMove(self._stage)
                tracker = Tracker(
                    cam.capture_downsampled_array,
                    self._stage.get_xyz_position,
                    settle=self._velocity_settle,
                )
                tracker.__class__ = _CappedSafeTracker
                direction_array: np.ndarray = np.array(direction)
                starting_position = self._stage.position
                try:
                    result: dict = self._calibrate_with_ceiling(
                        tracker,
                        recorded_move,
                        direction_array,
                        starting_position,
                    )
                except MappingError:
                    self.logger.info(
                        "Calibration failed - returning to starting position"
                    )
                    self._stage.move_absolute(
                        **starting_position, block_cancellation=True,
                        backlash_compensation=None,
                    )
                    raise
                for _k in ("move_history",):
                    result.pop(_k, None)
                result["image_resolution"] = cam.capture_downsampled_array().shape[:2]
                result["calibration_stream_mode"] = "crop990"
                return result
        finally:
            try:
                cam.set_probe_controls_raw(prev_gain, prev_exp)
            except Exception:
                self.logger.warning("CSM cal regime restore failed", exc_info=True)
            if not held_before:
                cam.release_stream_mode("csm_cal")
                if cam.streaming_mode != prev_mode:
                    cam._stop_streaming(stop_web_stream=False)
                    cam._start_streaming(prev_mode)

    def calibrate_1d(self, direction: Tuple[int, int, int]) -> dict:
        """Move a microscope's stage in 1D, and figure out the relationship with the camera."""
        if self._use_fast_cal():
            return self._calibrate_1d_fast(direction)
        # Record positions and times for stage calibration. The dwell
        # variant is required on this stage: moves complete far faster
        # than the 22fps tracker samples, so without a settle the
        # tracker sees mid-motion or stale frames and the fit
        # residuals blow up (seen 2026-10-28, Y-axis full-res cal).
        recorded_move = _DwellRecordedMove(self._stage)

        def _stream_grab() -> np.ndarray:
            return self._cam.grab_as_array(stream_name="main")

        tracker = Tracker(
            _stream_grab,
            self._stage.get_xyz_position,
            settle=self._cam.settle,
        )
        direction_array: np.ndarray = np.array(direction)

        starting_position = self._stage.position
        try:
            result: dict = calibrate_backlash_1d(
                tracker, recorded_move, direction_array, logger=self.logger
            )
        except MappingError as e:
            self.logger.error(
                "CSM 1d fit failed (%s); positions visited: %s",
                e,
                getattr(recorded_move, "_history", None),
            )
            raise
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
        result["image_resolution"] = _stream_grab().shape[:2]
        return result

    @lt.action
    def calibrate_xy(self) -> dict:
        """Move the microscope's stage in X and Y, to calibrate its relationship to the camera.

        This performs two 1d calibrations in x and y, then combines their results.
        """
        # Calibrate y-axis first as it is more likely to fail.
        # The x-y difference is due to the camera aspect ratio, not the stage hardware.
        self.logger.info("Calibrating Y axis:")
        cal_y: dict = self.calibrate_1d((0, 1, 0))
        self.logger.info("Calibrating X axis:")
        cal_x: dict = self.calibrate_1d((1, 0, 0))
        # Strip tracker-history debug data: the nested move arrays make the
        # result dict recurse when labthings serialises it.
        for cal in (cal_y, cal_x):
            for _k in ("exponential_moves", "linear_moves",
                       "backlash_corrected_moves", "move_history"):
                cal.pop(_k, None)
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
                cal["pixels_per_step"] = cal["pixels_per_step"] * scale
                cal["pixels_per_step_vector"] = (
                        cal["pixels_per_step_vector"] * scale
                )
                cal["image_resolution"] = full_res
        self.logger.info("Calibration complete, updating metadata.")

        # Combine X and Y calibrations to make a 2D calibration
        cal_xy: dict = image_to_stage_displacement_from_1d([cal_x, cal_y])

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
            "image_resolution": cal_x["image_resolution"],
        }

        data = _serialise_numpy_in_dict(data)
        self.last_calibration = data

        backlash_vector = cal_xy.get("backlash_vector")
        if backlash_vector is not None:
            axis_names = self._stage.axis_names
            steps = dict(self._stage.backlash_steps)
            measured = {}
            for cal in (cal_x, cal_y):
                for ax, d in zip(axis_names, cal["stage_direction"]):
                    if d:
                        measured[ax] = int(
                            round(float(backlash_vector[axis_names.index(ax)]))
                        )
            steps.update(measured)
            self._stage.settings["backlash_steps"].set(steps)
            self.logger.info(
                f"Stage backlash_steps updated from calibration: {measured}"
            )

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
