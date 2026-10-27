"""Scan workflows set different ways that smart scan can behave.

This module contains the base ``ScanWorkflow`` class that all workflows should subclass,
as well as specific workflows.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import (
    Generic,
    Literal,
    Mapping,
    Optional,
    Protocol,
    TypeVar,
)

import labthings_fastapi as lt
from pydantic import BaseModel

from openflexure_microscope_server.scan_planners import (
    RegularGridPlanner,
    ScanPlanner,
    SmartSpiral,
)
from openflexure_microscope_server.stitching import (
    TARGET_STITCHING_DIMENSION,
    StitchingSettings,
)
from openflexure_microscope_server.things import RelativeDataPath
from openflexure_microscope_server.things.autofocus import (
    MAX_TEST_IMAGE_COUNT,
    MIN_TEST_IMAGE_COUNT,
    AutofocusParams,
    AutofocusThing,
    SmartStackParams,
    StackParams,
)
from openflexure_microscope_server.things.background_detect import (
    ChannelDeviationLUV,
)
from openflexure_microscope_server.things.camera import BaseCamera, CaptureParams
from openflexure_microscope_server.things.camera_stage_mapping import CameraStageMapper
from openflexure_microscope_server.things.stage import BaseStage
from openflexure_microscope_server.ui import (
    UI_ELEMENT_RESPONSE,
    Accordion,
    HeaderBlock,
    PropertyControl,
    TextBlock,
    UIElementList,
    action_button_for,
    property_control_for,
)

SettingModelType = TypeVar("SettingModelType", bound=BaseModel)


class WorkflowStartError(lt.exceptions.InvocationError):
    """The scan workflow cannot start, as the requested configuration is invalid."""


class ScanWorkflow(Generic[SettingModelType], lt.Thing):
    """A base class for all Scanworkflows.

    Scan workflows set the behaviour of a scan, including the background detection,
    scan planning, acquisition routine.
    """

    _class_settings = {"validate_properties_on_set": True}

    display_name: str = lt.property(default="Base Workflow", readonly=True)
    ui_blurb: str = lt.property(
        default="If you see this message, something is wrong.", readonly=True
    )

    _settings_model: type[SettingModelType]

    # All workflows must have a set class for scan planning
    _planner_cls: type[ScanPlanner]

    # All workflows set a save resolution
    capture_mode: str = lt.setting(default="standard")
    """A tuple of the image resolution to capture."""

    # CSM may not be set, and isn't required for a workflow. Allow for it to exist or be None
    _csm: Optional[CameraStageMapper] = lt.thing_slot()
    # Camera, stage and autofocus are all required by any scan workflow
    _cam: BaseCamera = lt.thing_slot()
    _stage: BaseStage = lt.thing_slot()
    _autofocus: AutofocusThing = lt.thing_slot()

    # The noqa statement is because scan_name is unused but is needed for equivalence
    # with other workflows that may want to validate the scan name.
    def check_before_start(self, scan_name: str) -> None:  # noqa: ARG002
        """Check before the scan starts. Throw an error if the scan shouldn't start.

        The scan_name is passed to this function to enable workflows to validate the
        scan name if needed.
        """
        if self.capture_mode not in self._cam.capture_modes:
            cam_name = type(self._cam).__name__
            raise WorkflowStartError(
                f"{cam_name} has no capure mode {self.capture_mode}"
            )

    @lt.property
    def ready(self) -> bool:
        """Whether this scanworkflow is ready to start."""
        raise NotImplementedError(
            "Each specific ScanWorkflow must implement a ready property."
        )

    def all_settings(
            self, images_dir: RelativeDataPath
    ) -> tuple[SettingModelType, Optional[StitchingSettings], tuple[int, int]]:
        """Return the scan settings and the stitching settings.

        - The specific settings for this scan workflow are returned as a Base Model of
            the type set when defining the class.
        - Stitiching settings are returned either as a StitchingSettings object or None
            is returned if it is not possible to stitch the scan.
        - The save resolution as determined by a test image.
        """
        raise NotImplementedError(
            "Each specific ScanWorkflow must implement a `all_settings` method."
        )

    def _get_save_resolution(self) -> tuple[int, int]:
        """Return the save resolution as determined by a test image."""
        # Capture an example image.
        image = self._cam._capture_image(capture_mode=self.capture_mode)
        # Check size to create a unit faction for downsampling.
        return image.size

    def pre_scan_routine(self, settings: SettingModelType) -> None:
        """Overload to set the routine that happens before each scan."""
        raise NotImplementedError(
            "Each specific ScanWorkflow must implement a pre-scan routine."
        )

    def new_scan_planner(
            self, settings: SettingModelType, position: Mapping[str, int]
    ) -> ScanPlanner:
        """Return the a new scan planner object for a scan."""
        raise NotImplementedError(
            "Each specific ScanWorkflow must implement a ``new_scan_planner`` method."
        )

    def acquisition_routine(
            self, settings: SettingModelType, xyz_pos: tuple[int, int, int]
    ) -> tuple[bool, Optional[int], int]:
        """Overload to set the acquisition routine that happens at each scan site.

        :param settings: The settings for this scan, which should be a SettingModelType
        :param xyz_pos: The current position as a tuple or 3 ints.
        :return: A tuple of whether an image was taken, the z-position for focus, and
            the number of images captured at this site.
            If failed to find focus, returns for the focus z-position.
        """
        raise NotImplementedError(
            "Each specific ScanWorkflow must implement an acquisition routine"
        )

    def _autofocus_and_capture(
            self,
            xyz_pos: tuple[int, int, int],
            dz: int,
            images_dir: RelativeDataPath,
            capture_mode: str,
    ) -> tuple[bool, Optional[int], int]:
        """Autofocus and then capture, this can be used as an acquisition routine.

        :param dz: The dz for autofocus.
        :param images_dir: The path to the directory for saving images.
        :param capture_mode: The name of the camera capture mode.

        :return: A tuple ready to pass out of acquisition routine. In this method,
            image is always taken, so first return is True, and the last is 1.

        """
        self._autofocus.fast_autofocus(dz=dz)
        focus_height = self._stage.get_xyz_position()[2]
        filename = f"img_{xyz_pos[0]}_{xyz_pos[1]}_{focus_height}.jpeg"
        self._cam.capture_and_save_to_path(
            path=images_dir.join(filename),
            capture_mode=capture_mode,
        )

        return True, focus_height, 1

    @lt.endpoint("get", "settings_ui", responses=UI_ELEMENT_RESPONSE)
    def settings_ui(self) -> UIElementList:
        """Return the UI for the workflow's settings in the scan tab."""
        raise NotImplementedError(
            "Each scan workflow must implement a settings_ui method."
        )


class RectGridSettingsModel(BaseModel):
    """Base setting model for all RectGrid workflows."""

    overlap: float
    dx: int
    dy: int
    capture_params: CaptureParams
    autofocus_params: AutofocusParams


RectGridSettingModelType = TypeVar(
    "RectGridSettingModelType", bound=RectGridSettingsModel
)


class RectGridWorkflow(
    ScanWorkflow[RectGridSettingModelType], Generic[RectGridSettingModelType]
):
    """A generic workflow for any scan that captures images on a rectilinear grid."""

    # Redefine _csm Thing Slot, as CSM is required for any RectGridWorkflow
    _csm: CameraStageMapper = lt.thing_slot()

    overlap: float = lt.setting(default=0.35, ge=0.1, le=0.7)
    """The fraction that adjacent images should overlap in x and y.

    This must be between 0.1 and 0.7.
    """

    autofocus_dz: int = lt.setting(default=1000, ge=400, le=3000)
    """The z distance to perform an autofocus in steps.

    Must be greater than or equal to 400, and less than or equal to 3000.

    Note that 200 steps is the backlash correction size, so the value
    must be above this. 3000 is a sensible limit for 20x objectives.
    """

    def check_before_start(self, scan_name: str) -> None:
        """Before starting a scan, check that camera-stage-mapping is set.

        Raise error if:
          - camera stage mapping is not set
        """
        super().check_before_start(scan_name)
        if self._csm.calibration_required:
            raise WorkflowStartError("Camera Stage Mapping is not calibrated.")

    def _calc_displacement_from_overlap(self, overlap: float) -> tuple[int, int]:
        """Use camera stage mapping to calculate x and y displacement from given overlap.

        :param overlap: The desired overlap as a fraction of the image. i.e. 0.5 means
            that each image should overlap its nearest neighbour by 50%.

        :returns: (dx, dy) - the x and y displacements in steps

        :raises RuntimeError: If there is no camera stage mapper Thing available or if CMS isn't calibrated.
        """
        csm_image_res = self._csm.image_resolution
        if csm_image_res is None:
            raise RuntimeError("CSM not set. Scan shouldn't have progressed this far.")

        # Calculate displacements in image coordinates
        dx_img = csm_image_res[1] * (1 - overlap)
        dy_img = csm_image_res[0] * (1 - overlap)

        x_move_stage = self._csm.convert_image_to_stage_coordinates(x=dx_img, y=0)
        y_move_stage = self._csm.convert_image_to_stage_coordinates(x=0, y=dy_img)

        # Assume no rotation or skew and take only the aligned axis of vector.
        # Coerce to positive integer, but correct if x and y are flipped
        if abs(x_move_stage["x"]) > abs(x_move_stage["y"]):
            return x_move_stage["x"], y_move_stage["y"]
        # If not use the other stage axes. Note "dx" will be the movement in camera y.

        self.logger.info(
            f"Based on an overlap of {self.overlap}, the stage will make steps of "
            f"{y_move_stage['x']}, {x_move_stage['y']}"
        )
        return y_move_stage["x"], x_move_stage["y"]

    def log_movement_distances(self, scan_settings: RectGridSettingModelType) -> None:
        """Log the dx and dy movements for the started scan."""
        self.logger.info(
            f"Scanning with steps of dx={scan_settings.dx} and dy={scan_settings.dy}."
        )

    def _get_stitching_settings_model(
            self, save_resolution: tuple[int, int]
    ) -> StitchingSettings:
        """Return a stitching settings model based on current settings."""
        width, height = save_resolution
        # Target area in pixels
        target_area = TARGET_STITCHING_DIMENSION ** 2
        # Find N so that (width/N) * (height/N) ~ target_area
        # N^2 ~ (width * height) / target_area
        downsample_factor = max(1, round((width * height / target_area) ** 0.5))
        correlation_resize = 1 / downsample_factor

        return StitchingSettings(
            overlap=self.overlap,
            correlation_resize=correlation_resize,
        )

    def _build_scan_settings(self, base_kwargs: dict) -> RectGridSettingModelType:
        """Construct the _settings_model."""
        # Developer Note: This needs to be overridden if the settings model for this
        # class contains extra keys.
        return self._settings_model(**base_kwargs)

    def all_settings(
            self, images_dir: RelativeDataPath
    ) -> tuple[RectGridSettingModelType, Optional[StitchingSettings], tuple[int, int]]:
        """Return scan settings and the stitching settings.

        :param images_dir: The directory that images are to be written to.
        :return: A tuple containing the settings model for this workflow, the
            settings model for stitching, and the save resolution.
        """
        save_resolution = self._get_save_resolution()
        # Developer Note: When subclassing RectGridWorkflow rather than override
        # this method first consider overriding _build_scan_settings
        stitching_settings = self._get_stitching_settings_model(save_resolution)
        dx, dy = self._calc_displacement_from_overlap(self.overlap)

        base_kwargs = {
            "overlap": self.overlap,
            "dx": dx,
            "dy": dy,
            "capture_params": CaptureParams(
                images_dir=images_dir, capture_mode=self.capture_mode
            ),
            "autofocus_params": AutofocusParams(dz=self.autofocus_dz),
        }

        scan_settings = self._build_scan_settings(base_kwargs)

        # Log the dx, dy once scan_settings are finalised
        self.log_movement_distances(scan_settings)

        return scan_settings, stitching_settings, save_resolution

    @lt.property
    def ready(self) -> bool:
        """Whether this scanworkflow is ready to start."""
        return not self._csm.calibration_required


class SmartStackCompatibleSettings(Protocol):
    """A protocol for the minimum settings needed for smart stack to work."""

    capture_params: CaptureParams
    autofocus_params: AutofocusParams
    smart_stack_params: SmartStackParams


class SmartStackMixin:
    """A mixin for scan workflows that use smart stacking."""

    stack_images_to_save: int = lt.setting(default=1, ge=1, le=9)
    """The number of images to save in a stack.

    Defaults to 1 unless you need to see either side of focus
    """

    stack_sharp_set: bool = lt.setting(default=False)
    """True: distribute images_to_save samples within +-stack_dz of focus
    so every slice stays sharp (set stack_dz to FWHM/2 or less; the
    tab-open default does this). False: fixed stack_dz spacing, N sets
    z-coverage."""

    stack_min_images_to_test: int = lt.setting(
        default=9, ge=MIN_TEST_IMAGE_COUNT, le=MAX_TEST_IMAGE_COUNT
    )
    """The minimum number of images to capture in a stack.

    This many images are captured and tested for focus, if the focus is not central
    enough more images may be captured. After new images are captured, this value sets
    the number of images used for checking if focus is achieved.

    Defaults to 9 which balances reliability and speed.
    """

    stack_dz: int = lt.setting(default=50, ge=10, le=400)
    """Distance in steps between images in a z-stack.

    Suggested values:

    * 50 for 60-100x
    * 100 for 40x
    * 200 for 20x
    """

    @property
    def as_workflow(self) -> ScanWorkflow:
        """Return self as a ScanWorkflow.

        Ensures this mixin is only used with ScanWorkflow instances,
        raising TypeError otherwise.
        """
        if not isinstance(self, ScanWorkflow):
            raise TypeError("SmartStackMixin must be mixed into a ScanWorkflow")
        return self

    def create_smart_stack_params(self, save_on_failure: bool) -> SmartStackParams:
        """Set up the parameters used for all smart stacks in a scan.

        :returns: A StackSmartParams object with the required parameters.
        """
        # Coerce min_images_to_test parameter
        min_images_to_test = self.stack_min_images_to_test
        if min_images_to_test % 2 == 0:
            min_images_to_test += 1
            self.as_workflow.logger.warning(
                "Minimum number of images to test should be odd, setting to "
                f"{min_images_to_test}."
            )
        # Set the Thing property to the coerced value
        self.stack_min_images_to_test = min_images_to_test

        # images_to_save is the ladder length around the engine's focus;
        # it no longer relates to min_images_to_test (engine finds focus,
        # the ladder is only the saved product).
        images_to_save = self.stack_images_to_save

        return SmartStackParams(
            stack_dz=self.stack_dz,
            images_to_save=self.stack_images_to_save,
            min_images_to_test=self.stack_min_images_to_test,
            save_on_failure=save_on_failure,
        )

    def _perform_smart_stack(
            self, settings: SmartStackCompatibleSettings, xyz_pos: tuple[int, int, int]
    ) -> tuple[bool, Optional[int], int]:
        """Perform acquisition a smart stack.

        :param settings: The settings for this scan as a HistoScanSettingsModel
        :param xyz_pos: The current position as a tuple or 3 ints.
        :return: A tuple of whether an image was taken, and the z-position for focus.
            If failed to find focus, returns for the focus z-position.
        """
        focus_height: Optional[int]
        focused, focus_height, image_count = (
            self.as_workflow._autofocus.run_smart_stack(
                stack_parameters=settings.smart_stack_params,
                capture_parameters=settings.capture_params,
                autofocus_parameters=settings.autofocus_params,
            )
        )
        # An image was captured if we are focussed or we are not skipping background.
        imaged = focused or settings.smart_stack_params.save_on_failure

        if not imaged:
            msg = f"Stack failed at {xyz_pos}. Treating as background."
            self.as_workflow.logger.info(msg)

        # run_smart_stage always returns a focus height for the sharpest image even
        # if it failed to find a good focus. Set to None if not focussed.
        if not focused:
            focus_height = None

        return (
            imaged,
            focus_height,
            settings.smart_stack_params.images_to_save if imaged else 0,
        )

    def _refresh_stack_dz_default(self) -> None:
        """Set stack_dz to half the engine's latest measured FWHM
        (Nyquist for z-reconstruction). Runs on every tab open: the field
        is always the live recommended default. A user-typed value lasts
        for the scans launched from that tab visit; leaving and reopening
        the tab resets to the fresh default."""
        af = getattr(self.as_workflow, "_autofocus", None)
        fwhm = getattr(type(af), "_CAL_FWHM", None) if af else None
        if not fwhm:
            return
        self.stack_dz = max(10, min(400, int(round(fwhm / 2))))

    _save_executor = ThreadPoolExecutor(max_workers=1)
    _pending_saves: list = []

    def _drain_pending_saves(self, timeout: float = 30.0) -> None:
        for fut in list(self._pending_saves):
            try:
                fut.result(timeout=timeout)
            except Exception:
                self.logger.exception("Tile save failed")
        self._pending_saves.clear()

    def _focus_and_ladder(
        self, settings: SmartStackCompatibleSettings, xyz_pos: tuple[int, int, int]
    ) -> tuple[bool, Optional[int], int]:
        """Find focus with the smooth engine, then capture a short ladder
        around it: images_to_save images spaced stack_dz apart, centred
        so the sharpest-z sits in the middle. images_to_save=1 saves a
        single image at the focused z. Sensor grabs stay synchronous at
        the parked position; JPEG encode+write run on a worker thread
        and are joined before the next tile's grab."""
        af_params = settings.smart_stack_params
        images_to_save = max(1, af_params.images_to_save)
        step = af_params.stack_dz
        scan_af = getattr(
            self.as_workflow._autofocus, "scan_autofocus", None
        )
        if scan_af is not None:
            scan_af(span=0.0)
        else:
            self.as_workflow._autofocus.fast_autofocus(
                dz=settings.autofocus_params.dz
            )
        if images_to_save > 1:
            focus_height: Optional[int] = (
                self.as_workflow._stage.get_xyz_position()[2]
            )
        else:
            focus_height = None
        if images_to_save > 1 and self.stack_sharp_set:
            half = step
            step = (2 * half) // (images_to_save - 1) if images_to_save > 1 else 0
        else:
            half = step * (images_to_save - 1) // 2
        if images_to_save > 1 and half:
            self.as_workflow._stage.move_relative(z=-half)
        for k in range(images_to_save):
            if k and step:
                self.as_workflow._stage.move_relative(z=step)
            z_now = self.as_workflow._stage.get_xyz_position()[2]
            filename = f"img_{xyz_pos[0]}_{xyz_pos[1]}_{z_now}.jpeg"
            self._drain_pending_saves()
            buf = self.as_workflow._cam.capture_to_memory(
                capture_mode=settings.capture_params.capture_mode
            )
            path = settings.capture_params.images_dir.join(filename)
            cam = self.as_workflow._cam
            raw_save = getattr(
                cam.save_from_memory, "__wrapped__", None
            )
            if raw_save is None:
                desc = type(cam).save_from_memory
                raw_save = getattr(desc, "func", cam.save_from_memory)
            self._pending_saves.append(
                self._save_executor.submit(
                    raw_save, cam, path=path, buffer_id=buf,
                )
            )
        return True, focus_height, images_to_save

    def smart_stack_property_controls(self) -> list[PropertyControl]:
        """Return smart stack property controls for the UI."""
        return [
            property_control_for(
                self.as_workflow,
                "stack_images_to_save",
                label="Images in Stack to Save",
            ),
            property_control_for(
                self.as_workflow,
                "stack_min_images_to_test",
                label="Minimum number of images to test for focus",
            ),
            property_control_for(
                self.as_workflow, "stack_dz", label="Stack dz (steps)", step=5
            ),
            property_control_for(
                self.as_workflow,
                "stack_sharp_set",
                label="Sharp set: spread samples within +-Stack dz",
            ),
        ]


class HistoScanSettingsModel(RectGridSettingsModel):
    """The settings for a scan with the HistoScanWorkflow.

    This includes settings calculated when starting. This will be held by smart scan
    during a scan and serialised to disk.
    """

    max_dist: int
    skip_background: bool
    smart_stack_params: SmartStackParams


class HistoScanWorkflow(RectGridWorkflow[HistoScanSettingsModel], SmartStackMixin):
    """A workflow optimised for scanning Histopathology samples.

    This workflow automatically plans its own path around a sample spiralling out from
    the centre position, scanning only where it detects sample.
    """

    display_name: str = lt.property(default="Histo Scan", readonly=True)
    ui_blurb: str = lt.property(
        default=(
            "This scan workflow is optimised for scanning H&E stained biopsies. It "
            "spirals out from the starting location, scanning only where it detects "
            "sample. It also works well for many other flat, well-featured samples."
        ),
        readonly=True,
    )

    _settings_model = HistoScanSettingsModel
    _planner_cls = SmartSpiral
    # Thing Slots
    _background_detector: ChannelDeviationLUV = lt.thing_slot()

    # Scan settings

    skip_background: bool = lt.setting(default=True)
    """Whether to detect and skip empty fields of view.

    This uses the settings from the ``BackgroundDetectThing``.
    """

    max_range: int = lt.setting(default=45000, ge=0)
    """The maximum distance in steps from the centre of the scan."""

    equal_distances: bool = lt.setting(default=False)
    """Make the distances in x and y equal in motor steps, rather than in overlap.

    Uses the shorter distance (usually dy) as both dx and dy"""

    # The noqa statement is because scan_name is unused but is needed for equivalence
    # with other workflows that may want to validate the scan name.
    def check_before_start(self, scan_name: str) -> None:  # noqa: ARG002
        """Before starting a scan, check that background and CSM are set.

        Raise error if:
          - background is to be skipped but is not set
          - camera stage mapping is not set

        Raise warning if not using background detect that scan will go on until max
        steps reached.
        """
        super().check_before_start(scan_name)
        if self._csm.calibration_required:
            raise WorkflowStartError("Camera Stage Mapping is not calibrated.")

        if self.skip_background:
            if not self._background_detector.ready:
                raise WorkflowStartError(
                    "Background is not set: you need to calibrate background detection."
                )
        else:
            self.logger.warning(
                "This scan will run in a spiral from the starting point "
                f"until you cancel it, or until it has moved by {self.max_range} steps "
                "in every direction. Make sure you watch it run to stop it leaving "
                "the area of interest, or (worse) leading the microscope's range "
                "of motion."
            )

    @lt.property
    def ready(self) -> bool:
        """Whether this scanworkflow is ready to start."""
        if self._csm.calibration_required:
            return False
        if not self.skip_background:
            return True
        return self._background_detector.ready

    def _build_scan_settings(self, base_kwargs: dict) -> HistoScanSettingsModel:
        """Construct the SettingModel for all_settings.

        Adjust dx and dy to be equal if `equal_distances` is set.
        """
        # Make dx and dy equal if requested
        if self.equal_distances:
            dx = abs(base_kwargs.get("dx", 0))
            dy = abs(base_kwargs.get("dy", 0))
            min_displacement = min(dx, dy)
            base_kwargs["dx"] = min_displacement
            base_kwargs["dy"] = min_displacement

        return HistoScanSettingsModel(
            **base_kwargs,
            max_dist=self.max_range,
            skip_background=self.skip_background,
            smart_stack_params=self.create_smart_stack_params(
                save_on_failure=not self.skip_background
            ),
        )

    def pre_scan_routine(self, settings: HistoScanSettingsModel) -> None:
        """Autofocus before starting the scan.

        :param settings: The settings for this scan as a HistoScanSettingsModel
        """
        self._autofocus.looping_autofocus(
            dz=settings.autofocus_params.dz, start="centre"
        )

    def new_scan_planner(
            self, settings: HistoScanSettingsModel, position: Mapping[str, int]
    ) -> ScanPlanner:
        """Return a new scan planner object.

        :param settings: The settings for this scan as a HistoScanSettingsModel
        :param position: The starting position as a mapping of axes names to int.
        """
        # The initial plan for the scan should be a single x,y position. All future
        # moves will be planned around this point. In future, route planner could
        # have multiple starting positions, each of which will be visited before the
        # scan can end.
        planner_settings = {
            "dx": settings.dx,
            "dy": settings.dy,
            "max_dist": settings.max_dist,
        }
        return self._planner_cls(
            initial_position=(position["x"], position["y"]),
            planner_settings=planner_settings,
        )

    def acquisition_routine(
            self, settings: HistoScanSettingsModel, xyz_pos: tuple[int, int, int]
    ) -> tuple[bool, Optional[int], int]:
        """Overload to set the acquisition routine that happens at each scan site.

        :param settings: The settings for this scan, which should be a SettingModelType
        :param xyz_pos: The current position as a tuple or 3 ints.
        :return: A tuple of whether an image was taken, the z-position for focus, and
            the number of images captured at this site.
            If failed to find focus, returns for the focus z-position.
        """
        if settings.skip_background:
            # If skipping background, take an image to check if current field of view
            # is background
            image_array = self._cam.grab_as_array(stream_name="lores")
            capture_image, bg_message = self._background_detector.image_is_sample(
                image_array
            )
            del image_array

            if not capture_image:
                # Return early if the image is background.
                msg = f"Skipping {xyz_pos} as it is {bg_message}."
                self.logger.info(msg)
                return False, None, 0

        return self._focus_and_ladder(settings, xyz_pos)

    @lt.action
    def check_background(self) -> str:
        """Check if sample is background.

        This action is a pre-run check for feeding back to the user.
        """
        image_array = self._cam.grab_as_array(stream_name="lores")
        is_sample, bg_message = self._background_detector.image_is_sample(image_array)
        label = "sample" if is_sample else "background"

        return f"Current image is {label} ({bg_message})"

    @lt.action
    def set_background(self) -> None:
        """Set the background for this background detector.

        This sets the background for this workflow's background detector as opposed to
        the active background detector for the camera.
        """
        image_array = self._cam.grab_as_array(stream_name="lores")
        self._background_detector.set_background(image_array)

    @lt.endpoint("get", "settings_ui", responses=UI_ELEMENT_RESPONSE)
    def settings_ui(self) -> UIElementList:
        """Return the UI for the workflow's settings in the scan tab."""
        self._refresh_stack_dz_default()
        scan_settings = UIElementList(
            [
                property_control_for(
                    self, "overlap", label="Image Overlap (0.1-0.7)", step=0.05
                ),
                *self.smart_stack_property_controls(),
                property_control_for(
                    self, "autofocus_dz", label="Autofocus Range (steps)", step=200
                ),
                property_control_for(
                    self, "max_range", label="Maximum Distance (steps)", step=1000
                ),
                property_control_for(
                    self, "skip_background", label="Detect and Skip Empty Fields"
                ),
                property_control_for(
                    self, "equal_distances", label="Set Equal x and y Distances"
                ),
            ]
        )
        background_ui = self._background_detector.settings_ui()
        set_bg_button = action_button_for(
            self,
            "set_background",
            poll_interval=0.1,
            submit_label="Set Background",
            can_terminate=False,
            notify_on_success=True,
            success_message="Background image has been updated",
            update_interface_on_response=True,
        )
        check_bg_button = action_button_for(
            self,
            "check_background",
            poll_interval=0.1,
            submit_label="Check Current Image",
            disabled=not self._background_detector.ready,
            can_terminate=False,
            notify_on_success=True,
            response_is_success_message=True,
        )

        background_ui.root += [set_bg_button, check_bg_button]

        return UIElementList(
            [
                HeaderBlock(text=self.display_name, level=4),
                TextBlock(text=self.ui_blurb),
                Accordion(title="Background Detect", children=background_ui),
                Accordion(
                    title="Scan Settings",
                    children=scan_settings,
                ),
            ]
        )


class RegularGridSettingsModel(RectGridSettingsModel):
    """The settings for a scan with a regular grid of dx and dy for x_count, y_count steps.

    This includes settings calculated when starting. This will be held by smart scan
    during a scan and serialised to disk.
    """

    x_count: int
    y_count: int
    smart_stack_params: SmartStackParams
    style: Literal["snake", "raster"]


RegGridSettingModelType = TypeVar(
    "RegGridSettingModelType", bound=RegularGridSettingsModel
)


class RegularGridWorkflow(
    RectGridWorkflow[RegGridSettingModelType],
    SmartStackMixin,
    Generic[RegGridSettingModelType],
):
    """A base workflow for any workflow that uses a regular rectangular grid."""

    x_count: int = lt.setting(default=3, ge=1)
    """The number of columns in the scan."""
    y_count: int = lt.setting(default=2, ge=1)
    """The number of rows in the scan."""

    _settings_model: type[RegGridSettingModelType]
    _planner_cls = RegularGridPlanner
    _grid_style: Literal["snake", "raster"]

    def _build_scan_settings(self, base_kwargs: dict) -> RegGridSettingModelType:
        """Construct the SettingModel for all_settings."""
        return self._settings_model(
            **base_kwargs,
            x_count=self.x_count,
            y_count=self.y_count,
            style=self._grid_style,
            smart_stack_params=self.create_smart_stack_params(save_on_failure=True),
        )

    def pre_scan_routine(self, settings: RegGridSettingModelType) -> None:
        """Perform these steps before starting the scan.

        In this case, only autofocus.

        :param settings: The settings for this scan as as the relevant SettingsModel type.
        """
        self._autofocus.looping_autofocus(
            dz=settings.autofocus_params.dz, start="centre"
        )

    def new_scan_planner(
            self, settings: RegGridSettingModelType, position: Mapping[str, int]
    ) -> ScanPlanner:
        """Return a new scan planner object.

        :param settings: The settings for this scan as the relevant SettingsModel type.
        :param position: The starting position as a mapping of axes names to int.
        """
        planner_settings = {
            "dx": settings.dx,
            "dy": settings.dy,
            "x_count": settings.x_count,
            "y_count": settings.y_count,
            "style": settings.style,
        }
        return self._planner_cls(
            initial_position=(position["x"], position["y"]),
            planner_settings=planner_settings,
        )

    def acquisition_routine(
            self, settings: RegGridSettingModelType, xyz_pos: tuple[int, int, int]
    ) -> tuple[bool, Optional[int], int]:
        """Autofocus and capture.

        :param settings: The settings for this scan as the relevant SettingsModel type.
        :param xyz_pos: The current position as a tuple or 3 ints.
        :return: A tuple of whether an image was taken, the z-position for focus, and
            the number of images captured at this site.
            If failed to find focus, returns for the focus z-position.
        """
        return self._focus_and_ladder(settings, xyz_pos)

    @lt.endpoint("get", "settings_ui", responses=UI_ELEMENT_RESPONSE)
    def settings_ui(self) -> UIElementList:
        """Return the UI for the workflow's settings in the scan tab."""
        self._refresh_stack_dz_default()
        scan_settings = UIElementList(
            [
                property_control_for(
                    self, "overlap", label="Image Overlap (0.1-0.7)", step=0.05
                ),
                property_control_for(self, "x_count", label="Number of columns"),
                property_control_for(self, "y_count", label="Number of rows"),
                *self.smart_stack_property_controls(),
                property_control_for(
                    self, "autofocus_dz", label="Autofocus Range (steps)"
                ),
            ]
        )
        return UIElementList(
            [
                HeaderBlock(text=self.display_name, level=4),
                TextBlock(text=self.ui_blurb),
                Accordion(
                    title="Scan Settings",
                    children=scan_settings,
                ),
            ]
        )


class SnakeWorkflow(RegularGridWorkflow[RegularGridSettingsModel]):
    """A workflow optimised for snaking around samples.

    This workflow generates a list of coordinates in a rectangle, and snakes
    around them from the top left (assuming positive dx and dy).
    """

    display_name: str = lt.property(default="Snake Scan", readonly=True)
    ui_blurb: str = lt.property(
        default=(
            "This scan workflow is optimised for scanning over a rectangle. It "
            "snakes down and right from the starting point, over a defined grid."
        ),
        readonly=True,
    )
    _settings_model = RegularGridSettingsModel
    _grid_style = "snake"


class RasterWorkflow(RegularGridWorkflow[RegularGridSettingsModel]):
    """A workflow optimised for snaking around samples.

    This workflow generates a list of coordinates in a rectangle, and always
    moves right across a row, then moves down a row while moving to the starting
    column (assuming positive dx and dy).
    """

    display_name: str = lt.property(default="Raster Scan", readonly=True)
    ui_blurb: str = lt.property(
        default=(
            "This scan workflow is optimised for performing a raster scan over a rectangle. It "
            "always moves down and right from the starting point, over a defined grid."
        ),
        readonly=True,
    )
    _settings_model = RegularGridSettingsModel
    _grid_style = "raster"


class CChipScanSettingsModel(RectGridSettingsModel):
    """The settings for a scan with the CChipWorkflow.

    This includes settings calculated when starting. This will be held by smart scan
    during a scan and serialised to disk.
    """

    x_count: int
    y_count: int
    stack_params: StackParams
    style: Literal["snake", "raster"]


class CChipWorkflow(RectGridWorkflow[CChipScanSettingsModel]):
    """A workflow optimised for scanning the well of a CChip.

    This workflow generates a list of coordinates in a rectangle, and snakes
    around them from the top left (assuming positive dx and dy), stacking the
    grid and above.
    """

    display_name: str = lt.property(default="C-Chip Scan", readonly=True)
    ui_blurb: str = lt.property(
        default=(
            "This scan workflow is optimised for scanning a C-Chip. It focuses on "
            "a grid, then stacks images above the grid to complete a volumetric scan."
        ),
        readonly=True,
    )
    _grid_style: Literal["snake"] = "snake"
    _planner_cls = RegularGridPlanner
    _settings_model = CChipScanSettingsModel

    overlap: float = lt.setting(default=0.1, ge=0.1, le=0.7)
    """The fraction that adjacent images should overlap in x and y.

    This must be between 0.1 and 0.7.
    """
    x_count: int = lt.setting(default=5, readonly=False)
    """The number of columns in the scan."""
    y_count: int = lt.setting(default=7, readonly=False)
    """The number of rows in the scan."""

    stack_images_to_save: int = lt.setting(default=9, readonly=False)
    """The number of images to save in a stack.

    Defaults to 1 unless you need to see either side of focus
    """

    stack_dz: int = lt.setting(default=500, readonly=False)
    """Distance in steps between images in a z-stack."""

    def create_stack_params(
            self,
    ) -> StackParams:
        """Set up the parameters used for all stacks in a scan.

        :returns: A StackSmartParams object with the required parameters.
        """
        return StackParams(
            stack_dz=self.stack_dz, images_to_save=self.stack_images_to_save
        )

    def _build_scan_settings(self, base_kwargs: dict) -> CChipScanSettingsModel:
        """Construct the SettingModel for all_settings."""
        stack_params = self.create_stack_params()

        return self._settings_model(
            **base_kwargs,
            x_count=self.x_count,
            y_count=self.y_count,
            style=self._grid_style,
            stack_params=stack_params,
        )

    def new_scan_planner(
            self, settings: CChipScanSettingsModel, position: Mapping[str, int]
    ) -> ScanPlanner:
        """Return a new scan planner object.

        :param settings: The settings for this scan as the relevant SettingsModel type.
        :param position: The starting position as a mapping of axes names to int.
        """
        planner_settings = {
            "dx": settings.dx,
            "dy": settings.dy,
            "x_count": settings.x_count,
            "y_count": settings.y_count,
            "style": settings.style,
        }
        return self._planner_cls(
            initial_position=(position["x"], position["y"]),
            planner_settings=planner_settings,
        )

    def pre_scan_routine(self, settings: CChipScanSettingsModel) -> None:
        """No autofocus, a looping autofocus on a CChip could corrupt the entire scan."""
        pass

    def acquisition_routine(
            self,
            settings: CChipScanSettingsModel,
            xyz_pos: tuple[int, int, int],  # noqa: ARG002
    ) -> tuple[bool, Optional[int], int]:
        """Autofocus and capture a z-stack starting at the focused position.

        The routine performs a fast autofocus using the provided ``dz``.
        The focused z-height becomes the starting position for the stack
        and is also the height of the first captured image.

        A stack of ``self.stack_images_to_save`` images is then acquired.
        Each subsequent image is captured after moving the stage upward
        by ``self.stack_dz`` steps in z.

        :param xyz_pos: The (x, y, z) position associated with this acquisition.
        :param settings: The settings for this scan as a CChipSettingsModel

        :return: (True, focus_height, image_count) where focus_height is the
            autofocus z-position and the height of the first image in the stack,
            and image_count is the number of images captured at this site.
        """
        # Perform autofocus
        self._autofocus.fast_autofocus(dz=settings.autofocus_params.dz)
        focus_height = self._stage.get_xyz_position()[2]
        self._autofocus.run_basic_stack(settings.stack_params, settings.capture_params)

        return True, focus_height, settings.stack_params.images_to_save

    @lt.property
    def settings_ui(self) -> list[PropertyControl]:
        """A list of PropertyControl objects to create the settings in the scan tab."""
        return [
            property_control_for(self, "overlap", label="Image Overlap (0.1-0.7)"),
            property_control_for(self, "x_count", label="Number of columns"),
            property_control_for(self, "y_count", label="Number of rows"),
            property_control_for(self, "autofocus_dz", label="Autofocus Range (steps)"),
            property_control_for(
                self, "stack_dz", label="Distance in z between images in stack (steps)"
            ),
            property_control_for(
                self, "stack_images_to_save", label="Images to save per xy site"
            ),
        ]
