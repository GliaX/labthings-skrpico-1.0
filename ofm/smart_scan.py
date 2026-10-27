"""The core sample scanning functionality for the OpenFlexure Microscope.

SmartScan provides sample scanning functionality. This functionality can be customised
by different ``ScanWorkflow`` Things which control the path planning and acquisition
routines. It manages the directories of past scans via `scan_directories`.
It also controls external processes for live stitching composite images, and
the creation of the final stitched images.
"""

import os
import shutil
import threading
import time
from datetime import datetime
from types import TracebackType
from typing import (
    Annotated,
    Any,
    Callable,
    Concatenate,
    Literal,
    Mapping,
    Optional,
    ParamSpec,
    Self,
    TypeVar,
)

import labthings_fastapi as lt
from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, PlainSerializer

from openflexure_microscope_server import stitching
from openflexure_microscope_server.scan_directories import (
    BaseScanData,
    ScanDirectory,
    ScanGalleryDBEngine,
    ScanGalleryInfo,
    latest_scan_preview,
)
from openflexure_microscope_server.things import OFMThing
from openflexure_microscope_server.ui import ActionButton, action_button_for
from openflexure_microscope_server.utilities import (
    check_free_disk_space,
    coerce_thing_selector,
)
# Things
from .camera import BaseCamera
from .scan_workflows import ScanWorkflow
from .stage import BacklashCompensation, BaseStage

T = TypeVar("T")
P = ParamSpec("P")

MIN_IMAGES_TO_STITCH = 2

# This allows ActiveScanData to hold arbitrary workflow settings models during a scan.
AnyModel = Annotated[
    BaseModel,
    PlainSerializer(lambda value: value.model_dump(), return_type=dict),
]


class LiveScanDetails(BaseModel):
    """Details of ongoing scan used for the UI.

    This is also used to populate the "current scan settings" info modal in the
    scan tab, so it contains the workflow name and strings / dicts of the
    workflow/stitching settings.

    ``duration_seconds`` is monitored server-side and polled with other details,
    rather than being monitored in the front end while also polling the back end.
    """

    name: str
    stitch_timestamp: Optional[float] = None
    image_count: int = 0
    scan_phase: Literal["Setup", "Scanning", "Stitching", "Complete"] = "Setup"
    duration_seconds: Optional[float] = None
    workflow: Optional[str] = None
    settings: Optional[dict[str, Any]] = None
    stitching_settings: Optional[dict[str, Any]] = None
    save_resolution: Optional[tuple[int, int]] = None


class ActiveScanData(BaseScanData):
    """A model for the scan data during an ongoing scan.

    This differs from HistoricScanData as in this model ``workflow_settings`` are the
    model specified for the current ScanWorkflow. HistoricScanData loads
    ``workflow_settings`` into a dictionary.
    """

    workflow_settings: AnyModel
    """The settings for the ongoing workflow."""

    def set_final_data(
            self, result: str, completion_reason: Optional[str] = None
    ) -> None:
        """Set the final data for the scan, scan duration is automatically calculated.

        :param result: A string describing the result.
        :param completion_reason: An optional, more detailed human readable
            explanation of why the scan ended - e.g. why the scan planner
            stopped adding new locations. Only set for a successful scan;
            for cancellations and errors this detail already lives in
            ``result``.
        """
        self.duration = datetime.now() - self.start_time
        self.scan_result = result
        if completion_reason is not None:
            self.completion_reason = completion_reason


class JPEGBlob(lt.blob.Blob):
    """A class representing a JPEG image as a LabThings FastAPI Blob."""

    media_type: str = "image/jpeg"


class ZipBlob(lt.blob.Blob):
    """A class representing a Zip file as a LabThings FastAPI Blob."""

    media_type: str = "application/zip"


class ScanNotRunningError(RuntimeError):
    """Exception called when scan not running that requires a scan to be running."""


def _scan_running(
        method: Callable[Concatenate["SmartScanThing", P], T],
) -> Callable[Concatenate["SmartScanThing", P], T]:
    """Decorate a method so that it will error if a scan is not running.

    This decorator is used by all methods in SmartScanThing that are using
    the variables set for the scan. It will throw a runtime error if
    self._scan_lock is not locked, as all scan variables are set at
    the same time and released with the lock
    """

    def scan_running_wrapper(
            self: "SmartScanThing", *args: P.args, **kwargs: P.kwargs
    ) -> T:
        """Only start the requested method if the scan is running."""
        if self._scan_lock.locked():
            return method(self, *args, **kwargs)
        raise ScanNotRunningError(
            "Calling a @scan_running method can only be done while a scan is running!"
        )

    return scan_running_wrapper


class SmartScanThing(OFMThing):
    """A Thing for scanning samples and interacting with past scans.

    SmartScanThing exposes all functionality for automatically scanning samples,
    previewing live stitching, retrieving data from past scans, and for deleting
    past scans.
    """

    _class_settings = {"validate_properties_on_set": True}

    _cam: BaseCamera = lt.thing_slot()
    _stage: BaseStage = lt.thing_slot()
    _all_workflows: Mapping[str, ScanWorkflow] = lt.thing_slot()

    def __init__(
            self,
            thing_server_interface: lt.ThingServerInterface,
            default_workflow: str,
    ) -> None:
        """Initialise a SmartScanThing saving to and loading from the input directory.

        :param default_workflow: The default workflows that smart scan uses if nothing
            is set in settings.
        """
        super().__init__(thing_server_interface)
        self._scan_lock = threading.Lock()
        self._default_workflow = default_workflow
        self._workflow_name = default_workflow

    def __enter__(self) -> Self:
        """Open hardware connection when the Thing context manager is opened."""
        super().__enter__()
        self._gallery_db_engine = ScanGalleryDBEngine(self.data_dir, self)
        valid_name = coerce_thing_selector(
            thing_mapping=self._all_workflows,
            selected=self.workflow_name,
            default=self._default_workflow,
        )
        if valid_name is None:
            raise RuntimeError(
                "Could not set Scan Workflow. A Scan Workflow must be present in your "
                "configuration."
            )
        self._workflow_name = valid_name
        return self

    def __exit__(
            self,
            _exc_type: type[BaseException],
            _exc_value: Optional[BaseException],
            _traceback: Optional[TracebackType],
    ) -> None:
        """Clean up after context manager is closed."""
        self.gallery_db_engine.dispose()

    # Register with gallery.
    _show_data_in_gallery = True

    _gallery_db_engine: Optional[ScanGalleryDBEngine] = None

    @property
    def gallery_db_engine(self) -> ScanGalleryDBEngine:
        """The database engine."""
        if self._gallery_db_engine is None:
            raise RuntimeError(
                "Gallery engine has not been created yet. Has the LabThings server "
                "started?."
            )
        return self._gallery_db_engine

    def delete_all_gallery_items(self) -> None:
        """Delete all the scans on the microscope.

        **This will irreversibly remove all scanned data from the microscope!**

        Use with extreme caution.
        """
        for scan_name in self.gallery_db_engine.get_all_paths():
            lt.raise_if_cancelled()
            self.logger.info(f"Deleting: {scan_name}")
            self._delete_scan(scan_name)

    def get_gallery_bulk_actions(self) -> list[ActionButton]:
        """Return the bulk gallery actions for smart scan."""
        # Stitch all scans
        return [
            action_button_for(
                self,
                "stitch_all_scans",
                submit_label="Stitch All Unstitched Scans",
                can_terminate=True,
                button_primary=False,
                modal_progress=True,
                requires_confirmation=True,
                confirmation_message=(
                    "<h3>Stitch all unstitched scans?</h3>"
                    "<br>Depending on the number and size of scans, this may be slow, "
                    "and your microscope should not be used during the stitching.'"
                ),
            )
        ]

    # Note that the default detector name is set at init. This is over written if
    # setting is loaded from disk.
    @lt.setting
    def workflow_name(self) -> str:
        """The name of the scan workflow selector."""
        return self._workflow_name

    @workflow_name.setter
    def _set_workflow_name(self, name: str) -> None:
        """Validate and set workflow_name."""
        if name not in self._all_workflows:
            self.logger.warning(f"'{name}' is not a valid scan workflow name.")
            return
        self._workflow_name = name

    @property
    def _workflow(self) -> ScanWorkflow:
        """The active scan workflow object."""
        return self._all_workflows[self.workflow_name]

    _ongoing_scan: Optional[ScanDirectory] = None

    @property
    def ongoing_scan(self) -> ScanDirectory:
        """The ScanDirectory object of the ongoing scan.

        Only read this property is a scan is ongoing or it will raise an error.
        """
        if self._ongoing_scan is None:
            raise ScanNotRunningError("Cannot get ongoing scan if scan is not running.")
        return self._ongoing_scan

    def _get_scan_dir(self, scan_name: str) -> ScanDirectory:
        return ScanDirectory(scan_name, self.data_dir)

    @lt.property
    def all_workflow_names(self) -> list[str]:
        """Return a list of all available Scan Workflows."""
        return list(self._all_workflows.keys())

    @lt.property
    def workflow_display_names(self) -> dict[str, str]:
        """Return a list of the display names of all available Scan Workflows."""
        return {name: wf.display_name for name, wf in self._all_workflows.items()}

    _scan_data: Optional[ActiveScanData] = None

    @property
    def scan_data(self) -> ActiveScanData:
        """The ActiveScanData object holding information about the of the ongoing scan.

        Only read this property is a scan is ongoing or it will raise an error.
        """
        if self._scan_data is None:
            raise ScanNotRunningError("Cannot get scan data if scan is not running.")
        return self._scan_data

    _preview_stitcher: Optional[stitching.PreviewStitcher] = None

    _latest_scan_live_details: Optional[LiveScanDetails] = None

    @lt.property
    def latest_scan_live_details(self) -> Optional[LiveScanDetails]:
        """The details of any ongoing scan, or the previous scan if no scan ongoing."""
        if self._latest_scan_live_details is None:
            return None
        # Update image count and stitch timestamp if scan is ongoing.
        if self._scan_data is not None:
            self._latest_scan_live_details.image_count = self._scan_data.image_count

            # Keep duration live while scan is running.
            self._latest_scan_live_details.duration_seconds = (
                    datetime.now() - self._scan_data.start_time
            ).total_seconds()

        # Use walrus operator to not need to read filesystem timestamp twice
        if (stitch_time := self.latest_preview_stitch_time) is not None:
            self._latest_scan_live_details.stitch_timestamp = stitch_time
        return self._latest_scan_live_details

    @lt.action
    def sample_scan(self, scan_name: str = "") -> None:
        """Move the stage to cover an area, taking images.

        The way the stage moves depends on the selected workflow.
        If images overlap for a scan workflow then the images can be stitched together
        into a larger composite image.
        """
        got_lock = self._scan_lock.acquire(timeout=0.1)
        if not got_lock:
            raise RuntimeError("Trying to run scan while scan is already running!")

        # Set the scan details to None. So we know at the end if the scan started
        self._latest_scan_live_details = None

        try:
            # `scan_data` should already be None. This is added as a precaution as
            # the presence of `scan_data` is used during error handling to
            # determine whether the scan started.
            self._scan_data = None
            # probably make workflow a context manager with a lock?
            workflow = self._workflow

            workflow.check_before_start(scan_name)

            unique_name = self.gallery_db_engine.unique_scan_name(scan_name)
            self._ongoing_scan = ScanDirectory.new_scan_dir(unique_name, self.data_dir)
            self._latest_scan_live_details = LiveScanDetails(
                name=self.ongoing_scan.name
            )
            self._run_scan(workflow)
            # Save any final messages.
            self._ongoing_scan.save_scan_log(self)
        except Exception as e:
            # If _scan_data is set then scan started
            if self._scan_data is not None:
                self._return_to_starting_position()
            if self._ongoing_scan is not None:
                # Save any final messages even if there is an exception.
                self._ongoing_scan.save_scan_log(self)
            # Error must be raised so UI gives correct output
            raise e
        except lt.exceptions.InvocationCancelledError as e:
            # InvocationCancelledError is caught in `_run_scan` if it happens
            # while scanning, but it propagates if it occurs during stitching.
            # If raised we still need to save the scan log.
            if self._ongoing_scan is not None:
                self._ongoing_scan.save_scan_log(self)
            # Reraise so the action reports as cancelled if stitching is cancelled.
            raise e
        finally:
            self._set_final_details_add_to_db()

            # However the scan finishes, unset all variables and release lock
            self._ongoing_scan = None
            self._scan_data = None
            self._scan_lock.release()
            # Ensure any PreviewStitcher created cannot be reused.
            self._preview_stitcher = None
            # Remove any scan folders containing zero images.
            self.purge_empty_scans()

    def _set_final_details_add_to_db(self) -> None:
        """Set the final Live Scan details and add scan to the gallery database."""
        if self._latest_scan_live_details is None:
            return
        # The method must fail as it is called from a finally block.
        try:
            self._latest_scan_live_details.scan_phase = "Complete"

            if self._scan_data is not None:
                self._latest_scan_live_details.image_count = self._scan_data.image_count
                duration = (datetime.now() - self._scan_data.start_time).total_seconds()
                self._latest_scan_live_details.duration_seconds = duration

            if self._preview_stitcher is not None:
                self._latest_scan_live_details.stitch_timestamp = (
                    self.latest_preview_stitch_time
                )

            # If the scan started then add it to the database.
            self.gallery_db_engine.add(self._latest_scan_live_details.name)

        except BaseException as e:
            # As this method is called from a finally block it must not be allowed to
            # error.
            self.logger.exception(e)

    @_scan_running
    def _move_to_next_point(
            self, next_point: tuple[int, int], z_estimate: Optional[int] = None
    ) -> tuple[int, int, int]:
        """Move the stage to the next position.

        If no z_estimate is given then the current stage position is used. Must move
        to the estimated focused position (although moving below would be marginally
        faster) because background detect is most reliable at the focused position.

        :returns: the (x,y,z) with the chosen z_estimate
        """
        if z_estimate is None:
            z_estimate = self._stage.position["z"]

        self.logger.info(f"Moving to {next_point}")
        self._stage.move_absolute(
            x=next_point[0],
            y=next_point[1],
            z=z_estimate,
            backlash_compensation=BacklashCompensation.XY_ONLY,
        )

        return (next_point[0], next_point[1], z_estimate)

    @_scan_running
    def _collect_scan_data(self, workflow: ScanWorkflow) -> ActiveScanData:
        """Collect and return the data for this scan so it cannot be changed mid-scan."""
        # Record starting position so it can be returned to at end of scan.
        starting_position = self._stage.position

        images_dir = self.ongoing_scan.images_dir
        # Type narrowing
        if images_dir is None:
            raise RuntimeError("Couldn't run scan, images directory was not created.")

        workflow_settings, stitching_settings, save_resolution = workflow.all_settings(
            images_dir=self.create_data_path(images_dir, absolute=True)
        )

        # If stitching settings is None then this workflow doesn't support stitching.
        auto_stitch = self.stitch_automatically and stitching_settings is not None

        # Fix scan parameters in case UI is updated during scan.
        return ActiveScanData(
            scan_name=self.ongoing_scan.name,
            starting_position=starting_position,
            start_time=datetime.now(),
            stitch_automatically=auto_stitch,
            save_resolution=save_resolution,
            workflow=type(workflow).__name__,
            workflow_settings=workflow_settings,
            stitching_settings=stitching_settings,
        )

    @_scan_running
    def _save_final_scan_data(
            self, scan_result: str, completion_reason: Optional[str] = None
    ) -> None:
        """Update scan data JSON file with data only known at the end of the scan.

        Takes scan_result, a string that is either "success", "cancelled by user",
        or the error that ended the scan. ``completion_reason`` optionally gives
        more detail on why a successful scan stopped (e.g. range limit reached).
        """
        self.scan_data.set_final_data(
            result=scan_result, completion_reason=completion_reason
        )
        self.ongoing_scan.save_scan_data(self.scan_data)

    @_scan_running
    def _manage_stitching_threads(self) -> None:
        """Manage the stitching threads, starting them if needed and not already running."""
        if self._preview_stitcher is None:
            # This scan can't stitch.
            return
        # Begin stitching once two images are captured
        if (
                self.scan_data.image_count >= MIN_IMAGES_TO_STITCH
                and not self._preview_stitcher.running
        ):
            self._preview_stitcher.start()

    @_scan_running
    def _run_scan(self, workflow: ScanWorkflow) -> None:
        """Prepare and run the main scan, and perform final actions on completion.

        The result (or exception) from the main scan loop determines whether the
        scan should be stitched and whether  the microscope should return to the
        starting x,y,z position.
        """
        try:
            # Change into streaming mode instantly so mode is correct when collecting
            # Scan Data
            self._cam.change_streaming_mode(mode="full_resolution")
            self._cam._scan_hold = True
            self._scan_data = self._collect_scan_data(workflow)
            images_dir = self.ongoing_scan.images_dir
            # Type narrowing
            if images_dir is None:
                raise RuntimeError(
                    "Couldn't run scan, images directory was not created."
                )

            # Populate static workflow & stitching settings at start
            if self._latest_scan_live_details is not None:
                self._latest_scan_live_details.workflow = self._scan_data.workflow
                self._latest_scan_live_details.settings = (
                    self._scan_data.workflow_settings.model_dump(mode="json")
                )
                self._latest_scan_live_details.stitching_settings = (
                    self._scan_data.stitching_settings.model_dump(mode="json")
                    if self._scan_data.stitching_settings is not None
                    else None
                )
                self._latest_scan_live_details.save_resolution = (
                    self._scan_data.save_resolution
                )
                self._latest_scan_live_details.scan_phase = "Scanning"

            self.ongoing_scan.save_scan_data(self._scan_data)
            workflow.pre_scan_routine(self._scan_data.workflow_settings)

            # If stitching settings are None then this type of scan can't be stitched
            if self.scan_data.stitching_settings is not None:
                # Settings exist, so create preview stitcher
                stitching_settings = self.scan_data.stitching_settings
                self._preview_stitcher = stitching.PreviewStitcher(
                    images_dir,
                    overlap=stitching_settings.overlap,
                    correlation_resize=stitching_settings.correlation_resize,
                )
            if self._latest_scan_live_details is not None:
                self._latest_scan_live_details.scan_phase = "Scanning"
            # This is the main loop of the scan!
            completion_reason = self._main_scan_loop(workflow)
            self._save_final_scan_data(
                scan_result="success", completion_reason=completion_reason
            )

        except lt.exceptions.InvocationCancelledError as e:
            self.logger.info("Stopping scan because it was cancelled.")
            self._save_final_scan_data(scan_result="cancelled by user")
            if self.scan_data.image_count < MIN_IMAGES_TO_STITCH:
                # If too few images to stitch then, return to the start and report
                # cancelled
                self._return_to_starting_position()
                raise e
        except Exception as e:
            err_name = type(e).__name__
            if self._scan_data is not None:
                self._save_final_scan_data(scan_result=f"{err_name}: {e}")
            self.logger.error(
                f"The scan stopped because of an error: {e}",
                exc_info=e,
            )
            raise e
        finally:
            self._cam._scan_hold = False
            # Don't set Preview Stitcher to None yet. It is used by
            # _perform_final_stitch, which may also be run after this function completes
            # if it ended due to an exception.

            # Start streaming in the default resolution again as soon as possible
            self._cam.change_streaming_mode(mode="default")

        # This is what happens if the scan completes successfully or the
        # user cancels it.
        self._return_to_starting_position()
        self._perform_final_stitch()

    @_scan_running
    def _main_scan_loop(self, workflow: ScanWorkflow) -> str:
        """Run the main loop of the scan.

        This loop runs during a scan, until no more scan x,y positions
        are remaining.

        :return: A human readable explanation of why the scan planner finished,
            e.g. it reached a range limit or ran out of sample to follow. This
            is also written to the scan log.
        """
        workflow_settings = self.scan_data.workflow_settings
        route_planner = workflow.new_scan_planner(
            workflow_settings, self._stage.position
        )

        # The loop tests if the scan should continue, moves to the next position,
        # decides whether to capture an image, autofocuses if necessary,
        # captures an image if necessary, updates the scan path and future path,
        # and updates the zip file with new images.
        while not route_planner.scan_complete:
            check_free_disk_space(self.data_dir)
            self._manage_stitching_threads()

            next_pos_xy, z_est = route_planner.get_next_location_and_z_estimate()
            new_pos_xyz = self._move_to_next_point(next_pos_xy, z_est)
            current_pos_xyz = (
                new_pos_xyz[0],
                new_pos_xyz[1],
                self._stage.position["z"],
            )
            imaged, focus_height, site_image_count = workflow.acquisition_routine(
                workflow_settings, current_pos_xyz
            )

            if focus_height is None:
                focused = False
            else:
                focused = True
                current_pos_xyz = (new_pos_xyz[0], new_pos_xyz[1], focus_height)

            route_planner.mark_location_visited(
                current_pos_xyz, imaged=imaged, focused=focused
            )

            if imaged:
                self.scan_data.image_count += site_image_count
                # Add it to the incremental zip
                self.ongoing_scan.zip_files()
            self.ongoing_scan.save_scan_log(self)

        completion_reason = route_planner.completion_reason
        drain = getattr(workflow, "_drain_pending_saves", None)
        if drain is not None:
            drain()
        self.logger.info(f"Scan finished after {self.scan_data.image_count} images.")
        self.logger.info(f"{completion_reason}")
        return completion_reason

    @_scan_running
    def _return_to_starting_position(self) -> None:
        """Return to the initial scan position, if set."""
        self.logger.info("Returning to starting position.")

        if self._scan_data is not None:
            self._stage.move_absolute(
                **self.scan_data.starting_position,
                block_cancellation=True,
                backlash_compensation=None,
            )

    @property
    def thing_state(self) -> Mapping[str, Any]:
        """Return a metadata dict for ongoing scan to populate."""
        if self._ongoing_scan is None:
            return {}
        return {"scan_name": self.ongoing_scan.name}

    @_scan_running
    def _perform_final_stitch(self) -> None:
        """Update the scan zip and perform final stitch of the data."""
        if self._latest_scan_live_details is not None:
            self._latest_scan_live_details.scan_phase = "Stitching"
        if self.scan_data.image_count <= 1:
            self.logger.info("Not performing a stitch as not enough images captured")
            return

        self.ongoing_scan.zip_files()

        self.logger.info("Waiting for background processes to finish...")

        # Check the sticher exists rather than using self.preview_sticher as
        # this method can be called during exception handling.
        if self._preview_stitcher is not None:
            self._preview_stitcher.wait()

        stitching_settings = self.scan_data.stitching_settings
        if self.scan_data.stitch_automatically and stitching_settings is not None:
            self.logger.info("Stitching final image (may take some time)...")
            self.stitch_scan(scan_name=self.ongoing_scan.name)

    stitch_tiff: bool = lt.setting(default=False)
    """Whether or not to also produce a pyramidal tiff at the end of a scan."""

    stitch_automatically: bool = lt.setting(default=True)
    """Whether to run a final stitch at the end of a successful scan."""

    @lt.endpoint(
        "delete",
        "scans/{scan_name}",
        responses={
            200: {"description": "Successfully deleted scan"},
            400: {"description": "An error occurred while trying to delete scan"},
        },
    )
    def delete_scan(self, scan_name: str) -> None:
        """Delete the folder for the specified scan.

        This endpoint allows scans to be deleted from disk.

        :param scan_name: The name of the scan to delete
        """
        if not os.path.isdir(os.path.join(self.data_dir, scan_name)):
            self.logger.warning(f"Cannot find a scan of name {scan_name}")
            raise HTTPException(400, "Scan not found")
        deleted_scan_success = self._delete_scan(scan_name)
        if not deleted_scan_success:
            raise HTTPException(400, "Couldn't delete scan, check log for details")

    @lt.action
    def purge_empty_scans(self) -> None:
        """Delete all scan folders containing no images at the top level."""
        # Use the gallery database to check for empty scans.
        for scan_entry in self.gallery_db_engine.get_all_entries():
            scan_info = ScanGalleryInfo.model_validate(scan_entry.gallery_info)
            if scan_info.number_of_images == 0:
                self._delete_scan(scan_entry.path)

    def _delete_scan(self, scan_name: str) -> bool:
        """Delete a scan."""
        if self._ongoing_scan is not None and scan_name == self.ongoing_scan.name:
            self.logger.error("Attempted to delete ongoing scan.")
            return False
        try:
            shutil.rmtree(os.path.join(self.data_dir, scan_name))
            self.gallery_db_engine.delete(scan_name)
            return True
        except Exception as e:
            self.logger.warning(
                "Attempted to delete scan " + scan_name + ", which failed."
                                                          " Server sent response" + str(e)
            )
            return False

    @property
    def latest_preview_stitch_path(self) -> Optional[str]:
        """The path of the latest preview stitched image, or None if not available."""
        if self._latest_scan_live_details is None:
            return None

        return latest_scan_preview(self._latest_scan_live_details.name, self.data_dir)

    @property
    def latest_preview_stitch_time(self) -> Optional[float]:
        """The modification time of the latest preview image, to allow live updating.

        This will return None (``null`` to JS) if there is no preview image to return.

        This is used for two reasons:

        1. If all caching was turned off this stitch would be sent over the network
           repeatedly
        2. If caching was is on, then the stitch will not update when needed.
        """
        if self.latest_preview_stitch_path is None:
            return None
        return os.path.getmtime(self.latest_preview_stitch_path)

    @lt.endpoint(
        "get",
        "latest_preview_stitch.jpg",
        responses={
            200: {
                "description": "A preview-quality stitched image",
                "content": {"image/jpeg": {}},
            },
            404: {"description": "File not found"},
        },
    )
    def get_latest_preview(self) -> FileResponse:
        """Retrieve the latest preview image."""
        preview_path = self.latest_preview_stitch_path
        if preview_path is None:
            raise HTTPException(404, "File not found")
        return FileResponse(preview_path)

    @lt.action(use_global_lock=False)
    def stitch_scan(self, scan_name: str) -> None:
        """Generate a stitched image based on stage position metadata."""
        scan_dir_obj = self._get_scan_dir(scan_name)
        scan_data = scan_dir_obj.get_scan_data()
        if scan_data is None or scan_dir_obj.images_dir is None:
            self.logger.warning(
                "Couldn't read scan data - it may be missing or corrupt."
            )
            return
        if scan_data.stitching_settings is None:
            # If the stitching settings are none then this type of scan cannot be
            # stitiched.
            return

        final_stitcher = stitching.FinalStitcher(
            scan_dir_obj.images_dir,
            logger=self.logger,
            stitch_tiff=self.stitch_tiff,
            stitching_settings=scan_data.stitching_settings,
        )
        try:
            final_stitcher.run()
        except lt.exceptions.InvocationCancelledError as e:
            # Sleep for 1 second just to allow invocation logs to pass to user.
            time.sleep(1)
            raise e
        except ChildProcessError as e:
            self.logger.error(f"Stitching failed: {e}", exc_info=e)
        finally:
            # Ignore missing when rebuilding as a scan in collection is not in the
            # database
            self.gallery_db_engine.rebuild(scan_name, ignore_missing=True)

    @lt.action(use_global_lock=False)
    def download_zip(
            self,
            scan_name: str,
    ) -> ZipBlob:
        """Return zip after including any files left until the end.

        The zipfile is returned as a Blob.
        """
        zip_fname = self._get_scan_dir(scan_name).zip_files(final_version=True)
        return ZipBlob.from_file(zip_fname)

    @lt.action(use_global_lock=False)
    def stitch_all_scans(self) -> None:
        """Check the list of scans, and stitch any that don't have a DZI associated with it.

        :raises RuntimeError: if the microscope is currently running a scan
        """
        if self._scan_lock.locked():
            raise RuntimeError("Can't stitch previous scans while a scan is ongoing")
        # Use the scan list (the data read by the gallery) to find any scans that
        # need stitching.
        for scan_entry in self.gallery_db_engine.get_all_entries():
            scan_info = ScanGalleryInfo.model_validate(scan_entry.gallery_info)
            n_images = scan_info.number_of_images
            dzi = scan_info.dzi
            if dzi is None and n_images >= MIN_IMAGES_TO_STITCH:
                self.logger.info(f"Stitching {scan_entry.path}")
                self.stitch_scan(scan_name=scan_entry.path)
