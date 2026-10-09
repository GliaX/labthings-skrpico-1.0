"""Submodule for interacting with a Raspberry Pi camera using the Picamera2 library.

The Picamera2 library uses LibCamera as the underlying camera stack. This gives us
some control of the GPU pipeline for the image.

The API documentation for PiCamera2 is unfortunately not in a standard auto-generated
website. For documentation of the PiCamera2 API there is a PDF called
"The Picamera2 Library" available at:
https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf

For information on the algorithms used to tune/calibrate the Raspberry Pi Camera see
the guide called "Raspberry Pi Camera Algorithm and Tuning Guide"
Available at:
https://datasheets.raspberrypi.com/camera/raspberry-pi-camera-guide.pdf
"""

from __future__ import annotations

import copy
import json
import logging
import os
import tempfile
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from threading import RLock
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Self,
)

import numpy as np
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import Output
from PIL import Image

import labthings_fastapi as lt
from labthings_fastapi.exceptions import ServerNotRunningError
from labthings_fastapi.types.numpy import NDArray

from openflexure_microscope_server.things.background_detect import ChannelBlankError
from openflexure_microscope_server.ui import (
    ActionButton,
    PropertyControl,
    action_button_for,
    property_control_for,
)

from . import BaseCamera, CaptureMode, StreamingMode
from . import picamera_recalibrate_utils as recalibrate_utils
from . import picamera_tuning_file_utils as tf_utils

if TYPE_CHECKING:
    from libcamera import Request

LOGGER = logging.getLogger(__name__)


class PicameraModelError(RuntimeError):
    """There is a problem Picamera sensor model set by the configuration."""


class MissingCalibrationError(RuntimeError):
    """Picamera tuning file is missing or doesn't contain the requested algorithm."""


class PicameraStreamOutput(Output):
    """An Output class that sends frames to a stream."""

    def __init__(self, stream: lt.outputs.MJPEGStream, encoder: MJPEGEncoder) -> None:
        """Create an output that puts frames in an MJPEGStream.

        :param stream: The labthings MJPEGStream to send frames to.
        :param encoder: The encoder that is encoding the frames. This is needed to get
            the full timestamp.
        """
        Output.__init__(self)
        self.stream = stream
        self.encoder = encoder

    def outputframe(
        self,
        frame: bytes,
        _keyframe: Optional[bool] = True,
        timestamp: Optional[int] = None,
        _packet: Any = None,
        _audio: bool = False,
    ) -> None:
        """Add a frame to the stream's ringbuffer."""
        if timestamp is None or self.encoder.firsttimestamp is None:
            frame_timestamp = None
        else:
            # The difference between unix time and the CPU time. Do not cache this
            # number as it can change with clock slew or other time adjustments.
            cpu_dt = time.time() - time.monotonic()
            # Reconstruct full CPU time of the frame in us
            timestamp += self.encoder.firsttimestamp
            # Convert to datetime
            frame_timestamp = datetime.fromtimestamp(cpu_dt + timestamp / 1e6)
        self.stream.add_frame(frame, frame_timestamp)


class PiCamera2StreamingMode(StreamingMode):
    """Streaming mode configuration for the PiCamera2."""

    main_resolution: tuple[int, int]
    lores_resolution: tuple[int, int]
    sensor_mode_resolution: tuple[int, int]
    bit_depth: int
    use_lores_as_preview: bool
    buffer_count: int = 6
    scaler_crop: Optional[tuple[int, int, int, int]] = None
    frame_rate: Optional[float] = None

    @property
    def sensor_mode_dict(self) -> dict[str, Any]:
        """The dictionary expected by PiCamera2 for setting sensor mode."""
        return {"output_size": self.sensor_mode_resolution, "bit_depth": self.bit_depth}


class PiCamera2CaptureMode(CaptureMode):
    """Capture mode configuration for the PiCamera2."""

    streaming_mode: Optional[str]
    """The streaming mode the camera should be in for capture.

    Use None to use the active mode.
    """
    stream_name: Literal["lores", "main"]
    """The Picamera stream name to capture."""
    timeout: float = 5.0
    """The timeout. A number above 10 risks hardlocking the Pi."""


class StreamingPiCamera2(BaseCamera, ABC):
    """A Thing that provides and interface to the Raspberry Pi Camera.

    This is an abstract base class for all picamera models. Use a subclass specific
    to the camera model.
    """

    _focus_fom: int
    supports_focus_fom: bool = True
    tuning: dict = lt.setting(default_factory=dict, readonly=True)
    """The Raspberry PiCamera Tuning File JSON."""

    # Subclasses should defined both of these.
    _camera_board: str
    _sensor_info: recalibrate_utils.SensorInfo

    def __init__(
        self,
        thing_server_interface: lt.ThingServerInterface,
        camera_num: int = 0,
    ) -> None:
        """Initialise the camera with the given camera number.

        This makes no connection to the camera (except to get the default tuning file).

        :param thing_server_interface: The interface between this Thing and the server.
        :param camera_num: The number of the camera. This should generally be left as 0
            as most Raspberry Pi boards only support 1 camera.
        """
        super().__init__(thing_server_interface)
        self._setting_save_in_progress = False
        self._camera_num = camera_num

        # Check the subclass defined the _camera_board and _sensor_info
        if not hasattr(self, "_camera_board") or not hasattr(self, "_sensor_info"):
            raise AttributeError(
                f"{type(self).__name__} must define both both _camera_board and "
                "_sensor_info attributes"
            )

        self._picamera_lock = RLock()
        self._picamera = None

        # Load the tuning file for the specified sensor mode.
        self.default_tuning = tf_utils.load_default_tuning(
            self._sensor_info.sensor_model
        )

        # Set tuning to default tuning. This will be overwritten when the Thing is
        # connected to the server if tuning is saved to disk.
        try:
            self.tuning = copy.deepcopy(self.default_tuning)
        except ServerNotRunningError as e:
            # This will throw an error after setting as we are not connected to
            # a server. But we know this, so we ignore the error as long as the
            # tuning data is set.
            if "version" not in self.tuning:
                raise RuntimeError("Tuning file could not be set.") from e

        # Also set the colour gains based on the tuning. Set to _colour_gains to not
        # trigger a ServerNotRunningError
        self._colour_gains = tf_utils.get_colour_gains_from_lst(self.tuning)

    mjpeg_bitrate: Optional[int] = lt.property(default=100000000)
    """Bitrate for MJPEG stream (None for default)."""

    stream_active: bool = lt.property(default=False, readonly=True)
    """Whether the MJPEG stream is active."""

    def save_settings(self) -> None:
        """Override save_settings to ensure that camera properties don't recurse.

        This method is run by any Thing when a setting is saved. However, the
        method reads the setting. As reading the setting talks to the
        camera and calls save_settings if the value is not as expected, this could
        cause recursion. Also this means that saving one setting causes all others
        to be read each time.
        """
        try:
            self._setting_save_in_progress = True
            super().save_settings()
        finally:
            self._setting_save_in_progress = False

    @lt.property
    def calibration_required(self) -> bool:
        """Whether the camera needs calibrating."""
        # Check if the lens shading table is calibrated.
        return not tf_utils.lst_calibrated(self.tuning)

    ## Persistent controls! These are settings

    _analogue_gain: float = 1.0

    @lt.setting
    def analogue_gain(self) -> float:
        """The Analogue gain applied by the camera sensor."""
        if not self._setting_save_in_progress and self.streaming:
            with self._streaming_picamera() as cam:
                cam_value = cam.capture_metadata()["AnalogueGain"]
            if cam_value != self._analogue_gain:
                self._analogue_gain = cam_value
                self.save_settings()
        return self._analogue_gain

    @analogue_gain.setter
    def _set_analogue_gain(self, value: float) -> None:
        self._analogue_gain = value
        if self.streaming:
            with self._streaming_picamera() as cam:
                cam.set_controls({"AnalogueGain": value})

    _colour_gains: tuple[float, float] = (1.0, 1.0)

    @lt.setting
    def colour_gains(self) -> tuple[float, float]:
        """The red and blue colour gains, must be between 0.0 and 32.0."""
        if not self._setting_save_in_progress and self.streaming:
            with self._streaming_picamera() as cam:
                cam_value = cam.capture_metadata()["ColourGains"]
            if cam_value != self._colour_gains:
                self._colour_gains = cam_value
                self.save_settings()
        return self._colour_gains

    @colour_gains.setter
    def _set_colour_gains(self, value: tuple[float, float]) -> None:
        self._colour_gains = value
        if self.streaming:
            with self._streaming_picamera() as cam:
                cam.set_controls({"ColourGains": value})

    _exposure_time: int = 500

    @lt.setting
    def exposure_time(self) -> int:
        """The camera exposure time in microseconds.

        When setting this property the camera will adjust the set value
        to the nearest allowed value that is lower than the current setting.
        """
        if not self._setting_save_in_progress and self.streaming:
            with self._streaming_picamera() as cam:
                cam_value = cam.capture_metadata()["ExposureTime"]
            if cam_value != self._exposure_time:
                self._exposure_time = cam_value
                self.save_settings()
        return self._exposure_time

    @exposure_time.setter
    def _set_exposure_time(self, value: int) -> None:
        self._exposure_time = value
        if self.streaming:
            with self._streaming_picamera() as cam:
                # Note: This set a value 1 higher than requested as picamera2 always
                # sets a lower value than requested, even if the requested is allowed
                cam.set_controls({"ExposureTime": value + 1})

    def _get_persistent_controls(self) -> dict:
        if self.streaming:
            self.discard_frames()
        return {
            "AeEnable": False,
            "AnalogueGain": self.analogue_gain,
            "AwbEnable": False,
            "Brightness": 0,
            "ColourGains": self.colour_gains,
            "Contrast": 1,
            # Must also set plus 1 or the exposure drifts with start and stop stream.
            "ExposureTime": self.exposure_time + 1,
            "Saturation": 1,
            "Sharpness": 1,
        }

    @lt.property
    def sensor_resolution(self) -> Optional[tuple[int, int]]:
        """The native resolution of the camera's sensor."""
        with self._streaming_picamera() as cam:
            return cam.sensor_resolution

    def _initialise_picamera(self, check_sensor_model: bool = False) -> None:
        """Acquire the picamera device and store it as ``self._picamera``.

        This duplicates logic in ``Picamera2.__init__`` to provide a tuning file that
        will be read when the camera system initialises.

        :param check_sensor_model: Set to true to check the sensor model is the
            expected sensor model. This is used on ``__enter__`` to confirm that the
            real camera matches the expected camera.

        :raises PicameraModelError: If check_sensor_model is True and the real
            camera sensor model doesn't match the expected sensor model.
        """
        with self._picamera_lock, tempfile.NamedTemporaryFile("w") as tuning_file:
            json.dump(self.tuning, tuning_file)
            tuning_file.flush()  # but leave it open as closing it will delete it
            os.environ["LIBCAMERA_RPI_TUNING_FILE"] = tuning_file.name

            if self._picamera is not None:
                LOGGER.info("Closing picamera object for reinitialisation")
                LOGGER.info(
                    "Camera object already exists, closing for reinitialisation"
                )
                self._picamera.close()
                LOGGER.info("Picamera closed, deleting picamera")
                del self._picamera
                recalibrate_utils.recreate_camera_manager()

            LOGGER.info("Creating new Picamera2 object")
            # Specify tuning file otherwise it will be overwritten with None.
            self._picamera = Picamera2(
                camera_num=self._camera_num,
                tuning=self.tuning,
            )
            if self._picamera is None:
                # Type narrow (error if failure)
                raise RuntimeError("Failed to start Picamera")

            self._picamera.pre_callback = self._on_frame_complete

            if check_sensor_model:
                hw_sensor_model = self._picamera.camera_properties["Model"]
                if hw_sensor_model != self._sensor_info.sensor_model:
                    raise PicameraModelError(
                        f"Wrong Picamera model. Expecting {self._sensor_info.sensor_model}, "
                        f"but found {hw_sensor_model}."
                    )

    @property
    def focus_fom(self) -> int:
        """Return the focus figure of merit."""
        return self._focus_fom

    def _on_frame_complete(self, request: Request) -> None:
        md = request.get_metadata()
        fom = md.get("FocusFoM")
        if fom is not None:
            self._focus_fom = fom

    def __enter__(self) -> Self:
        """Start streaming when the Thing context manager is opened.

        This opens the picamera connection, initialises the camera, sets the
        property, and then starts the streams.
        """
        super().__enter__()
        self._initialise_picamera(check_sensor_model=True)
        self._start_streaming()
        return self

    @property
    def streaming(self) -> bool:
        """True if the camera is streaming."""
        return self._picamera is not None and self._picamera.started

    @contextmanager
    def _streaming_picamera(self, pause_stream: bool = False) -> Iterator[Picamera2]:
        """Lock access to picamera and return the underlying ``Picamera2`` instance.

        Optionally the stream can be paused to allow updating the camera settings.

        :param pause_stream: If False the ``Picamera2`` instance is simply yielded.
            If True:

                * Stop the MJPEG Stream
                * Yield the ``Picamera2`` instance for function calling the context manager to
                    make changes.
                * On closing of the context manager the stream will restart.
        """
        already_streaming = self.stream_active
        streaming_mode = self.streaming_mode
        with self._picamera_lock:
            if pause_stream and already_streaming:
                self._stop_streaming(stop_web_stream=False)
            try:
                yield self._picamera
            finally:
                if pause_stream and already_streaming:
                    self._start_streaming(streaming_mode)

    def __exit__(
        self,
        exc_type: type[BaseException],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Close the picamera connection when the Thing context manager is closed."""
        self._stop_streaming()
        with self._streaming_picamera() as cam:
            cam.close()
        del self._picamera
        super().__exit__(exc_type, exc_value, traceback)

    @abstractmethod
    @lt.property
    def streaming_modes(self) -> Mapping[str, PiCamera2StreamingMode]:
        """Modes the camera can stream in."""

    @abstractmethod
    @lt.property
    def capture_modes(self) -> Mapping[str, PiCamera2CaptureMode]:
        """Modes the camera can use for capturing."""

    def _create_picam_config_from_mode_info(
        self,
        picam: Picamera2,
        controls: dict[str, Any],
        mode_info: PiCamera2StreamingMode,
    ) -> dict[str, Any]:
        """Create the dictionary to pass to ``Picamera2.configure``.

        :param controls: The controls for the camera. Note that running
            ``_get_persistent_controls()`` should be done before getting the picamera
            lock to ensure that the current settings are read from the camera.
        """
        if mode_info.scaler_crop is not None:
            controls["ScalerCrop"] = mode_info.scaler_crop
        if mode_info.frame_rate is not None:
            controls["FrameRate"] = mode_info.frame_rate

        stream_config = picam.create_video_configuration(
            main={"size": mode_info.main_resolution},
            lores={"size": mode_info.lores_resolution, "format": "YUV420"},
            sensor=mode_info.sensor_mode_dict,
            controls=controls,
        )
        stream_config["buffer_count"] = mode_info.buffer_count
        return stream_config

    def _start_streaming(self, mode: str = "default") -> None:
        """Start the MJPEG stream. This is where persistent controls are sent to camera.

        Sets the camera stream resoltuons based on input mode

        Create two streams:

        * ``lores_mjpeg_stream`` for autofocus at ``lores_resolution``
        * ``mjpeg_stream`` for preview. This is at the ``main_resolution`` unless
            ``use_lores_as_preview`` is True, in which case it is a copy of
            ``lores_mjpeg_stream``
        """
        if mode not in self.streaming_modes:
            raise ValueError(f"Unknown mode {mode}")
        self.streaming_mode = mode

        # This must be before getting the picamera hardware lock.
        controls = self._get_persistent_controls()

        mode_info = self.streaming_modes[mode]

        with self._streaming_picamera() as picam:
            try:
                if picam.started:
                    picam.stop()
                    picam.stop_encoder()  # make sure there are no other encoders going
                stream_config = self._create_picam_config_from_mode_info(
                    picam=picam,
                    controls=controls,
                    mode_info=mode_info,
                )
                picam.configure(stream_config)
                LOGGER.info("Starting picamera MJPEG stream...")
                stream_name = "lores" if mode_info.use_lores_as_preview else "main"
                encoder = MJPEGEncoder(self.mjpeg_bitrate)
                picam.start_recording(
                    encoder,
                    PicameraStreamOutput(self.mjpeg_stream, encoder),
                    name=stream_name,
                )
                encoder = MJPEGEncoder(100000000)
                picam.start_encoder(
                    encoder,
                    PicameraStreamOutput(self.lores_mjpeg_stream, encoder),
                    name="lores",
                )
            except Exception as e:
                LOGGER.error(f"Error while starting preview: {e}.")
            else:
                self.stream_active = True
                LOGGER.debug("Started MJPEG stream in %s mode.", mode)

    def _stop_streaming(self, stop_web_stream: bool = True) -> None:
        """Stop the MJPEG stream."""
        with self._streaming_picamera() as picam:
            try:
                picam.stop_recording()  # This should also stop the extra lores encoder
            except Exception as e:
                LOGGER.info("Stopping recording failed")
                LOGGER.exception(e)
            else:
                self.stream_active = False
                if stop_web_stream:
                    self.mjpeg_stream.stop()
                    self.lores_mjpeg_stream.stop()
                LOGGER.info("Stopped MJPEG stream.")

            # Adding a sleep to prevent camera getting confused by rapid commands
            time.sleep(self._sensor_info.short_pause)

    @lt.action
    def discard_frames(self) -> None:
        """Discard frames so that the next frame captured is fresh."""
        with self._streaming_picamera() as cam:
            cam.capture_metadata()

    @contextmanager
    def _ensure_mode_for_capture(
        self, capture_mode_info: PiCamera2CaptureMode
    ) -> Iterator[Picamera2]:
        """Ensure in correct mode for capture.

        If the camera is already in the correct mode, the stream isn't paused and
        this is the same as using ``self._streaming_picamera()``.

        Otherwise, pause stream, and switch mode. Mode is reset and stream
        restarts after the context manager closes.
        """
        required_streaming_mode = capture_mode_info.streaming_mode
        if (
            required_streaming_mode is None
            or required_streaming_mode == self.streaming_mode
        ):
            with self._streaming_picamera() as cam:
                yield cam
        else:
            streaming_mode_info = self.streaming_modes[required_streaming_mode]

            # This must be before getting the picamera hardware lock.
            controls = self._get_persistent_controls()

            with self._streaming_picamera(pause_stream=True) as cam:
                LOGGER.debug("Reconfiguring camera for full resolution capture")
                stream_config = self._create_picam_config_from_mode_info(
                    picam=cam,
                    controls=controls,
                    mode_info=streaming_mode_info,
                )
                cam.configure(stream_config)
                cam.start()
                time.sleep(self._sensor_info.short_pause)
                yield cam

    def _capture_image(self, capture_mode: str = "standard") -> Image.Image:
        """Acquire one image from the camera and return it as a PIL Image.

        :param capture_mode: The capture mode to use. See the description field of each
            mode in ``capture_modes`` for more detail.

        :raises TimeoutError: if this time is exceeded during capture.
        """
        capture_mode = self._validate_capture_mode(capture_mode)
        capture_mode_info = self.capture_modes[capture_mode]

        with self._ensure_mode_for_capture(capture_mode_info) as cam:
            return cam.capture_image(
                capture_mode_info.stream_name, wait=capture_mode_info.timeout
            )

    @lt.action
    def capture_as_array(
        self,
        capture_mode: str = "standard",
        raw: bool = False,
    ) -> NDArray:
        """Acquire one image from the camera and return as an array.

        This function will produce a nested list containing an uncompressed RGB image.
        It's likely to be highly inefficient - raw and/or uncompressed captures using
        binary image formats will be added in due course.

        :param capture_mode: (Optional) The name of the capture mode as defined by the
            camera.
        :param raw: Whether to capture RAW data. Capturing RAW data may ignore some
            of the camera mode settings.

        :raises TimeoutError: if this time is exceeded during capture.
        """
        if raw:
            # Raw cannot use _capture_image.
            capture_mode = self._validate_capture_mode(capture_mode)
            capture_mode_info = self.capture_modes[capture_mode]

            with self._ensure_mode_for_capture(capture_mode_info) as cam:
                return cam.capture_array(name="raw", wait=capture_mode_info.timeout)

        # Note that internally the PiCamera creates a PIL image and then converts to
        # numpy with ``np.array(Image.open(io.BytesIO(self.make_buffer(name))))``.
        # As such we use _capture_image to get an Image from the picamera and return
        # as array
        return np.array(self._capture_image(capture_mode))

    @lt.property
    def camera_configuration(self) -> Mapping:
        """The "configuration" dictionary of the picamera2 object.

        The "configuration" sets the resolution and format of the camera's streams.
        Together with the "tuning" it determines how the sensor is configured and
        how the data is processed.

        Note that the configuration may be modified when taking still images, and
        this property refers to whatever configuration is currently in force -
        usually the one used for the preview stream.
        """
        with self._streaming_picamera() as cam:
            return cam.camera_configuration()

    @lt.property
    def capture_metadata(self) -> dict:
        """Return the metadata from the camera."""
        with self._streaming_picamera() as cam:
            return cam.capture_metadata()

    @lt.action
    def auto_expose_from_minimum(
        self,
        target_white_level: Optional[int] = None,
        percentile: float = 99.9,
    ) -> None:
        """Adjust exposure until a the target white level is reached.

        Starting from the minimum exposure, gradually increase exposure until
        the image reaches the specified white level.

        :param target_white_level: Raw target white level, this should be an integer
            within the range set by the bit-depth of the camera sensor (10-bit for
            PiCamera v2, 12 Bit for Picamera HQ. If None the default will be used for
            the current sensor. This is approximately 40% saturated, but after gamma
            curve is applied, the pixel values will have a value around 200.
        :param percentile: The percentile to use instead of maximum. Default 99.9. When
            calculating the brightest pixel, a percentile is used rather than the
            maximum in order to be robust to a small number of noisy/bright pixels.
        """
        if target_white_level is None:
            target_white_level = self._sensor_info.default_target_white_level

        with self._streaming_picamera(pause_stream=True) as cam:
            recalibrate_utils.adjust_shutter_and_gain_from_raw(
                cam,
                self._sensor_info,
                target_white_level=target_white_level,
                percentile=percentile,
            )

    @lt.action
    def calibrate_lens_shading(self) -> None:
        """Take an image and use it for flat-field correction.

        This method requires an empty (i.e. bright) field of view. It will take
        a raw image and effectively divide every subsequent image by the current
        one. This uses the camera's "tuning" file to correct the preview and
        the processed images. It should not affect raw images.
        """
        with self._streaming_picamera(pause_stream=True) as cam:
            # Suppress lint warning that L, Cr, and Cb are not lowercase, as these are
            # the standard mathematical terms for:
            # luminance (L), red-difference chroma (Cr), and blue-difference chroma
            # (Cb).
            L, Cr, Cb = recalibrate_utils.lst_from_camera(cam, self._sensor_info)  # noqa: N806
            self.tuning = tf_utils.set_lst(
                self.tuning,
                luminance=L,
                cr=Cr,
                cb=Cb,
                colour_temp=tf_utils.CALIBRATED_COLOUR_TEMP,
            )

            # Re-initialise the picamera to reload the tuning file.
            self._initialise_picamera()

        self.colour_gains = tf_utils.get_colour_gains_from_lst(self.tuning)

    @lt.property
    def colour_correction_matrix(
        self,
    ) -> tuple[float, float, float, float, float, float, float, float, float]:
        """The ``colour_correction_matrix`` from the tuning file.

        This is broken out into its own property for convenience and compatibility with
        the micromanager API

        It is a 9 value tuple used to specify the 3x3 matrix that the GPU pipeline uses
        to convert from the camera R,G,B vector to the standard R,G,B.
        """
        return tuple(tf_utils.get_ccm(self.tuning))

    @colour_correction_matrix.setter  # type: ignore
    def colour_correction_matrix(
        self,
        value: tuple[float, float, float, float, float, float, float, float, float],
    ) -> None:
        self.tuning = tf_utils.set_ccm(self.tuning, value)

        if self._picamera is not None:
            with self._streaming_picamera(pause_stream=True):
                self._initialise_picamera()

    @lt.action
    def reset_ccm(self) -> None:
        """Overwrite the colour correction matrix in camera tuning with default values."""
        self.tuning = tf_utils.copy_algo_from_other_tuning(
            algo="rpi.ccm",
            base_tuning_file=self.tuning,
            copy_from=self.default_tuning,
        )

    @lt.property
    def gamma_correction(self) -> list[int]:
        """Return the gamma correction curve from the tuning file."""
        return tf_utils.get_gamma_curve(self.tuning)

    @lt.action
    def set_static_green_equalisation(self, offset: int = 65535) -> None:
        """Set the green equalisation to a static value.

        Green equalisation avoids the debayering algorithm becoming confused
        by the two green channels having different values, which is a problem
        when the chief ray angle isn't what the sensor was designed for, and
        that's the case in e.g. a microscope using camera module v2.

        A value of 0 here does nothing, a value of 65535 is maximum correction.
        """
        with self._streaming_picamera(pause_stream=True):
            self.tuning = tf_utils.set_static_geq(self.tuning, offset)
            self._initialise_picamera()

    @lt.action
    def set_ce_enable_to_off(self) -> None:
        """Set the contrast enhancement to disabled.

        Adaptive contrast enhancement modifies settings to adapt to each field
        of view, causing inconsistent settings when capturing.
        """
        with self._streaming_picamera(pause_stream=True):
            self.tuning = tf_utils.set_ce_to_disabled(self.tuning)
            self._initialise_picamera()

    @lt.action
    def full_auto_calibrate(self) -> None:
        """Perform a full auto-calibration.

        This function will call the other calibration actions in sequence:

        * ``flat_lens_shading`` to disable flat-field
        * ``auto_expose_from_minimum``
        * ``set_static_green_equalisation`` to set geq offset to max
        * ``calibrate_lens_shading`` (also sets colour gains for white balance)
        * ``set_background``
        """
        self.flat_lens_shading()
        self.auto_expose_from_minimum()
        self.set_static_green_equalisation()
        self.set_ce_enable_to_off()
        self.calibrate_lens_shading()
        if self.background_detector is not None:
            for _i in range(3):
                try:
                    time.sleep(self._sensor_info.long_pause)
                    self.set_background()
                    # Return if background is set
                    return
                except ChannelBlankError:
                    # If channel is blank, sleep a second and try again.
                    pass
        raise RuntimeError("Couldn't set background")

    @lt.property
    def primary_calibration_actions(self) -> list[ActionButton]:
        """The calibration actions for both calibration wizard and settings panel."""
        return [
            action_button_for(
                self,
                "full_auto_calibrate",
                submit_label="Full Auto-Calibrate",
                can_terminate=False,
                requires_confirmation=True,
                confirmation_message=(
                    "Start recalibration? This may take a while, and the microscope "
                    "will be locked during this time."
                ),
                notify_on_success=True,
                success_message="Finished recalibration.",
            ),
        ]

    @lt.property
    def secondary_calibration_actions(self) -> list[ActionButton]:
        """The calibration actions that appear only in settings panel."""
        return [
            action_button_for(
                self,
                "auto_expose_from_minimum",
                submit_label="Auto Gain & Shutter Speed",
                can_terminate=False,
                button_primary=False,
            ),
            action_button_for(
                self,
                "calibrate_lens_shading",
                submit_label="Auto Flat Field Correction",
                can_terminate=False,
                button_primary=False,
                requires_confirmation=True,
                confirmation_message=(
                    "Is the microscope looking at an evenly illuminated, empty field "
                    "of view? If not, the current image will show through in any "
                    "images captured afterwards."
                ),
            ),
            action_button_for(
                self,
                "flat_lens_shading",
                submit_label="Disable Flat Field Correction",
                can_terminate=False,
                button_primary=False,
            ),
            action_button_for(
                self,
                "flat_lens_shading_chrominance",
                submit_label="Disable Flat Field Chrominance",
                can_terminate=False,
                button_primary=False,
            ),
            action_button_for(
                self,
                "reset_lens_shading",
                submit_label="Reset Flat Field Correction",
                can_terminate=False,
                button_primary=False,
            ),
        ]

    @lt.property
    def manual_camera_settings(self) -> list[PropertyControl]:
        """The camera settings to expose as property controls in the settings panel."""
        return [
            property_control_for(
                self,
                "exposure_time",
                label="Exposure Time (0-33251)",
                read_back=True,
                read_back_delay=1000,
            ),
            property_control_for(
                self,
                "analogue_gain",
                label="Analogue Gain",
                read_back=True,
                read_back_delay=1000,
            ),
            property_control_for(
                self,
                "colour_gains",
                label="Colour Gains",
                read_back=True,
                read_back_delay=1000,
            ),
        ]

    @lt.property
    def lens_shading_tables(self) -> Optional[tf_utils.LensShadingModel]:
        """The current lens shading (i.e. flat-field correction).

        Return the current lens shading correction, as three 2D lists each with
        dimensions 16x12.

        The colour temperature is returned. If the colour temperature us 5000 then this
        means the lens shading tables have been calibrated (with our illumination which
        has a 5000k colour temperature). Other numbers are set when flatening or
        resetting the table.
        """
        return tf_utils.get_lst(self.tuning)

    @lt.action
    def flat_lens_shading(self) -> None:
        """Disable flat-field correction.

        This method will set a completely flat lens shading table. It is not the
        same as the default behaviour, which is to use an adaptive lens shading
        table.

        This flat table is used to take an image with no lens shading so that the
        correct lens shading table can be calibrated.
        """
        with self._streaming_picamera(pause_stream=True):
            self.tuning = tf_utils.flatten_lst(self.tuning)
            self._initialise_picamera()

    @lt.action
    def flat_lens_shading_chrominance(self) -> None:
        """Disable flat-field correction for colour only.

        This method will set the chrominance of the lens shading table to be
        flat, i.e. we'll correct vignetting of intensity, but not any change in
        colour across the image.
        """
        with self._streaming_picamera(pause_stream=True):
            self.tuning = tf_utils.flatten_lst(self.tuning, keep_luminance=True)
            self._initialise_picamera()

    @lt.action
    def reset_lens_shading(self) -> None:
        """Revert to default lens shading settings.

        This method will restore the default "adaptive" lens shading method used
        by the Raspberry Pi camera.
        """
        with self._streaming_picamera(pause_stream=True):
            self.tuning = tf_utils.copy_algo_from_other_tuning(
                algo="rpi.alsc",
                base_tuning_file=self.tuning,
                copy_from=self.default_tuning,
            )
            self._initialise_picamera()

    @property
    def thing_state(self) -> Mapping[str, Any]:
        """Update generic camera metadata with Picamera-specific data."""
        state = dict(super().thing_state)
        state["camera_board"] = self._camera_board
        state["tuning"] = {
            "exposure_time": self.exposure_time,
            "colour_gains": self.colour_gains,
            "analogue_gain": self.analogue_gain,
            "gamma_correction": self.gamma_correction,
        }
        return state


class PiCameraV2(StreamingPiCamera2):
    """A Thing that provides and interface to the Raspberry Pi Camera V2."""

    _camera_board = "picamera_v2"
    _sensor_info = recalibrate_utils.IMX219_SENSOR_INFO

    @lt.property
    def streaming_modes(self) -> Mapping[str, PiCamera2StreamingMode]:
        """Modes the camera can stream in."""
        return {
            "default": PiCamera2StreamingMode(
                description=(
                    "The standard mode that balances capture resolution and stream "
                    "size."
                ),
                main_resolution=(820, 616),
                lores_resolution=(320, 240),
                sensor_mode_resolution=(3280, 2464),
                bit_depth=10,
                use_lores_as_preview=False,
            ),
            "full_resolution": PiCamera2StreamingMode(
                description=(
                    "Streaming the camera in 8MP full resolution. The preview stream "
                    "sent to the UI will be the low resolution (lores) stream. "
                    "This allows better image capture at expense of preview quality."
                ),
                main_resolution=(3280, 2464),
                lores_resolution=(320, 240),
                sensor_mode_resolution=(3280, 2464),
                bit_depth=10,
                use_lores_as_preview=True,
            ),
        }

    @lt.property
    def capture_modes(self) -> Mapping[str, PiCamera2CaptureMode]:
        """Modes the camera can use for capturing."""
        return {
            "standard": PiCamera2CaptureMode(
                description=(
                    "The standard mode, 2MP image created from downsampling an 8MP "
                    "capture."
                ),
                save_resolution=(1640, 1232),
                streaming_mode="full_resolution",
                stream_name="main",
            ),
            "full": PiCamera2CaptureMode(
                description="A full resolution 8MP capture.",
                streaming_mode="full_resolution",
                stream_name="main",
            ),
            "quick": PiCamera2CaptureMode(
                description="Capture without altering the stream settings.",
                streaming_mode=None,
                stream_name="main",
            ),
        }


class PiCameraHQ(StreamingPiCamera2):
    """A Thing that provides and interface to the Raspberry Pi Camera HQ."""

    _camera_board = "picamera_hq"
    _sensor_info = recalibrate_utils.IMX477_SENSOR_INFO

    @lt.property
    def streaming_modes(self) -> Mapping[str, PiCamera2StreamingMode]:
        """Modes the camera can stream in."""
        return {
            "default": PiCamera2StreamingMode(
                description=(
                    "The standard mode that balances capture resolution and stream "
                    "size."
                ),
                main_resolution=(700, 700),
                lores_resolution=(350, 350),
                sensor_mode_resolution=(4056, 3040),
                bit_depth=12,
                use_lores_as_preview=False,
                scaler_crop=(628, 120, 2800, 2800),
            ),
            "full_resolution": PiCamera2StreamingMode(
                description=(
                    "Streaming the camera with an 8MP area in full-resolution area "
                    "from the centre of the 12MP sensor.. The preview stream sent to "
                    "the UI will be the low resolution (lores) stream. This allows "
                    "better image capture at expense of preview quality."
                ),
                main_resolution=(2800, 2800),
                lores_resolution=(350, 350),
                sensor_mode_resolution=(4056, 3040),
                bit_depth=12,
                use_lores_as_preview=True,
                scaler_crop=(628, 120, 2800, 2800),
            ),
            "fast_preview": PiCamera2StreamingMode(
                description=(
                    "High frame rate mode for autofocus sweeps. The sensor is read "
                    "out 2x2 binned at 2028x1520, raising the stream from ~12 to "
                    "~37 fps. Stills requested in full-res capture modes "
                    "automatically reconfigure and restore."
                ),
                main_resolution=(700, 700),
                lores_resolution=(350, 350),
                sensor_mode_resolution=(2028, 1520),
                bit_depth=10,
                use_lores_as_preview=False,
                scaler_crop=(314, 60, 1400, 1400),
            ),
            "crop990": PiCamera2StreamingMode(
                description=(
                    "Cropped 1332x990 sensor readout for high frame rate autofocus "
                    "experiments. Narrower field of view than fast_preview; same "
                    "lores stream size."
                ),
                main_resolution=(700, 700),
                lores_resolution=(350, 350),
                sensor_mode_resolution=(1332, 990),
                bit_depth=10,
                use_lores_as_preview=False,
                scaler_crop=None,
                frame_rate=120.0,
            ),
        }

    @lt.property
    def capture_modes(self) -> Mapping[str, PiCamera2CaptureMode]:
        """Modes the camera can use for capturing."""
        return {
            "standard": PiCamera2CaptureMode(
                description=(
                    "The standard mode, 2MP image created from downsampling an 8MP "
                    "capture from the centre of the 12MP sensor."
                ),
                save_resolution=(1400, 1400),
                streaming_mode="full_resolution",
                stream_name="main",
            ),
            "full": PiCamera2CaptureMode(
                description=(
                    "An 8MP full resolution capture from the centre of the 12MP sensor."
                ),
                streaming_mode="full_resolution",
                stream_name="main",
            ),
            "quick": PiCamera2CaptureMode(
                description="Capture without altering the stream settings.",
                streaming_mode=None,
                stream_name="main",
            ),
        }
