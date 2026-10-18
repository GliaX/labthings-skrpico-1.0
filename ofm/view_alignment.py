"""View-alignment detection: measures how stage X/Y jogs move the image
and solves the preview rotation (and mirror) so on-screen navigation
matches the rotated view. Provided as a mixin for CameraStageMapper to
keep the diff in camera_stage_mapping.py minimal.
"""

from typing import List, Tuple

import labthings_fastapi as lt
import numpy as np
from pydantic import Field


def _pixel_shift(reference: np.ndarray, moved: np.ndarray) -> List[float]:
    def gray(img: np.ndarray) -> np.ndarray:
        g = img[..., 1].astype(np.float32)
        while min(g.shape) > 512:
            g = g[::2, ::2]
        return g - g.mean()

    a = gray(reference)
    b = gray(moved)
    fa = np.fft.rfft2(a)
    fb = np.fft.rfft2(b)
    corr = np.fft.irfft2(fa * np.conj(fb), s=a.shape)
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    dy, dx = peak
    h, w = a.shape
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    return [float(dx), float(dy)]


def _stage_to_image_rotation(
        vx_px: List[float],
        vy_px: List[float],
) -> Tuple[int, bool]:
    """Quantized rotation from stage axes to image axes, plus mirror.

    vx_px/vy_px are the measured image shifts (screen px, y down) for
    +X and +Y stage moves. Returns (rotation_deg, mirror) where
    rotation_deg is the +X shift angle quantized to 90 degrees.
    """
    det = vx_px[0] * vy_px[1] - vx_px[1] * vy_px[0]
    mirror = det < 0
    ang = round(np.degrees(np.arctan2(vx_px[1], vx_px[0])) / 90.0) * 90
    return int(ang) % 360, mirror


class ViewAlignmentMixin:
    """Mixin adding view-alignment detection to CameraStageMapper.

    Relies on the host Thing's ``_cam`` and ``_stage`` thing slots.
    """

    @lt.action
    def detect_view(
            self,
            step: int = Field(200, gt=0),
    ) -> dict:
        """Measure how stage X/Y moves displace the image.

        Moves +X and +Y by ``step`` units (returning after each),
        cross-correlating frames to measure the true image shift.

        :param step: stage units to move along each axis.
        :return: pixel shift vectors for +X and +Y, the stage-to-image
            rotation in degrees, and whether the optics mirror the image.
        """
        img0 = self._cam.capture_as_array()
        vx_px = self._measure_axis(img0, step, "x")
        vy_px = self._measure_axis(img0, step, "y")
        rot, mirror = _stage_to_image_rotation(vx_px, vy_px)
        return {
            "vx_px": vx_px,
            "vy_px": vy_px,
            "stage_to_image_rotation": rot,
            "mirror": mirror,
        }

    def _measure_axis(
            self,
            reference: np.ndarray,
            step: int,
            axis: str,
    ) -> List[float]:
        """Planned G1 move out and back along one stage axis; return
        the image shift in screen pixels (x right, y down)."""
        kwargs = {axis: step}
        self._stage.move_relative(**kwargs)
        moved = self._cam.capture_as_array()
        self._stage.move_relative(**{axis: -step})
        return _pixel_shift(reference, moved)
