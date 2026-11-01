"""View-alignment detection: measures how stage X/Y jogs move the image
and solves the preview rotation (and mirror) so on-screen navigation
matches the rotated view. Provided as a mixin for CameraStageMapper to
keep the diff in camera_stage_mapping.py minimal.

Peak selection is joint across the two axes: the true shift pair must
be near-perpendicular with similar magnitudes (rigid mapping), which
rejects decoy peaks from quasi-periodic samples without any external
prior. This makes repeated runs deterministic (idempotent).
"""

import math
from typing import List, Optional, Tuple

import labthings_fastapi as lt
import numpy as np
import time
from pydantic import Field


def _prep(img: np.ndarray) -> np.ndarray:
    g = img[..., 1].astype(np.float32)
    while min(g.shape) > 640:
        g = g[::2, ::2]
    return g - g.mean()


def _unwrap(v: int, n: int) -> int:
    return v - n if v > n // 2 else v


def _corr_and_peaks(
        reference: np.ndarray,
        moved: np.ndarray,
        k: int = 8,
        min_sep: float = 24.0,
) -> List[Tuple[float, float, float]]:
    """Cross-correlate two frames; return up to ``k`` peaks as
    (strength, dx, dy) in downsampled screen pixels, greedily
    de-duplicated by separation."""
    a = _prep(reference)
    b = _prep(moved)
    h, w = a.shape
    corr = np.fft.irfft2(
        np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=a.shape
    )
    order = np.argsort(corr.ravel())[::-1][: k * 8]
    peaks: List[Tuple[float, float, float]] = []
    for idx in order:
        j, i = divmod(int(idx), w)
        dy, dx = _unwrap(j, h), _unwrap(i, w)
        if any(
                math.hypot(dx - px, dy - py) < min_sep
                for _, px, py in peaks
        ):
            continue
        peaks.append((float(corr[j, i]), float(dx), float(dy)))
        if len(peaks) >= k:
            break
    return peaks


def _pick_pair(
        px_peaks: List[Tuple[float, float, float]],
        py_peaks: List[Tuple[float, float, float]],
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Choose the (vx, vy) peak pair that is jointly perpendicular
    (within 25 deg) and magnitude-matched (ratio < ~1.6). Returns
    None when no pair satisfies the rigid-mapping constraints."""
    best = None
    for _, xdx, xdy in px_peaks:
        magx = math.hypot(xdx, xdy)
        if magx < 2.0:
            continue
        angx = math.degrees(math.atan2(xdy, xdx))
        for _, ydx, ydy in py_peaks:
            magy = math.hypot(ydx, ydy)
            if magy < 2.0:
                continue
            angy = math.degrees(math.atan2(ydy, ydx))
            ang_err = abs((angx - angy) % 180.0 - 90.0)
            if ang_err > 25.0:
                continue
            ratio = max(magx, magy) / max(min(magx, magy), 1e-6)
            if ratio > 1.6:
                continue
            score = ang_err + 30.0 * abs(math.log(ratio))
            if best is None or score < best[0]:
                best = (score, (xdx, xdy), (ydx, ydy))
    if best is None:
        return None
    return best[1], best[2]


def _stage_to_image_rotation(
        vx_px: List[float],
        vy_px: List[float],
) -> Tuple[int, bool]:
    """Quantized rotation from stage axes to image axes, plus mirror."""
    det = vx_px[0] * vy_px[1] - vx_px[1] * vy_px[0]
    mirror = det < 0
    ang = round(np.degrees(np.arctan2(vx_px[1], vx_px[0])) / 90.0) * 90
    return int(ang) % 360, mirror


class ViewAlignmentMixin:
    """Mixin adding view-alignment detection to CameraStageMapper.

    Relies on the host Thing's ``_cam`` and ``_stage`` thing slots.
    """

    def _probe_step(self) -> int:
        """Probe distance in stage units: roughly a quarter of the
        field of view. Falls back to 200 without a calibration."""
        m = self.image_to_stage_displacement_matrix
        if m is None:
            return 200
        M = np.asarray(m)
        norms = [
            float(np.linalg.norm(M[:, k])) for k in range(2)
        ]
        if min(norms) <= 0:
            return 200
        try:
            px = min(self._cam.capture_as_array().shape[:2])
        except Exception:
            return 200
        return max(20, int(0.25 * px * min(norms)))

    def _move_gently(self, axis: str, step: int) -> None:
        """Move |step| units along ``axis`` in 4 sub-steps, returning
        exactly to the commanded net displacement."""
        sign = 1 if step >= 0 else -1
        total = abs(step)
        chunks = 4
        sub, rem = divmod(total, chunks)
        for i in range(chunks):
            d = sub + (rem if i == 0 else 0)
            if d:
                self._stage.move_relative(**{axis: sign * d})

    @lt.action
    def detect_view(
            self,
            step: int = Field(0, ge=0),
    ) -> dict:
        """Measure how stage X/Y moves displace the image.

        Moves +X and +Y (in small sub-steps, returning exactly after
        each), cross-correlating frames. The reported shift pair is
        selected jointly for perpendicularity and magnitude match, so
        decoy peaks from quasi-periodic samples are rejected and
        repeated runs give the same answer.

        :param step: stage units to move along each axis; 0 selects a
            quarter of the field of view automatically.
        :return: pixel shift vectors for +X and +Y, the stage-to-image
            rotation in degrees, whether the optics mirror the image,
            and whether the rigid-pair selection succeeded.
        """
        step = step if step > 0 else self._probe_step()
        img0 = self._cam.capture_as_array()
        self._move_gently("x", step)
        time.sleep(0.3)
        moved_x = self._cam.capture_as_array()
        self._move_gently("x", -step)
        px_peaks = _corr_and_peaks(img0, moved_x)
        self._move_gently("y", step)
        time.sleep(0.3)
        moved_y = self._cam.capture_as_array()
        self._move_gently("y", -step)
        py_peaks = _corr_and_peaks(img0, moved_y)
        pair = _pick_pair(px_peaks, py_peaks)
        rigid = pair is not None
        if rigid:
            vx, vy = pair
        else:
            vx = (px_peaks[0][1], px_peaks[0][2])
            vy = (py_peaks[0][1], py_peaks[0][2])
        rot, mirror = _stage_to_image_rotation(list(vx), list(vy))
        return {
            "vx_px": [vx[0], vx[1]],
            "vy_px": [vy[0], vy[1]],
            "stage_to_image_rotation": rot,
            "mirror": mirror,
            "step": step,
            "rigid_pair_found": rigid,
        }
