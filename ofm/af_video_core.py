"""Pure helpers for the AF point-of-view video recorder (no labthings)."""

import numpy as np

lores_w = 350
lores_h = 350
FRAME_BYTES = lores_w * lores_h * 3 // 2
DEFAULT_MAX_FRAMES = 1200


def compact_nv12(arr: np.ndarray) -> bytes:
    """Compact a stride-padded NV12 lores array to dense width x height.

    libcamera pads each plane row to the hardware stride; ffmpeg's rawvideo
    input needs the dense 350x350 frame, so both planes are trimmed.
    """
    y_rows = arr.shape[0] * 2 // 3
    y = np.ascontiguousarray(arr[:y_rows, :lores_w])
    uv = np.ascontiguousarray(arr[y_rows:, :lores_w])
    return y.tobytes() + uv.tobytes()


def parse_ts(text: str):
    """Parse the timestamp TSV into a list of floats; strict on garbage."""
    out = []
    for i, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        out.append(float(line))
    return out


def expected_frame_count(raw_size: int) -> int:
    if raw_size % FRAME_BYTES:
        raise ValueError(
            f"raw size {raw_size} not a multiple of frame size {FRAME_BYTES}"
        )
    return raw_size // FRAME_BYTES
