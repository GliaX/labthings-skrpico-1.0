"""Unit tests for af_video_core (pure helpers, no labthings/camera needed)."""

import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from af_video_core import (
    FRAME_BYTES,
    compact_nv12,
    expected_frame_count,
    lores_h,
    lores_w,
    parse_ts,
)

rng = np.random.default_rng(7)

# --- compact_nv12: stride padding is trimmed, planes preserved ---
stride = 384  # plausible libcamera stride > width
y_h = lores_h
uv_h = lores_h // 2
y = rng.integers(0, 256, size=(y_h, stride), dtype=np.uint8)
uv = rng.integers(0, 256, size=(uv_h, stride), dtype=np.uint8)
arr = np.vstack([y, uv])
blob = compact_nv12(arr)
assert len(blob) == FRAME_BYTES, (len(blob), FRAME_BYTES)
y2 = np.frombuffer(blob[: y_h * lores_w], dtype=np.uint8).reshape(y_h, lores_w)
uv2 = np.frombuffer(blob[y_h * lores_w:], dtype=np.uint8).reshape(uv_h, lores_w)
assert (y2 == y[:, :lores_w]).all()
assert (uv2 == uv[:, :lores_w]).all()

# exact-width arrays pass through unchanged
arr2 = np.vstack([
    rng.integers(0, 256, size=(y_h, lores_w), dtype=np.uint8),
    rng.integers(0, 256, size=(uv_h, lores_w), dtype=np.uint8),
])
assert compact_nv12(arr2) == arr2.tobytes()

print("compact_nv12: OK")

# --- parse_ts: floats, blank-line tolerant, strict on garbage ---
ts = parse_ts("12.5\n13.0\n\n14.25\n")
assert ts == [12.5, 13.0, 14.25]
for bad in ("1.0\nabc\n", "1.0\n1.2.3\n", "1,5\n"):
    try:
        parse_ts(bad)
        raise SystemExit(f"parse_ts accepted garbage: {bad!r}")
    except ValueError:
        pass
print("parse_ts: OK")

# --- expected_frame_count ---
assert expected_frame_count(FRAME_BYTES * 5) == 5
assert expected_frame_count(0) == 0
for bad in (1, FRAME_BYTES * 2 + 7):
    try:
        expected_frame_count(bad)
        raise SystemExit(f"expected_frame_count accepted {bad}")
    except ValueError:
        pass
print("expected_frame_count: OK")

# --- end-to-end against real ffmpeg if present ---
import shutil
import subprocess

ffmpeg = shutil.which("ffmpeg")
if ffmpeg:
    n = 30
    frames = rng.integers(0, 256, size=(n, y_h + uv_h, lores_w), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as td:
        raw = os.path.join(td, "v.nv12")
        with open(raw, "wb") as f:
            for fr in frames:
                f.write(fr.tobytes())
        out = os.path.join(td, "v.mp4")
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "nv12",
             "-s", f"{lores_w}x{lores_h}", "-r", "120", "-i", raw,
             "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", out],
            check=True, timeout=120,
        )
        assert os.path.getsize(out) > 100
        # decode one frame back and check dimensions
        dec = subprocess.run(
            [ffmpeg, "-loglevel", "error", "-i", out, "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "nv12", "-"],
            capture_output=True, timeout=120,
        )
        frame = dec.stdout
        assert len(frame) == FRAME_BYTES, (len(frame), FRAME_BYTES)
        row = np.frombuffer(frame[: lores_w], dtype=np.uint8)
        ref = np.frombuffer(frames[0].tobytes()[:lores_w], dtype=np.uint8)
        assert (row == ref).all(), "decoded first row does not match source"
    print(f"ffmpeg round-trip ({n} frames): OK")
else:
    print("ffmpeg not found; skipping round-trip test")

print("ALL AF-VIDEO CORE TESTS PASS")
