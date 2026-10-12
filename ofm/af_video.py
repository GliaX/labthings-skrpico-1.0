"""Capture the AF lores NV12 stream to disk and encode it to mp4.

Arms a sink inside the camera's AF pre-callback (via the picamera module's
_AF_SINK hook) so the frames that normally never reach the encoder are
captured. Disk-backed and frame-capped; pure logic lives in
af_video_core (unit-tested without labthings).
"""

import os
import subprocess
import threading
import time

import labthings_fastapi as lt
from pydantic import BaseModel, Field

try:
    from . import af_video_core as _core
except ImportError:
    import af_video_core as _core

try:
    from .af_video_core import (
        DEFAULT_MAX_FRAMES,
        expected_frame_count,
        lores_h,
        lores_w,
        parse_ts,
    )
except ImportError:
    from af_video_core import (
        DEFAULT_MAX_FRAMES,
        expected_frame_count,
        lores_h,
        lores_w,
        parse_ts,
    )


class ArmParams(BaseModel):
    max_frames: int = Field(default=DEFAULT_MAX_FRAMES, gt=0, le=20000)


class ArmResult(BaseModel):
    armed: bool
    raw_path: str
    ts_path: str
    max_frames: int


class DisarmResult(BaseModel):
    armed: bool
    frames: int


class EncodeParams(BaseModel):
    fps: float = Field(default=120.0, gt=0)
    crf: int = Field(default=18, ge=0, le=51)


class EncodeResult(BaseModel):
    video_path: str
    frames: int
    bytes_out: int
    ffmpeg: str


class AfVideoThing(lt.Thing):
    """Captures the AF lores NV12 stream and encodes it to video."""

    def __init__(self, thing_server_interface=None):
        super().__init__(thing_server_interface=thing_server_interface)
        self._lock = threading.Lock()
        self._f = None
        self._ts_f = None
        self._raw_path = None
        self._ts_path = None
        self._count = 0
        self._max = 0

    def _sink(self, ts, arr):
        with self._lock:
            if self._f is None:
                return
            self._f.write(_core.compact_nv12(arr))
            self._ts_f.write("%.6f\n" % ts)
            self._count += 1
            if self._count >= self._max:
                self._close()

    def _close(self):
        if self._f is not None:
            self._f.close()
            self._f = None
        if self._ts_f is not None:
            self._ts_f.close()
            self._ts_f = None

    @lt.action
    def arm(self, params: ArmParams) -> ArmResult:
        """Start capturing AF lores frames to disk."""
        from openflexure_microscope_server.things.camera import (
            picamera as _pcam,
        )

        with self._lock:
            self._close()
            base = "/home/admin/af_view_%d" % int(time.time())
            self._raw_path = base + ".nv12"
            self._ts_path = base + ".tsv"
            self._f = open(self._raw_path, "wb")
            self._ts_f = open(self._ts_path, "w")
            self._count = 0
            self._max = params.max_frames
            _pcam._AF_SINK = self._sink
            return ArmResult(
                armed=True,
                raw_path=self._raw_path,
                ts_path=self._ts_path,
                max_frames=self._max,
            )

    @lt.action
    def disarm(self) -> DisarmResult:
        """Stop capturing; returns the captured frame count."""
        from openflexure_microscope_server.things.camera import (
            picamera as _pcam,
        )

        _pcam._AF_SINK = None
        with self._lock:
            self._close()
            return DisarmResult(armed=False, frames=self._count)

    @lt.action
    def encode(self, params: EncodeParams) -> EncodeResult:
        """Wrap the captured raw NV12 into an mp4 (h264, yuv420p)."""
        with self._lock:
            raw, ts, n = self._raw_path, self._ts_path, self._count
            self._close()
        if raw is None or n == 0:
            raise RuntimeError("nothing captured; arm first")
        if os.path.getsize(raw) != n * (lores_w * lores_h * 3 // 2):
            raise RuntimeError(
                f"raw size disagrees with frame count {n}"
            )
        out = raw[:-5] + ".mp4"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "nv12",
            "-s", f"{lores_w}x{lores_h}", "-r", f"{params.fps}",
            "-i", raw,
            "-c:v", "libx264", "-crf", str(params.crf),
            "-pix_fmt", "yuv420p", out,
        ]
        subprocess.run(cmd, check=True, timeout=300)
        size = os.path.getsize(out)
        with open(ts) as f:
            n_ts = len(parse_ts(f.read()))
        if n_ts != n:
            raise RuntimeError(
                f"frame/ts mismatch: {n} frames vs {n_ts} timestamps"
            )
        return EncodeResult(
            video_path=out, frames=n, bytes_out=size, ffmpeg=" ".join(cmd)
        )
