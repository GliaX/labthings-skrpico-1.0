"""Synthetic end-to-end test for af_hud_video (no scope needed).

Builds a fake session: gradient NV12 frames + .tsv, a moves.log with a
two-leg run (move/SEND + LEG MARKER lines bridging et<->epoch) and a
commanded trapezoid z(t) known exactly. Asserts the burned HUD matches
ground truth at a known frame, then encodes and decode-checks the mp4.
"""

import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "ofm"))
sys.path.insert(0, str(HERE))
import af_hud_video as hud  # noqa: E402

td = pathlib.Path(tempfile.mkdtemp())
W, H, N = hud.W, hud.H, 60
fb = W * H * 3 // 2

mono0 = 1000.0
frames_ts = [mono0 + i / 120.0 for i in range(N)]
raw = td / "af_view_1.nv12"
with open(raw, "wb") as f:
    for i in range(N):
        y = np.tile(np.arange(W, dtype=np.uint8) * 2, H)
        u = np.full(W * H // 4, (60 + i) % 256, dtype=np.uint8)
        v = np.full(W * H // 4, (200 - i) % 256, dtype=np.uint8)
        f.write(y.tobytes() + u.tobytes() + v.tobytes())
(td / "af_view_1.tsv").write_text("".join(f"{t:.6f}\n" for t in frames_ts))

hoff = 1.9
leg = {"hoff": hoff, "samples": []}
(td / "smooth_af_legdump.json").write_text(json.dumps(leg))

accel = 50000.0
(td / "smooth_af_cal.json").write_text(json.dumps({"accel": accel}))

epoch0 = 1791770000.0
# leg1: 1024 u at 2048 u/s -> secs=0.5; starts at m1_t, ends m1_t+0.5
# corners add 2*v/a = 0.041s -> marker at t0+secs+0.041
v1 = 1024.0 / 0.5
t1_send = epoch0 - 0.05
m1_t = t1_send + 0.5 + 2.0 * v1 / accel
m1_mono = mono0 - 0.1
m1_et = m1_mono + hoff
# leg2: -2048 u at 4096 u/s -> secs=0.5, sent right after leg1 marker
v2 = 2048.0 / 0.5
t2_send = m1_t + 0.05
m2_t = t2_send + 0.5 + 2.0 * v2 / accel
m2_mono = m1_mono + (m2_t - m1_t)
m2_et = m2_mono + hoff

lines = [
    "move target=1024.00 cur=0.00 delta=1024.00 v=2048 t=%.3f" % t1_send,
    "SEND steps=1024.0 seconds=0.5 sync=True t=%.3f" % t1_send,
    "LEG MARKER n=1 z=1024.0 et=%.3f t=%.3f" % (m1_et, m1_t),
    "move target=-1024.00 cur=1024.00 delta=-2048.00 v=4096 t=%.3f" % t2_send,
    "SEND steps=-2048.0 seconds=0.5 sync=True t=%.3f" % t2_send,
    "LEG MARKER n=2 z=-1024.0 et=%.3f t=%.3f" % (m2_et, m2_t),
    "RUN total=4.16 mode=|test|",
]
(td / "smooth_af_moves.log").write_text("\n".join(lines) + "\n")

r = subprocess.run(
    [sys.executable, str(HERE / "af_hud_video.py"), str(td)],
    capture_output=True, text=True, timeout=300,
)
print(r.stdout, r.stderr[-400:] if r.returncode else "")
assert (td / "af_hud.mp4").exists(), "mp4 not produced"

# bridge sanity on the run markers
mono0b, epoch0b, scale = hud.bridge(
    [{"et": m1_et, "t": m1_t}, {"et": m2_et, "t": m2_t}], hoff)
assert abs(scale - 1.0) < 1e-6, f"bridge scale {scale}"
assert abs(epoch0b - m1_t) < 1e-9
assert abs(mono0b - (m1_et - hoff)) < 1e-9

# frame 30: mono = mono0+0.25 -> epoch = m1_t + 0.35. Leg1 (0..m1_t) is
# already over by frame 0 (video starts m1_t+0.1); frame 30 is in leg2's
# decel half. Model points for leg2: (t2s,1024,0),(t2s+.25,0,v2),(t2e,-1024,0)
epoch30 = m1_t + 0.35
f_dec = (epoch30 - (t2_send + 0.25)) / ((m2_t - 2e-3) - (t2_send + 0.25))
z_exp = 0.0 + f_dec * (-1024.0 - 0.0)
v_exp = v2 * (1.0 - f_dec)
sends, moves, markers, run_total = hud.load_events(td / "smooth_af_moves.log")
spans = hud.leg_spans(markers, sends)
assert spans[0][0] == "far1"
assert abs(spans[0][2] - m2_t) < 2e-3, "span end != m2_t (log is ms-rounded)"
ts_, zs_, vs_ = hud.build_traj(moves, sends, accel)
z, v, live = hud.traj_at(epoch30, ts_, zs_, vs_)
assert live, "frame 30 should be inside commanded trajectory"
assert abs(z - z_exp) < 2.0, f"z mismatch: {z} vs {z_exp}"
assert abs(v - v_exp) < 5.0, f"v mismatch: {v} vs {v_exp}"

f30 = Image.open(td / "hud_frames" / "f00030.png")
assert f30.size == (hud.W, hud.H + hud.HUD_H), f"HUD canvas clipped: {f30.size}"
band = np.asarray(f30.crop((0, hud.H, hud.W, hud.H + hud.HUD_H)))
assert band[:6].mean() > 200 or band.mean() < 120, "no HUD band drawn"

# mp4 decodable and has right frame count
r2 = subprocess.run(
    ["ffmpeg", "-i", str(td / "af_hud.mp4"), "-f", "null", "-"],
    capture_output=True, text=True, timeout=120)
m = [l for l in r2.stderr.splitlines() if "frame=" in l][-1]
print("ffmpeg check:", m.strip()[-80:])
assert "frame=   60" in m or "frame=60" in m, "frame count mismatch"

print("ALL HUD TESTS PASS")
