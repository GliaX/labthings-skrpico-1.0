"""Burn a telemetry HUD onto the AF point-of-view video.

Inputs (one AF session from the scope): af_view_*.nv12 + .tsv (frames and
sensor timestamps), smooth_af_moves.log (SEND/move/LEG MARKER lines),
smooth_af_cal.json (accel). The video timestamps are monotonic-domain and
the moves.log is epoch-domain; they are bridged with the LEG MARKER lines
of the selected run (each carries both et and epoch t; et-hoff is
monotonic). z/v shown are the COMMANDED trajectory reconstructed from the
send trapezoids using the calibrated accel, tagged 'cmd' on the HUD.

Usage: python af_hud_video.py <dir> [--f0 N] [--f1 N] [--out X]
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 350, 350
HUD_H = 64
FRAME_BYTES = W * H * 3 // 2
FONT = None
LEG_NAMES = ["far1", "far2", "claw->tiny", "tiny", "park"]


def make_font():
    for p in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, 15)
        except Exception:
            pass
    return ImageFont.load_default()


def nv12_to_rgb(buf):
    arr = np.frombuffer(buf, dtype=np.uint8)
    y = arr[: W * H].reshape(H, W)
    c = arr[W * H:].reshape(H // 2, W // 2, 2)
    u = c[:, :, 0].astype(np.float32).repeat(2, axis=0).repeat(2, axis=1)
    v = c[:, :, 1].astype(np.float32).repeat(2, axis=0).repeat(2, axis=1)
    yf = y.astype(np.float32)
    r = yf + 1.402 * (v - 128)
    g = yf - 0.344136 * (u - 128) - 0.714136 * (v - 128)
    b = yf + 1.772 * (u - 128)
    return np.clip(np.dstack([r, g, b]), 0, 255).astype(np.uint8)


def load_events(path):
    lines = pathlib.Path(path).read_text(errors="replace").splitlines()
    sends, moves, markers, run_total = [], [], [], None
    hoffs = []          # (line_idx, hoff) from WS HANDSHAKE lines
    for idx, l in enumerate(lines):
        m = re.match(r"WS HANDSHAKE ([\d.]+) t=([\d.]+)", l)
        if m:
            hoffs.append((idx, float(m.group(1)),
                          float(m.group(2))))
            continue
        m = re.match(
            r"SEND steps=(-?[\d.]+) seconds=([\d.]+).*t=([\d.]+)", l)
        if m:
            sends.append({"t": float(m.group(3)),
                          "steps": float(m.group(1)),
                          "secs": float(m.group(2))})
            continue
        m = re.match(
            r"move target=(-?[\d.]+) cur=(-?[\d.]+) delta=(-?[\d.]+)"
            r" v=([\d.]+) t=([\d.]+)", l)
        if m:
            moves.append({"t": float(m.group(5)),
                          "target": float(m.group(1)),
                          "cur": float(m.group(2))})
            continue
        m = re.match(
            r"LEG MARKER n=(\d+) z=(-?[\d.]+) et=([\d.]+) t=([\d.]+)", l)
        if m:
            markers.append({"z": float(m.group(2)),
                            "et": float(m.group(3)),
                            "t": float(m.group(4)),
                            "idx": idx})
            continue
        m = re.match(r"RUN total=([\d.]+) mode=(.*)", l)
        if m:
            run_total = (float(m.group(1)), m.group(2))
    return sends, moves, markers, run_total, hoffs


def last_run(markers, gap=30.0):
    """Markers of the final AF run (gap in epochs splits runs)."""
    if not markers:
        return []
    groups = [[markers[0]]]
    for m in markers[1:]:
        if m["t"] - groups[-1][-1]["t"] > gap:
            groups.append([m])
        else:
            groups[-1].append(m)
    return groups[-1]


def bridge(run_markers, hoff):
    """mono(t) = mono0 + (t - epoch0) * scale; scale must be ~1."""
    a, b = run_markers[0], run_markers[-1]
    ta, tb = a["t"], b["t"]
    ma, mb = a["et"] - hoff, b["et"] - hoff
    scale = (mb - ma) / (tb - ta) if tb > ta else 1.0
    return ma, ta, scale


def leg_spans(run_markers, sends):
    spans = []
    for i in range(len(run_markers) - 1):
        t0, t1 = run_markers[i]["t"], run_markers[i + 1]["t"]
        span = sum(s["secs"] for s in sends if t0 <= s["t"] <= t1)
        spans.append((LEG_NAMES[i] if i < len(LEG_NAMES) else f"leg{i}",
                      t0, t1, span))
    return spans


def build_traj(moves, sends, accel):
    """Commanded z/v piecewise model. Returns (times, zs, vs) sorted."""
    pts = []
    for mv, sd in zip(moves, sends):
        if sd["secs"] <= 0:
            continue
        v = abs(sd["steps"]) / sd["secs"]
        t0 = sd["t"]
        t1 = t0 + sd["secs"] + (2.0 * v / accel if accel else 0.0)
        pts.append((t0, mv["cur"], 0.0))
        pts.append((t0 + sd["secs"] / 2.0, mv["cur"] + sd["steps"] / 2.0, v))
        pts.append((t1, mv["target"], 0.0))
    pts.sort()
    return ([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts])


def traj_at(t, ts_, zs_, vs_):
    if len(ts_) == 0:
        return 0.0, 0.0, False
    i = np.searchsorted(ts_, t)
    if i == 0:
        return zs_[0], vs_[0], True
    if i >= len(ts_):
        return zs_[-1], 0.0, False
    z0, z1 = zs_[i - 1], zs_[i]
    v0, v1 = vs_[i - 1], vs_[i]
    f = (t - ts_[i - 1]) / max(ts_[i] - ts_[i - 1], 1e-6)
    return z0 + f * (z1 - z0), v0 + f * (v1 - v0), True


def draw_hud(frame_rgb, st):
    global FONT
    if FONT is None:
        FONT = make_font()
    img = Image.new("RGB", (W, H + HUD_H))
    img.paste(Image.fromarray(frame_rgb), (0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle((0, H, W, H + HUD_H), fill=(12, 14, 20, 235))
    d.text((6, H + 4), st["leg"], font=FONT, fill=(125, 211, 252))
    d.text((80, H + 4), "z= %.1f u %s" % (st["z"], st["src"]),
           font=FONT, fill=(240, 240, 240))
    d.text((240, H + 4), "v= %.0f u/s" % st["v"], font=FONT,
           fill=(240, 240, 240))
    d.text((6, H + 26), "t+ %.2fs" % st["tleg"], font=FONT,
           fill=(200, 200, 200))
    bw = int((W - 160) * max(0.0, min(1.0, st["frac"])))
    d.rectangle((70, H + 28, W - 90, H + 42), fill=(40, 44, 52, 255))
    if bw > 0:
        d.rectangle((70, H + 28, 70 + bw, H + 42), fill=(80, 200, 120, 255))
    d.text((W - 86, H + 26), "%.0f u" % st["span"], font=FONT,
           fill=(200, 200, 200))
    if not st["live"]:
        d.text((W - 110, H + 4), "NO TELEMETRY", font=FONT, fill=(255, 90, 90))
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--f0", type=int, default=0)
    ap.add_argument("--f1", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    base = pathlib.Path(args.dir)

    nv12s = sorted(base.glob("af_view_*.nv12"))
    if not nv12s:
        sys.exit("no af_view_*.nv12 in " + str(base))
    raw = nv12s[-1]
    tsf = raw.with_suffix(".tsv")

    leg = json.load(open(base / "smooth_af_legdump.json"))
    hoff = leg["hoff"]
    accel = 31609.0
    calp = base / "smooth_af_cal.json"
    if calp.exists():
        try:
            accel = float(json.load(open(calp)).get("accel") or accel)
        except Exception:
            pass

    sends, moves, markers, run_total, hoffs = load_events(
        base / "smooth_af_moves.log")
    run_markers = last_run(markers)
    if len(run_markers) < 2:
        sys.exit("no complete AF run in moves.log")
    knot_mono = np.array([m["et"] - hoff for m in run_markers])
    knot_epoch = np.array([m["t"] for m in run_markers])
    seg_scales = np.diff(knot_mono) / np.diff(knot_epoch)
    if np.max(np.abs(seg_scales - 1.0)) > 0.02:
        print("NOTE: per-seg bridge scales:",
              " ".join("%.3f" % s for s in seg_scales))
    spans = []
    for i in range(len(run_markers) - 1):
        span = sum(s["secs"] for s in sends
                   if knot_epoch[i] <= s["t"] <= knot_epoch[i + 1])
        spans.append((LEG_NAMES[i] if i < len(LEG_NAMES) else f"leg{i}",
                      knot_epoch[i], knot_epoch[i + 1], span))
    ts_, zs_, vs_ = build_traj(moves, sends, accel)

    tss = np.array([float(x)
                    for x in tsf.read_text().split() if x.strip()])
    f0 = args.f0
    f1 = args.f1 if args.f1 is not None else min(len(tss), 10**9)
    tss = tss[f0:f1]
    n = len(tss)
    data = np.fromfile(raw, dtype=np.uint8,
                       count=n * FRAME_BYTES, offset=f0 * FRAME_BYTES)
    data = data.reshape(n, FRAME_BYTES)

    out_dir = base / "hud_frames"
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("f*.png"):
        old.unlink()
    last = None
    for i in range(n):
        epoch = float(np.interp(tss[i], knot_mono, knot_epoch))
        z, v, live = traj_at(epoch, ts_, zs_, vs_)
        leg_name, frac, span, tleg = "idle", 0.0, 0.0, 0.0
        for nm, t0, t1, sp in spans:
            if t0 <= epoch <= t1:
                leg_name, span = nm, sp
                frac = (epoch - t0) / (t1 - t0)
                tleg = epoch - t0
                break
        st = {"leg": leg_name, "z": z, "v": v, "frac": frac,
              "span": span, "tleg": tleg, "live": live, "src": "cmd"}
        frame = draw_hud(nv12_to_rgb(data[i]), st)
        Image.fromarray(frame).save(out_dir / f"f{i:05d}.png")
        last = st
    out = args.out or str(base / "af_hud.mp4")
    probe = Image.open(out_dir / "f00000.png")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", "120", "-i", str(out_dir / "f%05d.png"),
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", out,
    ], check=True, timeout=900)
    print("saved", out, "| frame size:", probe.size)
    print("frames:", n, "| run:", run_total, "| last state:", last)


if __name__ == "__main__":
    main()
