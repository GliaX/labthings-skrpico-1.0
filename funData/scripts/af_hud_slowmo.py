"""Extend the slowmo HUD with the sharpness/metric panel (right side).

For each window frame: current leg (spans from marker anchors), and
within the leg, per-frame sharpness/pooled from the captured run.

Data reality for the captured run:
- tiny leg: smooth_af_seekpts_tiny.json has 25 samples (z 8..116, s,
  c, p) covering the 120u sweep. Sample index = position along leg.
- far legs: seekpts empty (fit failed before dump) -> no per-frame
  metrics; show s/c/p traces as 'no data' band but keep commanded z/v.
- pooled16 for this session: flat ~371-390 (no contrast spike), so the
  pool trace shows the flatness that made claw_far fail.

Frame->leg mapping reuses the content anchor: burst 28..66 spans
marker epochs linearly; leg boundaries are the intermediate markers.
"""

import json
import pathlib
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import af_hud_video as hud  # noqa: E402

W = H = 350
PANEL_W = 210
PAD = 12

INK = (235, 240, 245)
DIM = (120, 128, 138)
CYAN = (97, 218, 251)
GREEN = (80, 220, 120)
RED = (240, 98, 90)
AMBER = (255, 193, 87)


def energy_burst(raw_path):
    fb = W * H * 3 // 2
    data = np.fromfile(raw_path, dtype=np.uint8)
    n = len(data) // fb
    Y = data[: n * fb].reshape(n, fb)[:, : W * H].reshape(n, H, W)
    e = np.abs(np.diff(Y[:, ::4, :].astype(np.int16), axis=0)).mean(
        axis=(1, 2))
    above = np.where(e > 0.22 * float(e.max()))[0]
    return int(above[0]), int(above[-1]), n


def load_metrics(base):
    """Per-leg metric curves from seekpts jsons: list of (z, s, c, p)."""
    out = {}
    for tag in ("far", "far2", "tiny"):
        p = base / f"smooth_af_seekpts_{tag}.json"
        try:
            dd = json.load(open(p))
        except Exception:
            dd = {"z": [], "s": [], "c": [], "p": []}
        out[tag] = list(zip(dd["z"], dd["s"], dd["c"], dd["p"]))
    return out


def draw_panel(img, st, leg_tag, leg_pts, leg_frac, smin, smax, pmin, pmax):
    """Right-side panel: current sharpness + pooled value and the curve
    of the current leg with a position cursor."""
    from PIL import ImageDraw
    d = ImageDraw.Draw(img, "RGBA")
    x0 = W + 6
    d.text((x0, 4), "leg " + leg_tag, font=hud.make_font(), fill=CYAN)
    if not leg_pts:
        d.text((x0, 22), "no per-frame metrics", font=hud.make_font(),
               fill=DIM)
        d.text((x0, 40), "(leg predates capture)", font=hud.make_font(),
               fill=DIM)
        return
    z, s, c, p = leg_pts[-1]
    # nearest sample to current commanded z
    best_i = min(range(len(leg_pts)), key=lambda i: abs(leg_pts[i][0] - st["z"]))
    zz, ss, cc, pp = leg_pts[best_i]
    d.text((x0, 22), f"s={ss:6.1f}", font=hud.make_font(), fill=INK)
    d.text((x0 + 100, 22), f"c={cc:5.2f}", font=hud.make_font(),
           fill=GREEN if cc > 1.15 else (AMBER if cc > 1.05 else DIM))
    d.text((x0, 40), f"pool={pp:6.1f}", font=hud.make_font(), fill=INK)
    if pmax - pmin < 1e-6:
        d.text((x0, 58), "pool flat", font=hud.make_font(), fill=RED)
    # curve: s vs sample index
    cw, ch = PANEL_W - 18, 54
    cx, cy = x0, 74
    d.rectangle((cx, cy, cx + cw, cy + ch), outline=(60, 66, 76))
    smin_l = min(q[1] for q in leg_pts)
    smax_l = max(q[1] for q in leg_pts)
    rng = max(smax_l - smin_l, 1e-6)
    pts = [(cx + i * cw / (len(leg_pts) - 1),
            cy + ch - (q[1] - smin_l) / rng * (ch - 6) - 3)
           for i, q in enumerate(leg_pts)]
    d.line(pts, fill=CYAN, width=1)
    ci = cx + best_i * cw / (len(leg_pts) - 1)
    d.line([(ci, cy), (ci, cy + ch)], fill=AMBER, width=1)
    # curve: pooled vs index
    cy2 = cy + ch + 8
    d.rectangle((cx, cy2, cx + cw, cy2 + ch), outline=(60, 66, 76))
    pmin_l = min(q[3] for q in leg_pts)
    pmax_l = max(q[3] for q in leg_pts)
    prng = max(pmax_l - pmin_l, 1e-6)
    pts = [(cx + i * cw / (len(leg_pts) - 1),
            cy2 + ch - (q[3] - pmin_l) / prng * (ch - 6) - 3)
           for i, q in enumerate(leg_pts)]
    d.line(pts, fill=GREEN, width=1)
    d.line([(ci, cy2), (ci, cy2 + ch)], fill=AMBER, width=1)
    d.text((cx, cy2 + ch + 4), "sharpness", font=hud.make_font(), fill=DIM)
    d.text((cx + 100, cy2 + ch + 4), "pooled16", font=hud.make_font(),
           fill=DIM)


def main():
    base = pathlib.Path(sys.argv[1])
    slow = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0

    raw = next(base.glob("af_view_*.nv12"))
    f_m0, f_m1, n = energy_burst(raw)

    sends, moves, markers, run_total, _ = hud.load_events(
        base / "smooth_af_moves.log")
    run_markers = hud.last_run(markers)
    e0, e1 = run_markers[0]["t"], run_markers[-1]["t"]
    accel = 45000.0
    cal = base / "smooth_af_cal.json"
    if cal.exists():
        try:
            accel = float(json.load(open(cal)).get("accel") or accel)
        except Exception:
            pass
    ts_, zs_, vs_ = hud.build_traj(moves, sends, accel)
    spans = hud.leg_spans(run_markers, sends)
    leg_tags = {"far1": "far", "far2": "far2", "claw->tiny": "tiny", "tiny": "tiny"}
    metrics = load_metrics(base)
    allp = [q[3] for tag in metrics for q in metrics[tag]]

    f0, f1 = max(0, f_m0 - PAD), min(n - 1, f_m1 + PAD)
    fb = W * H * 3 // 2
    data = np.fromfile(raw, dtype=np.uint8)

    def epoch_of(f):
        if f <= f_m0:
            return e0 + (f - f_m0) * (0.5 / max(f_m0, 1))
        if f >= f_m1:
            return e1 + (f - f_m1) * (0.5 / max(n - 1 - f_m1, 1))
        return e0 + (f - f_m0) * (e1 - e0) / (f_m1 - f_m0)

    out_w, out_h = W + PANEL_W + 12, H + hud.HUD_H
    frames_dir = base / "hud_frames_slowmo"
    frames_dir.mkdir(exist_ok=True)
    for old in frames_dir.glob("f*.png"):
        old.unlink()

    for f in range(f0, f1 + 1):
        ep = epoch_of(f)
        z, v, live = hud.traj_at(ep, ts_, zs_, vs_)
        leg_name, frac, span, tleg = "idle", 0.0, 0.0, 0.0
        for nm, t0, t1, sp in spans:
            if t0 <= ep <= t1:
                leg_name, span = nm, sp
                frac = (ep - t0) / (t1 - t0)
                tleg = ep - t0
                break
        st = {"leg": leg_name, "z": z, "v": v, "frac": frac,
              "span": span, "tleg": tleg, "live": live, "src": "cmd"}
        buf = data[f * fb:(f + 1) * fb]
        frame = hud.nv12_to_rgb(buf)
        band_img = Image.fromarray(hud.draw_hud(frame, st))
        img = Image.new("RGB", (out_w, out_h), (10, 12, 16))
        img.paste(band_img, (0, 0))
        tag = leg_tags.get(leg_name)
        pts = metrics.get(tag, []) if tag else []
        draw_panel(img, st, leg_name, pts, frac,
                   0, 0, min(allp) if allp else 0, max(allp) if allp else 0)
        img.save(frames_dir / f"f{f:05d}.png")

    lst = base / "slowmo_concat.txt"
    with open(lst, "w") as f:
        files = sorted(frames_dir.glob("f*.png"))
        for ff in files:
            f.write("file '%s'\n" % ff.as_posix())
            f.write("duration %.6f\n" % max(slow / 21.0, 1.0 / 240.0))
        f.write("file '%s'\n" % files[-1].as_posix())
    out = base / "af_hud_slowmo.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(lst),
        "-fps_mode", "vfr", "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p", str(out),
    ], check=True, timeout=300)
    lst.unlink()
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,duration,width,height",
         "-of", "default=noprint_wrappers=1", str(out)],
        capture_output=True, text=True)
    print("saved", out)
    print(r.stdout.strip())


if __name__ == "__main__":
    main()
