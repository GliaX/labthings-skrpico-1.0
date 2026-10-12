"""AF motion story: one chart from the microscope's own telemetry.

P1: sharpness vs z (the mountain the AF climbs) + leg completion markers.
P2: leg wall-times from marker event spacing.
P3: motion-report samples of the last labeled leg (z + v trapezoid).
"""

import json
import re
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
BASE = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
lines = (BASE / "smooth_af_moves.log").read_text().splitlines()

run_idx = max(i for i, l in enumerate(lines) if l.startswith("RUN "))
start_idx = max(i for i, l in enumerate(lines[:run_idx])
                if l.startswith("FARREGIME_V"))
moves, markers = [], []
for l in lines[start_idx:run_idx + 1]:
    m = re.match(
        r"move target=(-?[\d.]+) cur=(-?[\d.]+) delta=(-?[\d.]+) v=(\d+)", l)
    if m:
        moves.append({"target": float(m.group(1)), "cur": float(m.group(2)),
                      "v": float(m.group(4)),
                      "t": float(l.rsplit("t=", 1)[1])})
        continue
    m = re.match(r"LEG MARKER n=(\d+) z=(-?[\d.]+) et=([\d.]+) t=([\d.]+)", l)
    if m:
        markers.append({"z": float(m.group(2)), "t": float(m.group(4))})

pts = []
for name in ("smooth_af_seekpts_far.json", "smooth_af_seekpts_tiny.json"):
    d = json.load(open(BASE / name))
    pts += list(zip(d["z"], d["s"]))
pts.sort()

leg = json.load(open(BASE / "smooth_af_legdump.json"))
hoff = leg["hoff"]
samp = sorted((s[4] - hoff, s[1], s[2])
              for s in leg["samples"] if len(s) > 4 and s[4])
leg_t0 = samp[0][0]

fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(12, 9),
    gridspec_kw={"height_ratios": [3, 1.2, 1.4]},
)
fig.suptitle("Autofocus motion story - from the microscope's own telemetry",
             fontsize=13, fontweight="bold")

# --- P1: the mountain ---
zs = [p[0] for p in pts]
ss = [p[1] for p in pts]
ax1.plot(zs, ss, "-", color="#dc2626", lw=1.6)
pk = max(pts, key=lambda p: p[1])
ax1.annotate(f"peak z={pk[0]:.0f} (parked here)", (pk[0], pk[1]),
             xytext=(10, 6), textcoords="offset points",
             color="#dc2626", fontsize=9, fontweight="bold")
ax1.axvline(pk[0], color="#dc2626", ls=":", alpha=0.4)
labels = ["far1 end", "far2 end", "claw->tiny", "tiny end", "park"]
for i, mk in enumerate(markers):
    lab = labels[i] if i < len(labels) else f"m{i}"
    ax1.axvline(mk["z"], color="#2563eb", alpha=0.35, lw=1)
    ax1.annotate(lab, (mk["z"], ax1.get_ylim()[1]), rotation=90,
                 va="top", ha="right", fontsize=8, color="#2563eb",
                 xytext=(-2, -4), textcoords="offset points")
ax1.set_xlabel("z (stage units)")
ax1.set_ylabel("sharpness (frame metric)")
ax1.set_title("What the camera saw while the stage swept", fontsize=10)

# --- P2: leg wall-times from marker spacing ---
tm = [m["t"] for m in markers]
durs = [b - a for a, b in zip(tm, tm[1:])]
names = [f"{a}->{b}" for a, b in
         (("far1", "far2"), ("far2", "appr"), ("appr", "tiny"),
          ("tiny", "park"))][:len(durs)]
bars = ax2.bar(names, durs, color="#059669", width=0.55)
for b, d in zip(bars, durs):
    ax2.annotate(f"{d:.2f}s", (b.get_x() + b.get_width() / 2, d),
                 ha="center", va="bottom", fontsize=9)
ax2.set_ylabel("seconds between completion markers")
ax2.set_title("Leg wall-times (event-to-event, no clocks involved)",
              fontsize=10)

# --- P3: telemetry of the final leg ---
ax3.plot([s[0] - leg_t0 for s in samp], [s[1] for s in samp], "-o",
         color="#059669", ms=4, lw=1.4, label="reported z (10ms tick)")
ax3b = ax3.twinx()
ax3b.plot([s[0] - leg_t0 for s in samp], [s[2] for s in samp], "-",
          color="#7c3aed", lw=1.2, alpha=0.8, label="reported v")
ax3b.set_ylabel("v (u/s)", color="#7c3aed")
ax3b.tick_params(axis="y", labelcolor="#7c3aed")
ax3.set_ylabel("z (u)")
ax3.set_xlabel("seconds (leg-local)")
ax3.set_title("Motion-report samples of the last labeled leg "
              "(trapezoid: accel, cruise, decel)", fontsize=10)
ax3.legend(loc="upper left", fontsize=8)
ax3b.legend(loc="lower right", fontsize=8)

fig.tight_layout(rect=(0, 0, 1, 0.95))
out = str(BASE / "af_motion_story.png")
fig.savefig(out, dpi=140)
print("saved", out)
print(f"moves={len(moves)} markers={len(markers)} pts={len(pts)} samp={len(samp)}")
