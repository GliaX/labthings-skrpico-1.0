"""History charts from smooth_af_moves.log: accel-cal health, focus
drift, run-duration eras (time- and distance-normalized), and a
distance-vs-duration scatter.

Usage: python hist_charts.py [dir-with-moves.log]
       defaults to ../data next to this script.

Event lines lack timestamps; each is interpolated between the nearest
timestamped neighbor lines (within a run those are milliseconds apart).
Run distance = sum of |steps| of the SENDs a run consumed (sends
precede their RUN summary line; each RUN claims all sends up to its
own timestamp).
"""

import bisect
import re
import statistics
import sys
import pathlib
from datetime import datetime, timezone, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
BASE = (pathlib.Path(sys.argv[1]) if len(sys.argv) > 1
        else HERE.parent / "data")
LOCAL = timezone(timedelta(hours=2))
lines = (BASE / "smooth_af_moves.log").read_text(errors="replace").splitlines()

T_EVENT = 1791765000
T_6000 = 1791762500
COLS = {"old": "#9ca3af", "mid": "#f59e0b", "new": "#059669"}
ERA_LABEL = {"old": "old chain", "mid": "6000/et-gate", "new": "event-driven"}


def dt(epoch):
    return datetime.fromtimestamp(epoch, LOCAL)


# ---------- timestamp index from lines that carry trailing t= ----------
ts_re = re.compile(r"t=([\d.]+)\s*$")
ts_idx = []
for i, l in enumerate(lines):
    m = ts_re.search(l)
    if not m:
        continue
    t = float(m.group(1))
    if t < 1.0e9:
        continue
    ts_idx.append((i, t))


def time_of(i):
    j = bisect.bisect_left([x[0] for x in ts_idx], i)
    if j == 0:
        return ts_idx[0][1]
    if j >= len(ts_idx):
        return ts_idx[-1][1]
    (i0, t0), (i1, t1) = ts_idx[j - 1], ts_idx[j]
    if i1 == i0:
        return t0
    f = (i - i0) / (i1 - i0)
    return t0 + f * (t1 - t0)


# ---------- parse events ----------
accel_events, saves, runs, all_sends = [], [], [], []
for i, l in enumerate(lines):
    m = re.match(r"ACCELTEL (?:out-of-range )?a=([\d.]+)", l)
    if m:
        accel_events.append((time_of(i), float(m.group(1)),
                             "rej" if "out-of-range" in l else "ok"))
        continue
    m = re.match(r"CALACCEL loaded ([\d.]+)", l)
    if m:
        accel_events.append((time_of(i), float(m.group(1)), "load"))
        continue
    m = re.match(r"SAVE anchor=(-?[\d.]+)", l)
    if m:
        saves.append((time_of(i), float(m.group(1))))
        continue
    m = re.match(r"SEND steps=(-?[\d.]+) seconds=([\d.]+).*t=([\d.]+)", l)
    if m:
        all_sends.append((float(m.group(3)), abs(float(m.group(1)))))
        continue
    m = re.match(r"RUN total=([\d.]+) mode=(.*)", l)
    if m:
        runs.append({"t": time_of(i), "total": float(m.group(1)),
                     "mode": m.group(2)})

# Sends precede their RUN line: each RUN consumes all sends up to its
# own timestamp.
si = 0
for r in runs:
    d = 0.0
    while si < len(all_sends) and all_sends[si][0] <= r["t"]:
        d += all_sends[si][1]
        si += 1
    r["dist"] = d

for r in runs:
    r["era"] = ("new" if r["t"] >= T_EVENT
                else "mid" if r["t"] >= T_6000 else "old")

# ---------- chart 2: accel cal health ----------
fig, ax = plt.subplots(figsize=(11, 4.2))
ok = [(dt(t), v) for (t, v, k) in accel_events if k == "ok"]
bad = [(dt(t), v) for (t, v, k) in accel_events if k == "rej"]
lod = [(dt(t), v) for (t, v, k) in accel_events if k == "load"]
if lod:
    ax.step([d for d, _ in lod], [v for _, v in lod], where="post",
            color="#2563eb", lw=1.8, label="persisted cal (loaded)")
if ok:
    ax.scatter([d for d, _ in ok], [v for _, v in ok], marker="^",
               s=70, color="#059669", zorder=3,
               label="fresh estimate (accepted)")
if bad:
    ax.scatter([d for d, _ in bad], [v for _, v in bad], marker="x",
               s=70, color="#dc2626", zorder=3,
               label="rejected (out of range)")
ax.set_ylabel("accel estimate (u/s²)")
ax.set_title("Acceleration calibration health - estimates vs persisted cal")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(str(BASE / "hist_accel_health.png"), dpi=140)
print("accel events:", len(accel_events), "(ok", len(ok), "rej", len(bad),
      "load", len(lod), ")")

# ---------- chart 3: focus drift ----------
fig, ax = plt.subplots(figsize=(11, 4.2))
ds = [dt(t) for t, _ in saves]
zs = [z for _, z in saves]
ax.plot(ds, zs, "-o", color="#dc2626", ms=5, lw=1.2)
for d, z in zip(ds, zs):
    ax.annotate(f"{z:.0f}", (d, z), xytext=(0, 6),
                textcoords="offset points", fontsize=7,
                ha="center", color="#7f1d1d")
ax.set_ylabel("focus anchor z (stage units)")
ax.set_title(f"Focus position over time ({len(saves)} AF saves)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(str(BASE / "hist_focus_drift.png"), dpi=140)
print("saves:", len(saves),
      "span:", f"{(max(t for t, _ in saves) - min(t for t, _ in saves)) / 3600:.1f}h")

# ---------- chart 4: run duration + throughput by era ----------
fig, (axa, axb) = plt.subplots(
    2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [1.4, 1]})
axa.scatter([dt(r["t"]) for r in runs], [r["total"] for r in runs],
            c=[COLS[r["era"]] for r in runs], s=18, alpha=0.75)
for tv, lab, c in ((T_6000, "6000/et-gate era", "#f59e0b"),
                   (T_EVENT, "event-driven live", "#059669")):
    axa.axvline(dt(tv), color=c, ls="--", alpha=0.6)
    axa.annotate(lab, (dt(tv), axa.get_ylim()[1]), rotation=90,
                 fontsize=8, color=c, va="top", ha="right",
                 xytext=(-3, -4), textcoords="offset points")
axa.set_ylabel("RUN total (s)")
axa.set_title("Autofocus duration per run, colored by chain era "
              "(REFUSED runs included)")
axa.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

bins = [i * 200 for i in range(0, 40)]
for e in ("old", "mid", "new"):
    vals = [r["dist"] / r["total"] for r in runs
            if r["era"] == e and r["dist"] > 0 and r["total"] > 0]
    if vals:
        axb.hist(vals, bins=bins, alpha=0.55, color=COLS[e],
                 label=f"{ERA_LABEL[e]} (n={len(vals)}, "
                       f"med={statistics.median(vals):.0f} u/s)",
                 edgecolor="white")
axb.set_xlabel("effective sweep throughput (stage-u per second of RUN total)")
axb.set_ylabel("runs")
axb.set_title("Throughput by era: total sweep distance / run duration")
axb.legend(fontsize=8)
fig.tight_layout()
fig.savefig(str(BASE / "hist_run_eras.png"), dpi=140)
for e in ("old", "mid", "new"):
    sel = [r for r in runs if r["era"] == e and r["dist"] > 0]
    if sel:
        print(f"{e}: n={len(sel)} eff_med={statistics.median([r['dist'] / r['total'] for r in sel]):.0f}u/s "
              f"dist_med={statistics.median([r['dist'] for r in sel]):.0f}u "
              f"time_med={statistics.median([r['total'] for r in sel]):.2f}s")

# ---------- chart 5: distance vs duration per run, by era ----------
fig, ax = plt.subplots(figsize=(11, 5.8))
for e in ("old", "mid", "new"):
    pts = [(r["dist"], r["total"]) for r in runs
           if r["era"] == e and r["dist"] > 0]
    if pts:
        ax.scatter([x for x, _ in pts], [y for _, y in pts],
                   c=COLS[e], s=24, alpha=0.75, zorder=3,
                   label=f"{ERA_LABEL[e]} (n={len(pts)}, med "
                         f"{statistics.median([x / y for x, y in pts]):.0f} u/s)")
dmax = max(r["dist"] for r in runs) * 1.06
tmax = max(r["total"] for r in runs) * 1.15
grid = np.linspace(1, dmax, 200)
for rate in (250, 500, 1000, 2000, 4000):
    ax.plot(grid, grid / rate, color="#555555", lw=0.8, ls=":",
            alpha=0.55, zorder=1)
    yl = dmax / rate
    if yl < tmax:
        ax.annotate(f"{rate} u/s", (dmax * 0.985, yl * 0.93), fontsize=7.5,
                    color="#444444", ha="right", va="top", rotation=-24)
ax.set_xlim(0, dmax)
ax.set_ylim(0, tmax)
ax.set_xlabel("total sweep distance per run (stage-u)")
ax.set_ylabel("RUN total (s)")
ax.set_title("Run duration vs distance - iso-throughput diagonals "
             "(down-right = more efficient; REFUSED runs included)")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(str(BASE / "hist_distance_eras.png"), dpi=140)
print("saved 4 charts to", BASE)
