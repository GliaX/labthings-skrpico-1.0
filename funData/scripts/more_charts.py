"""Charts 6-9: outcome forensics, dual-estimate deltas, capture
self-destruction, velocity personality. Reads smooth_af_moves.log plus
(optionally) the capture .tsv from the armed session.

Usage: python more_charts.py [dir-with-moves.log] [--tsv capture.tsv]
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

DATA = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else \
    pathlib.Path(__file__).resolve().parent.parent / "data"
LOCAL = timezone(timedelta(hours=2))
OUT = DATA.parent / "charts"
OUT.mkdir(exist_ok=True)

lines = (DATA / "smooth_af_moves.log").read_text(errors="replace").splitlines()


def dt(epoch):
    return datetime.fromtimestamp(epoch, LOCAL)


ts_re = re.compile(r"t=([\d.]+)\s*$")
ts_idx = []
for i, l in enumerate(lines):
    m = ts_re.search(l)
    if m:
        t = float(m.group(1))
        if t >= 1.0e9:
            ts_idx.append((i, t))


def time_of(i):
    j = bisect.bisect_left([x[0] for x in ts_idx], i)
    if j == 0:
        return ts_idx[0][1]
    if j >= len(ts_idx):
        return ts_idx[-1][1]
    i0, t0 = ts_idx[j - 1]
    i1, t1 = ts_idx[j]
    f = (i - i0) / (i1 - i0)
    return t0 + f * (t1 - t0)


# T_EVENT / T_6000: same era boundaries as hist_charts
T_EVENT = 1791765000
T_6000 = 1791762500
ERAS = (("old", 0.0), ("mid", T_6000), ("new", T_EVENT))
ERA_COLORS = {"old": "#9ca3af", "mid": "#f59e0b", "new": "#059669"}


def era_of(t):
    name = "old"
    for e, tv in ERAS:
        if t >= tv:
            name = e
    return name


# ================= chart 6: outcome forensics =================
OUTCOMES = [
    ("near_fit", "near sweep fit"),
    ("claw_pool", "claw pooled16"),
    ("claw_far", "claw sharp/far"),
    ("claw_sharp", "claw sharp"),
    ("tiny_fit", "tiny verified"),
    ("tiny_fail", "tiny refused"),
    ("no_anchor", "no anchor"),
    ("REFUSED", "hard refuse"),
    ("RETRY", "retry"),
]
out_events = []          # (epoch, tag)
for i, l in enumerate(lines):
    m = re.match(r"RUN total=([\d.]+) mode=(.*)", l)
    if not m:
        continue
    t = time_of(i)
    mode = m.group(2)
    for key, _label in OUTCOMES:
        if key in mode:
            out_events.append((t, key))
            break

fig, ax = plt.subplots(figsize=(11, 4.6))
keys_order = ["near_fit", "claw_pool", "claw_sharp", "claw_far",
              "tiny_fit", "tiny_fail", "no_anchor", "REFUSED"]
colors = plt.cm.tab10(range(len(keys_order)))
cmap = dict(zip(keys_order, colors))
series = {k: ([], []) for k in keys_order}
for t, k in out_events:
    series[k][0].append(dt(t))
    series[k][1].append(1)
for k in keys_order:
    xs, ys = series[k]
    if xs:
        ax.scatter(xs, [k] * len(xs), s=26, color=cmap[k], alpha=0.8,
                   label=f"{k} (n={len(xs)})")
for _e, tv in ERAS[1:]:
    ax.axvline(dt(tv), color="#666", ls="--", alpha=0.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
ax.set_title("AF outcome detectors over time (first matching tag per run)")
ax.grid(alpha=0.3, axis="x")
ax.legend(fontsize=8, loc="upper left", ncol=2)
fig.tight_layout()
fig.savefig(str(OUT / "chart_outcome_forensics.png"), dpi=140)
plt.close(fig)
print("chart 6: outcomes n=", len(out_events))

# rates per era
print("outcome rates per era:")
tot = {"old": 0, "mid": 0, "new": 0}
cnt = {("old", k): 0 for k in keys_order}
cnt.update({("mid", k): 0 for k in keys_order})
cnt.update({("new", k): 0 for k in keys_order})
run_times = []
for i, l in enumerate(lines):
    if re.match(r"RUN total=", l):
        run_times.append(time_of(i))
for t in run_times:
    tot[era_of(t)] += 1
for t, k in out_events:
    cnt[(era_of(t), k)] += 1
for e in ("old", "mid", "new"):
    top = sorted(((cnt[(e, k)] / tot[e] * 100, k)
                  for k in keys_order), reverse=True)[:3]
    print(f"  {e}: n={tot[e]:3d} | " +
          " ".join(f"{k}={p:.0f}%" for p, k in top))

# ================= chart 7: dual-estimate deltas =================
# tiny_fit from mode + SAVE anchor per run; plus fwhm tags
deltas, fwhms = [], []
tiny_now = None
fwhm_now = None
for i, l in enumerate(lines):
    m = re.match(r"RUN total=", l)
    if m:
        t = time_of(i)
        mm = re.search(r"tiny_fit(-?[\d.]+)", l)
        tiny_now = (t, float(mm.group(1))) if mm else None
        mf = re.search(r"\|fwhm([\d.]+)", l)
        if mf and tiny_now:
            fwhms.append((t, float(mf.group(1))))
        continue
    m = re.match(r"SAVE anchor=(-?[\d.]+)", l)
    if m and tiny_now:
        t = time_of(i)
        deltas.append((tiny_now[0], abs(tiny_now[1] - float(m.group(1))),
                       tiny_now[1]))
        tiny_now = None

fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
if deltas:
    a1.plot([dt(t) for t, _, _ in deltas], [d for _, d, _ in deltas],
            "o", ms=5, color="#7c3aed")
    a1.set_ylabel("|tiny_fit - SAVE anchor| (u)")
    a1.set_title("Dual focus estimates: verification gap over time")
    a1.grid(alpha=0.3)
if fwhms:
    a2.plot([dt(t) for t, _ in fwhms], [f for _, f in fwhms],
            "s", ms=5, color="#0d9488")
    a2.set_ylabel("measured fwhm (u)")
    a2.set_title("Tiny-sweep measured FWHM per run")
    a2.grid(alpha=0.3)
a2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
fig.tight_layout()
fig.savefig(str(OUT / "chart_dual_estimate.png"), dpi=140)
plt.close(fig)
if deltas:
    print("chart 7: dual estimates n=%d med gap=%.1fu max=%.1fu" % (
        len(deltas), statistics.median(d for _, d, _ in deltas),
        max(d for _, d, _ in deltas)))
else:
    print("chart 7: no dual-estimate pairs in log")
print("chart 7: fwhm points n=", len(fwhms))

# ================= chart 8: capture self-destruction =================
tsv = DATA / "af_view_capture.tsv"
if not tsv.exists():
    # fall back: original capture pulled earlier
    cand = list(DATA.parent.glob("charts/af_view*.tsv")) + \
        list(DATA.glob("af_view*.tsv"))
    tsv = cand[0] if cand else None
if tsv:
    tss = [float(x) for x in tsv.read_text().split()]
    iv = [1000.0 * (b - a) for a, b in zip(tss, tss[1:])]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    base_t = tss[0]
    ax.plot([t - base_t for t in tss[1:]], iv, lw=0.8, color="#b91c1c")
    ax.axhline(8.33, color="#059669", ls="--", lw=1,
               label="healthy 120fps (8.33ms)")
    arm_t = None
    for i, l in enumerate(lines):
        if l.startswith("SEND steps=1024"):
            arm_t = tss[0]
            break
    ax.set_xlabel("capture time (s)")
    ax.set_ylabel("camera frame interval (ms)")
    ax.set_title("The self-destruction: camera frame interval while the "
                 "NV12 sink was armed\n(throttle onset visible; sink wrote "
                 "184KB/frame synchronously in the pre-callback)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(OUT / "chart_capture_throttle.png"), dpi=140)
    plt.close(fig)
    print("chart 8: intervals n=%d med=%.1fms p90=%.1fms" % (
        len(iv), statistics.median(iv),
        sorted(iv)[int(len(iv) * 0.9)]))
else:
    print("chart 8: no capture tsv found, skipped")

# gcode dispatch latency: SEND -> SENT pairs
lat = []
last_send = None
for l in lines:
    m = re.match(r"SEND steps=.* t=([\d.]+)", l)
    if m:
        last_send = float(m.group(1))
        continue
    m = re.match(r"SENT ok t=([\d.]+)", l)
    if m and last_send:
        lat.append((last_send, float(m.group(1)) - last_send))
        last_send = None
if lat:
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.plot([dt(t) for t, _ in lat], [d * 1000 for _, d in lat],
            lw=0.7, color="#7c2d12")
    ax.set_ylabel("SEND -> SENT round-trip (ms)")
    ax.set_title("Gcode dispatch latency over time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(OUT / "chart_gcode_latency.png"), dpi=140)
    plt.close(fig)
    print("chart 8b: gcode latency n=%d med=%.0fms p95=%.0fms" % (
        len(lat), statistics.median(d for _, d in lat) * 1000,
        sorted(d for _, d in lat)[int(len(lat) * 0.95)] * 1000))

# ================= chart 9: velocity personality =================
sends = []
for i, l in enumerate(lines):
    m = re.match(r"SEND steps=(-?[\d.]+) seconds=([\d.]+).*t=([\d.]+)", l)
    if m:
        steps = abs(float(m.group(1)))
        secs = float(m.group(2))
        t = float(m.group(3))
        if t >= 1.0e9 and secs > 0:
            sends.append((t, steps / secs, steps))
fig, ax = plt.subplots(figsize=(11, 5.6))
rng = [(0, 400), (400, 1500), (1500, 4500), (4500, 12000)]
labels = ["near sweep (<400)", "labeled mid", "far/cal (1.5-4.5k)",
          "far/park (4.5k+)"]
palette = ["#6366f1", "#0ea5e9", "#f59e0b", "#dc2626"]
for (lo, hi), lab, c in zip(rng, labels, palette):
    pts = [(dt(t), v) for t, v, s in sends if lo <= v < hi]
    if pts:
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=12,
                   alpha=0.6, color=c, label=f"{lab} (n={len(pts)})")
for _e, tv in ERAS[1:]:
    ax.axvline(dt(tv), color="#666", ls="--", alpha=0.5)
ax.set_yscale("log")
ax.set_ylabel("SEND velocity (u/s, log)")
ax.set_title("Velocity personality: every move the AF ever ordered")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(str(OUT / "chart_velocity_personality.png"), dpi=140)
plt.close(fig)
print("chart 9: sends n=", len(sends))
print("saved to", OUT)
