import math
import statistics
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class CurvePoint:
    z: float
    sharpness: float
    brightness: Optional[float] = None


@dataclass
class FitOutcome:
    ok: bool
    reason: str = ""
    peak: Optional[float] = None
    fwhm: Optional[float] = None
    r2: Optional[float] = None
    snr: Optional[float] = None


def fit_log_gaussian(points: List[CurvePoint], window: float = 0.7) -> FitOutcome:
    """Fit a Gaussian in log-sharpness space; reject curves without a real peak."""
    pts = sorted(
        (p.z, math.log(max(p.sharpness, 1e-9)))
        for p in points
        if p.sharpness and p.sharpness > 0
    )
    if len(pts) < 5:
        return FitOutcome(ok=False, reason=f"only {len(pts)} valid points")
    logs = [v for _, v in pts]
    if max(logs) - min(logs) < 0.15:
        return FitOutcome(ok=False, reason="flat curve, no focus feature")
    i = max(range(len(pts)), key=lambda k: pts[k][1])
    lo = i
    while lo > 0 and pts[lo][1] > pts[i][1] - window:
        lo -= 1
    hi = i
    while hi < len(pts) - 1 and pts[hi][1] > pts[i][1] - window:
        hi += 1
    seg = pts[lo : hi + 1]
    n = len(seg)
    if n < 5:
        return FitOutcome(ok=False, reason="fit window too narrow")
    u = np.array([z for z, _ in seg])
    v = np.array([val for _, val in seg])
    ub = float(u.mean())
    u = u - ub
    su2 = float(np.sum(u * u))
    su4 = float(np.sum(u * u * u * u))
    sv = float(np.sum(v))
    su2v = float(np.sum(u * u * v))
    suv = float(np.sum(u * v))
    det = n * su4 - su2 * su2
    if abs(det) < 1e-12:
        return FitOutcome(ok=False, reason="singular fit")
    a = (n * su2v - su2 * sv) / det
    b = suv / su2 if abs(su2) > 1e-12 else 0.0
    if a >= 0:
        return FitOutcome(ok=False, reason="no concave peak")
    peak = ub - b / (2 * a)
    z_lo, z_hi = pts[0][0], pts[-1][0]
    margin = 0.1 * (z_hi - z_lo)
    if not (z_lo - margin <= peak <= z_hi + margin):
        return FitOutcome(ok=False, reason="peak outside sampled window")
    c = (sv - a * su2) / n
    fwhm = 2 * math.sqrt(math.log(2) / (-a))
    ss_res = float(np.sum((v - (a * u * u + b * u + c)) ** 2))
    ss_tot = float(np.sum((v - v.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    baseline = statistics.median(sorted(logs)[: max(2, len(logs) * 3 // 10)])
    snr = math.exp(pts[i][1] - baseline)
    return FitOutcome(ok=True, peak=float(peak), fwhm=fwhm, r2=r2, snr=snr)


def median_filter_curve(points: List[CurvePoint], window: int = 3) -> List[CurvePoint]:
    """Median-filter sharpness over z-ordered samples (kills motion-blur spikes)."""
    pts = sorted(points, key=lambda p: p.z)
    if len(pts) < window:
        return pts
    half = window // 2
    out = []
    for i, p in enumerate(pts):
        vals = [q.sharpness for q in pts[max(0, i - half): i + half + 1]]
        out.append(CurvePoint(z=p.z, sharpness=statistics.median(vals),
                              brightness=p.brightness))
    return out


def cluster_curve(points: List[CurvePoint]) -> List[CurvePoint]:
    """Median-collapse samples at the same z (drops first sample of each cluster)."""
    clusters: dict = {}
    for p in points:
        clusters.setdefault(round(p.z, 3), []).append(p.sharpness)
    out = []
    for z, vals in sorted(clusters.items()):
        if len(vals) >= 2:
            out.append(CurvePoint(z=z, sharpness=statistics.median(vals[1:])))
        else:
            out.append(CurvePoint(z=z, sharpness=vals[0]))
    return out
