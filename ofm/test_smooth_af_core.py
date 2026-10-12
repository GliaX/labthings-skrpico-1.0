import math
import random
import sys

sys.path.insert(
    0,
    r"C:\glia\EU_dotacje\mikroskop\OFM-server\openflexure-microscope-server"
    r"\src\openflexure_microscope_server\things",
)
from smooth_af_core import CurvePoint, cluster_curve, fit_log_gaussian

random.seed(7)


def gauss_curve(peak, fwhm, amp, base, lo, hi, n, noise):
    pts = []
    for i in range(n):
        z = lo + (hi - lo) * i / (n - 1)
        s = base + amp * math.exp(-4 * math.log(2) * (z - peak) ** 2 / fwhm**2)
        s *= 1 + random.uniform(-noise, noise)
        pts.append(CurvePoint(z=round(z, 4), sharpness=s))
    return pts


r = fit_log_gaussian(gauss_curve(26.2, 3.0, 10, 5, 20, 32, 25, 0.05))
print("clean gaussian:", r)
assert r.ok and abs(r.peak - 26.2) < 0.15 and 2.0 < r.fwhm < 4.5
assert r.snr > 2 and r.r2 > 0.9

r = fit_log_gaussian(gauss_curve(26.2, 3.0, 10, 5, 20, 32, 25, 0.25))
print("noisy gaussian:", r)
assert r.ok and abs(r.peak - 26.2) < 0.6

r = fit_log_gaussian([CurvePoint(z=z, sharpness=8.0) for z in range(20, 30)])
print("flat curve:", r)
assert not r.ok

r = fit_log_gaussian([CurvePoint(z=z, sharpness=8.0 + z * 0.1) for z in range(20, 30)])
print("gentle ramp:", r)
assert not r.ok

r = fit_log_gaussian([CurvePoint(z=z, sharpness=5.0 + z * 0.5) for z in range(20, 30)])
print("steep ramp:", r)
assert not r.ok

r = fit_log_gaussian([CurvePoint(z=20, sharpness=5), CurvePoint(z=21, sharpness=6)])
print("too few points:", r)
assert not r.ok

raw = [CurvePoint(z=1.0, sharpness=10), CurvePoint(z=1.0, sharpness=20),
       CurvePoint(z=1.0, sharpness=30), CurvePoint(z=2.0, sharpness=5)]
cl = cluster_curve(raw)
print("clustered:", [(p.z, round(p.sharpness, 2)) for p in cl])
assert len(cl) == 2
assert cl[0].sharpness == 25.0

print("\nALL SMOOTH-AF CORE TESTS PASS")
