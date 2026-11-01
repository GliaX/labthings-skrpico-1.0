import math
import random

from openflexure_microscope_server.things.smooth_af_core import (
    CurvePoint,
    fit_log_gaussian,
    median_filter_curve,
)


def gauss_curve(peak, fwhm, amp, base, lo, hi, n, noise):
    pts = []
    for i in range(n):
        z = lo + (hi - lo) * i / (n - 1)
        s = base + amp * math.exp(-4 * math.log(2) * (z - peak) ** 2 / fwhm**2)
        s *= 1 + random.uniform(-noise, noise)
        pts.append(CurvePoint(z=round(z, 4), sharpness=s))
    return pts


def test_clean_gaussian_recovers_peak():
    random.seed(7)
    r = fit_log_gaussian(gauss_curve(26.2, 3.0, 10, 5, 20, 32, 25, 0.05))
    assert r.ok
    assert abs(r.peak - 26.2) < 0.15
    assert 2.0 < r.fwhm < 4.5
    assert r.snr > 2
    assert r.r2 > 0.9


def test_noisy_gaussian_recovers_peak():
    random.seed(7)
    r = fit_log_gaussian(gauss_curve(26.2, 3.0, 10, 5, 20, 32, 25, 0.25))
    assert r.ok
    assert abs(r.peak - 26.2) < 0.6


def test_flat_curve_rejected():
    r = fit_log_gaussian([CurvePoint(z=z, sharpness=8.0) for z in range(20, 30)])
    assert not r.ok


def test_gentle_ramp_rejected():
    r = fit_log_gaussian(
        [CurvePoint(z=z, sharpness=8.0 + z * 0.1) for z in range(20, 30)]
    )
    assert not r.ok


def test_steep_ramp_rejected():
    r = fit_log_gaussian(
        [CurvePoint(z=z, sharpness=5.0 + z * 0.5) for z in range(20, 30)]
    )
    assert not r.ok


def test_too_few_points_rejected():
    r = fit_log_gaussian([CurvePoint(z=20, sharpness=5), CurvePoint(z=21, sharpness=6)])
    assert not r.ok


def test_median_filter_kills_spike():
    pts = [CurvePoint(z=z, sharpness=10.0) for z in range(10)]
    pts[5].sharpness = 100.0
    out = median_filter_curve(pts, window=3)
    assert out[5].sharpness == 10.0
