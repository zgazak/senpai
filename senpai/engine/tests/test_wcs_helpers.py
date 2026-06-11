"""Equivalence tests for the optimized WCS-helper hot paths.

``find_local_maxima`` gained a threshold-ladder fast path and
``match_stars_to_detections`` a vectorized cost matrix; both must be
interchangeable with their original full-computation forms.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from senpai.engine.models.starfield import StarInImage
from senpai.engine.utils.wcs_helpers import find_local_maxima, match_stars_to_detections


def _smooth_peaky_image(seed=3, size=1200, n_peaks=120):
    rng = np.random.default_rng(seed)
    img = rng.normal(0.0, 1.0, (size, size))
    ys = rng.integers(40, size - 40, n_peaks)
    xs = rng.integers(40, size - 40, n_peaks)
    img[ys, xs] += rng.uniform(50, 5000, n_peaks)
    # Kernel-smoothed, like the convolved images this runs on
    return gaussian_filter(img, sigma=4.0)


def test_find_local_maxima_fast_path_matches_full_computation():
    img = _smooth_peaky_image()
    k = 50
    fast = find_local_maxima(img, min_distance=30, max_detections=k)
    # max_detections=None takes the original maximum_filter path over all
    # maxima; its brightest k must equal the fast path's result.
    full = find_local_maxima(img, min_distance=30, max_detections=None)[:k]
    assert np.array_equal(fast, full)


def test_find_local_maxima_respects_threshold():
    img = _smooth_peaky_image(seed=4)
    thresh = float(np.percentile(img, 99.999))
    got = find_local_maxima(img, min_distance=30, threshold=thresh, max_detections=20)
    full = find_local_maxima(img, min_distance=30, threshold=thresh, max_detections=None)[:20]
    assert np.array_equal(got, full)
    assert all(img[y, x] > thresh for y, x in got)


def _match_reference(stars, detected_points, max_distance=20):
    """The original (pre-vectorization) implementation, kept as the oracle."""
    from scipy.optimize import linear_sum_assignment

    cost = np.zeros((len(stars), len(detected_points)))
    for i, star in enumerate(stars):
        if star is None:
            cost[i, :] = np.inf
            continue
        for j, (y, x) in enumerate(detected_points):
            cost[i, j] = np.sqrt((star.x - x) ** 2 + (star.y - y) ** 2)
    row, col = linear_sum_assignment(cost)
    pairs = [(i, j) for i, j in zip(row, col, strict=False) if cost[i, j] <= max_distance]
    un_s = [i for i in range(len(stars)) if i not in {p[0] for p in pairs}]
    un_d = [j for j in range(len(detected_points)) if j not in {p[1] for p in pairs}]
    return pairs, un_s, un_d


def test_match_stars_to_detections_matches_reference():
    rng = np.random.default_rng(9)
    stars = [
        StarInImage(x=float(x), y=float(y), counts=1.0)
        for x, y in rng.uniform(0, 2000, (300, 2))
    ]
    stars[7] = None  # None stars get infinite-cost rows
    # Detections: noisy copies of a subset of stars plus a few spurious points
    dets = [(s.y + rng.normal(0, 2), s.x + rng.normal(0, 2)) for s in stars[:40] if s]
    dets += [tuple(p) for p in rng.uniform(0, 2000, (10, 2))]

    got = match_stars_to_detections(stars, dets, max_distance=20)
    ref = _match_reference(stars, dets, max_distance=20)
    assert sorted(got[0]) == sorted(ref[0])
    assert sorted(got[1]) == sorted(ref[1])
    assert sorted(got[2]) == sorted(ref[2])
