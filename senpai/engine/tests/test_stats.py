"""Tests for the strided background statistics helper.

``robust_background_stats`` replaced full-frame ``sigma_clipped_stats`` in
the detection stack; these pin the contract that the strided estimate is
interchangeable with the full computation for detection thresholding.
"""

from __future__ import annotations

import numpy as np
from astropy.stats import sigma_clipped_stats

from senpai.engine.utils.stats import robust_background_stats


def _field(shape=(2400, 2400), background=120.0, noise=8.0, seed=11):
    rng = np.random.default_rng(seed)
    image = np.full(shape, background) + rng.normal(0.0, noise, shape)
    # Bright contaminants the sigma clip must reject (stars / hot pixels).
    ys = rng.integers(0, shape[0], 400)
    xs = rng.integers(0, shape[1], 400)
    image[ys, xs] += rng.uniform(500, 50000, 400)
    return image


def test_strided_stats_match_full_computation():
    image = _field()
    full_mean, full_median, full_std = sigma_clipped_stats(
        image, sigma=3.0, maxiters=5
    )
    mean, median, std = robust_background_stats(image)
    # The quantity detection consumes is median + n*std: both terms must be
    # interchangeable with the full-frame computation at the noise scale.
    assert abs(median - full_median) < 0.05 * full_std
    assert abs(std - full_std) / full_std < 0.01
    assert abs(mean - full_mean) < 0.05 * full_std


def test_small_images_are_not_subsampled():
    # At or below target_npix the input must be used in full — identical
    # results to sigma_clipped_stats, so unit-scale callers see no change.
    image = _field(shape=(256, 256))
    expected = sigma_clipped_stats(image, sigma=3.0, maxiters=5)
    result = robust_background_stats(image)
    assert result == expected


def test_gradient_background_is_sampled_fairly():
    # A vignetting-like gradient: the stride covers the whole frame, so the
    # estimate must track the global median, not a corner's.
    rng = np.random.default_rng(3)
    yy, _xx = np.mgrid[0:2200, 0:2200]
    image = 100.0 + 30.0 * (yy / 2200.0) + rng.normal(0.0, 5.0, (2200, 2200))
    _, full_median, full_std = sigma_clipped_stats(image, sigma=3.0, maxiters=5)
    _, median, std = robust_background_stats(image)
    assert abs(median - full_median) < 0.05 * full_std
    assert abs(std - full_std) / full_std < 0.01
