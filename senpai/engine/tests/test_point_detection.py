"""Behavioural tests for point-source detection.

Covers:
  - ``senpai.engine.detection.point.sidereal`` (star detection + FWHM helpers)
  - ``senpai.engine.detection.point.fwhm`` (catalog-star FWHM measurement)
  - ``senpai.engine.detection.point.satellite`` (rate-frame point detection,
    SNR/brightness filtering)

All tests run on synthetic Gaussian star fields with seeded noise so that
detection counts, positions, and recovered FWHM are deterministic.  No
network, no Astrometry.net, no GUI (matplotlib stays on the Agg backend).
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from datetime import UTC, datetime

import numpy as np
import pytest
from astropy.io import fits

from senpai.core.config import get_config, initialize_config
from senpai.core.constants import CONFIG_DIR
from senpai.engine.detection.point.fwhm import measure_fwhm_from_catalog_stars
from senpai.engine.detection.point.satellite import (
    extract_point_sources as extract_satellite_points,
)
from senpai.engine.detection.point.sidereal import (
    detect_sources_classic,
    estimate_fwhm,
    extract_point_sources,
)
from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.models.metadata import DetectionMetadata, ImageMetadata
from senpai.engine.models.senpai import RateTrackFrame
from senpai.engine.models.starfield import StarField, StarInSpace

SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ~2.3548


@pytest.fixture(scope="module", autouse=True)
def _config():
    """Initialise the process-wide config singleton from a shipped YAML."""
    initialize_config(CONFIG_DIR / "burr.yaml")
    cfg = get_config()
    cfg.plotting.debug = False
    return cfg


# ---------------------------------------------------------------------------
# Synthetic field helpers
# ---------------------------------------------------------------------------
def _add_gaussian(image: np.ndarray, x: float, y: float, flux: float, sigma: float) -> None:
    """Add a normalised 2D Gaussian (total ~flux) to ``image`` in place."""
    h, w = image.shape
    half = int(np.ceil(5 * sigma))
    x0, y0 = round(x), round(y)
    xlo, xhi = max(0, x0 - half), min(w, x0 + half + 1)
    ylo, yhi = max(0, y0 - half), min(h, y0 + half + 1)
    yy, xx = np.mgrid[ylo:yhi, xlo:xhi]
    g = np.exp(-(((xx - x) ** 2) + ((yy - y) ** 2)) / (2.0 * sigma**2))
    norm = 2.0 * np.pi * sigma**2
    image[ylo:yhi, xlo:xhi] += (flux / norm) * g


def _star_field(
    positions: list[tuple[float, float]],
    fluxes: list[float],
    sigma: float = 2.0,
    shape: tuple[int, int] = (256, 256),
    background: float = 100.0,
    noise: float = 5.0,
    seed: int = 1234,
) -> np.ndarray:
    """Build a synthetic star field with Poisson-like Gaussian read noise."""
    rng = np.random.default_rng(seed)
    image = np.full(shape, background, dtype=float)
    image += rng.normal(0.0, noise, shape)
    for (x, y), flux in zip(positions, fluxes, strict=True):
        _add_gaussian(image, x, y, flux, sigma)
    return image


def _processed_image(data: np.ndarray, image_id: str = "synthetic") -> ProcessedFitsImage:
    header = fits.Header()
    header["NAXIS1"] = data.shape[1]
    header["NAXIS2"] = data.shape[0]
    header["EXPTIME"] = 1.0
    metadata = ImageMetadata(
        image_id=image_id,
        width=data.shape[1],
        height=data.shape[0],
        exposure_time=1.0,
    )
    return ProcessedFitsImage(
        data=data,
        header=header,
        data_type=data.dtype,
        metadata=metadata,
    )


def _match_count(detected: list[tuple[float, float]], truth: list[tuple[float, float]], tol: float) -> int:
    """Number of truth positions matched by a detection within ``tol`` pixels."""
    matched = 0
    for tx, ty in truth:
        for dx, dy in detected:
            if (dx - tx) ** 2 + (dy - ty) ** 2 <= tol**2:
                matched += 1
                break
    return matched


# ---------------------------------------------------------------------------
# detect_sources_classic
# ---------------------------------------------------------------------------
def test_detect_sources_classic_finds_known_stars():
    truth = [(50.0, 60.0), (120.0, 90.0), (200.0, 180.0), (80.0, 210.0)]
    data = _star_field(truth, [40000.0] * 4, sigma=2.0)
    sources = detect_sources_classic(data, max_sources=20, fwhm=4.0, threshold_sigma=5.0)
    assert len(sources) >= len(truth)
    detected = [(float(s["xcentroid"]), float(s["ycentroid"])) for s in sources]
    assert _match_count(detected, truth, tol=1.5) == len(truth)


def test_detect_sources_classic_positions_within_one_pixel():
    truth = [(64.0, 64.0), (160.0, 100.0)]
    data = _star_field(truth, [60000.0, 50000.0], sigma=2.0, noise=3.0)
    sources = detect_sources_classic(data, max_sources=10, fwhm=4.0, threshold_sigma=5.0)
    detected = [(float(s["xcentroid"]), float(s["ycentroid"])) for s in sources]
    for tx, ty in truth:
        nearest = min(detected, key=lambda d: (d[0] - tx) ** 2 + (d[1] - ty) ** 2)
        assert abs(nearest[0] - tx) <= 1.0
        assert abs(nearest[1] - ty) <= 1.0


def test_detect_sources_classic_respects_max_sources():
    truth = [(x, y) for x in (40, 90, 140, 190) for y in (40, 90, 140, 190)]
    data = _star_field(truth, [30000.0] * len(truth), sigma=2.0)
    sources = detect_sources_classic(data, max_sources=5, fwhm=4.0, threshold_sigma=5.0)
    assert len(sources) == 5


def test_detect_sources_classic_empty_on_blank_field():
    rng = np.random.default_rng(7)
    data = 100.0 + rng.normal(0.0, 5.0, (128, 128))
    sources = detect_sources_classic(data, max_sources=10, fwhm=4.0, threshold_sigma=8.0)
    assert len(sources) == 0


# ---------------------------------------------------------------------------
# estimate_fwhm
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sigma", [1.5, 2.0, 3.0])
def test_estimate_fwhm_recovers_known_sigma(sigma):
    x, y = 64.0, 64.0
    data = _star_field([(x, y)], [80000.0], sigma=sigma, shape=(128, 128), noise=2.0)
    fwhm = estimate_fwhm(data, x, y, box_size=24)
    assert fwhm is not None
    assert fwhm == pytest.approx(SIGMA_TO_FWHM * sigma, abs=0.5)


def test_estimate_fwhm_returns_none_for_tiny_box():
    data = _star_field([(64.0, 64.0)], [80000.0], sigma=2.0, shape=(128, 128))
    # A box of 4 px straddling the edge is too small for a meaningful fit.
    assert estimate_fwhm(data, 0.0, 0.0, box_size=4) is None


# ---------------------------------------------------------------------------
# extract_point_sources (sidereal)
# ---------------------------------------------------------------------------
def test_extract_point_sources_counts_and_positions():
    truth = [(40.0, 50.0), (110.0, 70.0), (180.0, 160.0), (70.0, 200.0), (220.0, 60.0)]
    data = _star_field(truth, [50000.0] * len(truth), sigma=2.0, noise=4.0)
    image = _processed_image(data)
    # Cap detections at the true count: the brightest sources are the planted
    # stars, so all of them are recovered at sub-pixel accuracy.  (The 3-sigma
    # second pass also picks up noise peaks, hence we don't assert an upper
    # bound on the total count here.)
    starlist, _ = extract_point_sources(image, max_detections=len(truth))
    detected = [(d.x, d.y) for d in starlist.detections]
    assert _match_count(detected, truth, tol=1.5) == len(truth)


def test_extract_point_sources_reports_reasonable_fwhm():
    sigma = 2.5
    truth = [(50.0, 50.0), (150.0, 90.0), (90.0, 190.0), (200.0, 200.0)]
    data = _star_field(truth, [70000.0] * len(truth), sigma=sigma, noise=3.0)
    image = _processed_image(data)
    _, fwhm = extract_point_sources(image, max_detections=50)
    assert fwhm == pytest.approx(SIGMA_TO_FWHM * sigma, abs=1.0)


def test_extract_point_sources_faint_below_threshold_not_detected():
    bright = (64.0, 64.0)
    faint = (180.0, 180.0)
    # Bright star is ~50000 total over the PSF; faint star is buried in the
    # noise floor (~1.5x sigma peak) and must not be detected.
    data = _star_field([bright, faint], [50000.0, 30.0], sigma=2.0, noise=5.0)
    image = _processed_image(data)
    starlist, _ = extract_point_sources(image, max_detections=50)
    detected = [(d.x, d.y) for d in starlist.detections]
    assert _match_count(detected, [bright], tol=1.5) == 1
    assert _match_count(detected, [faint], tol=2.0) == 0


def test_extract_point_sources_respects_max_detections():
    truth = [(x, y) for x in (40, 80, 120, 160, 200) for y in (40, 80, 120, 160, 200)]
    data = _star_field(truth, [40000.0] * len(truth), sigma=2.0)
    image = _processed_image(data)
    starlist, _ = extract_point_sources(image, max_detections=10)
    assert len(starlist.detections) <= 10


def test_extract_point_sources_min_separation_dedupes():
    # Two stars closer than the enforced minimum separation: only one survives.
    truth = [(100.0, 100.0), (103.0, 100.0)]
    data = _star_field(truth, [60000.0, 55000.0], sigma=2.0)
    image = _processed_image(data)
    starlist, _ = extract_point_sources(image, max_detections=50, min_separation=10.0)
    near = [d for d in starlist.detections if abs(d.x - 100.0) < 6 and abs(d.y - 100.0) < 6]
    assert len(near) == 1


def test_extract_point_sources_empty_field_returns_default_fwhm():
    rng = np.random.default_rng(99)
    data = 100.0 + rng.normal(0.0, 5.0, (128, 128))
    image = _processed_image(data)
    starlist, fwhm = extract_point_sources(image, max_detections=50)
    assert len(starlist.detections) == 0
    assert fwhm == pytest.approx(4.0)  # DEFAULT_FWHM when nothing is found


# ---------------------------------------------------------------------------
# measure_fwhm_from_catalog_stars
# ---------------------------------------------------------------------------
def test_measure_fwhm_from_catalog_stars_recovers_sigma():
    sigma = 2.0
    # Well-separated stars so the isolation filter keeps them all.
    truth = [(40.0, 40.0), (120.0, 60.0), (60.0, 160.0), (200.0, 120.0), (180.0, 210.0)]
    data = _star_field(truth, [80000.0] * len(truth), sigma=sigma, noise=3.0)
    image = _processed_image(data)
    catalog = [
        StarInSpace(ra=0.0, dec=0.0, magnitude=12.0 + i, x=x, y=y)
        for i, (x, y) in enumerate(truth)
    ]
    stats = measure_fwhm_from_catalog_stars(image, catalog, initial_fwhm=4.0)
    assert stats.n_measurements >= 2
    assert stats.median_fwhm == pytest.approx(SIGMA_TO_FWHM * sigma, abs=1.0)


def test_measure_fwhm_skips_stars_without_positions():
    sigma = 2.0
    truth = [(50.0, 50.0), (150.0, 150.0)]
    data = _star_field(truth, [70000.0] * 2, sigma=sigma, noise=3.0)
    image = _processed_image(data)
    catalog = [
        StarInSpace(ra=0.0, dec=0.0, magnitude=12.0, x=50.0, y=50.0),
        StarInSpace(ra=1.0, dec=1.0, magnitude=13.0, x=None, y=None),
        StarInSpace(ra=2.0, dec=2.0, magnitude=14.0, x=150.0, y=150.0),
    ]
    stats = measure_fwhm_from_catalog_stars(image, catalog, initial_fwhm=4.0)
    # Only the two positioned stars (plus the seeded initial value) contribute.
    assert stats.median_fwhm == pytest.approx(SIGMA_TO_FWHM * sigma, abs=1.0)


def test_measure_fwhm_empty_catalog_falls_back_to_initial():
    data = _star_field([(64.0, 64.0)], [50000.0], sigma=2.0, shape=(128, 128))
    image = _processed_image(data)
    stats = measure_fwhm_from_catalog_stars(image, [], initial_fwhm=3.7)
    # No catalog measurements -> single value -> falls back to initial.
    assert stats.median_fwhm == pytest.approx(3.7, abs=1e-6)


# ---------------------------------------------------------------------------
# satellite.extract_point_sources (rate frame)
# ---------------------------------------------------------------------------
def _rate_frame(data: np.ndarray, pixel_fwhm: float = 4.0) -> RateTrackFrame:
    image = _processed_image(data, image_id="rate")
    metadata = image.metadata
    starfield = StarField(
        detections=[],
        image_metadata=metadata,
        wcs=None,
        detection_metadata=DetectionMetadata(pixel_fwhm=pixel_fwhm),
    )
    return RateTrackFrame(
        frame=image,
        starfield=starfield,
        index=0,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_satellite_extract_detects_point_source():
    cfg = get_config()
    cfg.detection.snr_threshold = 3.0
    sigma = 2.0  # FWHM ~4.7 px, matches pixel_fwhm guess
    # A grid of point sources so DAOStarFinder reaches its >=50 source floor;
    # one bright target plus many fainter companions.
    rng = np.random.default_rng(5)
    positions = [(float(x), float(y)) for x in range(30, 480, 40) for y in range(30, 480, 40)]
    fluxes = [40000.0] * len(positions)
    target = (256.0, 256.0)
    positions.append(target)
    fluxes.append(120000.0)
    data = np.full((512, 512), 100.0) + rng.normal(0.0, 5.0, (512, 512))
    for (x, y), flux in zip(positions, fluxes, strict=True):
        _add_gaussian(data, x, y, flux, sigma)
    frame = _rate_frame(data, pixel_fwhm=SIGMA_TO_FWHM * sigma)
    result = extract_satellite_points(frame)
    detected = [(d.x, d.y) for d in result.detections]
    assert _match_count(detected, [target], tol=2.0) == 1
    for det in result.detections:
        assert det.snr is not None and det.snr > cfg.detection.snr_threshold


def test_satellite_extract_high_snr_threshold_filters_all():
    sigma = 2.0
    rng = np.random.default_rng(6)
    positions = [(float(x), float(y)) for x in range(30, 480, 40) for y in range(30, 480, 40)]
    data = np.full((512, 512), 100.0) + rng.normal(0.0, 5.0, (512, 512))
    for x, y in positions:
        _add_gaussian(data, x, y, 30000.0, sigma)
    frame = _rate_frame(data, pixel_fwhm=SIGMA_TO_FWHM * sigma)
    cfg = get_config()
    original = cfg.detection.snr_threshold
    cfg.detection.snr_threshold = 1.0e9  # impossibly high -> nothing survives
    try:
        result = extract_satellite_points(frame)
    finally:
        cfg.detection.snr_threshold = original
    assert len(result.detections) == 0


def test_satellite_extract_assigns_no_radec_without_wcs():
    sigma = 2.0
    rng = np.random.default_rng(8)
    positions = [(float(x), float(y)) for x in range(30, 480, 40) for y in range(30, 480, 40)]
    data = np.full((512, 512), 100.0) + rng.normal(0.0, 5.0, (512, 512))
    for x, y in positions:
        _add_gaussian(data, x, y, 45000.0, sigma)
    frame = _rate_frame(data, pixel_fwhm=SIGMA_TO_FWHM * sigma)
    cfg = get_config()
    cfg.detection.snr_threshold = 3.0
    result = extract_satellite_points(frame)
    assert len(result.detections) > 0
    for det in result.detections:
        assert det.ra is None
        assert det.dec is None
        assert det.pixel_fwhm is not None and det.pixel_fwhm > 0


# ---------------------------------------------------------------------------
# Large-format paths: binned second pass + FWHM-crop fallback
#
# On large frames the second detection pass runs on a 2x2-binned frame when
# the PSF is fat enough (FWHM >= 6 px) and accepted centroids are re-measured
# at full resolution; pass 1 (FWHM estimation) runs on a central crop with a
# full-frame fallback. These tests pin both the routing (which branch ran)
# and the contract that matters downstream: sub-pixel centroid accuracy.
# ---------------------------------------------------------------------------
def _large_field(
    sigma: float,
    shape: tuple[int, int] = (2304, 2304),
    margin: int = 150,
    spacing: int = 250,
    flux: float = 60000.0,
    seed: int = 42,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """A large synthetic field with sub-pixel jittered star positions."""
    rng = np.random.default_rng(seed)
    h, w = shape
    truth = []
    image = np.full(shape, 100.0) + rng.normal(0.0, 5.0, shape)
    for gy in range(margin, h - margin, spacing):
        for gx in range(margin, w - margin, spacing):
            x = gx + float(rng.uniform(-0.5, 0.5))
            y = gy + float(rng.uniform(-0.5, 0.5))
            _add_gaussian(image, x, y, flux, sigma)
            truth.append((x, y))
    return image, truth


def _centroid_rms(detected, truth, tol=1.5):
    """(n_matched, rms residual) of truth stars matched by a detection."""
    residuals = []
    for tx, ty in truth:
        best = None
        for dx, dy in detected:
            d2 = (dx - tx) ** 2 + (dy - ty) ** 2
            if d2 <= tol**2 and (best is None or d2 < best):
                best = d2
        if best is not None:
            residuals.append(best)
    if not residuals:
        return 0, np.inf
    return len(residuals), float(np.sqrt(np.mean(residuals)))


def _track_refiner_calls(monkeypatch):
    """Record invocations of the full-res centroid refiner (binned path only)."""
    import senpai.engine.detection.point.sidereal as sid

    calls = []
    original = sid._refine_centroid_full_res

    def wrapper(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(sid, "_refine_centroid_full_res", wrapper)
    return calls


def test_binned_pass2_runs_and_keeps_subpixel_accuracy(monkeypatch):
    sigma = 3.5  # FWHM ~8.2 px -> binned branch
    image, truth = _large_field(sigma)
    refiner_calls = _track_refiner_calls(monkeypatch)

    starlist, fwhm = extract_point_sources(
        _processed_image(image), max_detections=len(truth)
    )

    assert refiner_calls, "fat-PSF large frame must take the binned pass-2 path"
    assert fwhm == pytest.approx(SIGMA_TO_FWHM * sigma, abs=1.0)
    detected = [(d.x, d.y) for d in starlist.detections]
    n_matched, rms = _centroid_rms(detected, truth)
    assert n_matched >= 0.95 * len(truth)
    # The contract that matters for astrometry: binning must not cost
    # sub-pixel accuracy (full-res refinement recovers it).
    assert rms < 0.15


def test_small_fwhm_large_frame_stays_unbinned(monkeypatch):
    sigma = 1.7  # FWHM ~4.0 px -> binning would undersample; must not bin
    image, truth = _large_field(sigma)
    refiner_calls = _track_refiner_calls(monkeypatch)

    starlist, fwhm = extract_point_sources(
        _processed_image(image), max_detections=len(truth)
    )

    assert not refiner_calls, "well-sampled PSF must use the full-res path"
    assert fwhm == pytest.approx(SIGMA_TO_FWHM * sigma, abs=1.0)
    detected = [(d.x, d.y) for d in starlist.detections]
    n_matched, rms = _centroid_rms(detected, truth)
    assert n_matched >= 0.95 * len(truth)
    assert rms < 0.15


def test_fwhm_pass_measures_full_frame_even_with_sparse_center():
    # Stars only in the outer band of a large frame. Pass 1 must scan the
    # full frame (a central-crop variant was reverted after it skewed the
    # source-peak saturation percentile on a real calsat field and biased
    # the FWHM from a true ~9 px down to 3.1 px), so an empty center must
    # not degrade the FWHM estimate.
    sigma = 3.5
    shape = (4608, 4608)
    rng = np.random.default_rng(7)
    image = np.full(shape, 100.0) + rng.normal(0.0, 5.0, shape)
    truth = []
    band = [(x, y) for x in range(80, 4530, 220) for y in (80, 180, 4430, 4530)]
    band += [(x, y) for y in range(400, 4200, 220) for x in (80, 180, 4430, 4530)]
    for gx, gy in band:
        x, y = gx + float(rng.uniform(-0.5, 0.5)), gy + float(rng.uniform(-0.5, 0.5))
        _add_gaussian(image, x, y, 60000.0, sigma)
        truth.append((x, y))

    starlist, fwhm = extract_point_sources(
        _processed_image(image), max_detections=len(truth)
    )

    assert fwhm == pytest.approx(SIGMA_TO_FWHM * sigma, abs=1.5)
    detected = [(d.x, d.y) for d in starlist.detections]
    n_matched, _ = _centroid_rms(detected, truth)
    assert n_matched >= 0.9 * len(truth)


def test_satellite_threshold_search_matches_daostarfinder():
    """The satellite detector's shared-convolution threshold search must
    reproduce DAOStarFinder exactly at any threshold. This pins the
    private-API reimplementation (_StarFinderKernel/_DAOStarFinderCatalog):
    if a photutils upgrade moves those internals, this fails loudly.
    """
    from photutils.detection import DAOStarFinder
    from photutils.detection.daofinder import _StarFinderKernel
    from scipy.signal import fftconvolve

    from senpai.engine.detection.point.satellite import (
        _dao_sources_at_threshold,
        _local_maxima_above,
    )

    sigma = 2.2
    rng = np.random.default_rng(12)
    data = rng.normal(0.0, 5.0, (512, 512))
    positions = [(float(x), float(y)) for x in range(40, 480, 45) for y in range(40, 480, 45)]
    fluxes = rng.uniform(2000.0, 80000.0, len(positions))
    for (x, y), f in zip(positions, fluxes, strict=True):
        _add_gaussian(data, x, y, float(f), sigma)
    data = data.astype(np.float32)

    fwhm = SIGMA_TO_FWHM * sigma
    std = float(np.std(data[data < np.percentile(data, 90)]))
    kernel = _StarFinderKernel(float(fwhm), ratio=1.0, theta=0.0, sigma_radius=1.5)
    convolved = fftconvolve(data, kernel.data.astype(np.float32), mode="same")
    ys, xs, vals = _local_maxima_above(
        convolved, kernel.mask.astype(bool), 3.0 * std * kernel.relerr
    )
    cand_xy = np.column_stack((xs, ys))

    for thr_sigma in (3.0, 8.0, 25.0):
        thr = thr_sigma * std
        ref = DAOStarFinder(
            fwhm=float(fwhm), threshold=thr, sharplo=0.1, sharphi=1.5,
            roundlo=-1.5, roundhi=1.5, brightest=None, peakmax=None,
        )(data)
        got = _dao_sources_at_threshold(
            data, convolved, kernel, cand_xy, vals, thr,
            sharplo=0.1, sharphi=1.5, roundlo=-1.5, roundhi=1.5,
        )
        n_ref = 0 if ref is None else len(ref)
        n_got = 0 if got is None else len(got)
        assert n_got == n_ref, f"count mismatch at {thr_sigma} sigma: {n_got} != {n_ref}"
        if n_ref:
            assert np.allclose(np.sort(np.asarray(got["xcentroid"])), np.sort(np.asarray(ref["xcentroid"])), atol=1e-3)
            assert np.allclose(np.sort(np.asarray(got["ycentroid"])), np.sort(np.asarray(ref["ycentroid"])), atol=1e-3)


# ---------------------------------------------------------------------------
# measure_fwhm_from_catalog_stars: saturation + winged-PSF behavior
# ---------------------------------------------------------------------------
def test_catalog_fwhm_skips_clipped_stars_and_measures_truth():
    # A field with a saturated pile (clipped cores) plus unsaturated stars:
    # the measured FWHM must come from the unsaturated cohort and match the
    # true PSF width — the old Gaussian-fit path measured only the faintest
    # stars (broken catalog-sample sat level) and read ~1.5x wide.
    sigma = 3.8  # FWHM ~8.9 px
    rng = np.random.default_rng(21)
    data = np.full((1400, 1400), 0.0) + rng.normal(0.0, 5.0, (1400, 1400))
    catalog = []
    grid = [(x, y) for x in range(120, 1300, 130) for y in range(120, 1300, 130)]
    for i, (gx, gy) in enumerate(grid):
        x, y = gx + 0.3, gy + 0.4
        saturated = i % 3 == 0  # every third star is clipped
        flux = 4.0e6 if saturated else 3.0e5
        _add_gaussian(data, x, y, flux, sigma)
        catalog.append(
            StarInSpace(ra=0.0, dec=0.0, magnitude=8.0 + 0.05 * i, x=x, y=y)
        )
    ceiling = 42000.0
    np.minimum(data, ceiling, out=data)  # clip the bright cores

    image = _processed_image(data)
    stats = measure_fwhm_from_catalog_stars(image, catalog, initial_fwhm=8.0)
    assert stats.median_fwhm == pytest.approx(SIGMA_TO_FWHM * sigma, abs=1.0)


def test_catalog_fwhm_uses_detection_sat_level_when_provided():
    sigma = 3.0
    rng = np.random.default_rng(22)
    data = np.full((900, 900), 0.0) + rng.normal(0.0, 4.0, (900, 900))
    catalog = []
    for i, (gx, gy) in enumerate([(x, y) for x in range(100, 850, 120) for y in range(100, 850, 120)]):
        _add_gaussian(data, gx, gy, 2.0e5, sigma)
        catalog.append(StarInSpace(ra=0.0, dec=0.0, magnitude=9.0 + 0.1 * i, x=float(gx), y=float(gy)))

    image = _processed_image(data)
    # An absurdly low explicit sat level marks every star saturated ->
    # no measurements -> falls back to the initial value. Proves the
    # passed-through level is honored rather than re-estimated.
    stats = measure_fwhm_from_catalog_stars(
        image, catalog, initial_fwhm=6.5, sat_level=10.0
    )
    assert stats.median_fwhm == pytest.approx(6.5, abs=1e-6)

    # And with the true (permissive) level, the real PSF width is measured.
    stats = measure_fwhm_from_catalog_stars(
        image, catalog, initial_fwhm=6.5, sat_level=1.0e9
    )
    assert stats.median_fwhm == pytest.approx(SIGMA_TO_FWHM * sigma, abs=1.0)


def test_catalog_fwhm_reads_profile_width_on_winged_psf():
    # Gaussian core + broad shallow wings (Moffat-like): the FWHM is the
    # composite profile's half-max width — slightly above the core's, far
    # below what a wing-absorbing single-Gaussian fit can drift to, and far
    # above the in-box-background half-max-area measure (which folds wing
    # flux into the sky and read ~30% narrow on real winged frames).
    sigma = 3.8
    rng = np.random.default_rng(23)
    data = np.full((1200, 1200), 0.0) + rng.normal(0.0, 4.0, (1200, 1200))
    catalog = []
    for i, (gx, gy) in enumerate([(x, y) for x in range(140, 1100, 150) for y in range(140, 1100, 150)]):
        x, y = float(gx) + 0.2, float(gy) - 0.3
        _add_gaussian(data, x, y, 2.5e5, sigma)
        _add_gaussian(data, x, y, 1.2e5, sigma * 3.0)  # wings
        catalog.append(StarInSpace(ra=0.0, dec=0.0, magnitude=9.0 + 0.1 * i, x=x, y=y))

    image = _processed_image(data)
    stats = measure_fwhm_from_catalog_stars(image, catalog, initial_fwhm=8.0)
    # Wings raise the half-max slightly; allow modest tolerance but pin it
    # far below the Gaussian-fit failure mode (~1.5x).
    assert stats.median_fwhm < 1.25 * SIGMA_TO_FWHM * sigma
    assert stats.median_fwhm > 0.8 * SIGMA_TO_FWHM * sigma
