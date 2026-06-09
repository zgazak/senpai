"""FWHM measurement from catalog stars."""

import logging

import numpy as np

from senpai.engine.detection.point.sidereal import estimate_fwhm, sat_level_from_peaks
from senpai.engine.models.metadata import FWHMMetadata

logger = logging.getLogger(__name__)


def measure_fwhm_from_catalog_stars(
    fits_image,
    catalog_stars: list,
    initial_fwhm: float,
    config=None,
) -> FWHMMetadata:
    """
    Measure FWHM from well-isolated catalog stars.

    Parameters
    ----------
    fits_image : ProcessedFitsImage
        The FITS image containing the stars
    catalog_stars : list
        List of catalog stars with x, y coordinates
    initial_fwhm : float
        Initial FWHM estimate from detection
    config : AppConfig, optional
        Configuration object for target FWHM

    Returns
    -------
    FWHMMetadata
        FWHM statistics and measurements
    """
    logger.info(f"Measuring FWHM from {len(catalog_stars)} catalog stars")

    if config is None:
        from senpai.core.config import get_config

        config = get_config()

    # Limit the number of catalog stars considered for FWHM measurements to
    # keep this step fast even when Gaia returns tens of thousands of stars.
    # We only need a modest number of well-isolated stars to get robust FWHM
    # statistics; beyond a few dozen measurements the median is very stable.
    # The candidate pool must reach well past the frame's saturated bright
    # end though: the saturation limit moves ~2.5 mag between 1 s and 10 s
    # exposures, and with a 100-star pool a long exposure's pool was almost
    # entirely saturated/bloomed stars — Gaussian fits on those returned
    # 2-3x the true FWHM, which (via FWHM-scaled apertures) silently
    # destroyed long-exposure photometric SNR and depth.
    MAX_CATALOG_FOR_FWHM = 400
    MAX_FWHM_MEASUREMENTS = 30

    # Filter to stars with valid positions
    valid_catalog_stars = [
        star for star in catalog_stars if star.x is not None and star.y is not None
    ]

    # Prefer brighter stars when sub-selecting
    valid_catalog_stars.sort(
        key=lambda s: s.magnitude if getattr(s, "magnitude", None) is not None else 99.0
    )
    if len(valid_catalog_stars) > MAX_CATALOG_FOR_FWHM:
        valid_catalog_stars = valid_catalog_stars[:MAX_CATALOG_FOR_FWHM]

    # Per-frame saturation level from the candidates' own core peaks (the
    # bright end piles up just under the clipped-and-subtracted ceiling);
    # saturated stars are excluded from FWHM fitting below. Same logic the
    # first-pass detection estimator uses — see sat_level_from_peaks.
    data = fits_image.data
    star_peaks: dict[int, float] = {}
    for star in valid_catalog_stars:
        x0, y0 = int(round(star.x)), int(round(star.y))
        core = data[max(0, y0 - 2): y0 + 3, max(0, x0 - 2): x0 + 3]
        if core.size:
            star_peaks[id(star)] = float(core.max())
    sat_level = sat_level_from_peaks(list(star_peaks.values()))

    # Build array of positions
    positions = np.array(
        [[star.x, star.y] for star in valid_catalog_stars], dtype=float
    )
    n = len(valid_catalog_stars)

    fwhm_measurements: list[float] = []
    fwhm_vs_position: list[tuple[float, float, float]] = []
    fwhm_vs_magnitude: list[tuple[float, float]] = []
    fwhm_vs_counts: list[tuple[float, float]] = []

    if n > 0:
        # Compute pairwise distances once, and mark isolated stars as those
        # without neighbors within 5 * initial_fwhm.
        dx = positions[:, 0][:, None] - positions[:, 0][None, :]
        dy = positions[:, 1][:, None] - positions[:, 1][None, :]
        dist2 = dx * dx + dy * dy

        r_iso = 5.0 * initial_fwhm
        r2_iso = float(r_iso * r_iso)

        # Mask out self-distances
        np.fill_diagonal(dist2, np.inf)
        neighbor_mask = dist2 < r2_iso
        isolated_mask = ~neighbor_mask.any(axis=1)

        n_saturated = 0
        for star, is_isolated in zip(valid_catalog_stars, isolated_mask, strict=False):
            if not is_isolated:
                continue
            if len(fwhm_measurements) >= MAX_FWHM_MEASUREMENTS:
                break
            peak = star_peaks.get(id(star))
            if peak is not None and peak >= sat_level:
                n_saturated += 1
                continue  # saturated/bloomed — a Gaussian fit returns garbage

            try:
                fwhm = estimate_fwhm(fits_image.data, star.x, star.y)
                if fwhm is not None and fwhm > 0:
                    fwhm_measurements.append(fwhm)
                    fwhm_vs_position.append((star.x, star.y, fwhm))
                    if star.magnitude is not None:
                        fwhm_vs_magnitude.append((star.magnitude, fwhm))
                    if hasattr(star, "counts") and star.counts is not None:
                        fwhm_vs_counts.append((star.counts, fwhm))
            except Exception as e:
                logger.debug(f"FWHM estimation failed for catalog star: {e!s}")
                continue

        logger.info(
            "Catalog FWHM sample: %d measured, %d saturated skipped (sat_level=%s)",
            len(fwhm_measurements), n_saturated,
            f"{sat_level:.0f}" if np.isfinite(sat_level) else "inf",
        )

    # Combine with initial measurement
    fwhm_measurements.append(initial_fwhm)

    # Robustly clip outliers before computing statistics
    if len(fwhm_measurements) >= 3:
        vals = np.array(fwhm_measurements, dtype=float)
        median = float(np.median(vals))
        mad = float(np.median(np.abs(vals - median))) or 1.0
        good_mask = np.abs(vals - median) <= 5.0 * mad
        vals = vals[good_mask]
    else:
        vals = np.array(fwhm_measurements, dtype=float)

    if vals.size >= 2:
        median_fwhm = float(np.median(vals))
        mean_fwhm = float(np.mean(vals))
        std_fwhm = float(np.std(vals))
        min_fwhm = float(np.min(vals))
        max_fwhm = float(np.max(vals))
    else:
        median_fwhm = mean_fwhm = initial_fwhm
        std_fwhm = 0.0
        min_fwhm = max_fwhm = initial_fwhm

    fwhm_stats = FWHMMetadata(
        n_measurements=int(vals.size),
        median_fwhm=median_fwhm,
        mean_fwhm=mean_fwhm,
        std_fwhm=std_fwhm,
        min_fwhm=min_fwhm,
        max_fwhm=max_fwhm,
        fwhm_vs_position=fwhm_vs_position,
        fwhm_vs_magnitude=fwhm_vs_magnitude,
        fwhm_vs_counts=fwhm_vs_counts,
        is_oversampled=median_fwhm > config.calibrations.target_fwhm,
        recommended_scale_factor=(
            median_fwhm / config.calibrations.target_fwhm
            if median_fwhm > config.calibrations.target_fwhm
            else None
        ),
    )

    logger.info(
        "Refined FWHM from %.3f to %.3f using %d stars (after clipping)",
        initial_fwhm,
        median_fwhm,
        vals.size,
    )

    return fwhm_stats
