"""
Simple photometry utilities for extracting basic photometric information from astronomical images.

This module provides basic photometry tools for:
- Simple aperture photometry with fixed aperture size
- Basic background estimation
- Simple quality assessment
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from senpai.engine.models.metadata import StreakMetadata
    from senpai.engine.photometry.color_terms import MultiBandCalibration

import numpy as np
from photutils.aperture import (
    CircularAnnulus,
    CircularAperture,
    RectangularAnnulus,
    RectangularAperture,
    aperture_photometry,
)
from scipy.spatial import cKDTree

from senpai.core.config import PhotometryConfig, get_config
from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.models.senpai import RateTrackFrame, SiderealFrame
from senpai.engine.models.starfield import SatelliteListImage, StarField, StarInSpace

logger = logging.getLogger(__name__)


def _normalize_photometry_config(
    config: PhotometryConfig | None = None,
) -> PhotometryConfig:
    """Return the given PhotometryConfig, or a default one if None."""

    if isinstance(config, PhotometryConfig):
        return config
    return PhotometryConfig()


# Backward-compatible alias: the photometry config now lives in senpai.core.config.
SimplePhotometryConfig = PhotometryConfig


@dataclass
class SimplePhotometryResult:
    """Result of simple photometric measurement for a single star."""

    # Star information
    star: StarInSpace

    # Basic photometry results
    flux: float  # Raw flux in ADU
    flux_err: float  # Uncertainty in flux
    snr: float  # Signal-to-noise ratio

    # Background
    background_level: float  # Background level per pixel
    background_std: float  # Background noise per pixel (standard deviation)

    # Aperture info
    aperture_radius: float  # Aperture radius in pixels

    # Quality metrics
    crowding_factor: float  # Crowding factor (0-1)
    quality_flag: bool  # Overall quality flag

    # Additional info
    instrumental_magnitude: float | None = None  # Instrumental magnitude

    def __post_init__(self):
        self.flux = round(self.flux, 2)
        self.flux_err = round(self.flux_err, 2)
        self.snr = round(self.snr, 2)
        self.background_level = round(self.background_level, 2)
        self.background_std = round(self.background_std, 2)
        self.aperture_radius = round(self.aperture_radius, 2)
        self.crowding_factor = round(self.crowding_factor, 3)
        if self.instrumental_magnitude is not None:
            self.instrumental_magnitude = round(self.instrumental_magnitude, 3)


@dataclass
class SimplePhotometrySummary:
    """Summary statistics from simple photometry measurements."""

    n_stars: int  # Number of stars measured
    n_quality: int  # Number of quality measurements
    median_snr: float  # Median signal-to-noise ratio
    median_background: float  # Median background level
    limiting_magnitude: float  # Estimated limiting magnitude (at configured completeness, typically 50%)
    zero_point: float | None = None  # Photometric zero point
    zero_point_err: float | None = None  # Zero point uncertainty
    limiting_snr: float | None = None  # SNR threshold used for limiting magnitude calculation
    limiting_magnitude_50: float | None = None  # Limiting magnitude at 50% completeness
    limiting_magnitude_90: float | None = None  # Limiting magnitude at 90% completeness
    multiband_calibration: "MultiBandCalibration | None" = None  # Multi-band ZP with color terms
    # Per-frame completeness curve: parallel arrays of (magnitude, detection %)
    completeness_mag: list[float] | None = None
    completeness_pct: list[float] | None = None
    # Per-detected-star catalog magnitude and measured SNR — parallel arrays
    # used by downstream observability plots (search rate, SNR-vs-time binned
    # by magnitude). Optional because the legacy zero/no-results path leaves
    # them None.
    stars_mag: list[float] | None = None
    stars_snr: list[float] | None = None
    # Per-star zero-point offset (m_cat − m_inst), parallel to stars_mag.
    # Feeds the per-star extinction curve (offset vs airmass); None entries
    # mark stars whose instrumental magnitude couldn't be measured.
    stars_zp_offset: list[float | None] | None = None
    # Per-star isolation flag, parallel to stars_mag: False when a brighter
    # catalog star sits within the aperture footprint (the measured flux is
    # then the blend's, not this star's — the dominant fake-faint-SNR source).
    # Downstream plots/aggregates should drop non-isolated stars.
    stars_isolated: list[bool] | None = None

    def __post_init__(self):
        self.median_snr = round(self.median_snr, 2)
        self.median_background = round(self.median_background, 2)
        self.limiting_magnitude = round(self.limiting_magnitude, 3)
        if self.zero_point is not None:
            self.zero_point = round(self.zero_point, 3)
        if self.zero_point_err is not None:
            self.zero_point_err = round(self.zero_point_err, 4)
        if self.limiting_magnitude_50 is not None:
            self.limiting_magnitude_50 = round(self.limiting_magnitude_50, 3)
        if self.limiting_magnitude_90 is not None:
            self.limiting_magnitude_90 = round(self.limiting_magnitude_90, 3)


def empirical_background_std_adu(image_data: np.ndarray) -> float:
    """Robust per-pixel background noise (ADU) measured from the pixels.

    Frames are background-subtracted upstream (row/col median), so the
    background *level* near any star is ~0 and carries no noise information:
    a level-based Poisson model collapses flux_err to sqrt(source) and
    inflates faint-star SNR ~8× (mag-20 "SNR 7" on frames whose detection
    limit is mag 18). The empirical pixel std already contains sky Poisson +
    read noise, so noise models built on it must not add those terms again.

    MAD over an 8×-strided subsample is robust to stars and matches the
    full-frame robust std to ~1% at 1/64 the cost. Returns 0.0 for degenerate
    (constant/empty) images — callers should fall back to a level model.
    """
    sub = image_data[::8, ::8]
    sub = sub[np.isfinite(sub)]
    if not sub.size:
        return 0.0
    return float(np.median(np.abs(sub - np.median(sub))) * 1.4826)


def measure_simple_star_photometry(
    image: ProcessedFitsImage,
    star: StarInSpace,
    fwhm: float,
    config: SimplePhotometryConfig | None = None,
) -> SimplePhotometryResult | None:
    """
    Perform simple photometry on a single star.

    Parameters
    ----------
    image : ProcessedFitsImage
        The image containing the star
    star : StarInSpace
        Star object with x, y coordinates
    fwhm : float
        FWHM of the star in pixels
    config : SimplePhotometryConfig, optional
        Photometry configuration

    Returns
    -------
    SimplePhotometryResult or None
        Photometry results if successful, None if failed
    """
    config = _normalize_photometry_config(config)

    if star.x is None or star.y is None:
        return None

    # Get image dimensions
    height, width = image.data.shape

    # Calculate fixed aperture and background annulus
    aperture_radius = config.aperture_radius_factor * fwhm
    bg_inner = config.bg_inner_factor * fwhm
    bg_outer = config.bg_outer_factor * fwhm

    # Check if star is within image bounds with margin
    margin = int(bg_outer + 10)
    if star.x < margin or star.x >= width - margin or star.y < margin or star.y >= height - margin:
        return None

    try:
        # Create apertures
        aperture = CircularAperture((star.x, star.y), r=aperture_radius)
        bg_aperture = CircularAnnulus((star.x, star.y), r_in=bg_inner, r_out=bg_outer)

        # Measure aperture photometry (single pass for both apertures)
        phot = aperture_photometry(image.data, [aperture, bg_aperture])

        # Get raw fluxes
        flux_sum = float(phot["aperture_sum_0"][0])
        bg_sum = float(phot["aperture_sum_1"][0])
        bg_area = bg_aperture.area

        # Calculate background level per pixel
        background_level = bg_sum / bg_area if bg_area > 0 else 0

        # Background subtraction
        flux = flux_sum - (background_level * aperture.area)

        # Estimate background noise (standard deviation per pixel) using background annulus
        bg_std = 0.0
        try:
            bg_masks = bg_aperture.to_mask(method="center")

            # photutils may return a single mask or a list of masks
            if isinstance(bg_masks, list):
                bg_pixels_list = []
                for mask in bg_masks:
                    data_sub = mask.multiply(image.data)
                    bg_pixels_list.append(data_sub[mask.data > 0])
                if bg_pixels_list:
                    bg_pixels = np.concatenate(bg_pixels_list)
                else:
                    bg_pixels = np.array([])
            else:
                data_sub = bg_masks.multiply(image.data)
                bg_pixels = data_sub[bg_masks.data > 0]

            # Remove non-finite values
            bg_pixels = bg_pixels[np.isfinite(bg_pixels)]
            if bg_pixels.size > 0:
                bg_std = float(np.std(bg_pixels))
        except Exception as e:
            logger.debug(
                f"Background noise estimation from annulus failed for star at ({star.x:.1f}, {star.y:.1f}): {e}"
            )

        # Fallback: use a local box around the star if annulus-based estimate failed
        if bg_std <= 0:
            try:
                y_min = max(0, int(star.y - bg_outer))
                y_max = min(image.data.shape[0], int(star.y + bg_outer))
                x_min = max(0, int(star.x - bg_outer))
                x_max = min(image.data.shape[1], int(star.x + bg_outer))
                region = image.data[y_min:y_max, x_min:x_max]
                if region.size > 0:
                    bg_std = float(np.std(region))
            except Exception as e:
                logger.debug(
                    f"Background noise fallback estimation failed for star at ({star.x:.1f}, {star.y:.1f}): {e}"
                )
                bg_std = 0.0

        # Simple uncertainty estimation (background-noise dominated)
        flux_err = bg_std * np.sqrt(aperture.area) if bg_std > 0 else 0.0

        # Calculate SNR
        snr = flux / flux_err if flux_err > 0 else 0

        # Calculate instrumental magnitude
        instrumental_magnitude = None
        if flux > 0:
            instrumental_magnitude = -2.5 * np.log10(flux)

        # Simple quality assessment
        crowding_factor = _calculate_simple_crowding(image.data, star.x, star.y, aperture_radius)
        quality_flag = _assess_simple_quality(flux, snr, crowding_factor, config)

        return SimplePhotometryResult(
            star=star,
            flux=flux,
            flux_err=flux_err,
            snr=snr,
            background_level=background_level,
            background_std=bg_std,
            aperture_radius=aperture_radius,
            crowding_factor=crowding_factor,
            quality_flag=quality_flag,
            instrumental_magnitude=instrumental_magnitude,
        )

    except Exception as e:
        logger.debug(f"Photometry failed for star at ({star.x:.1f}, {star.y:.1f}): {e}")
        return None


def _chunked_aperture_sums(data, positions, build_apertures, bbox_side):
    """Run aperture_photometry in position-chunks to bound peak memory.

    photutils materializes every aperture's bounding-box mask at once, so a
    bulk call over thousands of FWHM-/streak-scaled apertures can allocate
    tens of GB and draw the OOM killer (observed twice in burr night runs:
    rate streak apertures, then circular apertures on a 112k-star field).
    Slice the positions into chunks sized so each call's masks stay near
    ~2 GB; results are bit-for-bit identical to one call.

    build_apertures(pos_subset) -> [aperture, bg_aperture].
    Returns (aperture_sum_0_array, aperture_sum_1_array) as float arrays.
    """
    n = len(positions)
    per_star_bytes = 3 * 8 * bbox_side * bbox_side  # masks + cutout products
    chunk = max(64, min(n, int(2e9 / max(per_star_bytes, 1))))
    f0, f1 = [], []
    for lo in range(0, n, chunk):
        aps = build_apertures(positions[lo:lo + chunk])
        t = aperture_photometry(data, aps, method="subpixel", subpixels=5)
        f0.append(np.asarray(t["aperture_sum_0"], dtype=float))
        f1.append(np.asarray(t["aperture_sum_1"], dtype=float))
    return np.concatenate(f0), np.concatenate(f1)


def measure_simple_starfield_photometry(
    image: ProcessedFitsImage,
    starfield: StarField,
    config: SimplePhotometryConfig | None = None,
    frame_index: int | None = None,
) -> tuple[list[SimplePhotometryResult], SimplePhotometrySummary]:
    """
    Perform simple photometry on all stars in a starfield.

    Parameters
    ----------
    image : ProcessedFitsImage
        The image containing the stars
    starfield : StarField
        Starfield with detected and catalog stars
    config : SimplePhotometryConfig, optional
        Photometry configuration

    Returns
    -------
    tuple
        (photometry_results, summary_statistics)
    """
    config = _normalize_photometry_config(config)

    # Get FWHM from starfield
    if starfield.fwhm_stats is None:
        logger.warning("No FWHM information available in starfield")
        return [], SimplePhotometrySummary(
            n_stars=0,
            n_quality=0,
            median_snr=0,
            median_background=0,
            limiting_magnitude=0,
        )

    fwhm = starfield.fwhm_stats.median_fwhm

    # Only use catalog stars
    catalog_stars = starfield.catalog_stars or []

    if not catalog_stars:
        logger.warning("No catalog stars found in starfield for photometry")
        return [], SimplePhotometrySummary(
            n_stars=0,
            n_quality=0,
            median_snr=0,
            median_background=0,
            limiting_magnitude=0,
        )

    # ------------------------------------------------------------------
    # Vectorized aperture photometry for all catalog stars at once
    # ------------------------------------------------------------------
    height, width = image.data.shape

    aperture_radius = config.aperture_radius_factor * fwhm
    bg_inner = config.bg_inner_factor * fwhm
    bg_outer = config.bg_outer_factor * fwhm

    # Margin to keep stars away from the edges (so bg annulus fits in frame)
    margin = int(bg_outer + 10)

    positions = []
    valid_stars: list[StarInSpace] = []
    for star in catalog_stars:
        if star.x is None or star.y is None:
            continue
        if star.x < margin or star.x >= width - margin or star.y < margin or star.y >= height - margin:
            continue
        positions.append((star.x, star.y))
        valid_stars.append(star)

    if not valid_stars:
        logger.warning("No catalog stars within valid bounds for photometry")
        return [], SimplePhotometrySummary(
            n_stars=0,
            n_quality=0,
            median_snr=0,
            median_background=0,
            limiting_magnitude=0,
        )

    # Sample stars by magnitude bins to limit processing time while maintaining
    # good statistics across the magnitude range for limiting magnitude calculation
    MAX_STARS_PER_BIN = 500
    MAG_BIN_WIDTH = 1.0  # 1 magnitude bin width

    # Extract magnitudes for all valid stars
    star_magnitudes = []
    for star in valid_stars:
        # Get best available magnitude
        mag = None
        if hasattr(star, "magnitude") and star.magnitude is not None:
            mag = star.magnitude
        elif hasattr(star, "magnitudes") and star.magnitudes:
            # Use first available magnitude from dict
            mag = next(iter(star.magnitudes.values()))

        star_magnitudes.append(mag if mag is not None else 99.0)  # Default to very faint if no mag

    star_magnitudes = np.array(star_magnitudes)

    # Bin stars by magnitude
    valid_mags = star_magnitudes[star_magnitudes < 99.0]
    if len(valid_mags) == 0:
        # Fallback: no valid magnitudes, use all stars (but limit total)
        sampled_stars = valid_stars[:5000] if len(valid_stars) > 5000 else valid_stars
        logger.info(
            f"Sampling photometry: {len(sampled_stars)}/{len(valid_stars)} stars "
            f"(no magnitude info, using first {len(sampled_stars)})"
        )
    else:
        mag_min = float(np.nanmin(valid_mags))
        mag_max = float(np.nanmax(valid_mags))

        # Create magnitude bins
        n_bins = int(np.ceil((mag_max - mag_min) / MAG_BIN_WIDTH)) + 1
        bin_edges = np.linspace(mag_min, mag_max + MAG_BIN_WIDTH, n_bins + 1)

        sampled_indices = []
        for i in range(n_bins):
            bin_mask = (star_magnitudes >= bin_edges[i]) & (star_magnitudes < bin_edges[i + 1])
            bin_indices = np.where(bin_mask)[0]

            if len(bin_indices) > MAX_STARS_PER_BIN:
                # Deterministically sample evenly spaced stars from this bin
                # Sort by magnitude first for consistent sampling
                bin_mags = star_magnitudes[bin_indices]
                sorted_idx = np.argsort(bin_mags)
                sorted_bin_indices = bin_indices[sorted_idx]
                step = len(sorted_bin_indices) / MAX_STARS_PER_BIN
                selected_idx = [sorted_bin_indices[int(j * step)] for j in range(MAX_STARS_PER_BIN)]
                sampled_indices.extend(selected_idx)
            else:
                # Use all stars in this bin
                sampled_indices.extend(bin_indices)

        sampled_indices = np.array(sampled_indices)
        sampled_stars = [valid_stars[i] for i in sampled_indices]

        logger.info(
            f"Sampling photometry: {len(sampled_stars)}/{len(valid_stars)} stars "
            f"(max {MAX_STARS_PER_BIN} per {MAG_BIN_WIDTH:.1f} mag bin, "
            f"range {mag_min:.1f}-{mag_max:.1f} mag)"
        )

    # Rebuild positions array with sampled stars
    positions = np.array([(star.x, star.y) for star in sampled_stars])
    valid_stars = sampled_stars

    # Run photutils aperture photometry. method="subpixel" (vs default
    # "exact") for speed; <0.01% flux difference.
    #
    # Chunked: photutils materializes every aperture's bounding-box mask at
    # once. Apertures are FWHM-scaled, so on a dense field (thousands of
    # mag-sampled stars) with an inflated FWHM the annulus bboxes are huge —
    # a single call peaked ~42 GB and drew the OOM killer (observed live on a
    # 112k-star galactic field, _full7). Slice positions into ~2 GB chunks
    # sized from the annulus bbox; results are identical. Mirrors the rate path.
    bbox_side = 2.0 * bg_outer + 2.0
    per_star_bytes = 3 * 8 * bbox_side * bbox_side  # masks + cutout products
    chunk_size = max(64, min(len(positions), int(2e9 / max(per_star_bytes, 1))))

    flux_parts, bg_parts = [], []
    for lo in range(0, len(positions), chunk_size):
        pos_chunk = positions[lo:lo + chunk_size]
        aperture = CircularAperture(pos_chunk, r=aperture_radius)
        bg_aperture = CircularAnnulus(pos_chunk, r_in=bg_inner, r_out=bg_outer)
        phot = aperture_photometry(
            image.data, [aperture, bg_aperture], method="subpixel", subpixels=5
        )
        flux_parts.append(np.asarray(phot["aperture_sum_0"], dtype=float))
        bg_parts.append(np.asarray(phot["aperture_sum_1"], dtype=float))

    flux_sum = np.concatenate(flux_parts)
    bg_sum = np.concatenate(bg_parts)

    # Areas below use the full-position apertures (area is per-aperture
    # identical for circles), so rebuild lightweight aperture objects once.
    aperture = CircularAperture(positions, r=aperture_radius)
    bg_aperture = CircularAnnulus(positions, r_in=bg_inner, r_out=bg_outer)

    # Areas can be scalar or array depending on photutils version / usage
    aper_area = aperture.area
    bg_area = bg_aperture.area
    if hasattr(aper_area, "__len__"):
        aper_area = np.asarray(aper_area, dtype=float)
    else:
        aper_area = float(aper_area)
    if hasattr(bg_area, "__len__"):
        bg_area = np.asarray(bg_area, dtype=float)
    else:
        bg_area = float(bg_area)

    # Broadcast areas if needed
    aper_area_arr = aper_area if isinstance(aper_area, np.ndarray) else np.full_like(flux_sum, aper_area, dtype=float)
    bg_area_arr = bg_area if isinstance(bg_area, np.ndarray) else np.full_like(bg_sum, bg_area, dtype=float)

    # Background level per pixel and background-subtracted flux (ADU)
    background_level = np.where(bg_area_arr > 0, bg_sum / bg_area_arr, 0.0)
    flux = flux_sum - background_level * aper_area_arr

    # Noise model: gain/read noise from config, background noise EMPIRICAL.
    cfg = get_config()
    phot_cfg = getattr(cfg, "photometry", None)
    if phot_cfg is not None:
        gain = float(getattr(phot_cfg, "gain", 1.0))  # e-/ADU
        include_read_noise = bool(getattr(phot_cfg, "include_read_noise", False))
        read_noise = float(getattr(phot_cfg, "read_noise", 0.0)) if include_read_noise else 0.0  # e-
    else:
        gain = 1.0
        read_noise = 0.0

    n_pix = aper_area_arr

    # Convert to electrons
    source_e = np.clip(flux, a_min=0.0, a_max=None) * gain

    # Empirical background noise — see empirical_background_std_adu for why
    # the annulus level must not be used on background-subtracted frames.
    bg_std_adu = empirical_background_std_adu(image.data)

    if bg_std_adu > 0:
        bg_std_e = bg_std_adu * gain
        noise_e = np.sqrt(source_e + (bg_std_e**2) * n_pix)
        background_std = np.full_like(flux_sum, bg_std_adu)
    else:
        # Degenerate frame (constant image): fall back to the level model.
        bg_e = np.clip(background_level, a_min=0.0, a_max=None) * gain
        noise_e = np.sqrt(source_e + bg_e * n_pix + (read_noise**2) * n_pix)
        background_std = np.where(
            gain > 0,
            np.sqrt(bg_e + read_noise**2) / gain,
            0.0,
        )

    # Convert back to ADU
    flux_err = np.where(noise_e > 0, noise_e / gain, 0.0)

    # SNR
    snr = np.where(flux_err > 0, flux / flux_err, 0.0)

    # Build SimplePhotometryResult objects
    results: list[SimplePhotometryResult] = []
    for i, star in enumerate(valid_stars):
        this_flux = float(flux[i])
        this_err = float(flux_err[i])
        this_snr = float(snr[i])
        this_bg = float(background_level[i])
        this_bg_std = float(background_std[i])

        # Simple crowding placeholder (vectorized crowding is expensive; we
        # treat crowding as zero here and rely on SNR / flux for quality).
        crowding_factor = 0.0
        quality_flag = _assess_simple_quality(this_flux, this_snr, crowding_factor, config)

        instrumental_magnitude = None
        if this_flux > 0:
            instrumental_magnitude = -2.5 * np.log10(this_flux)

        results.append(
            SimplePhotometryResult(
                star=star,
                flux=this_flux,
                flux_err=this_err,
                snr=this_snr,
                background_level=this_bg,
                background_std=this_bg_std,
                aperture_radius=aperture_radius,
                crowding_factor=crowding_factor,
                quality_flag=quality_flag,
                instrumental_magnitude=instrumental_magnitude,
            )
        )

    # Calculate summary statistics
    summary = _calculate_simple_photometry_summary(results, starfield, config, frame_index=frame_index)

    logger.info(f"Measured photometry for {len(results)}/{len(catalog_stars)} catalog stars")

    # Diagnostic limiting-mag plot for the sidereal path. The rate-track path
    # gets an equivalent ``frame_<idx>_limiting_mag.png`` via WCS refinement
    # (estimate_limiting_magnitude_from_photometry); sidereal frames never go
    # through that path, so without this they'd silently skip the plot even
    # with ``plotting.photometry`` on. Same axes/lines as the rate version.
    if (
        frame_index is not None
        and get_config().plotting.photometry
        and summary.stars_mag and summary.stars_snr
    ):
        _save_simple_limiting_mag_plot(
            mags=summary.stars_mag, snrs=summary.stars_snr,
            limiting_mag=summary.limiting_magnitude_50 or summary.limiting_magnitude,
            min_snr=float(get_config().photometry.limiting_snr or 3.0),
            output_path=get_config().runtime.output_dir
                       / f"frame_{frame_index}_limiting_mag.png",
        )

    return results, summary


def _save_simple_limiting_mag_plot(
    mags: list[float],
    snrs: list[float],
    limiting_mag: float | None,
    min_snr: float,
    output_path,
) -> None:
    """Sidereal counterpart to the rate-track limiting-mag diagnostic.
    Same axes (mag vs log10 SNR) + threshold and limit reference lines."""

    import matplotlib.pyplot as plt
    import numpy as np

    mags_arr = np.asarray(mags, dtype=float)
    snrs_arr = np.asarray(snrs, dtype=float)
    valid = snrs_arr > 0
    if not np.any(valid):
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(mags_arr[valid], np.log10(snrs_arr[valid]),
               c="blue", alpha=0.4, s=14, label=f"Stars (n={int(valid.sum())})")
    ax.axhline(y=np.log10(min_snr), color="g", linestyle=":",
               label=f"SNR={min_snr} threshold")
    if limiting_mag is not None:
        ax.axvline(x=limiting_mag, color="k", linestyle="--",
                   label=f"Limiting mag = {limiting_mag:.2f}")
    ax.set_xlabel("magnitude")
    ax.set_ylabel("log10(SNR)")
    ax.set_title("Sidereal limiting magnitude estimation")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved sidereal limiting-mag diagnostic to %s", output_path)


def measure_rate_starfield_photometry(
    image: ProcessedFitsImage,
    starfield: StarField,
    streak: "StreakMetadata",
    config: SimplePhotometryConfig | None = None,
    frame_index: int | None = None,
) -> tuple[list[SimplePhotometryResult], SimplePhotometrySummary]:
    """
    Perform photometry on rate-track starfield using rectangular apertures.

    Uses rectangular apertures aligned with the streak orientation to properly
    measure flux from streaked stars.

    Parameters
    ----------
    image : ProcessedFitsImage
        The image containing the stars
    starfield : StarField
        Starfield with detected and catalog stars
    streak : StreakMetadata
        Measured streak parameters (length, angle, FWHM)
    config : SimplePhotometryConfig, optional
        Photometry configuration

    Returns
    -------
    tuple
        (photometry_results, summary_statistics)
    """
    from senpai.engine.models.metadata import StreakMetadata

    if not isinstance(streak, StreakMetadata):
        raise TypeError(f"streak must be StreakMetadata, got {type(streak)}")

    config = _normalize_photometry_config(config)

    # Get FWHM from starfield or streak
    if starfield.fwhm_stats is not None:
        fwhm = starfield.fwhm_stats.median_fwhm
    else:
        fwhm = streak.fwhm
        logger.warning("No FWHM stats in starfield, using streak FWHM")

    # Only use catalog stars
    catalog_stars = starfield.catalog_stars or []

    if not catalog_stars:
        logger.warning("No catalog stars found in starfield for photometry")
        return [], SimplePhotometrySummary(
            n_stars=0,
            n_quality=0,
            median_snr=0,
            median_background=0,
            limiting_magnitude=0,
        )

    # Calculate rectangular aperture dimensions from streak
    # Width: perpendicular to streak (FWHM-based)
    width_pixels = fwhm * config.aperture_radius_factor * 2  # Similar to circular radius
    # Length: along streak direction (streak length + margin)
    length_pixels = streak.pixel_length + fwhm * 2

    # Background annulus dimensions
    bg_width_in = width_pixels + fwhm * config.bg_inner_factor
    bg_width_out = width_pixels + fwhm * config.bg_outer_factor
    bg_length_in = length_pixels + fwhm * config.bg_inner_factor
    bg_length_out = length_pixels + fwhm * config.bg_outer_factor

    # Angle for photutils (needs to be in radians, and photutils convention differs)
    theta = streak.radian_angle() + np.pi / 2  # photutils angle convention

    height, width = image.data.shape

    # Margin to keep stars away from edges
    max_dimension = max(bg_length_out, bg_width_out)
    margin = int(max_dimension / 2 + 10)

    positions = []
    valid_stars: list[StarInSpace] = []
    for star in catalog_stars:
        if star.x is None or star.y is None:
            continue
        if star.x < margin or star.x >= width - margin or star.y < margin or star.y >= height - margin:
            continue
        positions.append((star.x, star.y))
        valid_stars.append(star)

    if not valid_stars:
        logger.warning("No catalog stars within valid bounds for photometry")
        return [], SimplePhotometrySummary(
            n_stars=0,
            n_quality=0,
            median_snr=0,
            median_background=0,
            limiting_magnitude=0,
        )

    # Sample stars by magnitude (same logic as circular version)
    MAX_STARS_PER_BIN = 500
    MAG_BIN_WIDTH = 1.0

    star_magnitudes = []
    for star in valid_stars:
        mag = None
        if hasattr(star, "magnitude") and star.magnitude is not None:
            mag = star.magnitude
        elif hasattr(star, "magnitudes") and star.magnitudes:
            mag = next(iter(star.magnitudes.values()))
        star_magnitudes.append(mag if mag is not None else 99.0)

    star_magnitudes = np.array(star_magnitudes)

    valid_mags = star_magnitudes[star_magnitudes < 99.0]
    if len(valid_mags) == 0:
        sampled_stars = valid_stars[:5000] if len(valid_stars) > 5000 else valid_stars
        logger.info(f"Sampling photometry: {len(sampled_stars)}/{len(valid_stars)} stars (no magnitude info)")
    else:
        mag_min = float(np.nanmin(valid_mags))
        mag_max = float(np.nanmax(valid_mags))
        n_bins = int(np.ceil((mag_max - mag_min) / MAG_BIN_WIDTH)) + 1
        bin_edges = np.linspace(mag_min, mag_max + MAG_BIN_WIDTH, n_bins + 1)

        sampled_indices = []
        for i in range(n_bins):
            bin_mask = (star_magnitudes >= bin_edges[i]) & (star_magnitudes < bin_edges[i + 1])
            bin_indices = np.where(bin_mask)[0]

            if len(bin_indices) > MAX_STARS_PER_BIN:
                bin_mags = star_magnitudes[bin_indices]
                sorted_idx = np.argsort(bin_mags)
                sorted_bin_indices = bin_indices[sorted_idx]
                step = len(sorted_bin_indices) / MAX_STARS_PER_BIN
                selected_idx = [sorted_bin_indices[int(j * step)] for j in range(MAX_STARS_PER_BIN)]
                sampled_indices.extend(selected_idx)
            else:
                sampled_indices.extend(bin_indices)

        sampled_indices = np.array(sampled_indices)
        sampled_stars = [valid_stars[i] for i in sampled_indices]

        logger.info(
            f"Sampling photometry: {len(sampled_stars)}/{len(valid_stars)} stars "
            f"(max {MAX_STARS_PER_BIN} per {MAG_BIN_WIDTH:.1f} mag bin)"
        )

    positions = np.array([(star.x, star.y) for star in sampled_stars])
    valid_stars = sampled_stars

    # Run photutils aperture photometry. method="subpixel" instead of the
    # default "exact": exact polygon-clipping of thousands of *rotated*
    # rectangular apertures on a 66 MP frame costs minutes; subpixel(5) is ~26x
    # faster with <0.01% flux difference on these large (~50 px) apertures.
    #
    # Chunked: photutils materializes every aperture's bounding-box mask at
    # once, and streak-length apertures have bboxes of ~(1.4*length)^2 floats
    # each — on a dense field (thousands of sampled stars) times aperture +
    # annulus masks plus their data cutouts, a single call peaked >30 GB and
    # drew the OOM killer. Slices keep identical results at bounded memory.
    bbox_side = 1.5 * max(bg_length_out, bg_width_out)
    per_star_bytes = 3 * 8 * bbox_side * bbox_side  # masks + cutout products
    chunk_size = max(64, min(len(positions), int(2e9 / max(per_star_bytes, 1))))

    flux_parts, bg_parts = [], []
    for lo in range(0, len(positions), chunk_size):
        pos_chunk = positions[lo:lo + chunk_size]
        apertures = RectangularAperture(pos_chunk, w=width_pixels, h=length_pixels, theta=theta)
        bg_apertures = RectangularAnnulus(
            pos_chunk,
            w_in=bg_width_in,
            w_out=bg_width_out,
            h_in=bg_length_in,
            h_out=bg_length_out,
            theta=theta,
        )
        phot = aperture_photometry(
            image.data, [apertures, bg_apertures], method="subpixel", subpixels=5
        )
        flux_parts.append(np.asarray(phot["aperture_sum_0"], dtype=float))
        bg_parts.append(np.asarray(phot["aperture_sum_1"], dtype=float))

    flux_sum = np.concatenate(flux_parts)
    bg_sum = np.concatenate(bg_parts)

    # Get areas (can be scalar or array)
    aper_area = apertures.area
    bg_area = bg_apertures.area
    if hasattr(aper_area, "__len__"):
        aper_area = np.asarray(aper_area, dtype=float)
    else:
        aper_area = float(aper_area)
    if hasattr(bg_area, "__len__"):
        bg_area = np.asarray(bg_area, dtype=float)
    else:
        bg_area = float(bg_area)

    # Broadcast areas if needed
    aper_area_arr = aper_area if isinstance(aper_area, np.ndarray) else np.full_like(flux_sum, aper_area, dtype=float)
    bg_area_arr = bg_area if isinstance(bg_area, np.ndarray) else np.full_like(bg_sum, bg_area, dtype=float)

    # Background level per pixel and background-subtracted flux
    background_level = np.where(bg_area_arr > 0, bg_sum / bg_area_arr, 0.0)
    flux = flux_sum - background_level * aper_area_arr

    # Noise model (same as circular version: empirical background std).
    cfg = get_config()
    phot_cfg = getattr(cfg, "photometry", None)
    if phot_cfg is not None:
        gain = float(getattr(phot_cfg, "gain", 1.0))
        include_read_noise = bool(getattr(phot_cfg, "include_read_noise", False))
        read_noise = float(getattr(phot_cfg, "read_noise", 0.0)) if include_read_noise else 0.0
    else:
        gain = 1.0
        read_noise = 0.0

    n_pix = aper_area_arr

    # Convert to electrons
    source_e = np.clip(flux, a_min=0.0, a_max=None) * gain

    # Empirical background noise — see empirical_background_std_adu for why
    # the annulus level must not be used on background-subtracted frames.
    bg_std_adu = empirical_background_std_adu(image.data)

    if bg_std_adu > 0:
        bg_std_e = bg_std_adu * gain
        noise_e = np.sqrt(source_e + (bg_std_e**2) * n_pix)
        background_std = np.full_like(flux_sum, bg_std_adu)
    else:
        # Degenerate frame (constant image): fall back to the level model.
        bg_e = np.clip(background_level, a_min=0.0, a_max=None) * gain
        noise_e = np.sqrt(source_e + bg_e * n_pix + (read_noise**2) * n_pix)
        background_std = np.where(
            gain > 0,
            np.sqrt(bg_e + read_noise**2) / gain,
            0.0,
        )

    # Convert back to ADU
    flux_err = np.where(noise_e > 0, noise_e / gain, 0.0)

    # SNR
    snr = np.where(flux_err > 0, flux / flux_err, 0.0)

    # Build results
    results: list[SimplePhotometryResult] = []
    for i, star in enumerate(valid_stars):
        this_flux = float(flux[i])
        this_err = float(flux_err[i])
        this_snr = float(snr[i])
        this_bg = float(background_level[i])
        this_bg_std = float(background_std[i])

        crowding_factor = 0.0  # Placeholder (same as circular version)
        quality_flag = _assess_simple_quality(this_flux, this_snr, crowding_factor, config)

        instrumental_magnitude = None
        if this_flux > 0:
            instrumental_magnitude = -2.5 * np.log10(this_flux)

        # Use average of width/length as "aperture_radius" for compatibility
        aperture_size = (width_pixels + length_pixels) / 2

        results.append(
            SimplePhotometryResult(
                star=star,
                flux=this_flux,
                flux_err=this_err,
                snr=this_snr,
                background_level=this_bg,
                background_std=this_bg_std,
                aperture_radius=aperture_size,  # Average dimension for reporting
                crowding_factor=crowding_factor,
                quality_flag=quality_flag,
                instrumental_magnitude=instrumental_magnitude,
            )
        )

    # Calculate summary statistics (same function as circular version)
    summary = _calculate_simple_photometry_summary(results, starfield, config, frame_index=frame_index)

    logger.info(
        f"Measured rate-track photometry for {len(results)}/{len(catalog_stars)} catalog stars "
        f"(rectangular apertures: {width_pixels:.1f}x{length_pixels:.1f}px, θ={np.rad2deg(theta):.1f}°)"
    )
    return results, summary


def measure_detection_photometry(
    image: ProcessedFitsImage,
    detections: SatelliteListImage,
    zero_point: float,
    zero_point_err: float | None = None,
    exposure_time: float | None = None,
    config: SimplePhotometryConfig | None = None,
    multiband_calibration: "MultiBandCalibration | None" = None,
    observation_filter: str | None = None,
) -> None:
    """
    Perform aperture photometry on satellite/object detections and assign calibrated magnitudes.

    Uses the zero point derived from streaked catalog star photometry to calibrate
    instrumental magnitudes into the catalog system. Updates detection fields in-place.

    When multiband_calibration is provided, computes per-band calibrated magnitudes
    using each band's ZP (no color term applied for unknown objects).

    Parameters
    ----------
    image : ProcessedFitsImage
        The image containing the detections
    detections : SatelliteListImage
        Detected point sources (satellites/asteroids) to measure
    zero_point : float
        Photometric zero point (ZP = m_cat + 2.5 * log10(flux_cat / t_exp))
    zero_point_err : float, optional
        Uncertainty in the zero point
    exposure_time : float, optional
        Exposure time in seconds (defaults to 1.0 if not provided)
    config : SimplePhotometryConfig, optional
        Photometry configuration
    multiband_calibration : MultiBandCalibration, optional
        Multi-band calibration with per-band ZPs and color terms
    observation_filter : str, optional
        Observation filter name (e.g. "Clear", "V")
    """
    config = _normalize_photometry_config(config)

    if not detections or not detections.detections:
        return

    if exposure_time is None or exposure_time <= 0:
        exposure_time = 1.0

    height, width = image.data.shape

    # Gather valid detections with positions and FWHM
    valid_indices = []
    positions = []
    fwhms = []
    for i, det in enumerate(detections.detections):
        if det.x is None or det.y is None:
            continue
        fwhm = det.pixel_fwhm if det.pixel_fwhm is not None and det.pixel_fwhm > 0 else 4.0
        aperture_radius = config.aperture_radius_factor * fwhm
        bg_outer = config.bg_outer_factor * fwhm
        margin = int(bg_outer + 10)
        if det.x < margin or det.x >= width - margin or det.y < margin or det.y >= height - margin:
            continue
        valid_indices.append(i)
        positions.append((det.x, det.y))
        fwhms.append(fwhm)

    if not valid_indices:
        logger.info("No detections within valid bounds for photometry")
        return

    # Get gain and read noise from config
    cfg = get_config()
    phot_cfg = getattr(cfg, "photometry", None)
    if phot_cfg is not None:
        gain = float(getattr(phot_cfg, "gain", 1.0))
        include_read_noise = bool(getattr(phot_cfg, "include_read_noise", False))
        read_noise = float(getattr(phot_cfg, "read_noise", 0.0)) if include_read_noise else 0.0
    else:
        gain = 1.0
        read_noise = 0.0

    # Empirical background noise, once per frame — see
    # empirical_background_std_adu for why the annulus level must not be used
    # on background-subtracted frames.
    bg_std_adu = empirical_background_std_adu(image.data)
    bg_std_e = bg_std_adu * gain

    # Process each detection individually (different FWHM -> different aperture sizes)
    n_measured = 0
    for idx, pos, fwhm in zip(valid_indices, positions, fwhms):
        det = detections.detections[idx]
        try:
            aperture_radius = config.aperture_radius_factor * fwhm
            bg_inner = config.bg_inner_factor * fwhm
            bg_outer = config.bg_outer_factor * fwhm

            aperture = CircularAperture(pos, r=aperture_radius)
            bg_aperture = CircularAnnulus(pos, r_in=bg_inner, r_out=bg_outer)

            phot = aperture_photometry(image.data, [aperture, bg_aperture])

            flux_sum = float(phot["aperture_sum_0"][0])
            bg_sum = float(phot["aperture_sum_1"][0])
            bg_area = bg_aperture.area
            aper_area = aperture.area

            # Background level per pixel and background-subtracted flux
            background_level = bg_sum / bg_area if bg_area > 0 else 0.0
            flux = flux_sum - background_level * aper_area

            if flux <= 0:
                continue

            # Noise model: Poisson source + empirical background noise
            n_pix = aper_area
            source_e = max(0.0, flux) * gain
            if bg_std_e > 0:
                noise_e = np.sqrt(source_e + (bg_std_e**2) * n_pix)
            else:
                bg_e = max(0.0, background_level) * gain
                noise_e = np.sqrt(source_e + bg_e * n_pix + (read_noise**2) * n_pix)
            flux_err = noise_e / gain if noise_e > 0 else 0.0

            # SNR
            snr = flux / flux_err if flux_err > 0 else 0.0

            # Instrumental magnitude
            flux_per_sec = flux / exposure_time
            instrumental_mag = -2.5 * np.log10(flux_per_sec)

            # Flux-based magnitude error component: 2.5/ln(10) * flux_err/flux
            mag_err_flux = 1.0857 * flux_err / flux

            # Update detection fields in-place
            det.flux = float(flux)
            det.flux_err = float(flux_err)
            det.snr = float(snr)
            det.instrumental_magnitude = float(instrumental_mag)
            det.observation_filter = observation_filter

            # Per-band calibrated magnitudes
            if multiband_calibration is not None and multiband_calibration.bands:
                cal_mags = {}
                cal_errs = {}
                for band_name, band_cal in multiband_calibration.bands.items():
                    cal_mags[band_name] = float(band_cal.zero_point + instrumental_mag)
                    band_zp_err = band_cal.zero_point_err if band_cal.zero_point_err else 0.0
                    cal_errs[band_name] = float(np.sqrt(mag_err_flux**2 + band_zp_err**2))
                det.calibrated_magnitudes = cal_mags
                det.magnitude_errs = cal_errs
            else:
                # Fallback: single ZP calibration
                calibrated_mag = zero_point + instrumental_mag
                if zero_point_err is not None:
                    mag_err = float(np.sqrt(mag_err_flux**2 + zero_point_err**2))
                else:
                    mag_err = float(mag_err_flux)
                band_key = observation_filter if observation_filter else "instrumental"
                det.calibrated_magnitudes = {band_key: float(calibrated_mag)}
                det.magnitude_errs = {band_key: float(mag_err)}
            n_measured += 1

        except Exception as e:
            logger.debug(f"Detection photometry failed for detection at ({det.x:.1f}, {det.y:.1f}): {e}")
            continue

    logger.info(f"Measured detection photometry for {n_measured}/{len(detections.detections)} detections")


def _calculate_simple_crowding(
    data: np.ndarray,
    x: float,
    y: float,
    radius: float,
) -> float:
    """Calculate simple crowding factor."""

    # Check a region 3x the aperture radius
    check_radius = radius * 3

    # Extract region around the star
    y_min = max(0, int(y - check_radius))
    y_max = min(data.shape[0], int(y + check_radius))
    x_min = max(0, int(x - check_radius))
    x_max = min(data.shape[1], int(x + check_radius))

    region = data[y_min:y_max, x_min:x_max]

    # Simple crowding: count bright pixels outside the aperture
    center_y, center_x = int(y - y_min), int(x - x_min)
    bright_threshold = np.median(region) + 2 * np.std(region)

    crowding_count = 0
    for i in range(region.shape[0]):
        for j in range(region.shape[1]):
            dist = np.sqrt((i - center_y) ** 2 + (j - center_x) ** 2)
            if dist > radius and dist < check_radius and region[i, j] > bright_threshold:
                crowding_count += 1

    # Normalize
    crowding_area = np.pi * (check_radius**2 - radius**2)
    crowding_factor = min(crowding_count / max(crowding_area, 1), 1.0)

    return crowding_factor


def _assess_simple_quality(
    flux: float,
    snr: float,
    crowding_factor: float,
    config: SimplePhotometryConfig,
) -> bool:
    """Assess simple quality of photometric measurement."""

    # Reject negative flux
    if flux <= 0:
        return False

    # Reject negative or zero SNR
    if snr <= 0:
        return False

    # Enforce minimum SNR threshold from configuration
    if snr < config.min_snr:
        return False
    # Reject too crowded
    if crowding_factor > config.max_crowding:
        return False

    return True


def _has_bright_neighbor(
    star: StarInSpace,
    mag: float,
    catalog_stars: list[StarInSpace],
    iso_radius_pix: float,
    delta_mag: float,
    mag_cache: dict[int, float | None] | None = None,
    kdtree: tuple[cKDTree, list[int]] | None = None,
) -> bool:
    """
    Check if a star has a significantly brighter neighbor within a given radius.

    This is used to avoid using severely blended stars (e.g., a faint catalog
    star nearly coincident with a much brighter star) for zero-point and
    limiting-magnitude calibration, since their measured SNR will be dominated
    by the brighter neighbor.

    Optimized version using KD-tree for spatial queries and pre-computed magnitudes.
    """
    if star.x is None or star.y is None:
        return False

    if not catalog_stars:
        return False

    # Use KD-tree if provided for fast spatial queries
    if kdtree is not None:
        tree, position_to_star = kdtree
        # Find all neighbors within iso_radius_pix
        star_pos = np.array([star.x, star.y])
        neighbor_kdtree_indices = tree.query_ball_point(star_pos, iso_radius_pix)

        # Check if any neighbor is significantly brighter
        for kdtree_idx in neighbor_kdtree_indices:
            catalog_idx = position_to_star[kdtree_idx]
            other = catalog_stars[catalog_idx]
            if other is star:
                continue

            # Get magnitude from cache if available, otherwise compute
            if mag_cache is not None:
                other_mag = mag_cache.get(id(other))
            else:
                other_mag = _get_best_magnitude(other)

            if other_mag is None:
                continue

            # Only consider neighbors that are significantly brighter
            if other_mag + delta_mag < mag:
                return True

        return False

    # Fallback to original O(n) scan if no KD-tree provided
    iso_radius_sq = iso_radius_pix * iso_radius_pix

    for other in catalog_stars:
        # Skip self
        if other is star:
            continue

        if other.x is None or other.y is None:
            continue

        # Get magnitude from cache if available, otherwise compute
        if mag_cache is not None:
            other_mag = mag_cache.get(id(other))
        else:
            other_mag = _get_best_magnitude(other)

        if other_mag is None:
            continue

        # Only consider neighbors that are significantly brighter
        if other_mag + delta_mag >= mag:
            continue

        dx = star.x - other.x
        dy = star.y - other.y
        if dx * dx + dy * dy <= iso_radius_sq:
            return True

    return False


def _get_best_magnitude(star: StarInSpace, preferred_filters: list[str] = None) -> float | None:
    """Get the best available magnitude for a star."""
    # Use a simple cache key based on star's magnitude attributes
    # Since stars are objects, we can't use lru_cache directly, but we can
    # cache based on the magnitude values themselves

    if preferred_filters is None:
        preferred_filters = [
            "Johnson_V",
            "Johnson_R",
            "Sloan_r",
            "Gaia_G",
            "Sloan_g",
            "Johnson_B",
        ]

    # Try preferred filters from magnitudes dict first
    if hasattr(star, "magnitudes") and star.magnitudes:
        for filter_name in preferred_filters:
            if filter_name in star.magnitudes:
                return star.magnitudes[filter_name]

        # Fallback to first available magnitude from dict
        if star.magnitudes:
            return next(iter(star.magnitudes.values()))

    # Fallback to primary magnitude if no magnitudes dict
    if hasattr(star, "magnitude") and star.magnitude is not None:
        return star.magnitude

    return None


def _find_common_magnitude_system(stars: list[StarInSpace], preferred_filters: list[str] = None) -> str | None:
    """
    Find the best magnitude system with the most star coverage.

    Tries each preferred filter in order and picks the first one that covers
    at least 50% of stars.  Among filters that meet the threshold, the
    preference order breaks ties.  This avoids mixing magnitude systems while
    remaining robust when not every star has every band.

    Returns the filter name if found, None otherwise.
    """
    if preferred_filters is None:
        preferred_filters = [
            "Johnson_V",
            "Johnson_R",
            "Sloan_r",
            "Gaia_G",
            "Sloan_g",
            "Johnson_B",
        ]

    n_stars = len(stars)
    if n_stars == 0:
        return None

    # Count how many stars have each preferred filter
    best_filter = None
    best_coverage = 0
    for filter_name in preferred_filters:
        count = 0
        for star in stars:
            has_dict = (
                hasattr(star, "magnitudes") and star.magnitudes is not None and len(star.magnitudes) > 0
            )
            if has_dict and filter_name in star.magnitudes:
                count += 1
        coverage = count / n_stars

        # Accept the first preferred filter with >=50% coverage
        if coverage >= 0.5 and best_filter is None:
            best_filter = filter_name
            best_coverage = coverage
            # If we have 100% coverage, no need to keep looking
            if coverage == 1.0:
                break

    if best_filter is not None:
        if best_coverage < 1.0:
            logger.info(
                "Using magnitude system '%s' (%.0f%% coverage, %d/%d stars)",
                best_filter,
                best_coverage * 100,
                int(best_coverage * n_stars),
                n_stars,
            )
        return best_filter

    # Fallback: if stars have primary magnitude but no magnitudes dict,
    # we can use that (assuming it's consistent)
    if all(hasattr(star, "magnitude") and star.magnitude is not None for star in stars):
        has_any_magnitudes_dict = any(
            hasattr(star, "magnitudes") and star.magnitudes is not None and len(star.magnitudes) > 0 for star in stars
        )
        if not has_any_magnitudes_dict:
            return "primary"

    return None


def _precompute_star_magnitudes(
    stars: list[StarInSpace], preferred_filters: list[str] = None
) -> dict[int, float | None]:
    """Pre-compute magnitudes for all stars using a consistent magnitude system.

    Returns a dict mapping id(star) to magnitude, since StarInSpace objects
    are not hashable.

    This ensures all stars use the same magnitude system to avoid mixing
    different photometric systems.
    """
    if preferred_filters is None:
        preferred_filters = [
            "Johnson_V",
            "Johnson_R",
            "Sloan_r",
            "Gaia_G",
            "Sloan_g",
            "Johnson_B",
        ]

    # Find a common magnitude system for all stars
    common_filter = _find_common_magnitude_system(stars, preferred_filters)

    mag_cache = {}
    if common_filter == "primary":
        # Use primary magnitude for all stars
        for star in stars:
            mag_cache[id(star)] = star.magnitude if hasattr(star, "magnitude") else None
    elif common_filter is not None:
        # Use the common filter for all stars
        for star in stars:
            if hasattr(star, "magnitudes") and star.magnitudes is not None and len(star.magnitudes) > 0:
                mag_cache[id(star)] = star.magnitudes.get(common_filter)
            else:
                mag_cache[id(star)] = None
    else:
        # Fallback: use _get_best_magnitude (but log a warning)
        logger.warning(
            "Could not find a common magnitude system for all stars. "
            "Falling back to per-star best magnitude selection. "
            "This may mix magnitude systems!"
        )
        for star in stars:
            mag_cache[id(star)] = _get_best_magnitude(star, preferred_filters)

    return mag_cache


def _isotonic_completeness(comp_mag, comp_pct):
    """Monotonic-decreasing (isotonic) smoothing of a completeness curve.

    Completeness is physically non-increasing with magnitude, so isotonic
    regression de-spikes the (often noisy) binned curve without imposing a
    parametric shape — unlike a logistic, whose forced 0/1 asymptotes bias the
    50% point when a contaminated faint tail flattens above 0. Returns
    (x_sorted, y_isotonic)."""
    x = np.asarray(comp_mag, dtype=float)
    y = np.asarray(comp_pct, dtype=float) / 100.0
    order = np.argsort(x)
    x, y = x[order], y[order]
    try:
        from sklearn.isotonic import IsotonicRegression

        ys = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0).fit_transform(x, y)
    except Exception:
        ys = np.minimum.accumulate(y)  # fallback: monotone via running min bright->faint
    return x, np.asarray(ys, dtype=float)


def _completeness_limits(
    comp_mag: list[float],
    comp_pct: list[float],
    target: float = 0.5,
) -> tuple[float | None, float | None, float | None]:
    """Limiting magnitudes from a completeness curve via isotonic smoothing +
    threshold crossing (no parametric fit).

    The curve is de-spiked with a monotonic-decreasing regression, then we read
    the faint-most magnitude where it crosses each level. This ignores a flat
    faint tail (the crossing happens on the roll-off, before any floor) and is
    robust to spiky bins. Returns ``(m_target, m50, m90)`` or Nones if the curve
    never crosses the level.
    """
    if not comp_mag or len(comp_mag) < 3:
        return None, None, None

    x, ys = _isotonic_completeness(comp_mag, comp_pct)

    def _cross(level: float) -> float | None:
        # Faint-most bright->faint crossing of `level`, linearly interpolated.
        idx = np.where((ys[:-1] >= level) & (ys[1:] < level))[0]
        if len(idx) == 0:
            return None
        i = int(idx[-1])
        if ys[i + 1] == ys[i]:
            return float(x[i])
        return float(x[i] + (level - ys[i]) * (x[i + 1] - x[i]) / (ys[i + 1] - ys[i]))

    return _cross(target), _cross(0.5), _cross(0.9)


def _isolated_result_mask(
    results: list[SimplePhotometryResult],
    starfield: StarField,
    pad: float = 2.0,
) -> list[bool]:
    """True for results with no *brighter* catalog star within the aperture
    footprint.

    A faint star whose aperture overlaps a brighter neighbor picks up that
    neighbor's flux and reports a spuriously high SNR — which inflates the
    faint-end completeness into a fake floor (and reads as "recovering" stars
    well past the real limit). Restricting the completeness sample to isolated
    stars removes that, so the curve actually rolls to ~0. The footprint is
    ``pad x aperture_radius`` (apertures of the two stars overlap within ~2x a
    radius). Comparison uses each star's primary magnitude for consistency.
    """
    catalog = getattr(starfield, "catalog_stars", None) or []
    cx, cy, cm = [], [], []
    for s in catalog:
        if s.x is None or s.y is None or s.magnitude is None:
            continue
        cx.append(s.x); cy.append(s.y); cm.append(s.magnitude)
    n = len(results)
    if len(cx) < 2:
        return [True] * n

    from scipy.spatial import cKDTree

    cx = np.asarray(cx); cy = np.asarray(cy); cm = np.asarray(cm)
    tree = cKDTree(np.column_stack([cx, cy]))

    keep: list[bool] = []
    for r in results:
        s = r.star
        smag = getattr(s, "magnitude", None)
        if s.x is None or s.y is None or smag is None:
            keep.append(False)
            continue
        radius = pad * float(getattr(r, "aperture_radius", 0.0) or 8.0)
        neigh = tree.query_ball_point((s.x, s.y), radius)
        # Excluded if any neighbor is meaningfully brighter (lower mag).
        keep.append(not any(cm[j] < smag - 0.1 for j in neigh))
    return keep


def _save_completeness_plot(comp_mag, comp_pct, m_target, m50, m90, output_path) -> None:
    """Plot the completeness curve, its isotonic smooth, and the limiting-mag
    crossings — the diagnostic for what the limiting-mag readout is doing."""
    import matplotlib.pyplot as plt

    try:
        x = np.asarray(comp_mag, dtype=float)
        y = np.asarray(comp_pct, dtype=float) / 100.0
        xs, ys = _isotonic_completeness(comp_mag, comp_pct)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, y, "o-", color="orange", alpha=0.7, label="completeness (isolated)")
        ax.plot(xs, ys, "r-", label="isotonic smooth")
        ax.axhline(0.5, ls=":", color="k", alpha=0.5)
        ax.axhline(0.9, ls=":", color="gray", alpha=0.5)
        if m50 is not None:
            ax.axvline(m50, ls="--", color="k", label=f"50% = {m50:.2f}")
        if m90 is not None:
            ax.axvline(m90, ls="--", color="gray", label=f"90% = {m90:.2f}")
        ax.set_xlabel("magnitude")
        ax.set_ylabel("completeness")
        ax.set_ylim(0, 1.05)
        ax.set_title("Completeness curve + limiting magnitude")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info("Saved completeness diagnostic to %s", output_path)
    except Exception as e:
        logger.warning("Completeness plot failed: %s", e)


def compute_completeness_curve(
    results: list[SimplePhotometryResult],
    starfield: StarField,
    config: SimplePhotometryConfig | None = None,
    bin_width: float = 0.5,
    min_stars_per_bin: int = 3,
    isolate: bool = False,
) -> tuple[list[float], list[float]]:
    """Compute per-frame magnitude vs detection completeness curve.

    For each magnitude bin, computes the fraction of catalog stars that were
    detected above the configured SNR threshold. This gives a per-frame
    "probability of detecting an object at magnitude X" curve that accounts
    for the actual seeing, sky background, and noise conditions of that frame.

    Args:
        results: Photometry results from measure_*_photometry.
        starfield: StarField with catalog stars for consistent magnitude system.
        config: Photometry config (uses limiting_snr for threshold).
        bin_width: Magnitude bin width (default 0.5 mag).
        min_stars_per_bin: Minimum stars in a bin for meaningful statistics.

    Returns:
        (completeness_mag, completeness_pct) — parallel arrays sorted bright
        to faint. completeness_pct[i] is the detection percentage at magnitude
        completeness_mag[i].
    """
    config = _normalize_photometry_config(config)
    snr_threshold = float(config.limiting_snr)

    # Get consistent magnitudes for all measured stars. NOTE:
    # _precompute_star_magnitudes takes (stars, preferred_filters) and keys the
    # cache by id(star); it must be given the result stars (not the results
    # list) and a filter list (not the starfield), or every lookup below misses
    # and the completeness curve comes back empty.
    stars = [r.star for r in results]
    preferred = getattr(config, "preferred_filters", None)
    mag_cache = _precompute_star_magnitudes(stars, preferred)

    # Optional isolation cut: drop measured stars with a brighter neighbor in
    # the aperture footprint (their SNR is contaminated). NOTE: applied to the
    # *measured* subset in a dense field this removes most stars and empties the
    # faint bins, so the curve no longer rolls over — it would need a
    # full-catalog denominator (which we don't have; faint stars are skipped for
    # speed). Off by default; the isotonic-crossing readout already handles the
    # faint contamination floor by crossing before it.
    if isolate:
        isolated = _isolated_result_mask(results, starfield)
        logger.info(
            "Completeness sample: %d/%d isolated stars (brighter-neighbor cut)",
            int(np.sum(isolated)), len(results),
        )
    else:
        isolated = [True] * len(results)

    mags_list: list[float] = []
    snrs_list: list[float] = []
    for r, keep in zip(results, isolated):
        if not keep:
            continue
        mag = mag_cache.get(id(r.star))
        if mag is None or r.snr is None or np.isnan(mag):
            continue
        mags_list.append(mag)
        snrs_list.append(max(0.0, r.snr))

    if len(mags_list) < min_stars_per_bin:
        return [], []

    mags_arr = np.array(mags_list)
    snrs_arr = np.array(snrs_list)

    min_mag = float(np.floor(np.min(mags_arr)))
    max_mag = float(np.ceil(np.max(mags_arr)))
    bins = np.arange(min_mag, max_mag + bin_width, bin_width)

    completeness_mag: list[float] = []
    completeness_pct: list[float] = []

    for i in range(len(bins) - 1):
        m_lo = bins[i]
        m_hi = bins[i + 1]
        in_bin = (mags_arr >= m_lo) & (mags_arr < m_hi)
        n_bin = int(np.sum(in_bin))

        if n_bin < min_stars_per_bin:
            continue

        frac = float(np.mean(snrs_arr[in_bin] >= snr_threshold))
        completeness_mag.append(round((m_lo + m_hi) / 2.0, 2))
        completeness_pct.append(round(frac * 100.0, 1))

    return completeness_mag, completeness_pct


def _calculate_simple_photometry_summary(
    results: list[SimplePhotometryResult],
    starfield: StarField,
    config: SimplePhotometryConfig | None = None,
    frame_index: int | None = None,
) -> SimplePhotometrySummary:
    """Calculate summary statistics from simple photometry results."""

    config = _normalize_photometry_config(config)

    if not results:
        limiting_snr = float(config.limiting_snr) if config else None
        return SimplePhotometrySummary(
            n_stars=0,
            n_quality=0,
            median_snr=0,
            median_background=0,
            limiting_magnitude=0,
            limiting_snr=limiting_snr,
        )

    n_stars = len(results)
    n_quality = sum(1 for r in results if r.quality_flag)

    # SNR statistics
    snrs = [r.snr for r in results if r.snr > 0]
    median_snr = float(np.median(snrs)) if snrs else 0

    # Background statistics
    backgrounds = [r.background_level for r in results]
    median_background = float(np.median(backgrounds)) if backgrounds else 0

    # Calculate zero point if catalog stars are available
    zero_point, zero_point_err = _calculate_simple_zero_point(results, starfield, config)

    # Estimate limiting magnitude using zero point and sky background when possible
    limiting_magnitude, limiting_magnitude_50, limiting_magnitude_90 = _estimate_simple_limiting_magnitude(
        results, starfield, zero_point=zero_point, config=config
    )

    # Multi-band calibration with color terms
    multiband_calibration = None
    if config.enable_color_terms and config.target_bands:
        from senpai.engine.photometry.color_terms import calculate_multiband_calibration

        multiband_calibration = calculate_multiband_calibration(results, starfield, config.target_bands, config)

    # Get limiting SNR from config
    limiting_snr = float(config.limiting_snr) if config else None

    # Compute per-frame completeness curve and read the limiting magnitudes off
    # an isotonic smooth + threshold crossing — robust to spiky bins and to a
    # residual faint tail (no parametric shape imposed). The crossing happens on
    # the roll-off, before any contamination floor.
    comp_mag, comp_pct = compute_completeness_curve(
        results, starfield, config, isolate=config.completeness_isolate
    )
    completeness_target = float(config.limiting_completeness_fraction)
    lim_target, lim_50, lim_90 = _completeness_limits(
        comp_mag, comp_pct, target=completeness_target
    )
    if lim_50 is not None:
        limiting_magnitude = round(lim_target, 3)
        limiting_magnitude_50 = round(lim_50, 3)
        limiting_magnitude_90 = round(lim_90, 3) if lim_90 is not None else None
        logger.info(
            "Limiting mag from completeness curve (isotonic crossing): "
            "target(%.2f)=%.2f, 50%%=%.2f, 90%%=%s",
            completeness_target, lim_target, lim_50,
            f"{lim_90:.2f}" if lim_90 is not None else "n/a",
        )
    else:
        logger.info(
            "Completeness curve never crosses target; keeping scan/SNR-fit "
            "limiting mag (50%%=%s)", limiting_magnitude_50,
        )

    # Always emit the completeness diagnostic when we have a curve — so the
    # readout is visible even when it falls back. Uses the GLOBAL config for the
    # plotting flag (the photometry `config` here has no `.plotting`).
    if comp_mag and frame_index is not None and get_config().plotting.photometry:
        _save_completeness_plot(
            comp_mag, comp_pct,
            lim_target, lim_50 or limiting_magnitude_50, lim_90 or limiting_magnitude_90,
            get_config().runtime.output_dir / f"frame_{frame_index}_completeness.png",
        )

    # Compact per-star (catalog_mag, snr, zp_offset) arrays for downstream
    # observability. Only stars with both a known catalog magnitude AND a
    # positive SNR are retained; that filter keeps the arrays useful for
    # binned aggregates. zp_offset = m_cat − m_inst (per-star zero point) is
    # None where the instrumental magnitude couldn't be measured.
    stars_mag: list[float] = []
    stars_snr: list[float] = []
    stars_zp_offset: list[float | None] = []
    stars_isolated: list[bool] = []
    iso_mask = _isolated_result_mask(results, starfield)
    for r, iso in zip(results, iso_mask):
        mag = getattr(r.star, "magnitude", None)
        if mag is None or r.snr is None or r.snr <= 0:
            continue
        stars_mag.append(round(float(mag), 3))
        stars_snr.append(round(float(r.snr), 3))
        stars_zp_offset.append(
            round(float(mag) - r.instrumental_magnitude, 3)
            if r.instrumental_magnitude is not None else None
        )
        stars_isolated.append(bool(iso))

    return SimplePhotometrySummary(
        n_stars=n_stars,
        n_quality=n_quality,
        median_snr=median_snr,
        median_background=median_background,
        limiting_magnitude=limiting_magnitude,
        zero_point=zero_point,
        zero_point_err=zero_point_err,
        limiting_snr=limiting_snr,
        limiting_magnitude_50=limiting_magnitude_50,
        limiting_magnitude_90=limiting_magnitude_90,
        multiband_calibration=multiband_calibration,
        completeness_mag=comp_mag if comp_mag else None,
        completeness_pct=comp_pct if comp_pct else None,
        stars_mag=stars_mag or None,
        stars_snr=stars_snr or None,
        stars_zp_offset=stars_zp_offset or None,
        stars_isolated=stars_isolated or None,
    )


def _estimate_simple_limiting_magnitude(
    results: list[SimplePhotometryResult],
    starfield: StarField,
    zero_point: float | None = None,
    config: SimplePhotometryConfig | None = None,
) -> tuple[float, float | None, float | None]:
    """
    Estimate limiting magnitude at multiple completeness levels.

    Returns:
        Tuple of (limiting_magnitude, limiting_magnitude_50, limiting_magnitude_90)
        where limiting_magnitude is at the configured completeness target.
    """
    """
    Estimate limiting magnitude from simple photometry results.

    Primary method (empirical / recommended):
        Use the actual measured SNR of catalog stars in this image
        to find the magnitude range where SNR ≈ 5, and take a robust
        faint-end estimate of that distribution.

    Fallback:
        If there are not enough stars with SNR≈5, fall back to a
        sky-noise + zero-point based estimate, and finally to the
        original empirical log10(SNR) vs magnitude fit.
    """

    if not results:
        return 0.0

    config = _normalize_photometry_config(config)

    # Pre-compute isolation radius for neighbor checks (in pixels)
    iso_radius_pix: float | None = None
    catalog_stars = starfield.catalog_stars or []
    mag_cache: dict[int, float | None] | None = None
    kdtree: tuple[cKDTree, list[int]] | None = None

    if starfield.fwhm_stats is not None:
        fwhm = starfield.fwhm_stats.median_fwhm
        aperture_radius = config.aperture_radius_factor * fwhm
        iso_radius_pix = config.isolation_radius_factor * aperture_radius

        # Pre-compute magnitudes for all catalog stars and build KD-tree for fast neighbor queries
        if catalog_stars and iso_radius_pix is not None:
            mag_cache = _precompute_star_magnitudes(catalog_stars, config.preferred_filters if config else None)

            # Build KD-tree for spatial queries (only for stars with valid positions)
            positions = []
            position_to_star = []  # Maps KD-tree index to catalog_stars index
            for i, star in enumerate(catalog_stars):
                if star.x is not None and star.y is not None:
                    positions.append([star.x, star.y])
                    position_to_star.append(i)

            if positions:
                kdtree = (
                    cKDTree(positions),
                    position_to_star,
                )  # Store mapping with tree
            else:
                kdtree = None

    # ------------------------------------------------------------------
    # 1) Empirical limit from measured SNR vs magnitude
    #    (hybrid completeness + SNR-fit approach)
    # ------------------------------------------------------------------
    empirical_points: list[tuple[float, float]] = []
    mags_all_list: list[float] = []
    snrs_all_list: list[float] = []

    # Pre-compute magnitudes for result stars using a consistent magnitude system
    # Find common magnitude system first to ensure all stars use the same system
    result_stars = [r.star for r in results]
    common_filter = _find_common_magnitude_system(result_stars, config.preferred_filters if config else None)

    if common_filter is not None:
        logger.info(
            f"Using consistent magnitude system: {common_filter} (ensures all stars use same photometric system)"
        )
    else:
        logger.warning("Could not find common magnitude system, may mix different systems!")

    result_mag_cache = {}
    if common_filter == "primary":
        # Use primary magnitude for all stars
        for r in results:
            star_id = id(r.star)
            if star_id not in result_mag_cache:
                result_mag_cache[star_id] = r.star.magnitude if hasattr(r.star, "magnitude") else None
    elif common_filter is not None:
        # Use the common filter for all stars
        for r in results:
            star_id = id(r.star)
            if star_id not in result_mag_cache:
                if hasattr(r.star, "magnitudes") and r.star.magnitudes is not None and len(r.star.magnitudes) > 0:
                    result_mag_cache[star_id] = r.star.magnitudes.get(common_filter)
                else:
                    result_mag_cache[star_id] = None
    else:
        # Fallback: use per-star best magnitude (not ideal)
        for r in results:
            star_id = id(r.star)
            if star_id not in result_mag_cache:
                result_mag_cache[star_id] = _get_best_magnitude(r.star, config.preferred_filters if config else None)

    for r in results:
        mag = result_mag_cache[id(r.star)]
        if mag is None or r.snr is None:
            continue

        # All stars (quality + poor) contribute to completeness statistics
        # Floor negative SNR at 0, but still include these stars
        snr_floor = max(0.0, r.snr) if r.snr is not None else 0.0
        mags_all_list.append(mag)
        snrs_all_list.append(snr_floor)

        # Only quality, isolated stars contribute to the SNR–mag fit
        # Use original SNR (not floored) for the fit
        if not r.quality_flag:
            continue

        if iso_radius_pix is not None and catalog_stars:
            if _has_bright_neighbor(
                r.star,
                mag,
                catalog_stars,
                iso_radius_pix,
                config.isolation_delta_mag,
                mag_cache=mag_cache,
                kdtree=kdtree,
            ):
                continue

        # Use original SNR for fit (not floored)
        empirical_points.append((mag, r.snr))

    # Log magnitude range for completeness calculation
    if mags_all_list:
        logger.debug(
            f"Completeness calculation: {len(mags_all_list)} stars with valid mag+SNR "
            f"(range {min(mags_all_list):.2f}-{max(mags_all_list):.2f} mag)"
        )

    mags_all = np.array(mags_all_list)
    snrs_all = np.array(snrs_all_list)

    if len(empirical_points) >= 10:
        mags = np.array([m for m, _ in empirical_points])
        snrs = np.array([s for _, s in empirical_points])

        snr_threshold = float(config.limiting_snr)
        completeness_target = float(config.limiting_completeness_fraction)

        # --- 1a) Completeness-based limits at multiple levels ---
        # Calculate limiting magnitudes at different completeness levels (50%, 90%)
        completeness_limit: float | None = None
        completeness_limit_50: float | None = None
        completeness_limit_90: float | None = None

        def find_completeness_limit(target_completeness: float) -> float | None:
            """Helper function to find limiting magnitude at a given completeness level."""
            try:
                # Use same bin width as plotting code for consistency (0.25 mag)
                bin_width = 0.25
                min_mag = float(np.floor(np.min(mags_all)))
                max_mag_actual = float(np.max(mags_all))  # Actual max, not rounded up
                max_mag = float(np.ceil(max_mag_actual))  # For binning, round up to include last bin
                bins = np.arange(min_mag, max_mag + bin_width, bin_width)

                # Build list of valid bins (with at least 1 star) and their completeness
                # Match plotting behavior: only skip empty bins, not sparse ones
                # This ensures consistency between plot and calculation
                valid_bins = []
                for i in range(len(bins) - 1):
                    m_lo = bins[i]
                    m_hi = bins[i + 1]
                    bin_center = (m_lo + m_hi) / 2.0
                    in_bin = (mags_all >= m_lo) & (mags_all < m_hi)
                    n_bin = int(np.sum(in_bin))

                    # Skip only empty bins (match plotting behavior)
                    # Sparse bins are still included for completeness calculation
                    if n_bin == 0:
                        continue

                    frac = float(np.mean(snrs_all[in_bin] >= snr_threshold))
                    valid_bins.append((bin_center, frac, m_lo, m_hi))

                if len(valid_bins) == 0:
                    return None

                # Scan from faint to bright to find where completeness drops below target
                # This is more reliable for limiting magnitude (faintest mag where completeness >= target)
                prev_frac: float | None = None
                prev_bin_center: float | None = None

                # Log first few bins for debugging (DEBUG level)
                if target_completeness == 0.5:
                    logger.debug(
                        f"Scanning for {target_completeness:.2f} completeness crossing. "
                        f"Valid bins: {len(valid_bins)}, "
                        f"faintest: {valid_bins[-1][0]:.2f} (completeness={valid_bins[-1][1]:.3f}), "
                        f"brightest: {valid_bins[0][0]:.2f} (completeness={valid_bins[0][1]:.3f})"
                    )

                for bin_center, frac, _, _ in reversed(valid_bins):
                    # Check if completeness equals the target exactly (within small tolerance)
                    if abs(frac - target_completeness) < 0.01:
                        logger.debug(
                            f"Found exact match for {target_completeness:.2f} completeness at mag={bin_center:.2f}"
                        )
                        return bin_center

                    # Check if we've crossed from above-target to below-target
                    # Since we're scanning faint-to-bright (reversed), we encounter:
                    # - Faint bins first (e.g., 17.88, 15.12)
                    # - Bright bins later (e.g., 14.88, 6.88)
                    # So prev is fainter (higher mag), current is brighter (lower mag)
                    # We want: prev (fainter) >= target AND current (brighter) < target
                    # OR: prev (fainter) < target AND current (brighter) >= target (going the other way)
                    if prev_frac is not None:
                        # Check for crossing: prev (fainter) >= target, current (brighter) < target
                        # This means completeness dropped as we go from faint to bright
                        if prev_frac >= target_completeness and frac < target_completeness:
                            # We've crossed the threshold - interpolate to find exact crossing point
                            if prev_frac != frac:  # Avoid division by zero
                                mag_crossing = prev_bin_center + (target_completeness - prev_frac) * (
                                    bin_center - prev_bin_center
                                ) / (frac - prev_frac)
                                logger.debug(
                                    f"Found crossing for {target_completeness:.2f} completeness: "
                                    f"{prev_frac:.3f} -> {frac:.3f} between mag "
                                    f"{prev_bin_center:.2f} (faint) and {bin_center:.2f} (bright), "
                                    f"interpolated to {mag_crossing:.2f}"
                                )
                                return mag_crossing
                            else:
                                logger.debug(
                                    f"Found crossing (equal fractions) for {target_completeness:.2f} "
                                    f"at mag={(prev_bin_center + bin_center) / 2.0:.2f}"
                                )
                                return (prev_bin_center + bin_center) / 2.0
                        # Also check the reverse: prev (fainter) < target, current (brighter) >= target
                        # This means completeness increased as we go from faint to bright
                        # We want the faintest mag where completeness >= target, so interpolate between them
                        elif prev_frac < target_completeness and frac >= target_completeness:
                            # We've crossed the threshold - interpolate to find exact crossing point
                            if prev_frac != frac:  # Avoid division by zero
                                mag_crossing = prev_bin_center + (target_completeness - prev_frac) * (
                                    bin_center - prev_bin_center
                                ) / (frac - prev_frac)
                                logger.debug(
                                    f"Found crossing (reverse) for {target_completeness:.2f} completeness: "
                                    f"{prev_frac:.3f} -> {frac:.3f} between mag "
                                    f"{prev_bin_center:.2f} (faint) and {bin_center:.2f} (bright), "
                                    f"interpolated to {mag_crossing:.2f}"
                                )
                                return mag_crossing
                            else:
                                logger.debug(
                                    f"Found crossing (equal fractions, reverse) for {target_completeness:.2f} "
                                    f"at mag={bin_center:.2f}"
                                )
                                return bin_center
                        elif prev_frac > target_completeness and frac <= target_completeness:
                            # Also catch the case where we go from > target to = target
                            if prev_frac != frac:
                                mag_crossing = prev_bin_center + (target_completeness - prev_frac) * (
                                    bin_center - prev_bin_center
                                ) / (frac - prev_frac)
                                logger.debug(
                                    f"Found crossing (> to =) for {target_completeness:.2f} completeness: "
                                    f"{prev_frac:.3f} -> {frac:.3f} between mag "
                                    f"{prev_bin_center:.2f} and {bin_center:.2f}, "
                                    f"interpolated to {mag_crossing:.2f}"
                                )
                                return mag_crossing
                            else:
                                logger.debug(
                                    f"Found crossing (> to =, equal) for {target_completeness:.2f} "
                                    f"at mag={bin_center:.2f}"
                                )
                                return bin_center

                    # Track previous bin for interpolation
                    prev_frac = frac
                    prev_bin_center = bin_center

                # If we never found a crossing, check if all bins are above or below target
                # But first, log what we scanned through to help debug
                if valid_bins:
                    faintest_bin_center, faintest_frac, _, _ = valid_bins[-1]
                    brightest_bin_center, brightest_frac, _, _ = valid_bins[0]

                    # Log summary if no crossing found (DEBUG level)
                    if target_completeness == 0.5:
                        logger.debug(
                            f"No crossing found for {target_completeness:.2f} completeness after scanning. "
                            f"Faintest bin: mag={faintest_bin_center:.2f}, completeness={faintest_frac:.3f}. "
                            f"Brightest bin: mag={brightest_bin_center:.2f}, completeness={brightest_frac:.3f}. "
                            f"Total valid bins: {len(valid_bins)}"
                        )

                    # Only return None if we're absolutely sure there's no crossing
                    # Don't use the edge case checks - they might be wrong if bins are sparse
                    # Instead, if we scanned through all bins and didn't find a crossing,
                    # it means there really isn't one (or the bins aren't adjacent)
                    logger.warning(
                        f"Could not find crossing for {target_completeness:.2f} completeness "
                        f"after scanning {len(valid_bins)} bins. "
                        f"Faintest: {faintest_bin_center:.2f} ({faintest_frac:.3f}), "
                        f"Brightest: {brightest_bin_center:.2f} ({brightest_frac:.3f})"
                    )
                    return None

                logger.warning(f"No valid bins found for {target_completeness:.2f} completeness calculation")
                return None
            except Exception as e:
                logger.debug(f"Completeness limit calculation failed for target {target_completeness}: {e}")
                return None

        # Calculate limits at different completeness levels
        try:
            # Use same bin width as plotting code for consistency (0.25 mag)
            bin_width = 0.25
            min_mag = float(np.floor(np.min(mags_all)))
            max_mag_actual = float(np.max(mags_all))  # Actual max, not rounded up
            max_mag = float(np.ceil(max_mag_actual))  # For binning, round up to include last bin
            bins = np.arange(min_mag, max_mag + bin_width, bin_width)

            # Count stars per bin for debugging
            n_bins_with_stars = 0
            n_bins_empty = 0
            for i in range(len(bins) - 1):
                in_bin = (mags_all >= bins[i]) & (mags_all < bins[i + 1])
                n_bin = int(np.sum(in_bin))
                if n_bin > 0:
                    n_bins_with_stars += 1
                else:
                    n_bins_empty += 1

            logger.info(
                f"Calculating completeness-based limiting magnitudes: "
                f"SNR threshold={snr_threshold:.1f}, "
                f"mag range={min_mag:.1f}-{max_mag_actual:.2f} (bins extend to {max_mag:.1f}), "
                f"bin_width={bin_width:.2f}, {n_bins_with_stars} bins with data, {n_bins_empty} empty bins"
            )

            # Calculate at configured target, 50%, and 90%
            completeness_limit = find_completeness_limit(completeness_target)
            completeness_limit_50 = find_completeness_limit(0.5)
            completeness_limit_90 = find_completeness_limit(0.9)

            if completeness_limit is not None:
                logger.info(
                    f"Completeness-based limiting magnitude (target={completeness_target:.2f}): "
                    f"{completeness_limit:.2f}"
                )
            if completeness_limit_50 is not None:
                logger.info(f"Limiting magnitude at 50% completeness: {completeness_limit_50:.2f}")
            if completeness_limit_90 is not None:
                logger.info(f"Limiting magnitude at 90% completeness: {completeness_limit_90:.2f}")
            if completeness_limit is None:
                logger.warning(
                    "Completeness-based limiting magnitude could not be determined "
                    f"(no crossing found, completeness never dropped below {completeness_target:.2f})"
                )
        except Exception as e:
            logger.warning(
                f"Completeness-based limiting magnitude estimation failed: {e}",
                exc_info=True,
            )
            completeness_limit = None
            completeness_limit_50 = None
            completeness_limit_90 = None

        # --- 1b) SNR-fit based limit ---
        snr_fit_limit: float | None = None
        snr_min_fit = 0.5  # allow lower SNR into the fit, but still avoid pure noise
        snr_max_fit = 100.0
        fit_mask = (snrs >= snr_min_fit) & (snrs <= snr_max_fit)

        if np.count_nonzero(fit_mask) >= 3:
            mags_fit = mags[fit_mask]
            snrs_fit = snrs[fit_mask]

            log_snrs_fit = np.log10(snrs_fit)
            coeffs = np.polyfit(mags_fit, log_snrs_fit, 1)

            limiting_log_snr = np.log10(snr_threshold)
            snr_fit_limit = float((limiting_log_snr - coeffs[1]) / coeffs[0])
            logger.info(
                f"SNR-fit limiting magnitude: {snr_fit_limit:.2f} "
                f"(fit: log10(SNR) = {coeffs[0]:.3f} * mag + {coeffs[1]:.3f}, "
                f"n_points={np.count_nonzero(fit_mask)})"
            )
        else:
            logger.info(
                f"SNR-fit limiting magnitude: not calculated (insufficient points: {np.count_nonzero(fit_mask)} < 3)"
            )

        # --- 1c) Combine empirical candidates ---
        # Prioritize completeness_limit since it matches the user's configured
        # limiting_completeness_fraction. If completeness_limit is available,
        # use it; otherwise fall back to SNR-fit limit.
        if completeness_limit is not None:
            logger.info(f"Using completeness-based limiting magnitude: {completeness_limit:.2f}")
            return (
                float(completeness_limit),
                completeness_limit_50,
                completeness_limit_90,
            )
        elif snr_fit_limit is not None:
            # Check if SNR-fit limit is extrapolated beyond actual data range
            max_mag_actual = float(np.max(mags_all)) if len(mags_all) > 0 else None
            if max_mag_actual is not None and snr_fit_limit > max_mag_actual:
                logger.warning(
                    f"SNR-fit limiting magnitude ({snr_fit_limit:.2f}) is extrapolated "
                    f"beyond actual data range (max sampled mag={max_mag_actual:.2f}). "
                    f"This may be unreliable."
                )
            logger.info(f"Using SNR-fit limiting magnitude (completeness limit unavailable): {snr_fit_limit:.2f}")
            # For SNR-fit, we don't have completeness-based 50% and 90% values
            return (float(snr_fit_limit), completeness_limit_50, completeness_limit_90)
        else:
            logger.warning(
                "Neither completeness nor SNR-fit limiting magnitude available, falling back to alternative methods"
            )

    # ------------------------------------------------------------------
    # 2) Fallback: sky-noise + zero-point based limit (standard formula)
    # ------------------------------------------------------------------
    # If zero point was not provided, compute it (backward compatibility)
    if zero_point is None:
        zero_point, _ = _calculate_simple_zero_point(results, starfield, config)

    try:
        if zero_point is not None and starfield.fwhm_stats is not None:
            # Representative sky noise per pixel from background_std across stars
            bg_stds = [r.background_std for r in results if getattr(r, "background_std", 0.0) > 0]
            if len(bg_stds) >= 10:
                sigma_sky = float(np.median(bg_stds))

                # Exposure time
                exposure_time = getattr(starfield.image_metadata, "exposure_time", None)
                if exposure_time is None or exposure_time <= 0:
                    exposure_time = 1.0  # Default to 1 second if not available

                # Number of pixels in the photometric aperture
                fwhm = starfield.fwhm_stats.median_fwhm
                aperture_radius = config.aperture_radius_factor * fwhm
                n_pix = np.pi * (aperture_radius**2)

                # Limiting magnitude at configured SNR (sky-noise dominated)
                snr_limit = float(config.limiting_snr)
                flux_limit = snr_limit * sigma_sky * np.sqrt(n_pix)

                if flux_limit > 0:
                    limiting_magnitude = zero_point - 2.5 * np.log10(flux_limit / exposure_time)
                    logger.info(
                        f"Using sky-noise based limiting magnitude (fallback): "
                        f"{limiting_magnitude:.2f} (ZP={zero_point:.2f}, "
                        f"sigma_sky={sigma_sky:.2f}, n_pix={n_pix:.1f})"
                    )
                    return (float(limiting_magnitude), None, None)
    except Exception as e:
        logger.debug(f"Sky-noise based limiting magnitude estimation failed: {e}")

    # ------------------------------------------------------------------
    # 3) Final fallback: original empirical fit of log10(SNR) vs magnitude
    # ------------------------------------------------------------------
    catalog_results = []
    # Reuse magnitude cache from above if available, otherwise compute
    for r in results:
        if not r.quality_flag or r.snr is None or r.snr <= 0:
            continue
        # Use cached magnitude if available
        star_id = id(r.star)
        if star_id in result_mag_cache:
            mag = result_mag_cache[star_id]
        else:
            mag = _get_best_magnitude(r.star, config.preferred_filters if config else None)
        if mag is not None:
            catalog_results.append((mag, r.snr))

    if len(catalog_results) < 3:
        logger.warning(
            f"Final fallback: insufficient catalog results ({len(catalog_results)} < 3), "
            "using conservative default: 15.0"
        )
        return 15.0  # Conservative default

    magnitudes = [item[0] for item in catalog_results]
    log_snrs = [np.log10(item[1]) for item in catalog_results]

    # Linear fit
    coeffs = np.polyfit(magnitudes, log_snrs, 1)

    limiting_log_snr = np.log10(float(config.limiting_snr))
    limiting_magnitude = (limiting_log_snr - coeffs[1]) / coeffs[0]

    logger.info(
        f"Using final fallback limiting magnitude: {limiting_magnitude:.2f} "
        f"(fit: log10(SNR) = {coeffs[0]:.3f} * mag + {coeffs[1]:.3f}, "
        f"n_points={len(catalog_results)})"
    )
    return (float(limiting_magnitude), None, None)


def _calculate_simple_zero_point(
    results: list[SimplePhotometryResult],
    starfield: StarField,
    config: SimplePhotometryConfig | None = None,
) -> tuple[float | None, float | None]:
    """Calculate simple photometric zero point."""

    # Normalize config so we always have SimplePhotometryConfig
    config = _normalize_photometry_config(config)

    # Use quality measurements with known magnitudes
    catalog_results = []

    # Pre-compute isolation radius for neighbor checks (in pixels)
    iso_radius_pix: float | None = None
    catalog_stars = starfield.catalog_stars or []
    mag_cache: dict[int, float | None] | None = None
    kdtree: tuple[cKDTree, list[int]] | None = None

    if starfield.fwhm_stats is not None:
        fwhm = starfield.fwhm_stats.median_fwhm
        aperture_radius = config.aperture_radius_factor * fwhm
        iso_radius_pix = config.isolation_radius_factor * aperture_radius

        # Pre-compute magnitudes for all catalog stars and build KD-tree for fast neighbor queries
        if catalog_stars and iso_radius_pix is not None:
            mag_cache = _precompute_star_magnitudes(catalog_stars, config.preferred_filters if config else None)

            # Build KD-tree for spatial queries (only for stars with valid positions)
            positions = []
            position_to_star = []  # Maps KD-tree index to catalog_stars index
            for i, star in enumerate(catalog_stars):
                if star.x is not None and star.y is not None:
                    positions.append([star.x, star.y])
                    position_to_star.append(i)

            if positions:
                kdtree = (
                    cKDTree(positions),
                    position_to_star,
                )  # Store mapping with tree
            else:
                kdtree = None

    # Pre-compute magnitudes for result stars
    result_mag_cache = {}
    for r in results:
        star_id = id(r.star)
        if star_id not in result_mag_cache:
            result_mag_cache[star_id] = _get_best_magnitude(r.star, config.preferred_filters if config else None)

    def _select(min_snr: float) -> list[tuple[float, float]]:
        """Catalog (mag, flux) pairs from clean, well-measured stars at/above
        ``min_snr``, excluding crowded and bright-neighbour-blended sources."""
        sel: list[tuple[float, float]] = []
        for r in results:
            if not r.quality_flag or r.flux <= 0 or r.snr < min_snr:
                continue
            if r.crowding_factor > config.zp_max_crowding:
                continue
            mag = result_mag_cache[id(r.star)]
            if mag is None:
                continue
            if iso_radius_pix is not None and catalog_stars:
                if _has_bright_neighbor(
                    r.star, mag, catalog_stars, iso_radius_pix,
                    config.isolation_delta_mag, mag_cache=mag_cache, kdtree=kdtree,
                ):
                    continue
            sel.append((mag, r.flux))
        return sel

    # Prefer the high-SNR, uncrowded sample; if a frame is too shallow/sparse to
    # supply enough such stars, relax the SNR floor toward the detection limit so
    # we still return a (lower-confidence) ZP rather than nothing.
    catalog_results = _select(config.zp_min_snr)
    if len(catalog_results) < config.zp_min_stars:
        relaxed = _select(max(config.limiting_snr, 5.0))
        if len(relaxed) > len(catalog_results):
            catalog_results = relaxed

    if len(catalog_results) < 3:
        return None, None

    # Get exposure time (handle None case)
    exposure_time = getattr(starfield.image_metadata, "exposure_time", None)
    if exposure_time is None or exposure_time <= 0:
        exposure_time = 1.0  # Default to 1 second if not available

    # Per-star zero points: ZP = m + 2.5 * log10(flux/texp)
    mags_arr = np.array([m for m, _ in catalog_results], dtype=float)
    flux_arr = np.array([f for _, f in catalog_results], dtype=float)
    zps = mags_arr + 2.5 * np.log10(flux_arr / exposure_time)

    # Median + sigma-clip: robust to any residual blend/saturation outliers on
    # either tail (median, not mean; the old mean let the faint tail pull the ZP).
    from astropy.stats import mad_std, sigma_clip

    clipped = sigma_clip(zps, sigma=config.zp_sigma_clip, maxiters=5, masked=True)
    kept = zps[~clipped.mask] if np.ma.is_masked(clipped) else zps
    if kept.size < 3:
        kept = zps
    zero_point = float(np.median(kept))
    robust_sigma = float(mad_std(kept)) if kept.size > 1 else 0.0
    zero_point_err = robust_sigma / np.sqrt(kept.size) if kept.size else 0.0

    return zero_point, zero_point_err


def calculate_star_snrs_with_aperture_photometry(
    frame: SiderealFrame | RateTrackFrame, catalog_stars: list[StarInSpace], plot: bool = True
) -> list[tuple[StarInSpace, float, float]]:
    """Calculate SNRs for catalog stars using proper aperture photometry.

    This is a shared photometry utility used by both sidereal and rate-track
    pipelines as well as WCS refinement code.
    """
    from photutils.aperture import (
        CircularAnnulus,
        CircularAperture,
        RectangularAnnulus,
        RectangularAperture,
        aperture_photometry,
    )

    # Determine frame type
    is_sidereal = isinstance(frame, SiderealFrame)

    # Filter stars with valid positions in the image bounds
    height, width = frame.frame.data.shape
    _cached = getattr(frame.frame, "_photometry_counts", None)
    if _cached is not None:
        counts_array = _cached
    else:
        counts_array = frame.frame.data.copy()
        counts_array -= np.min(counts_array)
        object.__setattr__(frame.frame, "_photometry_counts", counts_array)
    # Empirical per-pixel noise: the min-shift above makes the background
    # *level* an arbitrary offset, so Poisson-from-level is meaningless here
    # (and the frame is background-subtracted upstream anyway) — see
    # empirical_background_std_adu.
    bg_std_counts = empirical_background_std_adu(counts_array)
    margin = 10
    valid_stars: list[StarInSpace] = []
    positions: list[tuple[float, float]] = []

    for star in catalog_stars:
        if star.x is not None and star.y is not None:
            if margin <= star.x < width - margin and margin <= star.y < height - margin:
                valid_stars.append(star)
                positions.append((star.x, star.y))

    if not valid_stars:
        return []

    results: list[tuple[StarInSpace, float, float]] = []

    if is_sidereal:
        # For sidereal frames, use circular apertures (can process all at once)
        fwhm = frame.starfield.detection_metadata.pixel_fwhm
        radius = max(1.5 * fwhm, 3.0)  # Use at least 3 pixels radius

        # Chunked to bound peak memory (see _chunked_aperture_sums): on a
        # dense field this runs over all catalog stars with FWHM-scaled
        # apertures and a single call peaked ~42 GB → OOM (_full7).
        aper_sum_0, aper_sum_1 = _chunked_aperture_sums(
            counts_array, positions,
            lambda p: [
                CircularAperture(p, r=radius),
                CircularAnnulus(p, r_in=radius * 1.5, r_out=radius * 2.5),
            ],
            bbox_side=2.0 * (radius * 2.5) + 2.0,
        )
        # Lightweight objects for .area only (scalar; no mask materialization)
        apertures = CircularAperture(positions, r=radius)
        bg_apertures = CircularAnnulus(positions, r_in=radius * 1.5, r_out=radius * 2.5)

        # Calculate background-subtracted counts and SNR for each star
        for i, star in enumerate(valid_stars):
            aperture_sum = float(aper_sum_0[i])
            bg_sum = float(aper_sum_1[i])

            # Get areas - for multiple apertures, these are arrays
            bg_area = bg_apertures.area
            aperture_area = apertures.area

            # If we have multiple apertures, get the specific one for this star
            if hasattr(bg_area, "__len__"):
                bg_area = bg_area[i]
                aperture_area = aperture_area[i]

            # Calculate background per pixel and subtract from aperture
            bg_per_pixel = bg_sum / bg_area
            counts = aperture_sum - (bg_per_pixel * aperture_area)

            # Calculate noise (Poisson source + empirical background noise)
            bg_noise = (
                bg_std_counts * np.sqrt(aperture_area)
                if bg_std_counts > 0
                else np.sqrt(bg_per_pixel * aperture_area)
            )
            source_noise = np.sqrt(max(0, counts))
            total_noise = np.sqrt(source_noise**2 + bg_noise**2)

            # Calculate SNR
            snr = counts / total_noise if total_noise > 0 else 0

            results.append((star, snr, counts))
    else:
        # For rate-track frames, process all stars at once with rotated apertures
        streak = frame.streak
        width_pixels = streak.fwhm * 4
        length_pixels = streak.pixel_length + streak.fwhm * 2
        theta = streak.radian_angle() + np.pi / 2  # photutils angle convention

        # Chunked to bound peak memory (see _chunked_aperture_sums): streak-
        # length rectangular apertures over all catalog stars otherwise
        # materialize tens of GB of masks at once.
        aper_sum_0, aper_sum_1 = _chunked_aperture_sums(
            counts_array, positions,
            lambda p: [
                RectangularAperture(p, w=width_pixels, h=length_pixels, theta=theta),
                RectangularAnnulus(
                    p,
                    w_in=width_pixels + 2,
                    w_out=width_pixels + 6,
                    h_in=length_pixels + 2,
                    h_out=length_pixels + 6,
                    theta=theta,
                ),
            ],
            bbox_side=1.5 * max(length_pixels + 6, width_pixels + 6),
        )
        # Lightweight objects for .area only (scalar; no mask materialization)
        apertures = RectangularAperture(positions, w=width_pixels, h=length_pixels, theta=theta)
        bg_apertures = RectangularAnnulus(
            positions,
            w_in=width_pixels + 2,
            w_out=width_pixels + 6,
            h_in=length_pixels + 2,
            h_out=length_pixels + 6,
            theta=theta,
        )

        # Calculate background-subtracted counts and SNR for each star
        for i, star in enumerate(valid_stars):
            aperture_sum = float(aper_sum_0[i])
            bg_sum = float(aper_sum_1[i])

            # Get areas - for multiple apertures, these are arrays
            bg_area = bg_apertures.area
            aperture_area = apertures.area

            # If we have multiple apertures, get the specific one for this star
            if hasattr(bg_area, "__len__"):
                bg_area = bg_area[i]
                aperture_area = aperture_area[i]

            # Calculate background per pixel and subtract from aperture
            bg_per_pixel = bg_sum / bg_area
            counts = aperture_sum - (bg_per_pixel * aperture_area)

            # Calculate noise (Poisson source + empirical background noise)
            bg_noise = (
                bg_std_counts * np.sqrt(aperture_area)
                if bg_std_counts > 0
                else np.sqrt(bg_per_pixel * aperture_area)
            )
            source_noise = np.sqrt(max(0, counts))
            total_noise = np.sqrt(source_noise**2 + bg_noise**2)

            # Calculate SNR
            snr = counts / total_noise if total_noise > 0 else 0

            results.append((star, snr, counts))

    # Diagnostic plot emitted for *both* sidereal (circular apertures) and
    # rate-track (rectangular apertures) — previously only the rate branch
    # produced this, so sidereal frames had no aperture overlay despite the
    # config flag being on.
    if plot and get_config().plotting.photometry:
        from senpai.engine.plotting.images import plot_photometry_frame

        plot_photometry_frame(
            counts_array,
            apertures=apertures,
            annuli=bg_apertures,
            output_file=get_config().runtime.output_dir / f"frame_{frame.index}_aperture_photometry_stars.png",
        )
        logger.info(
            "Saved aperture photometry plot to %s",
            get_config().runtime.output_dir / f"frame_{frame.index}_aperture_photometry_stars.png",
        )

    return results


def estimate_limiting_magnitude_from_photometry(
    frame: SiderealFrame | RateTrackFrame,
    star_snr_results: list[tuple[StarInSpace, float, float]],
    min_snr: float = 3.0,
) -> float:
    """Estimate limiting magnitude using proper photometry results.

    This implementation is shared between sidereal and rate-track pipelines,
    and is used by WCS refinement code. The default SNR threshold is 3σ,
    configurable via the min_snr parameter.
    """
    from senpai.engine.models.senpai import RateTrackFrame

    # Determine frame type
    is_rate_track = isinstance(frame, RateTrackFrame)

    # Extract magnitude and SNR pairs
    mag_snr_pairs = [(star.magnitude, snr) for star, snr, _ in star_snr_results if star.magnitude is not None]

    if not mag_snr_pairs:
        return 15.0 if is_rate_track else 16.0  # Conservative default

    # Sort by magnitude
    mag_snr_pairs.sort(key=lambda x: x[0])

    # Try to fit a linear relationship between magnitude and log(SNR)
    magnitudes = np.array([m for m, _ in mag_snr_pairs])
    log_snrs = np.array([np.log10(max(s, 0.1)) for _, s in mag_snr_pairs])

    # Filter out stars with artificially capped SNR values
    valid_indices = [i for i, (_, snr) in enumerate(mag_snr_pairs) if snr > 0.1]
    if valid_indices:
        filtered_magnitudes = magnitudes[valid_indices]
        filtered_log_snrs = log_snrs[valid_indices]
    else:
        filtered_magnitudes = magnitudes
        filtered_log_snrs = log_snrs

    # Simple linear regression + completeness hybrid
    try:
        cfg = get_config()
        completeness_target = float(getattr(cfg.photometry, "limiting_completeness_fraction", 0.5))

        # Group stars by magnitude bins to calculate completeness and weights
        bin_width = 0.5  # magnitude bin width
        min_mag = np.floor(np.min(filtered_magnitudes))
        max_mag = np.ceil(np.max(filtered_magnitudes))
        bins = np.arange(min_mag, max_mag + bin_width, bin_width)

        # Initialize weights
        weights = np.ones_like(filtered_magnitudes)

        # Calculate completeness and variance in each bin and assign weights
        if len(filtered_magnitudes) > 10:
            bin_indices = np.digitize(filtered_magnitudes, bins)

            # Parameters to control weighting
            min_stars_per_bin = 3
            max_weight_factor = 10.0
            variance_floor = 0.01

            bin_variances: list[float] = []
            bin_counts: list[int] = []

            # For completeness
            completeness_candidates: list[float] = []

            for bin_idx in range(1, len(bins)):
                bin_mask = bin_indices == bin_idx
                bin_count = int(np.sum(bin_mask))
                bin_counts.append(bin_count)

                if bin_count >= min_stars_per_bin:
                    bin_var = float(np.var(filtered_log_snrs[bin_mask]))
                    bin_var = max(bin_var, variance_floor)

                    # Completeness in this bin
                    bin_snrs = 10 ** filtered_log_snrs[bin_mask]
                    frac = float(np.mean(bin_snrs >= min_snr))
                    if frac >= completeness_target:
                        completeness_candidates.append(bins[bin_idx])
                else:
                    bin_var = 1.0  # Default high variance for sparse bins

                bin_variances.append(bin_var)

            completeness_limit: float | None = None
            if completeness_candidates:
                completeness_limit = max(completeness_candidates)

            if bin_variances:
                inverse_variances = [1.0 / var for var in bin_variances]
                median_inv_var = float(np.median(inverse_variances))

                for bin_idx in range(1, len(bins)):
                    bin_mask = bin_indices == bin_idx
                    if np.sum(bin_mask) > 0:
                        bin_var = bin_variances[bin_idx - 1]

                        weight = 1.0 / bin_var
                        weight = min(weight, median_inv_var * max_weight_factor)

                        if bin_counts[bin_idx - 1] < min_stars_per_bin:
                            weight *= (bin_counts[bin_idx - 1] / min_stars_per_bin) ** 2

                        weights[bin_mask] = weight

        # Use weighted least squares for the fit
        coeffs = np.polyfit(filtered_magnitudes, filtered_log_snrs, 1, w=weights)
        slope, intercept = coeffs

        # SNR-fit limiting magnitude at threshold
        fit_limiting_mag = (np.log10(min_snr) - intercept) / slope
        fit_limiting_mag = max(12.0, fit_limiting_mag)

        # Choose the limiting magnitude. completeness_limit is the faintest bin
        # still above the target completeness; if that sits at the faint edge of
        # the data, the curve never actually rolled over — it's just where the
        # (deliberately shallow, for speed) catalog ran out, not a real 50%
        # limit. In that case use the SNR-fit crossing (the fitted trend
        # extrapolated to the threshold). Only trust completeness when it rolls
        # over *within* the data.
        data_faint_edge = float(np.max(filtered_magnitudes))
        comp = completeness_limit if "completeness_limit" in locals() else None
        if comp is not None and comp < data_faint_edge - 0.5:
            limiting_mag = comp  # genuine completeness roll-over within the data
        else:
            limiting_mag = fit_limiting_mag  # truncated → extrapolate trend to SNR threshold

        # Optional diagnostic plot
        if cfg.plotting.photometry:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 6))

            ax.scatter(
                filtered_magnitudes,
                filtered_log_snrs,
                c="blue",
                alpha=0.5,
                s=weights * 20 / np.max(weights),
                label="Stars",
            )

            mag_range = np.linspace(
                float(np.min(filtered_magnitudes)),
                float(np.max(filtered_magnitudes) + 2),
                100,
            )
            fitted_line = slope * mag_range + intercept
            ax.plot(mag_range, fitted_line, "r--", label="Fitted Trend")

            ax.axhline(
                y=np.log10(min_snr),
                color="g",
                linestyle=":",
                label=f"SNR={min_snr} Threshold",
            )
            ax.axvline(x=limiting_mag, color="k", linestyle="--", label="Limiting Magnitude")

            ax.set_xlabel("Magnitude")
            ax.set_ylabel("log10(SNR)")
            ax.set_title("Limiting Magnitude Estimation")
            ax.grid(True, alpha=0.3)
            ax.legend()

            output_path = cfg.runtime.output_dir / f"frame_{frame.index}_limiting_mag.png"
            plt.savefig(output_path)
            plt.close()
            logger.info("Saved limiting magnitude diagnostic to %s", output_path)

        return float(limiting_mag)

    except Exception:
        # Fallback if fitting fails: use faintest star above threshold
        good_stars = [(m, s) for m, s in mag_snr_pairs if s >= min_snr]
        if good_stars:
            faintest_good_mag = max(m for m, _ in good_stars)
            margin = 0.5  # Conservative margin
            return float(faintest_good_mag + margin)
        else:
            return 12.0 if is_rate_track else 13.0


# Keep the old function names for backward compatibility
PhotometryResult = SimplePhotometryResult
PhotometrySummary = SimplePhotometrySummary
measure_star_photometry = measure_simple_star_photometry
measure_starfield_photometry = measure_simple_starfield_photometry
