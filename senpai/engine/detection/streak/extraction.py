import logging

import numpy as np
from scipy.ndimage import median_filter, rotate
from scipy.optimize import curve_fit
from scipy.signal import convolve

from senpai.core.config import get_config
from senpai.engine.detection.kernels import rectangle_pyramoid
from senpai.engine.detection.streak.masking import (
    analyze_source_shape_fwhm,
    map_cluster,
    mask_all_but_border,
    remove_streak_at_point,
    remove_streak_at_point_enriched,
)
from senpai.engine.models.astrometry import WCSModel
from senpai.engine.models.metadata import FrameMetadata
from senpai.engine.models.streak_measurement import StreakMeasurement

logger = logging.getLogger(__name__)


def extract_streak_from_metadata(
    metadata: FrameMetadata, plate_scale_arcsec: float, wcs_model: WCSModel
) -> StreakMeasurement | None:
    if (
        not metadata.track_rate_dec_arcsec_per_second
        and not metadata.track_rate_ra_arcsec_per_second
    ):
        return None

    if not metadata.exposure_time_seconds:
        return None

    # Calculate streak length in arcseconds
    streak_length_arcsec = (
        np.sqrt(
            metadata.track_rate_ra_arcsec_per_second**2
            + metadata.track_rate_dec_arcsec_per_second**2
        )
        * metadata.exposure_time_seconds
    )
    streak_length_pixels = streak_length_arcsec / plate_scale_arcsec

    # Get the PC matrix elements
    pc1_1 = wcs_model.PC1_1
    pc1_2 = wcs_model.PC1_2
    pc2_1 = wcs_model.PC2_1
    pc2_2 = wcs_model.PC2_2

    # Calculate how a vector in RA/Dec space transforms to pixel space
    # The track rate is in (RA, Dec) space, so we need to transform it to pixel space
    # Track rate vector in RA/Dec space (normalized)
    ra_rate = metadata.track_rate_ra_arcsec_per_second
    dec_rate = metadata.track_rate_dec_arcsec_per_second

    # Apply the PC matrix transformation to get the direction in pixel space
    # Note: The PC matrix transforms from world to pixel coordinates
    dx = pc1_1 * ra_rate + pc1_2 * dec_rate
    dy = pc2_1 * ra_rate + pc2_2 * dec_rate

    # Calculate the angle in the image plane
    # Match the PCA-based angle calculation used in image measurements
    # PCA measures angle from vertical axis using arctan2(eigenvectors[0, 0], eigenvectors[1, 0])
    # For a vector (dx, dy), this is equivalent to arctan2(dy, dx) + 90
    streak_rotation = np.degrees(np.arctan2(dy, dx))

    # Normalize to [0, 180) range and handle 180° ambiguity
    # First ensure we're in [0, 360)
    streak_rotation = streak_rotation % 360
    # Then map to [0, 180) by reflecting angles > 180
    if streak_rotation >= 180:
        streak_rotation -= 180

    logger.info(
        f"Calculated streak rotation: {streak_rotation:.2f}° from RA/Dec track rates using PC matrix"
    )
    logger.info(f"Calculated streak length: {streak_length_pixels:.2f} pixels")

    return StreakMeasurement(
        rotation=streak_rotation,
        length=streak_length_pixels,
        fwhm=None,
    )


def prepare_rate_frame(rate_frame, padding: float = 0.05) -> np.ndarray:
    """Prepare rate frame for cross-correlation.

    Parameters
    ----------
    rate_frame : RateTrackFrame
        The rate frame to prepare
    padding : float
        Fraction of image to remove from edges (default: 0.1)

    Returns
    -------
    np.ndarray
        Prepared frame data
    """
    # Get the data after any scaling has been applied
    rate_data = rate_frame.frame.data.copy().astype(np.float32)

    # Calculate padding based on original dimensions from metadata
    height = rate_frame.frame.data.shape[0]
    width = rate_frame.frame.data.shape[1]

    # Calculate padding in original dimensions
    p10w = int(height * padding)
    p10h = int(width * padding)

    # Apply padding crop
    rate_data = rate_data[p10w:-p10w, p10h:-p10h]

    # Normalize
    rate_data -= np.min(rate_data)

    return rate_data


def prepare_sidereal_frame(
    sidereal_frame, padding: float = 0.05
) -> tuple[np.ndarray, bool]:
    """Prepare sidereal frame for cross-correlation.

    Parameters
    ----------
    sidereal_frame : SiderealFrame
        The sidereal frame to prepare
    padding : float
        Fraction of image to remove from edges (default: 0.1)

    Returns
    -------
    tuple[np.ndarray, bool]
        Prepared frame data and whether synthetic frame was used
    """
    # Check if we should use synthetic frame
    if sidereal_frame.starfield and sidereal_frame.starfield.wcs:
        from senpai.engine.utils.simulation import simulated_sidereal_frame

        # Generate synthetic frame using the same scaling process as the real image
        synthetic_frame = simulated_sidereal_frame(sidereal_frame.starfield)

        if synthetic_frame is not None:
            # Calculate padding based on synthetic frame dimensions (which should match scaled real frame)
            current_height, current_width = synthetic_frame.shape

            # Calculate padding in current dimensions
            p10w = int(current_height * padding)
            p10h = int(current_width * padding)

            # Apply padding crop
            synthetic_frame = synthetic_frame[p10w:-p10w, p10h:-p10h]
            return synthetic_frame, True

    # Get the data after any scaling has been applied
    sidereal_data = sidereal_frame.frame.data.copy().astype(np.float32)

    # Calculate padding based on original dimensions
    height = sidereal_frame.frame.metadata.height
    width = sidereal_frame.frame.metadata.width

    # Calculate padding in original dimensions
    p10w = int(height * padding)
    p10h = int(width * padding)

    # Apply padding crop
    sidereal_data = sidereal_data[p10w:-p10w, p10h:-p10h]

    # Apply median filter to reduce noise
    sidereal_data = median_filter(sidereal_data, size=2)

    # Normalize
    sidereal_data -= np.min(sidereal_data)

    return sidereal_data, False


def refine_robust_streak(
    psf: np.ndarray, seed: StreakMeasurement, frame_index: int = None
) -> tuple[StreakMeasurement, float]:
    """Refine streak parameters using a two-stage approach.

    Stage 1: Use moderate threshold to find orientation and core
    Stage 2: Use low threshold to capture all pixels (including faint wings),
             then project onto the principal axis for full length measurement

    This prevents underestimating length when the intensity threshold cuts off
    the fainter portions of the streak PSF.

    Args:
        psf: 2D numpy array containing the extracted PSF (normalized 0-1)
        seed: Initial StreakMeasurement with rotation, length, fwhm estimates
        frame_index: Optional frame index for labeling output plots

    Returns:
        tuple: (refined StreakMeasurement, measured_fwhm)
    """
    if psf is None or seed is None:
        logger.warning("Invalid inputs to refine_robust_streak")
        return seed, seed.fwhm if seed else None

    logger.info(
        f"Refining streak: seed L={seed.length:.1f}, θ={seed.rotation:.1f}°, W={seed.fwhm:.1f}"
    )

    # Normalize PSF against a *stable* max, not the single brightest pixel.
    # A lone hot/cosmic pixel inflates np.max → every fractional threshold (0.5
    # FWHM included) lands too low → length is over-measured. A 3x3 median
    # rejects single-pixel spikes while preserving the streak, so its max is the
    # true peak scale.
    from scipy.ndimage import median_filter

    psf_norm = psf.copy().astype(float)
    psf_norm -= np.min(psf_norm)
    stable_max = float(np.max(median_filter(psf_norm, size=3)))
    if stable_max <= 0:
        stable_max = float(np.max(psf_norm))
    if stable_max > 0:
        psf_norm /= stable_max  # hot pixels may exceed 1.0; harmless for thresholding

    # Rotate PSF to align streak horizontally
    rotated_psf = rotate(psf_norm, angle=seed.rotation, mode="constant", cval=0)

    # STAGE 1: Use moderate thresholds to measure core properties and width
    # These measurements help us understand the bright core
    core_thresholds = [0.25, 0.35, 0.45, 0.55]
    core_widths = []
    core_lengths = []  # Store for comparison/diagnostics

    for thresh in core_thresholds:
        mask = rotated_psf > thresh

        if not np.any(mask):
            continue

        from scipy.ndimage import label

        labeled_mask, num_features = label(mask)

        if num_features == 0:
            continue

        component_sizes = np.bincount(labeled_mask.ravel())[1:]
        if len(component_sizes) == 0:
            continue

        largest_component = np.argmax(component_sizes) + 1
        largest_mask = labeled_mask == largest_component

        y_coords, x_coords = np.where(largest_mask)

        if len(y_coords) < 5:
            continue

        # Width measurement (perpendicular to streak)
        width_measured = np.max(y_coords) - np.min(y_coords)
        core_widths.append(width_measured)

        # Store core length for diagnostics
        length_core = np.max(x_coords) - np.min(x_coords)
        core_lengths.append((thresh, length_core))

    # STAGE 2: Use flood fill from center with low threshold to capture wings
    # This ensures we follow the streak and don't pick up disconnected noise

    # Find the brightest point as starting location
    center_y, center_x = np.unravel_index(np.argmax(rotated_psf), rotated_psf.shape)

    # Use flood fill with a moderate threshold to capture wings without noise
    # Start with 0.15, which should capture wings but avoid background noise
    wing_threshold = 0.15

    # Import flood fill function from masking module
    from senpai.engine.detection.streak.masking import map_cluster

    # Use flood fill to get connected region starting from the brightest point
    full_mask = map_cluster(
        rotated_psf, (center_y, center_x), wing_threshold, pad_size=0
    )

    # Flood-fill extent is a diagnostic only now (the 1D-FWHM below is
    # authoritative), so a failed/tiny fill must NOT abort — e.g. when the
    # brightest pixel is a hot pixel the fill captures nothing.
    if np.any(full_mask):
        y_coords_all, x_coords_all = np.where(full_mask)
        flood_fill_length = (
            float(np.max(x_coords_all) - np.min(x_coords_all))
            if len(y_coords_all) >= 5 else 0.0
        )
    else:
        flood_fill_length = 0.0

    # Also compute a "profile extent" length that does NOT assume connectivity.
    # If the streak is broken into multiple blobs (e.g., due to a faint gap or
    # a processing/rotation artifact), flood-fill and connected-components can
    # underestimate length by measuring only one blob. A 1D profile along the
    # streak axis is robust to that: we take the min/max x where the profile
    # indicates any streak signal within a narrow band around the streak center.
    try:
        from scipy.ndimage import gaussian_filter1d

        # Use a narrow strip around the streak center to avoid unrelated structure.
        # Seed.fwhm is in pixels; use a generous multiplier to include wings.
        half_band = int(max(6, round((seed.fwhm or 4.0) * 3.0)))
        y0 = max(0, center_y - half_band)
        y1 = min(rotated_psf.shape[0], center_y + half_band + 1)
        strip = rotated_psf[y0:y1, :]

        x_profile = np.max(strip, axis=0)
        x_profile = gaussian_filter1d(x_profile, sigma=2.0)

        # Threshold relative to profile peak, but never below a small floor.
        profile_thresh = max(0.06, 0.12 * float(np.max(x_profile)))
        above = x_profile > profile_thresh

        if np.any(above):
            idx = np.where(above)[0]
            profile_extent_length = float(idx[-1] - idx[0])
        else:
            profile_extent_length = 0.0
    except Exception as e:
        logger.debug("Profile-extent length estimation failed: %s", e)
        profile_extent_length = 0.0

    logger.info(f"Core lengths at thresholds: {core_lengths}")
    logger.info(f"Flood fill length (thresh={wing_threshold}): {flood_fill_length:.1f}")
    if profile_extent_length > 0:
        logger.info(
            "Profile extent length (strip around center, thresh=%.3f): %.1f",
            profile_thresh,
            profile_extent_length,
        )

    # Smart length selection: Use flood fill ONLY when core measurements show significant drop
    if core_lengths:
        core_length_values = [length for _, length in core_lengths]
        min_core_length = min(core_length_values)
        max_core_length = max(core_length_values)

        logger.info(
            f"Core length range: [{min_core_length:.1f}, {max_core_length:.1f}] pixels"
        )

        # Calculate how much the length drops from lowest (0.25) to highest (0.55) threshold
        # A big drop indicates faint wings are being cut off at higher thresholds
        length_drop_ratio = (
            (max_core_length - min_core_length) / max_core_length
            if max_core_length > 0
            else 0
        )

        logger.info(
            f"Length drop across thresholds: {length_drop_ratio:.2%} "
            f"({max_core_length:.1f} → {min_core_length:.1f})"
        )

        # Strategy: Only use flood fill when there's a SIGNIFICANT drop (>30%) across thresholds
        # This indicates the thresholds are cutting off faint wings
        if length_drop_ratio > 0.30:
            # Significant drop - wings are being cut off, flood fill can help
            logger.info(
                f"Large threshold drop ({length_drop_ratio:.1%}), considering flood fill"
            )

            if (
                flood_fill_length >= min_core_length
                and flood_fill_length <= max_core_length
            ):
                # Flood fill within bounds - use it to capture wings
                refined_length = flood_fill_length
                logger.info(
                    f"Using flood fill ({flood_fill_length:.1f}) to capture faint wings"
                )
            elif flood_fill_length > max_core_length:
                # Flood fill too long - cap at max core
                refined_length = max_core_length
                logger.info(
                    f"Flood fill ({flood_fill_length:.1f}) exceeds max core, "
                    f"using max core: {refined_length:.1f}"
                )
            else:
                # Flood fill failed - use max core
                refined_length = max_core_length
                logger.info(
                    f"Flood fill ({flood_fill_length:.1f}) too short, "
                    f"using max core: {refined_length:.1f}"
                )
        else:
            # Small drop - core measurements are stable
            # Use measurement closest to FWHM (0.5) or 0.45 threshold, not the longest (0.25)
            # Find the threshold closest to 0.45-0.50 range
            target_thresholds = [0.45, 0.55]  # Prioritize these for stable cases

            # Filter core_lengths to those in our target range
            target_measurements = [
                (thresh, length)
                for thresh, length in core_lengths
                if thresh in target_thresholds
            ]

            if target_measurements:
                # Use the measurement from 0.45 or 0.55 threshold
                refined_length = target_measurements[0][1]  # Take first match
                logger.info(
                    f"Stable core measurements (drop={length_drop_ratio:.1%}), "
                    f"using threshold {target_measurements[0][0]} measurement: {refined_length:.1f}"
                )
            else:
                # Fallback: use the median of available measurements
                refined_length = np.median(core_length_values)
                logger.info(
                    f"Stable core measurements (drop={length_drop_ratio:.1%}), "
                    f"using median: {refined_length:.1f}"
                )
    else:
        # No valid core measurements, have to trust flood fill
        refined_length = flood_fill_length
        logger.warning("No core measurements available, using flood fill only")

    # If the streak is broken into multiple blobs, flood-fill/core methods can
    # underestimate. If the profile-extent is substantially larger (but still
    # plausible), prefer it. Also adopt when profile and seed agree closely —
    # in crowded fields the connected-component "core" lengths collapse with
    # threshold while the 1D strip-profile and the matched-filter seed both
    # remain reliable, so their agreement is itself strong evidence.
    # Profile-extent adoption removed: it preferred the faint full-PSF edge
    # (~the whole trail incl. wings), over-measuring length. The authoritative
    # length/width are now the 1D-collapse FWHM computed below; core/flood/
    # profile values above are retained only as diagnostics for the debug plot.
    profile_adopted = False

    # Width refinement using core measurements
    if core_widths:
        # Use median of core width measurements
        refined_width = np.median(core_widths)
    else:
        logger.warning("No valid width measurements, using seed")
        refined_width = seed.fwhm

    # CRITICAL: When we have core measurements, the final result MUST stay within their bounds
    # Core measurements are direct PSF observations and define the valid range.
    # Exception: if the strip-profile extent was adopted (lines above), the
    # core lengths have already been judged unreliable for this streak —
    # typically a long streak in a crowded field — and clamping back to them
    # would re-introduce the bug we just escaped.
    if core_lengths and not profile_adopted:
        # Absolutely enforce core bounds - no seed validation can override this
        if refined_length < min_core_length:
            logger.warning(
                f"Refined length ({refined_length:.1f}) below min core ({min_core_length:.1f}), "
                f"clamping to min core"
            )
            refined_length = min_core_length
        elif refined_length > max_core_length:
            logger.warning(
                f"Refined length ({refined_length:.1f}) above max core ({max_core_length:.1f}), "
                f"clamping to max core"
            )
            refined_length = max_core_length

        # Log final measurement relative to seed
        if abs(refined_length - seed.length) > seed.length * 0.5:
            logger.info(
                f"Core-based measurement ({refined_length:.1f}) differs significantly from seed ({seed.length:.1f}), "
                f"trusting core measurements"
            )
        else:
            logger.info(
                f"Using core-based measurement: {refined_length:.1f} (seed was {seed.length:.1f})"
            )
    else:
        # No core measurements - fall back to stricter seed validation
        if refined_length < seed.length * 0.7:
            logger.info(f"Length too small ({refined_length:.1f}), using 0.9x seed")
            refined_length = seed.length * 0.9
        elif refined_length > seed.length * 1.8:
            logger.info(f"Length too large ({refined_length:.1f}), using 1.3x seed")
            refined_length = seed.length * 1.3
        else:
            logger.info(f"Using measured length: {refined_length:.1f}")

    # --- FINAL length & width: 1D-collapse FWHM (robust to streak break-up) ---
    # Collapse the rotated streak onto each axis and take the extent between the
    # outermost crossings of 0.5 x a *stable* peak. A long trail can dip below
    # 0.5xmax mid-streak (tracking jitter, optics) and fragment into blobs, which
    # makes 2D connected-component length under-measure; the 1D extent spans the
    # whole trail because the outer points stay above the half level. The stable
    # peak (95th pct of the smoothed profile) keeps a lone hot pixel from setting
    # the scale.
    from scipy.ndimage import gaussian_filter1d as _g1d
    from scipy.ndimage import median_filter as _medfilt

    def _stable_half_extent(prof: np.ndarray, frac: float = 0.5) -> float:
        if prof.size == 0:
            return 0.0
        # Median filter rejects isolated hot-pixel columns (which would set a
        # spuriously high peak and exclude the real streak); then light smooth.
        sm = _g1d(_medfilt(prof.astype(float), size=5), sigma=1.5)
        peak = float(np.max(sm))
        if peak <= 0:
            return 0.0
        idx = np.where(sm >= frac * peak)[0]
        return float(idx[-1] - idx[0]) if idx.size >= 2 else 0.0

    # Hot-pixel-robust center: argmax of a median-filtered PSF (the global argmax
    # can be a hot pixel, which would put the measurement band off the streak).
    _clean = _medfilt(rotated_psf, size=3)
    cprof_y, cprof_x = np.unravel_index(int(np.argmax(_clean)), _clean.shape)

    # Length from the along-streak profile (max over a ~seeing perpendicular band).
    _lband = int(max(4, round((seed.fwhm or 4.0) * 2.0)))
    _ly0 = max(0, cprof_y - _lband)
    _ly1 = min(rotated_psf.shape[0], cprof_y + _lband + 1)
    along_profile = np.max(rotated_psf[_ly0:_ly1, :], axis=0)
    fwhm_length = _stable_half_extent(along_profile)
    if fwhm_length > 0:
        refined_length = fwhm_length
    else:
        logger.warning("1D FWHM length failed; keeping %.1f", refined_length)

    # Width from the perpendicular profile (max over an along band covering the trail).
    _hl = int(max(4, round(refined_length / 2.0)))
    _wx0 = max(0, cprof_x - _hl)
    _wx1 = min(rotated_psf.shape[1], cprof_x + _hl + 1)
    perp_profile = np.max(rotated_psf[:, _wx0:_wx1], axis=1)
    fwhm_width = _stable_half_extent(perp_profile)
    refined_width = fwhm_width if fwhm_width > 0 else (seed.fwhm or 4.0)

    # Clamp width to a sane seeing-scale bound: a streak's perpendicular profile
    # is the PSF (≈ seeing FWHM), never tens of pixels. Guards against a residual
    # fat/clipped stack inflating the width (the seed FWHM tracks the seeing once
    # saturated candidates are rejected upstream).
    max_width = 2.5 * (seed.fwhm or 4.0)
    if refined_width > max_width:
        logger.info(
            "Clamping streak width %.1f -> %.1f (> 2.5x seed FWHM %.1f)",
            refined_width, max_width, seed.fwhm or 4.0,
        )
        refined_width = max_width

    logger.info(
        f"Refined (1D FWHM @0.5xstable-max): L={refined_length:.1f} "
        f"(seed {seed.length:.1f}), W={refined_width:.1f} (seed {seed.fwhm:.1f})"
    )

    refined_measurement = StreakMeasurement(
        rotation=seed.rotation,  # Keep rotation from seed
        length=refined_length,
        fwhm=refined_width,
    )

    # Plot the PSF/refinement diagnostic. Gated on the dedicated `streak` flag
    # (small, ~<1MB) OR the broad `debug` flag (which also emits the heavy
    # kernel/CC plots) — so these can be kept on inline without the brutal ones.
    _plt_cfg = getattr(get_config(), "plotting", None)
    if _plt_cfg is not None and (_plt_cfg.streak or _plt_cfg.debug):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot rotated PSF with threshold contours
        im0 = axes[0].imshow(rotated_psf, origin="lower", cmap="viridis")
        axes[0].set_title(f"Rotated PSF (θ={seed.rotation:.1f}°)")

        # Show core thresholds in red
        for thresh in core_thresholds:
            axes[0].contour(
                rotated_psf, levels=[thresh], colors="red", linewidths=1, alpha=0.5
            )

        # Show wing threshold in cyan
        axes[0].contour(
            rotated_psf, levels=[wing_threshold], colors="cyan", linewidths=2, alpha=0.8
        )

        plt.colorbar(im0, ax=axes[0], label="Normalized Intensity")

        # Plot measurements showing core vs full length
        if core_lengths:
            thresholds_used = [t for t, _ in core_lengths]
            lengths_core = [l for _, l in core_lengths]
            axes[1].plot(
                thresholds_used,
                lengths_core,
                "o-",
                label="Core Length (threshold-limited)",
                markersize=8,
                color="orange",
            )

        axes[1].axhline(
            seed.length, color="r", linestyle="--", label="Seed Length", linewidth=2
        )
        axes[1].axhline(
            flood_fill_length,
            color="cyan",
            linestyle=":",
            label=f"Flood Fill (thresh={wing_threshold})",
            linewidth=2,
            alpha=0.7,
        )
        axes[1].axhline(
            refined_length,
            color="g",
            linestyle="-",
            label="Final Refined Length",
            linewidth=2,
        )
        if profile_extent_length and profile_extent_length > 0:
            axes[1].axhline(
                profile_extent_length,
                color="purple",
                linestyle="-.",
                label="Profile Extent Length",
                linewidth=2,
                alpha=0.8,
            )
        axes[1].set_xlabel("Intensity Threshold", fontsize=12)
        axes[1].set_ylabel("Length (pixels)", fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title(f"Length: {seed.length:.1f} → {refined_length:.1f} px")

        plt.tight_layout()

        # Include frame index in filename if provided
        if frame_index is not None:
            filename = f"streak_psf_refinement_frame_{frame_index:04d}.png"
        else:
            filename = "streak_psf_refinement.png"

        output_path = get_config().runtime.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Plotted PSF refinement to {output_path}")

    return refined_measurement, refined_width


def refine_streak_len(
    psf: np.array,
    pixel_fwhm: float | None,
    rotation: float,
    half_max_value: float = 0.55,
) -> float:
    """refine streak length in cases of noisy PSF by looking along streak angle

    Args:
        psf (np.array): PSF
        pixel_fwhm (float): estimate of pixel FWHM
        rotation (float): streak rotation
        half_max_value (float, optional): value of PSF (0-1) at which to cut for length. default 0.55

    Returns:
        float: refined estimate of streak length
    """
    rotated_psf = rotate(psf, angle=rotation, mode="constant", cval=np.min(psf))
    rotated_psf /= np.max(rotated_psf)

    if pixel_fwhm is None:
        pixel_fwhm = streak_fwhm_from_cutout(rotated_psf, 0)

    start_point = np.unravel_index(np.argmax(rotated_psf), rotated_psf.shape)
    cutout = rotated_psf[
        int(start_point[0] - pixel_fwhm) : int(start_point[0] + pixel_fwhm), :
    ]

    valids = np.where(np.max(cutout, 0) > half_max_value)

    # calculate length and return
    return np.max(valids) - np.min(valids)


def extract_streak_dims_mapping(
    data: np.ndarray,
    n_streaks: int = 5,
) -> tuple[StreakMeasurement, np.ndarray]:
    # Make a copy of the data for streak detection
    data_mapped = data.copy()
    # Make another copy for outlier removal only
    data_outliers_removed = data.copy()

    # Lists to store streak metadata
    all_streaks = []

    # First pass: collect all potential streaks
    max_iterations = n_streaks * 5  # Limit iterations to avoid infinite loops
    iteration = 0

    fill_min = np.median(data_mapped) + 0.5 * np.std(data_mapped)

    while iteration < max_iterations and len(all_streaks) < n_streaks * 2:
        iteration += 1

        # Find the brightest point
        if np.max(data_mapped) <= fill_min:
            break  # No more significant streaks to find

        x_max, y_max = np.unravel_index(np.argmax(data_mapped), data_mapped.shape)

        # Extract the streak and get its metadata
        data_mapped, metadata = remove_streak_at_point_enriched(
            data_mapped, (x_max, y_max), fill_min
        )

        # Skip streaks with very few pixels (likely noise)
        if metadata["num_pixels"] < 5:
            continue

        # Log FWHM-based measurement details for debugging
        if "fwhm_threshold" in metadata and "fwhm_pixels" in metadata:
            logger.debug(
                f"Streak {len(all_streaks) + 1}: fill_min={fill_min:.2f}, "
                f"fwhm_threshold={metadata['fwhm_threshold']:.2f}, "
                f"total_pixels={metadata['num_pixels']}, "
                f"fwhm_pixels={metadata['fwhm_pixels']}, "
                f"length={metadata['length']:.1f}"
            )

        all_streaks.append(metadata)

    # Filter out round objects before clustering
    filtered_streaks = []
    outlier_streaks = []

    for streak in all_streaks:
        # Calculate roundness ratio
        roundness_ratio = (
            streak["fwhm_major"] / streak["fwhm_minor"]
            if streak["fwhm_minor"] > 0
            else float("inf")
        )

        # Keep only elongated objects (ratio > 1.3)
        if roundness_ratio >= 1.3:
            filtered_streaks.append(streak)
        else:
            outlier_streaks.append(streak)

    # If we have no elongated streaks, fall back to using all streaks
    if not filtered_streaks:
        logger.warning("No elongated streaks found, using all detected objects")
        filtered_streaks = all_streaks
        outlier_streaks = []

    # Find the most consistent group using a clustering approach

    # Calculate statistics for lengths and orientations using only filtered streaks
    lengths = np.array([s["length"] for s in filtered_streaks])
    orientations = np.array([s["orientation"] for s in filtered_streaks])

    # Handle orientation wrapping (e.g., 179° and 1° are close)
    # Convert orientations to be within -90 to 90 degrees
    orientations = orientations % 180
    orientations = np.where(orientations > 90, orientations - 180, orientations)

    # Calculate standard deviations for clustering thresholds
    length_mean, length_std = np.mean(lengths), np.std(lengths) or 1.0
    orient_mean, orient_std = (
        np.mean(orientations),
        np.std(orientations) or 10.0,
    )  # Default to 10° if std is 0

    # Create a similarity matrix that combines both length and orientation
    n_streaks = len(filtered_streaks)
    similarity_matrix = np.zeros((n_streaks, n_streaks))

    for i in range(n_streaks):
        for j in range(n_streaks):
            if i == j:
                similarity_matrix[i, j] = 1.0  # Perfect similarity with self
                continue

            # Length similarity (normalized by standard deviation)
            length_diff = abs(lengths[i] - lengths[j]) / length_std

            # Orientation similarity (handle circular nature)
            orient_diff = (
                min(
                    abs(orientations[i] - orientations[j]),
                    abs(orientations[i] - orientations[j] + 180),
                    abs(orientations[i] - orientations[j] - 180),
                )
                / orient_std
            )

            # Combined similarity score (higher is more similar)
            # Weight orientation more heavily than length (3:1 ratio)
            similarity = 1.0 / (1.0 + length_diff + 3.0 * orient_diff)
            similarity_matrix[i, j] = similarity

    # For each streak, count how many similar streaks it has
    # A streak is considered similar if similarity > threshold
    similarity_threshold = 0.5
    similar_counts = np.sum(similarity_matrix > similarity_threshold, axis=1)

    # Find all potential seed points (streaks with many similar neighbors)
    potential_seeds = np.where(similar_counts >= 2)[0]

    if len(potential_seeds) > 0:
        # For each potential seed, evaluate the quality of its cluster
        best_cluster = None
        best_cluster_score = -1

        for seed in potential_seeds:
            # Get all streaks similar to this seed
            similar_to_seed = np.where(similarity_matrix[seed] > similarity_threshold)[
                0
            ]

            # Calculate the average pairwise similarity within this cluster
            cluster_similarities = []
            for i in range(len(similar_to_seed)):
                for j in range(i + 1, len(similar_to_seed)):
                    cluster_similarities.append(
                        similarity_matrix[similar_to_seed[i], similar_to_seed[j]]
                    )

            # Cluster score is the product of size and average similarity
            if cluster_similarities:
                avg_similarity = np.mean(cluster_similarities)
                cluster_score = len(similar_to_seed) * avg_similarity

                if cluster_score > best_cluster_score:
                    best_cluster_score = cluster_score
                    best_cluster = similar_to_seed

        if best_cluster is not None:
            # Create masks for characteristic and outlier streaks
            characteristic_mask = np.zeros(n_streaks, dtype=bool)
            characteristic_mask[best_cluster] = True

            # Select streaks
            characteristic_streaks = [
                s for i, s in enumerate(filtered_streaks) if characteristic_mask[i]
            ]
            # We already have outlier_streaks from the roundness filtering
            # Add any filtered streaks that weren't in the best cluster
            outlier_streaks.extend(
                [
                    s
                    for i, s in enumerate(filtered_streaks)
                    if not characteristic_mask[i]
                ]
            )
        else:
            # Fallback if no good cluster found
            characteristic_streaks = filtered_streaks
            # outlier_streaks already contains the round objects
    else:
        # Fallback if no potential seeds found
        characteristic_streaks = filtered_streaks
        outlier_streaks = []

    # Final validation: check if the characteristic streaks are actually consistent
    if len(characteristic_streaks) >= 2:
        char_lengths = np.array([s["length"] for s in characteristic_streaks])
        char_orients = np.array([s["orientation"] for s in characteristic_streaks])

        # Calculate coefficient of variation for length
        length_cv = np.std(char_lengths) / np.mean(char_lengths)

        # Calculate circular standard deviation for orientation
        orient_diffs = []
        for i in range(len(char_orients)):
            for j in range(i + 1, len(char_orients)):
                diff = min(
                    abs(char_orients[i] - char_orients[j]),
                    abs(char_orients[i] - char_orients[j] + 180),
                    abs(char_orients[i] - char_orients[j] - 180),
                )
                orient_diffs.append(diff)

        orient_std_dev = np.std(orient_diffs) if orient_diffs else 0

        # If the characteristic streaks are not consistent, try a different approach
        if length_cv > 0.2 or orient_std_dev > 15.0:
            # Look for the most consistent group in the original streaks
            # Group streaks by similar length
            length_groups = {}
            for i, length in enumerate(lengths):
                length_key = round(length / (length_std * 0.5)) * (length_std * 0.5)
                if length_key not in length_groups:
                    length_groups[length_key] = []
                length_groups[length_key].append(i)

            # Find the largest length group
            largest_group = max(length_groups.values(), key=len)

            # If this group has multiple streaks, check if orientations are consistent
            if len(largest_group) >= 2:
                group_orients = orientations[largest_group]

                # Group by orientation within this length group
                orient_groups = {}
                for i, orient in enumerate(group_orients):
                    orient_key = round(orient / 10) * 10  # Group by 10-degree bins
                    if orient_key not in orient_groups:
                        orient_groups[orient_key] = []
                    orient_groups[orient_key].append(largest_group[i])

                # Find the largest orientation group
                largest_orient_group = max(orient_groups.values(), key=len)

                # If we found a consistent group, use it
                if len(largest_orient_group) >= 2:
                    characteristic_mask = np.zeros(n_streaks, dtype=bool)
                    characteristic_mask[largest_orient_group] = True

                    characteristic_streaks = [
                        s
                        for i, s in enumerate(filtered_streaks)
                        if characteristic_mask[i]
                    ]
                    outlier_streaks = [
                        s
                        for i, s in enumerate(filtered_streaks)
                        if not characteristic_mask[i]
                    ]

    # Remove only round objects from the second copy of the data
    # These are the objects we filtered out earlier based on roundness ratio
    round_objects = [
        streak
        for streak in all_streaks
        if (
            streak["fwhm_major"] / streak["fwhm_minor"]
            if streak["fwhm_minor"] > 0
            else float("inf")
        )
        < 1.4
    ]

    for round_obj in round_objects:
        center = np.array(round_obj["center"]).astype(int)
        data_outliers_removed = remove_streak_at_point(
            data_outliers_removed, center, fill_min
        )

    # Limit to exactly n_streaks if we have more
    characteristic_streaks = characteristic_streaks[:n_streaks]

    streak_measurement = StreakMeasurement(
        rotation=np.median([s["orientation"] for s in characteristic_streaks]),
        length=np.median([s["length"] for s in characteristic_streaks]),
        fwhm=np.median([s["fwhm_minor"] for s in characteristic_streaks]),
    )
    print(streak_measurement)

    # Create and return the StreakMeasurement object

    return streak_measurement, data_outliers_removed


def refine_streak_length_by_overhang(
    data: np.ndarray,
    streak_mask: np.ndarray,
    initial_length: float,
    rotation: float,
    fwhm: float,
) -> float:
    """
    Refine streak length by detecting overhang at the ends.
    Looks for sharp drops in intensity that indicate the streak extends beyond the data.

    Args:
        data: Original image data
        streak_mask: Boolean mask of the detected streak
        initial_length: Initial length estimate
        rotation: Streak rotation in degrees
        fwhm: FWHM of the PSF

    Returns:
        float: Refined length estimate
    """
    # Get coordinates of streak pixels
    y_coords, x_coords = np.where(streak_mask)

    if len(y_coords) < 10:
        return initial_length

    # Calculate streak center
    center_y, center_x = np.mean(y_coords), np.mean(x_coords)

    # Create unit vector along streak direction
    angle_rad = np.deg2rad(rotation)
    streak_direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])

    # Project all streak points onto the streak direction
    points = np.column_stack([x_coords - center_x, y_coords - center_y])
    projections = np.dot(points, streak_direction)

    # Sort projections to get ordered points along streak
    sorted_indices = np.argsort(projections)
    sorted_projections = projections[sorted_indices]
    sorted_x = x_coords[sorted_indices]
    sorted_y = y_coords[sorted_indices]

    # Calculate intensity profile along streak
    intensities = data[sorted_y, sorted_x]

    # Smooth the intensity profile to reduce noise
    from scipy.ndimage import gaussian_filter1d

    smoothed_intensities = gaussian_filter1d(intensities, sigma=1.0)

    # Find the peak intensity
    peak_idx = np.argmax(smoothed_intensities)
    peak_intensity = smoothed_intensities[peak_idx]

    # Calculate threshold for significant drop (e.g., 50% of peak)
    drop_threshold = peak_intensity * 0.5

    # Look for drops from both ends
    left_end = 0
    right_end = len(sorted_projections) - 1

    # Find left end (start from peak and go left)
    for i in range(peak_idx, 0, -1):
        if smoothed_intensities[i] < drop_threshold:
            left_end = i
            break

    # Find right end (start from peak and go right)
    for i in range(peak_idx, len(smoothed_intensities)):
        if smoothed_intensities[i] < drop_threshold:
            right_end = i
            break

    # Calculate refined length
    refined_length = sorted_projections[right_end] - sorted_projections[left_end]

    # Ensure minimum length
    min_length = fwhm * 2
    refined_length = max(refined_length, min_length)

    # Sanity check: don't let it get too much shorter
    if refined_length < initial_length * 0.5:
        logger.warning(
            f"Refined length ({refined_length:.1f}) too much shorter than initial ({initial_length:.1f})"
        )
        refined_length = (
            initial_length * 0.8
        )  # Use 80% of initial as conservative estimate

    logger.info(
        f"Length refinement: {initial_length:.1f} -> {refined_length:.1f} pixels"
    )

    return refined_length


def extract_streak_dims_simple_long(
    data: np.ndarray,
    length: float = None,
    rotation: float = None,
    fwhm: float = 4.0,
) -> tuple[StreakMeasurement, np.ndarray, float]:
    """
    Extract streak dimensions for very long streaks using a simple threshold-based approach.
    This method is optimized for streaks longer than ~100 pixels.

    Args:
        data: Input image data
        length: Initial estimate of streak length in pixels (for guidance)
        rotation: Initial estimate of streak rotation in degrees (for guidance)
        fwhm: Estimated FWHM of the PSF in pixels

    Returns:
        tuple: (StreakMeasurement, psf, measured_fwhm)
    """
    logger.info(
        f"Using simple long streak extraction method (est: len={length:.1f}, rot={rotation:.1f}°)"
    )

    # Create a working copy
    working_data = data.copy()

    # Calculate background statistics
    bg_median = np.median(working_data)
    bg_std = np.std(working_data[working_data < np.percentile(working_data, 80)])

    # Use a conservative threshold for long streaks
    # Long streaks are usually bright and well-defined
    threshold = bg_median + 2.0 * bg_std

    # Find the brightest point as starting point
    y_max, x_max = np.unravel_index(np.argmax(working_data), working_data.shape)

    # Use flood fill to map the streak
    streak_mask = map_cluster(working_data, (y_max, x_max), threshold)

    # Get coordinates of all points in the streak
    y_coords, x_coords = np.where(streak_mask)

    if len(y_coords) < 10:  # Need minimum points
        logger.warning("Not enough pixels in streak mask")
        return None, None, fwhm

    # Use PCA to measure the streak properties
    points = np.column_stack([y_coords, x_coords])
    points_centered = points - np.mean(points, axis=0)

    try:
        cov = np.cov(points_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Calculate orientation
        streak_angle = np.degrees(np.arctan2(eigenvectors[0, 0], eigenvectors[1, 0]))
        streak_angle = streak_angle % 180

        # Calculate initial length using eigenvalues
        # For long streaks, use a more conservative factor
        initial_length = 3 * np.sqrt(eigenvalues[0])  # 3 sigma instead of 4

        # Calculate width
        measured_width = 3 * np.sqrt(eigenvalues[1])

        # Calculate aspect ratio
        aspect_ratio = (
            initial_length / measured_width if measured_width > 0 else float("inf")
        )

        logger.info(
            f"Initial measurement: length={initial_length:.1f}, angle={streak_angle:.1f}°, aspect_ratio={aspect_ratio:.1f}"
        )

        # Sanity checks
        if aspect_ratio < 2.0:
            logger.warning(
                f"Low aspect ratio ({aspect_ratio:.1f}) - may not be a streak"
            )
            return None, None, fwhm

        if initial_length < fwhm * 2:
            logger.warning(f"Initial length ({initial_length:.1f}) too short")
            return None, None, fwhm

        # Refine length by checking for overhang
        refined_length = refine_streak_length_by_overhang(
            data, streak_mask, initial_length, streak_angle, measured_width
        )

        # Create a simple PSF from the refined parameters
        psf = rectangle_pyramoid(
            refined_length,
            np.sin(np.deg2rad(streak_angle)),
            np.cos(np.deg2rad(streak_angle)),
            int(measured_width),
            upsample=50,  # Lower upsample for speed
            halo_fwhm=2,
            halo_level=0,
        )

        # Normalize PSF
        psf -= np.min(psf)
        psf /= np.max(psf) if np.max(psf) > 0 else 1.0

        streak_measurement = StreakMeasurement(
            rotation=streak_angle,
            length=refined_length,
            fwhm=measured_width,
        )

        return streak_measurement, psf, measured_width

    except np.linalg.LinAlgError:
        logger.warning("PCA failed for streak measurement")
        return None, None, fwhm


def _estimate_streak_seed(
    data: np.ndarray, fwhm: float = 4.0, crop: int = 2048
) -> tuple[float, float]:
    """Coarse, header-free (length, rotation) seed for the anchor frame.

    Found by a normalized matched-filter scale-space search on a central crop:
    a unit-L2 streak kernel is correlated with the image and the (length,
    angle) with the strongest peak response wins (the normalization makes
    responses comparable across kernel sizes, so the best match is the true
    streak scale). The search is two-stage rather than an exhaustive grid —
    angle first at a mid-scale length, then length at the winning angle. An
    elongated kernel's angle response peaks at the true angle at any
    plausible length, so the stages decouple (verified identical to the full
    7x12 grid on real 8k frames at ~1/4 the trials). Trials run in
    float32: this is peak-picking, not photometry. Replaces the old
    ``min(shape)*0.05`` default, which on a large frame was ~10x too long
    and produced a smeared, oversized PSF. This is only a seed — the robust
    extractor refines it.
    """
    try:
        from scipy.signal import fftconvolve

        h, w = data.shape
        s = min(crop, h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        c = data[y0:y0 + s, x0:x0 + s].astype(np.float32)
        c -= np.median(c)

        def response(length: float, ang: float) -> float:
            k = rectangle_pyramoid(
                length, np.sin(np.deg2rad(ang)), np.cos(np.deg2rad(ang)),
                int(fwhm * 2), upsample=1, halo_fwhm=4, halo_level=0,
            )
            norm = float(np.sqrt(np.sum(k * k)))
            if norm <= 0:
                return -np.inf
            kn = (k / norm).astype(np.float32)
            return float(np.max(fftconvolve(c, kn, mode="same")))

        lengths = (12, 18, 27, 40, 60, 90, 135)
        mid_length = 40

        best_rot, best_resp = None, -np.inf
        for ang in range(0, 180, 15):
            resp = response(mid_length, ang)
            if resp > best_resp:
                best_rot, best_resp = float(ang), resp
        if best_rot is None or not np.isfinite(best_resp):
            raise ValueError("no matched-filter response")

        best_len, best_resp = None, -np.inf
        for length in lengths:
            resp = response(length, best_rot)
            if resp > best_resp:
                best_len, best_resp = float(length), resp
        if best_len is None:
            raise ValueError("no matched-filter response")
        logger.info(
            "Multi-scale seed estimate: L=%.0f px, rot=%.0f deg", best_len, best_rot
        )
        return best_len, best_rot
    except Exception as e:
        logger.warning("Multi-scale seed estimate failed (%s); using 40 px fallback", e)
        return 40.0, 0.0


def extract_streak_dims_robust(
    data: np.ndarray,
    n_streaks: int = 10,
    length: float = None,
    rotation: float = None,
    fwhm: float | None = None,
) -> tuple[StreakMeasurement, np.ndarray, float]:
    """
    Extract streak dimensions using a robust approach that combines matched filtering,
    morphological analysis, and statistical validation.

    Args:
        data: Input image data
        n_streaks: Maximum number of streaks to extract
        length: Initial estimate of streak length in pixels
        rotation: Initial estimate of streak rotation in degrees
        fwhm: Estimated FWHM of the PSF in pixels (if None, will be measured)

    Returns:
        tuple: (rotation, length, psf, measured_fwhm)
    """
    # Only fall back to the simple long-streak extractor for *very* long streaks.
    # The previous 150px threshold split a single set across two extractors
    # (e.g. a calsat's frames at 102/126/167px used robust/robust/simple_long and
    # gave inconsistent results). The robust path now rejects saturated
    # candidates and clamps width, so it handles normal streaks consistently;
    # reserve simple_long for genuinely huge trails.
    if length is not None and length > 400:
        logger.info(
            f"Streak length ({length:.1f} px) exceeds threshold, using simple long streak extraction method"
        )
        return extract_streak_dims_simple_long(data, length, rotation, fwhm or 4.0)

    # If FWHM is not provided, make an initial estimate
    if fwhm is None:
        # Start with a reasonable default
        fwhm = 4.0
        logger.info(f"No FWHM provided, using initial estimate of {fwhm:.1f} pixels")

    length_str = f"{length:.1f}" if length is not None else "None"
    rotation_str = f"{rotation:.1f}°" if rotation is not None else "None"
    logger.info(
        f"Extracting streak parameters using robust method (initial est: len={length_str}, rot={rotation_str}, fwhm={fwhm:.1f})"
    )

    # Step 1: Create a clean working copy of the data
    working_data = data.copy()
    # Keep an UNMODIFIED copy for cutout extraction (so masking doesn't corrupt the PSF)
    original_data = data.copy()

    # Background statistics for thresholding
    bg_median = np.median(working_data)
    bg_std = np.std(working_data[working_data < np.percentile(working_data, 80)])

    # Step 2: Create matched filter kernel based on initial estimates
    # Use defaults if not provided - estimate from image size for length, 0 for rotation
    if length is None or rotation is None:
        # No prior (anchor frame): estimate a coarse (length, rotation) seed
        # from the image via a matched-filter scale-space search, rather than
        # the old min(shape)*0.05 default (~10x too long on large frames →
        # oversized smeared PSF).
        est_len, est_rot = _estimate_streak_seed(working_data, fwhm)
        if length is None:
            length = est_len
        if rotation is None:
            rotation = est_rot

    kernel = rectangle_pyramoid(
        length,
        np.sin(np.deg2rad(rotation)),
        np.cos(np.deg2rad(rotation)),
        int(fwhm * 2),
        upsample=100,
        halo_fwhm=4,
        halo_level=0,
    )

    mask_kernel = rectangle_pyramoid(
        length * 1.2,
        np.sin(np.deg2rad(rotation)),
        np.cos(np.deg2rad(rotation)),
        int(fwhm * 2.2),
        upsample=100,
        halo_fwhm=4,
        halo_level=0,
    )

    # Step 3: Apply matched filter
    from senpai.engine.utils.stats import fft_workers

    with fft_workers():
        filtered_data = convolve(working_data, kernel, mode="same")

    # Step 4: Clean up borders to avoid edge artifacts
    border_width = max(10, int(length * 0.5))
    filtered_data[:border_width, :] = bg_median
    filtered_data[-border_width:, :] = bg_median
    filtered_data[:, :border_width] = bg_median
    filtered_data[:, -border_width:] = bg_median

    # Step 5: Extract streak candidates
    streak_candidates = []
    streak_metrics = []

    # Size of cutout region (make it generously larger than expected streak)
    cutout_size = int(max(length * 1.3, fwhm * 10))

    # Minimum distance between streak centers to avoid duplicates
    min_distance = max(int(length * 0.5), 10)

    # Keep track of already processed regions
    processed_mask = np.zeros_like(filtered_data, dtype=bool)

    # Precompute the self-convolved mask kernel + background once. `_mask` marks a
    # picked region's area of influence directly into the matched-filter response
    # (-inf) so the next argmax skips it — no full-frame copy or boolean scan per
    # iteration (only the small bounded region is touched). Mutates in place.
    eff_kernel = streak_mask_effective_kernel(mask_kernel)

    def _mask(yy, xx):
        mask_streak_region(
            processed_mask, working_data, yy, xx, mask_kernel,
            effective_kernel=eff_kernel, bg_value=bg_median, response=filtered_data,
        )

    for _ in range(min(30, n_streaks * 3)):  # Try more candidates than needed
        if np.all(
            processed_mask[border_width:-border_width, border_width:-border_width]
        ):
            break  # Stop if we've processed all valid regions

        # Next-brightest unprocessed peak — already-masked regions are -inf in the
        # response (set by _mask), so no per-iteration full-frame masking needed.
        y_max, x_max = np.unravel_index(np.argmax(filtered_data), filtered_data.shape)

        # Skip if too close to edge for full cutout
        if (
            y_max < cutout_size
            or y_max >= working_data.shape[0] - cutout_size
            or x_max < cutout_size
            or x_max >= working_data.shape[1] - cutout_size
        ):
            # Mark this region as processed and mask the working data
            _mask(y_max, x_max)
            continue

        # Extract cutout from ORIGINAL unmodified data (not the progressively-masked working_data)
        # This prevents artificial breaks from previous mask operations
        cutout = original_data[
            y_max - cutout_size : y_max + cutout_size,
            x_max - cutout_size : x_max + cutout_size,
        ].copy()

        # Check if the cutout overlaps with previously masked regions
        if not is_valid_psf(cutout, processed_mask, y_max, x_max, cutout_size):
            logger.debug(
                f"Rejecting PSF at ({y_max}, {x_max}) due to overlap with masked regions"
            )
            # Mark this region as processed and mask the working data
            _mask(y_max, x_max)
            continue

        # Calculate SNR and other metrics
        local_bg = np.median(cutout)
        local_noise = np.std(cutout[cutout < local_bg + 2 * bg_std])
        peak_value = np.max(cutout)
        snr = (peak_value - local_bg) / local_noise if local_noise > 0 else 0

        # Skip low SNR detections
        if snr < 3.0:
            # Mark this region as processed and mask the working data
            _mask(y_max, x_max)
            continue

        # Normalize cutout for analysis
        norm_cutout = cutout.copy()
        norm_cutout -= local_bg
        norm_cutout = np.clip(norm_cutout, 0, None)  # Remove negative values
        norm_cutout /= np.max(norm_cutout) if np.max(norm_cutout) > 0 else 1.0

        # Reject saturated/clipped streaks (background-independent): a clipped
        # star is flat-topped — most of its core sits at the clip level — so a
        # large fraction of its bright pixels pin near the peak. Stacking those
        # makes the PSF a fat uniform block with a runaway width. A real
        # (unsaturated) streak has a Gaussian cross-section, so only its thin
        # ridge approaches the peak. (Note: the upstream saturation cut compares
        # to 0.9*full_well, which misses clips here because the row/col-median
        # subtraction pulls 65535 down to ~57000 — below that threshold.)
        core = norm_cutout > 0.5
        if np.any(core):
            plateau_frac = float(np.mean(norm_cutout[core] >= 0.95))
            if plateau_frac > 0.5:
                logger.debug(
                    "Rejecting clipped/saturated candidate at (%d,%d): "
                    "plateau_frac=%.2f", y_max, x_max, plateau_frac,
                )
                _mask(y_max, x_max)
                continue

        # Analyze shape using connected components
        binary_cutout = norm_cutout > 0.3  # Threshold at 30% of peak

        # Skip if no pixels above threshold
        if not np.any(binary_cutout):
            # Mark this region as processed and mask the working data
            _mask(y_max, x_max)
            continue

        # Find connected component containing peak
        from scipy import ndimage

        labeled, num_features = ndimage.label(binary_cutout)
        peak_y, peak_x = np.unravel_index(np.argmax(norm_cutout), norm_cutout.shape)
        peak_label = labeled[peak_y, peak_x]

        if peak_label == 0:  # No label at peak (shouldn't happen)
            # Mark this region as processed and mask the working data
            _mask(y_max, x_max)
            continue

        # Extract just the connected component containing the peak
        component_mask = labeled == peak_label

        # Calculate shape metrics
        y_indices, x_indices = np.where(component_mask)

        # Skip if too few pixels
        if len(y_indices) < 5:
            # Mark this region as processed and mask the working data
            _mask(y_max, x_max)
            continue

        # Calculate covariance matrix for shape analysis
        points = np.column_stack([y_indices, x_indices])
        points_centered = points - np.mean(points, axis=0)
        cov = np.cov(points_centered, rowvar=False)

        # Get eigenvalues and eigenvectors
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Sort in descending order
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Calculate aspect ratio
            aspect_ratio = (
                np.sqrt(eigenvalues[0] / eigenvalues[1]) if eigenvalues[1] > 0 else 10.0
            )

            # Calculate orientation
            streak_angle = np.degrees(
                np.arctan2(eigenvectors[0, 0], eigenvectors[1, 0])
            )
            streak_angle = streak_angle % 180

            # Calculate length using eigenvalues
            # The length is approximately 4 * sqrt(largest eigenvalue)
            # This corresponds to ~95% of the mass of a Gaussian distribution
            component_length = 4 * np.sqrt(eigenvalues[0])

            # Calculate width similarly
            component_width = 4 * np.sqrt(eigenvalues[1])

            # Calculate angle difference from expected
            angle_diff = min(
                abs(streak_angle - rotation),
                abs(streak_angle - rotation + 180),
                abs(streak_angle - rotation - 180),
            )

            # Calculate length difference from expected
            length_ratio = component_length / length if length > 0 else 1.0

            # Skip if aspect ratio is too low (not streak-like)
            if aspect_ratio < 1.5:
                # Mark this region as processed and mask the working data
                _mask(y_max, x_max)
                continue

            # Calculate overall quality score
            # Higher for: high SNR, high aspect ratio, angle close to expected, length close to expected
            quality_score = (
                snr
                * aspect_ratio
                * (1.0 / (1.0 + angle_diff / 10))
                * (1.0 / (1.0 + abs(length_ratio - 1.0)))
            )

            # Store candidate and metrics
            streak_candidates.append(norm_cutout)
            streak_metrics.append(
                {
                    "snr": snr,
                    "aspect_ratio": aspect_ratio,
                    "angle": streak_angle,
                    "length": component_length,
                    "width": component_width,
                    "quality_score": quality_score,
                }
            )

            logger.debug(
                f"Found streak candidate: SNR={snr:.1f}, AR={aspect_ratio:.1f}, "
                f"angle={streak_angle:.1f}°, length={component_length:.1f}, "
                f"score={quality_score:.1f}"
            )

        except np.linalg.LinAlgError:
            # Skip if eigenvalue decomposition fails
            pass

        # After processing, mark this region as processed and mask the working data
        _mask(y_max, x_max)

    # Step 6: Select best candidates and create PSF
    if not streak_candidates:
        logger.warning("No valid streak candidates found")
        return None, None, fwhm

    # Sort by quality score
    sorted_indices = np.argsort([m["quality_score"] for m in streak_metrics])[::-1]

    # Take top n candidates (or all if fewer)
    top_n = min(n_streaks, len(streak_candidates))
    selected_indices = sorted_indices[:top_n]

    selected_streaks = [streak_candidates[i] for i in selected_indices]
    selected_metrics = [streak_metrics[i] for i in selected_indices]

    logger.info(
        f"Selected {top_n} best streak candidates out of {len(streak_candidates)}"
    )

    # Step 7: Align streaks before stacking
    aligned_streaks = []
    for i, streak in enumerate(selected_streaks):
        # Rotate to align with horizontal axis
        angle_to_horizontal = selected_metrics[i]["angle"] - 90
        aligned = rotate(
            streak, angle_to_horizontal, reshape=False, mode="constant", cval=0
        )
        aligned_streaks.append(aligned)

    # Step 8: Create PSF by stacking aligned streaks
    psf = np.median(np.stack(aligned_streaks), axis=0)

    # Rotate back to original orientation
    psf = rotate(
        psf,
        90 - np.median([m["angle"] for m in selected_metrics]),
        reshape=False,
        mode="constant",
        cval=0,
    )

    # Normalize PSF
    psf -= np.min(psf)
    psf /= np.max(psf) if np.max(psf) > 0 else 1.0

    # Step 9: Calculate final streak parameters
    # Use median of individual measurements for robustness
    raw_length = np.median([m["length"] for m in selected_metrics])

    # Ensure the corrected length is not negative or too small
    final_length = max(raw_length, fwhm * 0.5)

    final_angle = np.median([m["angle"] for m in selected_metrics])

    # Log both raw and corrected measurements
    logger.info(
        f"Raw streak length: {raw_length:.1f}, corrected to: {final_length:.1f} after PSF subtraction"
    )

    # Sanity check on length
    if final_length < fwhm:
        logger.warning(
            f"Corrected length ({final_length:.1f}) is smaller than FWHM ({fwhm:.1f})"
        )
        # Fall back to original estimate if available and reasonable
        if length is not None and length > final_length:
            logger.info(f"Using original length estimate: {length:.1f}")
            final_length = length

    # Sanity check on angle
    angle_diffs = [
        min(
            abs(m["angle"] - rotation),
            abs(m["angle"] - rotation + 180),
            abs(m["angle"] - rotation - 180),
        )
        for m in selected_metrics
    ]

    """
    if np.median(angle_diffs) > 20:  # If median angle differs by more than 20 degrees
        logger.warning(f"Measured angle ({final_angle:.1f}°) differs significantly from expected ({rotation:.1f}°)")
        # Fall back to original estimate
        if rotation is not None:
            logger.info(f"Using original angle estimate: {rotation:.1f}°")
            final_angle = rotation
            logger.info(f"Using original length estimate: {length:.1f}")
            final_length = length
            # Skip further length corrections since we're using the original length
            logger.info(f"Final streak parameters: length={final_length:.1f}, angle={final_angle:.1f}°")
            return final_angle, final_length, psf, fwhm
    """

    logger.info(
        f"Final streak parameters: length={final_length:.1f}, angle={final_angle:.1f}°"
    )

    # After creating the PSF, measure its FWHM perpendicular to the streak direction
    measured_fwhm = measure_psf_fwhm(psf, final_angle)

    if measured_fwhm is not None and measured_fwhm > 0:
        logger.info(f"Measured PSF FWHM: {measured_fwhm:.1f} pixels")
        # Use the measured FWHM for length correction
        fwhm_for_correction = measured_fwhm
    else:
        logger.warning("Could not measure PSF FWHM, using initial estimate")
        fwhm_for_correction = fwhm

    # Calculate raw and corrected lengths
    raw_length = final_length

    # Correct for PSF blurring by subtracting FWHM
    # Option 1: Subtract more than one FWHM (e.g., 1.5 times)
    corrected_length = raw_length - (1.5 * fwhm_for_correction)

    # Ensure the corrected length is not negative or too small
    final_length = max(corrected_length, fwhm_for_correction * 0.5)

    logger.info(
        f"Raw streak length: {raw_length:.1f}, corrected to: {final_length:.1f} after PSF subtraction"
    )

    # Sanity checks and fallbacks as before...
    streak_measurement = StreakMeasurement(
        rotation=final_angle,
        length=final_length,
        fwhm=measured_fwhm,
    )
    # Return the measured FWHM along with other parameters
    return streak_measurement, psf, measured_fwhm


def extract_streak_dims(
    data,
    n_streaks=5,
    length=None,
    rotation=None,
    fwhm: float = 4.0,
):
    logger.info("extracting streak params from image")

    kernel = rectangle_pyramoid(
        length,
        np.sin(np.deg2rad(rotation)),
        np.cos(np.deg2rad(rotation)),
        int(fwhm * 2),
        upsample=100,
        halo_fwhm=4,
        halo_level=0,
    )

    # clear out borders on data frame for any bright edge effects
    pixel_cut = np.mean(data) + 3 * np.std(data)
    fill_min = np.median(data) + 0.5 * np.std(data)

    border_pixels = mask_all_but_border(data, 2)
    while np.max(border_pixels) > pixel_cut:
        # logger.info("removing border point")
        start_point = np.unravel_index(np.argmax(border_pixels), data.shape)
        data = remove_streak_at_point(data, start_point, fill_min)
        border_pixels = mask_all_but_border(data, 2)

    conv = convolve(data, kernel, mode="same")
    cfill_min = np.median(conv) + 0.5 * np.std(conv)

    cutout_shape = 2 * int(length)

    attempts = 0
    streaks = []
    streak_snrs = []  # Track SNR of each streak candidate

    while len(streaks) < n_streaks and attempts < 30:
        attempts += 1

        maxx, maxy = np.unravel_index(np.argmax(conv), shape=conv.shape)

        # Check if we're too close to the edge for a full cutout
        if (
            maxx < cutout_shape
            or maxx > data.shape[0] - cutout_shape
            or maxy < cutout_shape
            or maxy > data.shape[1] - cutout_shape
        ):
            # Skip this candidate and move to the next
            conv = remove_streak_at_point(conv, [maxx, maxy], cfill_min)
            data = remove_streak_at_point(data, [maxx, maxy], fill_min)
            continue

        cutout = data[
            maxx - cutout_shape : maxx + cutout_shape,
            maxy - cutout_shape : maxy + cutout_shape,
        ].copy()

        if cutout.shape == tuple([cutout_shape * 2, cutout_shape * 2]):
            # Calculate SNR before normalization
            # Use edge pixels for background estimate (avoid including the streak itself)
            edge_width = max(5, cutout_shape // 10)
            edge_pixels = np.concatenate(
                [
                    cutout[:edge_width, :].ravel(),
                    cutout[-edge_width:, :].ravel(),
                    cutout[:, :edge_width].ravel(),
                    cutout[:, -edge_width:].ravel(),
                ]
            )
            bg_level = np.median(edge_pixels)
            bg_noise = np.std(edge_pixels)
            peak_value = np.max(cutout)
            snr = (peak_value - bg_level) / bg_noise if bg_noise > 0 else 0

            # Subtract background using edge-based estimate (not global median)
            cutout_clean = cutout - bg_level
            cutout_clean = np.clip(cutout_clean, 0, None)  # Remove negative values

            # Normalize
            cutout_max = np.max(cutout_clean)
            if cutout_max > 0:
                cutout_clean /= cutout_max
            else:
                # Skip this cutout if it's all background
                conv = remove_streak_at_point(conv, [maxx, maxy], cfill_min)
                data = remove_streak_at_point(data, [maxx, maxy], fill_min)
                continue

            # plot_single_frame(cutout_clean, output_file="test2.png")

            streaks.append(cutout_clean)
            streak_snrs.append(snr)

        conv = remove_streak_at_point(conv, [maxx, maxy], cfill_min)
        data = remove_streak_at_point(data, [maxx, maxy], fill_min)

    if not streaks:
        logger.warning("No valid streak candidates found")
        return rotation, length, None

    # If we have multiple streaks, use the top 50% by SNR
    if len(streaks) > 1:
        # Sort streaks by SNR
        sorted_indices = np.argsort(streak_snrs)[::-1]  # Descending order
        # Take the top half (or at least 1)
        top_n = max(1, len(streaks) // 2)
        selected_indices = sorted_indices[:top_n]
        selected_streaks = [streaks[i] for i in selected_indices]
        logger.info(
            f"Selected {top_n} highest SNR streaks out of {len(streaks)} candidates"
        )
        psf = np.median(np.stack(selected_streaks), 0)
    else:
        psf = streaks[0]

    # scale to 0 -> 1 (this is idealized PSF)
    psf -= np.min(psf)
    psf /= np.max(psf)
    # plot_single_frame(psf, output_file="test3.png")

    streak_len = streak_length_from_cutout(psf)
    fwhm_pixel = streak_fwhm_from_cutout(psf, rotation)

    # in case weird case with extremely precise point sources
    streak_len = np.max([fwhm, streak_len])

    streak_len_refined = refine_streak_len(psf, fwhm, rotation)
    logger.info(f"Refined length estimate: {streak_len_refined:.1f}")

    # If refined length is suspiciously short, use the original estimate
    if streak_len_refined < length * 0.5:
        logger.warning(
            f"Refined length ({streak_len_refined:.1f}) seems too short compared to expected ({length:.1f})"
        )
        return rotation, length, psf

    return rotation, streak_len_refined, psf


def streak_fwhm_from_cutout(cutout_frame: np.ndarray, rotation: float) -> float:
    """Measure the Full Width at Half Maximum (FWHM) of a streak PSF.

    Args:
        cutout_frame: 2D array containing the PSF
        rotation: Angle to rotate the cutout (degrees) to align streak vertically

    Returns:
        float: FWHM in pixels, or None if measurement fails
    """
    if rotation != 0:
        rotated_cutout = rotate(
            cutout_frame, angle=rotation, mode="constant", cval=np.nan
        )
    else:
        rotated_cutout = cutout_frame

    # Compress along horizontal axis using mean to get vertical profile
    vertical_profile = np.nanmean(rotated_cutout, axis=1)

    # Remove any NaN values
    vertical_profile = vertical_profile[~np.isnan(vertical_profile)]

    # Find peak value and location
    peak_value = np.max(vertical_profile)
    peak_idx = np.argmax(vertical_profile)

    # Calculate half maximum value
    half_max = peak_value / 2

    # Find points where profile crosses half maximum
    above_half = vertical_profile >= half_max

    # Use more robust method to find FWHM
    # Look for crossings on both sides of the peak separately
    left_side = above_half[:peak_idx]
    right_side = above_half[peak_idx:]

    if len(left_side) > 0 and len(right_side) > 0:
        # Find left crossing (last True before peak)
        if not np.all(left_side):
            left_idx = len(left_side) - 1 - np.argmax(left_side[::-1] == False)
        else:
            left_idx = 0

        # Find right crossing (first False after peak)
        if not np.all(right_side):
            right_idx = peak_idx + np.argmax(right_side == False)
        else:
            right_idx = len(vertical_profile) - 1

        # Calculate FWHM in pixels
        fwhm = right_idx - left_idx
        return float(fwhm)
    else:
        # If can't find clear crossings, return None
        return None


def streak_length_from_cutout(cutout_frame, plot=True):
    """Calculate streak length using FWHM-based analysis for robustness."""
    subcc = cutout_frame.copy()
    subcc = subcc.copy() - np.median(subcc)
    subcc /= np.max(subcc)

    # Use the old threshold approach to get initial pixel mapping
    fill_min = 0.50  # FWHM
    start_point = np.unravel_index(np.argmax(subcc), subcc.shape)
    mapped = map_cluster(subcc, start_point, fill_min)

    # Get coordinates of all points in the mapped region
    y_coords, x_coords = np.where(mapped)

    if len(y_coords) == 0:
        return 0.0

    # Use the FWHM-based analysis for more robust length calculation
    analysis_result = analyze_source_shape_fwhm(subcc, y_coords, x_coords)

    return analysis_result["length"]


def streak_parameters_from_xcorr(
    cutout_frame: np.ndarray,
    plate_scale_arcsec: float,
    seeing_fwhm_pixels: float,
    expected_max_star_distance_arcsec: float | None = None,
) -> tuple[StreakMeasurement, np.ndarray]:
    """Extract streak parameters from cross-correlation frame with quality assessment.

    Args:
        cutout_frame: Cross-correlation between sidereal and rate frames
        plate_scale_arcsec: Plate scale in arcsec/pixel
        seeing_fwhm_pixels: Seeing FWHM in pixels
        expected_max_star_distance_arcsec: Expected maximum star travel distance

    Returns:
        tuple: (StreakMeasurement with quality info, processed cutout)
    """
    cutout_frame = (cutout_frame.copy() - np.min(cutout_frame)) / np.max(cutout_frame)

    center = np.array(cutout_frame.shape) / 2
    subcc = cutout_frame.copy()

    # Zero out the center to avoid correlation artifacts
    subcc[int(center[0]), int(center[1])] = 0

    # Apply windowing if we have expected scale
    pixel_scale = None
    if plate_scale_arcsec is not None and expected_max_star_distance_arcsec is not None:
        pixel_scale = int(2 * expected_max_star_distance_arcsec / plate_scale_arcsec)

    if pixel_scale is not None:
        # Ensure indices stay within frame boundaries
        x_min = max(0, int(center[0]) - pixel_scale + 1)
        x_max = min(subcc.shape[0], int(center[0]) + pixel_scale)
        y_min = max(0, int(center[1]) - pixel_scale + 1)
        y_max = min(subcc.shape[1], int(center[1]) + pixel_scale)
        subcc = subcc[x_min:x_max, y_min:y_max]

    # Normalize the cutout
    subcc = subcc.copy() - np.median(subcc)
    subcc /= np.max(subcc) if np.max(subcc) > 0 else 1.0
    subc = np.array(subcc.shape) / 2

    # Use multiple thresholds to assess robustness
    thresholds = [0.3, 0.4, 0.5, 0.6]
    length_estimates = []
    rotation_estimates = []
    quality_metrics = {}

    for thresh in thresholds:
        try:
            start_point = np.unravel_index(np.argmax(subcc), subcc.shape)
            mapped = map_cluster(subcc, start_point, thresh)

            if not np.any(mapped):
                continue

            # Get coordinates of detected pixels
            y_coords, x_coords = np.where(mapped)

            if len(y_coords) < 5:  # Need minimum points for reliable measurement
                continue

            # Method 1: Min/max coordinate difference (original method)
            xlen_minmax = max(x_coords) - min(x_coords)
            ylen_minmax = max(y_coords) - min(y_coords)
            length_minmax = np.sqrt(xlen_minmax**2 + ylen_minmax**2)

            # Method 2: PCA-based length (more robust)
            points = np.column_stack([y_coords, x_coords])
            points_centered = points - np.mean(points, axis=0)

            if len(points_centered) > 1:
                cov_matrix = np.cov(points_centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

                # Sort eigenvalues in descending order
                idx = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]

                # Length is 4 standard deviations along major axis (covers ~95% of points)
                length_pca = 4 * np.sqrt(eigenvalues[0]) if eigenvalues[0] > 0 else 0

                # Calculate aspect ratio for quality assessment
                aspect_ratio = (
                    np.sqrt(eigenvalues[0] / eigenvalues[1])
                    if eigenvalues[1] > 0
                    else float("inf")
                )

                # Calculate rotation angle
                rotation_deg = np.degrees(
                    np.arctan2(eigenvectors[0, 0], eigenvectors[1, 0])
                )
                rotation_deg = rotation_deg % 180

                # Use PCA length as primary estimate
                length_estimates.append(length_pca)
                rotation_estimates.append(rotation_deg)

                # Store quality metrics for this threshold
                quality_metrics[thresh] = {
                    "length_minmax": length_minmax,
                    "length_pca": length_pca,
                    "aspect_ratio": aspect_ratio,
                    "n_pixels": len(y_coords),
                    "peak_value": np.max(subcc[mapped]),
                }

        except (np.linalg.LinAlgError, ValueError):
            continue

    # Assess measurement quality and select best estimate
    if not length_estimates:
        logger.warning("No valid streak detected in cross-correlation frame")
        return (
            StreakMeasurement(
                rotation=0.0,
                length=seeing_fwhm_pixels,  # Fallback to seeing disk
                fwhm=seeing_fwhm_pixels,
            ),
            subcc,
        )

    # Calculate statistics across thresholds
    median_length = np.median(length_estimates)
    std_length = np.std(length_estimates)
    median_rotation = np.median(rotation_estimates)

    # Quality assessment
    length_cv = std_length / median_length if median_length > 0 else float("inf")

    # Check for consistency across thresholds
    is_reliable = True
    warning_messages = []

    if length_cv > 0.3:  # High coefficient of variation
        is_reliable = False
        warning_messages.append(
            f"High length variability across thresholds (CV={length_cv:.2f})"
        )

    if median_length < 2 * seeing_fwhm_pixels:
        is_reliable = False
        warning_messages.append(
            f"Detected length ({median_length:.1f}) is very small compared to seeing ({seeing_fwhm_pixels:.1f})"
        )

    # Check aspect ratios
    aspect_ratios = [
        metrics["aspect_ratio"]
        for metrics in quality_metrics.values()
        if metrics["aspect_ratio"] != float("inf")
    ]
    if aspect_ratios and np.median(aspect_ratios) < 2.0:
        is_reliable = False
        warning_messages.append(
            f"Low aspect ratio ({np.median(aspect_ratios):.1f}) suggests round rather than elongated feature"
        )

    # Apply conservative length correction
    # Only subtract a fraction of the seeing FWHM, not the full amount
    corrected_length = median_length
    if median_length > 3 * seeing_fwhm_pixels:
        # Only apply correction for clearly elongated features
        corrected_length = median_length - 0.5 * seeing_fwhm_pixels
        logger.info(
            f"Applied seeing correction: {median_length:.1f} -> {corrected_length:.1f} pixels"
        )

    # Ensure minimum length
    final_length = max(corrected_length, seeing_fwhm_pixels)

    # Log warnings if measurement is unreliable
    if not is_reliable:
        logger.warning("Cross-correlation streak measurement may be unreliable:")
        for msg in warning_messages:
            logger.warning(f"  - {msg}")
        logger.warning(f"Using length estimate: {final_length:.1f} pixels")
    else:
        logger.info(
            f"Cross-correlation streak measurement: length={final_length:.1f}px, rotation={median_rotation:.1f}°"
        )

    # Log detailed quality metrics for debugging
    logger.debug("Cross-correlation quality metrics:")
    for thresh, metrics in quality_metrics.items():
        logger.debug(
            f"  Threshold {thresh}: length_pca={metrics['length_pca']:.1f}, "
            f"aspect_ratio={metrics['aspect_ratio']:.1f}, n_pixels={metrics['n_pixels']}"
        )

    return (
        StreakMeasurement(
            rotation=median_rotation,
            length=final_length,
            fwhm=seeing_fwhm_pixels,
        ),
        subcc,
    )


def measure_gaussian_shift(centered_cutout: np.ndarray) -> tuple[np.ndarray, float]:
    """Measure the shift of a PSF from the center and its FWHM by fitting a Gaussian.

    Args:
        centered_cutout: 2D array containing a centered PSF

    Returns:
        tuple: (shift_vector, fwhm) where shift_vector is the offset from center and fwhm is the
               full width at half maximum of the fitted Gaussian in pixels
    """
    # Find the peak location
    psf_center = np.unravel_index(np.argmax(centered_cutout), centered_cutout.shape)
    shift = psf_center - np.array(centered_cutout.shape) / 2

    # Extract profiles through the peak
    y_profile = centered_cutout[psf_center[0], :]
    x_profile = centered_cutout[:, psf_center[1]]

    # Normalize profiles
    y_profile = y_profile / np.max(y_profile)
    x_profile = x_profile / np.max(x_profile)

    # Create coordinate arrays
    y_coords = np.arange(len(y_profile))
    x_coords = np.arange(len(x_profile))

    # Define 1D Gaussian function
    def gaussian(x, amplitude, center, sigma, offset):
        return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma**2)) + offset

    # Initial parameter guesses
    p0_y = [1.0, np.argmax(y_profile), 3.0, 0.0]
    p0_x = [1.0, np.argmax(x_profile), 3.0, 0.0]

    try:
        # Fit Gaussians to both profiles
        popt_y, _ = curve_fit(gaussian, y_coords, y_profile, p0=p0_y)
        popt_x, _ = curve_fit(gaussian, x_coords, x_profile, p0=p0_x)

        # Extract sigma values
        sigma_y = abs(popt_y[2])
        sigma_x = abs(popt_x[2])

        # Calculate FWHM (FWHM = 2.355 * sigma for a Gaussian)
        fwhm_y = 2.355 * sigma_y
        fwhm_x = 2.355 * sigma_x

        # Use the average FWHM
        fwhm = (fwhm_x + fwhm_y) / 2

    except (RuntimeError, ValueError):
        # If fitting fails, estimate FWHM using half-maximum points
        fwhm = estimate_fwhm_from_profiles(x_profile, y_profile)

    return shift, fwhm


def estimate_fwhm_from_profiles(x_profile, y_profile):
    """Estimate FWHM from profiles when curve fitting fails."""
    # Find half-maximum points in both profiles
    half_max = 0.5

    # Process x profile
    above_half_x = x_profile >= half_max
    if np.any(above_half_x):
        left_x = np.argmax(above_half_x)
        right_x = len(above_half_x) - np.argmax(above_half_x[::-1]) - 1
        fwhm_x = right_x - left_x
    else:
        fwhm_x = 4.0  # Default value

    # Process y profile
    above_half_y = y_profile >= half_max
    if np.any(above_half_y):
        left_y = np.argmax(above_half_y)
        right_y = len(above_half_y) - np.argmax(above_half_y[::-1]) - 1
        fwhm_y = right_y - left_y
    else:
        fwhm_y = 4.0  # Default value

    return (fwhm_x + fwhm_y) / 2


def measure_psf_shift_parameter_free(
    centered_cutout: np.ndarray, expected_max_distance: float | None = None
) -> np.ndarray:
    """Measure the shift of a PSF from the center without requiring streak parameters.

    This function finds the brightest point in the cutout and returns its offset
    from the center. It can optionally mask the center region to avoid correlation
    artifacts.

    Args:
        centered_cutout: 2D array containing the cross-correlation result
        expected_max_distance: Optional maximum expected distance from center (pixels)

    Returns:
        np.ndarray: Shift vector from center to peak
    """
    center = np.array(centered_cutout.shape) / 2

    # Create a working copy
    working_cutout = centered_cutout.copy()

    # Optionally mask the center to avoid correlation artifacts
    # This is useful for cross-correlation results where the center might have artifacts
    center_mask_radius = 2  # Small radius to mask center artifacts
    y_center, x_center = int(center[0]), int(center[1])

    # Create circular mask around center
    y_indices, x_indices = np.ogrid[
        : working_cutout.shape[0], : working_cutout.shape[1]
    ]
    center_mask = (y_indices - y_center) ** 2 + (
        x_indices - x_center
    ) ** 2 <= center_mask_radius**2

    # Set center region to minimum value
    if np.any(center_mask):
        working_cutout[center_mask] = np.min(working_cutout)

    # If we have an expected maximum distance, limit search to that region
    if expected_max_distance is not None:
        # Create a mask for the search region
        search_mask = (y_indices - y_center) ** 2 + (
            x_indices - x_center
        ) ** 2 <= expected_max_distance**2

        # Set everything outside search region to minimum
        working_cutout[~search_mask] = np.min(working_cutout)

    # Find the peak
    peak_y, peak_x = np.unravel_index(np.argmax(working_cutout), working_cutout.shape)

    # Calculate shift from center
    shift = np.array([peak_x - center[1], peak_y - center[0]])  # Return as [x, y]

    return shift


def measure_streak_shift_centroid(
    centered_cutout: np.ndarray,
    expected_max_distance: float | None = None,
    intensity_threshold: float = 0.5,
) -> np.ndarray:
    """Measure the shift of a streak from the center using centroid calculation.

    This function is designed for cross-correlation results where the feature is a streak
    (e.g., correlating point sources with streaks). Instead of finding just the peak,
    it finds the center of mass of the streak.

    Args:
        centered_cutout: 2D array containing the cross-correlation result with a streak
        expected_max_distance: Optional maximum expected distance from center (pixels)
        intensity_threshold: Fraction of peak intensity to threshold at (default: 0.5)

    Returns:
        np.ndarray: Shift vector from center to streak centroid [x, y]
    """
    from scipy.ndimage import center_of_mass, label

    center = np.array(centered_cutout.shape) / 2
    y_center, x_center = int(center[0]), int(center[1])

    # Normalize the cutout
    working_cutout = centered_cutout.copy()
    working_cutout -= np.min(working_cutout)
    if np.max(working_cutout) > 0:
        working_cutout /= np.max(working_cutout)

    # Mask the center to avoid correlation artifacts
    center_mask_radius = 2
    y_indices, x_indices = np.ogrid[
        : working_cutout.shape[0], : working_cutout.shape[1]
    ]
    center_mask = (y_indices - y_center) ** 2 + (
        x_indices - x_center
    ) ** 2 <= center_mask_radius**2
    if np.any(center_mask):
        working_cutout[center_mask] = 0

    # If we have an expected maximum distance, limit search to that region
    if expected_max_distance is not None:
        search_mask = (y_indices - y_center) ** 2 + (
            x_indices - x_center
        ) ** 2 <= expected_max_distance**2
        working_cutout[~search_mask] = 0

    # Threshold at a fraction of the peak to get the main streak feature
    threshold = intensity_threshold
    binary_mask = working_cutout > threshold

    if not np.any(binary_mask):
        logger.warning(
            f"No pixels above threshold {threshold:.2f}, falling back to peak finding"
        )
        return measure_psf_shift_parameter_free(centered_cutout, expected_max_distance)

    # Find connected components
    labeled_mask, num_features = label(binary_mask)

    if num_features == 0:
        logger.warning("No connected components found, falling back to peak finding")
        return measure_psf_shift_parameter_free(centered_cutout, expected_max_distance)

    # Find the largest connected component
    component_sizes = np.bincount(labeled_mask.ravel())[1:]  # Skip background
    largest_component = np.argmax(component_sizes) + 1
    largest_mask = labeled_mask == largest_component

    # Calculate the intensity-weighted centroid of the largest component
    # Use original (normalized) intensities for weighting
    masked_intensities = working_cutout * largest_mask
    centroid_y, centroid_x = center_of_mass(masked_intensities)

    # Calculate shift from center
    shift = np.array([centroid_x - center[1], centroid_y - center[0]])  # [x, y]

    logger.info(
        f"Streak centroid shift: ({shift[0]:.2f}, {shift[1]:.2f}), "
        f"component size: {component_sizes[largest_component - 1]} pixels"
    )

    return shift


def measure_psf_shift(
    centered_cutout: np.ndarray, length: float, rotation: float, pixel_fwhm: float
) -> np.ndarray:
    """Legacy function - now just calls the parameter-free version with a warning."""
    logger.warning(
        "measure_psf_shift is deprecated - using parameter-free version instead"
    )

    # Estimate expected distance based on length
    expected_distance = length * 0.75 if length > 0 else None

    return measure_psf_shift_parameter_free(centered_cutout, expected_distance)


def cross_corr(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Cross correlate two images using FFT.

    Args:
        img1 (np.ndarray): First image to cross correlate
        img2 (np.ndarray): Second image to cross correlate

    Returns:
        np.ndarray: Cross correlated image
    """
    # scipy.fft: same pocketfft as numpy's, multithreaded with workers —
    # identical values, several times faster on full frames.
    from scipy import fft as sfft

    ccf = np.roll(
        sfft.ifft2(
            sfft.fft2(img1, workers=-1).conj() * sfft.fft2(img2, workers=-1),
            workers=-1,
        ).real,
        np.array([img1.shape[0] - 1, img1.shape[1] - 1]) // 2,
        axis=(0, 1),
    )

    return ccf


def measure_psf_fwhm(data: np.ndarray, rotation: float | None = None) -> float:
    """
    Measure the FWHM of the PSF perpendicular to the streak direction with sub-pixel precision.

    Args:
        data: 2D array containing a normalized PSF or source
        rotation: Angle of the streak in degrees (if None, will try to determine)

    Returns:
        float: FWHM in pixels
    """
    # If rotation is not provided, try to determine it
    if rotation is None:
        # Use PCA to find principal axes
        y_indices, x_indices = np.where(data > 0.5 * np.max(data))
        if len(y_indices) < 5:  # Not enough points
            return None

        points = np.column_stack([y_indices, x_indices])
        points_centered = points - np.mean(points, axis=0)

        try:
            cov = np.cov(points_centered, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Sort in descending order
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Calculate orientation (perpendicular to streak)
            streak_angle = np.degrees(
                np.arctan2(eigenvectors[0, 0], eigenvectors[1, 0])
            )
            # The width direction is perpendicular to the streak
            width_angle = (streak_angle + 90) % 180
        except np.linalg.LinAlgError:
            return None
    else:
        # Width direction is perpendicular to streak
        width_angle = (rotation + 90) % 180

    # Rotate the data to align the width direction with horizontal axis
    rotated_data = rotate(data, width_angle, reshape=False, mode="constant", cval=0)

    # Find the peak
    peak_y, peak_x = np.unravel_index(np.argmax(rotated_data), rotated_data.shape)

    # Extract a profile through the peak perpendicular to the streak
    profile = rotated_data[peak_y, :]

    # Normalize the profile
    profile = profile / np.max(profile)

    # Find the half-maximum value
    half_max = 0.5

    # Find indices where profile crosses half-maximum
    above_half = profile >= half_max
    if not np.any(above_half):
        return None

    # Find approximate crossing points
    left_idx = np.argmax(above_half)  # First crossing
    right_idx = len(above_half) - np.argmax(above_half[::-1]) - 1  # Last crossing

    # Refine left crossing with linear interpolation
    if left_idx > 0:
        y1 = profile[left_idx - 1]
        y2 = profile[left_idx]
        if y2 > y1:  # Ensure we're crossing upward
            # Linear interpolation: x = x1 + (target_y - y1) * (x2 - x1) / (y2 - y1)
            left_precise = (left_idx - 1) + (half_max - y1) / (y2 - y1)
        else:
            left_precise = float(left_idx)
    else:
        left_precise = float(left_idx)

    # Refine right crossing with linear interpolation
    if right_idx < len(profile) - 1:
        y1 = profile[right_idx]
        y2 = profile[right_idx + 1]
        if y1 > y2:  # Ensure we're crossing downward
            # Linear interpolation: x = x1 + (target_y - y1) * (x2 - x1) / (y2 - y1)
            right_precise = right_idx + (half_max - y1) / (y2 - y1)
        else:
            right_precise = float(right_idx)
    else:
        right_precise = float(right_idx)

    # Calculate FWHM with sub-pixel precision
    fwhm = right_precise - left_precise

    # If FWHM is too small, it might be noise - set a minimum
    fwhm = max(fwhm, 2.0)

    return fwhm


_EK_CACHE: dict = {}  # last effective-kernel, keyed by a cheap kernel fingerprint
_BG_CACHE: dict = {}  # background per working-data array (weakref-validated)


def streak_mask_effective_kernel(kernel: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Binary 'effective area of influence' = kernel self-convolved + thresholded.

    Depends only on the kernel, so it's cached: the candidate loop reuses the same
    streak kernel ~30x, and the self-convolution of a ~85x207 kernel costs ~0.8 s
    each — recomputing it every call dominated runtime."""
    from scipy import signal

    key = (kernel.shape, float(kernel.sum()), round(threshold, 6))
    cached = _EK_CACHE.get(key)
    if cached is not None:
        return cached
    ek = signal.convolve2d(kernel, kernel, mode="full")
    ek = (ek / np.max(ek)) > threshold
    _EK_CACHE.clear()  # only ever need the current kernel's
    _EK_CACHE[key] = ek
    return ek


def _cached_working_bg(working_data: np.ndarray) -> float:
    """Median background of working_data, cached by array identity (weakref-
    validated against id reuse). Masking a handful of small regions doesn't shift
    a 66 MP median, so computing it once per extraction (not ~30x) is exact enough
    and avoids ~0.8 s/call."""
    import weakref

    key = id(working_data)
    ent = _BG_CACHE.get(key)
    if ent is not None and ent[0]() is working_data:
        return ent[1]
    bg = float(np.median(working_data))
    _BG_CACHE.clear()
    _BG_CACHE[key] = (weakref.ref(working_data), bg)
    return bg


def mask_streak_region(
    processed_mask: np.ndarray,
    working_data: np.ndarray,
    y_max: int,
    x_max: int,
    kernel: np.ndarray,
    threshold: float = 0.01,
    effective_kernel: np.ndarray | None = None,
    bg_value: float | None = None,
    response: np.ndarray | None = None,
    response_fill: float = -np.inf,
):
    """
    Mask a streak region's area of influence in ``processed_mask`` /
    ``working_data`` (and optionally a ``response`` map used for peak-finding).

    For speed in the candidate loop, pass a precomputed ``effective_kernel``
    (see :func:`streak_mask_effective_kernel`) and a ``bg_value`` so this routine
    doesn't recompute the kernel self-convolution or a full-frame median on every
    call, and applies the mask to a bounded slice rather than allocating a
    full-frame boolean array. Results are identical to the old per-call version.

    Args:
        effective_kernel: precomputed binary influence area (else computed here).
        bg_value: fill value for masked working_data (else full-frame median).
        response: optional peak-response array to also mask (set to
            ``response_fill``, default -inf, so a masked region won't be re-picked
            by argmax — replaces copying the whole response map each iteration).

    Returns:
        tuple: (updated_processed_mask, updated_working_data)
    """
    if effective_kernel is None:
        effective_kernel = streak_mask_effective_kernel(kernel, threshold)
    if bg_value is None:
        bg_value = _cached_working_bg(working_data)

    ek_height, ek_width = effective_kernel.shape
    y_start = max(0, y_max - ek_height // 2)
    y_end = min(processed_mask.shape[0], y_max + ek_height // 2)
    x_start = max(0, x_max - ek_width // 2)
    x_end = min(processed_mask.shape[1], x_max + ek_width // 2)

    # Calculate kernel indices
    k_y_start = max(0, ek_height // 2 - y_max)
    k_y_end = min(ek_height, k_y_start + (y_end - y_start))
    k_x_start = max(0, ek_width // 2 - x_max)
    k_x_end = min(ek_width, k_x_start + (x_end - x_start))

    kernel_part = effective_kernel[k_y_start:k_y_end, k_x_start:k_x_end]

    # Apply directly to the bounded slices (no full-frame allocation).
    mask_height, mask_width = kernel_part.shape
    if mask_height > 0 and mask_width > 0:  # Ensure we have a valid region
        processed_mask[y_start:y_end, x_start:x_end] |= kernel_part
        working_data[y_start:y_end, x_start:x_end][kernel_part] = bg_value
        if response is not None:
            response[y_start:y_end, x_start:x_end][kernel_part] = response_fill

    return processed_mask, working_data


def is_valid_psf(cutout, processed_mask, y_max, x_max, cutout_size):
    """
    Check if a PSF cutout is valid by ensuring it doesn't overlap with masked regions.

    Args:
        cutout: The extracted PSF cutout
        processed_mask: The mask of processed regions
        y_max, x_max: The coordinates of the detected streak
        cutout_size: Size of the cutout

    Returns:
        bool: True if the PSF is valid, False otherwise
    """
    # Extract the corresponding region from the processed mask
    y_start = max(0, y_max - cutout_size)
    y_end = min(processed_mask.shape[0], y_max + cutout_size)
    x_start = max(0, x_max - cutout_size)
    x_end = min(processed_mask.shape[1], x_max + cutout_size)

    mask_cutout = processed_mask[y_start:y_end, x_start:x_end]

    # Check if the cutout overlaps with previously masked regions
    # Allow some overlap (e.g., less than 10% of pixels)
    overlap_fraction = np.sum(mask_cutout) / mask_cutout.size

    # Return True if overlap is below threshold
    return overlap_fraction < 0.1  # Adjust threshold as needed
