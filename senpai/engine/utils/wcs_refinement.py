"""Sidereal and rate-track WCS refinement engines."""

import logging

import numpy as np
from scipy.signal import convolve

from senpai.core.config import get_config
from senpai.engine.detection.jacobian import get_local_streak_kernel
from senpai.engine.detection.kernels import sidereal_kernel
from senpai.engine.models.astrometry import WCSMetadata, WCSModel, WCSStatus
from senpai.engine.models.senpai import RateTrackFrame, SiderealFrame
from senpai.engine.models.starfield import StarInImage, StarInSpace, StarListImage
from senpai.engine.photometry.utils import calculate_star_snrs_with_aperture_photometry
from senpai.engine.plotting.images import plot_single_frame
from senpai.engine.plotting.wcs_diagnostics import (
    plot_variable_kernel_grid,
    plot_variable_kernel_star_diagnostic,
)
from senpai.engine.utils.stats import fft_workers
from senpai.engine.utils.wcs_helpers import (
    calculate_spatial_coverage,
    compute_snr_and_filter_stars,
    find_local_maxima,
    fit_and_validate_wcs,
    match_stars_to_detections,
    reject_outlier_shifts_by_mad,
    update_starfield_wcs,
)
from senpai.engine.utils.wcs_ops import (
    catalog_stars_from_wcs,
    compute_wcs_distortion_metrics,
    existing_stars_from_wcs,
    filter_catalog_stars_by_radius,
    shift_wcs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate-track: kernel-convolution refinement (top-level entry point)
# ---------------------------------------------------------------------------


def refine_wcs_by_kernel_convolution(frame: RateTrackFrame) -> tuple[float, float]:
    """Refine the WCS by convolving the image with a streak kernel.

    Args:
        frame (RateTrackFrame): The frame for which to refine the WCS.

    Returns:
        tuple[float, float]: The correction (delta_x, delta_y) in pixels applied during refinement.
                            These are adjustments to the existing WCS, not absolute shifts.
    """
    config = get_config()

    if frame.starfield.wcs_status != WCSStatus.PIXEL_SHIFTED_WCS:
        logger.error(
            "WCS status is not PIXEL_SHIFTED_WCS, skipping kernel convolution [call senpai.engine.utils.propagate_wcs.shift_wcs_by_pixel_shift first]"
        )
        raise ValueError(
            "WCS status is not PIXEL_SHIFTED_WCS, skipping kernel convolution"
        )

    # Decide whether to enable variable kernels for this frame based on WCS distortion.
    use_variable_kernels = False
    distortion_metrics = None
    try:
        # Variable-kernel control is global for the run; we also require a valid WCS.
        if config.streak.variable_kernel.enable and frame.starfield.wcs is not None:
            distortion_metrics = compute_wcs_distortion_metrics(
                frame.starfield.wcs,
                frame.frame.data.shape,
            )

            if distortion_metrics:
                # Persist scalar metrics on the starfield for later inspection/serialization
                frame.starfield.distortion_metrics = distortion_metrics

                max_angle = distortion_metrics.get("max_angle_variation_deg", 0.0)
                max_length_frac = distortion_metrics.get(
                    "max_length_variation_fraction", 0.0
                )

                logger.info(
                    "Rate-track WCS distortion (frame %d): "
                    "delta_J=%.3g, max_angle_variation_deg=%.3f, max_length_variation_fraction=%.3f",
                    frame.index,
                    distortion_metrics.get("delta_J", 0.0),
                    max_angle,
                    max_length_frac,
                )

                angle_thresh = config.streak.variable_kernel.angle_thresh_deg
                length_thresh = config.streak.variable_kernel.length_thresh_fraction

                if max_angle >= angle_thresh or max_length_frac >= length_thresh:
                    use_variable_kernels = True
                    logger.info(
                        "Enabling variable streak kernels for frame %d "
                        "(angle variation %.3f° >= %.3f° or length variation %.3f >= %.3f)",
                        frame.index,
                        max_angle,
                        angle_thresh,
                        max_length_frac,
                        length_thresh,
                    )
                else:
                    logger.info(
                        "Keeping single streak kernel for frame %d "
                        "(angle variation %.3f° < %.3f°, length variation %.3f < %.3f)",
                        frame.index,
                        max_angle,
                        angle_thresh,
                        max_length_frac,
                        length_thresh,
                    )
    except Exception as e:
        logger.warning(
            "Error while evaluating variable-kernel decision for frame %d: %s. Falling back to single-kernel behavior.",
            frame.index,
            e,
        )
        use_variable_kernels = False

    # Store decision on the streak metadata so downstream refinement can act on it.
    if frame.streak is not None:
        # type: ignore[attr-defined] - added dynamically for backward compatibility
        frame.streak.use_variable_kernel = use_variable_kernels

    # Get the kernel
    kernel = frame.streak.to_pyramoid()
    with fft_workers():
        convolved_image = convolve(frame.frame.data, kernel, mode="same")

    # First pass: Get global shift using astrometric fit stars
    global_shift_x, global_shift_y = get_global_shift_from_astrometric_stars(
        frame, convolved_image
    )

    logger.info(
        f"Calculated diagnostic global shift: ({global_shift_x:.2f}, {global_shift_y:.2f})"
    )

    # Use the existing WCS without additional shifting (it's already been positioned correctly)
    original_wcs_model = frame.starfield.wcs
    updated_wcs_model = shift_wcs(original_wcs_model, -global_shift_x, -global_shift_y)

    # Update the WCS with the global shift (no limiting magnitude — need all stars for refinement)
    update_starfield_wcs(frame, updated_wcs_model, limiting_magnitude=None)

    if config.plotting.debug:
        plot_single_frame(
            frame.frame.data,
            starfield=frame.starfield,
            streak=frame.streak,
            output_file=config.runtime.output_dir
            / f"{frame.index}_kernel_1_global.png",
        )

    # Second pass: Refine WCS using catalog stars
    refined_wcs = refine_wcs_with_catalog_stars(frame, convolved_image)

    if refined_wcs is not None:
        # Update with the refined WCS if successful. Keep the catalog DEEP (no
        # limiting-magnitude trim): this query only refreshes the *stored*
        # catalog_stars used downstream for photometry/completeness — refinement
        # is already done above and selects its own bright stars internally.
        # Trimming to the measured limiting magnitude here is circular: it caps
        # the photometry catalog at the depth we just measured, so subsequent
        # frames' completeness curves can't roll over and their limiting mag is
        # just where the (truncated) catalog ran out. The completeness/ZP code
        # now handles faint contamination itself (isolation cut + robust ZP).
        update_starfield_wcs(frame, refined_wcs, limiting_magnitude=None)

        if config.plotting.debug:
            plot_single_frame(
                frame.frame.data,
                starfield=frame.starfield,
                streak=frame.streak,
                output_file=config.runtime.output_dir
                / f"{frame.index}_kernel_3_refined.png",
            )
    else:
        logger.info("Using existing WCS without further refinement")
        if config.plotting.debug:
            plot_single_frame(
                frame.frame.data,
                starfield=frame.starfield,
                streak=frame.streak,
                output_file=config.runtime.output_dir
                / f"{frame.index}_kernel_final.png",
            )

    # Update WCS status
    frame.starfield.wcs_status = WCSStatus.KERNEL_REFINED_WCS

    return global_shift_x, global_shift_y


# ---------------------------------------------------------------------------
# Shared: global shift measurement
# ---------------------------------------------------------------------------


def get_global_shift_from_astrometric_stars(
    frame: RateTrackFrame, convolved_image: np.ndarray
) -> tuple[float, float]:
    """Get global shift using astrometric fit stars.

    Args:
        frame (RateTrackFrame): The frame containing the stars.
        convolved_image (np.ndarray): The convolved image.

    Returns:
        tuple[float, float]: The median shifts in x and y.
    """
    logger.info("Measuring global shift from astrometric fit stars")

    # Use astrometric_fit_stars directly from the starfield
    astrometric_stars = frame.starfield.catalog_stars

    if not astrometric_stars:
        logger.warning(
            "No astrometric fit stars found, using catalog stars for global shift"
        )
        astrometric_stars = (
            frame.starfield.astrometric_fit_stars
            if frame.starfield.astrometric_fit_stars
            else []
        )

    # Find local maxima in the convolved image
    detected_points = find_local_maxima(
        convolved_image, min_distance=30, max_detections=50
    )
    logger.info(f"Found {len(detected_points)} local maxima in the convolved image")

    # Get the stars in the frame as StarInImage objects
    stars_in_image = []
    for star in astrometric_stars:
        if star.x is not None and star.y is not None:
            stars_in_image.append(StarInImage(x=star.x, y=star.y, counts=None))

    # Match stars to detections - using max_distance instead of max_match_distance
    matched_pairs, unmatched_stars, unmatched_detections = match_stars_to_detections(
        stars_in_image, detected_points, max_distance=50
    )

    logger.info(
        f"Matched {len(matched_pairs)} stars out of {len(stars_in_image)} catalog stars and {len(detected_points)} detections"
    )
    logger.info(
        f"Unmatched stars: {len(unmatched_stars)}, unmatched detections: {len(unmatched_detections)}"
    )

    # Define minimum number of stars needed for reliable shift calculation
    MIN_STARS_FOR_SHIFT = 3  # Minimum stars needed for reliable shift calculation

    # Track shifts for each matched star
    x_shifts = []
    y_shifts = []

    # Create a list to store detected stars with their new positions
    detected_stars = []

    # Calculate shifts for matched stars
    for star_idx, detection_idx in matched_pairs:
        y, x = detected_points[detection_idx]

        # Calculate shift from original position
        original_x = stars_in_image[star_idx].x
        original_y = stars_in_image[star_idx].y

        # Record the shifts
        x_shift = x - original_x
        y_shift = y - original_y
        x_shifts.append(x_shift)
        y_shifts.append(y_shift)

        # Debug: Show which stars are being used
        logger.info(
            f"Matched star {star_idx}: expected at ({original_x:.1f}, {original_y:.1f}), detected at ({x:.1f}, {y:.1f}), shift=({x_shift:.2f}, {y_shift:.2f})"
        )

        # Create StarInImage for this detection (without counts for now)
        star_in_image = StarInImage(x=float(x), y=float(y), counts=None)
        detected_stars.append(star_in_image)

    # Use calculate_star_snrs_with_aperture_photometry to efficiently get counts for all stars at once
    if detected_stars:
        # Create temporary StarInSpace objects with the detected positions
        temp_space_stars = []
        for star in detected_stars:
            # Create a minimal StarInSpace with just the position information
            temp_space_star = StarInSpace(
                ra=0.0,  # Dummy value, not used for photometry
                dec=0.0,  # Dummy value, not used for photometry
                x=star.x,
                y=star.y,
                magnitude=None,
                catalog=None,
                catalog_id=None,
            )
            temp_space_stars.append(temp_space_star)

        # Get SNR and counts for all stars at once (no plot — this is a shift-measurement step)
        star_snr_results = calculate_star_snrs_with_aperture_photometry(
            frame, temp_space_stars, plot=False
        )

        # Update the detected stars with their counts
        for i, (_temp_star, _snr, counts) in enumerate(star_snr_results):
            detected_stars[i].counts = counts

        # Add to detections if not already present
        for star in detected_stars:
            if star not in frame.starfield.detections:
                frame.starfield.detections.append(star)

    # Calculate median shifts (more robust than mean)
    if len(x_shifts) >= MIN_STARS_FOR_SHIFT:
        median_x_shift = float(np.median(x_shifts))
        median_y_shift = float(np.median(y_shifts))
        logger.info(
            f"Global shift: x={median_x_shift:.2f}, y={median_y_shift:.2f} from {len(x_shifts)} matched stars"
        )
        logger.info(
            f"Individual shifts - x: {[f'{s:.2f}' for s in x_shifts[:5]]}... (showing first 5)"
        )
        logger.info(
            f"Individual shifts - y: {[f'{s:.2f}' for s in y_shifts[:5]]}... (showing first 5)"
        )
    else:
        median_x_shift = 0.0
        median_y_shift = 0.0
        logger.warning(
            f"Not enough matched stars ({len(x_shifts)}) for reliable shift calculation. Using zero shift."
        )

    return median_x_shift, median_y_shift


# ---------------------------------------------------------------------------
# Sidereal: top-level entry point
# ---------------------------------------------------------------------------


def refine_sidereal_frame(frame: SiderealFrame) -> None:
    """Refine WCS for sidereal frames using catalog stars from brightest to dimmest.

    Args:
        frame (SiderealFrame): The sidereal frame containing the stars.
    """
    config = get_config()

    with fft_workers():
        convolved_image = convolve(
            frame.frame.data,
            sidereal_kernel(frame.starfield.detection_metadata.pixel_fwhm),
            mode="same",
        )

    wcs_model = refine_sidereal_with_catalog_stars(frame, convolved_image)

    frame.starfield.wcs = wcs_model
    frame.starfield.wcs_metadata = WCSMetadata.from_wcsmodel(wcs_model)

    # Get catalog stars and apply radius filtering if configured
    catalog_stars = catalog_stars_from_wcs(wcs_model)
    if config.astrometry.reduce_field_by_radius is not None:
        catalog_stars = filter_catalog_stars_by_radius(
            catalog_stars,
            frame.frame.metadata,
            config.astrometry.reduce_field_by_radius,
        )
        logger.info(
            "Filtered catalog stars to %i stars within %.2f%% of image circle",
            len(catalog_stars.stars),
            config.astrometry.reduce_field_by_radius * 100,
        )

    # CRITICAL: Update pixel coordinates using full WCS with SIP distortion
    frame.starfield.catalog_stars = existing_stars_from_wcs(
        wcs_model, catalog_stars.stars
    )

    if config.plotting.debug:
        plot_single_frame(
            frame.frame.data,
            starfield=frame.starfield,
            markersize=2 * frame.starfield.detection_metadata.pixel_fwhm,
            output_file=config.runtime.output_dir
            / f"{frame.index}_side_kernel_3_refit.png",
        )


# ---------------------------------------------------------------------------
# Sidereal: catalog-star refinement
# ---------------------------------------------------------------------------


def refine_sidereal_with_catalog_stars(
    frame: SiderealFrame, convolved_image: np.ndarray
) -> WCSModel:
    """Refine WCS for sidereal frames using catalog stars from brightest to dimmest.

    Args:
        frame (SiderealFrame): The sidereal frame containing the stars.
        convolved_image (np.ndarray): The convolved image (with a 2D Gaussian kernel).

    Returns:
        WCSModel: The refined WCS model, or None if refinement failed.
    """
    config = get_config()

    # First pass: Get global shift using astrometric fit stars
    global_shift_x, global_shift_y = get_global_shift_from_astrometric_stars(
        frame, convolved_image
    )

    # Apply the global shift to the WCS
    original_wcs_model = frame.starfield.wcs
    updated_wcs_model = shift_wcs(original_wcs_model, -global_shift_x, -global_shift_y)

    if config.plotting.debug:
        plot_single_frame(
            frame.frame.data,
            starfield=frame.starfield,
            markersize=2 * frame.starfield.detection_metadata.pixel_fwhm,
            output_file=config.runtime.output_dir
            / f"{frame.index}_side_kernel_0_init.png",
        )

    # Update the WCS with the global shift (no limiting magnitude — need all stars for refinement)
    update_starfield_wcs(frame, updated_wcs_model, limiting_magnitude=None)

    if config.plotting.debug:
        plot_single_frame(
            frame.frame.data,
            starfield=frame.starfield,
            markersize=2 * frame.starfield.detection_metadata.pixel_fwhm,
            output_file=config.runtime.output_dir
            / f"{frame.index}_side_kernel_1_global.png",
        )

    logger.info("Refining WCS for sidereal frame with catalog stars")

    # Get catalog stars and sort by magnitude (brightest first)
    catalog_stars = frame.starfield.catalog_stars
    catalog_stars.sort(
        key=lambda star: star.magnitude if star.magnitude is not None else float("inf")
    )

    # SNR + magnitude filtering (shared helper)
    filtered_catalog_stars, _limiting_mag = compute_snr_and_filter_stars(
        frame, catalog_stars
    )

    # --- WCS Refinement via fit_wcs_from_points ---
    # Measure actual star positions in the convolved image and refit WCS
    height, width = frame.frame.data.shape
    search_radius = 10  # pixels
    min_separation = 15  # minimum distance between used stars
    MIN_STARS_FOR_WCS = 4
    edge_margin = 15  # pixels from image edge

    filtered_star_data = []
    for star in filtered_catalog_stars:
        x, y = star.x, star.y
        if x is None or y is None:
            continue

        # Skip stars too close to already-used stars
        too_close = False
        for prev_det, _, _, _ in filtered_star_data:
            dist = np.sqrt((x - prev_det.x) ** 2 + (y - prev_det.y) ** 2)
            if dist < min_separation:
                too_close = True
                break
        if too_close:
            continue

        # Find local max in convolved image near catalog position
        x_min = max(0, int(x - search_radius))
        x_max = min(width, int(x + search_radius + 1))
        y_min = max(0, int(y - search_radius))
        y_max = min(height, int(y + search_radius + 1))

        local_region = convolved_image[y_min:y_max, x_min:x_max]
        if local_region.size == 0:
            continue

        max_idx = np.argmax(local_region)
        local_y, local_x = np.unravel_index(max_idx, local_region.shape)
        measured_x = float(x_min + local_x)
        measured_y = float(y_min + local_y)

        # Skip stars too close to the edge
        if (
            measured_x < edge_margin
            or measured_x > width - edge_margin
            or measured_y < edge_margin
            or measured_y > height - edge_margin
        ):
            continue

        detection = StarInImage(x=measured_x, y=measured_y, counts=star.counts)
        filtered_star_data.append((detection, star, measured_x, measured_y))

    logger.info(
        "Found %d well-separated, high-SNR stars for sidereal WCS refinement",
        len(filtered_star_data),
    )

    if len(filtered_star_data) < MIN_STARS_FOR_WCS:
        logger.warning(
            "Not enough stars (%d) for WCS refit, returning shifted WCS",
            len(filtered_star_data),
        )
        return updated_wcs_model

    # MAD-based outlier rejection on shift magnitudes (shared helper)
    shifts = []
    for _detection, star, measured_x, measured_y in filtered_star_data:
        dx = measured_x - star.x
        dy = measured_y - star.y
        shift_mag = np.sqrt(dx * dx + dy * dy)
        shifts.append({
            "magnitude": shift_mag,
            "star": star,
            "measured_x": measured_x,
            "measured_y": measured_y,
        })

    world_coords, pixel_coords = reject_outlier_shifts_by_mad(
        shifts, min_stars=MIN_STARS_FOR_WCS
    )

    if len(world_coords) < MIN_STARS_FOR_WCS:
        logger.warning(
            "Not enough stars (%d) after outlier rejection for WCS refit",
            len(world_coords),
        )
        return updated_wcs_model

    # Check spatial coverage
    star_positions = np.array(pixel_coords)
    coverage_metrics = calculate_spatial_coverage(star_positions, frame.frame.data.shape)
    logger.info("Spatial coverage metrics: %s", coverage_metrics)

    if (
        coverage_metrics["quadrant_coverage"] < 2
        or coverage_metrics["convex_hull_area_ratio"] < 0.15
    ):
        logger.warning(
            "Poor spatial distribution (quadrants: %d/4, hull ratio: %.2f). Using shifted WCS.",
            coverage_metrics["quadrant_coverage"],
            coverage_metrics["convex_hull_area_ratio"],
        )
        return updated_wcs_model

    # Fit and validate WCS (shared helper)
    return fit_and_validate_wcs(
        world_coords,
        pixel_coords,
        frame.frame.data.shape,
        fallback_wcs=updated_wcs_model,
        sip_refit_order=config.astrometry.sip_refit_order,
        sip_refit_enabled=config.astrometry.sip_refit_enabled,
    )


# ---------------------------------------------------------------------------
# Rate-track: catalog-star refinement
# ---------------------------------------------------------------------------


def refine_wcs_with_catalog_stars(
    frame: RateTrackFrame, convolved_image: np.ndarray
) -> WCSModel:
    """Refine WCS using catalog stars from brightest to dimmest.

    Args:
        frame (RateTrackFrame): The frame containing the stars.
        convolved_image (np.ndarray): The convolved image.

    Returns:
        WCSModel: The refined WCS model, or None if refinement failed.
    """
    config = get_config()
    logger.info("Second pass: Refining WCS with catalog stars")
    vk_cfg = config.streak.variable_kernel

    # Check whether variable, distortion-aware kernels should be used for this frame.
    use_variable_kernel = bool(getattr(frame.streak, "use_variable_kernel", False))
    astropy_wcs_for_kernels = None
    if use_variable_kernel:
        if frame.starfield.wcs is None:
            logger.warning(
                "Variable kernels requested for frame %d but starfield WCS is missing; "
                "falling back to single-kernel refinement.",
                frame.index,
            )
            use_variable_kernel = False
        else:
            astropy_wcs_for_kernels = frame.starfield.wcs.to_astropy_wcs()
            if astropy_wcs_for_kernels is None:
                logger.warning(
                    "Variable kernels requested for frame %d but WCS conversion failed; "
                    "falling back to single-kernel refinement.",
                    frame.index,
                )
                use_variable_kernel = False
            else:
                # Ensure image dimensions are available for Jacobian sampling
                height, width = frame.frame.data.shape
                astropy_wcs_for_kernels.array_shape = (height, width)
                logger.info(
                    "Using variable, distortion-aware streak kernels for frame %d",
                    frame.index,
                )
                if config.plotting.debug:
                    try:
                        plot_variable_kernel_grid(
                            frame,
                            astropy_wcs_for_kernels,
                            nx=vk_cfg.diagnostics_grid_nx,
                            ny=vk_cfg.diagnostics_grid_ny,
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to generate variable-kernel grid diagnostic for frame %d: %s",
                            frame.index,
                            e,
                        )

    # Get catalog stars and sort by magnitude (brightest first)
    catalog_stars = frame.starfield.catalog_stars
    catalog_stars.sort(
        key=lambda star: star.magnitude if star.magnitude is not None else float("inf")
    )

    # SNR + magnitude filtering (shared helper — with conservative pre-filter for rate-track)
    filtered_catalog_stars, _limiting_mag = compute_snr_and_filter_stars(
        frame, catalog_stars, conservative_mag_cutoff=16.5
    )

    # Minimum separation between stars to use for WCS refinement
    min_separation = 15

    # Get image dimensions
    height, width = frame.frame.data.shape

    # Calculate the safety margin based on streak properties
    # This ensures we don't use stars that might be cut off at the edges
    streak_length = frame.streak.pixel_length
    streak_angle_rad = frame.streak.radian_angle()

    # Calculate the maximum possible projection of the streak in x and y directions
    dx_max = abs(streak_length * np.cos(streak_angle_rad))
    dy_max = abs(streak_length * np.sin(streak_angle_rad))

    # Set safety margin to half the streak length in each direction
    safety_margin_x = dx_max / 2
    safety_margin_y = dy_max / 2

    # Ensure minimum safety margin
    min_margin = 15  # pixels
    safety_margin_x = max(safety_margin_x, min_margin)
    safety_margin_y = max(safety_margin_y, min_margin)

    logger.info(
        f"Using safety margins: x={safety_margin_x:.1f}, y={safety_margin_y:.1f} pixels to avoid edge-cut streaks"
    )

    # Instead of just storing detections, store (detection, star, measured_x, measured_y) tuples
    # during the filtering process
    filtered_star_data = []
    diag_star_counter = 0
    variable_kernel_used_count = 0

    # Maximum number of stars to process for WCS refinement
    MAX_STARS_FOR_REFINEMENT = 250

    # Process filtered catalog stars from brightest to dimmest
    for star in filtered_catalog_stars:
        # Early termination if we have enough stars
        if len(filtered_star_data) >= MAX_STARS_FOR_REFINEMENT:
            logger.info(
                f"Reached maximum stars for refinement ({MAX_STARS_FOR_REFINEMENT}), stopping early"
            )
            break
        # Skip stars that are too close to already processed stars
        too_close = False
        for processed_data in filtered_star_data:
            processed_detection = processed_data[0]  # Get the detection from the tuple
            dist = np.sqrt(
                (star.x - processed_detection.x) ** 2
                + (star.y - processed_detection.y) ** 2
            )
            if dist < min_separation:
                too_close = True
                break

        if too_close:
            continue

        # Get current position
        x, y = star.x, star.y

        measured_x = None
        measured_y = None
        variable_kernel_used = False

        if (
            use_variable_kernel
            and astropy_wcs_for_kernels is not None
            and frame.streak is not None
        ):
            # Use a local, distortion-aware kernel and correlate in a cutout around the star.
            streak = frame.streak

            # Define a cutout window large enough to contain the streak plus margin.
            length_pixels = streak.pixel_length + 2 * streak.fwhm
            width_pixels = streak.fwhm * 4
            half_size = int(max(length_pixels, width_pixels) / 2) + 4

            x0 = round(x)
            y0 = round(y)
            x_min = max(0, x0 - half_size)
            x_max = min(width, x0 + half_size + 1)
            y_min = max(0, y0 - half_size)
            y_max = min(height, y0 + half_size + 1)

            if x_max > x_min and y_max > y_min:
                image_cutout = frame.frame.data[y_min:y_max, x_min:x_max]

                try:
                    local_kernel = get_local_streak_kernel(
                        astropy_wcs_for_kernels,
                        streak,
                        x=float(x),
                        y=float(y),
                        scale_width=True,
                        upsample=100,
                        halo_fwhm=None,
                        halo_level=1e-3,
                        verbose=False,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to build local streak kernel at (%.1f, %.1f) for frame %d: %s. "
                        "Falling back to global kernel at this star.",
                        x,
                        y,
                        frame.index,
                        e,
                    )
                    local_kernel = None

                if local_kernel is not None:
                    correlation = convolve(image_cutout, local_kernel, mode="same")
                    if correlation.size > 0:
                        max_idx = int(np.argmax(correlation))
                        local_y, local_x = np.unravel_index(max_idx, correlation.shape)
                        measured_x = x_min + local_x
                        measured_y = y_min + local_y
                        variable_kernel_used = True

                        if (
                            config.plotting.debug
                            and diag_star_counter < vk_cfg.diagnostics_max_stars
                        ):
                            try:
                                plot_variable_kernel_star_diagnostic(
                                    frame,
                                    image_cutout,
                                    local_kernel,
                                    correlation,
                                    x_min,
                                    y_min,
                                    measured_x,
                                    measured_y,
                                    diag_star_counter,
                                )
                                diag_star_counter += 1
                            except Exception as e:
                                logger.warning(
                                    "Failed to generate variable-kernel star diagnostic for frame %d, star %.1f/%.1f: %s",
                                    frame.index,
                                    x,
                                    y,
                                    e,
                                )

        if measured_x is None or measured_y is None:
            # Fallback / default behavior: use the globally convolved image near the catalog position.
            search_radius = 10  # pixels
            x_min, x_max = max(0, int(x - search_radius)), min(
                width, int(x + search_radius + 1)
            )
            y_min, y_max = max(0, int(y - search_radius)), min(
                height, int(y + search_radius + 1)
            )

            local_region = convolved_image[y_min:y_max, x_min:x_max]
            if local_region.size == 0:
                continue

            max_idx = np.argmax(local_region)
            local_y, local_x = np.unravel_index(max_idx, local_region.shape)
            measured_x = x_min + local_x
            measured_y = y_min + local_y

        # Double-check that the measured position isn't too close to the edge
        if (
            measured_x < safety_margin_x
            or measured_x > width - safety_margin_x
            or measured_y < safety_margin_y
            or measured_y > height - safety_margin_y
        ):
            logger.debug(
                f"Skipping star with measured position ({measured_x:.1f}, {measured_y:.1f}) - too close to frame edge"
            )
            continue

        # Counts are filled in one batched aperture pass after the loop —
        # they don't influence acceptance, and a per-star photutils call
        # here cost ~21 ms x 250 stars per frame.
        detection = StarInImage(x=float(measured_x), y=float(measured_y), counts=0.0)

        # Store the detection along with the star and measured position
        filtered_star_data.append((detection, star, measured_x, measured_y))

        if variable_kernel_used:
            variable_kernel_used_count += 1

        logger.debug(
            f"Added star with magnitude {star.magnitude:.2f}, SNR {getattr(star, 'snr', 'N/A'):.1f} at ({measured_x:.1f}, {measured_y:.1f})"
        )

    if use_variable_kernel:
        logger.info(
            "Variable streak kernels contributed measurements for %d of %d catalog stars in frame %d",
            variable_kernel_used_count,
            len(filtered_catalog_stars),
            frame.index,
        )

    logger.info(
        f"Found {len(filtered_star_data)} well-separated, high-SNR stars for WCS refinement"
    )

    # Fill detection counts in one batched, shared-shape aperture pass
    # (identical aperture geometry to extract_counts_with_rectangular_aperture).
    if filtered_star_data:
        from photutils.aperture import RectangularAnnulus, RectangularAperture

        from senpai.engine.photometry.utils import _shared_shape_aperture_sums

        ap_width = frame.streak.fwhm * 4
        ap_length = frame.streak.pixel_length + frame.streak.fwhm * 2
        ap_theta = frame.streak.radian_angle() + np.pi / 2
        positions = np.array(
            [(d.x, d.y) for d, _, _, _ in filtered_star_data], dtype=float
        )
        flux_sum, bg_sum = _shared_shape_aperture_sums(
            frame.frame.data,
            positions,
            lambda p: [
                RectangularAperture(p, w=ap_width, h=ap_length, theta=ap_theta),
                RectangularAnnulus(
                    p,
                    w_in=ap_width,
                    w_out=ap_width + 4,
                    h_in=ap_length,
                    h_out=ap_length + 4,
                    theta=ap_theta,
                ),
            ],
        )
        aper_area = ap_width * ap_length
        bg_area = (ap_width + 4) * (ap_length + 4) - aper_area
        bg_per_pixel = np.where(bg_area > 0, bg_sum / bg_area, 0.0)
        counts = flux_sum - bg_per_pixel * aper_area
        for (detection, _, _, _), c in zip(filtered_star_data, counts, strict=True):
            detection.counts = float(c)

    # Update the detections list with just the detection objects
    frame.starfield.detections = [data[0] for data in filtered_star_data]

    # Define minimum number of stars needed for reliable WCS fitting
    MIN_STARS_FOR_WCS = 4

    # First pass - calculate all shifts
    shifts = []
    for detection, star, measured_x, measured_y in filtered_star_data:
        dx = measured_x - star.x
        dy = measured_y - star.y
        shift_magnitude = np.sqrt(dx * dx + dy * dy)
        shifts.append(
            {
                "dx": dx,
                "dy": dy,
                "magnitude": shift_magnitude,
                "detection": detection,
                "star": star,
                "measured_x": measured_x,
                "measured_y": measured_y,
            }
        )

    # MAD-based outlier rejection — rate-track also updates detections list
    if len(shifts) >= MIN_STARS_FOR_WCS:
        magnitudes = np.array([s["magnitude"] for s in shifts])
        median_magnitude = np.median(magnitudes)
        mad_magnitude = np.median(np.abs(magnitudes - median_magnitude))
        magnitude_threshold = 4.0

        good_shifts = []
        for shift in shifts:
            is_magnitude_outlier = (
                abs(shift["magnitude"] - median_magnitude)
                > magnitude_threshold * mad_magnitude
            )

            if not (is_magnitude_outlier):
                good_shifts.append(shift)
            else:
                logger.warning(
                    f"Excluding outlier star at ({shift['measured_x']:.1f}, {shift['measured_y']:.1f}), "
                    f"shift magnitude: {shift['magnitude']:.1f} (median: {median_magnitude:.1f}), "
                )
        # Update filtered_star_data with only good shifts
        filtered_star_data = [
            (s["detection"], s["star"], s["measured_x"], s["measured_y"])
            for s in good_shifts
        ]

        # Update the detections list
        frame.starfield.detections = [s["detection"] for s in good_shifts]

        logger.info(f"Filtered out {len(shifts) - len(good_shifts)} outlier shifts")

        world_coords = [(s["star"].ra, s["star"].dec) for s in good_shifts]
        pixel_coords = [(s["measured_x"], s["measured_y"]) for s in good_shifts]
    else:
        world_coords = [(star.ra, star.dec) for _, star, mx, my in filtered_star_data]
        pixel_coords = [(mx, my) for _, star, mx, my in filtered_star_data]

    if len(world_coords) < MIN_STARS_FOR_WCS:
        logger.warning(
            f"Not enough stars ({len(world_coords)}) for reliable WCS refinement. Minimum required: {MIN_STARS_FOR_WCS}"
        )
        return frame.starfield.wcs  # Return the original WCS model

    # Check spatial distribution of reference stars
    star_positions = np.array(pixel_coords)
    coverage_metrics = calculate_spatial_coverage(
        star_positions, frame.frame.data.shape
    )

    # Log the metrics
    logger.info(f"Spatial coverage metrics: {coverage_metrics}")

    # Check if coverage is too poor
    if (
        coverage_metrics["quadrant_coverage"] < 3
        or coverage_metrics["convex_hull_area_ratio"] < 0.3
    ):
        logger.warning(
            f"Poor spatial distribution of reference stars. "
            f"Quadrants covered: {coverage_metrics['quadrant_coverage']}/4, "
            f"Convex hull area ratio: {coverage_metrics['convex_hull_area_ratio']:.2f}"
        )

        # If coverage is really bad, you might want to return the original WCS
        if (
            coverage_metrics["quadrant_coverage"] < 2
            or coverage_metrics["convex_hull_area_ratio"] < 0.15
        ):
            logger.error(
                "Extremely poor spatial distribution of reference stars. Using original WCS."
            )
            return frame.starfield.wcs

    logger.info(
        f"Using {len(world_coords)} well-separated star positions for WCS fitting"
    )

    if config.plotting.debug:
        markersize = 2 * frame.seeing.pixel_fwhm if frame.seeing is not None else 10
        logger.info(f"config.runtime.output_dir / {frame.index}_kernel_2_torefit.png")
        plot_single_frame(
            frame.frame.data,
            starlist=StarListImage(
                detections=frame.starfield.detections,
                image_metadata=frame.starfield.image_metadata,
            ),
            markersize=markersize,
            output_file=config.runtime.output_dir
            / f"{frame.index}_kernel_2_torefit.png",
        )

    # Fit and validate WCS (shared helper)
    return fit_and_validate_wcs(
        world_coords,
        pixel_coords,
        frame.frame.data.shape,
        fallback_wcs=frame.starfield.wcs,
        sip_refit_order=config.astrometry.sip_refit_order,
        sip_refit_enabled=config.astrometry.sip_refit_enabled,
    )
