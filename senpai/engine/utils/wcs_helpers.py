"""Detection, matching, fitting, and shared helpers for WCS refinement."""

import logging

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull

from senpai.core.config import get_config
from senpai.engine.models.astrometry import WCSMetadata, WCSModel
from senpai.engine.models.metadata import StreakMetadata
from senpai.engine.models.starfield import StarInImage, StarInSpace
from senpai.engine.photometry.utils import (
    calculate_star_snrs_with_aperture_photometry,
    estimate_limiting_magnitude_from_photometry,
)
from senpai.engine.utils.wcs_ops import (
    catalog_stars_from_wcs,
    existing_stars_from_wcs,
    filter_catalog_stars_by_radius,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _local_maxima_above_floor(image, half, floor):
    """(ys, xs, values) of pixels above ``floor`` that equal the maximum of
    their (2*half+1)^2 window, matching ``maximum_filter(mode='constant')``
    semantics for positive floors (out-of-bounds zeros never beat an
    above-floor pixel). Offsets run nearest-first so almost every
    non-maximum dies on an immediate neighbor before the wide scans."""
    ys, xs = np.nonzero(image > floor)
    vals = image[ys, xs]
    h, w = image.shape
    for radius in range(1, half + 1):
        ring = [
            (dy, dx)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
            if max(abs(dy), abs(dx)) == radius
        ]
        for dy, dx in ring:
            if ys.size == 0:
                return ys, xs, vals
            yy, xx = ys + dy, xs + dx
            keep = np.ones(ys.size, dtype=bool)
            inb = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
            keep[inb] = vals[inb] >= image[yy[inb], xx[inb]]
            ys, xs, vals = ys[keep], xs[keep], vals[keep]
    return ys, xs, vals


def find_local_maxima(image, min_distance=30, threshold=None, max_detections=None):
    """
    Find local maxima in an image with minimum separation distance.

    Args:
        image: 2D numpy array
        min_distance: Minimum pixel separation between maxima
        threshold: Optional intensity threshold
        max_detections: Maximum number of detections to return (returns brightest ones)

    Returns:
        Array of (y, x) coordinates of maxima
    """
    size = 2 * min_distance + 1
    floor_base = max(0.0, float(threshold) if threshold is not None else 0.0)

    if max_detections is not None:
        # Fast path: only the brightest max_detections maxima are wanted, so
        # probe with a descending ladder of intensity floors and test only
        # above-floor pixels against their neighborhood — a full-frame
        # 61x61 maximum_filter costs ~1.3 s per call. Any maximum excluded
        # by a floor is dimmer than the ones found above it, so once enough
        # are found the brightest set is exact; identical to the filter
        # path (which this falls back to if the ladder never yields enough).
        sample = image[::8, ::8]
        for q in (99.99, 99.9, 99.0):
            floor = float(np.percentile(sample, q))
            if floor <= floor_base:
                break
            ys, xs, vals = _local_maxima_above_floor(image, min_distance, floor)
            if ys.size >= max_detections:
                order = np.argsort(-vals)[:max_detections]
                return np.column_stack((ys[order], xs[order]))

    # Apply threshold if provided
    if threshold is not None:
        mask = image > threshold
        filtered_image = image * mask
    else:
        filtered_image = image.copy()

    # Apply maximum filter
    maximum_filtered = maximum_filter(filtered_image, size=size, mode="constant")

    # Find points that are local maxima
    maxima = (filtered_image == maximum_filtered) & (filtered_image > 0)

    # Get coordinates and values of maxima in one step
    y_coords, x_coords = np.where(maxima)

    if len(y_coords) == 0:
        return np.array([])

    # Get values at these coordinates
    values = filtered_image[y_coords, x_coords]

    # Sort by intensity (brightest first)
    sort_indices = np.argsort(-values)  # Negative for descending order

    # Limit to max_detections if specified
    if max_detections is not None and max_detections < len(sort_indices):
        sort_indices = sort_indices[:max_detections]

    # Return sorted coordinates
    return np.column_stack((y_coords[sort_indices], x_coords[sort_indices]))


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------


def match_stars_to_detections(
    stars: list[StarInImage],
    detected_points: list[tuple[float, float]],
    max_distance: float = 20,
):
    """
    Match catalog stars to detected points using bipartite matching.

    Args:
        stars: List of StarInImage objects
        detected_points: Array of (y, x) coordinates from local maxima detection
        max_distance: Maximum allowed matching distance in pixels

    Returns:
        matched_pairs: List of (star_idx, detection_idx) pairs
        unmatched_stars: List of star indices with no match
        unmatched_detections: List of detection indices with no match
    """
    if not stars or len(detected_points) == 0:
        return [], list(range(len(stars))), list(range(len(detected_points)))

    # Distance matrix, vectorized: the python double loop cost ~2 s per call
    # against a full 18k-star catalog. None stars get infinite cost rows
    # (never assigned), as before.
    star_x = np.array([s.x if s is not None else np.nan for s in stars], dtype=float)
    star_y = np.array([s.y if s is not None else np.nan for s in stars], dtype=float)
    det = np.asarray(detected_points, dtype=float)  # rows are (y, x)
    cost_matrix = np.hypot(
        star_x[:, None] - det[None, :, 1], star_y[:, None] - det[None, :, 0]
    )
    cost_matrix[~np.isfinite(cost_matrix)] = np.inf

    # If all costs are infinite, return empty matches
    if not np.isfinite(cost_matrix).any():
        return [], list(range(len(stars))), list(range(len(detected_points)))

    # Solve the assignment problem
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except ValueError:
        return [], list(range(len(stars))), list(range(len(detected_points)))

    # Filter out assignments with distance exceeding max_distance
    matched_pairs = []
    unmatched_stars = list(range(len(stars)))
    unmatched_detections = list(range(len(detected_points)))

    for i, j in zip(row_ind, col_ind, strict=False):
        if cost_matrix[i, j] <= max_distance:
            matched_pairs.append((i, j))
            unmatched_stars.remove(i)
            unmatched_detections.remove(j)

    return matched_pairs, unmatched_stars, unmatched_detections


# ---------------------------------------------------------------------------
# Photometry / aperture helpers
# ---------------------------------------------------------------------------


def extract_counts_with_rectangular_aperture(
    image, x, y, streak: StreakMetadata, background_annulus=True
):
    """
    Extract counts from an image using a rectangular aperture aligned with a streak.

    Args:
        image: 2D numpy array containing the image data
        x: x-coordinate of the star center
        y: y-coordinate of the star center
        streak: Streak object containing length, width, and angle information
        background_annulus: Whether to subtract local background using an annulus

    Returns:
        counts: Background-subtracted counts within the aperture
        background: Local background level (per pixel)
    """
    from photutils.aperture import RectangularAnnulus, RectangularAperture

    # Create rectangular aperture aligned with the streak
    width = streak.fwhm * 4
    length = streak.pixel_length + streak.fwhm * 2
    theta = streak.radian_angle() + np.pi / 2  # Assuming angle is in radians

    # Create the aperture
    aperture = RectangularAperture((x, y), w=width, h=length, theta=theta)

    # Create background annulus if requested
    if background_annulus:
        # Make the annulus slightly larger than the aperture
        bg_aperture = RectangularAnnulus(
            (x, y),
            w_in=width,
            w_out=width + 4,
            h_in=length,
            h_out=length + 4,
            theta=theta,
        )

    # Perform photometry
    from photutils.aperture import aperture_photometry

    phot_table = aperture_photometry(image, aperture)
    aperture_sum = float(phot_table["aperture_sum"][0])

    # Calculate background if requested
    if background_annulus:
        bg_phot_table = aperture_photometry(image, bg_aperture)
        bg_sum = float(bg_phot_table["aperture_sum"][0])
        bg_area = bg_aperture.area
        aperture_area = aperture.area

        # Calculate background per pixel
        background = bg_sum / bg_area

        # Subtract background from aperture sum
        counts = aperture_sum - (background * aperture_area)
    else:
        background = 0.0
        counts = aperture_sum

    return counts, background


# ---------------------------------------------------------------------------
# Spatial coverage
# ---------------------------------------------------------------------------


def calculate_spatial_coverage(positions, image_shape):
    """Calculate metrics for spatial coverage of reference stars.

    Args:
        positions: Array of (x, y) coordinates
        image_shape: (height, width) of the image

    Returns:
        Dictionary of coverage metrics
    """
    height, width = image_shape
    metrics = {}

    # 1. Quadrant coverage - how many quadrants of the image have stars
    center_x, center_y = width / 2, height / 2
    quadrants = [0, 0, 0, 0]  # [top-right, top-left, bottom-left, bottom-right]

    for x, y in positions:
        if x >= center_x and y < center_y:
            quadrants[0] = 1
        elif x < center_x and y < center_y:
            quadrants[1] = 1
        elif x < center_x and y >= center_y:
            quadrants[2] = 1
        elif x >= center_x and y >= center_y:
            quadrants[3] = 1

    metrics["quadrant_coverage"] = sum(quadrants)

    # 2. Convex hull area ratio - area of the convex hull of stars divided by image area
    if len(positions) >= 3:  # Need at least 3 points for a convex hull
        try:
            hull = ConvexHull(positions)
            hull_area = hull.volume  # In 2D, volume is area
            image_area = width * height
            metrics["convex_hull_area_ratio"] = hull_area / image_area
        except Exception as e:
            logger.warning(f"Could not calculate convex hull: {e}")
            metrics["convex_hull_area_ratio"] = 0
    else:
        metrics["convex_hull_area_ratio"] = 0

    # 3. Standard deviation of x and y coordinates normalized by image dimensions
    # Higher values indicate better spread
    if len(positions) >= 2:
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        metrics["normalized_x_std"] = np.std(x_coords) / width
        metrics["normalized_y_std"] = np.std(y_coords) / height
    else:
        metrics["normalized_x_std"] = 0
        metrics["normalized_y_std"] = 0

    # 4. Distance to nearest edge - minimum distance of any star to image edge
    min_edge_distance = float("inf")
    for x, y in positions:
        edge_dist = min(x, y, width - x, height - y)
        min_edge_distance = min(min_edge_distance, edge_dist)

    metrics["min_edge_distance"] = min_edge_distance
    metrics["normalized_min_edge_distance"] = min_edge_distance / min(width, height)

    return metrics


# ---------------------------------------------------------------------------
# SIP order selection
# ---------------------------------------------------------------------------


def determine_optimal_sip_order(
    world_coords, pixel_coords, image_shape, max_order: int = 3
):
    """Determine optimal SIP order based on spatial coverage and number of reference stars.

    Args:
        world_coords: List of (ra, dec) pairs
        pixel_coords: List of (x, y) pairs
        image_shape: (height, width) of image

    Returns:
        int: Optimal SIP order (1-5)
    """
    n_stars = len(world_coords)

    # Start with base order based on number of stars
    if n_stars < 6:
        # Not enough stars for higher order
        sip_order = 1
    elif n_stars < 10:
        # Limited stars, use low order
        sip_order = 2
    elif n_stars < 20:
        # Moderate number of stars
        sip_order = 3
    elif n_stars < 40:
        # Good number of stars
        sip_order = 4
    else:
        # Many stars, can use higher order
        sip_order = 5

    # Check spatial coverage
    coverage_metrics = calculate_spatial_coverage(pixel_coords, image_shape)

    # If poor coverage, reduce order to avoid overfitting
    if (
        coverage_metrics["quadrant_coverage"] < 3
        or coverage_metrics["convex_hull_area_ratio"] < 0.2
    ):
        sip_order = max(1, sip_order - 1)

    return min(sip_order, max_order)


# ---------------------------------------------------------------------------
# Shared helpers for sidereal / rate-track refinement deduplication
# ---------------------------------------------------------------------------


def compute_snr_and_filter_stars(
    frame,
    catalog_stars: list[StarInSpace],
    min_snr: float = 8.0,
    min_stars_to_preserve: int = 6,
    margin: float = 0.5,
    conservative_mag_cutoff: float | None = None,
) -> tuple[list[StarInSpace], float | None]:
    """Calculate SNRs, estimate limiting magnitude, and filter catalog stars.

    Consolidates the identical photometry + magnitude-filter blocks used in both
    the sidereal and rate-track refinement paths.

    Args:
        frame: SiderealFrame or RateTrackFrame.
        catalog_stars: Catalog stars sorted by magnitude (brightest first).
        min_snr: Minimum SNR threshold for keeping a star.
        min_stars_to_preserve: Always keep at least this many brightest stars.
        margin: Magnitude margin subtracted from the limiting magnitude.
        conservative_mag_cutoff: If set, pre-filter stars dimmer than this
            magnitude before running photometry (saves computation).

    Returns:
        (filtered_stars, limiting_magnitude) where *filtered_stars* are the
        stars that passed SNR + magnitude gating.
    """
    stars_for_photometry = catalog_stars

    # Optional pre-filter by conservative magnitude cutoff
    if conservative_mag_cutoff is not None:
        initial_count = len(stars_for_photometry)
        stars_for_photometry = [
            star
            for star in stars_for_photometry
            if star.magnitude is None or star.magnitude <= conservative_mag_cutoff
        ]
        if initial_count > len(stars_for_photometry):
            logger.info(
                "Pre-filtered catalog stars from %d to %d using conservative magnitude cutoff %.2f",
                initial_count,
                len(stars_for_photometry),
                conservative_mag_cutoff,
            )

    # Limit the number of stars for photometry to avoid excessive computation
    MAX_STARS_FOR_PHOTOMETRY = 500
    if len(stars_for_photometry) > MAX_STARS_FOR_PHOTOMETRY:
        stars_for_photometry = stars_for_photometry[:MAX_STARS_FOR_PHOTOMETRY]
        logger.info(
            "Limited stars for photometry to %d brightest stars",
            MAX_STARS_FOR_PHOTOMETRY,
        )

    # Calculate proper SNRs using aperture photometry
    star_snr_results = calculate_star_snrs_with_aperture_photometry(
        frame, stars_for_photometry
    )

    # Store SNR with each star for later use
    for star, snr, counts in star_snr_results:
        star.snr = snr
        star.counts = counts

    # Filter stars by SNR
    filtered_catalog_stars = [
        star for star, snr, _ in star_snr_results if snr >= min_snr
    ]

    # Estimate limiting magnitude using shared photometry utility (3-sigma by default)
    limiting_magnitude = estimate_limiting_magnitude_from_photometry(
        frame, star_snr_results, min_snr=3.0
    )

    # Store the limiting magnitude in the starfield
    if (
        hasattr(frame.starfield, "limiting_magnitude")
        and limiting_magnitude is not None
    ):
        frame.starfield.limiting_magnitude = limiting_magnitude

    # Filter out stars that are too dim (beyond the limiting magnitude)
    # BUT always preserve the brightest N stars to ensure we have enough for WCS fitting
    if limiting_magnitude is not None:
        cutoff_mag = limiting_magnitude - margin
        before_count = len(filtered_catalog_stars)

        sorted_by_mag = sorted(
            filtered_catalog_stars,
            key=lambda s: s.magnitude if s.magnitude is not None else float("inf"),
        )
        brightest_stars = sorted_by_mag[:min_stars_to_preserve]
        other_stars = sorted_by_mag[min_stars_to_preserve:]

        filtered_other_stars = [
            star
            for star in other_stars
            if star.magnitude is None or star.magnitude <= cutoff_mag
        ]

        filtered_catalog_stars = brightest_stars + filtered_other_stars
        after_count = len(filtered_catalog_stars)

        logger.info(
            "Filtered catalog stars for WCS refinement from %d to %d using limiting magnitude %.2f (margin %.2f mag), "
            "preserving %d brightest stars",
            before_count,
            after_count,
            cutoff_mag,
            margin,
            min(min_stars_to_preserve, before_count),
        )

    logger.info(
        "Filtered catalog from %d to %d stars above SNR threshold",
        len(catalog_stars),
        len(filtered_catalog_stars),
    )

    return filtered_catalog_stars, limiting_magnitude


def reject_outlier_shifts_by_mad(
    shifts: list[dict],
    min_stars: int = 4,
    mad_threshold: float = 4.0,
) -> tuple[list[tuple], list[tuple]]:
    """MAD-based outlier rejection on shift magnitudes.

    Args:
        shifts: List of dicts, each with keys ``magnitude``, ``star``,
            ``measured_x``, ``measured_y``.
        min_stars: Minimum number of shifts required to attempt MAD rejection.
        mad_threshold: Number of MADs beyond which a shift is an outlier.

    Returns:
        (world_coords, pixel_coords) lists for inlier shifts.
    """
    world_coords: list[tuple] = []
    pixel_coords: list[tuple] = []

    if len(shifts) >= min_stars:
        magnitudes = np.array([s["magnitude"] for s in shifts])
        median_mag = np.median(magnitudes)
        mad_mag = np.median(np.abs(magnitudes - median_mag))

        good_shifts = []
        for shift in shifts:
            if abs(shift["magnitude"] - median_mag) > mad_threshold * mad_mag:
                logger.warning(
                    "Excluding outlier star at (%.1f, %.1f), shift: %.1f (median: %.1f)",
                    shift["measured_x"],
                    shift["measured_y"],
                    shift["magnitude"],
                    median_mag,
                )
            else:
                good_shifts.append(shift)

        logger.info("Filtered out %d outlier shifts", len(shifts) - len(good_shifts))

        for shift in good_shifts:
            world_coords.append((shift["star"].ra, shift["star"].dec))
            pixel_coords.append((shift["measured_x"], shift["measured_y"]))
    else:
        for shift in shifts:
            world_coords.append((shift["star"].ra, shift["star"].dec))
            pixel_coords.append((shift["measured_x"], shift["measured_y"]))

    return world_coords, pixel_coords


def fit_and_validate_wcs(
    world_coords: list[tuple],
    pixel_coords: list[tuple],
    image_shape: tuple[int, int],
    fallback_wcs: WCSModel,
    sip_refit_order: int,
    sip_refit_enabled: bool,
    max_acceptable_shift: float = 50.0,
) -> WCSModel:
    """Fit a new WCS from matched points and validate against a fallback.

    Consolidates the identical fit-then-validate blocks used in both the
    sidereal and rate-track refinement paths.

    Args:
        world_coords: List of (ra, dec) pairs.
        pixel_coords: List of (x, y) pairs.
        image_shape: (height, width).
        fallback_wcs: WCS to return if the fit is rejected.
        sip_refit_order: Maximum SIP order from config.
        sip_refit_enabled: Whether SIP refit is enabled in config.
        max_acceptable_shift: Maximum pixel shift between fallback and new WCS
            at image corners/center before the fit is rejected.

    Returns:
        The refined WCSModel, or *fallback_wcs* if validation fails.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.wcs.utils import fit_wcs_from_points

    ra_values = [wc[0] for wc in world_coords]
    dec_values = [wc[1] for wc in world_coords]
    sky_coords = SkyCoord(ra_values, dec_values, unit=u.deg)

    x_values = np.array([pc[0] for pc in pixel_coords])
    y_values = np.array([pc[1] for pc in pixel_coords])

    # Convert to FITS convention (1-indexed) if needed
    if x_values.min() < 1 or y_values.min() < 1:
        x_values = x_values + 1
        y_values = y_values + 1

    # Determine SIP order
    sip_degree = determine_optimal_sip_order(
        world_coords,
        pixel_coords,
        image_shape,
        max_order=sip_refit_order if sip_refit_enabled else 0,
    )
    logger.info(
        "Using SIP order %d for WCS fitting with %d reference stars",
        sip_degree,
        len(world_coords),
    )

    refined_astropy_wcs = fit_wcs_from_points(
        (x_values, y_values), sky_coords, proj_point="center", sip_degree=sip_degree
    )

    new_wcs_model = WCSModel.from_astropy_wcs(
        refined_astropy_wcs, image_shape=image_shape
    )

    # Consistency check: compare corners and center between fallback and refined WCS
    original_wcs = fallback_wcs.to_astropy_wcs()
    h, w = image_shape
    reference_pixels = [
        (0, 0),
        (w - 1, 0),
        (0, h - 1),
        (w - 1, h - 1),
        (w // 2, h // 2),
    ]
    original_world = original_wcs.all_pix2world(reference_pixels, 0)
    # quiet=True: fit_wcs_from_points fits no inverse SIP, so all_world2pix
    # inverts iteratively and raises NoConvergence on marginal fits — which
    # would kill the batch inside a *consistency check*. Take astropy's best
    # solution instead; a genuinely diverged inverse lands far from the
    # reference pixels and the max_shift test below rejects the refinement.
    new_pixels = refined_astropy_wcs.all_world2pix(original_world, 0, quiet=True)

    max_shift = max(
        np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        for (x1, y1), (x2, y2) in zip(reference_pixels, new_pixels, strict=False)
    )

    if max_shift > max_acceptable_shift:
        logger.warning(
            "Refined WCS differs too much from fallback WCS (max shift: %.1f pixels). "
            "Using fallback WCS.",
            max_shift,
        )
        return fallback_wcs

    logger.info(
        "Successfully refined WCS using %d catalog stars", len(world_coords)
    )
    return new_wcs_model


def update_starfield_wcs(
    frame,
    new_wcs: WCSModel,
    limiting_magnitude: float | None = None,
) -> None:
    """Apply a new WCS to a frame's starfield and refresh star positions.

    Consolidates the repeated 4-step WCS update pattern that appears multiple
    times across the sidereal and rate-track refinement paths.

    Args:
        frame: SiderealFrame or RateTrackFrame.
        new_wcs: The new WCS model to apply.
        limiting_magnitude: Optional limiting magnitude for catalog query.
    """
    config = get_config()

    frame.starfield.wcs = new_wcs
    frame.starfield.wcs_metadata = WCSMetadata.from_wcsmodel(new_wcs)

    # Update astrometric fit star positions
    frame.starfield.astrometric_fit_stars = existing_stars_from_wcs(
        new_wcs, frame.starfield.astrometric_fit_stars
    )

    # Query/refresh catalog stars
    catalog_stars = catalog_stars_from_wcs(new_wcs, limiting_magnitude=limiting_magnitude)

    # Apply radius filtering if configured
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
        new_wcs, catalog_stars.stars
    )
