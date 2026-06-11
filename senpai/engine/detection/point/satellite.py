"""
Satellite point source detection in rate track, assuming WCS already fit
"""

import logging
import warnings
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import fitting, models
from photutils.detection.daofinder import _DAOStarFinderCatalog, _StarFinderKernel
from photutils.utils.exceptions import NoDetectionsWarning
from scipy.ndimage import median_filter
from scipy.signal import fftconvolve

from senpai.core.config import get_config
from senpai.engine.detection.streak.masking import percent_difference
from senpai.engine.models.senpai import RateTrackFrame
from senpai.engine.models.starfield import SatelliteInImage, SatelliteListImage
from senpai.engine.plotting.images import plot_single_frame
from senpai.engine.utils.stats import fft_workers, robust_background_stats

logger = logging.getLogger(__name__)


def _median_filter_3x3(image: np.ndarray) -> np.ndarray:
    """3x3 median filter for hot-pixel removal.

    cv2.medianBlur is ~70x faster than scipy.ndimage.median_filter on large
    float32 frames and interior-identical (only the 1-px border differs:
    replicate vs reflect padding — border detections are discarded anyway).
    Falls back to scipy for dtypes cv2 doesn't support.
    """
    if image.dtype == np.float32:
        return cv2.medianBlur(np.ascontiguousarray(image), 3)
    return median_filter(image, size=3)


def _local_maxima_above(
    convolved: np.ndarray, footprint: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ys, xs, values) of local maxima within ``footprint`` above ``threshold``.

    Matches photutils ``find_peaks`` semantics exactly for positive
    thresholds — ``(data == maximum_filter(data, footprint,
    mode='constant', cval=0)) & (data > threshold)`` — but evaluates the
    neighborhood test only at above-threshold pixels, which is ~30x faster
    than the full-frame maximum_filter for detection-sized footprints.
    Out-of-bounds neighbors are the filter's constant zeros, which can
    never beat an above-(positive-)threshold pixel, so they are skipped.
    """
    ys, xs = np.nonzero(convolved > threshold)
    vals = convolved[ys, xs]
    h, w = convolved.shape
    cy, cx = (footprint.shape[0] - 1) // 2, (footprint.shape[1] - 1) // 2
    fy, fx = np.nonzero(footprint)
    # Nearest offsets first: in a kernel-smoothed image almost every
    # non-maximum candidate is beaten by an immediate neighbor, so the
    # candidate set collapses within the first ring and the remaining
    # offsets scan a tiny survivor list.
    offsets = sorted(
        zip(fy - cy, fx - cx, strict=True), key=lambda d: max(abs(d[0]), abs(d[1]))
    )
    for dy, dx in offsets:
        if (dy == 0 and dx == 0) or ys.size == 0:
            continue
        yy, xx = ys + dy, xs + dx
        keep = np.ones(ys.size, dtype=bool)
        inbounds = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        keep[inbounds] = vals[inbounds] >= convolved[yy[inbounds], xx[inbounds]]
        ys, xs, vals = ys[keep], xs[keep], vals[keep]
    return ys, xs, vals


def _dao_sources_at_threshold(
    data_sub: np.ndarray,
    convolved: np.ndarray,
    kernel,
    candidate_xy: np.ndarray,
    candidate_vals: np.ndarray,
    threshold: float,
    *,
    sharplo: float,
    sharphi: float,
    roundlo: float,
    roundhi: float,
):
    """Exact ``DAOStarFinder(...)(data_sub)`` result at ``threshold``,
    reusing a shared convolution and precomputed local maxima.

    The adaptive-threshold search varies only the threshold scalar, but
    ``DAOStarFinder`` recomputes the identical kernel convolution and
    full-frame peak search on every call. Local maxima above a higher
    threshold are exactly the precomputed maxima with values above it, so
    each attempt reduces to a 1D mask plus DAO's per-candidate property
    filters. Equivalence to DAOStarFinder is pinned by a regression test
    (photutils is version-locked; the test fails loudly if internals move).
    """
    selected = candidate_vals > threshold * kernel.relerr
    if not np.any(selected):
        return None
    catalog = _DAOStarFinderCatalog(
        data_sub,
        convolved,
        candidate_xy[selected],
        threshold,
        kernel,
        sharplo=sharplo,
        sharphi=sharphi,
        roundlo=roundlo,
        roundhi=roundhi,
        brightest=None,
        peakmax=None,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NoDetectionsWarning)
        catalog = catalog.apply_all_filters()
    if catalog is None:
        return None
    return catalog.to_table()


def cutout_gauss(
    sub_image: np.ndarray, pixel_seeing: float, plot: bool = False
) -> Tuple[float, float, float]:
    """
    Fit a 2D Gaussian to a sub-image and return FWHM measurements.

    Args:
        sub_image: Small image cutout centered on a detection
        pixel_seeing: Expected seeing in pixels (FWHM)
        plot: Whether to generate diagnostic plots

    Returns:
        Tuple of (FWHM_x, FWHM_y, average_FWHM)
    """
    size = sub_image.shape[0]

    # Remove background to improve Gaussian fitting
    sub_image = sub_image - np.median(sub_image).astype(sub_image.dtype)

    # Convert FWHM to standard deviation for Gaussian2D model
    # FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.355 * sigma
    # So sigma = FWHM / 2.355
    sigma_seeing = pixel_seeing / (2 * np.sqrt(2 * np.log(2)))

    # Set reasonable bounds: sigma should be between 0.1 and size/2 pixels
    min_sigma = 0.1
    max_sigma = size / 2.0

    # Ensure initial guess is within bounds
    sigma_init = max(min_sigma, min(max_sigma, sigma_seeing))

    # Fit a 2D Gaussian with bounds
    p_init = models.Gaussian2D(
        amplitude=np.max(sub_image),
        x_mean=size // 2,
        y_mean=size // 2,
        x_stddev=sigma_init,
        y_stddev=sigma_init,
    )

    # Set bounds on parameters
    p_init.x_stddev.bounds = (min_sigma, max_sigma)
    p_init.y_stddev.bounds = (min_sigma, max_sigma)
    p_init.x_mean.bounds = (0, size)
    p_init.y_mean.bounds = (0, size)
    p_init.amplitude.bounds = (0, None)

    fit_p = fitting.LevMarLSQFitter()
    y, x = np.mgrid[:size, :size]

    try:
        fitted_p = fit_p(p_init, x, y, sub_image)

        # Check if fit converged properly
        if fit_p.fit_info["ierr"] not in [1, 2, 3, 4]:
            # Fit may not have converged, check if values are reasonable
            if (
                fitted_p.x_stddev.value > max_sigma * 2
                or fitted_p.y_stddev.value > max_sigma * 2
                or fitted_p.x_stddev.value < min_sigma
                or fitted_p.y_stddev.value < min_sigma
            ):
                raise ValueError(
                    f"Fit produced unrealistic sigma values: "
                    f"x_stddev={fitted_p.x_stddev.value:.2f}, "
                    f"y_stddev={fitted_p.y_stddev.value:.2f}"
                )
    except Exception as e:
        raise ValueError(f"Gaussian fit failed: {str(e)}") from e

    sub_img_fit = fitted_p(x, y)

    if plot:
        _, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(sub_image, origin="lower", cmap="viridis")
        ax[0].set_title("Original Sub-Image")
        ax[1].imshow(sub_img_fit, origin="lower", cmap="viridis")
        ax[1].set_title("Fitted Gaussian Model")
        plt.savefig("gaussian_fit.png")
        plt.close("all")

    # Extract the FWHM in pixels (convert from standard deviation)
    fwhm_x = fitted_p.x_stddev.value * 2 * np.sqrt(2 * np.log(2))
    fwhm_y = fitted_p.y_stddev.value * 2 * np.sqrt(2 * np.log(2))
    fwhm_avg = (fwhm_x + fwhm_y) / 2

    return fwhm_x, fwhm_y, fwhm_avg


def find_two_brightest_points(
    arr: np.ndarray,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Find the coordinates of the two brightest points in a 2D array.

    Args:
        arr: 2D numpy array

    Returns:
        Coordinates of the two brightest points as ((y1, x1), (y2, x2))
    """
    # Find the coordinates of the brightest point
    brightest_point_1 = np.unravel_index(np.argmax(arr), arr.shape)

    # Copy the array and set the brightest point to a very low value
    arr_copy = arr.copy()
    arr_copy[brightest_point_1] = np.min(arr)

    # Find the coordinates of the second brightest point
    brightest_point_2 = np.unravel_index(np.argmax(arr_copy), arr_copy.shape)

    return brightest_point_1, brightest_point_2


def euclidean_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    """
    Calculate the Euclidean distance between two points.

    Args:
        point1: Coordinates of the first point (y, x)
        point2: Coordinates of the second point (y, x)

    Returns:
        Euclidean distance
    """
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def generate_cutout(
    frame: np.ndarray, detection: Tuple[float, float], side: int, plot: bool = False
) -> np.ndarray:
    """
    Generate a square cutout centered on a detection.

    Args:
        frame: Full image array
        detection: (x, y) coordinates of the detection
        side: Half-width of the cutout in pixels

    Returns:
        Square cutout of the image
    """
    x, y = detection
    y_min = max(0, int(round(y) - side))
    y_max = min(int(round(y) + side), frame.shape[0])
    x_min = max(0, int(round(x) - side))
    x_max = min(int(round(x) + side), frame.shape[1])

    if plot:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(frame[y_min:y_max, x_min:x_max], origin="lower", cmap="viridis")
        plt.savefig("cutout.png")
        plt.close("all")

    return frame[y_min:y_max, x_min:x_max].copy()


def filter_point_sources(
    frame: RateTrackFrame,
    detections: List[Tuple[float, float]],
    pixel_seeing: float,
    hot_pixel_threshold: float = 0.35,
) -> List[Tuple[float, float, float]]:
    """
    Filter out hot pixels, extended sources, and other non-point-like detections.

    Args:
        frame: RateTrackFrame containing the image data
        detections: List of (x, y) coordinates for potential point sources
        pixel_seeing: Expected seeing in pixels
        hot_pixel_threshold: Maximum fraction of flux allowed in the brightest pixel

    Returns:
        List of filtered (x, y) coordinates for confirmed point sources
    """
    config = get_config()
    filtered_detections = []
    cutout_size = int(3 * pixel_seeing)

    logger.info(f"Evaluating {len(detections)} detections")

    for idx, detection in enumerate(detections):
        # Generate cutout and check if it's on the edge
        cutout = generate_cutout(frame.frame.data, detection, cutout_size, plot=False)
        if cutout.shape[0] != cutout.shape[1]:
            if config.detection.verbose:
                logger.warning(f"[{idx + 1}] [FILTERING] Detection is on edge of image")
            continue

        # Normalize cutout
        cutout = cutout - np.min(cutout)
        if np.sum(cutout) == 0:
            if config.detection.verbose:
                logger.warning(f"[{idx + 1}] [FILTERING] No signal in cutout")
            continue

        # Check for hot pixels (too much flux in a single pixel)
        hot_pixel_concentration = np.max(cutout) / np.sum(cutout)
        if hot_pixel_concentration > hot_pixel_threshold:
            if config.detection.verbose:
                logger.warning(
                    f"[{idx + 1}] [FILTERING] Brightest pixel contains {hot_pixel_concentration:.2f} of total flux"
                )
            continue

        # Check if flux is concentrated (not two separate bright spots)
        # For bright point sources, the two brightest pixels might be separated by up to ~2x seeing
        # This is more lenient than the strict seeing limit to avoid filtering out valid bright sources
        p1, p2 = find_two_brightest_points(cutout)
        dist = euclidean_distance(p1, p2)
        max_separation = pixel_seeing * 2.0  # Allow up to 2x seeing for bright sources
        if dist > max_separation:
            if config.detection.verbose:
                logger.warning(
                    f"[{idx + 1}] [FILTERING] Two brightest pixels separated by {dist:.1f} pixels "
                    f"(seeing is {pixel_seeing:.1f}, max allowed is {max_separation:.1f})"
                )
            continue

        # Check PSF shape with Gaussian fitting
        try:
            fx, fy, fcomb = cutout_gauss(cutout, pixel_seeing, plot=False)

            # Check if PSF is too narrow
            if fx < pixel_seeing / 2.5 or fy < pixel_seeing / 2.5:
                if config.detection.verbose:
                    logger.warning(
                        f"[{idx + 1}] [FILTERING] PSF too narrow (FWHM={fcomb:.2f}) compared to seeing={pixel_seeing:.2f}"
                    )
                continue

            # Check if PSF is too wide - use stricter threshold (1.5x) to catch streak detections
            # Also check average FWHM to catch cases where one dimension is OK but average is too wide
            max_fwhm = max(fx, fy)
            if max_fwhm > pixel_seeing * 1.5 or fcomb > pixel_seeing * 1.5:
                if config.detection.verbose:
                    logger.warning(
                        f"[{idx + 1}] [FILTERING] PSF too wide (FWHM_x={fx:.2f}, FWHM_y={fy:.2f}, "
                        f"avg={fcomb:.2f}) compared to seeing={pixel_seeing:.2f}"
                    )
                continue

            # Check if PSF is non-circular (could be a streak)
            # Use stricter threshold (40% instead of 55%) to catch more streak-like detections
            if percent_difference(fx, fy) > 40:
                if config.detection.verbose:
                    logger.warning(
                        f"[{idx + 1}] [FILTERING] PSF not round (difference between x and y FWHM={percent_difference(fx, fy):.2f}%)"
                    )
                continue

            # Additional check: if one dimension is much larger than the other, likely a streak
            # For point sources, both dimensions should be similar
            fwhm_ratio = max(fx, fy) / min(fx, fy) if min(fx, fy) > 0 else float("inf")
            if fwhm_ratio > 2.0:
                if config.detection.verbose:
                    logger.warning(
                        f"[{idx + 1}] [FILTERING] PSF aspect ratio too high (ratio={fwhm_ratio:.2f}, "
                        f"likely a streak)"
                    )
                continue

        except Exception as e:
            if config.detection.verbose:
                logger.warning(
                    f"[{idx + 1}] [FILTERING] Failed to fit Gaussian: {str(e)}"
                )
            continue

        if config.plotting.debug:
            fig, ax = plot_single_frame(cutout)
            plt.savefig(config.runtime.output_dir / f"detection_cutout_{idx}.png")
            plt.close("all")

        pixel_fwhm = (fx + fy) / 2
        filtered_detections.append([detection[0], detection[1], pixel_fwhm])
        """
        # Refine centroid position
        try:
            masked_frame = frame.frame.data.copy()
            cutmask = mask_tol(masked_frame, [detection[1], detection[0]], pixel_tol=cutout_size)
            masked_frame *= cutmask.astype(masked_frame.dtype)

            # Find the true maximum within the masked region
            y_cent, x_cent = np.unravel_index(np.argmax(masked_frame), masked_frame.shape)

            # Use quadratic centroiding for sub-pixel precision
            centroid_x, centroid_y = centroid_quadratic(masked_frame, x_cent, y_cent)

            if np.isnan(centroid_x) or np.isnan(centroid_y):
                # If quadratic centroiding fails, fall back to the maximum pixel
                logger.warning(f"[{idx + 1}] [CENTROID] Quadratic centroiding failed, using maximum pixel")
                centroid_x, centroid_y = x_cent, y_cent

            logger.info(
                f"[{idx + 1}] [ACCEPTING] Detection with FWHM={fcomb:.2f} and "
                + f"brightest pixel flux contribution={hot_pixel_concentration:.2f}"
            )

            pixel_fwhm = (fx + fy) / 2
            filtered_detections.append([centroid_x, centroid_y, pixel_fwhm])
            if config.plotting.debug:
                fig, ax = plot_single_frame(masked_frame,scale=False)
                ax.scatter(centroid_x, centroid_y, color="red", marker="o", facecolors='none')
                plt.savefig(config.runtime.output_dir / f"detection_centroid_{idx}.png")
                plt.close("all")
            breakpoint()

        except Exception as e:
            if config.detection.verbose:
                logger.warning(f"[{idx + 1}] [FILTERING] Failed to measure centroid: {str(e)}")
            continue
        """

    return filtered_detections


def extract_point_sources(frame: RateTrackFrame) -> SatelliteListImage:
    """
    Extract point sources from a rate track frame.

    This function identifies point sources in astronomical frames where stars may be streaked.
    It uses a combination of techniques to distinguish point sources from streaks, hot pixels, and noise.

    Args:
        frame: A RateTrackFrame object containing the image data and metadata

    Returns:
        A SatelliteListImage containing detected point sources
    """
    logger.info("Extracting point sources")

    config = get_config()
    # Get the image data
    image_data = frame.frame.data

    # Apply 3x3 median filter to remove hot pixels before detection
    # This helps eliminate single-pixel hot pixels that could be mistaken for point sources
    image_data = _median_filter_3x3(image_data)
    logger.debug("Applied 3x3 median filter to remove hot pixels")

    # Calculate background statistics using sigma clipping
    mean, median, std = robust_background_stats(image_data)

    # Subtract background
    image_data_sub = image_data - median

    # Use DAOStarFinder with adaptive threshold to get between 3-100 sources
    threshold_min = 3.0 * std  # Minimum threshold
    threshold_max = 100.0 * std  # Maximum threshold (adjust as needed)
    threshold = 10.0 * std  # Start with 5.0 * std
    fwhm = 3.0
    if frame.starfield and frame.starfield.detection_metadata.pixel_fwhm is not None:
        fwhm = frame.starfield.detection_metadata.pixel_fwhm

    max_attempts = 10
    attempts = 0
    sources = None
    min_sources = 50
    max_sources = 300  # Adjust this value as needed

    # One kernel convolution serves every threshold attempt below (see
    # _dao_sources_at_threshold). The FFT convolution is the same linear
    # operation photutils applies directly. Candidates are gathered lazily
    # at the lowest threshold visited so far: a local maximum above any
    # lower floor filtered to the attempt threshold is exactly the maxima
    # set at that threshold, and the binary search starts at 10 sigma and
    # rarely descends toward the 3 sigma floor — where the candidate set is
    # ~50x larger and its local-maxima pass costs seconds.
    kernel = _StarFinderKernel(float(fwhm), ratio=1.0, theta=0.0, sigma_radius=1.5)
    with fft_workers():
        convolved = fftconvolve(
            image_data_sub.astype(np.float32),
            kernel.data.astype(np.float32),
            mode="same",
        )

    gathered_floor = None
    cand_xy = cand_vals = None

    def ensure_candidates(min_threshold: float) -> None:
        nonlocal gathered_floor, cand_xy, cand_vals
        if gathered_floor is not None and gathered_floor <= min_threshold:
            return
        ys, xs, vals = _local_maxima_above(
            convolved, kernel.mask.astype(bool), min_threshold * kernel.relerr
        )
        cand_xy = np.column_stack((xs, ys))
        cand_vals = vals
        gathered_floor = min_threshold

    while attempts < max_attempts:
        ensure_candidates(threshold)
        sources = _dao_sources_at_threshold(
            image_data_sub,
            convolved,
            kernel,
            cand_xy,
            cand_vals,
            threshold,
            sharplo=0.1,
            sharphi=1.5,
            roundlo=-1.5,
            roundhi=1.5,
        )

        source_count = 0 if sources is None else len(sources)
        logger.info(
            f"Attempt {attempts + 1}: threshold={threshold:.2f}, found {source_count} sources"
        )

        # Binary search adjustment
        if sources is None or source_count < min_sources:
            # Too few sources, decrease threshold
            threshold_max = threshold
            threshold = (threshold_min + threshold) / 2
            logger.info(f"Too few sources, decreasing threshold to {threshold:.2f}")
        elif source_count > max_sources:
            # Too many sources, increase threshold
            threshold_min = threshold
            threshold = (threshold + threshold_max) / 2
            logger.info(
                f"Too many sources ({source_count}), increasing threshold to {threshold:.2f}"
            )
        else:
            # Good number of sources
            logger.info(f"Found {source_count} sources with threshold {threshold:.2f}")
            break

        # Check if we've converged (thresholds very close)
        if abs(threshold_max - threshold_min) < 0.1 * std:
            logger.info(f"Threshold search converged at {threshold:.2f}")
            break

        attempts += 1

    # If no sources found after all attempts, return empty list
    if sources is None:
        logger.info("No sources detected by DAOStarFinder")
        return SatelliteListImage(
            detections=[], image_metadata=frame.starfield.image_metadata
        )

    # Extract initial detections
    initial_detections = [
        (float(src["xcentroid"]), float(src["ycentroid"]))
        for src in sources
        if src["flux"] > 0
    ]
    logger.info(f"Initial detection found {len(initial_detections)} potential sources")

    """
    # Add additional detections from a simple peak finder (limited to 10 brightest)
    # to catch obvious bright sources that might have been missed by DAOStarFinder
    peak_threshold = 20.0 * std  # Higher threshold for obvious peaks
    data_smooth = image_data_sub.copy()

    # Apply a small Gaussian filter to reduce noise
    from scipy.ndimage import gaussian_filter, label

    data_smooth = gaussian_filter(data_smooth, sigma=1.0)

    # Find local maxima
    from scipy.ndimage import generate_binary_structure, maximum_filter

    s = generate_binary_structure(2, 2)
    filtered = maximum_filter(data_smooth, size=3)
    maxima = data_smooth == filtered

    # Filter out background noise
    maxima[data_smooth < peak_threshold] = 0

    # Get coordinates of maxima
    labeled, num_objects = label(maxima)
    xy = np.array(np.nonzero(maxima)).T

    # Get values at these coordinates
    if len(xy) > 0:
        peak_values = np.array([data_smooth[y, x] for y, x in xy])

        # Sort by brightness and take only the top 10
        if len(peak_values) > 30:
            brightest_indices = np.argsort(peak_values)[-10:]
            xy = xy[brightest_indices]

        # Convert to list of (x, y) tuples and add to initial detections
        additional_peaks = [(float(x), float(y)) for y, x in xy]
        logger.info(f"Found {len(additional_peaks)} additional bright peaks")
    else:
        additional_peaks = []
        logger.info("No additional bright peaks found")
    """
    additional_peaks = []
    # Combine all detections
    all_detections = initial_detections + additional_peaks

    pixel_seeing = fwhm

    logger.info(f"Estimated seeing: {pixel_seeing:.2f} pixels")

    # Plot initial detections before filtering for debugging
    if config.plotting.debug and config.runtime.output_dir:
        try:
            # Create temporary detections list for plotting
            temp_detections = []
            for x, y in all_detections:
                temp_det = SatelliteInImage(
                    x=float(x),
                    y=float(y),
                    ra=None,
                    dec=None,
                    snr=0.0,
                    pixel_fwhm=pixel_seeing,
                )
                temp_detections.append(temp_det)

            temp_detections_list = SatelliteListImage(
                detections=temp_detections,
                image_metadata=frame.starfield.image_metadata,
            )

            plot_single_frame(
                image_data_sub,
                detections=temp_detections_list,
                output_file=config.runtime.output_dir / "detections_initial.png",
                scale=True,
            )
            logger.info(
                f"Plotted {len(all_detections)} initial detections to detections_initial.png"
            )
        except Exception as e:
            logger.warning(f"Failed to plot initial detections: {e}")

    # Filter out non-point sources
    filtered_detections = filter_point_sources(
        frame=frame, detections=all_detections, pixel_seeing=pixel_seeing
    )

    logger.info(f"After filtering, {len(filtered_detections)} point sources remain")

    # Deduplicate detections
    deduplicated_detections = []
    for detection in filtered_detections:
        # Check if this detection is a duplicate (within 1 pixel of an existing detection)
        is_duplicate = False
        for existing in deduplicated_detections:
            distance = np.sqrt(
                (detection[0] - existing[0]) ** 2 + (detection[1] - existing[1]) ** 2
            )
            if distance < 1.0:  # 1 pixel threshold for considering as duplicate
                is_duplicate = True
                break

        if not is_duplicate:
            deduplicated_detections.append(detection)

    if len(deduplicated_detections) < len(filtered_detections):
        logger.info(
            f"After deduplication, {len(deduplicated_detections)} detections remain"
        )

    # Convert to StarInImage objects
    stars = []
    for x, y, pixel_fwhm in deduplicated_detections:
        # Calculate SNR
        cutout = generate_cutout(image_data_sub, (x, y), int(pixel_fwhm * 2))
        peak_value = np.max(cutout)
        snr = peak_value / std

        # Convert pixel coordinates to world coordinates if WCS is available
        ra, dec = None, None
        if frame.starfield.wcs is not None:
            ra, dec = frame.starfield.wcs.pix2world_0based(x, y)

        # Create StarInImage object
        star = SatelliteInImage(
            x=float(x),
            y=float(y),
            ra=ra,
            dec=dec,
            snr=float(snr),
            pixel_fwhm=float(pixel_fwhm),
        )

        if config.detection.snr_threshold and star.snr > config.detection.snr_threshold:
            stars.append(star)

    if config.detection.snr_threshold:
        logger.info(f"After SNR filtering, {len(stars)} detections remain")

    return SatelliteListImage(
        detections=stars, image_metadata=frame.starfield.image_metadata
    )
