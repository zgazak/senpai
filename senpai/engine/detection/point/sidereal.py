import logging

import numpy as np
from astropy.table import Table
from photutils.detection import DAOStarFinder
from scipy.optimize import curve_fit

from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.models.starfield import StarInImage, StarListImage
from senpai.engine.utils.stats import robust_background_stats

logger = logging.getLogger(__name__)


def gaussian_2d(
    data: tuple[np.ndarray, np.ndarray],
    amp: float,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    offset: float,
) -> np.ndarray:
    """
    2D Gaussian function for curve fitting.
    """
    x, y = data
    x0 = float(x0)
    y0 = float(y0)
    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (2 * sigma_y**2)

    # Calculate the 2D Gaussian
    gaussian = offset + amp * np.exp(-(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2)))

    # Return the flattened Gaussian to match the flattened cutout data
    return gaussian.ravel()


def estimate_fwhm(
    image: np.ndarray,
    x_centroid: float,
    y_centroid: float,
    box_size: int = 20,
    fwhm_x_guess: float = 2.0,
    fwhm_y_guess: float = 2.0,
) -> float | None:
    """
    Estimate the FWHM of a bright star by fitting a 2D Gaussian to the source.

    Parameters:
    - image (numpy.ndarray): 2D array representing the image.
    - x_centroid, y_centroid (float): Coordinates of the star's centroid.
    - box_size (int): Size of the box to extract around the centroid for fitting.

    Returns:
    - float: Estimated FWHM.
    """
    # Extract a small box around the star
    x0, y0 = int(x_centroid), int(y_centroid)

    # Calculate box boundaries with boundary checks
    y_min = max(0, y0 - box_size // 2)
    y_max = min(image.shape[0], y0 + box_size // 2)
    x_min = max(0, x0 - box_size // 2)
    x_max = min(image.shape[1], x0 + box_size // 2)

    # Check if the box is too small for a meaningful fit
    if (y_max - y_min) < 5 or (x_max - x_min) < 5:
        logger.warning(f"Box too small for star at ({x0}, {y0})")
        return None

    cutout = image[y_min:y_max, x_min:x_max]

    # Reject saturated / near-saturated stars: a clipped, flat-topped core forces
    # the Gaussian fit to a wide sigma (the flat top can't be matched by a narrow
    # peak), which poisons the median FWHM and balloons the downstream detection
    # kernel cost (cost ~ FWHM²). A real PSF — even in poor seeing — has a single
    # smooth peak (~1 pixel at the max); a (near-)saturated core is a plateau of
    # many pixels pinned at the same clipped ceiling. This signature is
    # threshold-free (no absolute saturation level needed) and scales correctly
    # with seeing.
    if cutout.size:
        peak = float(cutout.max())
        # Tolerance scales with peak: after row/col-median subtraction a
        # clipped plateau is no longer flat-valued (pixels differ by the
        # subtracted medians, tens-to-hundreds of ADU), so a ±1 ADU test
        # never fires and bloomed cores reach the Gaussian fit. 0.5% of peak
        # catches those plateaus while a real PSF keeps <2 px that close to
        # its peak (the top 0.5% of a Gaussian is r < 0.1 sigma).
        flat_tol = max(2.0, 0.005 * abs(peak))
        if peak > 0 and int(np.count_nonzero(cutout >= peak - flat_tol)) >= 4:
            logger.debug(
                "Skipping FWHM at (%d, %d): saturated/flat-topped core", x0, y0
            )
            return None

    # Create x and y coordinates for fitting
    y, x = np.mgrid[: cutout.shape[0], : cutout.shape[1]]

    # Initial guess for fitting parameters
    try:
        # Adjust initial guess to account for potentially asymmetric box
        x_center = (x_max - x_min) // 2
        y_center = (y_max - y_min) // 2
        initial_guess = (
            cutout.max(),
            x_center,
            y_center,
            fwhm_x_guess,
            fwhm_y_guess,
            0,
            0,
        )
    except Exception as e:
        logger.error(f"Error estimating FWHM: {e}")
        return None

    try:
        # Constrain the fit to avoid pathological sigmas that lead to
        # unphysically large or tiny FWHM values.
        # Parameter order: amp, x0, y0, sigma_x, sigma_y, theta, offset
        lower_bounds = (
            0.0,  # amp >= 0
            0.0,
            0.0,
            0.5,  # sigma_x in [0.5, 20] px  -> FWHM in ~[1.2, 47] px
            0.5,  # sigma_y
            -np.pi,
            -np.inf,
        )
        upper_bounds = (
            np.inf,
            cutout.shape[1],
            cutout.shape[0],
            20.0,
            20.0,
            np.pi,
            np.inf,
        )

        popt, _ = curve_fit(
            gaussian_2d,
            (x, y),
            cutout.ravel(),
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=2000,
        )

        # Extract fitted parameters
        amp, x0_fit, y0_fit, sigma_x, sigma_y, theta, offset = popt

        # FWHM is approximately 2.355 * sigma for a Gaussian
        fwhm_x = 2.355 * sigma_x
        fwhm_y = 2.355 * sigma_y
        fwhm = float((fwhm_x + fwhm_y) / 2.0)

        # Sanity-check FWHM; reject clearly pathological results.
        if not np.isfinite(fwhm) or fwhm <= 0.3 or fwhm > 50.0:
            logger.debug(f"Discarding pathological FWHM estimate: {fwhm:.3f} pixels")
            return None

        return fwhm
    except RuntimeError:
        return None


def detect_sources_classic(
    image: np.ndarray,
    max_sources: int = 10,
    fwhm: float = 5.0,
    threshold_sigma: float = 5.0,
    sharplo: float = 0.2,
    sharphi: float = 2.0,
    bg_stats: tuple[float, float, float] | None = None,
) -> Table:
    """
    Detect point sources in a 2D image.

    Parameters:
    - image (numpy.ndarray): 2D array representing the image.
    - max_sources (int): Maximum number of sources to detect. Defaults to 10.
    - fwhm (float): Full-width half-maximum of the point sources. Defaults to 5.0.
    - threshold_sigma (float): Detection threshold in units of background RMS noise. Defaults to 5.0.
    - sharplo (float): Lower bound on sharpness for source detection.
    - sharphi (float): Upper bound on sharpness for source detection.
    - bg_stats: Precomputed (mean, median, std) background statistics; pass
      this when calling repeatedly on the same image so the sigma-clipped
      stats are computed once.

    Returns:
    - astropy.table.Table: A table containing the detected sources, sorted by brightness.
    """
    # Estimate background statistics with more robust parameters
    if bg_stats is None:
        bg_stats = robust_background_stats(image)
    mean, median, std = bg_stats

    # Define the DAOStarFinder object with more permissive criteria
    daofind = DAOStarFinder(
        fwhm=fwhm,
        threshold=threshold_sigma * std,
        sharplo=sharplo,
        sharphi=sharphi,
        roundlo=-1.0,  # More permissive roundness criteria
        roundhi=1.0,
        peakmax=None,  # Don't limit peak values
    )

    # Find stars in background-subtracted image
    sources = daofind(image - median)

    # Sort sources by peak brightness (flux) and limit to max_sources
    if sources is not None:
        sources.sort("flux", reverse=True)
        max_sources = min(max_sources, len(sources))
        return sources[:max_sources]

    return Table()


def _estimate_saturation_level(image: np.ndarray, sources) -> float:
    """Robust per-frame saturation level from detected source PEAKS.

    Raw uint16 frames clip at 65535; after the row/col-median background
    subtraction the clipped pixels land a few thousand ADU below that and are no
    longer flat-valued (so a flat-top test fails). Stars are a tiny fraction of
    the mostly-sky frame, so a whole-frame percentile lands in the noise — but the
    detected sources' core peaks are bimodal: saturated stars (and a bright star's
    bloom duplicates) pile tightly just under the clipped ceiling, well separated
    from unsaturated stars. The 90th percentile of source peaks sits in that pile;
    back off 10% for a conservative reject. Falls back to 0.8*max when too few
    sources for a stable percentile.
    """
    peaks = []
    for s in sources:
        x0, y0 = int(round(s["xcentroid"])), int(round(s["ycentroid"]))
        core = image[max(0, y0 - 2): y0 + 3, max(0, x0 - 2): x0 + 3]
        if core.size:
            peaks.append(float(core.max()))
    return sat_level_from_peaks(peaks)


def sat_level_from_peaks(peaks) -> float:
    """Saturation level from a sample of source core peaks (see
    _estimate_saturation_level for the rationale). Shared with the
    catalog-star FWHM path, whose star sample suffers the same
    saturated-pile-at-the-bright-end structure."""
    if len(peaks) < 10:
        return float("inf")  # too few sources to identify a saturated population
    peaks = np.asarray(peaks)
    level = 0.9 * float(np.percentile(peaks, 90))
    # Only apply the cut if it genuinely separates a saturated cluster from a
    # fainter population. If nearly all sources sit above it (a uniformly bright
    # field — or a synthetic test with identical-amplitude stars), there's no
    # saturation split to act on, so don't reject anything by level.
    n_below = int(np.count_nonzero(peaks < level))
    if n_below < max(3, int(0.15 * len(peaks))):
        return float("inf")
    return level


def _robust_source_fwhm(
    image: np.ndarray, x: float, y: float, sat_level: float, box_size: int = 21
) -> float | None:
    """Fit-free, saturation-aware FWHM: diameter of the connected above-half-max
    region around the centroid.

    No Gaussian assumption — the real PSFs here are messy and a curve_fit returns
    garbage (37/42/1.2 px) on them. Rejects saturated cores (peak >= sat_level)
    and isolates the source's own component via connected-component labelling so a
    neighbour blend can't inflate the width.
    """
    from scipy.ndimage import label

    x0, y0 = int(round(x)), int(round(y))
    h = box_size // 2
    y_min, y_max = max(0, y0 - h), min(image.shape[0], y0 + h + 1)
    x_min, x_max = max(0, x0 - h), min(image.shape[1], x0 + h + 1)
    cut = image[y_min:y_max, x_min:x_max].astype(float)
    if cut.size < 25:
        return None
    peak = float(cut.max())
    if peak >= sat_level:
        return None  # saturated — don't measure
    bg = float(np.median(cut))
    amp = peak - bg
    if amp <= 0:
        return None
    half = bg + 0.5 * amp
    lbl, _ = label(cut >= half)
    cy = min(cut.shape[0] - 1, max(0, y0 - y_min))
    cx = min(cut.shape[1] - 1, max(0, x0 - x_min))
    cid = lbl[cy, cx]
    if cid == 0:
        return None
    area = int(np.count_nonzero(lbl == cid))
    if area < 2:
        return None
    return 2.0 * float(np.sqrt(area / np.pi))


def _measure_fwhm_sample(
    data: np.ndarray,
    bg_stats: tuple[float, float, float],
    initial_fwhm: float,
) -> tuple[list[float], float, int]:
    """First-pass FWHM sample: detect bright sources and measure fit-free FWHMs.

    Sample deep (300): the brightest sources are almost all SATURATED (and a
    bright saturated star spawns many adjacent detections along its bloom), so
    we must reach past them to the fainter unsaturated stars. FWHM is measured
    only from unsaturated, de-blended stars via a fit-free half-max-area
    measure (the Gaussian fit returns garbage on these messy / saturated
    PSFs), deduped so a saturated star's bloom-duplicates don't crowd out the
    unsaturated sample.

    Returns (fwhms, sat_level, n_detected).
    """
    detected_sources = detect_sources_classic(
        data,
        max_sources=300,
        fwhm=initial_fwhm,
        threshold_sigma=5.0,
        sharplo=0.1,
        sharphi=2.0,
        bg_stats=bg_stats,
    )

    if detected_sources is None or len(detected_sources) == 0:
        return [], float("inf"), 0

    sat_level = _estimate_saturation_level(data, detected_sources)
    fwhms: list[float] = []
    measured_pos: list[tuple[float, float]] = []
    DEDUP_R2 = 12.0 ** 2
    for source in detected_sources:
        xc = float(source["xcentroid"])
        yc = float(source["ycentroid"])
        if any((xc - mx) ** 2 + (yc - my) ** 2 < DEDUP_R2 for mx, my in measured_pos):
            continue
        try:
            fwhm = _robust_source_fwhm(data, xc, yc, sat_level)
        except Exception as e:
            logger.debug(f"FWHM estimation failed: {str(e)}")
            continue
        if fwhm is not None and fwhm > 0:
            fwhms.append(fwhm)
            measured_pos.append((xc, yc))
            if len(fwhms) >= 40:
                break
    return fwhms, sat_level, len(detected_sources)


def _refine_centroid_full_res(
    data: np.ndarray, x: float, y: float, fwhm: float, bg_median: float
) -> tuple[float, float]:
    """Re-measure a binned-detection centroid on the full-resolution frame.

    Iterative Gaussian-windowed centroid (SExtractor's XWIN/YWIN scheme): a
    center of mass weighted by a PSF-sized Gaussian recentred on each
    iterate. The window suppresses wings, noise, and neighbours, and the
    fixed point of the iteration is the unweighted source center, so it
    converges to near-PSF-fit accuracy — a plain clipped center of mass
    plateaus an order of magnitude worse. Recovers the sub-pixel accuracy
    that 2x2-binned detection gives up (binned centroids carry up to
    ~0.5 px of full-res quantization). Returns the input position when the
    window holds no positive signal.
    """
    sigma_w = max(1.5, fwhm / 2.355)
    r = max(4, int(round(fwhm)))
    for _ in range(5):
        x0, y0 = int(round(x)), int(round(y))
        ylo, yhi = max(0, y0 - r), min(data.shape[0], y0 + r + 1)
        xlo, xhi = max(0, x0 - r), min(data.shape[1], x0 + r + 1)
        cut = data[ylo:yhi, xlo:xhi] - bg_median
        if cut.size == 0:
            return x, y
        yy, xx = np.mgrid[ylo:yhi, xlo:xhi]
        window = np.exp(-(((xx - x) ** 2) + ((yy - y) ** 2)) / (2.0 * sigma_w**2))
        weights = np.clip(cut, 0, None) * window
        total = float(weights.sum())
        if total <= 0:
            return x, y
        x_new = float((weights * xx).sum() / total)
        y_new = float((weights * yy).sum() / total)
        converged = abs(x_new - x) < 0.005 and abs(y_new - y) < 0.005
        x, y = x_new, y_new
        if converged:
            break
    return x, y


def extract_point_sources(
    image: ProcessedFitsImage, max_detections: int = 100, min_separation: float = None
) -> tuple[StarListImage, float]:
    logger.info("Extracting point sources from image %s", image.metadata.image_id)

    # Default FWHM values - more permissive for poor seeing conditions
    DEFAULT_FWHM = 4.0
    MIN_VALID_FWHM = 1.5
    # Pass 2 (detection) runs on a 2x2-binned frame when the PSF is fat
    # enough that binning keeps it well-sampled (binned FWHM >= 3 px);
    # accepted centroids are then re-measured at full resolution.
    BIN_MIN_FWHM = 6.0
    BIN_MIN_DIM = 2048

    data = image.data
    h, w = data.shape

    # Background statistics: computed once on a strided subsample and shared
    # by both detection passes (this was previously recomputed full-frame
    # inside each pass and dominated the total detection cost).
    bg_stats = robust_background_stats(data)

    # Pass 1 deliberately scans the FULL frame even though it only needs ~40
    # FWHM stars: the saturation level is a percentile of the detected
    # sources' peaks, and a central-crop population was observed to land the
    # percentile far below the true clip level (28.7k vs 42.3k on a real
    # calsat field), rejecting the bright unsaturated stars and biasing the
    # median FWHM to the noise-truncated faint end (3.1 px vs a true ~9 px).
    fwhms, sat_level, n_detected = _measure_fwhm_sample(data, bg_stats, DEFAULT_FWHM)

    if n_detected == 0:
        logger.warning("No bright sources detected in first pass")
        return StarListImage(detections=[], image_metadata=image.metadata), DEFAULT_FWHM

    # Use median FWHM if we have measurements, otherwise use default
    if len(fwhms) >= 2:
        fwhm_pixel = float(np.median(fwhms))
        logger.info(
            "Using median FWHM %.1f px from %d unsaturated stars (sat_level=%.0f)",
            fwhm_pixel, len(fwhms), sat_level,
        )
    else:
        fwhm_pixel = DEFAULT_FWHM
        logger.info(f"Using default FWHM ({DEFAULT_FWHM} pixels) due to insufficient valid measurements")

    # Only enforce minimum FWHM, be more permissive with maximum
    fwhm_pixel = max(MIN_VALID_FWHM, fwhm_pixel)

    # Set minimum separation based on FWHM if not provided
    if min_separation is None:
        if fwhm_pixel > 10:
            min_separation = max(1.0 * fwhm_pixel, 5.0)
        else:
            min_separation = max(1.5 * fwhm_pixel, 5.0)

    logger.info(f"Using FWHM: {fwhm_pixel:.1f} pixels, minimum separation: {min_separation:.1f} pixels")

    # Second pass: detect all sources with adapted FWHM — skip per-source
    # FWHM fitting.  DAOStarFinder's sharpness filter already rejects
    # non-stellar sources (cosmics, hot pixels, streak fragments).
    # The expensive per-source curve_fit was the dominant cost (~0.05s each
    # x 500 sources = 25s).  Using the median FWHM from the first pass
    # and trusting DAOStarFinder's built-in quality metrics is sufficient
    # for astrometry (which only needs positions, not per-star FWHM).
    threshold_sigma = 2.0 if fwhm_pixel > 10 else 3.0

    # With a fat PSF the DAO kernel convolution + peak search scale with
    # FWHM^2 over 66 Mpix; a 2x2 mean-bin quarters the pixels and halves the
    # kernel while the binned PSF (>= 3 px) stays well-sampled, so the
    # matched-filter depth and DAO's sharpness/roundness filters are
    # preserved. Accepted centroids are re-measured at full resolution in
    # the accept loop below.
    use_binned = fwhm_pixel >= BIN_MIN_FWHM and min(h, w) >= BIN_MIN_DIM
    if use_binned:
        h2, w2 = (h // 2) * 2, (w // 2) * 2
        det_data = data[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3))
        det_fwhm = fwhm_pixel / 2.0
        det_stats = robust_background_stats(det_data)
        logger.info(
            "Detecting on 2x2-binned frame (FWHM %.1f -> %.1f px)",
            fwhm_pixel,
            det_fwhm,
        )
    else:
        det_data = data
        det_fwhm = fwhm_pixel
        det_stats = bg_stats

    sources = detect_sources_classic(
        det_data,
        max_sources=max_detections * 3,
        fwhm=det_fwhm,
        threshold_sigma=threshold_sigma,
        sharplo=0.1,
        sharphi=2.0,
        bg_stats=det_stats,
    )

    stars = []
    star_positions = []

    def is_too_close_optimized(new_x: float, new_y: float, positions: list, min_sep: float) -> bool:
        """Optimized check if a new source is too close to any existing source."""
        min_sep_sq = min_sep * min_sep
        for x, y in positions:
            dist_sq = (new_x - x) ** 2 + (new_y - y) ** 2
            if dist_sq < min_sep_sq:
                return True
        return False

    sources_processed = 0
    sources_skipped_proximity = 0

    if sources is None:
        logger.warning("No sources detected in second pass")
        return StarListImage(detections=[], image_metadata=image.metadata), fwhm_pixel

    # Sort sources by flux to process brightest first
    sources.sort("flux", reverse=True)

    for source in sources:
        if len(stars) >= max_detections:
            break

        sources_processed += 1
        x_centroid = source["xcentroid"]
        y_centroid = source["ycentroid"]

        # Skip if NaN coordinates
        if np.isnan(x_centroid) or np.isnan(y_centroid):
            continue

        if use_binned:
            # Binned pixel (i, j) covers full-res pixels (2i, 2i+1) x
            # (2j, 2j+1), so a binned pixel-center coordinate maps to
            # 2*c + 0.5 at full resolution; then recover sub-pixel accuracy
            # on the unbinned frame.
            x_centroid = 2.0 * x_centroid + 0.5
            y_centroid = 2.0 * y_centroid + 0.5
            x_centroid, y_centroid = _refine_centroid_full_res(
                data, x_centroid, y_centroid, fwhm_pixel, float(bg_stats[1])
            )

        # Check minimum separation from existing sources
        if is_too_close_optimized(x_centroid, y_centroid, star_positions, min_separation):
            sources_skipped_proximity += 1
            continue

        # Accept all sources that passed DAOStarFinder's quality filters
        # (sharpness, roundness).  No per-source FWHM fitting needed.
        # (When binned, "flux" is in binned-pixel units — ~1/4 of the
        # full-res value — which preserves the brightness ranking that
        # astrometry consumes.)
        stars.append(StarInImage(x=x_centroid, y=y_centroid, counts=source["flux"]))
        star_positions.append((x_centroid, y_centroid))

    logger.info(f"Extracted {len(stars)} point sources from {sources_processed} candidates")
    logger.info(f"Skipped {sources_skipped_proximity} sources due to proximity")
    logger.info(f"Used minimum separation: {min_separation:.1f} pixels")

    starlist = StarListImage(
        detections=stars,
        image_metadata=image.metadata,
        sat_level=(float(sat_level) if np.isfinite(sat_level) else None),
    )
    return starlist, fwhm_pixel
