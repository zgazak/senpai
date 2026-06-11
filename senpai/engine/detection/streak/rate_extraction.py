"""Rate-track streak extraction and source detection utilities."""

import logging

import numpy as np
from scipy.signal import convolve

from senpai.engine.detection.kernels import rectangle_pyramoid
from senpai.engine.detection.streak.extraction import (
    extract_streak_dims_robust,
    prepare_rate_frame,
    refine_robust_streak,
)
from senpai.engine.models.metadata import StreakMetadata
from senpai.engine.models.starfield import StarInImage
from senpai.engine.models.streak_measurement import StreakMeasurement
from senpai.engine.utils.stats import fft_workers, robust_background_stats

logger = logging.getLogger(__name__)


def extract_rate_streak_measurement(
    rate_frame,
    *,
    n_streaks: int = 10,
    initial_fwhm: float | None = None,
) -> tuple[StreakMeasurement | None, np.ndarray | None, float | None]:
    """
    Measure the characteristic star-streak in a single rate-track frame.

    Returns:
        (measurement, psf, measured_fwhm)
    """
    # Work on a padded/cropped version to avoid edge artifacts.
    rate_data = prepare_rate_frame(rate_frame)

    measurement, psf, measured_fwhm = extract_streak_dims_robust(
        rate_data,
        n_streaks=n_streaks,
        rotation=None,
        length=None,
        fwhm=initial_fwhm,
    )
    if measurement is None:
        return None, None, measured_fwhm

    # Refine using the two-stage robust refiner when we have a PSF cutout.
    if psf is not None:
        measurement, measured_fwhm = refine_robust_streak(
            psf, measurement, frame_index=rate_frame.index
        )

    return measurement, psf, measured_fwhm


def build_streak_metadata(measurement: StreakMeasurement) -> StreakMetadata:
    theta = float(measurement.rotation)
    return StreakMetadata(
        pixel_length=float(measurement.length),
        sine_angle=float(np.sin(np.deg2rad(theta))),
        cosine_angle=float(np.cos(np.deg2rad(theta))),
        fwhm=float(measurement.fwhm if measurement.fwhm is not None else 0.0),
    )


def extract_streak_centers_as_sources(
    image: np.ndarray,
    *,
    streak: StreakMetadata | None = None,
    max_sources: int = 200,
    threshold_sigma: float = 3.0,
) -> list[StarInImage]:
    """
    Extract streak centroids using matched filtering with a rectangular kernel.

    Uses the measured streak parameters (length, angle, FWHM) to create a matched
    filter kernel, then finds local maxima in the convolved image.

    Args:
        image: Raw image data
        streak: Measured streak metadata (if None, uses conservative defaults)
        max_sources: Maximum number of detections to return
        threshold_sigma: Detection threshold in units of background RMS

    Returns:
        List of StarInImage objects representing streak centroids
    """
    # Use measured streak parameters if available, otherwise conservative defaults
    if streak is not None:
        streak_length = streak.pixel_length
        streak_angle_deg = streak.degree_angle()
        streak_fwhm = streak.fwhm
    else:
        # Fallback: estimate from image size
        streak_length = min(image.shape) * 0.05
        streak_angle_deg = 0.0
        streak_fwhm = 4.0
        logger.warning(
            "No streak metadata available, using defaults: "
            f"length={streak_length:.1f}px, angle={streak_angle_deg:.1f}\u00b0, fwhm={streak_fwhm:.1f}px"
        )

    # Create matched filter kernel matching the measured streak
    if streak is not None:
        sine_angle = streak.sine_angle
        cosine_angle = streak.cosine_angle
    else:
        sine_angle = np.sin(np.deg2rad(streak_angle_deg))
        cosine_angle = np.cos(np.deg2rad(streak_angle_deg))

    kernel = rectangle_pyramoid(
        streak_length,
        sine_angle,
        cosine_angle,
        int(streak_fwhm * 2),
        upsample=100,
        halo_fwhm=4,
        halo_level=0,
    )

    # Normalize kernel to have unit sum (for proper SNR calculation); keep
    # everything float32 \u2014 this stage picks peaks, it doesn't do photometry.
    kernel = (kernel / np.sum(kernel)).astype(np.float32)

    # Convolve image with matched filter
    logger.info(
        f"Convolving image with streak-matched kernel "
        f"(L={streak_length:.1f}px, \u03b8={streak_angle_deg:.1f}\u00b0, W={streak_fwhm:.1f}px)"
    )
    with fft_workers():
        convolved = convolve(image.astype(np.float32), kernel, mode="same")

    # Estimate background statistics from the convolved image: sigma-clipped
    # for outlier robustness, on a strided subsample (threshold shift vs the
    # full-frame stats is <0.01 sigma on real 8k frames, at ~70x less cost).
    _, median, std = robust_background_stats(convolved)

    # Detection threshold
    threshold = median + threshold_sigma * std
    logger.info(
        f"Detection threshold: {threshold:.1f} (median={median:.1f}, std={std:.1f}, "
        f"sigma={threshold_sigma:.1f})"
    )

    # Find local maxima above threshold. Same predicate as a full-frame
    # maximum_filter comparison (a pixel >= every neighbor in its window),
    # but evaluated only at above-threshold pixels — a tiny fraction of the
    # frame — instead of filtering all 66 Mpix. The neighborhood is based on
    # streak FWHM to avoid multiple detections per streak.
    neighborhood_size = max(3, int(streak_fwhm))
    half = neighborhood_size // 2

    above = convolved >= threshold
    # Border pixels can't be tested against a full window (and sit in the
    # convolution's edge-artifact zone anyway).
    above[:half, :] = False
    above[-half:, :] = False
    above[:, :half] = False
    above[:, -half:] = False
    y_coords, x_coords = np.nonzero(above)
    response_values = convolved[y_coords, x_coords]

    for dy in range(-half, half + 1):
        is_max = np.ones(y_coords.size, dtype=bool)
        for dx in range(-half, half + 1):
            if dy == 0 and dx == 0:
                continue
            is_max &= response_values >= convolved[y_coords + dy, x_coords + dx]
        # Compact survivors after each row of offsets: most candidates are
        # eliminated by their immediate neighbors, so later rows test a
        # shrinking set.
        if not is_max.all():
            y_coords = y_coords[is_max]
            x_coords = x_coords[is_max]
            response_values = response_values[is_max]

    if len(y_coords) == 0:
        logger.warning("No detections found above threshold")
        return []

    # Sort by response value (brightest first)
    sort_indices = np.argsort(response_values)[::-1]
    y_coords = y_coords[sort_indices]
    x_coords = x_coords[sort_indices]
    response_values = response_values[sort_indices]

    # Apply minimum separation to avoid duplicate detections
    # Minimum separation should be at least the streak length
    min_separation = max(streak_length * 0.5, streak_fwhm * 2)
    detections: list[StarInImage] = []
    seen_positions: list[tuple[float, float]] = []

    for i in range(
        min(len(y_coords), max_sources * 3)
    ):  # Check more candidates than needed
        if len(detections) >= max_sources:
            break

        x, y = float(x_coords[i]), float(y_coords[i])
        response = float(response_values[i])

        # Check minimum separation from existing detections
        too_close = False
        for existing_x, existing_y in seen_positions:
            dist = np.sqrt((x - existing_x) ** 2 + (y - existing_y) ** 2)
            if dist < min_separation:
                too_close = True
                break

        if too_close:
            continue

        # Use response value as a proxy for "counts" (SNR-weighted flux)
        detections.append(StarInImage(x=x, y=y, counts=response))
        seen_positions.append((x, y))

    logger.info(
        f"Extracted {len(detections)} streak centroids from {len(y_coords)} candidates "
        f"(threshold={threshold:.1f}, min_sep={min_separation:.1f}px)"
    )

    return detections
