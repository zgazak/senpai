import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve


from scipy.ndimage import gaussian_filter

# from senpai.engine.utils.preprocessing import background_subtract
from senpai.core.config import get_config
from senpai.engine.detection.kernels import rectangle_pyramoid
from senpai.engine.detection.streak.extraction import (
    cross_corr,
    extract_streak_dims_mapping,
    extract_streak_dims_robust,
    prepare_rate_frame,
    refine_robust_streak,
)
from senpai.engine.detection.streak.masking import (
    percent_difference,
    remove_border_crossing_streaks,
    remove_brightest_streak,
    remove_near_saturation_streaks,
    remove_streak_at_point_robust,
)
from senpai.engine.detection.streak.validation import validate_proposed_shift
from senpai.engine.models.metadata import StreakMetadata
from senpai.engine.models.senpai import FrameShift, RateTrackFrame
from senpai.engine.models.streak_measurement import (
    StreakMeasurement,
    StreakMeasurements,
    angular_difference,
)
from senpai.engine.utils.propagate_wcs import get_global_shift_from_astrometric_stars

logger = logging.getLogger(__name__)


def strip_unbalanced_streaks(rate1_img: np.ndarray, rate2_img: np.ndarray) -> None:
    max_r1 = np.percentile(rate1_img, 99.99)
    max_r2 = np.percentile(rate2_img, 99.99)
    pdthresh = 20.0
    pd = percent_difference(max_r1, max_r2)

    max_attempts = 5
    attempts = 0
    while pd > pdthresh and attempts < max_attempts:
        attempts += 1
        logger.info(
            f"percent difference {pd:.1f}% is greater than threshold of {pdthresh:.1f}%"
        )
        print(max_r1, max_r2)

        if max_r1 > max_r2:
            logger.info("removing near-saturation streak in rate1")
            fill_min = np.median(rate1_img) + 0.5 * np.std(rate1_img)
            rate1_img = remove_brightest_streak(rate1_img, fill_min)

        else:
            logger.info("removing near-saturation streak in rate2")
            fill_min = np.median(rate2_img) + 0.5 * np.std(rate2_img)
            rate2_img = remove_brightest_streak(rate2_img, fill_min)

        max_r1 = np.percentile(rate1_img, 99.99)
        max_r2 = np.percentile(rate2_img, 99.99)
        pd = percent_difference(max_r1, max_r2)

    if pd <= pdthresh:
        logger.info(
            f"percent difference {pd:.1f}% is below threshold of {pdthresh:.1f}%"
        )
    else:
        logger.warning(
            f"percent difference {pd:.1f}% is above threshold of {pdthresh:.1f}%"
        )


def refine_correlation_shift_by_global_shift(
    rate_frame_source: RateTrackFrame,
    rate_frame_target: RateTrackFrame,
    streak: StreakMetadata,
    shift: np.ndarray,
) -> np.ndarray:
    src_streak = rate_frame_source.streak
    if (src_streak is None or src_streak.pixel_length is None
            or src_streak.fwhm is None):
        # No streak model on the source frame (failed extraction) — the
        # refinement kernel can't be built; keep the unrefined CC shift.
        logger.warning(
            "Skipping correlation-shift refinement: source frame %s has no "
            "streak model", rate_frame_source.index,
        )
        return shift
    starfield = rate_frame_source.starfield
    if (starfield is None or not starfield.catalog_stars
            or not starfield.astrometric_fit_stars):
        # No catalog/astrometric stars on the source frame (failed catalog
        # query or WCS) — the global-shift refinement has nothing to anchor
        # to; keep the unrefined CC shift.
        logger.warning(
            "Skipping correlation-shift refinement: source frame %s has no "
            "catalog/astrometric stars", rate_frame_source.index,
        )
        return shift
    kernel = rectangle_pyramoid(
        rate_frame_source.streak.pixel_length,
        rate_frame_source.streak.sine_angle,
        rate_frame_source.streak.cosine_angle,
        int(rate_frame_source.streak.fwhm * 2),
        upsample=100,
        halo_fwhm=4,
        halo_level=0,
    )

    # Use mode='same' to ensure output has same size as input
    convolved_frame = convolve(rate_frame_target.frame.data, kernel, mode="same")

    shifted_catalog = [
        star.model_copy(update={"x": star.x - shift[0], "y": star.y - shift[1]})
        for star in rate_frame_source.starfield.catalog_stars
    ]
    shifted_astro = [
        star.model_copy(update={"x": star.x - shift[0], "y": star.y - shift[1]})
        for star in rate_frame_source.starfield.astrometric_fit_stars
    ]
    copied_rate_frame = rate_frame_source.model_copy(deep=False)
    copied_rate_frame.starfield = rate_frame_source.starfield.model_copy(
        update={"catalog_stars": shifted_catalog, "astrometric_fit_stars": shifted_astro}
    )

    x_shift, y_shift = get_global_shift_from_astrometric_stars(
        copied_rate_frame, convolved_frame
    )
    logger.info(
        f"refined correlation shift {shift[0]:.1f}, {shift[1]:.1f} -> {shift[0] - x_shift:.1f}, {shift[1] - y_shift:.1f}"
    )
    return np.array([shift[0] - x_shift, shift[1] - y_shift])


def whiten_image(im, sigma=3, eps=1e-6):
    # scipy.fft is the same pocketfft as numpy's but multithreaded with
    # workers — identical values, several times faster on full frames.
    from scipy import fft as sfft

    # subtract mean (optional but recommended)
    im0 = im - np.mean(im)

    # FFT
    F = sfft.fft2(im0, workers=-1)

    # Power spectrum
    P = np.abs(F) ** 2
    P_smooth = gaussian_filter(P, sigma=sigma)

    # Whiten
    F_white = F / np.sqrt(P_smooth + eps)

    # Back to image space
    return sfft.ifft2(F_white, workers=-1).real


def _block_median_downsample(img: np.ndarray, factor: int) -> np.ndarray:
    """Downsample by f×f block MEDIAN (not mean) — robust to hot pixels / cosmics
    that would otherwise spike the whitened cross-correlation. Trailing
    rows/cols that don't fill a block are dropped."""
    if factor <= 1:
        return img
    h, w = img.shape
    hc, wc = (h // factor) * factor, (w // factor) * factor
    blocks = img[:hc, :wc].reshape(hc // factor, factor, wc // factor, factor)
    return np.median(blocks, axis=(1, 3))


def cc_downsample_factor(
    fwhm: float | None, shape: tuple[int, int], target_fwhm: float = 3.0,
    min_side: int = 512,
) -> int:
    """Downsample factor for the coarse cross-correlation, chosen by information
    content: shed PSF oversampling down to ~target_fwhm px (FWHM 12 → factor 4),
    with a size floor so small frames (e.g. 512²) are left at full resolution. The
    shift only needs to be coarse — the catalog-star match refines to sub-pixel."""
    if not fwhm or fwhm <= target_fwhm:
        return 1
    f = max(1, int(round(fwhm / target_fwhm)))
    while f > 1 and min(shape[0] // f, shape[1] // f) < min_side:
        f -= 1
    return f


def solve_rate_from_rate(
    rate_frame_a: RateTrackFrame, rate_frame_b: RateTrackFrame, frame_shift: FrameShift
) -> None:
    # Return the modified object

    # Only the SOURCE frame needs a starfield: this solver reads
    # rate_frame_a.starfield.catalog_stars for cross-correlation masking and shift
    # validation (and would AttributeError on None), then the caller propagates
    # that WCS to the target. The TARGET's starfield is the *output* of this shift
    # — it's built downstream in collect.py ("Shifting WCS by pixel shift"), so a
    # target with starfield=None is the normal, expected pre-solve state. Requiring
    # it here wrongly skipped every rate->rate shift whose target wasn't already
    # anchored, killing propagation past the first rate frame.
    #
    # A fully-cloudy source gets no WCS / no starfield. Mark the shift processed-
    # but-invalid and return: the caller's loop pulls the next *unprocessed* shift
    # (SenpaiRun.get_next_shift), so returning without setting processed=True hands
    # the same shift back forever — a livelock (observed in _full7).
    if rate_frame_a.starfield is None:
        logger.warning(
            "Skipping rate-to-rate shift %d->%d: source frame missing starfield "
            "— frame likely had no WCS solution.",
            frame_shift.source_index, frame_shift.target_index,
        )
        frame_shift.processed = True
        frame_shift.is_valid = False
        frame_shift.error_message = "Missing starfield (no WCS solution)"
        return

    frame_exposure_gap_seconds = abs(
        (rate_frame_a.timestamp - rate_frame_b.timestamp).total_seconds()
    )
    rate_a_exposure_time = rate_frame_a.frame.header.get("EXPTIME", 1)
    rate_b_exposure_time = rate_frame_b.frame.header.get("EXPTIME", 1)

    # Get the average pixel track rate if available from both frames
    rates = []
    if rate_frame_a.pixel_track_rate_per_second is not None:
        rates.append(rate_frame_a.pixel_track_rate_per_second)
    if rate_frame_b.pixel_track_rate_per_second is not None:
        rates.append(rate_frame_b.pixel_track_rate_per_second)

    if rates:
        pixel_track_rate_per_second = np.mean(rates)
    else:
        pixel_track_rate_per_second = None

    fwhms = []
    if rate_frame_a.streak is not None:
        fwhms.append(rate_frame_a.streak.fwhm)
    if rate_frame_b.streak is not None:
        fwhms.append(rate_frame_b.streak.fwhm)

    # Both frames can lack a streak (failed extraction); downstream consumers
    # (cc_downsample_factor, the metadata headers) all accept fwhm=None.
    streak_fwhm = float(np.mean(fwhms)) if fwhms else None

    rate_a_data = prepare_rate_frame(rate_frame_a)
    rate_b_data = prepare_rate_frame(rate_frame_b)

    # whopping bright streaks can mess with correlation
    rate_a_data, n_rate_a_removed = remove_near_saturation_streaks(
        rate_a_data, rate_frame_a.frame.data_type
    )
    rate_b_data, n_rate_b_removed = remove_near_saturation_streaks(
        rate_b_data, rate_frame_b.frame.data_type
    )

    rate_a_data = remove_border_crossing_streaks(rate_a_data)
    rate_b_data = remove_border_crossing_streaks(rate_b_data)

    # strip_unbalanced_streaks(rate_a_data, rate_b_data)

    rate_a_data = rate_a_data / np.std(rate_a_data)
    rate_a_data -= np.mean(rate_a_data)
    rate_b_data = rate_b_data / np.std(rate_b_data)
    rate_b_data -= np.mean(rate_b_data)

    streak_measurements = StreakMeasurements()

    """
    streak_measurements.header = extract_streak_from_metadata(
        rate_frame_b.frame_metadata,
        plate_scale_arcsec=rate_frame_a.starfield.wcs_metadata.x_ifov_arcsec,
        wcs_model=rate_frame_a.starfield.wcs,
    )
    if streak_measurements.header is not None:
        streak_measurements.header.fwhm = streak_fwhm
    """
    # Cheap mapping extraction on frame B solely to seed the cross-correlation mask
    # radius below (the robust extraction + rate come later). Dropped here: the
    # frame-A mapping call and both "cleaned_*" outputs were entirely unused, and
    # previous_frame only fed the now-removed median blend.
    streak_measurements.streak_mapping, _ = extract_streak_dims_mapping(
        rate_b_data, n_streaks=5
    )

    # Fast fourier-based (whitened/phase) cross correlation. Downsample first
    # (block-MEDIAN — robust to hot pixels that spike the whitened CC) so the FFTs
    # run on far fewer pixels; the shift it needs is coarse and the catalog-star
    # match downstream refines to sub-pixel. The CC is upsampled back to full-res
    # extent so all the masking/peak/refine logic below is unchanged.
    cc_ds = cc_downsample_factor(streak_fwhm, rate_a_data.shape)
    if cc_ds > 1:
        from scipy.ndimage import zoom

        a_white = whiten_image(_block_median_downsample(rate_a_data, cc_ds))
        b_white = whiten_image(_block_median_downsample(rate_b_data, cc_ds))
        cc_small = gaussian_filter(cross_corr(a_white, b_white), sigma=3)
        cross_correlated_image = zoom(cc_small, cc_ds, order=1)
        logger.info(
            "Cross-correlation at 1/%d resolution (%s -> %s) for speed",
            cc_ds, rate_a_data.shape, cc_small.shape,
        )
    else:
        cross_correlated_image = gaussian_filter(
            cross_corr(whiten_image(rate_a_data), whiten_image(rate_b_data)), sigma=3
        )

    # Calculate expected shift from track rate if available
    expected_shift = None
    if pixel_track_rate_per_second is not None:
        # expected pixel shift...
        expected_shift = pixel_track_rate_per_second * (
            frame_exposure_gap_seconds
            + 0.5 * (rate_a_exposure_time + rate_b_exposure_time)
        )

    # After getting initial measurements, mask the cross-correlation frame based on expected shift
    if streak_measurements.streak_mapping is not None:
        # Calculate maximum expected shift from streak measurements
        # For rate-to-rate, we expect similar streak lengths in both frames
        streak_rate = streak_measurements.streak_mapping.length / rate_a_exposure_time

        # Total time for shift is just the gap between frames
        # (both frames are rate tracked, so object appears stationary during exposures)
        total_shift_time = frame_exposure_gap_seconds

        max_expected_shift = streak_rate * total_shift_time

        # Use the larger of our two estimates for masking. A failed streak
        # extraction can leave length/exposure NaN → NaN radius → int() crash;
        # the mask is only a CC-peak-suppression optimization, so fall back to
        # the expected_shift estimate or skip masking rather than dying.
        if not np.isfinite(max_expected_shift):
            logger.warning(
                "Streak-based shift estimate is not finite "
                "(length=%s, exposure=%s); masking from expected_shift only.",
                streak_measurements.streak_mapping.length, rate_a_exposure_time,
            )
            max_expected_shift = None
        if expected_shift is not None and max_expected_shift is not None:
            mask_radius = int(2.0 * max(max_expected_shift, expected_shift))
        elif expected_shift is not None:
            mask_radius = int(2.0 * expected_shift)
        elif max_expected_shift is not None:
            mask_radius = int(2.0 * max_expected_shift)
        else:
            mask_radius = None

        if mask_radius is not None:
            logger.info(
                f"Masking cross-correlation outside radius {mask_radius:.1f}px "
                f"(from streak length={streak_measurements.streak_mapping.length:.1f}px, "
                f"rate={streak_rate:.1f}px/s, time={total_shift_time:.1f}s)"
            )

            # Create circular mask centered on image
            center_y, center_x = (
                cross_correlated_image.shape[0] // 2,
                cross_correlated_image.shape[1] // 2,
            )
            y, x = np.ogrid[
                -center_y : cross_correlated_image.shape[0] - center_y,
                -center_x : cross_correlated_image.shape[1] - center_x,
            ]
            mask = x * x + y * y <= mask_radius * mask_radius

            # Set everything outside mask to minimum value
            cross_correlated_image[~mask] = np.min(cross_correlated_image)

        if get_config().plotting.debug:
            from senpai.engine.plotting.images import plot_single_frame

            plot_single_frame(
                cross_correlated_image,
                scale=False,
                output_file=get_config().runtime.output_dir
                / f"masked_cc_{rate_frame_a.index}_{rate_frame_b.index}.png",
            )

    # mask center of subcc
    mask_center = 0.2 * expected_shift if expected_shift else 3

    scale = 150
    if expected_shift:
        scale = int(expected_shift * 1.2)

    valid = False
    max_trials = 10
    trials = 0
    best_correlation_score = -1
    best_shift = None  # Track the best shift we've found
    shift_rate_to_rate_xy = None  # Ensure defined for downstream usage

    """
    # Early try: validate shift from prior pixel track rate and frame gap only
    if (
        (not valid)
        and (pixel_track_rate_per_second is not None)
        and (rate_frame_a.streak is not None)
    ):
        prior_gap_only_shift_mag = (
            pixel_track_rate_per_second * frame_exposure_gap_seconds
        )
        theta_rad = np.deg2rad(rate_frame_a.streak.degree_angle())
        expected_shift_x = -1 * prior_gap_only_shift_mag * np.cos(theta_rad)
        expected_shift_y = -1 * prior_gap_only_shift_mag * np.sin(theta_rad)

        logger.info(
            f"Trying prior shift first: ({expected_shift_x:.2f}, {expected_shift_y:.2f}) from rate={pixel_track_rate_per_second:.3f} px/s and gap={frame_exposure_gap_seconds:.3f}s"
        )

        # validate_proposed_shift(rate_frame_b, rate_frame_a, -expected_shift_x, -expected_shift_y, rate_frame_a.starfield.catalog_stars, trial=0)
        (
            prior_valid,
            prior_corr,
            prior_validation_measurement,
            prior_correction,
        ) = validate_proposed_shift(
            rate_frame_b,
            rate_frame_a,
            -expected_shift_x,
            -expected_shift_y,
            rate_frame_a.starfield.catalog_stars,
            fwhm_exclusion=3 * (rate_frame_a.seeing.pixel_fwhm if rate_frame_a.seeing else 4.0),
            trial=0,
        )

        if prior_valid:
            valid = True
            best_correlation_score = prior_corr
            best_shift = np.array([-expected_shift_x, -expected_shift_y]) - np.array(
                prior_correction
            )
            shift_rate_to_rate_xy = np.array([-expected_shift_x, -expected_shift_y])
            logger.info(
                f"Prior shift validated successfully (corr={prior_corr:.3f}). Using it and skipping search trials."
            )
        else:
            logger.info(
                f"Prior shift validation failed (corr={prior_corr:.3f}). Will try correlation search."
            )
    """
    while not valid and trials < max_trials:
        trials += 1

        # mask center of subcc
        subcc = cross_correlated_image.copy()
        center = np.array(cross_correlated_image.shape) / 2
        subcc[
            int(center[0] - mask_center) : int(center[0] + mask_center),
            int(center[1] - mask_center) : int(center[1] + mask_center),
        ] = np.min(subcc)

        # window down subcc
        # Ensure we stay within bounds and calculate proper limits
        y_min = max(0, int(center[0] - scale))
        y_max = min(subcc.shape[0], int(center[0] + scale))
        x_min = max(0, int(center[1] - scale))
        x_max = min(subcc.shape[1], int(center[1] + scale))

        subcc = subcc[y_min:y_max, x_min:x_max]

        cc_max = np.unravel_index(np.argmax(subcc), subcc.shape)

        # move back into original frame scale - adjust for actual window position
        # Calculate the actual offset from original center based on where we windowed
        cc_max = np.array(
            [
                cc_max[0] + y_min,  # Add y_min to y-coordinate
                cc_max[1] + x_min,  # Add x_min to x-coordinate
            ]
        )
        original_center = np.array(cross_correlated_image.shape) / 2

        # Calculate shift vector (from center to max correlation point)
        shift_rate_to_rate_xy = original_center - cc_max
        # Convert to (x,y) ordering
        shift_rate_to_rate_xy = shift_rate_to_rate_xy[::-1]

        shift_rate_to_rate_xy = refine_correlation_shift_by_global_shift(
            rate_frame_a, rate_frame_b, rate_frame_a.streak, shift_rate_to_rate_xy
        )

        valid, correlation_score, _, shift_correction = (
            validate_proposed_shift(
                rate_frame_b,
                rate_frame_a,
                shift_rate_to_rate_xy[0],
                shift_rate_to_rate_xy[1],
                rate_frame_a.starfield.catalog_stars,
                fwhm_exclusion=3 * (rate_frame_a.seeing.pixel_fwhm if rate_frame_a.seeing else 4.0),
                trial=trials,
            )
        )

        # Track the best correlation score and corresponding shift
        if correlation_score > best_correlation_score:
            best_correlation_score = correlation_score
            best_shift = shift_rate_to_rate_xy.copy() - shift_correction

        # Log the shift correction for analysis
        if shift_correction != (0.0, 0.0):
            logger.info(
                f"Trial {trials}: Shift correction = ({shift_correction[0]:.2f}, {shift_correction[1]:.2f})"
            )

        if get_config().plotting.debug:
            from senpai.engine.plotting.images import plot_single_frame

            fig, ax = plot_single_frame(
                cross_correlated_image,
                scale=False,
            )

            # Markers: red '+' at center, blue 'x' at cc_max
            center_y, center_x = (
                cross_correlated_image.shape[0] // 2,
                cross_correlated_image.shape[1] // 2,
            )
            ax.plot(
                center_x, center_y, "+", color="red", markersize=10, markeredgewidth=2
            )
            ax.plot(
                int(cc_max[1]),
                int(cc_max[0]),
                "x",
                color="blue",
                markersize=10,
                markeredgewidth=2,
            )

            plt.savefig(
                get_config().runtime.output_dir
                / f"rate_to_rate_cc_{rate_frame_a.index}_{rate_frame_b.index}_{trials}.png"
            )
            plt.close("all")

        if not valid:
            # Get the shift from the rate-to-rate alignment

            # Calculate expected streak length in a single frame (not shift between frames!)
            # Priority order: 1) Use previous frame streak, 2) Use track rate, 3) Calculate from shift
            if rate_frame_a.streak is not None:
                # Best: Use the actual measured streak length from frame A
                streak_length_expected = rate_frame_a.streak.pixel_length
                logger.info(
                    f"Using streak length from frame A: {streak_length_expected:.1f}px"
                )
            elif pixel_track_rate_per_second is not None:
                # Good: Use known track rate * exposure time
                streak_length_expected = (
                    pixel_track_rate_per_second * rate_b_exposure_time
                )
                logger.info(
                    f"Using streak length from track rate: {streak_length_expected:.1f}px "
                    f"(rate={pixel_track_rate_per_second:.1f}px/s * exp={rate_b_exposure_time:.1f}s)"
                )
            else:
                # Fallback: Calculate rate from shift, then multiply by exposure
                pixel_shift = np.linalg.norm(shift_rate_to_rate_xy)
                if pixel_shift > 0 and frame_exposure_gap_seconds > 0:
                    rate_estimate = pixel_shift / frame_exposure_gap_seconds
                    streak_length_expected = rate_estimate * rate_b_exposure_time
                    logger.info(
                        f"Using streak length from shift: {streak_length_expected:.1f}px "
                        f"(shift={pixel_shift:.1f}px / gap={frame_exposure_gap_seconds:.1f}s * exp={rate_b_exposure_time:.1f}s)"
                    )
                else:
                    streak_length_expected = 20  # Conservative fallback
                    logger.warning(
                        f"Using fallback streak length: {streak_length_expected}px"
                    )

            # Box size should be just large enough for the streak plus small margin
            # Keep it tight to avoid picking up background structure
            pixel_fwhm = rate_frame_a.seeing.pixel_fwhm if rate_frame_a.seeing else 4.0
            box_size = int(max(streak_length_expected * 0.75, 10 * pixel_fwhm))

            logger.info(
                f"Trial {trials}: Removing streak at cc_max=({cc_max[0]}, {cc_max[1]}), "
                f"box_size={box_size} (streak_length={streak_length_expected:.1f}px, "
                f"FWHM={pixel_fwhm:.1f}px)"
            )

            # Use robust streak removal that finds contiguous regions
            # This is more reliable than threshold-based approaches when
            # signal/variance is very high
            cross_correlated_image, removal_info = remove_streak_at_point_robust(
                cross_correlated_image,
                start_point=(int(cc_max[0]), int(cc_max[1])),
                box_size=box_size,
                pad_size=1,  # Light dilation to catch edges without over-masking
                logger=logger,
            )

            logger.info(
                f"Removed {removal_info['num_pixels']} pixels using robust method "
                f"(before dilation: {removal_info.get('num_pixels_before_dilation', 'N/A')})"
            )

    # If we never found a valid shift, mark the frame_shift as invalid
    if not valid:
        if best_correlation_score > 0.7:
            logger.warning(
                f"Failed to find valid shift after {trials} trials. Best correlation score: {best_correlation_score:.4f}, continuing"
            )
        else:
            logger.warning(
                f"Failed to find valid shift after {trials} trials. Best correlation score: {best_correlation_score:.4f}"
            )
            frame_shift.processed = True
            frame_shift.is_valid = False
            frame_shift.error_message = "Failed to validate shift"
            return

    # Calculate the magnitude of the shift - apply the -1 adjustment here for consistency
    # This ensures the magnitude calculation matches the adjusted shift values
    adjusted_shift = np.array([best_shift[0] - 1, best_shift[1] - 1])
    pixel_shift = np.linalg.norm(adjusted_shift)

    # Calculate rate based on the time between frames
    estimated_pixel_track_rate_per_second = pixel_shift / frame_exposure_gap_seconds

    logger.info(
        f"Pixel shift rate to rate: {pixel_shift:.1f} pixels, {estimated_pixel_track_rate_per_second:.1f} pixels/s"
    )

    frame_shift.x_shift = best_shift[0] - 1
    frame_shift.y_shift = best_shift[1] - 1

    frame_shift.processed = True
    frame_shift.is_valid = True
    frame_shift.error_message = None

    streak_length_expected_from_shift = (
        estimated_pixel_track_rate_per_second * rate_a_exposure_time
    )
    streak_orientation_expected_from_shift = np.rad2deg(
        np.arctan2(shift_rate_to_rate_xy[1], shift_rate_to_rate_xy[0])
    )
    streak_measurements.frame_to_frame = StreakMeasurement(
        rotation=streak_orientation_expected_from_shift,
        length=streak_length_expected_from_shift,
        fwhm=streak_fwhm,
    )

    frame_extraction_measurement, psf, fwhm = extract_streak_dims_robust(
        rate_b_data,
        n_streaks=5,
        rotation=streak_orientation_expected_from_shift,
        length=streak_length_expected_from_shift,
        fwhm=streak_fwhm,
    )

    streak_measurements.frame_extraction, fwhm = refine_robust_streak(
        psf, frame_extraction_measurement, frame_index=rate_frame_b.index
    )

    if get_config().plotting.debug and psf is not None:
        from senpai.engine.plotting.images import plot_single_frame

        plot_single_frame(
            psf,
            scale=False,
            output_file=get_config().runtime.output_dir
            / f"{rate_frame_b.index}_streak_psf.png",
        )

    # Sanity-check the (already star-validated) shift against the directly-measured
    # streak. We trust the streak (frame_extraction); both measure the same motion,
    # so a disagreement points to a registration error in the shift. When they
    # disagree, try the shift the trusted streak implies and adopt it ONLY if it
    # validates against the stars at least as well. We never discard the extraction
    # and never reject the frame on disagreement alone (the shift already passed star
    # validation in the trial loop, and the streak carries the geometry/rate anyway).
    fe = streak_measurements.frame_extraction
    ftf = streak_measurements.frame_to_frame
    if (
        fe is not None
        and ftf is not None
        and (
            abs(ftf.length - fe.length) > 0.4 * fe.length
            or angular_difference(ftf.rotation, fe.rotation) > 15.0
        )
    ):
        logger.warning(
            f"Shift-derived motion disagrees with the extracted streak "
            f"(shift: len={ftf.length:.1f}px rot={ftf.rotation:.1f}°; "
            f"streak: len={fe.length:.1f}px rot={fe.rotation:.1f}°) — trying a "
            f"streak-derived shift"
        )
        # Shift implied by the trusted streak: rate (len / exposure) over the gap,
        # along the streak orientation.
        expected_rate_per_second = fe.length / rate_b_exposure_time
        expected_shift_magnitude = expected_rate_per_second * frame_exposure_gap_seconds
        expected_rotation_rad = np.deg2rad(fe.rotation)
        expected_shift_x = expected_shift_magnitude * np.cos(expected_rotation_rad)
        expected_shift_y = expected_shift_magnitude * np.sin(expected_rotation_rad)

        expected_valid, expected_correlation, _, expected_shift_correction = (
            validate_proposed_shift(
                rate_frame_b,
                rate_frame_a,
                expected_shift_x,
                expected_shift_y,
                rate_frame_a.starfield.catalog_stars,
                fwhm_exclusion=3 * (rate_frame_a.seeing.pixel_fwhm if rate_frame_a.seeing else 4.0),
                trial=999,
            )
        )
        # Bias toward the streak-derived shift (>= 0.8x current correlation) — we
        # trust the streak — but require it to actually validate against the stars.
        if expected_valid and expected_correlation > 0.8 * best_correlation_score:
            logger.info(
                f"Adopting streak-derived shift (corr {expected_correlation:.3f} vs "
                f"{best_correlation_score:.3f})"
            )
            best_shift = np.array([expected_shift_x, expected_shift_y]) - np.array(
                expected_shift_correction
            )
            adjusted_shift = np.array([best_shift[0] - 1, best_shift[1] - 1])
            pixel_shift = np.linalg.norm(adjusted_shift)
            estimated_pixel_track_rate_per_second = (
                pixel_shift / frame_exposure_gap_seconds
            )
            frame_shift.x_shift = best_shift[0] - 1
            frame_shift.y_shift = best_shift[1] - 1
        else:
            logger.info(
                "Streak-derived shift did not validate better; keeping the original "
                "(star-validated) shift. Streak geometry/rate still come from the "
                "extraction."
            )

    # Trust the directly-measured streak (frame_extraction); fall back to the
    # shift-derived frame_to_frame only if the extraction failed or was discarded
    # by the consistency check above.
    streak = streak_measurements.frame_extraction
    if streak is None:
        streak = streak_measurements.frame_to_frame
        if streak is not None:
            logger.warning(
                "frame_extraction unavailable, using shift-derived frame_to_frame"
            )
    if streak is None:
        logger.error("No streak measurement available, cannot set streak metadata")
        return

    # Reported rate from the streak length when we have a direct extraction
    # (model-free: a trail of length L over exposure t IS a rate L/t). Otherwise
    # fall back to the clean rate-rate shift rate (shift / gap, both trail
    # midpoints — no x2 hybrid). Cross-check the two and log the ratio.
    if streak_measurements.frame_extraction is not None and rate_b_exposure_time:
        rate_from_streak = (
            streak_measurements.frame_extraction.length / rate_b_exposure_time
        )
        ratio = (
            estimated_pixel_track_rate_per_second / rate_from_streak
            if rate_from_streak
            else float("nan")
        )
        logger.info(
            f"Rate (frame {rate_frame_b.index}): {rate_from_streak:.2f}px/s from streak "
            f"(len={streak_measurements.frame_extraction.length:.1f}px / "
            f"exp={rate_b_exposure_time:.1f}s); shift cross-check ratio={ratio:.2f}"
        )
        if rate_from_streak > 0 and abs(ratio - 1.0) > 0.25:
            logger.warning(
                f"Rate cross-check off on frame {rate_frame_b.index}: shift-rate is "
                f"{ratio:.2f}x the streak-rate — bad extraction/shift or timing baseline"
            )
        reported_rate = rate_from_streak
    else:
        reported_rate = estimated_pixel_track_rate_per_second

    rate_frame_b.streak = StreakMetadata(
        pixel_length=streak.length,
        sine_angle=np.sin(np.deg2rad(streak.rotation)),
        cosine_angle=np.cos(np.deg2rad(streak.rotation)),
        fwhm=streak.fwhm,
    )
    rate_frame_b.seeing = rate_frame_a.seeing
    rate_frame_b.pixel_track_rate_per_second = reported_rate
