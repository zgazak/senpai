import logging

import matplotlib.pyplot as plt
import numpy as np

from senpai.core.config import get_config
from senpai.engine.detection.streak.extraction import (
    cross_corr,
    extract_streak_dims_robust,
    measure_streak_shift_centroid,
    prepare_rate_frame,
    prepare_sidereal_frame,
    refine_robust_streak,
    streak_parameters_from_xcorr,
)
from senpai.engine.detection.streak.masking import (
    remove_border_crossing_streaks,
    remove_streak_at_point_robust,
)
from senpai.engine.detection.streak.validation import validate_proposed_shift
from senpai.engine.models.metadata import StreakMetadata
from senpai.engine.models.senpai import FrameShift, RateTrackFrame, SiderealFrame
from senpai.engine.models.streak_measurement import (
    StreakMeasurement,
    StreakMeasurements,
    angular_difference,
    normalize_angle,
)
from senpai.engine.plotting.images import plot_single_frame

logger = logging.getLogger(__name__)


def solve_rate_from_sidereal(
    sidereal_frame: SiderealFrame, rate_frame: RateTrackFrame, frame_shift: FrameShift
):
    config = get_config()

    # A fully-cloudy anchor gets no WCS, so its starfield (and detection
    # metadata / fwhm) is None — reading it crashed the whole calsat batch.
    # Mark the shift processed-but-invalid and skip, like solve_rate_from_rate,
    # so the loop makes progress and the batch completes gracefully instead of
    # failing (the frame is unusable without a WCS anyway).
    if (sidereal_frame.starfield is None
            or sidereal_frame.starfield.detection_metadata is None):
        logger.warning(
            "Skipping sidereal-rate shift %d->%d: missing starfield/WCS.",
            frame_shift.source_index, frame_shift.target_index,
        )
        frame_shift.processed = True
        frame_shift.is_valid = False
        frame_shift.error_message = "Missing starfield (no WCS solution)"
        return

    frame_exposure_gap_seconds = abs(
        (sidereal_frame.timestamp - rate_frame.timestamp).total_seconds()
    )
    rate_exposure_time = rate_frame.frame.header.get("EXPTIME", 1)
    sidereal_exposure_time = sidereal_frame.frame.header.get("EXPTIME", 1)

    pixel_fwhm = sidereal_frame.starfield.detection_metadata.pixel_fwhm

    sidereal_data, is_synthetic = prepare_sidereal_frame(sidereal_frame)
    rate_data = prepare_rate_frame(rate_frame)

    # whopping bright streaks can mess with correlation

    """
    rate_data, removed_streaks = remove_near_saturation_streaks(
        rate_data, rate_frame.frame.data_type
    )
    if not is_synthetic:
        sidereal_data, removed_streaks = remove_n_brightest_streaks(
            sidereal_data, removed_streaks
        )
    """

    rate_data = remove_border_crossing_streaks(rate_data)
    sidereal_data = remove_border_crossing_streaks(sidereal_data)

    streak_measurements = StreakMeasurements()

    # Extract the rate-frame streak ONCE. The streak in rate_data is a single fixed
    # object, so one robust extraction is all we need — it serves as both the seed
    # for the cross-correlation mask radius below and the final streak measurement.
    # (Historically this was extracted ~3x/frame and blended via sigma_clipped_mean
    # with shift-derived estimates to outvote a flaky extractor; the robust method
    # is reliable now, so we trust it directly and drop the redundant passes + blend.)
    initial_measurement, psf, _ = extract_streak_dims_robust(
        rate_data,
        n_streaks=5,
        rotation=None,
        length=None,
        fwhm=pixel_fwhm,
    )
    if initial_measurement is not None:
        streak_measurements.frame_extraction, _ = refine_robust_streak(
            psf, initial_measurement, frame_index=rate_frame.index
        )
    if streak_measurements.frame_extraction is not None:
        fe = streak_measurements.frame_extraction
        logger.info(
            f"Rate-frame streak: length={fe.length:.1f}px, rotation={fe.rotation:.1f}°, "
            f"fwhm={fe.fwhm:.1f}"
        )
    else:
        logger.warning(
            "Could not extract rate-frame streak, proceeding without masking"
        )

    logger.info("Cross correlating rate and sidereal frames")

    # fast fourier-based cross correlation
    cross_correlated_image = cross_corr(sidereal_data, rate_data)

    # After getting initial measurements, mask the cross-correlation frame based on expected shift
    if (
        streak_measurements.frame_extraction is not None
        and streak_measurements.frame_extraction.length > 10
    ):
        # Only apply masking if we have a reasonable length estimate (> 10 pixels)
        # Calculate maximum expected shift from streak measurements
        streak_rate = streak_measurements.frame_extraction.length / rate_exposure_time
        total_shift_time = (
            frame_exposure_gap_seconds  # Time between frames (including settling)
            + 0.5 * sidereal_exposure_time  # Half sidereal exposure
            + 0.5 * rate_exposure_time  # Half rate exposure
        )
        max_expected_shift = streak_rate * total_shift_time

        # Use 2x the expected shift as our mask radius to be conservative
        mask_radius = int(2.0 * max_expected_shift)
        logger.info(
            f"Masking cross-correlation outside radius {mask_radius:.1f}px "
            f"(from streak length={streak_measurements.frame_extraction.length:.1f}px, "
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

        if config.plotting.debug:
            plot_single_frame(
                cross_correlated_image,
                scale=False,
                output_file=config.runtime.output_dir
                / f"masked_cc_{sidereal_frame.index}-{rate_frame.index}.png",
            )
    else:
        logger.info(
            "Skipping cross-correlation masking (no reliable initial estimate or length too short)"
        )

    # cross_correlated_image = background_subtract(cross_correlated_image, box_size=100)

    valid = False
    max_trials = 10
    trials = 0
    best_correlation_score = -1
    best_shift = None  # Track the best shift we've found

    """
    streak_measurements.header = extract_streak_from_metadata(
        rate_frame.frame_metadata,
        plate_scale_arcsec=sidereal_frame.starfield.wcs_metadata.x_ifov_arcsec,
        wcs_model=sidereal_frame.starfield.wcs,
    )
    """

    while not valid and trials < max_trials:
        trials += 1

        # extract rotation and streak length from cc, assuming alignment feature is brightest
        # rotation_estimate_1, length_estimate_1, subcc =
        # Re-extract the CC streak parameters each trial — the cross-correlation
        # image is progressively masked across trials, so this legitimately changes.
        # The rate-frame streak geometry (frame_extraction) was extracted once
        # before the loop and is reused here for expected_distance / validation.
        streak_measurements.cross_correlation, subcc = streak_parameters_from_xcorr(
            cross_correlated_image,
            plate_scale_arcsec=sidereal_frame.starfield.wcs_metadata.x_ifov_arcsec,
            seeing_fwhm_pixels=pixel_fwhm,
        )

        # Use centroid-based measurement for streaks in cross-correlation
        # The cross-correlation of point sources with streaks produces a streak
        # Calculate expected shift distance from streak measurements
        if streak_measurements.frame_extraction is not None:
            streak_rate = (
                streak_measurements.frame_extraction.length / rate_exposure_time
            )
            total_shift_time = (
                frame_exposure_gap_seconds
                + 0.5 * sidereal_exposure_time
                + 0.5 * rate_exposure_time
            )
            expected_distance = streak_rate * total_shift_time
            logger.info(
                f"Expected shift distance: {expected_distance:.1f}px "
                f"(rate={streak_rate:.1f}px/s, time={total_shift_time:.1f}s)"
            )
        else:
            expected_distance = None

        # we need to reverse this.  it's complicated
        pixel_shift_rate_to_sidereal_xy = -1 * measure_streak_shift_centroid(
            subcc, expected_max_distance=expected_distance, intensity_threshold=0.5
        )
        if config.plotting.debug:
            _, ax = plot_single_frame(
                subcc,
                scale=False,
            )

            # Add a marker at the center of the image + pixel shift

            center_y, center_x = subcc.shape[0] // 2, subcc.shape[1] // 2
            shifted_x = center_x - pixel_shift_rate_to_sidereal_xy[0]
            shifted_y = center_y - pixel_shift_rate_to_sidereal_xy[1]

            ax.plot(
                shifted_x,
                shifted_y,
                "x",
                color="blue",
                markersize=10,
                markeredgewidth=2,
            )

            ax.plot(
                center_x,
                center_y,
                "+",
                color="red",
                markersize=10,
                markeredgewidth=2,
            )

            plt.savefig(
                config.runtime.output_dir
                / f"sidereal_to_rate_cc_{sidereal_frame.index}-{rate_frame.index}-{trials}.png"
            )
            plt.close("all")

        # Pass streak rotation to validation to avoid sampling along streak direction
        streak_rotation = None
        if streak_measurements.frame_extraction is not None:
            streak_rotation = streak_measurements.frame_extraction.rotation

        valid, correlation_score, streak_measurements.validation, shift_correction = (
            validate_proposed_shift(
                rate_frame,
                sidereal_frame,
                pixel_shift_rate_to_sidereal_xy[0],
                pixel_shift_rate_to_sidereal_xy[1],
                sidereal_frame.starfield.catalog_stars,
                trials,
                fwhm_exclusion=3 * (sidereal_frame.seeing.pixel_fwhm if sidereal_frame.seeing else 4.0),
            )
        )

        # Track the best correlation score and corresponding shift
        if correlation_score > best_correlation_score:
            best_correlation_score = correlation_score
            best_shift = pixel_shift_rate_to_sidereal_xy.copy() - shift_correction

        # Log the shift correction for analysis
        if shift_correction != (0.0, 0.0):
            logger.info(
                f"Trial {trials}: Shift correction = ({shift_correction[0]:.2f}, {shift_correction[1]:.2f})"
            )

        if not valid:
            x, y = pixel_shift_rate_to_sidereal_xy.astype(int)
            # Get the shift from the rate-to-sidereal alignment
            center_y, center_x = (
                cross_correlated_image.shape[0] // 2,
                cross_correlated_image.shape[1] // 2,
            )
            # Use minus to match the plotting convention (see lines 222-223)
            shifted_x = center_x - x
            shifted_y = center_y - y

            logger.info(
                f"Shift calculation: x={x}, y={y}, "
                f"center=({center_y}, {center_x}), "
                f"shifted=({shifted_y}, {shifted_x})"
            )

            # Use robust streak removal that finds contiguous regions
            # This is more reliable than threshold-based approaches when
            # signal/variance is very high

            # Calculate expected streak length in a single frame (not shift between frames!)
            # Priority: 1) Use previous track rate, 2) Calculate from shift
            if (
                hasattr(rate_frame, "pixel_track_rate_per_second")
                and rate_frame.pixel_track_rate_per_second is not None
            ):
                # Use known track rate * exposure time
                streak_length_expected = (
                    rate_frame.pixel_track_rate_per_second * rate_exposure_time
                )
                logger.info(
                    f"Using streak length from track rate: {streak_length_expected:.1f}px "
                    f"(rate={rate_frame.pixel_track_rate_per_second:.1f}px/s * exp={rate_exposure_time:.1f}s)"
                )
            else:
                # Calculate rate from shift, then multiply by exposure
                pixel_shift = np.linalg.norm(pixel_shift_rate_to_sidereal_xy)
                if pixel_shift > 0 and frame_exposure_gap_seconds > 0:
                    rate_estimate = pixel_shift / frame_exposure_gap_seconds
                    streak_length_expected = rate_estimate * rate_exposure_time
                    logger.info(
                        f"Using streak length from shift: {streak_length_expected:.1f}px "
                        f"(shift={pixel_shift:.1f}px / gap={frame_exposure_gap_seconds:.1f}s * exp={rate_exposure_time:.1f}s)"
                    )
                else:
                    streak_length_expected = 20  # Conservative fallback
                    logger.warning(
                        f"Using fallback streak length: {streak_length_expected}px"
                    )

            # Box size should be just large enough for the streak plus small margin
            # Keep it tight to avoid picking up background structure
            box_size = int(max(streak_length_expected * 0.75, 10 * pixel_fwhm))

            logger.info(
                f"Using box_size={box_size} (streak_length={streak_length_expected:.1f}px, "
                f"FWHM={pixel_fwhm:.1f}px)"
            )

            cross_correlated_image, removal_info = remove_streak_at_point_robust(
                cross_correlated_image,
                [int(shifted_y), int(shifted_x)],
                box_size,
                pad_size=1,  # Light dilation to catch edges without over-masking
                logger=logger,
            )

            logger.info(
                f"Removed {removal_info['num_pixels']} pixels using robust method "
                f"(before dilation: {removal_info.get('num_pixels_before_dilation', 'N/A')}, "
                f"thresholds: {removal_info['thresholds_tried']}), "
                f"region bounds: y[{removal_info.get('y_min', 'N/A')}:{removal_info.get('y_max', 'N/A')}], "
                f"x[{removal_info.get('x_min', 'N/A')}:{removal_info.get('x_max', 'N/A')}]"
            )

    # If we never found a valid shift, mark the frame_shift as invalid
    if not valid:
        if best_correlation_score > 0.7:
            logger.warning(
                f"Failed to find valid shift after {trials} trials. "
                f"Best correlation score: {best_correlation_score:.4f}, continuing"
            )
        else:
            logger.warning(
                f"Failed to find valid shift after {trials} trials. "
                f"Best correlation score: {best_correlation_score:.4f}"
            )
            frame_shift.processed = True
            frame_shift.is_valid = False
            frame_shift.error_message = (
                "Could not validate shift between sidereal and rate frames"
            )
            return

    pixel_shift_rate_to_sidereal = np.linalg.norm(best_shift)

    # The shift gives us the streak DIRECTION (gold — it's star-validated) and a
    # cross-check on the magnitude. The magnitude formula below is suspect: the x2 +
    # (gap + 0.5*E_r) denominator is an inconsistent hybrid of two mount-cadence
    # models, and there's no trustworthy external rate to calibrate it against
    # (header track rates are unreliable). So we DON'T use it for the reported rate —
    # the streak length does that (model-free; see below). Kept only as a cross-check.
    shift_rate_per_second_suspect = (
        2
        * pixel_shift_rate_to_sidereal
        / (frame_exposure_gap_seconds + 0.5 * rate_exposure_time)
    )

    # Streak direction from the (star-validated) shift vector.
    rotation_estimate_frame_to_frame = np.rad2deg(
        np.arctan2(best_shift[1], best_shift[0])
    )

    streak_measurements.frame_to_frame = StreakMeasurement(
        rotation=rotation_estimate_frame_to_frame,
        length=shift_rate_per_second_suspect * rate_exposure_time,
        fwhm=pixel_fwhm,
    )

    # If we couldn't extract a valid streak from the frame, mark as invalid
    if streak_measurements.frame_extraction is None:
        logger.warning(
            f"Failed to extract streak from rate frame {rate_frame.index} - marking shift as invalid"
        )
        frame_shift.processed = True
        frame_shift.is_valid = False
        frame_shift.error_message = "Could not extract valid streak from rate frame"
        return

    if config.plotting.debug and psf is not None:
        plot_single_frame(
            psf,
            scale=False,
            output_file=config.runtime.output_dir
            / f"{rate_frame.index}_streak_psf.png",
        )

    # The reported rate comes from the streak length — model-free, no cross-frame
    # timing or mount-cadence assumptions: a trail of length L over exposure t IS a
    # rate L/t. The shift-derived rate is the cross-check, not the source of truth.
    # We log the shift/streak ratio every frame: a *systematic* departure from 1.0
    # is the suspect x2/baseline showing itself; a *one-off* departure flags a bad
    # extraction or shift on that frame. Direction comes from the validated shift.
    streak = streak_measurements.frame_extraction
    rate_from_streak = streak.length / rate_exposure_time if rate_exposure_time else 0.0
    if rate_from_streak > 0:
        shift_streak_ratio = shift_rate_per_second_suspect / rate_from_streak
        angle_disagreement = angular_difference(
            streak.rotation, rotation_estimate_frame_to_frame
        )
        logger.info(
            f"Rate (frame {rate_frame.index}): {rate_from_streak:.2f}px/s from streak "
            f"(len={streak.length:.1f}px / exp={rate_exposure_time:.1f}s); "
            f"shift cross-check ratio={shift_streak_ratio:.2f}, "
            f"dir disagreement={angle_disagreement:.1f}°"
        )
        if abs(shift_streak_ratio - 1.0) > 0.25 or angle_disagreement > 15.0:
            logger.warning(
                f"Rate cross-check off on frame {rate_frame.index}: shift-rate is "
                f"{shift_streak_ratio:.2f}x the streak-rate, streak rot "
                f"{streak.rotation:.1f}° vs shift dir "
                f"{normalize_angle(rotation_estimate_frame_to_frame):.1f}° "
                f"({angle_disagreement:.1f}° off) — bad extraction/shift, or the "
                f"shift-rate baseline is wrong (watch for this firing on EVERY frame)"
            )

    rate_frame.streak = StreakMetadata(
        pixel_length=streak.length,
        sine_angle=np.sin(np.deg2rad(streak.rotation)),
        cosine_angle=np.cos(np.deg2rad(streak.rotation)),
        fwhm=streak.fwhm,
    )

    frame_shift.x_shift = best_shift[0]
    frame_shift.y_shift = best_shift[1]
    frame_shift.is_valid = True
    frame_shift.processed = True
    frame_shift.error_message = None

    # Model-free rate from the directly-measured streak length (not the suspect
    # shift formula). Magnitude from the streak, direction from the validated shift.
    rate_frame.pixel_track_rate_per_second = rate_from_streak
    rate_frame.seeing = sidereal_frame.seeing
    return
