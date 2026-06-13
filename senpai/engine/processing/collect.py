"""SENPAI collect pipeline — multi-frame sidereal+rate processing."""

import logging
import time
from pathlib import Path

import numpy as np

from senpai.core.config import get_config
from senpai.engine.detection.jacobian import wcs_distortion_metrics
from senpai.engine.detection.point.satellite import extract_point_sources
from senpai.engine.detection.streak.frame_shift import solve_shift
from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.models.metadata import SeeingModel, TrackMode
from senpai.engine.models.senpai import RateTrackFrame, SenpaiRun, SiderealFrame
from senpai.engine.plotting.images import plot_single_frame
from senpai.engine.processing.sidereal import process_astrometry_fits_sidereal
from senpai.engine.utils.preprocessing import (
    scale_starfield_coordinates,
)
from senpai.engine.utils.propagate_wcs import (
    refine_sidereal_frame,
    refine_wcs_by_kernel_convolution,
    shift_wcs_by_pixel_shift,
)

logger = logging.getLogger(__name__)


def process_senpai_collect(
    file_list: list[ProcessedFitsImage],
    id: str = "senpai",
    force_track_mode: TrackMode | None = None,
) -> SenpaiRun:
    t_start = time.time()
    config = get_config()

    # Apply preprocessing to all frames before organizing
    from senpai.engine.utils.preprocessing import preprocess_image

    logger.info("Applying preprocessing to all frames...")
    for frame in file_list:
        preprocess_image(frame, config, store_intermediates=False)

        # Save the processed frame data for later export (replot reads these;
        # full-night runs skip them — ~260 MB/frame dominates the output dir).
        if (config.runtime.save_processed_fits
                and hasattr(frame, "file_path") and frame.file_path):
            # Create processed filename
            processed_path = Path(frame.file_path)
            processed_filename = (
                f"{processed_path.stem}_processed{processed_path.suffix}"
            )
            processed_file_path = config.runtime.output_dir / processed_filename

            # Save processed FITS file
            from astropy.io import fits

            hdu = fits.PrimaryHDU(frame.data, frame.header)
            hdu.writeto(processed_file_path, overwrite=True)

            # Store the processed file path in the frame for later access
            frame.processed_file_path = str(processed_file_path)
            logger.debug(f"Saved processed frame: {processed_file_path}")

    senpai_run = SenpaiRun.organize_senpai_frames(
        file_list, id=id, force_track_mode=force_track_mode
    )

    valid_sidereal_frame = False
    for image_frame in senpai_run.sidereal_frames:
        sidereal_wcs_starfield = process_astrometry_fits_sidereal(image_frame.frame)

        # Stop processing once we have a valid solution
        if sidereal_wcs_starfield.fit:
            logger.info(
                f"Found valid WCS solution in frame {image_frame.index}, initial sidereal processing complete"
            )
            valid_sidereal_frame = True
            image_frame.seeing = SeeingModel.from_fwhm_stats(
                sidereal_wcs_starfield.fwhm_stats
            )
            # Propagate seeing to all sidereal frames (they share the same optics)
            for other_frame in senpai_run.sidereal_frames:
                if other_frame.seeing is None:
                    other_frame.seeing = image_frame.seeing
            # Apply FWHM optimization if enabled
            if config.calibrations.auto_scale_images:
                # Get FWHM stats from the successfully solved sidereal frame
                fwhm_stats = sidereal_wcs_starfield.fwhm_stats
                if (
                    fwhm_stats
                    and fwhm_stats.recommended_scale_factor
                    and fwhm_stats.recommended_scale_factor > 1.0
                ):
                    scale_factor = fwhm_stats.recommended_scale_factor
                    logger.info(
                        f"Scaling all frames using FWHM stats from frame {image_frame.index}"
                    )
                    logger.info(
                        f"FWHM: {fwhm_stats.median_fwhm:.1f} -> {config.calibrations.target_fwhm:.1f} pixels (factor: {scale_factor:.2f})"
                    )
                    logger.info(
                        f"Method: {config.calibrations.scaling_method}, {fwhm_stats.n_measurements} FWHM measurements"
                    )

                    # Scale all sidereal frames
                    for frame in senpai_run.sidereal_frames:
                        frame.frame.scale_frame(
                            scale_factor, method=config.calibrations.scaling_method
                        )
                        if frame.starfield:
                            frame.starfield = scale_starfield_coordinates(
                                frame.starfield, scale_factor
                            )

                    # Scale all rate track frames
                    for frame in senpai_run.rate_track_frames:
                        frame.frame.scale_frame(
                            scale_factor, method=config.calibrations.scaling_method
                        )
                        if frame.starfield:
                            frame.starfield = scale_starfield_coordinates(
                                frame.starfield, scale_factor
                            )

                    # Get the actual scale factor used from the processing history
                    # For median_filter, this will be the rounded integer value
                    actual_scale_factor = scale_factor
                    if config.calibrations.scaling_method == "median_filter":
                        # Get the actual integer scale factor from the first frame's processing history
                        for frame in (
                            senpai_run.sidereal_frames + senpai_run.rate_track_frames
                        ):
                            if frame.frame.processing_history:
                                for step in reversed(frame.frame.processing_history):
                                    if step.step_type.value == "fwhm_optimization":
                                        actual_scale_factor = step.parameters.get(
                                            "scale_factor", scale_factor
                                        )
                                        break
                                if actual_scale_factor != scale_factor:
                                    break

                    # Store the actual scale_factor at the run level
                    senpai_run.scale_factor = actual_scale_factor
                    logger.info(
                        f"Stored actual scale_factor {actual_scale_factor} at run level (original recommended: {scale_factor:.2f})"
                    )

                    logger.info("All frames scaled successfully")

                    # Re-run astrometry on the scaled frame
                    sidereal_wcs_starfield = process_astrometry_fits_sidereal(
                        image_frame.frame
                    )
                    # The scale_factor is now stored at the run level, so we don't need to preserve it in individual starfields
                    image_frame.starfield = sidereal_wcs_starfield
                else:
                    image_frame.starfield = sidereal_wcs_starfield
            else:
                image_frame.starfield = sidereal_wcs_starfield

            # Once we have a valid sidereal WCS, measure field distortion via local Jacobians.
            # This is used later to decide whether variable kernels are needed for rate-track frames.
            try:
                if image_frame.starfield and image_frame.starfield.wcs is not None:
                    astropy_wcs = image_frame.starfield.wcs.to_astropy_wcs()
                    if astropy_wcs is not None:
                        # Ensure array_shape is set so jacobian sampling covers the full detector
                        height, width = image_frame.frame.data.shape
                        astropy_wcs.array_shape = (height, width)

                        # Use a nominal unit rate vector in RA; metrics are relative so the
                        # specific rate magnitude is not critical for distortion assessment.
                        metrics = wcs_distortion_metrics(
                            astropy_wcs, rate_ra=1.0, rate_dec=0.0, nx=5, ny=5
                        )

                        # Keep only compact scalar metrics on the starfield to avoid bloating results
                        distortion_summary = {
                            "delta_J": float(metrics["delta_J"]),
                            "max_angle_variation_deg": float(
                                metrics["max_angle_variation_deg"]
                            ),
                            "max_length_variation_fraction": float(
                                metrics["max_length_variation_fraction"]
                            ),
                        }
                        image_frame.starfield.distortion_metrics = distortion_summary

                        logger.info(
                            "Sidereal WCS distortion (frame %d): "
                            "delta_J=%.3g, max_angle_variation_deg=%.3f, max_length_variation_fraction=%.3f",
                            image_frame.index,
                            distortion_summary["delta_J"],
                            distortion_summary["max_angle_variation_deg"],
                            distortion_summary["max_length_variation_fraction"],
                        )
            except Exception as e:
                logger.warning(
                    "Failed to compute WCS distortion metrics for sidereal frame %d: %s",
                    image_frame.index,
                    e,
                )

            if config.plotting.review and config.plotting.debug:
                # otherwise this'll be plotted at the end (review True, debug False)
                target = senpai_run.get_frame_by_index(image_frame.index)

                plot_single_frame(
                    target.frame.data,
                    starfield=target.starfield,
                    detections=(
                        target.detections if isinstance(target, SiderealFrame) else None
                    ),
                    output_file=config.runtime.output_dir / f"final_{target.index}.png",
                )

            break

    if not valid_sidereal_frame and senpai_run.rate_track_frames:
        # Rate-only input — try WCS from first rate frame's streak centroids
        from senpai.astrometry import solve_field
        from senpai.catalog.runner import query_catalog
        from senpai.engine.detection.streak.rate_extraction import (
            build_streak_metadata,
            extract_rate_streak_measurement,
            extract_streak_centers_as_sources,
        )
        from senpai.engine.models.metadata import (
            DetectionMetadata,
            ImageMetadata,
        )
        from senpai.engine.models.starfield import StarListImage
        from senpai.engine.utils.fits_io import extract_boresight_from_header

        for image_frame in senpai_run.rate_track_frames:
            measurement, _psf, measured_fwhm = extract_rate_streak_measurement(
                image_frame, n_streaks=10, initial_fwhm=None
            )
            if measurement is None:
                continue

            if measurement.fwhm is None and measured_fwhm is not None:
                measurement.fwhm = float(measured_fwhm)
            if measurement.fwhm is None:
                measurement.fwhm = 4.0

            image_frame.streak = build_streak_metadata(measurement)
            image_frame.seeing = SeeingModel(
                pixel_fwhm=float(image_frame.streak.fwhm),
                pixel_fwhm_stdev=0.0,
                n_measurements=1,
            )

            sources = extract_streak_centers_as_sources(
                image_frame.frame.data,
                streak=image_frame.streak,
                max_sources=200,
            )
            if not sources:
                continue

            # Diagnostic overlay: show streak centroids handed to the solver,
            # so a WCS failure still produces a visible debug artifact.
            if (config.plotting.debug or config.plotting.review) and sources:
                sources_for_plot = StarListImage(
                    detections=sources,
                    image_metadata=ImageMetadata(
                        width=image_frame.frame.data.shape[1],
                        height=image_frame.frame.data.shape[0],
                    ),
                )
                plot_single_frame(
                    image_frame.frame.data,
                    starlist=sources_for_plot,
                    streak=image_frame.streak,
                    markersize=(image_frame.streak.fwhm * 2 if image_frame.streak else 10),
                    output_file=config.runtime.output_dir / f"rate_detections_{image_frame.index}.png",
                )

            boresight_ra, boresight_dec = extract_boresight_from_header(image_frame.frame.header)
            frame_meta = image_frame.frame_metadata
            img_meta = ImageMetadata(
                width=image_frame.frame.data.shape[1],
                height=image_frame.frame.data.shape[0],
                boresight_ra=boresight_ra,
                boresight_dec=boresight_dec,
                exposure_time=(
                    float(frame_meta.exposure_time_seconds)
                    if frame_meta and frame_meta.exposure_time_seconds
                    else None
                ),
            )
            starlist = StarListImage(detections=sources, image_metadata=img_meta)

            try:
                wcs_starfield = solve_field(starlist)
            except Exception as e:
                logger.warning("WCS solve failed for rate frame %d: %s", image_frame.index, e)
                continue

            if wcs_starfield and wcs_starfield.wcs:
                try:
                    catalog = query_catalog(wcs_starfield.wcs, max_stars=None, apply_sip=True)
                    wcs_starfield.catalog_stars = catalog.stars
                    base_md = wcs_starfield.image_metadata.model_dump()
                    for k, v in catalog.image_metadata.model_dump().items():
                        if v is not None:
                            base_md[k] = v
                    wcs_starfield.image_metadata = ImageMetadata(**base_md)
                except Exception as e:
                    logger.warning("Catalog query failed for rate frame %d: %s", image_frame.index, e)

                wcs_starfield.detection_metadata = DetectionMetadata(
                    pixel_fwhm=float(image_frame.streak.fwhm)
                )
                image_frame.starfield = wcs_starfield

                # Track rate in pixels/s
                exp = frame_meta.exposure_time_seconds if frame_meta else None
                if exp and exp > 0:
                    image_frame.pixel_track_rate_per_second = (
                        float(image_frame.streak.pixel_length) / float(exp)
                    )

                valid_sidereal_frame = True
                logger.info(
                    "Rate-only mode: WCS solved from rate frame %d streak centroids",
                    image_frame.index,
                )
                break

    if not valid_sidereal_frame:
        senpai_run.error_message = "No valid WCS solution found"
        logger.warning(senpai_run.error_message)
        return senpai_run

    # ok, find a valid path through all frames creating / using frame shifts
    senpai_run.create_valid_path()

    next_shift = senpai_run.get_next_shift()
    while next_shift is not None:
        if config.plotting.debug:
            plot_single_frame(
                senpai_run.get_frame_by_index(next_shift.target_index).frame.data,
                output_file=config.runtime.output_dir
                / f"{next_shift.target_index}_raw.png",
            )

        solve_shift(senpai_run, next_shift)

        # Backstop against a livelock: the loop pulls the next *unprocessed*
        # shift, so a solver that returns without setting processed=True hands
        # the same shift back forever (a missing-starfield early-return did
        # exactly this and spun 8 workers at 100% CPU for hours). A solver
        # must always mark the shift processed; if one didn't, retire it as
        # failed here so the loop can make progress.
        if not next_shift.processed:
            logger.error(
                "Shift %d->%d returned unprocessed from solver; force-retiring "
                "as failed to avoid livelock.",
                next_shift.source_index, next_shift.target_index,
            )
            next_shift.processed = True
            next_shift.is_valid = False
            next_shift.error_message = (
                next_shift.error_message or "Solver returned without processing"
            )

        senpai_run.update_valid_path()

        logger.info("Shifting WCS by pixel shift")

        if next_shift.is_valid and next_shift.processed:
            shift_wcs_by_pixel_shift(senpai_run, next_shift)

            target = senpai_run.get_frame_by_index(next_shift.target_index)

            if isinstance(target, SiderealFrame):
                logger.info("Refining WCS for sidereal frame %d", target.index)
                refine_sidereal_frame(target)

            elif isinstance(target, RateTrackFrame):
                logger.info("Refining WCS by kernel convolution")
                shift_correction_x, shift_correction_y = (
                    refine_wcs_by_kernel_convolution(target)
                )

                # Apply the correction to the existing shift
                original_x = next_shift.x_shift
                original_y = next_shift.y_shift
                next_shift.x_shift -= shift_correction_x
                next_shift.y_shift -= shift_correction_y
                logger.info(
                    f"Applied WCS refinement correction to shift {next_shift.source_index}->{next_shift.target_index}: "
                    f"({original_x:.2f}, {original_y:.2f}) + ({shift_correction_x:.2f}, {shift_correction_y:.2f}) "
                    f"= ({next_shift.x_shift:.2f}, {next_shift.y_shift:.2f})"
                )

                # Recalculate pixel_track_rate_per_second based on refined shift
                # This is critical for the next frame pair's validation attempt
                source = senpai_run.get_frame_by_index(next_shift.source_index)
                if isinstance(source, RateTrackFrame):
                    frame_gap_seconds = abs(
                        (target.timestamp - source.timestamp).total_seconds()
                    )
                    refined_shift_magnitude = np.sqrt(
                        next_shift.x_shift**2 + next_shift.y_shift**2
                    )
                    old_rate = target.pixel_track_rate_per_second
                    target.pixel_track_rate_per_second = (
                        refined_shift_magnitude / frame_gap_seconds
                    )
                    logger.info(
                        f"Updated pixel_track_rate_per_second for frame {target.index}: "
                        f"{old_rate:.3f} -> {target.pixel_track_rate_per_second:.3f} px/s "
                        f"(shift_mag={refined_shift_magnitude:.2f}px, gap={frame_gap_seconds:.2f}s)"
                    )
                if config.detection.detect:
                    target.detections = extract_point_sources(target)

            if config.plotting.review and config.plotting.debug:
                # otherwise this'll be plotted at the end (review True, debug False)
                plot_single_frame(
                    target.frame.data,
                    starfield=target.starfield,
                    detections=(
                        target.detections
                        if isinstance(target, RateTrackFrame)
                        else None
                    ),
                    streak=(
                        target.streak if isinstance(target, RateTrackFrame) else None
                    ),
                    output_file=config.runtime.output_dir / f"final_{target.index}.png",
                )

        senpai_run.log_analysis_chain()

        next_shift = senpai_run.get_next_shift()

    # --- Point source detection for rate-track frames that weren't shift targets ---
    if config.detection.detect:
        for image_frame in senpai_run.rate_track_frames:
            if image_frame.detections is not None:
                continue
            if image_frame.starfield is None or not image_frame.starfield.fit:
                continue
            image_frame.detections = extract_point_sources(image_frame)

    # --- Photometry ---
    from dataclasses import asdict

    from senpai.engine.photometry.utils import (
        measure_detection_photometry,
        measure_rate_starfield_photometry,
        measure_simple_starfield_photometry,
    )

    # Sidereal frames: simple circular aperture photometry
    for image_frame in senpai_run.sidereal_frames:
        if image_frame.starfield is None or not image_frame.starfield.fit:
            continue
        try:
            _, summary = measure_simple_starfield_photometry(
                image_frame.frame, image_frame.starfield, config.photometry,
                frame_index=image_frame.index,
            )
            image_frame.photometry_summary = asdict(summary)
            if summary.limiting_magnitude_50 is not None:
                image_frame.starfield.limiting_magnitude = summary.limiting_magnitude_50
            elif summary.limiting_magnitude:
                image_frame.starfield.limiting_magnitude = summary.limiting_magnitude
            logger.info(
                f"Sidereal frame {image_frame.index}: photometry ZP={summary.zero_point}, "
                f"limiting_mag={image_frame.starfield.limiting_magnitude}"
            )
        except Exception as e:
            logger.warning(f"Photometry failed for sidereal frame {image_frame.index}: {e}")

    # --- Catalog-filtered point source detections in sidereal frames ---
    # Match starfield.detections against catalog_stars; non-matched sources that
    # are bright enough to reliably be in the catalog are potential satellites/asteroids.
    from senpai.engine.models.starfield import SatelliteInImage, SatelliteListImage

    for image_frame in senpai_run.sidereal_frames:
        if image_frame.starfield is None or not image_frame.starfield.fit:
            continue
        if not image_frame.starfield.catalog_stars or not image_frame.starfield.detections:
            continue

        fwhm = 4.0
        if image_frame.starfield.detection_metadata and image_frame.starfield.detection_metadata.pixel_fwhm:
            fwhm = image_frame.starfield.detection_metadata.pixel_fwhm

        match_radius_sq = (2 * fwhm) ** 2

        catalog_positions = [
            (s.x, s.y)
            for s in image_frame.starfield.catalog_stars
            if s.x is not None and s.y is not None
        ]
        if not catalog_positions:
            continue

        catalog_xy = np.array(catalog_positions)

        # Separate matched vs unmatched detections and compute a brightness
        # threshold.  Only flag unmatched detections that are at least as bright
        # as the 25th-percentile of matched (catalog-confirmed) detections.
        # Fainter unmatched sources are overwhelmingly noise peaks.
        matched_counts = []
        unmatched = []
        for det in image_frame.starfield.detections:
            if det.x is None or det.y is None:
                continue
            dists_sq = (catalog_xy[:, 0] - det.x) ** 2 + (catalog_xy[:, 1] - det.y) ** 2
            if np.min(dists_sq) <= match_radius_sq:
                if det.counts is not None:
                    matched_counts.append(det.counts)
            else:
                unmatched.append(det)

        if not unmatched or not matched_counts:
            continue

        min_counts = float(np.percentile(matched_counts, 25))

        non_catalog = [
            det for det in unmatched
            if det.counts is not None and det.counts >= min_counts
        ]

        if non_catalog:
            satellites = []
            for det in non_catalog:
                ra_val, dec_val = None, None
                if image_frame.starfield.wcs:
                    try:
                        wcs = image_frame.starfield.wcs.to_astropy_wcs()
                        sky = wcs.pixel_to_world(det.x, det.y)
                        ra_val = float(sky.ra.deg)
                        dec_val = float(sky.dec.deg)
                    except Exception:
                        pass
                satellites.append(
                    SatelliteInImage(
                        x=det.x, y=det.y, snr=det.snr,
                        ra=ra_val, dec=dec_val,
                        pixel_fwhm=fwhm, detection_type="point",
                    )
                )

            img_meta = image_frame.starfield.image_metadata
            if image_frame.detections is None:
                image_frame.detections = SatelliteListImage(
                    detections=satellites, image_metadata=img_meta,
                )
            else:
                image_frame.detections.detections.extend(satellites)

        logger.info(
            "Sidereal frame %d: %d non-catalog point detections (%d unmatched, %d below brightness threshold, counts_thresh=%.0f)",
            image_frame.index, len(non_catalog), len(unmatched),
            len(unmatched) - len(non_catalog), min_counts,
        )

    # Rate-track frames: rectangular aperture photometry + detection photometry
    for image_frame in senpai_run.rate_track_frames:
        if image_frame.starfield is None or not image_frame.starfield.fit:
            continue
        if image_frame.streak is None:
            continue
        try:
            _, summary = measure_rate_starfield_photometry(
                image_frame.frame, image_frame.starfield, image_frame.streak,
                config.photometry, frame_index=image_frame.index,
            )
            image_frame.photometry_summary = asdict(summary)
            if summary.limiting_magnitude_50 is not None:
                image_frame.starfield.limiting_magnitude = summary.limiting_magnitude_50
            elif summary.limiting_magnitude:
                image_frame.starfield.limiting_magnitude = summary.limiting_magnitude
            logger.info(
                f"Rate frame {image_frame.index}: photometry ZP={summary.zero_point}, "
                f"limiting_mag={image_frame.starfield.limiting_magnitude}"
            )

            # Detection photometry if we have detections and a valid zero point
            if (
                image_frame.detections
                and image_frame.detections.detections
                and summary.zero_point is not None
            ):
                try:
                    exp_time = (
                        image_frame.frame_metadata.exposure_time_seconds
                        if image_frame.frame_metadata
                        else None
                    )
                    measure_detection_photometry(
                        image_frame.frame,
                        image_frame.detections,
                        summary.zero_point,
                        summary.zero_point_err,
                        exposure_time=exp_time,
                        config=config.photometry,
                        multiband_calibration=summary.multiband_calibration,
                        observation_filter=(
                            image_frame.frame_metadata.observation_filter
                            if image_frame.frame_metadata
                            else None
                        ),
                    )
                except Exception as e:
                    logger.warning(
                        f"Detection photometry failed for rate frame {image_frame.index}: {e}"
                    )
        except Exception as e:
            logger.warning(f"Photometry failed for rate frame {image_frame.index}: {e}")

    # --- Streak detection & cross-frame correlation ---
    if config.detection.detect and config.detection.detect_streaks:
        from senpai.engine.processing.rate_scan_confirmation import (
            confirm_streaks_via_rate_scan,
        )
        from senpai.engine.processing.streak_correlation import (
            correlate_rate_to_sidereal,
            detect_streaks_in_rate_frames,
            detect_streaks_in_sidereal_frames,
        )

        de_data = detect_streaks_in_sidereal_frames(senpai_run)
        detect_streaks_in_rate_frames(senpai_run)
        senpai_run.correlated_streaks = confirm_streaks_via_rate_scan(senpai_run, de_data)
        if senpai_run.rate_track_frames and senpai_run.sidereal_frames:
            correlate_rate_to_sidereal(senpai_run)

        # Clear unconfirmed rate-frame streak candidates — they are single-frame
        # detections that didn't pass multi-frame confirmation and would show
        # as false positives in annotations/output.
        for frame in senpai_run.rate_track_frames:
            frame.streak_candidates = []

        # Deduplicate: remove point-type detections that overlap with streak detections
        for frame in senpai_run.sidereal_frames + senpai_run.rate_track_frames:
            if frame.detections is None:
                continue
            streak_dets = [
                d for d in frame.detections.detections
                if getattr(d, "detection_type", None) == "streak"
            ]
            if not streak_dets:
                continue
            fwhm = 4.0
            if frame.starfield and frame.starfield.detection_metadata and frame.starfield.detection_metadata.pixel_fwhm:
                fwhm = frame.starfield.detection_metadata.pixel_fwhm
            radius_sq = (2 * fwhm) ** 2
            cleaned = []
            for d in frame.detections.detections:
                if getattr(d, "detection_type", None) == "point":
                    near_streak = any(
                        (d.x - s.x) ** 2 + (d.y - s.y) ** 2 < radius_sq
                        for s in streak_dets
                    )
                    if near_streak:
                        continue
                cleaned.append(d)
            n_removed = len(frame.detections.detections) - len(cleaned)
            if n_removed:
                frame.detections.detections = cleaned
                logger.info(
                    "Frame %d: removed %d point detections overlapping streak detections",
                    frame.index, n_removed,
                )

    t_end = time.time()
    senpai_run.completed = True
    senpai_run.error_message = None
    senpai_run.compute_seconds = round(t_end - t_start, 2)
    logger.info(f"Time taken to process set: {senpai_run.compute_seconds} seconds")

    return senpai_run


def _write_sequence_gif(image_paths: list, gif_path) -> None:
    """Write a per-frame animation, padding frames to a common shape first.

    Mixed sidereal/rate batches render their ``final_*`` plots at different pixel
    sizes (different overlays/colorbars), so a naive ``np.stack`` of the frames
    raises "all input arrays must have the same shape". We pad each frame to the
    max height/width before stacking. The GIF is a diagnostic nicety, so any
    failure is logged and swallowed rather than failing the batch."""

    try:
        import imageio.v3 as iio
        import numpy as np

        images = [iio.imread(str(f)) for f in image_paths]
        if not images:
            return
        h = max(im.shape[0] for im in images)
        w = max(im.shape[1] for im in images)
        padded = []
        for im in images:
            pad = [(0, h - im.shape[0]), (0, w - im.shape[1])]
            pad += [(0, 0)] * (im.ndim - 2)
            padded.append(np.pad(im, pad, mode="constant"))
        iio.imwrite(gif_path, padded, duration=400, loop=0)
        logger.info(f"Created animation at {gif_path}")
    except Exception as e:
        logger.warning(f"Skipping animation {gif_path}: {e}")


def final_plots(senpai_run: SenpaiRun, output_dir: Path):
    config = get_config()

    run_id = config.runtime.run_id

    # Per-frame empirical PSF panels (stacked stars for sidereal, stacked streak
    # for rate). A small .npy stamp is saved next to each PNG so the panel can be
    # regenerated later (see engine.plotting.replot) without the raw FITS.
    if config.plotting.psfs:
        from senpai.engine.plotting.psf import plot_rate_frame, plot_sidereal_frame

        for f in senpai_run.sidereal_frames:
            png = output_dir / f"frame_{f.index}_psf.png"
            if not png.exists():
                try:
                    plot_sidereal_frame(f, png, output_dir / f"frame_{f.index}_psf.npy")
                except Exception as e:
                    logger.warning("PSF panel failed for sidereal frame %s: %s",
                                   f.index, e)
        for f in senpai_run.rate_track_frames:
            png = output_dir / f"frame_{f.index}_streak.png"
            if not png.exists():
                try:
                    plot_rate_frame(f, png, output_dir / f"frame_{f.index}_streak.npy")
                except Exception as e:
                    logger.warning("PSF panel failed for rate frame %s: %s",
                                   f.index, e)

    for image_frame in senpai_run.sidereal_frames:
        output_file = output_dir / f"final_{image_frame.index}.png"
        if config.plotting.review and not output_file.exists():
            plot_single_frame(
                image_frame.frame.data,
                starfield=image_frame.starfield,
                detections=image_frame.detections,
                streak_candidates=image_frame.streak_candidates or None,
                output_file=output_file,
            )
        output_file = output_dir / f"raw_{image_frame.index}.png"
        if config.plotting.review and not output_file.exists():
            plot_single_frame(
                image_frame.frame.data,
                output_file=output_file,
            )

    for image_frame in senpai_run.rate_track_frames:
        output_file = output_dir / f"final_{image_frame.index}.png"
        if config.plotting.review and not output_file.exists():
            plot_single_frame(
                image_frame.frame.data,
                starfield=image_frame.starfield,
                streak=image_frame.streak,
                detections=image_frame.detections,
                streak_candidates=image_frame.streak_candidates or None,
                output_file=output_file,
            )

        output_file = output_dir / f"raw_{image_frame.index}.png"
        if config.plotting.review and not output_file.exists():
            plot_single_frame(
                image_frame.frame.data,
                output_file=output_file,
            )

    if config.plotting.review:
        # Collect all plot filenames and sort by frame index
        plot_files = []
        plot_rate_files = []
        plot_raw_files = []
        for frame in sorted(
            senpai_run.sidereal_frames + senpai_run.rate_track_frames,
            key=lambda x: x.index,
        ):
            plot_file = output_dir / f"final_{frame.index}.png"
            if plot_file.exists():
                plot_files.append(plot_file)

                if isinstance(frame, RateTrackFrame):
                    plot_rate_files.append(plot_file)

            plot_file = output_dir / f"raw_{frame.index}.png"
            if plot_file.exists():
                plot_raw_files.append(plot_file)

        if plot_files:
            _write_sequence_gif(plot_files, output_dir / f"{run_id}_sequence.gif")
        if plot_rate_files:
            _write_sequence_gif(plot_rate_files, output_dir / f"{run_id}_sequence_rate.gif")
        if plot_raw_files:
            _write_sequence_gif(plot_raw_files, output_dir / f"{run_id}_sequence_raw.gif")
