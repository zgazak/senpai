"""COCO format export functionality for SENPAI data."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import png
from astropy.io import fits

from senpai.engine.models.senpai import SenpaiRun, SenpaiRunResult
from senpai.engine.utils.file_io import load_fits_file

logger = logging.getLogger(__name__)


class SenpaiCocoExporter:
    """Export SENPAI runs to individual COCO format files per image."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        write_png: bool = False,
        write_fits: bool = True,
        save_annotated_images: bool = False,
        remove_median: bool = False,
        snr_cut: float = 0.5,
        box_size: int = 4,
        streak_box_size: int = 10,
        mask_radius: Optional[float] = None,
        max_streak_length: Optional[float] = None,
        process_sidereal: bool = True,  # Add parameter to control sidereal processing
    ):
        """Initialize the COCO exporter.

        Args:
            output_dir: Directory to save COCO files
            write_png: Whether to save PNG images
            write_fits: Whether to save FITS images
            save_annotated_images: Whether to save annotated images
            remove_median: Whether to remove median from images
            snr_cut: Minimum SNR for annotations
            box_size: Size of bounding boxes for point sources
            streak_box_size: Size of bounding boxes for satellites
            mask_radius: Radius to mask around center (pixels)
            max_streak_length: Maximum streak length to include
            process_sidereal: Whether to process sidereal frames
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.write_png = write_png
        self.write_fits = write_fits
        self.save_annotated_images = save_annotated_images
        self.remove_median = remove_median
        self.snr_cut = snr_cut
        self.box_size = box_size
        self.streak_box_size = streak_box_size
        self.mask_radius = mask_radius
        self.max_streak_length = max_streak_length
        self.process_sidereal = process_sidereal  # Store the parameter

    def export_senpai_run(
        self,
        senpai_run: Union[SenpaiRun, SenpaiRunResult],
        collect_id: str,
        apply_calibrations: bool = True,
        source_path=None,
    ) -> None:
        """Export a single SENPAI run to individual COCO format files.

        ``source_path`` is the run's JSON path; its sibling ``config.yaml`` is
        used to rebuild processed frames in place when *_processed.fits is
        absent (runs made with --no-processed-fits).
        """
        # Store reference to senpai_run for scale_factor access
        self.senpai_run = senpai_run
        self._source_path = source_path
        self._build_cfg = "unset"  # reset per-run config cache

        # Process frames based on run type
        if isinstance(senpai_run, SenpaiRunResult):
            # Process serializable frames
            if self.process_sidereal:
                for frame in senpai_run.sidereal_frames:
                    self._process_sidereal_frame_serializable(frame, collect_id, apply_calibrations)

            for frame in senpai_run.rate_track_frames:
                self._process_rate_frame_serializable(frame, collect_id, apply_calibrations)
        else:
            # Process live frames
            if self.process_sidereal:
                for frame in senpai_run.sidereal_frames:
                    self._process_sidereal_frame(frame, collect_id, apply_calibrations)

            for frame in senpai_run.rate_track_frames:
                self._process_rate_frame(frame, collect_id, apply_calibrations)

    def _build_config(self):
        """Config used to rebuild missing processed frames, loaded once from the
        run's own ``config.yaml`` (next to the source JSON) so the in-place
        preprocessing matches exactly what the night run did (same flat dir,
        row/col-median settings, etc.). Returns an AppConfig or None if it can't
        be resolved (then callers fall back to the raw, uncalibrated frame)."""
        if getattr(self, "_build_cfg", "unset") != "unset":
            return self._build_cfg
        cfg = None
        src = getattr(self, "_source_path", None)
        try:
            from senpai.core.config import AppConfig, load_yaml

            if src:
                cfg_path = Path(src).parent / "config.yaml"
                if cfg_path.is_file():
                    cfg = AppConfig(**load_yaml(cfg_path))  # load_yaml unwraps "app"
            if cfg is None:
                from senpai.core.config import get_config
                cfg = get_config()  # raises if no run config + none initialized
        except Exception as e:
            logger.warning(
                "No config for in-place processed-frame rebuild (%s); COCO will "
                "use raw uncalibrated frames where *_processed.fits is missing", e)
            cfg = None
        self._build_cfg = cfg
        return cfg

    def _load_frame_image(self, frame, apply_calibrations: bool, label: str):
        """(data, header, file_path) for a serialized frame.

        Prefers the written ``*_processed.fits``. When it's absent — e.g. a run
        made with ``--no-processed-fits`` — rebuild the processed frame IN PLACE
        from the raw original via the configured preprocessing pipeline (the same
        steps the night run applied), so COCO trains on calibrated frames instead
        of silently falling back to raw. Returns (None, None, None) if no source.
        """
        if frame.processed_frame_path and os.path.exists(frame.processed_frame_path):
            img = load_fits_file(frame.processed_frame_path)
            return img.data, img.header, frame.processed_frame_path
        if frame.original_frame_path and os.path.exists(frame.original_frame_path):
            img = load_fits_file(frame.original_frame_path)
            if apply_calibrations:
                cfg = self._build_config()
                if cfg is not None:
                    from senpai.engine.utils.preprocessing import preprocess_image
                    try:
                        preprocess_image(img, cfg)
                        logger.info("Built processed frame in place from raw for "
                                    "%s (no *_processed.fits on disk)", label)
                    except Exception as e:
                        logger.warning("In-place preprocess failed for %s (%s); "
                                       "using raw frame", label, e)
            return img.data, img.header, frame.original_frame_path
        return None, None, None

    def _process_sidereal_frame_serializable(
        self,
        frame,
        collect_id: str,
        apply_calibrations: bool,
    ) -> None:
        """Process a serializable sidereal frame for COCO export."""
        logger.info(f"Processing serializable sidereal frame {frame.index}")
        
        # Check if frame has WCS (starfield with fit=True)
        if not frame.starfield or not frame.starfield.fit:
            logger.info(f"Skipping sidereal frame {frame.index} - no valid WCS solution")
            return

        # Load frame data - prefer processed FITS files if available
        frame_data, header, file_path = self._load_frame_image(
            frame, apply_calibrations, f"sidereal frame {frame.index}")
        if frame_data is None:
            logger.warning(f"No frame data available for sidereal frame {frame.index}")
            return

        # Create image entry
        image_id = f"{collect_id}_sidereal_{frame.index}"

        # Save image files
        output_image_file = self._save_image_files(frame_data, header, image_id, file_path, frame.starfield)

        # Create image metadata
        image_metadata = self._create_image_metadata(
            frame_data, header, image_id, collect_id, "sidereal", output_image_file
        )

        # Get scale factor from run level or starfield if available
        scale_factor = self._get_scale_factor(frame.starfield, self.senpai_run)

        # Create point source annotations (stars)
        point_annotations = []
        star_centers = []

        if frame.starfield:
            # Use catalog stars if available (they have magnitudes)
            if frame.starfield.catalog_stars:
                for star in frame.starfield.catalog_stars:
                    # Scale coordinates back up if needed
                    x, y = self._scale_coordinates(star.x, star.y, scale_factor)
                    # Estimate SNR from magnitude (brighter stars = higher SNR)
                    estimated_snr = self._estimate_snr_from_magnitude(star.magnitude) if star.magnitude else 5.0
                    if estimated_snr >= self.snr_cut:
                        annotation = self._create_point_annotation(
                            x, y, image_id, self.box_size, estimated_snr, star.magnitude, len(point_annotations)
                        )
                        if annotation:
                            point_annotations.append(annotation)
                            star_centers.append([x, y])

            # Also use detections if available (they have counts)
            if frame.starfield.detections:
                for star in frame.starfield.detections:
                    # Scale coordinates back up if needed
                    x, y = self._scale_coordinates(star.x, star.y, scale_factor)
                    # Estimate SNR from counts (more counts = higher SNR)
                    estimated_snr = self._estimate_snr_from_counts(star.counts) if star.counts else 5.0
                    if estimated_snr >= self.snr_cut:
                        annotation = self._create_point_annotation(
                            x, y, image_id, self.box_size, estimated_snr, None, len(point_annotations)
                        )
                        if annotation:
                            point_annotations.append(annotation)
                            star_centers.append([x, y])

        # Create COCO datasets
        point_dataset = self._create_coco_dataset([image_metadata], point_annotations, "sidereal_star", "point_source")

        # Save individual COCO files
        self._save_coco_files(image_id, point_dataset, None, star_centers, frame_data, None, frame.starfield, None)

    def _process_rate_frame_serializable(
        self,
        frame,
        collect_id: str,
        apply_calibrations: bool,
    ) -> None:
        """Process a serializable rate track frame for COCO export."""
        logger.info(f"Processing serializable rate frame {frame.index}")
        
        # Check if frame has WCS (starfield with fit=True)
        if not frame.starfield or not frame.starfield.fit:
            logger.info(f"Skipping rate frame {frame.index} - no valid WCS solution")
            return

        # Early check for max_streak_length - skip entire frame if streak exceeds limit
        if self.max_streak_length is not None and frame.streak:
            scale_factor = self._get_scale_factor(frame.starfield, self.senpai_run)
            scaled_length = frame.streak.pixel_length * scale_factor
            if scaled_length > self.max_streak_length:
                logger.info(f"Skipping rate frame {frame.index} - streak length {scaled_length:.1f} > {self.max_streak_length}")
                return

        # Load frame data - prefer the written *_processed.fits; rebuild in
        # place from raw when absent (see _load_frame_image).
        frame_data, header, file_path = self._load_frame_image(
            frame, apply_calibrations, f"rate frame {frame.index}")
        if frame_data is None:
            logger.warning(f"No frame data available for rate frame {frame.index}")
            return

        # Create image entry
        image_id = f"{collect_id}_rate_{frame.index}"

        # Save image files
        output_image_file = self._save_image_files(frame_data, header, image_id, file_path, frame.starfield)

        # Create image metadata
        image_metadata = self._create_image_metadata(
            frame_data, header, image_id, collect_id, "rate", output_image_file
        )

        # Get scale factor from run level or starfield if available
        scale_factor = self._get_scale_factor(frame.starfield, self.senpai_run)

        # Create satellite annotations (point sources)
        satellite_annotations = []
        satellite_centers = []

        if frame.detections and frame.detections.detections:
            logger.debug(f"Processing satellite annotations for rate frame {frame.index}")
            logger.debug(f"Found {len(frame.detections.detections)} satellite detections")

            for satellite in frame.detections.detections:
                # Scale coordinates back up if needed
                x, y = self._scale_coordinates(satellite.x, satellite.y, scale_factor)
                # Estimate SNR if it's null
                if satellite.snr is not None:
                    snr_value = satellite.snr
                else:
                    snr_value = 5.0  # Default SNR for satellites
                    logger.debug(f"Using default SNR {snr_value} for satellite at ({x}, {y})")

                annotation = self._create_point_annotation(
                    x, y, image_id, self.streak_box_size, snr_value, None, len(satellite_annotations)
                )
                if annotation:
                    satellite_annotations.append(annotation)
                    satellite_centers.append([x, y])
                    logger.debug(f"Created satellite annotation at ({x}, {y}) with SNR {snr_value:.2f}")
        else:
            logger.debug(f"No satellite detections for rate frame {frame.index}")

        # Create streak annotations (lines)
        streak_annotations = []
        streak_lines = []

        if frame.starfield and frame.streak:
            logger.debug(f"Processing streak annotations for rate frame {frame.index}")
            
            # Use catalog stars if available (they have magnitudes), otherwise fall back to detections
            stars_to_process = []
            if frame.starfield.catalog_stars:
                stars_to_process = frame.starfield.catalog_stars
                logger.debug(f"Using {len(stars_to_process)} catalog stars for streak annotations")
            elif frame.starfield.detections:
                stars_to_process = frame.starfield.detections
                logger.debug(f"Using {len(stars_to_process)} detected stars for streak annotations (no catalog stars available)")
            else:
                logger.debug("No stars available for streak annotations")

            for star in stars_to_process:
                # Scale coordinates back up if needed
                x, y = self._scale_coordinates(star.x, star.y, scale_factor)
                
                # Estimate SNR from magnitude if available, otherwise from counts
                if hasattr(star, "magnitude") and star.magnitude is not None:
                    snr_value = self._estimate_snr_from_magnitude(star.magnitude)
                    logger.debug(f"Using magnitude {star.magnitude:.2f} for SNR estimation at ({x}, {y})")
                elif star.snr is not None:
                    snr_value = star.snr
                    logger.debug(f"Using provided SNR {snr_value:.2f} for star at ({x}, {y})")
                elif hasattr(star, "counts") and star.counts is not None:
                    snr_value = self._estimate_snr_from_counts(star.counts)
                    logger.debug(f"Estimated SNR {snr_value:.2f} from counts {star.counts:.0f} for star at ({x}, {y})")
                else:
                    snr_value = 5.0  # Default SNR
                    logger.debug(f"Using default SNR {snr_value} for star at ({x}, {y})")

                if snr_value >= self.snr_cut:
                    # Create streak line annotation with scaled coordinates
                    line = self._create_streak_line_scaled(star, frame.streak, frame_data, scale_factor)
                    if line:
                        annotation = self._create_streak_annotation_scaled(
                            star, line, image_id, len(streak_annotations), scale_factor
                        )
                        if annotation:
                            streak_annotations.append(annotation)
                            streak_lines.append(line)
                            logger.debug(f"Created streak annotation for star at ({x}, {y}) with SNR {snr_value:.2f}")
                else:
                    logger.debug(f"Skipping star at ({x}, {y}) with SNR {snr_value:.2f} < {self.snr_cut}")
        else:
            logger.debug(
                f"No streak data for rate frame {frame.index}: starfield={frame.starfield is not None}, streak={frame.streak is not None}"
            )

        # Create COCO datasets
        satellite_dataset = self._create_coco_dataset(
            [image_metadata], satellite_annotations, "satellite", "point_source"
        )
        streak_dataset = self._create_coco_dataset([image_metadata], streak_annotations, "rate_star", "streak_source")

        # Save individual COCO files
        self._save_coco_files(
            image_id,
            satellite_dataset,
            streak_dataset,
            satellite_centers,
            frame_data,
            streak_lines,
            frame.starfield,
            frame.streak,
        )

    def _process_sidereal_frame(
        self,
        frame,
        collect_id: str,
        apply_calibrations: bool,
    ) -> None:
        """Process a live sidereal frame for COCO export."""
        # Check if frame has WCS (starfield with fit=True)
        if not hasattr(frame, "starfield") or not frame.starfield or not frame.starfield.fit:
            logger.info(f"Skipping live sidereal frame {frame.index} - no valid WCS solution")
            return

        # Get frame data
        if hasattr(frame, "frame") and frame.frame is not None:
            frame_data = frame.frame.data
            header = frame.frame.header
            file_path = frame.frame.file_path

            # Apply calibrations if requested and we have processing history
            if apply_calibrations and frame.frame.processing_history:
                logger.debug(f"Applying calibrations from processing history for live sidereal frame {frame.index}")
                frame_data = self._apply_calibrations_from_processing_history(
                    frame_data,
                    header,
                    frame.frame.processing_history,
                    frame.frame.correction_frames if hasattr(frame.frame, "correction_frames") else None,
                )
        else:
            logger.warning(f"No frame data available for sidereal frame {frame.index}")
            return

        # Create image entry
        image_id = f"{collect_id}_sidereal_{frame.index}"

        # Save image files
        output_image_file = self._save_image_files(frame_data, header, image_id, file_path, frame.starfield)

        # Create image metadata
        image_metadata = self._create_image_metadata(
            frame_data, header, image_id, collect_id, "sidereal", output_image_file
        )

        # Create point source annotations (stars)
        point_annotations = []
        star_centers = []

        if hasattr(frame, "starfield") and frame.starfield is not None:
            starfield = frame.starfield
            if hasattr(starfield, "detections") and starfield.detections:
                for star in starfield.detections:
                    if star.snr and star.snr >= self.snr_cut:
                        annotation = self._create_point_annotation(
                            star.x, star.y, image_id, self.box_size, star.snr, None, len(point_annotations)
                        )
                        if annotation:
                            point_annotations.append(annotation)
                            star_centers.append([star.x, star.y])

        # Create COCO datasets
        point_dataset = self._create_coco_dataset([image_metadata], point_annotations, "sidereal_star", "point_source")

        # Save individual COCO files
        self._save_coco_files(image_id, point_dataset, None, star_centers, frame_data, frame.starfield, None)

    def _process_rate_frame(
        self,
        frame,
        collect_id: str,
        apply_calibrations: bool,
    ) -> None:
        """Process a live rate track frame for COCO export."""
        # Check if frame has WCS (starfield with fit=True)
        if not hasattr(frame, "starfield") or not frame.starfield or not frame.starfield.fit:
            logger.info(f"Skipping live rate frame {frame.index} - no valid WCS solution")
            return

        # Early check for max_streak_length - skip entire frame if streak exceeds limit
        if self.max_streak_length is not None and hasattr(frame, "streak") and frame.streak:
            scale_factor = self._get_scale_factor(frame.starfield, self.senpai_run)
            scaled_length = frame.streak.pixel_length * scale_factor
            if scaled_length > self.max_streak_length:
                logger.info(f"Skipping live rate frame {frame.index} - streak length {scaled_length:.1f} > {self.max_streak_length}")
                return

        # Get frame data
        if hasattr(frame, "frame") and frame.frame is not None:
            frame_data = frame.frame.data
            header = frame.frame.header
            file_path = frame.frame.file_path

            # Apply calibrations if requested and we have processing history
            if apply_calibrations and frame.frame.processing_history:
                logger.debug(f"Applying calibrations from processing history for live rate frame {frame.index}")
                frame_data = self._apply_calibrations_from_processing_history(
                    frame_data,
                    header,
                    frame.frame.processing_history,
                    frame.frame.correction_frames if hasattr(frame.frame, "correction_frames") else None,
                )
        else:
            logger.warning(f"No frame data available for rate frame {frame.index}")
            return

        # Create image entry
        image_id = f"{collect_id}_rate_{frame.index}"

        # Save image files
        output_image_file = self._save_image_files(frame_data, header, image_id, file_path, frame.starfield)

        # Create image metadata
        image_metadata = self._create_image_metadata(
            frame_data, header, image_id, collect_id, "rate", output_image_file
        )

        # Create satellite annotations (point sources)
        satellite_annotations = []
        satellite_centers = []

        if hasattr(frame, "detections") and frame.detections is not None:
            for satellite in frame.detections.detections:
                # Estimate SNR if it's null
                if satellite.snr is not None:
                    snr_value = satellite.snr
                else:
                    snr_value = 5.0  # Default SNR for satellites

                annotation = self._create_point_annotation(
                    satellite.x,
                    satellite.y,
                    image_id,
                    self.streak_box_size,
                    snr_value,
                    None,
                    len(satellite_annotations),
                )
                if annotation:
                    satellite_annotations.append(annotation)
                    satellite_centers.append([satellite.x, satellite.y])

        # Create streak annotations (lines)
        streak_annotations = []
        streak_lines = []

        if hasattr(frame, "starfield") and frame.starfield is not None:
            starfield = frame.starfield
            
            # Use catalog stars if available (they have magnitudes), otherwise fall back to detections
            stars_to_process = []
            if hasattr(starfield, "catalog_stars") and starfield.catalog_stars:
                stars_to_process = starfield.catalog_stars
                logger.debug(f"Using {len(stars_to_process)} catalog stars for streak annotations")
            elif hasattr(starfield, "detections") and starfield.detections:
                stars_to_process = starfield.detections
                logger.debug(f"Using {len(stars_to_process)} detected stars for streak annotations (no catalog stars available)")
            else:
                logger.debug("No stars available for streak annotations")

            for star in stars_to_process:
                # Estimate SNR from magnitude if available, otherwise from counts
                if hasattr(star, "magnitude") and star.magnitude is not None:
                    snr_value = self._estimate_snr_from_magnitude(star.magnitude)
                elif star.snr is not None:
                    snr_value = star.snr
                elif hasattr(star, "counts") and star.counts is not None:
                    snr_value = self._estimate_snr_from_counts(star.counts)
                else:
                    snr_value = 5.0  # Default SNR

                if snr_value >= self.snr_cut:
                    # Create streak annotation
                    if hasattr(frame, "streak") and frame.streak is not None:
                        line = self._create_streak_line(star, frame.streak, frame_data)
                        if line:
                            annotation = self._create_streak_annotation(
                                star, line, image_id, len(streak_annotations)
                            )
                            if annotation:
                                streak_annotations.append(annotation)
                                streak_lines.append(line)
        else:
            logger.debug(
                f"No streak data for live rate frame {frame.index}: starfield={hasattr(frame, 'starfield')}, streak={hasattr(frame, 'streak')}"
            )

        # Create COCO datasets
        satellite_dataset = self._create_coco_dataset(
            [image_metadata], satellite_annotations, "satellite", "point_source"
        )
        streak_dataset = self._create_coco_dataset([image_metadata], streak_annotations, "rate_star", "streak_source")

        # Save individual COCO files
        self._save_coco_files(
            image_id,
            satellite_dataset,
            streak_dataset,
            satellite_centers,
            frame_data,
            streak_lines,
            frame.starfield,
            frame.streak,
        )

    def _save_image_files(
        self,
        frame_data: np.ndarray,
        header: fits.Header,
        image_id: str,
        file_path: Optional[str] = None,
        starfield=None,
    ) -> Optional[str]:
        """Save image files (PNG and/or FITS) and return the output file path."""
        output_image_file = None

        if self.write_png:
            output_image_file = str(self.output_dir / f"{image_id}.png")
            self._save_png_image(frame_data, output_image_file)

        if self.write_fits:
            output_image_file = str(self.output_dir / f"{image_id}.fits")
            # Always create new FITS file with processed data (which may have calibrations applied)
            hdu = fits.PrimaryHDU(frame_data, header)
            # Add WCS as second ImageHDU if available in starfield
            if starfield is not None and hasattr(starfield, 'wcs') and starfield.wcs is not None:
                try:
                    # Convert SENPAI WCS to Astropy WCS
                    astropy_wcs = starfield.wcs.to_astropy_wcs()
                    
                    # Create a dummy data array for the WCS HDU (same shape as original)
                    wcs_data = np.zeros_like(frame_data)
                    
                    # Create ImageHDU with WCS header
                    wcs_hdu = fits.ImageHDU(data=wcs_data, header=astropy_wcs.to_header(), name='WCS')
                    
                    # Append WCS HDU to the FITS file
                    hdu = fits.HDUList([hdu, wcs_hdu])
                    
                    logger.debug(f"Added WCS as second ImageHDU for {image_id}")
                except Exception as e:
                    logger.warning(f"Failed to add WCS as second ImageHDU for {image_id}: {str(e)}")
                    # Continue without WCS HDU if conversion fails
            
            hdu.writeto(output_image_file, overwrite=True)

        return output_image_file

    def _create_image_metadata(
        self,
        frame_data: np.ndarray,
        header: fits.Header,
        image_id: str,
        collect_id: str,
        frame_type: str,
        output_image_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create image metadata for COCO format."""
        # Extract metadata from header
        exposure_time = header.get("EXPTIME", header.get("EXPOSURE", 1.0))
        gain = header.get("GAIN", 1.0)
        date_obs = header.get("DATE-OBS", "")
        altitude = header.get("SK.MOUNT.ALTITUDE_DEGS", header.get("CENTALT"))
        azimuth = header.get("SK.MOUNT.AZIMUTH_DEGS", header.get("CENTAZ"))

        # Determine file name based on what was actually saved
        if output_image_file:
            file_name = os.path.basename(output_image_file)
        elif self.write_fits:
            file_name = f"{image_id}.fits"
        elif self.write_png:
            file_name = f"{image_id}.png"
        else:
            file_name = f"{image_id}.fits"  # Default

        return {
            "file_name": file_name,
            "width": frame_data.shape[1],
            "height": frame_data.shape[0],
            "id": image_id,
            "collectId": collect_id,
            "type": frame_type,
            "exposure_seconds": exposure_time,
            "gain": gain,
            "date": date_obs,
            "altitude": altitude,
            "azimuth": azimuth,
        }

    def _create_point_annotation(
        self,
        x: float,
        y: float,
        image_id: str,
        box_size: int,
        snr: float,
        magnitude: Optional[float] = None,
        annotation_id: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """Create a point source annotation (bounding box)."""
        # Check mask radius
        if self.mask_radius is not None:
            frame_center = np.array([x, y])
            if np.linalg.norm(frame_center) > self.mask_radius:
                return None

        # Create bounding box
        box = np.zeros(4)
        centroid = [round(x, 3), round(y, 3)]
        box[:2] = np.array(centroid) - box_size / 2
        box[2:] = box_size

        return {
            "id": annotation_id,
            "centroid": centroid,
            "bbox": box.tolist(),
            "image_id": image_id,
            "type": "bbox",
            "snr": round(snr, 3),
            "vmag": round(magnitude, 3) if magnitude else None,
            "area": box_size * box_size,
        }

    def _create_streak_line(
        self,
        star,
        streak,
        frame_data: np.ndarray,
    ) -> Optional[List[float]]:
        """Create a streak line from star and streak parameters."""
        if not streak:
            return None

        # Use the streak metadata to create a proper line
        # StreakMetadata has: pixel_length, sine_angle, cosine_angle, fwhm
        # Line format: [x, y, dx, dy] where (x,y) is start point and (dx,dy) is direction vector

        # Calculate the start point at one end of the streak
        # Start at star position and go back by half the streak length
        half_length = streak.pixel_length / 2
        x_start = star.x - half_length * streak.cosine_angle
        y_start = star.y - half_length * streak.sine_angle

        # Calculate direction vector from streak angle
        # The streak angle is stored as sine and cosine components
        # We need to create a vector of length pixel_length in the streak direction
        dx = streak.pixel_length * streak.cosine_angle
        dy = streak.pixel_length * streak.sine_angle

        # Create line: [x_start, y_start, dx, dy]
        line = [x_start, y_start, dx, dy]

        return line

    def _create_streak_annotation(
        self,
        star,
        line: List[float],
        image_id: str,
        annotation_id: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """Create a streak annotation (line)."""
        # Check mask radius
        if self.mask_radius is not None:
            frame_center = np.array([star.x, star.y])
            if np.linalg.norm(frame_center) > self.mask_radius:
                return None

        return {
            "image_id": image_id,
            "line": [round(x, 3) for x in line],
            "mag": round(star.magnitude, 3) if hasattr(star, "magnitude") and star.magnitude else None,
            "id": annotation_id,
            "category_id": 0,
            "area": 1,
            "blend_perc": round(star.blend_perc, 2) if hasattr(star, "blend_perc") and star.blend_perc else 0.0,
            "snr": star.snr if star.snr else None,
            "type": "line",
        }

    def _create_coco_dataset(
        self,
        images: List[Dict[str, Any]],
        annotations: List[Dict[str, Any]],
        category_name: str,
        supercategory: str,
    ) -> Dict[str, Any]:
        """Create a COCO format dataset."""
        return {
            "images": images,
            "metadata": {
                "module": "senpai.export.coco",
                "version": "1.0.0",
            },
            "annotations": annotations,
            "categories": [{"id": 0, "name": category_name, "supercategory": supercategory}],
        }

    def _save_coco_files(
        self,
        image_id: str,
        point_dataset: Dict[str, Any],
        streak_dataset: Optional[Dict[str, Any]] = None,
        point_centers: Optional[List[List[float]]] = None,
        frame_data: Optional[np.ndarray] = None,
        streak_lines: Optional[List[List[float]]] = None,
        starfield=None,
        streak=None,
    ) -> None:
        """Save individual COCO format files for an image."""
        # Save point source annotations (if any)
        if point_dataset["annotations"]:
            point_file = self.output_dir / f"{image_id}_point_sat.json"
            logger.info(f"Writing point source annotations: {point_file}")
            with open(point_file, "w") as f:
                json.dump(point_dataset, f, indent=4)

        # Save streak annotations (if any)
        if streak_dataset and streak_dataset["annotations"]:
            streak_file = self.output_dir / f"{image_id}_line_star.json"
            logger.info(f"Writing streak annotations: {streak_file}")
            with open(streak_file, "w") as f:
                json.dump(streak_dataset, f, indent=4)

        # Save annotated image if requested
        if self.save_annotated_images and frame_data is not None:
            self._save_annotated_image(frame_data, starfield, streak, image_id)

    def _save_annotated_image(
        self,
        frame_data: np.ndarray,
        starfield,
        streak,
        image_id: str,
    ) -> None:
        """Save an annotated version of the image using scaled-up starfield and streak."""
        try:
            from senpai.engine.plotting.images import plot_single_frame

            # Scale up starfield and streak for plotting if needed
            scale_factor = self._get_scale_factor(starfield, self.senpai_run)
            if scale_factor != 1.0:
                logger.debug(f"Scaling starfield and streak for plotting by factor {scale_factor}")
                # We need to get the frame to determine the scaling method
                # For now, we'll use the default method since we don't have direct frame access here
                scaled_starfield = self._scale_starfield_for_plotting(starfield, scale_factor, None)
                scaled_streak = self._scale_streak_for_plotting(streak, scale_factor, None)
            else:
                scaled_starfield = starfield
                scaled_streak = streak

            output_path = self.output_dir / f"{image_id}_annotated.png"
            plot_single_frame(
                frame_data,
                starfield=scaled_starfield,
                streak=scaled_streak,
                output_file=output_path,
                centercross=False,
            )
            logger.info(f"Saved annotated image: {output_path}")
        except ImportError:
            logger.warning("Could not import plot_single_frame - skipping annotated image")
        except Exception as e:
            logger.warning(f"Failed to save annotated image for {image_id}: {e}")

    def _save_png_image(self, data: np.ndarray, output_path: str) -> None:
        """Save image data as PNG."""
        with open(output_path, "wb") as f:
            writer = png.Writer(
                width=data.shape[1],
                height=data.shape[0],
                compression=0,
                bitdepth=16,
                greyscale=True,
                alpha=False,
            )
            writer.write(f, data)

    def _estimate_snr_from_magnitude(self, magnitude: float) -> float:
        """Estimate SNR from magnitude (brighter stars = higher SNR)."""
        # Rough approximation: SNR ~ 10^(0.4 * (20 - magnitude))
        # This assumes a limiting magnitude of ~20 and typical SNR scaling
        if magnitude is None:
            return 5.0
        return max(1.0, 10 ** (0.4 * (20 - magnitude)))

    def _estimate_snr_from_counts(self, counts: float) -> float:
        """Estimate SNR from counts (more counts = higher SNR)."""
        # Rough approximation: SNR ~ sqrt(counts) for Poisson noise
        if counts is None or counts <= 0:
            return 5.0
        return max(1.0, np.sqrt(counts / 1000))  # Normalize by typical background

    def _apply_calibrations_from_processing_history(
        self,
        frame_data: np.ndarray,
        header: fits.Header,
        processing_history: List,
        correction_frames: Optional[Dict] = None,
    ) -> np.ndarray:
        """Apply the same calibration chain that was used during SENPAI processing."""
        from senpai.engine.models.images import ProcessingStep
        from senpai.engine.utils.darks import apply_dark_subtraction
        from senpai.engine.utils.preprocessing import remove_column_and_row_medians

        # Start with the raw frame data
        processed_data = frame_data.copy().astype(np.float64)

        # Apply processing steps in the same order they were applied
        for step in processing_history:
            step_type = step.step_type.value

            if step_type == "dark_subtract":
                # Apply dark subtraction
                if correction_frames and ProcessingStep.DARK_SUBTRACT in correction_frames:
                    dark_frame = correction_frames[ProcessingStep.DARK_SUBTRACT]
                    processed_data -= dark_frame
                    logger.debug("Applied dark subtraction from correction frame")
                else:
                    # Try to find and apply master dark
                    try:
                        from senpai.core.config import get_config

                        config = get_config()
                        if config.calibrations.auto_apply_darks and config.calibrations.master_darks_dir:
                            # Create a temporary ProcessedFitsImage to use the dark subtraction function
                            from senpai.engine.models.images import ProcessedFitsImage
                            from senpai.engine.models.metadata import ImageMetadata

                            temp_image = ProcessedFitsImage(
                                data=processed_data,
                                header=header,
                                metadata=ImageMetadata(width=processed_data.shape[1], height=processed_data.shape[0]),
                                data_type=processed_data.dtype,
                            )
                            temp_image = apply_dark_subtraction(temp_image, config.calibrations.master_darks_dir)
                            processed_data = temp_image.data
                            logger.debug("Applied dark subtraction from master dark")
                    except Exception as e:
                        logger.warning(f"Could not apply dark subtraction: {e}")

            elif step_type == "column_median_subtract" or step_type == "row_median_subtract":
                # Apply column and row median removal
                if correction_frames and ProcessingStep.COLUMN_MEDIAN_SUBTRACT in correction_frames:
                    column_medians = correction_frames[ProcessingStep.COLUMN_MEDIAN_SUBTRACT]
                    processed_data -= column_medians
                    logger.debug("Applied column median subtraction from correction frame")
                if correction_frames and ProcessingStep.ROW_MEDIAN_SUBTRACT in correction_frames:
                    row_medians = correction_frames[ProcessingStep.ROW_MEDIAN_SUBTRACT]
                    processed_data -= row_medians
                    logger.debug("Applied row median subtraction from correction frame")
                else:
                    # Apply the standard column and row median removal
                    try:
                        from senpai.engine.models.images import ProcessedFitsImage
                        from senpai.engine.models.metadata import ImageMetadata

                        temp_image = ProcessedFitsImage(
                            data=processed_data,
                            header=header,
                            metadata=ImageMetadata(width=processed_data.shape[1], height=processed_data.shape[0]),
                            data_type=processed_data.dtype,
                        )
                        temp_image = remove_column_and_row_medians(temp_image, store_intermediates=False)
                        processed_data = temp_image.data
                        logger.debug("Applied column and row median removal")
                    except Exception as e:
                        logger.warning(f"Could not apply column/row median removal: {e}")

            elif step_type == "background_subtract":
                # Apply background subtraction
                if correction_frames and ProcessingStep.BACKGROUND_SUBTRACT in correction_frames:
                    background = correction_frames[ProcessingStep.BACKGROUND_SUBTRACT]
                    processed_data -= background
                    logger.debug("Applied background subtraction from correction frame")
                else:
                    logger.warning("Background subtraction detected but no correction frame available - cannot apply")

            elif step_type == "flat_divide":
                # Apply flat division
                if correction_frames and ProcessingStep.FLAT_DIVIDE in correction_frames:
                    flat_frame = correction_frames[ProcessingStep.FLAT_DIVIDE]
                    processed_data /= flat_frame
                    logger.debug("Applied flat division from correction frame")
                else:
                    logger.warning("Flat division detected but no correction frame available - cannot apply")

        return processed_data

    def _apply_calibrations_from_header(
        self,
        image,
        header: Dict[str, Any],
    ):
        """Apply calibrations from header metadata."""
        # This is a placeholder - implement actual calibration logic
        return image

    def _get_scale_factor(self, starfield, senpai_run=None) -> float:
        """Get scale factor from run level or starfield if available."""
        # First check run level scale_factor
        if senpai_run and hasattr(senpai_run, "scale_factor") and senpai_run.scale_factor is not None:
            logger.debug(f"Found scale_factor at run level: {senpai_run.scale_factor}")
            return senpai_run.scale_factor

        # Fallback to starfield scale_factor (for backward compatibility)
        if starfield and hasattr(starfield, "scale_factor") and starfield.scale_factor is not None:
            logger.debug(f"Found scale_factor in starfield: {starfield.scale_factor}")
            return starfield.scale_factor

        logger.debug("No scale_factor found, using 1.0")
        return 1.0

    def _get_scaling_method(self, frame) -> str:
        """Get the scaling method used for a frame from its processing history."""
        if hasattr(frame, "frame") and hasattr(frame.frame, "processing_history"):
            for step in reversed(frame.frame.processing_history):
                if step.step_type.value == "fwhm_optimization":
                    return step.parameters.get("method", "block_median")
        return "block_median"  # Default fallback

    def _scale_starfield_for_plotting(self, starfield, scale_factor: float, frame=None):
        """Create a scaled-up copy of starfield for plotting using proper unscaling."""
        if starfield is None or scale_factor == 1.0:
            return starfield

        # Get the scaling method used
        scaling_method = self._get_scaling_method(frame) if frame else "block_median"

        # Use the proper unscaling function
        from senpai.engine.utils.preprocessing import unscale_starfield_coordinates

        return unscale_starfield_coordinates(starfield, scale_factor, scaling_method)

    def _scale_streak_for_plotting(self, streak, scale_factor: float, frame=None):
        """Create a scaled-up copy of streak metadata for plotting using proper unscaling."""
        if streak is None or scale_factor == 1.0:
            return streak

        # Get the scaling method used
        scaling_method = self._get_scaling_method(frame) if frame else "block_median"

        # Use the proper unscaling function
        from senpai.engine.utils.preprocessing import unscale_streak_metadata

        return unscale_streak_metadata(streak, scale_factor, scaling_method)

    def _scale_coordinates(self, x: float, y: float, scale_factor: float) -> tuple[float, float]:
        """Scale coordinates back up to original image size."""
        if scale_factor != 1.0:
            scaled_x, scaled_y = x * scale_factor, y * scale_factor
            logger.debug(f"Scaling coordinates ({x}, {y}) by factor {scale_factor} -> ({scaled_x}, {scaled_y})")
            return scaled_x, scaled_y
        return x, y

    def _create_streak_line_scaled(
        self,
        star,
        streak,
        frame_data: np.ndarray,
        scale_factor: float,
    ) -> Optional[List[float]]:
        """Create a streak line from star and streak parameters with proper scaling."""
        if not streak:
            return None

        # Scale star coordinates back up
        x_star, y_star = self._scale_coordinates(star.x, star.y, scale_factor)

        # Calculate the start point at one end of the streak
        # Start at star position and go back by half the streak length
        scaled_length = streak.pixel_length * scale_factor
        half_length = scaled_length / 2
        x_start = x_star - half_length * streak.cosine_angle
        y_start = y_star - half_length * streak.sine_angle

        # Calculate direction vector from streak angle (angle doesn't change with scaling)
        dx = scaled_length * streak.cosine_angle
        dy = scaled_length * streak.sine_angle

        # Create line: [x_start, y_start, dx, dy]
        line = [x_start, y_start, dx, dy]

        return line

    def _create_streak_annotation_scaled(
        self,
        star,
        line: List[float],
        image_id: str,
        annotation_id: int = 0,
        scale_factor: float = 1.0,
    ) -> Optional[Dict[str, Any]]:
        """Create a streak annotation (line) with proper scaling."""
        # Scale star coordinates for mask radius check
        x, y = self._scale_coordinates(star.x, star.y, scale_factor)
        if self.mask_radius is not None:
            frame_center = np.array([x, y])
            if np.linalg.norm(frame_center) > self.mask_radius:
                return None

        # Estimate SNR from counts if SNR is null
        if star.snr is not None:
            snr_value = star.snr
        elif hasattr(star, "counts") and star.counts is not None:
            snr_value = self._estimate_snr_from_counts(star.counts)
        else:
            snr_value = 5.0

        return {
            "image_id": image_id,
            "line": [round(x, 3) for x in line],
            "mag": round(star.magnitude, 3) if hasattr(star, "magnitude") and star.magnitude else None,
            "id": annotation_id,
            "category_id": 0,
            "area": 1,
            "blend_perc": round(star.blend_perc, 2) if hasattr(star, "blend_perc") and star.blend_perc else 0.0,
            "snr": round(snr_value, 3),
            "type": "line",
        }

    def export_batch(
        self,
        senpai_runs: List[Union[SenpaiRun, SenpaiRunResult]],
        collect_ids: List[str],
        apply_calibrations: bool = True,
    ) -> None:
        """Export multiple SENPAI runs to individual COCO format files."""
        for senpai_run, collect_id in zip(senpai_runs, collect_ids, strict=False):
            logger.info(f"Exporting SENPAI run {collect_id}")
            self.export_senpai_run(senpai_run, collect_id, apply_calibrations)
