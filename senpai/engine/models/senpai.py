import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel, field_serializer

import senpai
from senpai.engine.models.astrometry import WCSMetadata, WCSModel
from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.models.metadata import (
    CollectionMetadata,
    FrameMetadata,
    SeeingModel,
    StreakMetadata,
    TelescopeMetadata,
    TrackMode,
)
from senpai.engine.models.starfield import SatelliteInImage, SatelliteListImage, StarField
from senpai.engine.utils.frame_organization import extract_uct_time_from_header

logger = logging.getLogger(__name__)


class SiderealFrameSerializable(BaseModel):
    starfield: StarField | None = None
    seeing: SeeingModel | None = None
    hardware: TelescopeMetadata | None = None
    detections: SatelliteListImage | None = None
    original_frame_path: str | None = None
    processed_frame_path: str | None = None  # Path to the processed/reduced FITS file
    original_frame_header: dict | None = None
    processing_history: list | None = None  # Store processing history for calibration reproduction
    correction_frames: dict | None = None  # Store correction frames for calibration reproduction
    index: int
    timestamp: str | None = None  # ISO string; None when the frame has no time header
    frame_metadata: FrameMetadata | None = None
    photometry_summary: dict | None = None
    streak_candidates: list[dict] = []


class SiderealFrame(BaseModel):
    starfield: StarField | None = None
    seeing: SeeingModel | None = None
    hardware: TelescopeMetadata | None = None
    detections: SatelliteListImage | None = None
    frame: ProcessedFitsImage
    index: int
    timestamp: datetime | None = None  # None when the frame has no time header
    frame_metadata: FrameMetadata | None = None
    photometry_summary: dict | None = None
    streak_candidates: list = []


class RateTrackFrameSerializable(BaseModel):
    starfield: StarField | None = None
    streak: StreakMetadata | None = None
    seeing: SeeingModel | None = None
    hardware: TelescopeMetadata | None = None
    detections: SatelliteListImage | None = None
    frame_metadata: FrameMetadata | None = None
    original_frame_path: str | None = None
    processed_frame_path: str | None = None  # Path to the processed/reduced FITS file
    original_frame_header: dict | None = None
    processing_history: list | None = None  # Store processing history for calibration reproduction
    correction_frames: dict | None = None  # Store correction frames for calibration reproduction
    index: int
    timestamp: str | None = None  # ISO string; None when the frame has no time header
    pixel_track_rate_per_second: float | None = None
    photometry_summary: dict | None = None
    streak_candidates: list[dict] = []

    @field_serializer("pixel_track_rate_per_second")
    def serialize_rate(self, v: float | None) -> float | None:
        return round(v, 3) if v is not None else None


class RateTrackFrame(BaseModel):
    starfield: StarField | None = None
    streak: StreakMetadata | None = None
    seeing: SeeingModel | None = None
    hardware: TelescopeMetadata | None = None
    detections: SatelliteListImage | None = None
    frame_metadata: FrameMetadata | None = None
    frame: ProcessedFitsImage
    index: int
    timestamp: datetime | None = None  # None when the frame has no time header
    pixel_track_rate_per_second: float | None = None
    photometry_summary: dict | None = None
    streak_candidates: list = []

    @field_serializer("pixel_track_rate_per_second")
    def serialize_rate(self, v: float | None) -> float | None:
        return round(v, 3) if v is not None else None


class FrameShift(BaseModel):
    source_index: int
    target_index: int
    x_shift: float | None = None  # source to target
    y_shift: float | None = None  # source to target
    is_valid: bool = True
    processed: bool = False
    error_message: str | None = None


class FrameSummary(BaseModel):
    """Compact per-frame summary with actionable information only."""

    index: int
    timestamp: str | None = None  # ISO string; None when the frame has no time header
    track_mode: str | None = None  # "sidereal" / "rate"
    original_frame_path: str | None = None
    processed_frame_path: str | None = None

    # WCS
    wcs_status: str | None = None  # WCSStatus value
    wcs: WCSModel | None = None
    wcs_metadata: WCSMetadata | None = None  # FOV, plate scale, center RA/Dec

    # Observing
    frame_metadata: FrameMetadata | None = None
    seeing: SeeingModel | None = None

    # Photometry
    photometry_summary: dict | None = None
    limiting_magnitude: float | None = None

    # Object detections (satellites/asteroids)
    detections: list[SatelliteInImage] = []

    # Streak candidates (sidereal frames only)
    streak_candidates: list[dict] = []

    # Rate-track specific
    pixel_track_rate_per_second: float | None = None
    streak: StreakMetadata | None = None

    # Compact diagnostics
    distortion_metrics: dict[str, float] | None = None
    num_catalog_stars: int | None = None
    num_astrometric_fit_stars: int | None = None

    @field_serializer("pixel_track_rate_per_second")
    def serialize_rate(self, v: float | None) -> float | None:
        return round(v, 3) if v is not None else None


class CorrelatedStreak(BaseModel):
    """A streak confirmed across multiple frames with resolved direction."""

    streak_id: str
    frame_indices: list[int] = []
    positions_x: list[float] = []
    positions_y: list[float] = []
    ra: list[float] = []
    dec: list[float] = []
    timestamps_iso: list[str] = []
    angle_deg: float  # Refined angle in [0, 180)
    direction_deg: float | None = None  # Resolved direction in [0, 360), None if single-frame
    length_pixels: float | None = None  # Refined streak length from boxcar scan
    rate_pixels_per_sec: float | None = None
    rate_arcsec_per_sec: float | None = None
    rate_ra_arcsec_per_sec: float | None = None  # RA component of angular rate
    rate_dec_arcsec_per_sec: float | None = None  # Dec component of angular rate
    confirmed: bool = False  # True if matched across 2+ frames
    best_snr: float = 0.0
    best_flux: float | None = None
    best_calibrated_magnitudes: dict[str, float] | None = None
    best_magnitude_errs: dict[str, float] | None = None

    @field_serializer("angle_deg")
    def _round_angle(self, v: float) -> float:
        return round(v, 2)

    @field_serializer(
        "direction_deg", "rate_pixels_per_sec", "rate_arcsec_per_sec",
        "rate_ra_arcsec_per_sec", "rate_dec_arcsec_per_sec",
        "length_pixels", "best_snr", "best_flux",
    )
    def _round_optional(self, v: float | None) -> float | None:
        return round(v, 3) if v is not None else None


class SenpaiRunSummary(BaseModel):
    """Compact summary of an entire SenpaiRun, dropping bulk star lists and calibration data."""

    id: str
    num_frames: int
    completed: bool = False
    compute_seconds: float | None = None
    senpai_version: str = senpai.__version__
    error_message: str | None = None
    scale_factor: float | None = None
    collect_metadata: CollectionMetadata
    frame_shifts: list[FrameShift] = []
    frames: list[FrameSummary] = []
    correlated_streaks: list[CorrelatedStreak] = []

    def model_dump(self, **kwargs):
        """Override model_dump to ensure datetime fields are serialized as ISO format strings."""
        return super().model_dump(mode="json", **kwargs)


class SenpaiRunResult(BaseModel):
    id: str
    num_frames: int
    collect_metadata: CollectionMetadata
    completed: bool = False
    compute_seconds: float | None = None
    senpai_version: str = senpai.__version__
    error_message: str | None = None
    scale_factor: float | None = None  # Store the actual scale factor used for the entire run
    frame_shifts: list[FrameShift] = []
    frame_shifts_failed: list[FrameShift] = []
    sidereal_frames: list[SiderealFrameSerializable] = []
    rate_track_frames: list[RateTrackFrameSerializable] = []
    correlated_streaks: list[CorrelatedStreak] = []

    def model_dump(self, **kwargs):
        """Override model_dump to ensure datetime fields are serialized as ISO format strings."""
        return super().model_dump(mode="json", **kwargs)


class SenpaiRun(BaseModel):
    id: str
    num_frames: int
    completed: bool = False
    compute_seconds: float | None = None
    senpai_version: str = senpai.__version__
    error_message: str | None = None
    scale_factor: float | None = None  # Store the actual scale factor used for the entire run
    collect_metadata: CollectionMetadata
    frame_shifts: list[FrameShift] = []
    frame_shifts_failed: list[FrameShift] = []
    sidereal_frames: list[SiderealFrame] = []
    rate_track_frames: list[RateTrackFrame] = []
    correlated_streaks: list[CorrelatedStreak] = []

    @classmethod
    def organize_senpai_frames(
        cls,
        frames: list[ProcessedFitsImage],
        id: str = "",
        force_track_mode: TrackMode | None = None,
    ) -> "SenpaiRun":
        # Initialize empty lists and create a new SenpaiRun instancexxx
        sidereal_frames = []
        rate_track_frames = []

        # Sort frames by time. A frame with no usable date header keeps its
        # input position rather than crashing the run (focus/raw frames often
        # carry only NAXIS); timed frames sort ahead of untimed ones.
        timed: list[tuple[ProcessedFitsImage, datetime]] = []
        untimed: list[ProcessedFitsImage] = []
        for frame in frames:
            try:
                frame_time = extract_uct_time_from_header(frame.header)
                timed.append((frame, frame_time))
            except AttributeError:
                fname = Path(frame.file_path).name if frame.file_path else "?"
                logger.warning(
                    "Frame %s has no usable date/time header — keeping input order; "
                    "time-based frame ordering and correlation are disabled for it",
                    fname,
                )
                untimed.append(frame)

        timed.sort(key=lambda x: x[1])
        ordered: list[tuple[ProcessedFitsImage, datetime | None]] = (
            [(f, t) for f, t in timed] + [(f, None) for f in untimed]
        )

        # Process frames in time order and assign sequential indexes
        for index, (frame, timestamp) in enumerate(ordered):
            frame_metadata = FrameMetadata.from_header(frame.header)

            # Verbose audit: log every header-gated capability this frame loses.
            fname = Path(frame.file_path).name if frame.file_path else f"index {index}"
            frame_metadata.log_missing_capabilities(logger, label=f"Frame {fname}")

            header_track_type = frame_metadata.track_mode
            if force_track_mode is not None:
                if header_track_type is not None and header_track_type != force_track_mode:
                    fname = Path(frame.file_path).name if frame.file_path else "?"
                    logger.warning(
                        "Frame %d (%s): header reports %s but CLI forced %s — processing as %s",
                        index, fname, header_track_type.value, force_track_mode.value, force_track_mode.value,
                    )
                frame_metadata.track_mode = force_track_mode
            track_type = frame_metadata.track_mode

            if track_type == TrackMode.RATE:
                model = RateTrackFrame
                framelist = rate_track_frames
            elif track_type == TrackMode.SIDEREAL:
                model = SiderealFrame
                framelist = sidereal_frames
            else:
                logger.debug(f"unknown track type: {track_type}")
                if frame_metadata.track_rate_ra_arcsec_per_second and frame_metadata.track_rate_dec_arcsec_per_second:
                    overall_rate = (
                        frame_metadata.track_rate_ra_arcsec_per_second**2
                        + frame_metadata.track_rate_dec_arcsec_per_second**2
                    ) ** 0.5
                    if overall_rate > 1.0:
                        logger.debug("track rates are high, defaulting to rate track")
                        model = RateTrackFrame
                        framelist = rate_track_frames
                    else:
                        logger.debug("track rates are too low, defaulting to sidereal")
                        model = SiderealFrame
                        framelist = sidereal_frames
                else:
                    logger.debug("no track rates, defaulting to sidereal")
                    model = SiderealFrame
                    framelist = sidereal_frames

            framelist.append(
                model(
                    frame=frame,
                    index=index,
                    timestamp=timestamp,
                    frame_metadata=frame_metadata,
                )
            )

        # Log organized frame summary
        for sf in sidereal_frames:
            fname = Path(sf.frame.file_path).name if sf.frame.file_path else "?"
            exp = sf.frame_metadata.exposure_time_seconds if sf.frame_metadata else None
            ts = sf.timestamp.strftime("%H:%M:%S") if sf.timestamp else "?"
            logger.info("Frame %d: SIDEREAL  %s  exp=%.1fs  time=%s", sf.index, fname, exp or 0, ts)
        for rf in rate_track_frames:
            fname = Path(rf.frame.file_path).name if rf.frame.file_path else "?"
            exp = rf.frame_metadata.exposure_time_seconds if rf.frame_metadata else None
            ts = rf.timestamp.strftime("%H:%M:%S") if rf.timestamp else "?"
            rate_ra = rf.frame_metadata.track_rate_ra_arcsec_per_second if rf.frame_metadata else None
            rate_dec = rf.frame_metadata.track_rate_dec_arcsec_per_second if rf.frame_metadata else None
            logger.info(
                "Frame %d: RATE      %s  exp=%.1fs  time=%s  rate=(%.1f,%.1f)\"/s",
                rf.index, fname, exp or 0, ts, rate_ra or 0, rate_dec or 0,
            )
        logger.info(
            "Organized %d frames: %d sidereal + %d rate",
            len(frames), len(sidereal_frames), len(rate_track_frames),
        )

        return cls(
            id=id,
            num_frames=len(frames),
            collect_metadata=CollectionMetadata(),
            sidereal_frames=sidereal_frames,
            rate_track_frames=rate_track_frames,
        )

    def add_frame_shift(
        self,
        source_index: int,
        target_index: int,
        x_shift: float | None = None,
        y_shift: float | None = None,
        is_valid: bool = True,
        processed: bool = False,
        error_message: str | None = None,
    ) -> None:
        """Add a measured shift between two frames."""
        self.frame_shifts.append(
            FrameShift(
                source_index=source_index,
                target_index=target_index,
                x_shift=x_shift,
                y_shift=y_shift,
                is_valid=is_valid,
                processed=processed,
                error_message=error_message,
            )
        )

    def create_valid_path(self) -> None:
        """Find a valid path through all frames, starting from a sidereal frame."""
        # Get all frame indices
        all_frames = self.sidereal_frames + self.rate_track_frames
        if not all_frames:
            return

        # Sort frames by index
        all_frames.sort(key=lambda x: x.index)

        # Find a sidereal frame to start with
        start_frame = None
        if self.sidereal_frames:
            # Start with the first sidereal frame (lowest index)
            start_frame = min(self.sidereal_frames, key=lambda x: x.index)
        else:
            # If no sidereal frames, use the first frame
            start_frame = all_frames[0]

        start_index = start_frame.index

        # Create a list of frames in the order we want to process them
        ordered_frames = []

        # If starting frame is at index 0, go forward (0 -> max_index)
        if start_index == all_frames[0].index:
            ordered_frames = all_frames
        # Otherwise, expand outward from start: forward then backward.
        # This ensures the first cross-mode shift (e.g., sidereal→rate)
        # uses the start frame as the anchor, not the last forward frame.
        # Example: frames [0r,1r,2r,3r,4r,5r,6s,7s,8s,9s], start=6
        #   forward: 6→7→8→9
        #   backward: 6→5→4→3→2→1→0  (NOT 9→5, which has a large gap)
        else:
            # Get frames after the start frame (forward, ascending)
            after_frames = [f for f in all_frames if f.index > start_index]
            after_frames.sort(key=lambda x: x.index)

            # Get frames before the start frame (backward, descending)
            before_frames = [f for f in all_frames if f.index < start_index]
            before_frames.sort(key=lambda x: x.index, reverse=True)

            ordered_frames = [start_frame] + after_frames

        # Create shifts between consecutive frames in our ordered path
        for i in range(len(ordered_frames) - 1):
            current_frame = ordered_frames[i]
            next_frame = ordered_frames[i + 1]

            self.add_frame_shift(
                source_index=current_frame.index,
                target_index=next_frame.index,
                x_shift=None,
                y_shift=None,
                is_valid=False,
                error_message="Shift not yet measured",
            )

        # Add backward chain from start frame (separate branch)
        if start_index != all_frames[0].index:
            prev_frame = start_frame
            for bf in before_frames:
                self.add_frame_shift(
                    source_index=prev_frame.index,
                    target_index=bf.index,
                    x_shift=None,
                    y_shift=None,
                    is_valid=False,
                    error_message="Shift not yet measured",
                )
                prev_frame = bf

        # Log the analysis chain
        self.log_analysis_chain()

    def log_analysis_chain(self) -> None:
        """Log the analysis chain path with status indicators:
        ✅ - processed and valid
        ❓ - not processed yet
        ❌ - processed but failed
        """
        if not self.frame_shifts and not self.frame_shifts_failed:
            logger.info("No analysis chain to log (no frame shifts)")
            return

        # Create the chain representation with status indicators
        chain_parts = []

        # Add active shifts
        for shift in self.frame_shifts:
            status = "❓"  # Not processed yet
            if shift.processed:
                status = "✅" if shift.is_valid else "❌"  # Processed: valid or failed
            chain_parts.append(f"{shift.source_index}-{shift.target_index} {status}")

        # Add failed shifts that have been moved to the failed list
        for shift in self.frame_shifts_failed:
            chain_parts.append(f"{shift.source_index}-{shift.target_index} ❌")

        chain_str = "analysis chain: " + ", ".join(chain_parts)
        logger.info(chain_str)

    def get_next_shift(self) -> FrameShift:
        """Get the next unprocessed frame shift, or None if all are processed."""
        for shift in self.frame_shifts:
            if not shift.processed:
                return shift
        return None

    def update_valid_path(self) -> list[FrameShift]:
        """Update a valid path through all frames"""
        logger = logging.getLogger(__name__)

        # Get all failed shifts that have been processed
        failed_shifts = [shift for shift in self.frame_shifts if not shift.is_valid and shift.processed]

        if not failed_shifts:
            logger.info("No failed shifts to process")
            return self.frame_shifts

        logger.info(f"Processing {len(failed_shifts)} failed shifts")

        # Create a set of previously attempted shift pairs to avoid loops
        attempted_shifts = set()
        for shift in self.frame_shifts_failed:
            attempted_shifts.add((shift.source_index, shift.target_index))

        # Process each failed shift
        for failed_shift in failed_shifts:
            source_idx = failed_shift.source_index
            failed_target_idx = failed_shift.target_index

            logger.info(f"Processing failed shift {source_idx}-{failed_target_idx}")

            # Add this failed shift to our attempted set
            attempted_shifts.add((source_idx, failed_target_idx))

            # Find the position of the failed shift in the list
            failed_shift_index = self.frame_shifts.index(failed_shift)
            logger.info(f"Failed shift is at position {failed_shift_index} in frame_shifts")

            # Get all frame indices
            all_frames = sorted(self.sidereal_frames + self.rate_track_frames, key=lambda x: x.index)
            all_indices = sorted([frame.index for frame in all_frames])
            logger.info(f"All frame indices: {all_indices}")

            # Move the failed shift to the failed list
            if not hasattr(self, "frame_shifts_failed"):
                self.frame_shifts_failed = []
            self.frame_shifts_failed.append(failed_shift)
            self.frame_shifts.remove(failed_shift)  # Remove it first

            # Find and DELETE any unprocessed shifts that depend on the failed target
            dependent_shifts = [
                shift for shift in self.frame_shifts if shift.source_index == failed_target_idx and not shift.processed
            ]

            if dependent_shifts:
                logger.info(
                    f"Found {len(dependent_shifts)} unprocessed shifts that depend on failed target {failed_target_idx}"
                )
                for dep_shift in dependent_shifts:
                    logger.info(f"Deleting dependent shift {dep_shift.source_index}-{dep_shift.target_index}")
                    # Simply remove the shift without adding to failed list
                    self.frame_shifts.remove(dep_shift)

            # Determine the direction of the failed shift
            shift_direction = 1 if failed_target_idx > source_idx else -1

            # Find indices in the SAME direction as the failed shift
            # If we were going up (5->6), only consider higher indices (7,8,...)
            # If we were going down (5->4), only consider lower indices (3,2,...)
            if shift_direction > 0:
                next_indices = [idx for idx in all_indices if idx > failed_target_idx]
            else:
                next_indices = [idx for idx in all_indices if idx < failed_target_idx]

            logger.info(f"Potential next indices in direction {shift_direction}: {next_indices}")

            # Filter out indices that would create shifts we've already attempted
            next_indices = [idx for idx in next_indices if (source_idx, idx) not in attempted_shifts]

            if not next_indices:
                logger.info(
                    f"No untried indices available in direction {shift_direction} from source {source_idx}, giving up"
                )
                continue

            # Sort indices by proximity to failed_target_idx (maintaining direction)
            if shift_direction > 0:
                next_indices.sort()  # Ascending order for upward shifts
                next_target_idx = next_indices[0]  # Get the next higher index
            else:
                next_indices.sort(reverse=True)  # Descending order for downward shifts
                next_target_idx = next_indices[0]  # Get the next lower index

            logger.info(f"Selected next target index: {next_target_idx}")

            # Create new replacement shift
            new_shift = FrameShift(
                source_index=source_idx,
                target_index=next_target_idx,
                x_shift=None,
                y_shift=None,
                is_valid=False,
                processed=False,
                error_message="Replacement shift not yet measured",
            )

            logger.info(f"Created new shift {source_idx}-{next_target_idx}")

            # Insert the new shift at the same position
            self.frame_shifts.insert(failed_shift_index, new_shift)
            logger.info(f"Inserted new shift at position {failed_shift_index}")

        # Log the updated chain
        self.log_analysis_chain()
        return self.frame_shifts

    def get_frame_by_index(self, index: int) -> SiderealFrame | RateTrackFrame | None:
        """Get a frame by its index, regardless of whether it's sidereal or rate track."""
        for frame in self.sidereal_frames:
            if frame.index == index:
                return frame
        for frame in self.rate_track_frames:
            if frame.index == index:
                return frame
        return None

    def to_result(self) -> SenpaiRunResult:
        """Convert this SenpaiRun to a serializable SenpaiRunResult."""
        # Convert sidereal frames to serializable versions
        sidereal_frames_serializable = []
        for frame in self.sidereal_frames:
            # Get the file path and convert to absolute path if it's not already
            file_path = None
            if hasattr(frame.frame, "file_path") and frame.frame.file_path:
                file_path = os.path.abspath(frame.frame.file_path)

            # Get the processed file path if available
            processed_file_path = None
            if hasattr(frame.frame, "processed_file_path") and frame.frame.processed_file_path:
                processed_file_path = os.path.abspath(frame.frame.processed_file_path)

            # Convert header to a serializable dictionary
            frame_header = {}
            if hasattr(frame.frame, "header") and frame.frame.header:
                for key, value in frame.frame.header.items():
                    try:
                        # Test if the value is JSON serializable
                        json.dumps({key: value})
                        frame_header[key] = value
                    except (TypeError, OverflowError):
                        # Skip values that aren't JSON serializable
                        continue

            # Convert processing history to serializable format
            processing_history = None
            if hasattr(frame.frame, "processing_history") and frame.frame.processing_history:
                processing_history = []
                for step in frame.frame.processing_history:
                    try:
                        # Convert ProcessingStep enum to string
                        step_dict = {"step_type": step.step_type.value, "parameters": step.parameters}
                        processing_history.append(step_dict)
                    except Exception as e:
                        logger.warning(f"Could not serialize processing step: {e}")

            # Convert correction frames to serializable format
            correction_frames = None
            if hasattr(frame.frame, "correction_frames") and frame.frame.correction_frames:
                correction_frames = {}
                for step_type, correction_data in frame.frame.correction_frames.items():
                    try:
                        # Convert numpy arrays to lists for JSON serialization
                        if isinstance(correction_data, np.ndarray):
                            correction_frames[step_type.value] = correction_data.tolist()
                        else:
                            correction_frames[step_type.value] = correction_data
                    except Exception as e:
                        logger.warning(f"Could not serialize correction frame for {step_type}: {e}")

            # Serialize streak candidates (StreakCandidate pydantic models -> dicts)
            streak_cands_serialized = []
            for sc in frame.streak_candidates:
                if hasattr(sc, "model_dump"):
                    streak_cands_serialized.append(sc.model_dump(mode="json"))
                elif isinstance(sc, dict):
                    streak_cands_serialized.append(sc)

            serializable = SiderealFrameSerializable(
                starfield=frame.starfield,
                seeing=frame.seeing,
                hardware=frame.hardware,
                detections=frame.detections,
                original_frame_path=file_path,
                processed_frame_path=processed_file_path,
                index=frame.index,
                timestamp=frame.timestamp.isoformat() if frame.timestamp else None,
                original_frame_header=frame_header,
                processing_history=processing_history,
                correction_frames=correction_frames,
                frame_metadata=frame.frame_metadata.to_serializable() if frame.frame_metadata else None,
                photometry_summary=frame.photometry_summary,
                streak_candidates=streak_cands_serialized,
            )
            sidereal_frames_serializable.append(serializable)

        # Convert rate track frames to serializable versions
        rate_track_frames_serializable = []
        for frame in self.rate_track_frames:
            # Get the file path and convert to absolute path if it's not already
            file_path = None
            if hasattr(frame.frame, "file_path") and frame.frame.file_path:
                file_path = os.path.abspath(frame.frame.file_path)

            # Get the processed file path if available
            processed_file_path = None
            if hasattr(frame.frame, "processed_file_path") and frame.frame.processed_file_path:
                processed_file_path = os.path.abspath(frame.frame.processed_file_path)

            # Convert header to a serializable dictionary
            frame_header = {}
            if hasattr(frame.frame, "header") and frame.frame.header:
                for key, value in frame.frame.header.items():
                    try:
                        # Test if the value is JSON serializable
                        json.dumps({key: value})
                        frame_header[key] = value
                    except (TypeError, OverflowError):
                        # Skip values that aren't JSON serializable
                        continue

            # Convert processing history to serializable format
            processing_history = None
            if hasattr(frame.frame, "processing_history") and frame.frame.processing_history:
                processing_history = []
                for step in frame.frame.processing_history:
                    try:
                        # Convert ProcessingStep enum to string
                        step_dict = {"step_type": step.step_type.value, "parameters": step.parameters}
                        processing_history.append(step_dict)
                    except Exception as e:
                        logger.warning(f"Could not serialize processing step: {e}")

            # Convert correction frames to serializable format
            correction_frames = None
            if hasattr(frame.frame, "correction_frames") and frame.frame.correction_frames:
                correction_frames = {}
                for step_type, correction_data in frame.frame.correction_frames.items():
                    try:
                        # Convert numpy arrays to lists for JSON serialization
                        if isinstance(correction_data, np.ndarray):
                            correction_frames[step_type.value] = correction_data.tolist()
                        else:
                            correction_frames[step_type.value] = correction_data
                    except Exception as e:
                        logger.warning(f"Could not serialize correction frame for {step_type}: {e}")

            # Serialize streak candidates (StreakCandidate pydantic models -> dicts)
            rate_streak_cands_serialized = []
            for sc in frame.streak_candidates:
                if hasattr(sc, "model_dump"):
                    rate_streak_cands_serialized.append(sc.model_dump(mode="json"))
                elif isinstance(sc, dict):
                    rate_streak_cands_serialized.append(sc)

            serializable = RateTrackFrameSerializable(
                starfield=frame.starfield,
                streak=frame.streak,
                seeing=frame.seeing,
                hardware=frame.hardware,
                detections=frame.detections,
                original_frame_path=file_path,
                processed_frame_path=processed_file_path,
                index=frame.index,
                timestamp=frame.timestamp.isoformat() if frame.timestamp else None,
                pixel_track_rate_per_second=frame.pixel_track_rate_per_second,
                original_frame_header=frame_header,
                processing_history=processing_history,
                correction_frames=correction_frames,
                frame_metadata=frame.frame_metadata.to_serializable() if frame.frame_metadata else None,
                photometry_summary=frame.photometry_summary,
                streak_candidates=rate_streak_cands_serialized,
            )
            rate_track_frames_serializable.append(serializable)

        # Create and return the result
        return SenpaiRunResult(
            id=self.id,
            num_frames=self.num_frames,
            collect_metadata=self.collect_metadata,
            compute_seconds=self.compute_seconds,
            frame_shifts=self.frame_shifts,
            frame_shifts_failed=self.frame_shifts_failed,
            sidereal_frames=sidereal_frames_serializable,
            rate_track_frames=rate_track_frames_serializable,
            completed=self.completed,
            senpai_version=self.senpai_version,
            error_message=self.error_message,
            scale_factor=self.scale_factor,
            correlated_streaks=self.correlated_streaks,
        )

    def _build_frame_summary(
        self,
        frame: "SiderealFrame | RateTrackFrame",
        track_mode: str,
    ) -> FrameSummary:
        """Build a FrameSummary from a sidereal or rate-track frame."""
        # File paths
        original_path = None
        if hasattr(frame.frame, "file_path") and frame.frame.file_path:
            original_path = os.path.abspath(frame.frame.file_path)
        processed_path = None
        if hasattr(frame.frame, "processed_file_path") and frame.frame.processed_file_path:
            processed_path = os.path.abspath(frame.frame.processed_file_path)

        # Extract compact fields from starfield
        sf = frame.starfield
        wcs = sf.wcs if sf else None
        wcs_metadata = sf.wcs_metadata if sf else None
        wcs_status = sf.wcs_status.value if sf else None
        limiting_magnitude = sf.limiting_magnitude if sf else None
        distortion_metrics = sf.distortion_metrics if sf else None
        num_catalog = len(sf.catalog_stars) if sf and sf.catalog_stars else None
        num_astro = len(sf.astrometric_fit_stars) if sf and sf.astrometric_fit_stars else None

        # Detections (satellites/asteroids)
        detection_list = frame.detections.detections if frame.detections else []

        # Rate-track specific fields
        pixel_rate = getattr(frame, "pixel_track_rate_per_second", None)
        streak = getattr(frame, "streak", None)

        # Streak candidates (sidereal frames)
        streak_cands = []
        for sc in getattr(frame, "streak_candidates", []):
            if hasattr(sc, "model_dump"):
                streak_cands.append(sc.model_dump(mode="json"))
            elif isinstance(sc, dict):
                streak_cands.append(sc)

        return FrameSummary(
            index=frame.index,
            timestamp=frame.timestamp.isoformat() if frame.timestamp else None,
            track_mode=track_mode,
            original_frame_path=original_path,
            processed_frame_path=processed_path,
            wcs_status=wcs_status,
            wcs=wcs,
            wcs_metadata=wcs_metadata,
            frame_metadata=frame.frame_metadata.to_serializable() if frame.frame_metadata else None,
            seeing=frame.seeing,
            photometry_summary=frame.photometry_summary,
            limiting_magnitude=limiting_magnitude,
            detections=detection_list,
            streak_candidates=streak_cands,
            pixel_track_rate_per_second=pixel_rate,
            streak=streak,
            distortion_metrics=distortion_metrics,
            num_catalog_stars=num_catalog,
            num_astrometric_fit_stars=num_astro,
        )

    def to_summary(self) -> SenpaiRunSummary:
        """Build a compact summary dropping bulk star lists, headers, and calibration data."""
        frames: list[FrameSummary] = []
        for frame in self.sidereal_frames:
            frames.append(self._build_frame_summary(frame, "sidereal"))
        for frame in self.rate_track_frames:
            frames.append(self._build_frame_summary(frame, "rate"))

        # Sort all frames by index for a unified timeline
        frames.sort(key=lambda f: f.index)

        return SenpaiRunSummary(
            id=self.id,
            num_frames=self.num_frames,
            completed=self.completed,
            compute_seconds=self.compute_seconds,
            senpai_version=self.senpai_version,
            error_message=self.error_message,
            scale_factor=self.scale_factor,
            collect_metadata=self.collect_metadata,
            frame_shifts=self.frame_shifts,
            frames=frames,
            correlated_streaks=self.correlated_streaks,
        )
