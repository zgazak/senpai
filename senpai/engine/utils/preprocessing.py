import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)
from astropy.io import fits
from astropy.stats import SigmaClip
from photutils.background import Background2D
from scipy.ndimage import gaussian_filter, zoom

from senpai.engine.models.images import ProcessedFitsImage, ProcessingStep
from senpai.engine.models.metadata import FWHMMetadata
from senpai.engine.models.starfield import StarField
from senpai.engine.utils.darks import apply_dark_subtraction as _apply_dark_subtraction
from senpai.engine.utils.flats import apply_flat_field as _apply_flat_field


def handle_negative_values(image: ProcessedFitsImage) -> ProcessedFitsImage:
    """
    Handle negative values in the image by setting them to the maximum value
    based on the BITPIX header.

    Parameters
    ----------
    image : ProcessedFitsImage
        Image to process

    Returns
    -------
    ProcessedFitsImage
        Image with negative values replaced
    """
    # Get BITPIX from header
    bitpix = image.header.get("BITPIX", 16)  # Default to 16-bit if not found

    # Calculate max value for signed integers
    if bitpix > 0:
        max_value = 2 ** (bitpix - 1) - 1  # For signed integers
    else:
        max_value = 1e10  # For floating point

    # Find negative values
    negative_mask = image.data < 0
    n_negative = np.sum(negative_mask)

    if n_negative > 0:
        print(f"Found {n_negative} negative pixels, setting to max value {max_value} (BITPIX={bitpix})")
        image.data[negative_mask] = max_value

    return image


def remove_column_and_row_medians(image: ProcessedFitsImage, store_intermediates: bool = False) -> ProcessedFitsImage:
    """Remove the median value of each column and row from the image"""
    from senpai.engine.models.images import ProcessingMetadata

    array = image.data

    # Convert to a signed float so the subtraction can go negative. float32
    # on purpose: the whole downstream pipeline (detection convolutions,
    # statistics, photometry) inherits this dtype, and float64 doubles every
    # one of those costs for nothing — the data are ADU-scale (<~1e5), where
    # float32's 24-bit mantissa resolves ~0.005 ADU, far below read noise.
    array = array.astype(np.float32)

    # Subtract column medians (shape: (1, n_cols))
    column_medians = np.median(array, axis=0)[np.newaxis, :]
    array -= column_medians
    col_metadata = ProcessingMetadata(step_type=ProcessingStep.COLUMN_MEDIAN_SUBTRACT, parameters={})
    image.processing_history.append(col_metadata)

    # Subtract row medians (shape: (n_rows, 1))
    row_medians = np.median(array, axis=1)[:, np.newaxis]
    array -= row_medians
    row_metadata = ProcessingMetadata(step_type=ProcessingStep.ROW_MEDIAN_SUBTRACT, parameters={})
    image.processing_history.append(row_metadata)

    if store_intermediates:
        if image.correction_frames is None:
            image.correction_frames = {}
        image.correction_frames[ProcessingStep.COLUMN_MEDIAN_SUBTRACT] = column_medians
        image.correction_frames[ProcessingStep.ROW_MEDIAN_SUBTRACT] = row_medians

        if image.original_data is None:
            # save the original frame if not done
            image.original_data = image.data

    image.data = array

    return image


def remove_background(
    image: ProcessedFitsImage,
    box_size: int = 20,
    filter_size: int = 3,
    exclude_percentile: float = 50.0,
    sigma: float = 3.0,
    maxiters: int = 10,
    store_intermediates: bool = False,
) -> ProcessedFitsImage:
    from senpai.engine.models.images import ProcessingMetadata

    background = measure_background(image.data, box_size, filter_size, exclude_percentile, sigma, maxiters)

    if store_intermediates:
        if image.correction_frames is None:
            image.correction_frames = {}
        image.correction_frames[ProcessingStep.BACKGROUND_SUBTRACT] = background

        if image.original_data is None:
            # save the original frame if not done
            image.original_data = image.data.copy()

    bg_metadata = ProcessingMetadata(
        step_type=ProcessingStep.BACKGROUND_SUBTRACT,
        parameters={
            "box_size": box_size,
            "filter_size": filter_size,
            "exclude_percentile": exclude_percentile,
            "sigma": sigma,
            "maxiters": maxiters,
        },
    )
    image.processing_history.append(bg_metadata)

    # Background2D returns float64; don't let the subtraction silently
    # upcast the frame (the pipeline runs float32).
    image.data = (image.data - background).astype(np.float32, copy=False)
    image.data -= np.min(image.data)

    return image


def measure_background(
    image: np.ndarray,
    box_size: int = 20,
    filter_size: int = 3,
    exclude_percentile: float = 50.0,
    sigma: float = 3.0,
    maxiters: int = 10,
) -> np.ndarray:
    """
    Subtract the 2D background from an image using photutils Background2D.

    Parameters
    ----------
    image : np.ndarray
        Input image array
    box_size : int
        Size of the box in pixels used to calculate the background
    filter_size : int
        Size of the median filter to apply to the background mesh
    exclude_percentile : float
        Percentile to exclude when calculating the background
    sigma : float
        Sigma clipping parameter for identifying outliers
    maxiters : int
        Maximum number of sigma-clipping iterations

    Returns
    -------
    np.ndarray
        Background-subtracted image
    """
    try:
        # Create a SigmaClip object with the specified parameters
        sigma_clip = SigmaClip(sigma=sigma, maxiters=maxiters)

        # Try with the provided parameters
        background = Background2D(
            image,
            box_size=box_size,
            filter_size=filter_size,
            exclude_percentile=exclude_percentile,
            sigma_clip=sigma_clip,
        )
        return background.background
    except ValueError:
        # If the first attempt fails, try with more permissive parameters
        try:
            sigma_clip = SigmaClip(sigma=5.0, maxiters=5)
            background = Background2D(
                image,
                box_size=box_size,
                filter_size=filter_size,
                exclude_percentile=90.0,  # More permissive exclusion
                sigma_clip=sigma_clip,
                fill_value=np.median(image),  # Use median as fallback
            )
            return background.background
        except Exception:
            # If all else fails, fall back to a simple global median subtraction
            print("Warning: Background2D failed, falling back to global median subtraction")
            return np.median(image)


def background_subtract(
    image: np.ndarray,
    box_size: int = 20,
    filter_size: int = 3,
    exclude_percentile: float = 50.0,
    sigma: float = 3.0,
    maxiters: int = 10,
) -> np.ndarray:
    """
    Subtract the 2D background from an image using photutils Background2D.

    Parameters
    ----------
    image : np.ndarray
        Input image array
    box_size : int
        Size of the box in pixels used to calculate the background
    filter_size : int
        Size of the median filter to apply to the background mesh
    exclude_percentile : float
        Percentile to exclude when calculating the background
    sigma : float
        Sigma clipping parameter for identifying outliers
    maxiters : int
        Maximum number of sigma-clipping iterations

    Returns
    -------
    np.ndarray
        Background-subtracted image
    """

    background = measure_background(image, box_size, filter_size, exclude_percentile, sigma, maxiters)
    return image - background


def apply_flat_field(
    image: ProcessedFitsImage,
    master_flat: Union[str, Path, np.ndarray],
    store_intermediates: bool = False,
) -> ProcessedFitsImage:
    """
    Apply flat field correction to a ProcessedFitsImage.

    This is a wrapper around the flat field utilities that integrates with
    the preprocessing workflow.

    Parameters
    ----------
    image : ProcessedFitsImage
        Image to flat field correct
    master_flat : str, Path, or np.ndarray
        Master flat field. If string/Path, will load from file
    store_intermediates : bool
        Whether to store intermediate correction frames

    Returns
    -------
    ProcessedFitsImage
        Flat field corrected image
    """
    return _apply_flat_field(image, master_flat, store_intermediates)


def apply_dark_subtraction(
    image: ProcessedFitsImage,
    master_dark: Union[str, Path, np.ndarray],
    dark_exposure_time: Optional[float] = None,
    store_intermediates: bool = False,
) -> ProcessedFitsImage:
    """
    Apply dark subtraction to a ProcessedFitsImage.

    This is a wrapper around the dark subtraction utilities that integrates with
    the preprocessing workflow.

    Parameters
    ----------
    image : ProcessedFitsImage
        Image to dark subtract
    master_dark : str, Path, or np.ndarray
        Master dark frame. If string/Path, will load from file
    dark_exposure_time : float, optional
        Exposure time of the master dark. If None, will try to read from header
    store_intermediates : bool
        Whether to store intermediate correction frames

    Returns
    -------
    ProcessedFitsImage
        Dark-subtracted image
    """
    return _apply_dark_subtraction(image, master_dark, dark_exposure_time, store_intermediates)


def auto_apply_calibrations(
    image: ProcessedFitsImage,
    config: Optional[object] = None,
    store_intermediates: bool = False,
) -> ProcessedFitsImage:
    """
    Automatically apply calibration frames (flats, darks) based on configuration.

    Parameters
    ----------
    image : ProcessedFitsImage
        Image to calibrate
    config : AppConfig, optional
        Configuration object. If None, will try to get from global config
    store_intermediates : bool
        Whether to store intermediate correction frames

    Returns
    -------
    ProcessedFitsImage
        Calibrated image
    """
    if config is None:
        try:
            from senpai.core.config import get_config

            config = get_config()
        except ImportError:
            print("Warning: Could not import config, skipping auto-calibration")
            return image
        except RuntimeError:
            print("Warning: Config not initialized, skipping auto-calibration")
            return image

    # Check if calibrations config exists
    if not hasattr(config, "calibrations"):
        return image

    cal_config = config.calibrations

    # Helper function to check if a step has already been applied
    def step_already_applied(step_type):
        return any(step.step_type == step_type for step in image.processing_history)

    # Apply darks first if configured
    if cal_config.auto_apply_darks:
        if step_already_applied(ProcessingStep.DARK_SUBTRACT):
            print("Dark subtraction already applied, skipping")
        else:
            # Use intelligent dark matching that considers exposure time
            master_dark_path = _find_best_dark_calibration(
                image, cal_config.master_darks_dir, cal_config.dark_matching_headers, cal_config.max_dark_exposure_ratio
            )
            if master_dark_path:
                print(f"Applying master dark: {master_dark_path}")
                image = apply_dark_subtraction(image, master_dark_path, store_intermediates=store_intermediates)
            else:
                print("Warning: Auto-apply darks enabled but no matching master dark found")

    # Apply preprocessing steps based on configuration
    if cal_config.auto_remove_column_median:
        if step_already_applied(ProcessingStep.COLUMN_MEDIAN_SUBTRACT):
            print("Column median removal already applied, skipping")
        else:
            print("Applying column median removal")
            image = remove_column_and_row_medians(image, store_intermediates)
    elif cal_config.auto_remove_row_median:
        if step_already_applied(ProcessingStep.ROW_MEDIAN_SUBTRACT):
            print("Row median removal already applied, skipping")
        else:
            # If only row median removal is enabled, we need a separate function
            # For now, we'll still call the combined function but note this in the future
            print("Applying row median removal")
            image = remove_column_and_row_medians(image, store_intermediates)

    if cal_config.auto_subtract_background:
        if step_already_applied(ProcessingStep.BACKGROUND_SUBTRACT):
            print("Background subtraction already applied, skipping")
        else:
            print("Applying background subtraction")
            image = remove_background(
                image,
                box_size=cal_config.background_box_size,
                filter_size=cal_config.background_filter_size,
                exclude_percentile=cal_config.background_exclude_percentile,
                sigma=cal_config.background_sigma,
                maxiters=cal_config.background_maxiters,
                store_intermediates=store_intermediates,
            )

    # Apply flats if configured
    if cal_config.auto_apply_flats:
        if step_already_applied(ProcessingStep.FLAT_DIVIDE):
            print("Flat correction already applied, skipping")
        else:
            master_flat_path = _find_master_calibration(
                image, cal_config.master_flats_dir, cal_config.flat_matching_headers, "flat"
            )
            if master_flat_path:
                print(f"Applying master flat: {master_flat_path}")
                image = apply_flat_field(image, master_flat_path, store_intermediates)
            else:
                print("Warning: Auto-apply flats enabled but no matching master flat found")

    return image


def _find_master_calibration(
    image: ProcessedFitsImage, calibration_dir: Optional[str], matching_headers: list[str], calibration_type: str
) -> Optional[Path]:
    """
    Find the appropriate master calibration file for an image by matching FITS headers.

    Parameters
    ----------
    image : ProcessedFitsImage
        Image to find calibration for
    calibration_dir : str, optional
        Directory containing master calibration files
    matching_headers : list[str]
        FITS header keywords that must match between science and calibration frames
    calibration_type : str
        Type of calibration ("flat" or "dark")

    Returns
    -------
    Path or None
        Path to matching master calibration file, or None if not found
    """
    if not calibration_dir:
        return None

    calibration_dir = Path(calibration_dir)
    if not calibration_dir.exists():
        print(f"Warning: {calibration_type} directory does not exist: {calibration_dir}")
        return None

    # Get all FITS files in the calibration directory
    calib_files = list(calibration_dir.glob("*.fits")) + list(calibration_dir.glob("*.fit"))
    if not calib_files:
        print(f"Warning: No FITS files found in {calibration_type} directory: {calibration_dir}")
        return None

    # Extract target metadata from image header
    target_header = image.header
    target_metadata = {}

    for header_key in matching_headers:
        # Try common variations of header keys
        header_variations = [header_key, header_key.upper(), header_key.lower()]
        if header_key.upper() == "BINNING":
            header_variations.extend(["XBINNING", "BIN", "CCDSUM"])
        elif header_key.upper() == "FILTER":
            header_variations.extend(["FILTNAM", "FILTER1"])
        elif header_key.upper() == "EXPTIME":
            header_variations.extend(["EXPOSURE"])

        for variant in header_variations:
            if variant in target_header:
                value = target_header[variant]
                # Normalize string values to lowercase for comparison
                if isinstance(value, str):
                    value = value.lower().strip()
                target_metadata[header_key] = value
                break

        if header_key not in target_metadata:
            print(f"Warning: Required header '{header_key}' not found in science image")
            return None

    print(f"Looking for {calibration_type} with metadata: {target_metadata}")

    # Science frame time, for picking the nearest-in-time calibration when
    # several match (e.g. one master flat per night in a shared dir).
    target_time = None
    try:
        if "DATE-OBS" in target_header:
            from astropy.time import Time

            target_time = Time(target_header["DATE-OBS"]).unix
    except Exception:
        target_time = None

    # Collect every calibration file whose headers match
    candidates: list[tuple[Path, Optional[float], dict]] = []
    for calib_file in calib_files:
        try:
            with fits.open(calib_file) as hdul:
                calib_header = hdul[0].header

                # Check if all required headers match
                matches = True
                calib_metadata = {}

                for header_key in matching_headers:
                    # Try the same header variations for calibration files
                    header_variations = [header_key, header_key.upper(), header_key.lower()]
                    if header_key.upper() == "BINNING":
                        header_variations.extend(["XBINNING", "BIN", "CCDSUM"])
                    elif header_key.upper() == "FILTER":
                        header_variations.extend(["FILTNAM", "FILTER1"])
                    elif header_key.upper() == "EXPTIME":
                        header_variations.extend(["EXPOSURE"])

                    calib_value = None
                    for variant in header_variations:
                        if variant in calib_header:
                            calib_value = calib_header[variant]
                            break

                    if calib_value is None:
                        matches = False
                        break

                    # Normalize for comparison
                    if isinstance(calib_value, str):
                        calib_value = calib_value.lower().strip()

                    calib_metadata[header_key] = calib_value

                    if calib_value != target_metadata[header_key]:
                        matches = False
                        break

                if matches:
                    calib_time = None
                    try:
                        if "DATE-OBS" in calib_header:
                            from astropy.time import Time

                            calib_time = Time(calib_header["DATE-OBS"]).unix
                    except Exception:
                        calib_time = None
                    candidates.append((calib_file, calib_time, calib_metadata))

        except Exception as e:
            print(f"Warning: Could not read {calibration_type} file {calib_file}: {e}")

    if candidates:
        if target_time is not None and len(candidates) > 1:
            # Prefer the calibration taken closest in time; undated ones last.
            candidates.sort(
                key=lambda c: abs(c[1] - target_time) if c[1] is not None else float("inf")
            )
        calib_file, _, calib_metadata = candidates[0]
        print(
            f"Found matching {calibration_type}: {calib_file.name} with metadata: "
            f"{calib_metadata} ({len(candidates)} candidate(s))"
        )
        return calib_file

    print(f"No matching {calibration_type} found for metadata: {target_metadata}")

    # Show available calibration files for debugging
    print(f"Available {calibration_type} files:")
    for calib_file in calib_files[:5]:  # Show first 5 files
        try:
            with fits.open(calib_file) as hdul:
                calib_header = hdul[0].header
                calib_metadata = {}
                for header_key in matching_headers:
                    if header_key in calib_header:
                        value = calib_header[header_key]
                        if isinstance(value, str):
                            value = value.lower().strip()
                        calib_metadata[header_key] = value
                print(f"  {calib_file.name}: {calib_metadata}")
        except:
            print(f"  {calib_file.name}: <could not read headers>")

    return None


def _find_best_dark_calibration(
    image: ProcessedFitsImage, dark_dir: Optional[str], matching_headers: list[str], max_exposure_ratio: float
) -> Optional[Path]:
    """
    Find the best dark calibration file by matching headers and finding closest exposure time.

    Parameters
    ----------
    image : ProcessedFitsImage
        Image to find dark for
    dark_dir : str, optional
        Directory containing master dark files
    matching_headers : list[str]
        FITS header keywords that must match (excluding exposure time)
    max_exposure_ratio : float
        Maximum ratio between image and dark exposure times

    Returns
    -------
    Path or None
        Path to best matching master dark file, or None if not found
    """
    if not dark_dir:
        return None

    dark_dir = Path(dark_dir)
    if not dark_dir.exists():
        print(f"Warning: Dark directory does not exist: {dark_dir}")
        return None

    # Get all FITS files in the dark directory
    dark_files = list(dark_dir.glob("*.fits")) + list(dark_dir.glob("*.fit"))
    if not dark_files:
        print(f"Warning: No FITS files found in dark directory: {dark_dir}")
        return None

    # Get image exposure time
    image_header = image.header
    image_exptime = None

    # Try to get exposure time from image header
    for key in ["EXPTIME", "EXPOSURE", "TELAPSE"]:
        if key in image_header:
            image_exptime = float(image_header[key])
            break

    if image_exptime is None:
        print("Warning: Could not determine image exposure time, using exact header matching")
        return _find_master_calibration(image, dark_dir, matching_headers, "dark")

    # Extract target metadata from image header (excluding exposure time)
    target_metadata = {}
    for header_key in matching_headers:
        # Try common variations of header keys
        header_variations = [header_key, header_key.upper(), header_key.lower()]
        if header_key.upper() == "BINNING":
            header_variations.extend(["XBINNING", "BIN", "CCDSUM"])
        elif header_key.upper() == "FILTER":
            header_variations.extend(["FILTNAM", "FILTER1"])

        for variant in header_variations:
            if variant in image_header:
                value = image_header[variant]
                # Normalize string values to lowercase for comparison
                if isinstance(value, str):
                    value = value.lower().strip()
                target_metadata[header_key] = value
                break

        if header_key not in target_metadata:
            print(f"Warning: Required header '{header_key}' not found in science image")
            return None

    print(f"Looking for dark with metadata: {target_metadata} (image exposure: {image_exptime}s)")

    # Find all darks that match the required headers
    matching_darks = []

    for dark_file in dark_files:
        try:
            with fits.open(dark_file) as hdul:
                dark_header = hdul[0].header

                # Check if all required headers match
                matches = True
                for header_key in matching_headers:
                    # Try the same header variations for dark files
                    header_variations = [header_key, header_key.upper(), header_key.lower()]
                    if header_key.upper() == "BINNING":
                        header_variations.extend(["XBINNING", "BIN", "CCDSUM"])
                    elif header_key.upper() == "FILTER":
                        header_variations.extend(["FILTNAM", "FILTER1"])

                    dark_value = None
                    for variant in header_variations:
                        if variant in dark_header:
                            dark_value = dark_header[variant]
                            break

                    if dark_value is None:
                        matches = False
                        break

                    # Normalize for comparison
                    if isinstance(dark_value, str):
                        dark_value = dark_value.lower().strip()

                    if dark_value != target_metadata[header_key]:
                        matches = False
                        break

                if matches:
                    # Get dark exposure time
                    dark_exptime = None
                    for key in ["EXPTIME", "EXPOSURE", "TELAPSE"]:
                        if key in dark_header:
                            dark_exptime = float(dark_header[key])
                            break

                    if dark_exptime is not None:
                        # Check exposure time ratio
                        ratio = max(image_exptime / dark_exptime, dark_exptime / image_exptime)
                        if ratio <= max_exposure_ratio:
                            matching_darks.append((dark_file, dark_exptime, ratio))

        except Exception as e:
            print(f"Warning: Could not read dark file {dark_file}: {e}")

    if not matching_darks:
        print(f"No matching darks found for metadata: {target_metadata}")
        return None

    # Sort by exposure time ratio (closest match first)
    matching_darks.sort(key=lambda x: x[2])

    best_dark_file, best_dark_exptime, best_ratio = matching_darks[0]

    print(f"Found {len(matching_darks)} matching darks:")
    for dark_file, dark_exptime, ratio in matching_darks[:5]:  # Show first 5
        print(f"  {dark_file.name}: {dark_exptime}s (ratio: {ratio:.2f})")

    print(f"Selected best dark: {best_dark_file.name} ({best_dark_exptime}s, ratio: {best_ratio:.2f})")

    return best_dark_file


def preprocess_image(
    image: ProcessedFitsImage,
    config: Optional[object] = None,
    store_intermediates: bool = False,
) -> ProcessedFitsImage:
    """
    Complete preprocessing pipeline that applies all configured steps.
    """
    import numpy as np

    fname = Path(image.file_path).name if image.file_path else "?"

    def log_stats(stage, arr):
        # Median on a stride-8 subsample: this is a diagnostic log line, and
        # four full-frame medians per frame cost ~1 s each on 66 Mpix.
        logger.info(
            "[%s] %s: min=%.1f, max=%.1f, median=%.1f, mean=%.1f",
            fname, stage, np.min(arr), np.max(arr),
            np.median(arr[::8, ::8]), np.mean(arr),
        )

    log_stats("loaded", image.data)

    if config is None:
        try:
            from senpai.core.config import get_config

            config = get_config()
        except ImportError:
            print("Warning: Could not import config, applying default preprocessing")
            # Apply default preprocessing steps only if not already applied
            if not any(
                step.step_type in [ProcessingStep.ROW_MEDIAN_SUBTRACT, ProcessingStep.COLUMN_MEDIAN_SUBTRACT]
                for step in image.processing_history
            ):
                image = remove_column_and_row_medians(image, store_intermediates)
            if not any(step.step_type == ProcessingStep.BACKGROUND_SUBTRACT for step in image.processing_history):
                image = remove_background(image, store_intermediates=store_intermediates)
            return image
        except RuntimeError:
            print("Warning: Config not initialized, applying default preprocessing")
            # Apply default preprocessing steps only if not already applied
            if not any(
                step.step_type in [ProcessingStep.ROW_MEDIAN_SUBTRACT, ProcessingStep.COLUMN_MEDIAN_SUBTRACT]
                for step in image.processing_history
            ):
                image = remove_column_and_row_medians(image, store_intermediates)
            if not any(step.step_type == ProcessingStep.BACKGROUND_SUBTRACT for step in image.processing_history):
                image = remove_background(image, store_intermediates=store_intermediates)
            return image

    cal_config = config.calibrations

    logger.debug("[%s] Starting preprocessing pipeline", fname)

    # Handle negative values once at the beginning
    image = handle_negative_values(image)

    def step_already_applied(step_type):
        return any(step.step_type == step_type for step in image.processing_history)

    # Step 1: Apply darks first (if configured)
    if cal_config.auto_apply_darks:
        if step_already_applied(ProcessingStep.DARK_SUBTRACT):
            logger.debug("[%s] Dark subtraction already applied, skipping", fname)
        else:
            master_dark_path = _find_best_dark_calibration(
                image, cal_config.master_darks_dir, cal_config.dark_matching_headers, cal_config.max_dark_exposure_ratio
            )
            if master_dark_path:
                logger.info("[%s] Applying master dark: %s", fname, master_dark_path)
                image = apply_dark_subtraction(image, master_dark_path, store_intermediates=store_intermediates)

                # Optionally clip extreme negative values (hot pixel artifacts)
                if hasattr(cal_config, "clip_negative_after_dark") and cal_config.clip_negative_after_dark:
                    negative_threshold = getattr(cal_config, "negative_clip_threshold", -1000)
                    n_clipped = np.sum(image.data < negative_threshold)
                    if n_clipped > 0:
                        logger.info("[%s] Clipping %d pixels below %d ADU", fname, n_clipped, negative_threshold)
                        image.data = np.clip(image.data, negative_threshold, None)

                log_stats("after dark", image.data)
            else:
                logger.debug("[%s] No matching dark frame found", fname)
    else:
        logger.debug("[%s] Dark subtraction disabled", fname)

    # Step 2: Apply flats (if configured)
    if cal_config.auto_apply_flats:
        if step_already_applied(ProcessingStep.FLAT_DIVIDE):
            logger.debug("[%s] Flat correction already applied, skipping", fname)
        else:
            master_flat_path = _find_master_calibration(
                image, cal_config.master_flats_dir, cal_config.flat_matching_headers, "flat"
            )
            if master_flat_path:
                logger.info("[%s] Applying master flat: %s", fname, master_flat_path)
                image = apply_flat_field(image, master_flat_path, store_intermediates)
                log_stats("after flat", image.data)
            else:
                logger.debug("[%s] No matching flat frame found", fname)
    else:
        logger.debug("[%s] Flat correction disabled", fname)

    # Step 3: Row/column median removal (if configured)
    row_median_applied = step_already_applied(ProcessingStep.ROW_MEDIAN_SUBTRACT)
    column_median_applied = step_already_applied(ProcessingStep.COLUMN_MEDIAN_SUBTRACT)

    if cal_config.auto_remove_column_median and cal_config.auto_remove_row_median:
        if row_median_applied and column_median_applied:
            logger.debug("[%s] Row/col median already applied, skipping", fname)
        else:
            image = remove_column_and_row_medians(image, store_intermediates)
            log_stats("after row/col median", image.data)
    elif cal_config.auto_remove_column_median:
        if column_median_applied:
            logger.debug("[%s] Column median already applied, skipping", fname)
        else:
            image = remove_column_and_row_medians(image, store_intermediates)
            log_stats("after col median", image.data)
    elif cal_config.auto_remove_row_median:
        if row_median_applied:
            logger.debug("[%s] Row median already applied, skipping", fname)
        else:
            image = remove_column_and_row_medians(image, store_intermediates)
            log_stats("after row median", image.data)
    else:
        logger.debug("[%s] Row/col median removal disabled", fname)

    # Step 4: Background subtraction (if configured)
    if cal_config.auto_subtract_background:
        if step_already_applied(ProcessingStep.BACKGROUND_SUBTRACT):
            logger.debug("[%s] Background subtraction already applied, skipping", fname)
        else:
            image = remove_background(
                image,
                box_size=cal_config.background_box_size,
                filter_size=cal_config.background_filter_size,
                exclude_percentile=cal_config.background_exclude_percentile,
                sigma=cal_config.background_sigma,
                maxiters=cal_config.background_maxiters,
                store_intermediates=store_intermediates,
            )
            log_stats("after bg subtract", image.data)
    else:
        logger.debug("[%s] Background subtraction disabled", fname)

    return image


def collect_detailed_fwhm_stats(
    starfield: StarField, target_fwhm: float = 3.0, oversample_threshold: float = 4.0
) -> FWHMMetadata:
    """
    Collect comprehensive FWHM statistics from star detections.

    Parameters
    ----------
    starfield : StarField
        Starfield with detected stars
    target_fwhm : float
        Target FWHM for scaling recommendations
    oversample_threshold : float
        FWHM threshold above which image is considered oversampled

    Returns
    -------
    FWHMMetadata
        Comprehensive FWHM statistics
    """
    # Get FWHM measurements from different sources
    fwhm_values = []
    fwhm_vs_position = []
    fwhm_vs_magnitude = []
    fwhm_vs_counts = []

    # From detections (if they have FWHM info)
    for star in starfield.detections:
        if hasattr(star, "fwhm") and star.fwhm is not None:
            fwhm_values.append(star.fwhm)
            fwhm_vs_position.append((star.x, star.y, star.fwhm))
            if star.counts is not None:
                fwhm_vs_counts.append((star.counts, star.fwhm))

    # From catalog stars with measured FWHM
    if starfield.catalog_stars:
        for star in starfield.catalog_stars:
            if hasattr(star, "fwhm") and star.fwhm is not None:
                fwhm_values.append(star.fwhm)
                if star.x is not None and star.y is not None:
                    fwhm_vs_position.append((star.x, star.y, star.fwhm))
                if star.magnitude is not None:
                    fwhm_vs_magnitude.append((star.magnitude, star.fwhm))
                if star.counts is not None:
                    fwhm_vs_counts.append((star.counts, star.fwhm))

    # Fallback to detection metadata if available
    if not fwhm_values and starfield.detection_metadata:
        median_fwhm = starfield.detection_metadata.pixel_fwhm
        # Create approximate statistics
        fwhm_values = [median_fwhm]
        fwhm_vs_position = [(0, 0, median_fwhm)]  # Placeholder

    if not fwhm_values:
        raise ValueError("No FWHM measurements found in starfield")

    fwhm_array = np.array(fwhm_values)

    # Calculate basic statistics
    median_fwhm = np.median(fwhm_array)
    mean_fwhm = np.mean(fwhm_array)
    std_fwhm = np.std(fwhm_array)
    min_fwhm = np.min(fwhm_array)
    max_fwhm = np.max(fwhm_array)

    # Analyze spatial gradient (simple version)
    has_spatial_gradient = False
    spatial_gradient_info = None

    if len(fwhm_vs_position) > 5:
        positions = np.array([(x, y) for x, y, _ in fwhm_vs_position])
        fwhm_vals = np.array([fwhm for _, _, fwhm in fwhm_vs_position])

        # Simple gradient analysis: check if FWHM varies significantly across image
        if positions.shape[0] > 1:
            # Calculate correlation with position
            x_corr = np.corrcoef(positions[:, 0], fwhm_vals)[0, 1]
            y_corr = np.corrcoef(positions[:, 1], fwhm_vals)[0, 1]

            # Consider gradient significant if |correlation| > 0.3
            if abs(x_corr) > 0.3 or abs(y_corr) > 0.3:
                has_spatial_gradient = True
                spatial_gradient_info = {
                    "x_correlation": x_corr,
                    "y_correlation": y_corr,
                    "gradient_strength": max(abs(x_corr), abs(y_corr)),
                }

    # Determine if oversampled and calculate recommended scale factor
    is_oversampled = median_fwhm > oversample_threshold
    recommended_scale_factor = None
    if is_oversampled:
        recommended_scale_factor = median_fwhm / target_fwhm

    return FWHMMetadata(
        n_measurements=len(fwhm_values),
        median_fwhm=median_fwhm,
        mean_fwhm=mean_fwhm,
        std_fwhm=std_fwhm,
        min_fwhm=min_fwhm,
        max_fwhm=max_fwhm,
        fwhm_vs_position=fwhm_vs_position,
        fwhm_vs_magnitude=fwhm_vs_magnitude,
        fwhm_vs_counts=fwhm_vs_counts,
        has_spatial_gradient=has_spatial_gradient,
        spatial_gradient_info=spatial_gradient_info,
        is_oversampled=is_oversampled,
        recommended_scale_factor=recommended_scale_factor,
    )


def scale_image_block_median(image: ProcessedFitsImage, scale_factor: float) -> ProcessedFitsImage:
    """
    Scale image using block median method (fast + removes hot pixels).

    Parameters
    ----------
    image : ProcessedFitsImage
        Input image to scale
    scale_factor : float
        Factor to scale by (>1 means downsample)

    Returns
    -------
    ProcessedFitsImage
        Scaled image
    """
    from senpai.engine.models.images import ProcessingMetadata

    # Calculate block size first
    block_size = int(np.ceil(scale_factor))

    # Calculate padding needed to make dimensions divisible by block_size
    pad_y = (block_size - image.data.shape[0] % block_size) % block_size
    pad_x = (block_size - image.data.shape[1] % block_size) % block_size

    # Pad image to ensure clean block divisions
    padded_data = np.pad(image.data, ((0, pad_y), (0, pad_x)), mode="edge")

    # Calculate target dimensions based on padded size
    padded_height = padded_data.shape[0]
    padded_width = padded_data.shape[1]

    # Ensure dimensions are exactly divisible by block_size
    blocks_y = padded_height // block_size
    blocks_x = padded_width // block_size

    # Calculate final target dimensions
    target_height = int(image.data.shape[0] / scale_factor)
    target_width = int(image.data.shape[1] / scale_factor)

    # Reshape into blocks and take median of each block
    reshaped = padded_data.reshape(blocks_y, block_size, blocks_x, block_size)
    scaled_data = np.median(reshaped, axis=(1, 3))

    # Ensure exact target dimensions using array slicing
    if scaled_data.shape[0] > target_height:
        scaled_data = scaled_data[:target_height, :]
    if scaled_data.shape[1] > target_width:
        scaled_data = scaled_data[:, :target_width]

    # Update metadata with new dimensions
    new_metadata = image.metadata.model_copy()
    new_metadata.width = target_width
    new_metadata.height = target_height

    # Create new image with scaled data
    scaled_image = ProcessedFitsImage(
        data=scaled_data,
        header=image.header.copy(),
        data_type=image.data_type,
        metadata=new_metadata,
        file_path=image.file_path,
        original_data=image.original_data,
        correction_frames=image.correction_frames.copy() if image.correction_frames else {},
        processing_history=image.processing_history.copy(),
    )

    # Add processing step with metadata
    scaling_metadata = ProcessingMetadata(
        step_type=ProcessingStep.BACKGROUND_SUBTRACT,  # Reuse existing step
        parameters={"method": "block_median", "scale_factor": scale_factor},
    )
    scaled_image.processing_history.append(scaling_metadata)

    return scaled_image


def scale_image_blur_decimate(image: ProcessedFitsImage, scale_factor: float) -> ProcessedFitsImage:
    """
    Scale image using Gaussian blur + decimation (better photometry).

    Parameters
    ----------
    image : ProcessedFitsImage
        Input image to scale
    scale_factor : float
        Factor to scale by (>1 means downsample)

    Returns
    -------
    ProcessedFitsImage
        Scaled image
    """
    from senpai.engine.models.images import ProcessingMetadata

    # Calculate target dimensions first
    target_height = int(image.data.shape[0] / scale_factor)
    target_width = int(image.data.shape[1] / scale_factor)
    output_shape = (target_height, target_width)

    # Apply Gaussian blur to avoid aliasing
    # Sigma should be ~scale_factor/2 to prevent aliasing
    sigma = scale_factor / 2.0
    blurred_data = gaussian_filter(image.data, sigma=sigma)

    # Decimate using scipy.ndimage.zoom with exact output size
    scaled_data = zoom(
        blurred_data, (target_height / image.data.shape[0], target_width / image.data.shape[1]), order=1, mode="nearest"
    )

    # Update metadata with new dimensions
    new_metadata = image.metadata.model_copy()
    new_metadata.width = target_width
    new_metadata.height = target_height

    # Create new image with scaled data
    scaled_image = ProcessedFitsImage(
        data=scaled_data,
        header=image.header.copy(),
        data_type=image.data_type,
        metadata=new_metadata,
        file_path=image.file_path,
        original_data=image.original_data,
        correction_frames=image.correction_frames.copy() if image.correction_frames else {},
        processing_history=image.processing_history.copy(),
    )

    # Add processing step with metadata
    scaling_metadata = ProcessingMetadata(
        step_type=ProcessingStep.BACKGROUND_SUBTRACT,  # Reuse existing step
        parameters={"method": "blur_decimate", "scale_factor": scale_factor, "sigma": sigma},
    )
    scaled_image.processing_history.append(scaling_metadata)

    return scaled_image


def scale_image_to_target_fwhm(
    image: ProcessedFitsImage,
    fwhm_stats: FWHMMetadata,
    target_fwhm: float = 3.0,
    method: str = "block_median",
    oversample_threshold: float = 4.0,
) -> tuple[ProcessedFitsImage, float]:
    """
    Scale image to achieve target FWHM using specified method.

    Parameters
    ----------
    image : ProcessedFitsImage
        Input image to scale
    fwhm_stats : FWHMMetadata
        FWHM statistics for the image
    target_fwhm : float
        Target FWHM in pixels
    method : str
        Scaling method: "block_median" or "blur_decimate"
    oversample_threshold : float
        Only scale if median FWHM > this threshold

    Returns
    -------
    tuple[ProcessedFitsImage, float]
        (scaled_image, scale_factor_used)
    """
    if fwhm_stats.median_fwhm <= oversample_threshold:
        print(f"FWHM {fwhm_stats.median_fwhm:.1f} <= {oversample_threshold}, no scaling needed")
        return image, 1.0

    scale_factor = fwhm_stats.median_fwhm / target_fwhm

    print(f"Scaling image: {fwhm_stats.median_fwhm:.1f} -> {target_fwhm:.1f} pixels FWHM (factor: {scale_factor:.2f})")
    print(f"Method: {method}, {fwhm_stats.n_measurements} FWHM measurements")

    if method == "block_median":
        scaled_image = scale_image_block_median(image, scale_factor)
    elif method == "blur_decimate":
        scaled_image = scale_image_blur_decimate(image, scale_factor)
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    return scaled_image, scale_factor


def scale_starfield_coordinates(starfield: StarField, scale_factor: float) -> StarField:
    """
    Update starfield coordinates after image scaling.

    Parameters
    ----------
    starfield : StarField
        Starfield to update
    scale_factor : float
        Scale factor that was applied to the image

    Returns
    -------
    StarField
        Updated starfield with scaled coordinates
    """
    # Scale detection coordinates
    for star in starfield.detections:
        star.x /= scale_factor
        star.y /= scale_factor

    # Scale catalog star coordinates if they exist
    if starfield.catalog_stars:
        for star in starfield.catalog_stars:
            if star.x is not None:
                star.x /= scale_factor
            if star.y is not None:
                star.y /= scale_factor

    # Scale astrometric fit star coordinates if they exist
    if starfield.astrometric_fit_stars:
        for star in starfield.astrometric_fit_stars:
            if star.x is not None:
                star.x /= scale_factor
            if star.y is not None:
                star.y /= scale_factor

    # Update image metadata dimensions
    starfield.image_metadata.width = int(starfield.image_metadata.width / scale_factor)
    starfield.image_metadata.height = int(starfield.image_metadata.height / scale_factor)

    # Update detection metadata (FWHM should be scaled)
    if starfield.detection_metadata:
        starfield.detection_metadata.pixel_fwhm /= scale_factor

    # Update WCS if it exists
    if starfield.wcs:
        # Scale the PC matrix elements (this model uses PC + CDELT, not CD)
        starfield.wcs.PC1_1 *= scale_factor
        starfield.wcs.PC1_2 *= scale_factor
        starfield.wcs.PC2_1 *= scale_factor
        starfield.wcs.PC2_2 *= scale_factor

        # Scale the reference pixel
        starfield.wcs.CRPIX1 /= scale_factor
        starfield.wcs.CRPIX2 /= scale_factor

    # Update WCS metadata if it exists
    if starfield.wcs_metadata:
        starfield.wcs_metadata.x_ifov_arcsec *= scale_factor
        starfield.wcs_metadata.y_ifov_arcsec *= scale_factor

    # Store the scale factor
    starfield.scale_factor = scale_factor

    return starfield


def unscale_starfield_coordinates(
    starfield: StarField, scale_factor: float, scaling_method: str = "block_median"
) -> StarField:
    """
    Unscale starfield coordinates back to original image size.

    This is the reverse of scale_starfield_coordinates, accounting for the specific scaling method used.

    Parameters
    ----------
    starfield : StarField
        Starfield to unscale
    scale_factor : float
        Scale factor that was applied to the image
    scaling_method : str
        Scaling method that was used ("block_median" or "simple_integer")

    Returns
    -------
    StarField
        Updated starfield with unscaled coordinates
    """
    if scale_factor == 1.0:
        return starfield

    # For simple integer scaling or median filter, just multiply coordinates
    if scaling_method in ["simple_integer", "median_filter"]:
        # Scale detection coordinates
        for star in starfield.detections:
            star.x *= scale_factor
            star.y *= scale_factor

        # Scale catalog star coordinates if they exist
        if starfield.catalog_stars:
            for star in starfield.catalog_stars:
                if star.x is not None:
                    star.x *= scale_factor
                if star.y is not None:
                    star.y *= scale_factor

        # Scale astrometric fit star coordinates if they exist
        if starfield.astrometric_fit_stars:
            for star in starfield.astrometric_fit_stars:
                if star.x is not None:
                    star.x *= scale_factor
                if star.y is not None:
                    star.y *= scale_factor

        # Update image metadata dimensions
        starfield.image_metadata.width = int(starfield.image_metadata.width * scale_factor)
        starfield.image_metadata.height = int(starfield.image_metadata.height * scale_factor)

        # Update detection metadata (FWHM should be scaled)
        if starfield.detection_metadata:
            starfield.detection_metadata.pixel_fwhm *= scale_factor

        # Update WCS if it exists
        if starfield.wcs:
            # Scale the PC matrix elements back down (reverse of scaling)
            starfield.wcs.PC1_1 /= scale_factor
            starfield.wcs.PC1_2 /= scale_factor
            starfield.wcs.PC2_1 /= scale_factor
            starfield.wcs.PC2_2 /= scale_factor

            # Scale the reference pixel back up
            starfield.wcs.CRPIX1 *= scale_factor
            starfield.wcs.CRPIX2 *= scale_factor

        # Update WCS metadata if it exists
        if starfield.wcs_metadata:
            starfield.wcs_metadata.x_ifov_arcsec /= scale_factor
            starfield.wcs_metadata.y_ifov_arcsec /= scale_factor

        # Set scale_factor to 1.0 since this is now the "original" scale
        starfield.scale_factor = 1.0

    elif scaling_method == "block_median":
        # For block median scaling, we need to account for trimming
        # The trimming offset depends on the original image size and scale factor
        # This is more complex and requires the original image dimensions

        # For now, use simple scaling but warn that it might not be exact
        import warnings

        warnings.warn(
            "Block median unscaling may not be exact due to trimming. Consider using simple_integer scaling for precise coordinate mapping."
        )

        # Use the same logic as simple_integer for now
        for star in starfield.detections:
            star.x *= scale_factor
            star.y *= scale_factor

        if starfield.catalog_stars:
            for star in starfield.catalog_stars:
                if star.x is not None:
                    star.x *= scale_factor
                if star.y is not None:
                    star.y *= scale_factor

        if starfield.astrometric_fit_stars:
            for star in starfield.astrometric_fit_stars:
                if star.x is not None:
                    star.x *= scale_factor
                if star.y is not None:
                    star.y *= scale_factor

        starfield.image_metadata.width = int(starfield.image_metadata.width * scale_factor)
        starfield.image_metadata.height = int(starfield.image_metadata.height * scale_factor)

        if starfield.detection_metadata:
            starfield.detection_metadata.pixel_fwhm *= scale_factor

        if starfield.wcs:
            starfield.wcs.PC1_1 /= scale_factor
            starfield.wcs.PC1_2 /= scale_factor
            starfield.wcs.PC2_1 /= scale_factor
            starfield.wcs.PC2_2 /= scale_factor
            starfield.wcs.CRPIX1 *= scale_factor
            starfield.wcs.CRPIX2 *= scale_factor

        if starfield.wcs_metadata:
            starfield.wcs_metadata.x_ifov_arcsec /= scale_factor
            starfield.wcs_metadata.y_ifov_arcsec /= scale_factor

        starfield.scale_factor = 1.0

    else:
        raise ValueError(f"Unsupported scaling method for unscaling: {scaling_method}")

    return starfield


def unscale_streak_metadata(streak, scale_factor: float, scaling_method: str = "block_median"):
    """
    Unscale streak metadata back to original image size.

    Parameters
    ----------
    streak : StreakMetadata
        Streak metadata to unscale
    scale_factor : float
        Scale factor that was applied to the image
    scaling_method : str
        Scaling method that was used

    Returns
    -------
    StreakMetadata
        Updated streak metadata with unscaled values
    """
    if streak is None or scale_factor == 1.0:
        return streak

    # Import here to avoid circular imports
    from copy import deepcopy

    # Create a deep copy to avoid modifying the original
    unscaled_streak = deepcopy(streak)

    # Scale the pixel length
    if hasattr(unscaled_streak, "pixel_length"):
        unscaled_streak.pixel_length *= scale_factor

    return unscaled_streak


def apply_fwhm_optimization(
    image: ProcessedFitsImage, starfield: StarField, config: Optional[object] = None
) -> tuple[ProcessedFitsImage, StarField]:
    """
    Apply FWHM-based optimization after WCS fitting.

    This function:
    1. Collects detailed FWHM statistics
    2. Optionally scales the image to optimize FWHM
    3. Updates starfield coordinates accordingly

    Usage Example:
        # After WCS fitting in the main pipeline:
        if sidereal_frame.starfield and sidereal_frame.starfield.fit:
            # Apply FWHM optimization
            optimized_image, updated_starfield = apply_fwhm_optimization(
                sidereal_frame.frame,
                sidereal_frame.starfield,
                config
            )

            # Update frame with optimized data
            sidereal_frame.frame = optimized_image
            sidereal_frame.starfield = updated_starfield

            # Log scaling results
            if updated_starfield.scale_factor and updated_starfield.scale_factor > 1.0:
                logger.info(f"Image scaled by factor {updated_starfield.scale_factor:.2f} "
                           f"(FWHM: {updated_starfield.fwhm_stats.median_fwhm:.1f} → "
                           f"{updated_starfield.fwhm_stats.median_fwhm / updated_starfield.scale_factor:.1f} pixels)")

    Parameters
    ----------
    image : ProcessedFitsImage
        Processed image with starfield
    starfield : StarField
        Starfield with WCS solution and detections
    config : AppConfig, optional
        Configuration object

    Returns
    -------
    tuple[ProcessedFitsImage, StarField]
        (optimized_image, updated_starfield)
    """
    if config is None:
        try:
            from senpai.core.config import get_config

            config = get_config()
        except (ImportError, RuntimeError):
            print("Warning: Config not available, skipping FWHM optimization")
            return image, starfield

    # Check if calibrations config exists
    if not hasattr(config, "calibrations"):
        print("No calibrations config found, skipping FWHM optimization")
        return image, starfield

    cal_config = config.calibrations

    fwhm_stats = starfield.fwhm_stats

    # Step 2: Apply scaling if enabled and beneficial
    if cal_config.auto_scale_images and fwhm_stats.is_oversampled:
        try:
            print("Applying image scaling optimization...")
            scaled_image, scale_factor = scale_image_to_target_fwhm(
                image,
                fwhm_stats,
                target_fwhm=cal_config.target_fwhm,
                method=cal_config.scaling_method,
                oversample_threshold=cal_config.oversample_threshold,
            )

            if scale_factor > 1.0:
                # Update starfield coordinates
                print("Updating starfield coordinates for scaled image...")
                updated_starfield = scale_starfield_coordinates(starfield, scale_factor)

                print(
                    f"Image optimized: {image.data.shape} -> {scaled_image.data.shape} "
                    f"(scale factor: {scale_factor:.2f})"
                )

                return scaled_image, updated_starfield
            else:
                print("No scaling applied - image already optimal")
                return image, starfield

        except Exception as e:
            print(f"Warning: Image scaling failed: {e}")
            return image, starfield
    else:
        if not cal_config.auto_scale_images:
            print("Auto-scaling disabled in configuration")
        elif not fwhm_stats.is_oversampled:
            print(f"Image not oversampled (FWHM={fwhm_stats.median_fwhm:.1f} <= {cal_config.oversample_threshold})")

        return image, starfield
