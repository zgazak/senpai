import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import arrow
from astropy.io import fits

from senpai.engine.constants import DATE_HEADERS, DATE_TIME_HEADERS, TIME_HEADERS

logger = logging.getLogger(__name__)


def _parse_date_string(date_str: str) -> arrow.Arrow:
    """
    Parse a date string, trying arrow's default formats first,
    then falling back to MM/DD/YY and MM/DD/YYYY formats.
    """
    try:
        # First try arrow's default parsing (handles many formats)
        return arrow.get(date_str)
    except Exception:
        # If default parsing fails, try MM/DD/YY and MM/DD/YYYY formats
        try:
            return arrow.get(date_str, "M/D/YY")
        except Exception:
            try:
                return arrow.get(date_str, "MM/DD/YY")
            except Exception:
                try:
                    return arrow.get(date_str, "M/D/YYYY")
                except Exception:
                    return arrow.get(date_str, "MM/DD/YYYY")

    return None


def _parse_time_string(time_str: str) -> tuple[int, int, int, int] | None:
    """
    Parse a time string and return (hour, minute, second, microsecond) tuple.
    Returns None if parsing fails.
    """
    from datetime import datetime

    # Try parsing with datetime first
    time_formats = [
        "%H:%M:%S.%f",  # 21:36:28.604000
        "%H:%M:%S",  # 21:36:28
        "%H:%M:%S.%f000",  # sometimes microseconds are padded
    ]

    for fmt in time_formats:
        try:
            parsed_time = datetime.strptime(time_str, fmt).time()
            return (parsed_time.hour, parsed_time.minute, parsed_time.second, parsed_time.microsecond)
        except ValueError:
            continue

    # Fallback: try arrow's natural parsing and extract time components
    try:
        arrow_time = arrow.get(time_str)
        return (arrow_time.hour, arrow_time.minute, arrow_time.second, arrow_time.microsecond)
    except Exception:
        pass

    return None


def extract_uct_time_from_header(header: dict[str, Any]) -> datetime:
    for header_key in DATE_TIME_HEADERS:
        if header_key in header:
            try:
                arrow_time = _parse_date_string(str(header[header_key]))
                return arrow_time.datetime
            except Exception as e:
                logger.error(f"failed to parse time from {header_key}: {e}")
                continue

    arrow_date = None
    time_components = None
    for header_key in DATE_HEADERS:
        if header_key in header:
            try:
                arrow_date = _parse_date_string(str(header[header_key]))
            except Exception as e:
                logger.error(f"failed to parse time from {header_key}: {e}")
                continue

    if arrow_date is not None:
        for header_key in TIME_HEADERS:
            if header_key in header:
                try:
                    time_components = _parse_time_string(str(header[header_key]))
                    break
                except Exception as e:
                    logger.error(f"failed to parse time from {header_key}: {e}")
                    continue

    # If we have both date and time, combine them
    if arrow_date is not None and time_components is not None:
        hour, minute, second, microsecond = time_components
        combined_datetime = arrow_date.replace(hour=hour, minute=minute, second=second, microsecond=microsecond)
        return combined_datetime.datetime

    # Debug-level: callers that can tolerate a missing time catch this raise and
    # log a clean, actionable warning themselves (see organize_senpai_frames /
    # extract_observation_time_from_header). Logging ERROR here made a handled,
    # expected condition look like a failure (6 lines x 2 calls per frame).
    logger.debug("no valid date header found in header")
    logger.debug(f"available header: {', '.join(list(header.keys()))}")
    logger.debug(f"coded DATE_TIME headers: {', '.join(DATE_TIME_HEADERS)}")
    logger.debug(f"coded DATE headers: {', '.join(DATE_HEADERS)}")
    logger.debug(f"coded TIME headers: {', '.join(TIME_HEADERS)}")
    logger.debug("YOU MUST HAVE A DATE_TIME or a DATE and a TIME")
    raise AttributeError(f"no valid date header found in {header}")


def get_imageset_by_filename(data_directory: Path, string_match: str) -> list[str]:
    # Get all .fits files in directory that match the regex pattern
    fits_files = [str(f) for f in data_directory.glob("**/*.fits") if string_match in f.name]

    if not fits_files:
        logger.warning(f"No .fits files found matching '{string_match}'*.fits in {data_directory}")

    return sorted(fits_files)


def get_all_images_in_directory(data_directory: Path) -> list[str]:
    # Get all .fits files in directory and subdirectories
    fits_files = [str(f) for f in data_directory.glob("**/*.fits")]

    if not fits_files:
        logger.warning(f"No .fits files found in {data_directory}")

    return sorted(fits_files)


def extract_id_from_header(file: Path, header_key: str) -> str:
    with fits.open(file) as hdul:
        header = hdul[0].header

    if header_key not in header:
        logger.warning(f"header key {header_key} not found in {file}")
        return None

    if header_key == "ORCHCOMM":
        # ORCHCOMM looks something like this: &IMAGESETID@[ukr]#[1:6]%[OPEN]
        return header["ORCHCOMM"].split("&")[1].split("@")[0]

    return header[header_key]


def header_key_matches(file: Path, header_key: str, value: str) -> bool:
    with fits.open(file) as hdul:
        header = hdul[0].header

    if header_key not in header:
        return False

    if header_key == "ORCHCOMM":
        # ORCHCOMM looks something like this: &IMAGESETID@[ukr]#[1:6]%[OPEN]
        return value in header["ORCHCOMM"].split("&")[1].split("@")[0]

    return header[header_key] == value


def get_imageset_by_id(data_directory: Path, id: str, header_id_key: str) -> list[str]:
    # get all fits files in a directory that have the same value for the header_id_key
    fits_files = [str(f) for f in data_directory.glob("**/*.fits") if header_key_matches(f, header_id_key, id)]

    if not fits_files:
        logger.warning(f"No .fits files found with ID {id} in {data_directory}")

    return sorted(fits_files)
