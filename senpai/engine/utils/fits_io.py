import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import arrow
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.io.fits import Header
from astropy.io.fits import open as fits_open
from astropy.time import Time

from senpai.core.config import get_config, initialize_config
from senpai.core.constants import LOCAL_APP_CONFIG_OVERRIDE
from senpai.core.logging import set_log_level
from senpai.engine.models.metadata import SiteMetadata, TrackMode
from senpai.engine.utils.frame_organization import extract_uct_time_from_header

logger = logging.getLogger(__name__)


def sexagesimal_to_decimal(value: str, units: str = "degrees") -> float:
    """
    Convert sexagesimal coordinates to decimal degrees.

    Args:
        value: String in sexagesimal format (e.g., "+20 44 48.24" or "14 15 39.7")
        units: If provided, converts to the specified units

    Returns:
        float: Decimal degrees
    """
    # Clean up the input string
    value = value.strip()

    # Determine sign
    sign = 1
    if value.startswith(("-", "+")):
        sign = -1 if value.startswith("-") else 1
        value = value[1:].strip()

    # Replace any delimiters with spaces
    for char in "hdm°:'\"":
        value = value.replace(char, " ")

    # Split into components
    parts = [p for p in value.split() if p]

    # Handle different formats
    if len(parts) == 1:
        # Already decimal format
        return sign * float(parts[0])

    elif len(parts) == 2:
        # Degrees/Hours and Minutes format
        degrees = float(parts[0])
        minutes = float(parts[1])
        decimal = sign * (degrees + minutes / 60.0)

    elif len(parts) >= 3:
        # Degrees/Hours, Minutes, and Seconds format
        degrees = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        decimal = sign * (degrees + minutes / 60.0 + seconds / 3600.0)

    else:
        raise ValueError(f"Could not parse sexagesimal value: {value}")

    # Multiply by 15 if it's RA format (HMS to degrees)
    if units == "hours":
        decimal *= 15.0
    elif units == "degrees":
        pass
    else:
        raise ValueError(f"Unsupported units: {units}")

    return decimal


def extract_header_value(header: Header, key: str) -> Any:
    if key in header:
        return header[key]
    return None


def float_nsew_to_decimal(value: str) -> float:
    # Handle coordinates with cardinal directions (N, S, E, W)
    value = value.strip()
    cardinal = None

    # Extract the cardinal direction if present
    for direction in ["N", "S", "E", "W"]:
        if direction in value:
            cardinal = direction
            value = value.replace(direction, "").strip()
            break

    # Convert to float
    decimal = float(value)

    # Apply sign based on cardinal direction
    if cardinal in ["S", "W"]:
        decimal = -decimal

    return decimal


def convert_to_decimal_degrees_unknown_format(value: str, units: str = None) -> float:
    for converter in [sexagesimal_to_decimal, float, float_nsew_to_decimal]:
        try:
            value = converter(value)
            break
        except ValueError:
            pass

    if value is not None:
        if units == "degrees":
            return value
        elif units == "hours":
            return value * 15.0
        else:
            raise ValueError(f"Unsupported units: {units}")

    raise ValueError(f"Could not convert value to decimal degrees: {value}")


def convert_to_decimal_kilometers(value: str, units: str = None) -> float:
    if units == "kilometers":
        return float(value)
    elif units == "meters":
        return float(value) / 1000.0
    else:
        logger.warning(f"value: {value}.  Is this meters or kilometers?")
        raise ValueError(f"Unsupported units: {units}")


def convert_to_decimal_degrees(value, fmt: str = None, units: str = None) -> float:
    if fmt == "sexagesimal":
        return sexagesimal_to_decimal(str(value), units)
    elif fmt == "float":
        if units == "degrees":
            return float(value)
        elif units == "hours":
            return float(value) * 15.0
        else:
            raise ValueError(f"Unsupported units: {units}")
    elif fmt == "float NSEW":
        return float_nsew_to_decimal(str(value))

    else:
        raise ValueError(f"Unsupported format: {fmt}")


_CLEAR_FILTER_ALIASES = {"", "open", "l", "lum", "luminance", "clear", "none"}


def extract_filter_from_header(header: Header) -> str | None:
    """Extract observation filter from FITS header.

    Tries configured header keys and normalizes common clear/open filter
    values to "Clear".

    Returns
    -------
    str or None
        Normalized filter name, or None if not found.
    """
    config = get_config()
    for key in config.headers.filter_keys:
        value = extract_header_value(header, key)
        if value is not None:
            value_str = str(value).strip()
            if value_str.lower() in _CLEAR_FILTER_ALIASES:
                return "Clear"
            return value_str
    return None


def extract_observation_time_from_header(header: Header) -> datetime | None:
    config = get_config()
    for key in config.headers.observation_time.observation_time_keys:
        observation_time = extract_header_value(header, key)
        if observation_time is not None:
            try:
                if config.headers.observation_time.format == "iso":
                    observation_time = datetime.fromisoformat(observation_time)
                else:
                    observation_time = datetime.strptime(observation_time, config.headers.observation_time.format)
            except ValueError:
                observation_time = arrow.get(observation_time).datetime

            return observation_time

    # Last-ditch broad attempt. Returns None (rather than raising) when no date
    # header is present, so a header-sparse frame degrades gracefully instead of
    # crashing the run.
    try:
        return extract_uct_time_from_header(header)
    except AttributeError:
        logger.warning("No observation-time header found; observation time unavailable")
        return None


def extract_exposure_time_from_header(header: Header) -> float | None:
    config = get_config()
    for key in config.headers.exposure_time.exposure_time_keys:
        exposure_time = extract_header_value(header, key)
        if exposure_time is not None:
            exposure_time = float(exposure_time)
            logger.debug(f"Extracted exposure time from {key}: {exposure_time}")
            return exposure_time

    return None


def extract_observing_site_from_header(header: Header) -> SiteMetadata | None:
    config = get_config()

    latitude = None
    longitude = None
    altitude = None

    for key in config.headers.site.site_latitude_keys:
        latitude = extract_header_value(header, key)
        if latitude is not None:
            latitude = convert_to_decimal_degrees(
                latitude, fmt=config.headers.site.positional_format, units=config.headers.site.positional_unit
            )
            logger.debug(f"Extracted latitude from {key}: {latitude}")
            break

    for key in config.headers.site.site_longitude_keys:
        longitude = extract_header_value(header, key)

        if longitude is not None:
            longitude = convert_to_decimal_degrees(
                longitude, fmt=config.headers.site.positional_format, units=config.headers.site.positional_unit
            )
            logger.debug(f"Extracted longitude from {key}: {longitude}")
            break

    for key in config.headers.site.site_altitude_keys:
        altitude = extract_header_value(header, key)
        if altitude is not None:
            altitude = convert_to_decimal_kilometers(altitude, units=config.headers.site.altitude_unit)
            logger.debug(f"Extracted altitude from {key}: {altitude}")
            break

    if latitude and longitude:
        try:
            site = SiteMetadata(latitude=latitude, longitude=longitude, altitude_km=altitude)
        except Exception as e:
            logger.warning(f"could not extract site details! {e.__str__}")
            site = None
        return site
    else:
        logger.warning("Could not extract observing site from header")
        return None


def extract_boresight_from_header(header: Header) -> tuple[float, float]:
    config = get_config()

    ra = None
    dec = None
    azimuth = None
    altitude = None

    for key in config.headers.pointing.target_ra_keys:
        ra = extract_header_value(header, key)
        if ra is not None:
            ra = convert_to_decimal_degrees(
                ra,
                fmt=config.headers.pointing.ra_dec_format,
                units=config.headers.pointing.ra_units,
            )
            logger.debug(f"Extracted RA from {key}: {ra}")
            break

    if ra is not None:
        for key in config.headers.pointing.target_dec_keys:
            dec = extract_header_value(header, key)
            if dec is not None:
                dec = convert_to_decimal_degrees(
                    dec,
                    fmt=config.headers.pointing.ra_dec_format,
                    units=config.headers.pointing.dec_units,
                )
                logger.debug(f"Extracted DEC from {key}: {dec}")
                break

    if ra and dec:
        return ra, dec

    for key in config.headers.pointing.boresight_azimuth_keys:
        azimuth = extract_header_value(header, key)
        if azimuth is not None:
            azimuth = convert_to_decimal_degrees(azimuth, fmt="float", units="degrees")
            logger.debug(f"Extracted azimuth from {key}: {azimuth}")
            break

    if azimuth is not None:
        for key in config.headers.pointing.boresight_altitude_keys:
            altitude = extract_header_value(header, key)
            if altitude is not None:
                altitude = convert_to_decimal_degrees(altitude, fmt="float", units="degrees")
                logger.debug(f"Extracted altitude from {key}: {altitude}")
                break

    if azimuth and altitude:
        logger.debug("Extracted boresight azimuth, altitude, converting to RA, Dec")
    else:
        logger.warning("Could not extract boresight from header")
        return None, None

    logger.debug("Alt/Az -> RA/Dec requires observation time and site information")
    observation_time = extract_observation_time_from_header(header)

    if observation_time is not None:
        logger.debug("Alt/Az -> RA/Dec observation_time: %s", observation_time.isoformat())
    else:
        logger.warning("Could not extract observation time from header")
        return None, None

    site = extract_observing_site_from_header(header)
    if site is not None:
        logger.debug("Alt/Az -> RA/Dec site: %s", site)
    else:
        logger.warning("Could not extract site from header")
        return None, None

    # Convert Alt/Az to RA/Dec using astropy

    # Create EarthLocation object from site metadata
    location = EarthLocation(
        lat=site.latitude * u.deg,
        lon=site.longitude * u.deg,
        height=site.altitude_km * u.km if site.altitude_km is not None else 0 * u.m,
    )

    # Create Time object from observation time
    obstime = Time(observation_time)

    # Create AltAz coordinate
    altaz = AltAz(alt=altitude * u.deg, az=azimuth * u.deg, obstime=obstime, location=location)

    # Convert to RA/Dec
    radec = SkyCoord(altaz).icrs

    ra_deg = radec.ra.deg
    dec_deg = radec.dec.deg

    logger.debug(f"Converted Alt/Az to RA/Dec: {ra_deg}, {dec_deg}")

    return ra_deg, dec_deg


_RATE_UNIT_TO_ARCSEC_PER_SEC: dict[str, float] = {
    "arcseconds/second": 1.0,
    "arcsec/second": 1.0,
    "arcsec/s": 1.0,
    "degrees/second": 3600.0,
    "deg/second": 3600.0,
    "deg/s": 3600.0,
    "radians/second": 206264.80624709636,
    "rad/s": 206264.80624709636,
}


def _to_arcsec_per_second(value: float, unit: str) -> float:
    """Normalize a track-rate value to arcseconds/second using the unit string
    declared in ``config.headers.tracking.track_*_rate_unit``. Unknown unit
    strings are treated as arcsec/s (the senpai default) with a warning."""

    factor = _RATE_UNIT_TO_ARCSEC_PER_SEC.get(unit.strip().lower())
    if factor is None:
        logger.warning(
            "Unknown track-rate unit %r — treating value as arcsec/s. "
            "Known units: %s", unit, sorted(_RATE_UNIT_TO_ARCSEC_PER_SEC),
        )
        return value
    return value * factor


def extract_track_rates_from_header(header: Header) -> tuple[float, float, TrackMode]:
    config = get_config()

    ra_rate = None
    dec_rate = None
    track_mode = None

    for key in config.headers.tracking.track_ra_rate_keys:
        ra_rate = extract_header_value(header, key)
        if ra_rate is not None:
            ra_rate = _to_arcsec_per_second(
                float(ra_rate), config.headers.tracking.track_ra_rate_unit
            )
            logger.debug(f"Extracted RA rate from {key}: {ra_rate} arcsec/s")
            break

    for key in config.headers.tracking.track_dec_rate_keys:
        dec_rate = extract_header_value(header, key)
        if dec_rate is not None:
            dec_rate = _to_arcsec_per_second(
                float(dec_rate), config.headers.tracking.track_dec_rate_unit
            )
            logger.debug(f"Extracted DEC rate from {key}: {dec_rate} arcsec/s")
            break

    for key in config.headers.tracking.track_mode_keys:
        track_mode = extract_header_value(header, key)
        if track_mode is not None:
            track_mode = track_mode.strip()
            logger.debug(f"Extracted track mode from {key}: {track_mode}")
            break

    # Determine TrackMode enum value
    mode_enum = TrackMode.UNKNOWN

    if track_mode is not None:
        track_mode_lower = track_mode.lower()
        contains_rate = "rate" in track_mode_lower
        contains_sidereal = "sidereal" in track_mode_lower

        if contains_rate and not contains_sidereal:
            mode_enum = TrackMode.RATE
        elif contains_sidereal and not contains_rate:
            mode_enum = TrackMode.SIDEREAL
        else:
            # Contains both or neither - log warning and fall back to rate checking
            logger.warning(f"Ambiguous or unrecognized track mode: '{track_mode}'. Checking track rates for fallback.")

    # Fallback logic: if UNKNOWN, check track rates
    if mode_enum == TrackMode.UNKNOWN and ra_rate is not None and dec_rate is not None:
        if ra_rate == 0.0 and dec_rate == 0.0:
            mode_enum = TrackMode.SIDEREAL
            logger.debug("Track mode determined as SIDEREAL based on zero track rates")
        elif ra_rate != 0.0 or dec_rate != 0.0:
            mode_enum = TrackMode.RATE
            logger.debug("Track mode determined as RATE based on non-zero track rates")

    return ra_rate, dec_rate, mode_enum


def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract information from FITS files based on configuration.")
    parser.add_argument("fits_file", help="Path to the FITS file to analyze")
    parser.add_argument(
        "--config",
        "-c",
        help="Path to custom configuration file (optional, uses default if not provided)",
        default=LOCAL_APP_CONFIG_OVERRIDE,
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    set_log_level(level="DEBUG")

    # Initialize config (either default or from provided path)
    initialize_config(config_path=Path(args.config))

    try:
        # Open the FITS file
        with fits_open(args.fits_file) as hdul:
            header = hdul[0].header

            # Extract and display information
            print(f"Analyzing FITS file: {args.fits_file}")
            print("-" * 50)

            try:
                observation_time = extract_observation_time_from_header(header)
                print(f"Observation Time: {observation_time}")
            except Exception as e:
                print(f"Could not extract observation time: {e}")

            try:
                site = extract_observing_site_from_header(header)
                print(f"Observing Site: {site}")
            except Exception as e:
                print(f"Could not extract observing site: {e}")

            try:
                boresight = extract_boresight_from_header(header)
                print(f"Boresight (RA, Dec): {boresight}")
            except Exception as e:
                print(f"Could not extract boresight: {e}")

            try:
                exposure_time = extract_exposure_time_from_header(header)
                print(f"Exposure Time: {exposure_time} seconds")
            except Exception as e:
                print(f"Could not extract exposure time: {e}")

            try:
                track_rates = extract_track_rates_from_header(header)
                print(f"Track Rates: {track_rates}")
            except Exception as e:
                print(f"Could not extract track rates: {e}")

    except Exception as e:
        logger.error(f"Error processing FITS file: {e}")
        return 1

    return 0


if __name__ == "__main__":
    main()
