"""Per-night photometric calibration post-stage.

Consumes the manifest + per-batch ``SenpaiRun`` JSONs that
:mod:`senpai.cli.burr` writes, aggregates each frame's
:class:`~senpai.engine.photometry.utils.SimplePhotometrySummary` into a
:class:`NightCalibration`, and emits a calibration JSON + plot set:

* zero point vs airmass (Bouguer extinction fit per filter),
* limiting magnitude distribution per filter,
* Az/Alt coverage polar plot (one point per frame),
* ZP drift over the night.

The shape of this module is deliberately narrow: no FITS I/O, no astrometry,
no photometry. Frames without WCS still contribute to per-filter ZP medians;
geometric aggregates (extinction, Az/Alt coverage) drop those frames.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# --- per-frame extract --------------------------------------------------------


@dataclass(slots=True)
class FramePhoto:
    """Calibration-relevant slice of one frame's senpai output."""

    batch_id: str
    frame_index: int
    timestamp: datetime | None
    track_mode: str | None  # 'sidereal' or 'rate'
    filter_name: str | None
    exposure_time: float | None
    zero_point: float | None
    zero_point_err: float | None
    limiting_magnitude_50: float | None
    limiting_magnitude_90: float | None
    median_snr: float | None
    median_background: float | None
    n_stars: int | None
    n_quality: int | None

    # Geometry (None if WCS didn't solve).
    ra_center_deg: float | None = None
    dec_center_deg: float | None = None
    altitude_deg: float | None = None
    azimuth_deg: float | None = None
    airmass: float | None = None
    fov_sq_deg: float | None = None  # x_fov × y_fov from WCS, for area metrics
    # Moon geometry (filled by _add_moon_geometry; None if astropy/site absent).
    # Moonglow raises sky background → degrades SNR/depth, but NOT the zero
    # point (the star's flux is background-subtracted), so it's a depth-only
    # contaminant. Separation is field-center to Moon; alt < 0 means no glow.
    moon_sep_deg: float | None = None
    moon_alt_deg: float | None = None
    fwhm_px: float | None = None  # sidereal PSF FWHM median (detection_metadata)
    fwhm_std_px: float | None = None  # PSF FWHM spread across the field
    sky_adu: float | None = None  # flat-fielded sky level before row/col subtract
    pixel_scale_arcsec: float | None = None  # arcsec/pixel (x_fov / width)

    # Filter → ZP from the multiband calibration, when present.
    multiband_zps: dict[str, float] = field(default_factory=dict)

    # Per-star catalog magnitude + measured SNR pairs, lifted from the
    # SimplePhotometrySummary that senpai's collect pipeline now retains.
    # Empty (not None) when the frame has photometry but no qualifying stars.
    stars_mag: list[float] = field(default_factory=list)
    stars_snr: list[float] = field(default_factory=list)
    # Per-star ZP offset (m_cat − m_inst), parallel to stars_mag; None entries
    # mark stars without a measured instrumental magnitude. Empty on runs
    # predating stars_zp_offset retention.
    stars_zp_offset: list[float | None] = field(default_factory=list)
    # Per-star isolation flag, parallel to stars_mag (False = a brighter
    # catalog star inside the aperture footprint — blended flux). Empty on
    # runs predating retention; treat missing as isolated.
    stars_isolated: list[bool] = field(default_factory=list)

    # Rate-track geometry (None on sidereal frames). Retained for *all* frames
    # — including rate — so limiting-case studies are possible: ZP(streak
    # length), ZP(track rate), ΔSNR vs magnitude & streak length, and where
    # rate frames run out of stars (fast rates / small FoV). The night-level ZP
    # aggregation excludes rate frames (see _zp_frames), but the per-frame raw
    # numbers here are kept regardless.
    streak_length_px: float | None = None
    streak_fwhm_px: float | None = None
    pixel_track_rate: float | None = None  # px/s
    track_rate_arcsec_per_s: float | None = None  # on-sky |rate|, plate-scale independent

    @property
    def has_wcs(self) -> bool:
        return self.ra_center_deg is not None and self.dec_center_deg is not None


def _safe_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _airmass(altitude_deg: float | None) -> float | None:
    """Plane-parallel airmass = sec(zenith). Good to ~3.5; we clip above
    altitude < 3° to avoid divergence in plots."""
    if altitude_deg is None or altitude_deg <= 3.0:
        return None
    z = math.radians(90.0 - altitude_deg)
    return 1.0 / math.cos(z)


def _sky_mu(f: "FramePhoto") -> float | None:
    """Sky surface brightness in mag/arcsec², from the captured flat-fielded sky
    level. With m = ZP − 2.5·log10(flux/t_exp), one pixel's sky flux is at
    m_pix = ZP − 2.5·log10(sky_adu/t_exp); converting per-pixel → per-arcsec²
    adds 2.5·log10(pixel_area) = 5·log10(pixscale). Returns None if any input
    (ZP / sky / exposure / plate scale) is missing."""
    if (f.zero_point is None or f.sky_adu is None or f.sky_adu <= 0
            or not f.exposure_time or not f.pixel_scale_arcsec):
        return None
    return (f.zero_point - 2.5 * math.log10(f.sky_adu / f.exposure_time)
            + 5.0 * math.log10(f.pixel_scale_arcsec))


def _compute_alt_az(
    ra_deg: float, dec_deg: float, when: datetime, site: dict[str, Any] | None
) -> tuple[float, float] | None:
    """RA/Dec + UTC + site → (altitude_deg, azimuth_deg). Returns None if astropy
    is unavailable or site is incomplete. Imports are lazy so a calibration
    can be loaded without astropy when only ZP aggregates are needed."""

    if not site or site.get("latitude") is None or site.get("longitude") is None:
        return None

    try:
        from astropy.coordinates import AltAz, EarthLocation, SkyCoord
        from astropy.time import Time
        from astropy import units as u
    except ImportError:
        logger.debug("astropy unavailable; skipping alt/az conversion")
        return None

    location = EarthLocation(
        lat=site["latitude"] * u.deg,
        lon=site["longitude"] * u.deg,
        height=(site.get("altitude_km") or 0.0) * 1000.0 * u.m,
    )
    sky = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    aa = sky.transform_to(AltAz(obstime=Time(when), location=location))
    return float(aa.alt.deg), float(aa.az.deg)


def _add_moon_geometry(calib: "NightCalibration") -> None:
    """Fill per-frame Moon separation/altitude and the night's Moon illumination,
    in place. Vectorized (one astropy ephemeris call for all frames). No-op if
    astropy or the site is unavailable. Moonglow degrades depth (sky background)
    but not the zero point, so the depth/SNR plots use this to mask or model it."""

    site = calib.site
    if not site or site.get("latitude") is None or site.get("longitude") is None:
        return
    fr = [f for f in calib.frames
          if f.timestamp is not None and f.ra_center_deg is not None
          and f.dec_center_deg is not None]
    if not fr:
        return
    try:
        from astropy.coordinates import (AltAz, EarthLocation, SkyCoord,
                                         get_body, get_sun)
        from astropy.time import Time
        from astropy import units as u
        import numpy as np
    except ImportError:
        logger.debug("astropy unavailable; skipping moon geometry")
        return

    loc = EarthLocation(
        lat=site["latitude"] * u.deg, lon=site["longitude"] * u.deg,
        height=(site.get("altitude_km") or 0.0) * 1000.0 * u.m)
    times = Time([f.timestamp for f in fr], scale="utc")
    moon = get_body("moon", times, loc)
    fields = SkyCoord([f.ra_center_deg for f in fr] * u.deg,
                      [f.dec_center_deg for f in fr] * u.deg)
    seps = np.atleast_1d(fields.separation(moon).deg)
    malt = np.atleast_1d(moon.transform_to(AltAz(obstime=times, location=loc)).alt.deg)
    for f, s, a in zip(fr, seps, malt):
        f.moon_sep_deg = float(s)
        f.moon_alt_deg = float(a)
    # Illumination from the Sun–Moon elongation at mid-night.
    tmid = times[len(times) // 2]
    elong = get_body("moon", tmid, loc).separation(get_sun(tmid)).deg
    calib.moon_illumination = float(0.5 * (1 - math.cos(math.radians(elong))))
    logger.info("Moon: %.0f%% illuminated, separation median %.0f° (n=%d frames)",
                100 * calib.moon_illumination, float(np.median(seps)), len(fr))


def _extract_frame_photo(
    frame_dict: dict[str, Any],
    batch_id: str,
    site: dict[str, Any] | None,
    track_mode_default: str,
) -> FramePhoto | None:
    """Pull a FramePhoto from one serialized ``SiderealFrameSerializable`` or
    ``RateTrackFrameSerializable``. Returns None for frames with no photometry
    summary at all — they carry no calibration signal."""

    summary = frame_dict.get("photometry_summary")
    if not summary:
        # No photometry was measured; nothing to aggregate.
        return None

    fmd = frame_dict.get("frame_metadata") or {}
    starfield = frame_dict.get("starfield") or {}
    wcs_meta = (starfield or {}).get("wcs_metadata") or {}

    ts = _safe_iso(frame_dict.get("timestamp") or fmd.get("observation_time"))

    # Multiband ZPs, when present, are nested under photometry_summary.
    mb_cal = summary.get("multiband_calibration") or {}
    multiband_zps: dict[str, float] = {}
    for band_name, band in (mb_cal.get("bands") or {}).items():
        zp = band.get("zero_point") if isinstance(band, dict) else None
        if zp is not None:
            multiband_zps[band_name] = float(zp)

    ra_center = wcs_meta.get("RA_center_deg")
    dec_center = wcs_meta.get("Dec_center_deg")

    fov_x = wcs_meta.get("x_fov_degrees")
    fov_y = wcs_meta.get("y_fov_degrees")
    fov_sq_deg = float(fov_x) * float(fov_y) if fov_x and fov_y else None

    altitude = azimuth = None
    if ra_center is not None and dec_center is not None and ts is not None:
        alt_az = _compute_alt_az(ra_center, dec_center, ts, site)
        if alt_az is not None:
            altitude, azimuth = alt_az

    track_mode = fmd.get("track_mode") or track_mode_default

    # PSF FWHM (pixels) measured during detection — only meaningful on sidereal
    # frames (rate frames are streaked; their cross-streak width is streak_fwhm).
    fwhm_px = fwhm_std_px = None
    det_meta = (starfield or {}).get("detection_metadata") or {}
    if track_mode != "rate":
        # FWHM over REAL stars only. Sub-pixel (~1.2 px) "detections" are noise
        # spikes / hot pixels / cosmic rays; when transparency drops (cloud)
        # real stars vanish and these come to dominate, collapsing the naive
        # median toward the ~1 px floor — a contamination artifact, not better
        # seeing. Recompute the median over per-star FWHM ≥ _FWHM_MIN_PX; if too
        # few real stars survive, leave None (frame excluded from the FWHM plot).
        fs = det_meta.get("fwhm_stats") or {}
        positions = fs.get("fwhm_vs_position") or []
        if positions:
            real = [v[2] for v in positions
                    if v and len(v) > 2 and v[2] is not None and v[2] >= _FWHM_MIN_PX]
            if len(real) >= 5:
                import statistics as _st
                fwhm_px = float(_st.median(real))
                fwhm_std_px = float(_st.pstdev(real)) if len(real) > 1 else 0.0
        else:  # no per-star list (older serialization) → upstream median
            fwhm_px = det_meta.get("pixel_fwhm")
            fwhm_std_px = fs.get("std_fwhm")

    # Plate scale (arcsec/pixel) from the WCS field-of-view and image width —
    # needed to put the sky background in mag/arcsec².
    pixel_scale_arcsec = None
    img_meta = (starfield or {}).get("image_metadata") or {}
    img_w = img_meta.get("width")
    if fov_x and img_w:
        pixel_scale_arcsec = float(fov_x) * 3600.0 / float(img_w)

    # Physical sky level (ADU), captured pre-row/col-subtraction in the
    # column-median step metadata (only present on runs after that capture
    # landed; older runs leave this None and the sky plot is skipped).
    sky_adu = None
    for step in (frame_dict.get("processing_history") or []):
        if isinstance(step, dict) and "column_median_subtract" in str(
                step.get("step_type", "")).lower():
            sky_adu = (step.get("parameters") or {}).get("sky_median_adu")
            break

    # Rate-track geometry — only meaningful on rate frames. (Burr leaves a
    # non-zero residual RA/DEC rate in the header even on the sidereal leg, so
    # we must not read a "track rate" off sidereal frames.)
    streak_length_px = streak_fwhm_px = pixel_track_rate = None
    track_rate_arcsec_per_s = None
    if track_mode == "rate":
        streak = frame_dict.get("streak") or {}
        streak_length_px = streak.get("pixel_length")
        streak_fwhm_px = streak.get("fwhm")
        pixel_track_rate = frame_dict.get("pixel_track_rate_per_second")
        rate_ra = fmd.get("track_rate_ra_arcsec_per_second")
        rate_dec = fmd.get("track_rate_dec_arcsec_per_second")
        if rate_ra is not None and rate_dec is not None:
            track_rate_arcsec_per_s = math.hypot(rate_ra, rate_dec)

    return FramePhoto(
        batch_id=batch_id,
        frame_index=int(frame_dict.get("index", -1)),
        timestamp=ts,
        track_mode=track_mode,
        filter_name=fmd.get("observation_filter"),
        exposure_time=fmd.get("exposure_time_seconds"),
        zero_point=summary.get("zero_point"),
        zero_point_err=summary.get("zero_point_err"),
        limiting_magnitude_50=summary.get("limiting_magnitude_50"),
        limiting_magnitude_90=summary.get("limiting_magnitude_90"),
        median_snr=summary.get("median_snr"),
        median_background=summary.get("median_background"),
        n_stars=summary.get("n_stars"),
        n_quality=summary.get("n_quality"),
        ra_center_deg=ra_center,
        dec_center_deg=dec_center,
        altitude_deg=altitude,
        azimuth_deg=azimuth,
        airmass=_airmass(altitude),
        fov_sq_deg=fov_sq_deg,
        multiband_zps=multiband_zps,
        stars_mag=list(summary.get("stars_mag") or []),
        stars_snr=list(summary.get("stars_snr") or []),
        stars_zp_offset=list(summary.get("stars_zp_offset") or []),
        stars_isolated=list(summary.get("stars_isolated") or []),
        streak_length_px=streak_length_px,
        streak_fwhm_px=streak_fwhm_px,
        pixel_track_rate=pixel_track_rate,
        track_rate_arcsec_per_s=track_rate_arcsec_per_s,
        fwhm_px=fwhm_px,
        fwhm_std_px=fwhm_std_px,
        sky_adu=sky_adu,
        pixel_scale_arcsec=pixel_scale_arcsec,
    )


# --- aggregates ---------------------------------------------------------------


@dataclass(slots=True)
class ZeroPointStat:
    """Per-filter summary statistic of zero points across a night."""

    filter_name: str
    n: int
    median: float
    p16: float  # 16th percentile (lower 1-σ-ish)
    p84: float  # 84th percentile (upper 1-σ-ish)
    median_err: float | None = None


@dataclass(slots=True)
class ExtinctionFit:
    """Bouguer linear fit ``zero_point = m0 - k * airmass`` over a filter's
    frames. ``k`` is the extinction coefficient (mag/airmass, conventionally
    positive on clear nights); ``m0`` is the extra-atmospheric zero point."""

    filter_name: str
    m0: float          # zero point at zero airmass (extra-atmospheric)
    m0_err: float
    k: float           # extinction (mag/airmass) — positive on clear nights
    k_err: float
    n: int
    airmass_range: tuple[float, float]
    clear_fraction: float | None = None  # frac of frames near the clear-sky line


@dataclass(slots=True)
class NightCalibration:
    """Aggregated calibration products for one night."""

    night_id: str
    sensor: str | None
    site: dict[str, Any] | None
    n_frames_total: int
    n_frames_with_photometry: int
    n_frames_with_wcs: int
    frames: list[FramePhoto] = field(default_factory=list)
    zp_per_filter: dict[str, ZeroPointStat] = field(default_factory=dict)
    extinction_per_filter: dict[str, ExtinctionFit] = field(default_factory=dict)
    limiting_mag_p50_per_filter: dict[str, float] = field(default_factory=dict)
    limiting_mag_p90_per_filter: dict[str, float] = field(default_factory=dict)
    moon_illumination: float | None = None  # 0–1 fraction at mid-night
    # Night output dir (holds manifest.json + batches/). Kept so pixel-level
    # plots (PSF profile) can re-read the few frames they need + their raw FITS;
    # not a calibration product, so it is intentionally excluded from to_dict().
    source_dir: str | None = None

    def conditions(self) -> dict[str, Any]:
        """One-line-per-night observing-conditions summary (PSF, sky, extinction,
        Moon) for cross-night tracking. Medians over the night's frames."""
        import statistics as st

        def _med(xs):
            xs = [x for x in xs if x is not None]
            return float(st.median(xs)) if xs else None

        fwhm = _med([f.fwhm_px for f in self.frames])
        fwhm_spread = _med([f.fwhm_std_px for f in self.frames])
        sky_adu = _med([f.sky_adu for f in self.frames])
        sky_mu = _med([_sky_mu(f) for f in self.frames])
        moon_sep = _med([f.moon_sep_deg for f in self.frames])
        # Dominant-filter extinction (most frames).
        ext = max(self.extinction_per_filter.values(), key=lambda x: x.n,
                  default=None)
        lim50 = _med(list(self.limiting_mag_p50_per_filter.values())) \
            if self.limiting_mag_p50_per_filter else None
        return {
            "moon_illumination": self.moon_illumination,
            "moon_sep_median_deg": moon_sep,
            "extinction_k": ext.k if ext else None,
            "extinction_k_err": ext.k_err if ext else None,
            "zenith_transmission": (10 ** (-0.4 * ext.k)) if ext else None,
            "clear_fraction": ext.clear_fraction if ext else None,
            "fwhm_px_median": fwhm,
            "fwhm_px_spread": fwhm_spread,
            "sky_adu_median": sky_adu,
            "sky_mag_arcsec2_median": sky_mu,
            "limiting_mag_50_median": lim50,
            "n_frames_photometry": self.n_frames_with_photometry,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "night_id": self.night_id,
            "sensor": self.sensor,
            "site": self.site,
            "moon_illumination": self.moon_illumination,
            "n_frames_total": self.n_frames_total,
            "n_frames_with_photometry": self.n_frames_with_photometry,
            "n_frames_with_wcs": self.n_frames_with_wcs,
            "zp_per_filter": {
                k: _asdict_safe(v) for k, v in self.zp_per_filter.items()
            },
            "extinction_per_filter": {
                k: _asdict_safe(v) for k, v in self.extinction_per_filter.items()
            },
            "limiting_mag_p50_per_filter": self.limiting_mag_p50_per_filter,
            "limiting_mag_p90_per_filter": self.limiting_mag_p90_per_filter,
            "conditions": self.conditions(),
            "frames": [_asdict_safe(f) for f in self.frames],
        }


def _asdict_safe(obj: Any) -> Any:
    """Pure-stdlib dataclass→dict that handles our datetime fields."""
    from dataclasses import asdict, is_dataclass

    if is_dataclass(obj):
        d = asdict(obj)
        for k, v in d.items():
            if isinstance(v, datetime):
                d[k] = v.isoformat()
        return d
    return obj


# --- aggregation engine -------------------------------------------------------


# A Bouguer fit needs the airmass to actually vary; below this span the slope
# (extinction coefficient) is unconstrained and any fit is noise.
_MIN_AIRMASS_RANGE = 0.15
_FWHM_MIN_PX = 2.0  # below this a "star" is a noise spike, not a real PSF


def _zp_frames(frames: list[FramePhoto]) -> list[FramePhoto]:
    """Frames whose photometry is trustworthy for the night's zero point: the
    sidereal-tracked frames. Rate-tracked frames image stars as streaks, so
    their aperture photometry — and any ZP derived from it — is unreliable and
    must not define the night's photometric calibration."""

    return [f for f in frames if f.track_mode == "sidereal"]


def _isolated_flags(f: FramePhoto) -> list[bool]:
    """Per-star isolation flags parallel to ``stars_mag``; legacy runs that
    predate ``stars_isolated`` retention default to all-isolated."""

    return f.stars_isolated or [True] * len(f.stars_mag)


def _frame_task(f: FramePhoto) -> str:
    """Task token parsed from the burr batch_id (…_<task>_<target>_<hash>)."""
    bid = f.batch_id or ""
    for t in ("photometric", "coverage", "calsats", "twilight_flats"):
        if t in bid:
            return t
    return "other"


# Frames within this of an up Moon are sky-background-contaminated; the depth/
# SNR plots suppress them. Empirically moonglow degrades depth out to ~45° of a
# full Moon (at 30° the photometric exposures still sagged ~0.25 mag below the
# coverage √t line; at 45° that closes and the global exposure ladder is
# monotonic). Beyond ~45-60° the penalty is negligible.
_MOON_SEP_MIN_DEG = 45.0


def _moon_ok(f: FramePhoto, min_sep: float = _MOON_SEP_MIN_DEG) -> bool:
    """False when the frame is within ``min_sep`` of an *above-horizon* Moon
    (moonglow inflates sky background → depth loss). Frames with no Moon
    geometry, or taken with the Moon down, pass."""
    if f.moon_sep_deg is None:
        return True
    if f.moon_alt_deg is not None and f.moon_alt_deg < 0:
        return True  # Moon below horizon — no glow
    return f.moon_sep_deg >= min_sep


def _clear_sky_zp_band(frames: list[FramePhoto]) -> tuple[float, float] | None:
    """``(mode, sigma)`` of the clear-sky cluster in the per-frame ZP histogram.

    Cloud only *reduces* throughput, so the ZP distribution has a sharp
    clear-sky mode with a one-sided tail to low ZP. The mode is the night's
    photometric zero point and the scatter of frames within ±0.4 mag of it is
    the clear-sky stability. Frames within ±sigma of the mode are a "weather
    mask": photometric-condition frames with the cloud-attenuated ones dropped.
    Returns None if too few sidereal frames to estimate."""
    import numpy as np

    zps = np.array([f.zero_point for f in _zp_frames(frames)
                    if f.zero_point is not None])
    if len(zps) < 10:
        return None
    hist, edges = np.histogram(zps, bins=40)
    i = int(hist.argmax())
    mode = float(0.5 * (edges[i] + edges[i + 1]))
    near = zps[np.abs(zps - mode) <= 0.4]
    sigma = float(np.std(near)) if len(near) >= 3 else 0.2
    return mode, sigma


def _snr_consistent(snr: float, mag: float, f: FramePhoto,
                    tolerance: float = 5.0) -> bool:
    """True when a star's measured SNR is plausible for its catalog magnitude.

    By definition SNR ≈ limiting_snr (3) at the frame's lim50 and scales with
    flux (×10^0.4 per mag) in the background-dominated regime, so the frame
    predicts each star's SNR from its magnitude alone. Stars measured far
    above that (×{tolerance}) are real flux wrongly attributed — bright-star
    wings/spikes outside the isolation radius, variables, bad cross-matches.
    They are ~2% of isolated SNR≥5 stars but carry ×10–80 flux excess, enough
    to fabricate entire faint-end bins. Saturated/bright stars always pass
    (measured ≤ predicted there)."""

    if f.limiting_magnitude_50 is None:
        return True
    snr_pred = 3.0 * 10 ** (0.4 * (f.limiting_magnitude_50 - mag))
    return snr <= tolerance * snr_pred


def _summarize_zp(frames: list[FramePhoto]) -> dict[str, ZeroPointStat]:
    """Median + 16/84 percentile of zero_point per filter (sidereal frames)."""

    import statistics

    by_filter: dict[str, list[FramePhoto]] = {}
    for f in _zp_frames(frames):
        if f.zero_point is None:
            continue
        key = f.filter_name or "unknown"
        by_filter.setdefault(key, []).append(f)

    out: dict[str, ZeroPointStat] = {}
    for filt, fs in by_filter.items():
        zps = sorted(f.zero_point for f in fs)
        errs = [f.zero_point_err for f in fs if f.zero_point_err is not None]
        out[filt] = ZeroPointStat(
            filter_name=filt,
            n=len(zps),
            median=statistics.median(zps),
            p16=_percentile(zps, 0.16),
            p84=_percentile(zps, 0.84),
            median_err=statistics.median(errs) if errs else None,
        )
    return out


def _percentile(sorted_xs: list[float], q: float) -> float:
    if not sorted_xs:
        return float("nan")
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    idx = q * (len(sorted_xs) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_xs[lo]
    return sorted_xs[lo] + (idx - lo) * (sorted_xs[hi] - sorted_xs[lo])


_EXT_ENV_PCT = 85.0      # upper-envelope percentile = clear-sky proxy
_EXT_BIN_W = 0.1         # airmass bin width
_EXT_CLEAR_TOL = 0.1     # mag; frames within this below the line count as clear


def _extinction_envelope_fit(pairs: list[tuple[float, float]]) -> dict | None:
    """Cloud-robust Bouguer fit of frame zero point vs airmass.

    pairs = [(airmass, zero_point), ...] for one filter. Cloud only ATTENUATES
    (drops ZP, never raises it), so the clear-sky Bouguer line is the UPPER
    ENVELOPE of the ZP-vs-airmass cloud — cloudy frames scatter below it. We fit
    the per-airmass-bin upper percentile (the clean edge) rather than the median:
    the median is the central tendency of a one-sidedly cloud-eaten sample, i.e.
    survivorship-biased downward (and OLS over-steepens k by conflating cloud
    with extinction). Returns fit + the bin median/envelope points for plotting,
    plus a clear_fraction diagnostic. None if too few points / airmass range.
    """
    import numpy as np

    if len(pairs) < 3:
        return None
    X = np.array([p[0] for p in pairs])
    Z = np.array([p[1] for p in pairs])
    if X.max() - X.min() < _MIN_AIRMASS_RANGE:
        return None
    edges = np.arange(X.min(), X.max() + _EXT_BIN_W, _EXT_BIN_W)
    cx, cmed, cenv = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        zz = Z[(X >= lo) & (X < hi)]
        if len(zz) >= 5:
            cx.append(float((lo + hi) / 2))
            cmed.append(float(np.median(zz)))
            cenv.append(float(np.percentile(zz, _EXT_ENV_PCT)))
    if len(cx) >= 2:
        ex, ey, note = cx, cenv, f"{_EXT_ENV_PCT:.0f}th-pct envelope"
    else:
        ex, ey, note = list(map(float, X)), list(map(float, Z)), "raw OLS (sparse)"
    n = len(ex)
    mx = sum(ex) / n
    my = sum(ey) / n
    ssxx = sum((x - mx) ** 2 for x in ex)
    if ssxx <= 0:
        return None
    slope = sum((ex[i] - mx) * (ey[i] - my) for i in range(n)) / ssxx
    m0 = my - slope * mx
    resid = [ey[i] - (m0 + slope * ex[i]) for i in range(n)]
    sigma2 = sum(r * r for r in resid) / max(n - 2, 1)
    slope_err = math.sqrt(sigma2 / ssxx) if sigma2 > 0 else 0.0
    m0_err = math.sqrt(sigma2 * (1 / n + mx * mx / ssxx)) if sigma2 > 0 else 0.0
    clear_fraction = float(np.mean(Z >= (m0 + slope * X) - _EXT_CLEAR_TOL))
    return {
        "k": -slope, "k_err": slope_err, "m0": m0, "m0_err": m0_err,
        "n": len(pairs), "airmass_range": (float(X.min()), float(X.max())),
        "bin_centers": cx, "bin_median": cmed, "bin_envelope": cenv,
        "note": note, "clear_fraction": clear_fraction,
    }


def _fit_extinction(frames: list[FramePhoto]) -> dict[str, ExtinctionFit]:
    """Per-filter cloud-robust Bouguer fit ``zero_point = m0 - k * airmass`` via
    the upper-envelope method (see _extinction_envelope_fit). k = -slope. This is
    the authoritative extinction used in night_calibration.json and for the
    airmass-normalization across the SNR plots."""
    by_filter: dict[str, list[tuple[float, float]]] = {}
    for f in _zp_frames(frames):
        if f.zero_point is None or f.airmass is None:
            continue
        key = f.filter_name or "unknown"
        by_filter.setdefault(key, []).append((f.airmass, f.zero_point))

    out: dict[str, ExtinctionFit] = {}
    for filt, pairs in by_filter.items():
        r = _extinction_envelope_fit(pairs)
        if r is None:
            continue
        out[filt] = ExtinctionFit(
            filter_name=filt,
            m0=r["m0"], m0_err=r["m0_err"],
            k=r["k"], k_err=r["k_err"],
            n=r["n"], airmass_range=r["airmass_range"],
            clear_fraction=r["clear_fraction"],
        )
    return out


def _summarize_limiting_mag(
    frames: list[FramePhoto], attr: str
) -> dict[str, float]:
    """Median limiting magnitude per filter, from SIDEREAL frames only.

    Rate-tracked frames image stars as streaks; their forced-photometry
    completeness is dominated by faint catalog positions landing on brighter
    stars' trails (a spurious detection floor), so a limiting mag read off them
    is unreliable. The night's authoritative depth comes from the sidereal legs
    (the raw rate per-frame values are still retained on each FramePhoto for
    limiting-case studies)."""
    import statistics

    by_filter: dict[str, list[float]] = {}
    for f in _zp_frames(frames):
        v = getattr(f, attr)
        if v is None:
            continue
        key = f.filter_name or "unknown"
        by_filter.setdefault(key, []).append(v)
    return {k: statistics.median(v) for k, v in by_filter.items()}


# --- night loader -------------------------------------------------------------


def analyze_night(night_dir: str | Path) -> NightCalibration:
    """Build a :class:`NightCalibration` from the output of
    ``python -m senpai.cli.burr night <night_dir> -o <output>`` — i.e. a dir
    that contains ``manifest.json`` and a ``batches/`` tree of SenpaiRun JSONs."""

    night_dir = Path(night_dir)
    manifest_path = night_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No manifest.json at {manifest_path}. Run `senpai-burr night` first."
        )
    manifest = json.loads(manifest_path.read_text())

    frames: list[FramePhoto] = []
    n_total = 0
    for entry in manifest.get("batches", []):
        # A skipped batch (resumed --skip-existing run) still has valid output;
        # include it as long as its result JSON is present.
        result_path = entry.get("result_path")
        if not result_path:
            continue
        batch_id = entry["batch_id"]
        path = Path(result_path)
        if not path.is_file():
            # The manifest stores absolute paths; when the drive remounts
            # elsewhere they go stale. Re-anchor on this night_dir.
            path = night_dir / "batches" / batch_id / path.name
            if not path.is_file():
                continue
        try:
            run = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            logger.warning("Skipping unreadable %s: %s", path, e)
            continue
        for fd in run.get("sidereal_frames", []):
            n_total += 1
            fp = _extract_frame_photo(fd, batch_id, manifest.get("site"), "sidereal")
            if fp is not None:
                frames.append(fp)
        for fd in run.get("rate_track_frames", []):
            n_total += 1
            fp = _extract_frame_photo(fd, batch_id, manifest.get("site"), "rate")
            if fp is not None:
                frames.append(fp)

    n_with_wcs = sum(1 for f in frames if f.has_wcs)
    calib = NightCalibration(
        night_id=manifest.get("night_id", night_dir.name),
        sensor=manifest.get("sensor"),
        site=manifest.get("site"),
        n_frames_total=n_total,
        n_frames_with_photometry=len(frames),
        n_frames_with_wcs=n_with_wcs,
        frames=frames,
        source_dir=str(night_dir),
    )
    _add_moon_geometry(calib)
    calib.zp_per_filter = _summarize_zp(frames)
    calib.extinction_per_filter = _fit_extinction(frames)
    calib.limiting_mag_p50_per_filter = _summarize_limiting_mag(frames, "limiting_magnitude_50")
    calib.limiting_mag_p90_per_filter = _summarize_limiting_mag(frames, "limiting_magnitude_90")

    logger.info(
        "NightCalibration %s: %d/%d frames had photometry, %d had WCS; "
        "filters with ZP: %s; extinction fits: %s",
        calib.night_id, calib.n_frames_with_photometry, calib.n_frames_total,
        calib.n_frames_with_wcs,
        sorted(calib.zp_per_filter.keys()),
        sorted(calib.extinction_per_filter.keys()),
    )
    return calib


# --- persistence + plots ------------------------------------------------------


def save_calibration(calib: NightCalibration, output_dir: str | Path) -> Path:
    """Write the aggregated calibration JSON. Returns the file path."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "night_calibration.json"
    with open(path, "w") as f:
        json.dump(calib.to_dict(), f, indent=2)
    logger.info("Wrote %s", path)
    return path


def _jsonify(obj: Any) -> Any:
    """Recursively convert numpy / datetime values to JSON-native types so the
    plotted arrays round-trip through plot_data.json."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonify(obj.tolist())
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


# Per-plot (analysis, render) functions driving the calibration plot set.
#   analysis(calib) -> dict | None   (None = no data, plot skipped)
#   render(data, meta, output_dir, plt, np) -> Path | list[Path]
# Registered after their definitions below; build_plot_data runs every analysis,
# plot_calibration runs every render from the resulting plot_data dict.
_PLOT_BUILDERS: dict[str, tuple] = {}


def build_plot_data(calib: NightCalibration) -> dict:
    """Run ALL calibration plot analysis up front and return a JSON-serializable
    dict — the actual plotted arrays (gray clouds, binned points, fit lines),
    not the raw per-frame data. This is the single analysis stage; renderers
    consume only this dict, so plots can be regenerated from plot_data.json
    without reprocessing the batch JSONs."""
    meta = {
        "night_id": calib.night_id,
        "site": calib.site,
        "moon_illumination": calib.moon_illumination,
    }
    plots: dict[str, Any] = {}
    for name, (analysis, _render) in _PLOT_BUILDERS.items():
        d = analysis(calib)
        if d is not None:
            plots[name] = d
    return _jsonify({"version": 1, "meta": meta, "plots": plots})


def save_plot_data(plot_data: dict, output_dir: str | Path) -> Path:
    """Write the plotted-data dict to <output_dir>/plot_data.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "plot_data.json"
    with open(path, "w") as f:
        json.dump(plot_data, f, indent=2)
    logger.info("Wrote %s", path)
    return path


def load_plot_data(path: str | Path) -> dict:
    """Load a previously written plot_data.json."""
    with open(path) as f:
        return json.load(f)


# Columns for the cross-night conditions table: (header, conditions-key, fmt).
_NIGHTS_COLS = [
    ("night", "night_id", "{}"),
    ("moon%", "moon_illumination", "{:.0%}"),
    ("moonSep°", "moon_sep_median_deg", "{:.0f}"),
    ("k", "extinction_k", "{:.3f}"),
    ("T_zen", "zenith_transmission", "{:.0%}"),
    ("clear%", "clear_fraction", "{:.0%}"),
    ("FWHM_px", "fwhm_px_median", "{:.1f}"),
    ("FWHM_sd", "fwhm_px_spread", "{:.1f}"),
    ("sky_ADU", "sky_adu_median", "{:.0f}"),
    ("sky_μ", "sky_mag_arcsec2_median", "{:.1f}"),
    ("lim50", "limiting_mag_50_median", "{:.1f}"),
    ("nFrm", "n_frames_photometry", "{:d}"),
]


def summarize_nights(root: str | Path, csv_path: str | Path | None = None) -> str:
    """Aggregate every night's conditions into one table for tracking PSF / sky /
    extinction vs Moon phase & weather across nights. Reads each
    ``<root>/*/calibration/night_calibration.json`` (its ``conditions`` block).
    Returns the formatted table; optionally also writes a CSV."""
    root = Path(root)
    rows: list[dict] = []
    for nc_path in sorted(root.glob("*/calibration/night_calibration.json")):
        try:
            with open(nc_path) as f:
                nc = json.load(f)
        except Exception as e:
            logger.warning("skip %s: %s", nc_path, e)
            continue
        c = dict(nc.get("conditions") or {})
        c["night_id"] = nc.get("night_id") or nc_path.parts[-3]
        rows.append(c)
    if not rows:
        return f"No night_calibration.json found under {root}"

    headers = [h for h, _, _ in _NIGHTS_COLS]
    table_rows = []
    for c in rows:
        cells = []
        for _h, key, fmt in _NIGHTS_COLS:
            v = c.get(key)
            cells.append("—" if v is None else fmt.format(v))
        table_rows.append(cells)
    widths = [max(len(headers[i]), *(len(r[i]) for r in table_rows))
              for i in range(len(headers))]
    sep = "  "
    out = [sep.join(h.rjust(widths[i]) for i, h in enumerate(headers))]
    out.append(sep.join("-" * widths[i] for i in range(len(headers))))
    for r in table_rows:
        out.append(sep.join(r[i].rjust(widths[i]) for i in range(len(headers))))
    table = "\n".join(out)

    if csv_path is not None:
        import csv as _csv
        with open(csv_path, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow([h for h, _, _ in _NIGHTS_COLS])
            for c in rows:
                w.writerow([c.get(key) for _h, key, _f in _NIGHTS_COLS])
        logger.info("Wrote %s", csv_path)
    return table


def plot_calibration(source, output_dir: str | Path,
                     *, save_data: bool = True) -> list[Path]:
    """Render the calibration plot set. Quietly skips plots that have no data.

    ``source`` is either a NightCalibration (live: the analysis is run via
    build_plot_data and, unless save_data=False, dumped to plot_data.json) or a
    plot_data dict already loaded from plot_data.json (replot: no reprocessing).
    matplotlib is imported lazily so this module loads cheaply when only the
    aggregation is needed.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    if isinstance(source, NightCalibration):
        calib: NightCalibration | None = source
        plot_data = build_plot_data(calib)
        if save_data:
            save_plot_data(plot_data, output_dir)
    else:
        calib = None
        plot_data = source

    meta = plot_data.get("meta", {})
    for name, (_analysis, render) in _PLOT_BUILDERS.items():
        d = plot_data.get("plots", {}).get(name)
        if d is not None:
            out = render(d, meta, output_dir, plt, np)
            paths.extend(out if isinstance(out, list) else [out])

    return paths


def _save(fig, path: Path) -> Path:
    import matplotlib.pyplot as plt

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)
    return path


# --- Migrated plots: (analysis, render) pairs --------------------------------
# analysis(calib) -> JSON-serializable dict | None;  render(d, meta, out, plt, np) -> Path.
# Registered in _PLOT_BUILDERS at the bottom of this section.


def _data_limiting_magnitude_hist(calib: NightCalibration):
    by_filter: dict[str, list] = {}
    for f in _zp_frames(calib.frames):
        if f.limiting_magnitude_50 is not None:
            by_filter.setdefault(f.filter_name or "unknown", []).append(
                f.limiting_magnitude_50)
    return {"by_filter": by_filter} if by_filter else None


def _render_limiting_magnitude_hist(d, meta, output_dir, plt, np) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for filt in sorted(d["by_filter"]):
        xs = d["by_filter"][filt]
        ax.hist(xs, bins=30, alpha=0.5, label=f"{filt} (n={len(xs)})")
    ax.set_xlabel("limiting magnitude (50% completeness)")
    ax.set_ylabel("number of frames")
    ax.set_title(f"{meta['night_id']}: limiting magnitude distribution")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    return _save(fig, output_dir / "limiting_magnitude_hist.png")


def _data_extinction_curve(calib: NightCalibration):
    # Frame ZP vs airmass, fit cloud-robustly by the upper envelope (same
    # _extinction_envelope_fit that feeds night_calibration.json, so the plot's
    # red line == the reported k). Cloud drops points below the clear-sky line;
    # the per-bin median (shown for contrast) is dragged down by it, the
    # envelope tracks the clean edge.
    pairs_by_filter: dict[str, list] = {}
    for f in _zp_frames(calib.frames):
        if f.zero_point is None or f.airmass is None:
            continue
        pairs_by_filter.setdefault(f.filter_name or "unknown", []).append(
            (f.airmass, f.zero_point))
    series = []
    for filt, pairs in sorted(pairs_by_filter.items()):
        r = _extinction_envelope_fit(pairs)
        if r is None:
            continue
        series.append({
            "filter": filt,
            "airmass": [p[0] for p in pairs], "zp": [p[1] for p in pairs],
            "bin_centers": r["bin_centers"], "bin_median": r["bin_median"],
            "bin_envelope": r["bin_envelope"],
            "k": r["k"], "k_err": r["k_err"], "m0": r["m0"],
            "note": r["note"], "clear_fraction": r["clear_fraction"],
        })
    return {"series": series} if series else None


def _render_extinction_curve(d, meta, output_dir, plt, np) -> Path:
    fig, ax = plt.subplots(figsize=(10, 7))
    series = d["series"]
    multi = len(series) > 1
    cmap = plt.cm.viridis(np.linspace(0, 0.85, max(len(series), 1)))
    title_bits = []
    for color, s in zip(cmap, series):
        pre = f"{s['filter']} " if multi else ""
        X = np.array(s["airmass"])
        line_x = np.linspace(X.min(), X.max(), 50)
        sky = 10 ** (-0.4 * s["k"])
        flag = "  ⚠ NON-PHOTOMETRIC" if s["clear_fraction"] < 0.30 else ""
        if multi:
            ax.scatter(X, s["zp"], s=12, alpha=0.5, color=color,
                       label=f"{pre}frames (n={len(X)})")
            ax.plot(line_x, s["m0"] - s["k"] * line_x, color=color, lw=2,
                    label=f"  k={s['k']:.3f}±{s['k_err']:.3f} (T_zen={sky:.0%})")
        else:
            ax.scatter(X, s["zp"], s=12, alpha=0.4, color="lightgray",
                       label=f"frames (n={len(X)})")
            if s["bin_centers"]:
                ax.plot(s["bin_centers"], s["bin_median"], "o--",
                        color="darkorange", ms=6, alpha=0.7,
                        label="per-bin median (cloud-biased)")
                ax.plot(s["bin_centers"], s["bin_envelope"], "o", color="black",
                        ms=7, label=f"{_EXT_ENV_PCT:.0f}th-pct envelope (clear-sky)")
            ax.plot(line_x, s["m0"] - s["k"] * line_x, "r-", lw=2, alpha=0.85,
                    label=f"envelope fit: k={s['k']:.3f}±{s['k_err']:.3f}  "
                          f"(zenith T={sky:.0%})")
        title_bits.append(f"{pre}k={s['k']:.3f} (clear-frac {s['clear_fraction']:.0%}){flag}")
    ax.set_ylabel("zero point (instrumental → catalog mag)")
    ax.set_xlabel("Airmass")
    ax.set_title(f"{meta['night_id']}: extinction (cloud-robust upper-envelope)\n"
                 + " | ".join(title_bits))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax2 = ax.twiny()
    ticks = np.array([t for t in ax.get_xticks() if t >= 1.0])
    if len(ticks):
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(
            [f"{math.degrees(math.asin(1.0 / t)):.0f}°" for t in ticks])
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel("Altitude")
    return _save(fig, output_dir / "extinction_curve.png")


def _data_alt_az_coverage(calib: NightCalibration):
    aa = [(f.azimuth_deg, f.altitude_deg, f.timestamp) for f in calib.frames
          if f.azimuth_deg is not None and f.altitude_deg is not None]
    if not aa:
        return None
    ts = [t for _, _, t in aa if t is not None]
    ts0 = min(ts) if ts else None
    colors = [(t - ts0).total_seconds() if (ts0 is not None and t is not None)
              else 0 for _, _, t in aa]
    return {
        "thetas": [math.radians(p[0]) for p in aa],
        "rs": [90 - p[1] for p in aa],
        "colors": colors, "has_time": ts0 is not None, "n": len(aa),
    }


def _render_alt_az_coverage(d, meta, output_dir, plt, np) -> Path:
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    sc = ax.scatter(d["thetas"], d["rs"], c=d["colors"], s=12, cmap="plasma",
                    alpha=0.7)
    ax.set_ylim(0, 90)
    ax.set_yticks([15, 30, 45, 60, 75])
    ax.set_yticklabels([f"{90 - r}°" for r in [15, 30, 45, 60, 75]])
    ax.set_title(f"{meta['night_id']}: Az/Alt coverage  (n={d['n']})", pad=20)
    if d["has_time"]:
        cb = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.7)
        cb.set_label("seconds since first frame")
    return _save(fig, output_dir / "alt_az_coverage.png")


def _data_zp_drift(calib: NightCalibration):
    import numpy as np
    from datetime import timedelta

    drift = [(f.timestamp, f.zero_point, f.filter_name or "unknown")
             for f in _zp_frames(calib.frames)
             if f.timestamp is not None and f.zero_point is not None]
    if not drift:
        return None
    BIN_SECONDS = 1200.0
    per_filter = {}
    for filt in sorted({d[2] for d in drift}):
        pts = sorted((d[0], d[1]) for d in drift if d[2] == filt)
        xs = [p[0] for p in pts]
        ys = np.array([p[1] for p in pts])
        t0 = xs[0]
        secs = np.array([(x - t0).total_seconds() for x in xs])
        edges = np.arange(0.0, secs.max() + BIN_SECONDS, BIN_SECONDS)
        cx, cy, e_lo, e_hi = [], [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            in_bin = ys[(secs >= lo) & (secs < hi)]
            if len(in_bin) < 3:
                continue
            med = float(np.median(in_bin))
            cx.append((t0 + timedelta(seconds=(lo + hi) / 2)).isoformat())
            cy.append(med)
            e_lo.append(med - float(np.percentile(in_bin, 16)))
            e_hi.append(float(np.percentile(in_bin, 84)) - med)
        per_filter[filt] = {
            "scatter_t": [x.isoformat() for x in xs],
            "scatter_zp": [float(y) for y in ys],
            "binned_t": cx, "binned_zp": cy, "err_lo": e_lo, "err_hi": e_hi,
        }
    return {"per_filter": per_filter, "n_filters": len(per_filter)}


def _render_zp_drift(d, meta, output_dir, plt, np) -> Path:
    from datetime import datetime as _dt

    fig, ax = plt.subplots(figsize=(10, 5))
    for filt in sorted(d["per_filter"]):
        s = d["per_filter"][filt]
        xs = [_dt.fromisoformat(t) for t in s["scatter_t"]]
        ax.scatter(xs, s["scatter_zp"], label=f"{filt} (n={len(xs)})", s=10,
                   alpha=0.4)
        if s["binned_t"]:
            cx = [_dt.fromisoformat(t) for t in s["binned_t"]]
            lbl = ("binned (median ± 16/84%)" if d["n_filters"] == 1
                   else f"{filt} binned")
            ax.errorbar(cx, s["binned_zp"], yerr=[s["err_lo"], s["err_hi"]],
                        fmt="o", color="black", markersize=6, capsize=4,
                        capthick=1.5, elinewidth=1.5, alpha=0.85, zorder=5,
                        label=lbl)
    ax.set_xlabel("UTC time")
    ax.set_ylabel("zero point")
    ax.set_title(f"{meta['night_id']}: zero point drift")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    return _save(fig, output_dir / "zp_drift.png")


# --- inter-frame overhead (slew + settle + readout) model ---------------------
# DATE-OBS is the exposure START, so for time-ordered frames i -> i+1 the gap
# between successive starts decomposes as
#     Δt = exposure[i] + readout + settle + slew_time(separation)
# and the per-pair "overhead" Δt - exposure[i] is readout-only when the mount
# does not move (repeat exposures at one pointing) and gains a fixed settle step
# plus a distance-proportional slew term whenever it does. We fit that two-regime
# model from the night's own telemetry rather than assuming a flat overhead.
_SLEW_READOUT_MAX_DEG = 0.1   # below this separation: no slew, overhead==readout
_SLEW_MIN_DEG = 0.25          # at/above this separation: treat as a real slew
_SLEW_MAX_GAP_S = 300.0       # drop pairs spanning focus runs / weather pauses
_SLEW_ENV_PCTILE = 10         # lower-envelope percentile = physical floor per bin
_SLEW_ENV_MIN_PTS = 6         # min pairs per distance bin to anchor the envelope
_SLEW_DIST_BINS = [0.25, 0.5, 1, 2, 4, 8, 15, 25, 40, 60, 90, 180]


def _angsep_deg(alt1, az1, alt2, az2) -> float:
    """Great-circle separation (deg) in the mount's alt/az frame — the same
    slew metric burr's coverage optimizer uses."""
    a1, a2 = math.radians(alt1), math.radians(alt2)
    d = (math.sin((a2 - a1) / 2) ** 2
         + math.cos(a1) * math.cos(a2) * math.sin(math.radians(az2 - az1) / 2) ** 2)
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(d))))


def _fit_slew_model(frames: list[FramePhoto]) -> dict | None:
    """Fit the inter-frame overhead model (readout + settle + slew) from the
    night's time-ordered frames.

    Pairs are consecutive in time across ALL frames (sidereal and rate) — sorting
    a single track mode would skip over interleaved frames and inflate the gaps.
    The lower-envelope (per-bin p10) fit is robust to the minority of pairs whose
    separation is rate-track motion rather than a slew, and to contingent delays
    (plate-solve retries, downloads) that only ever sit *above* the floor.

    Returns the fitted scalars, the contiguous-grid cadence overhead (one
    FoV-width slew), and the raw (separation, overhead) cloud + envelope for
    plotting; or None if there aren't enough usable pairs.
    """
    import numpy as np

    fr = sorted(
        (f for f in frames
         if f.timestamp and f.exposure_time
         and f.altitude_deg is not None and f.azimuth_deg is not None),
        key=lambda f: f.timestamp,
    )
    dist, over = [], []
    for a, b in zip(fr, fr[1:]):
        dt = (b.timestamp - a.timestamp).total_seconds()
        ov = dt - a.exposure_time            # DATE-OBS = exposure start
        if ov <= 0 or dt >= _SLEW_MAX_GAP_S:  # clock glitch / long pause
            continue
        dist.append(_angsep_deg(a.altitude_deg, a.azimuth_deg,
                                b.altitude_deg, b.azimuth_deg))
        over.append(ov)
    if len(dist) < 2 * _SLEW_ENV_MIN_PTS:
        return None
    dist, over = np.array(dist), np.array(over)

    # Readout floor: overhead when the mount does not move (repeat exposures).
    same = dist < _SLEW_READOUT_MAX_DEG
    if int(same.sum()) < _SLEW_ENV_MIN_PTS:
        return None
    readout = float(np.median(over[same]))

    # Slew regime: per-bin lower envelope, then a count-weighted line
    # overhead = bias + separation / slew_rate  (bias = readout + settle).
    edges = np.array(_SLEW_DIST_BINS)
    ex, ey, en = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist >= lo) & (dist < hi)
        if int(m.sum()) >= _SLEW_ENV_MIN_PTS:
            ex.append(float(np.median(dist[m])))
            ey.append(float(np.percentile(over[m], _SLEW_ENV_PCTILE)))
            en.append(int(m.sum()))
    if len(ex) < 2:
        return None
    ex_a, ey_a, w = np.array(ex), np.array(ey), np.sqrt(np.array(en, float))
    A = np.vstack([np.ones_like(ex_a), ex_a]).T * w[:, None]
    (bias, slope), *_ = np.linalg.lstsq(A, ey_a * w, rcond=None)
    if slope <= 0:        # degenerate / inverted — no usable slew term
        return None
    slew_rate = 1.0 / slope
    settle = float(bias) - readout

    fovs = [f.fov_sq_deg for f in fr if f.fov_sq_deg]
    fov_width = math.sqrt(float(np.median(fovs))) if fovs else None
    # Contiguous grid search: each step slews one FoV width to the next tile.
    grid_overhead = float(bias) + (fov_width / slew_rate if fov_width else 0.0)

    return {
        "readout_s": readout,
        "settle_s": settle,
        "slew_rate_deg_s": slew_rate,
        "bias_s": float(bias),
        "fov_width_deg": fov_width,
        "grid_overhead_s": grid_overhead,
        "n_pairs": int(len(dist)),
        "n_slew": int((dist >= _SLEW_MIN_DEG).sum()),
        "dist": dist,
        "overhead": over,
        "env": {"x": ex, "y": ey, "n": en},
    }


def _data_search_rate(calib: NightCalibration):
    import numpy as np

    # Search rate = sky area/hour surveyable while still reaching a TARGET-σ (3σ)
    # detection per field. A star measured at SNR s in a known exposure pins its
    # flux, so the time to reach target_snr is t_req = exp·(target_snr/s)² — valid
    # for *any* s>0, giving each star a search rate fov/(t_req+overhead)·3600.
    #
    # Naively this leaves two artifacts at the faint/slow end:
    #   1. A blank band below fov·3600/(max_exp·9 + overhead): a star at the
    #      detection threshold in the LONGEST exposure sets the slowest rate the
    #      exposure ladder can probe (≈157 deg²/h here for 10s max). Slower rates
    #      need longer exposures, not present in this data.
    #   2. A noise floor: a blank aperture still returns SNR≈1, so measured SNR
    #      never falls to 0 even well past the limit (it plateaus at ~0.9σ).
    # We remove the noise floor by subtracting it in QUADRATURE — the standard
    # debiasing when noise adds in quadrature to a significance measurement:
    #   s_signal = √(max(s² − s_noise², 0))
    # Bright stars are essentially unchanged (√(s²−n²)≈s); stars at the noise
    # level go to zero signal → rate 0. This fills the faint/slow band with the
    # real (small) search rates that ARE in the data in aggregate, while the curve
    # still reaches 0 at the true limit instead of a 1σ cliff or a noise plateau.
    # s_noise is measured per-night as the median SNR of stars well past the
    # limiting magnitude, where the flux is negligible and the SNR is pure noise.
    #
    # The per-field cadence is exposure + inter-frame overhead, where the overhead
    # is the fitted contiguous-grid value (readout + settle + one-FoV-width slew)
    # rather than a flat guess; it falls back to 1.0s if the slew fit is unusable.
    target_snr, min_exposure_s = 3.0, 0.1
    slew = _fit_slew_model(calib.frames)
    overhead_s = slew["grid_overhead_s"] if slew else 1.0

    m_l, s_l, e_l, fov_l, lim_l, lim50s = [], [], [], [], [], []
    for f in _zp_frames(calib.frames):
        if not f.exposure_time or not f.fov_sq_deg or not f.stars_mag:
            continue
        if f.limiting_magnitude_50 is not None:
            lim50s.append(f.limiting_magnitude_50)
        lim = f.limiting_magnitude_50 if f.limiting_magnitude_50 is not None else np.nan
        for m, s, iso in zip(f.stars_mag, f.stars_snr, _isolated_flags(f)):
            if not iso:
                continue
            m_l.append(m); s_l.append(s); e_l.append(f.exposure_time)
            fov_l.append(f.fov_sq_deg); lim_l.append(lim)
    if not m_l:
        return None
    mg = np.array(m_l); sn = np.array(s_l); ex = np.array(e_l)
    fv = np.array(fov_l); lim = np.array(lim_l)
    median_lim50 = float(np.median(lim50s)) if lim50s else None

    # Noise-floor SNR: median SNR of stars ≥2.5 mag past the limit (pure noise).
    faint = (mg > median_lim50 + 2.5) if median_lim50 is not None \
        else (mg > np.percentile(mg, 95))
    s_noise = float(np.median(sn[faint])) if int(faint.sum()) >= 100 else 0.9

    # Reject spuriously-high SNR (bright-star wings, bad matches): >5× predicted.
    with np.errstate(over="ignore", invalid="ignore"):
        snr_pred = 3.0 * 10 ** (0.4 * (lim - mg))
    consistent = np.isnan(lim) | (sn <= 5.0 * snr_pred)

    s_sig = np.sqrt(np.clip(sn ** 2 - s_noise ** 2, 0.0, None))
    detect = s_sig > 0
    t_req = np.clip(ex * (target_snr / np.where(detect, s_sig, 1.0)) ** 2,
                    min_exposure_s, None)
    rate_all = np.where(detect, fv / (t_req + overhead_s) * 3600.0, 0.0)

    mags = mg[consistent]
    rates = rate_all[consistent]
    bin_width = 0.5
    bin_edges = np.arange(math.floor(mags.min() / bin_width) * bin_width,
                          mags.max() + bin_width, bin_width)
    centers, medians, err_lo, err_hi = [], [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = rates[(mags >= lo) & (mags < hi)]
        if len(in_bin) < 5:
            continue
        med = float(np.median(in_bin))
        centers.append((lo + hi) / 2)
        medians.append(med)
        err_lo.append(med - float(np.percentile(in_bin, 16)))
        err_hi.append(float(np.percentile(in_bin, 84)) - med)
    return {
        "mags": mags, "rates": rates,
        "binned": {"x": centers, "y": medians, "err_lo": err_lo, "err_hi": err_hi},
        "median_lim50": median_lim50,
        "n_stars": int(len(mags)), "target_snr": target_snr,
        "noise_floor_snr": s_noise,
        "overhead_s": overhead_s,
        "overhead_model": None if slew is None else {
            "readout_s": slew["readout_s"], "settle_s": slew["settle_s"],
            "slew_rate_deg_s": slew["slew_rate_deg_s"],
            "fov_width_deg": slew["fov_width_deg"],
        },
    }


def _render_search_rate(d, meta, output_dir, plt, np) -> Path:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(d["mags"], d["rates"], alpha=0.3, s=10, color="lightgray",
               label="Individual stars")
    b = d["binned"]
    if b["x"]:
        ax.errorbar(b["x"], b["y"], yerr=[b["err_lo"], b["err_hi"]], fmt="o",
                    color="black", markersize=7, capsize=4, capthick=1.5,
                    elinewidth=1.5, alpha=0.85,
                    label="Binned data (median ± 1σ percentiles)")
    if d["median_lim50"] is not None:
        ax.axvline(d["median_lim50"], color="firebrick", linestyle="--",
                   linewidth=1.5, alpha=0.8,
                   label=f"median lim. mag (50%) = {d['median_lim50']:.1f}")
    ax.set_xlabel("Apparent Magnitude (Catalog)")
    ax.set_ylabel(f"Search Rate (deg²/hour to TARGET {d['target_snr']:.0f}σ)")
    om = d.get("overhead_model")
    if om and om.get("fov_width_deg"):
        oh_txt = (f"grid cadence overhead {d['overhead_s']:.1f}s = "
                  f"readout {om['readout_s']:.1f}s + settle {om['settle_s']:.1f}s "
                  f"+ {om['fov_width_deg']:.1f}° slew @ "
                  f"{om['slew_rate_deg_s']:.1f}°/s")
    else:
        oh_txt = f"overhead {d['overhead_s']:.1f}s (default; slew fit unavailable)"
    ax.set_title(f"{meta['night_id']}: search rate vs magnitude "
                 f"({d['n_stars']} isolated stars, sidereal; SNR debiased by "
                 f"{d.get('noise_floor_snr', 0):.2f}σ noise floor, scaled to "
                 f"TARGET {d['target_snr']:.0f}σ)\n{oh_txt}", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    return _save(fig, output_dir / "search_rate.png")


def _data_slew_model(calib: NightCalibration):
    return _fit_slew_model(calib.frames)


def _render_slew_model(d, meta, output_dir, plt, np) -> Path:
    dist = np.clip(np.asarray(d["dist"], float), 0.01, None)  # log axis: >0
    over = np.asarray(d["overhead"], float)
    readout, settle = d["readout_s"], d["settle_s"]
    rate, bias = d["slew_rate_deg_s"], d["bias_s"]
    fovw, grid = d["fov_width_deg"], d["grid_overhead_s"]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(dist, over, s=10, alpha=0.25, color="lightgray",
               label=f"consecutive frame pairs (n={d['n_pairs']})")
    env = d["env"]
    if env["x"]:
        ax.scatter(env["x"], env["y"], color="black", s=45, zorder=5,
                   label=f"lower envelope (p{_SLEW_ENV_PCTILE} per bin)")
    ax.axhline(readout, color="steelblue", ls=":", lw=1.6,
               label=f"readout floor = {readout:.1f}s (no slew)")
    xs = np.linspace(_SLEW_MIN_DEG, float(dist.max()), 200)
    ax.plot(xs, bias + xs / rate, color="firebrick", lw=2.2,
            label=(f"slew fit: {bias:.1f}s + sep / {rate:.1f}°/s   "
                   f"(settle {settle:.1f}s, n_slew={d['n_slew']})"))
    if fovw:
        ax.axvline(fovw, color="seagreen", ls="--", lw=1.5, alpha=0.8)
        ax.scatter([fovw], [grid], color="seagreen", s=160, marker="*", zorder=6,
                   label=f"contiguous-grid step {fovw:.1f}° → cadence "
                         f"overhead {grid:.1f}s")
    ax.set_xscale("log")
    ax.set_xlabel("Slew separation between consecutive frames (deg, alt/az)")
    ax.set_ylabel("Inter-frame overhead  =  Δt − exposure  (s)")
    ax.set_ylim(0, float(np.percentile(over, 98)))
    ax.set_title(f"{meta['night_id']}: inter-frame overhead model "
                 f"(readout + settle + slew)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    return _save(fig, output_dir / "slew_model.png")


# --- empirical PSF profile by seeing band -------------------------------------
# The per-frame FWHM is one number; the actual PSF *shape* (core sharpness, wing
# strength, x/y elongation) needs the pixels. We bin the night's sidereal frames
# into a few seeing bands by detection FWHM, and for each band take the single
# clearest (highest-ZP) frame with enough stars, median-stack subpixel-aligned
# stamps of its bright, isolated, unsaturated catalog stars from the raw FITS,
# then read off the azimuthally-averaged radial profile + x/y cuts. This reuses
# the approach proven in scripts/psf_vs_exposure.py, re-binned by seeing.
_PSF_STAMP_HALF = 30      # stamp is (2*half+1)² px
_PSF_SAT_PEAK = 40000.0   # raw ADU; reject saturated stars (well below 65535)
_PSF_ISO_RADIUS = 60.0    # px; no neighbor brighter than mag+2 within this
_PSF_MAX_STARS = 200      # stamps to stack per frame
_PSF_MIN_STARS_FRAME = 200  # need a well-populated field to pick from
_PSF_MIN_STAMPS = 20      # min stacked stars for a usable profile
_PSF_MIN_PEAK_SNR = 20.0  # per-stamp peak/background-noise floor: below this the
                          # max pixel is a noise spike, so peak-normalization
                          # corrupts the stack (e.g. faint stars in a 1s frame)
_PSF_FWHM_SANITY = 2.5    # reject a stack whose cut FWHM exceeds this × the
                          # frame's detection FWHM (cosmic rays / settling-trailed
                          # short frames stack to a meaningless broad blob)
_PSF_N_BANDS = 3


def _batch_result_paths(source_dir: str) -> dict[str, Path]:
    """{batch_id: result JSON path}, re-anchored on source_dir when the
    manifest's absolute paths are stale (drive remounted elsewhere)."""
    import glob as _glob
    md = json.loads((Path(source_dir) / "manifest.json").read_text())
    out: dict[str, Path] = {}
    for e in md.get("batches", []):
        bid = e.get("batch_id")
        if not bid:
            continue
        rp = e.get("result_path")
        p = Path(rp) if rp else None
        if p is None or not p.is_file():
            hits = _glob.glob(str(Path(source_dir) / "batches" / bid / "senpai_*.json"))
            if not hits:
                continue
            p = Path(hits[0])
        out[bid] = p
    return out


def _sidereal_frame_dict(batch_path: Path, index: int) -> dict | None:
    try:
        run = json.loads(batch_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for fr in run.get("sidereal_frames", []):
        if int(fr.get("index", -2)) == index:
            return fr
    return None


def _psf_stack_stamp(fits_path: str, catalog_stars: list, fwhm: float, half: int):
    """Median-stacked, peak-normalized PSF stamp from a frame's bright, isolated,
    unsaturated catalog stars. Returns (stamp2d, n_stars) or (None, 0)."""
    import numpy as np
    from astropy.io import fits
    from scipy import ndimage
    from scipy.spatial import cKDTree

    try:
        data = fits.getdata(fits_path).astype(np.float64)
    except Exception:
        return None, 0
    h, w = data.shape
    keep = [s for s in catalog_stars
            if s.get("x") is not None and s.get("y") is not None]
    if len(keep) < 20:
        return None, 0
    xy = np.array([(s["x"], s["y"]) for s in keep])
    mags = np.array([s.get("magnitude") if s.get("magnitude") is not None
                     else np.inf for s in keep])
    tree = cKDTree(xy)
    n = 2 * half + 1
    stamps = []
    for i in np.argsort(mags):              # brightest first
        if len(stamps) >= _PSF_MAX_STARS:
            break
        x, y = xy[i]
        if not (half + 2 < x < w - half - 2 and half + 2 < y < h - half - 2):
            continue
        neigh = tree.query_ball_point((x, y), _PSF_ISO_RADIUS)
        if any(j != i and mags[j] < mags[i] + 2.0 for j in neigh):
            continue
        xi, yi = int(round(x)), int(round(y))
        st = data[yi - half:yi + half + 1, xi - half:xi + half + 1].copy()
        if st.shape != (n, n):
            continue
        ring = np.concatenate([st[0:4].ravel(), st[-4:].ravel(),
                               st[:, 0:4].ravel(), st[:, -4:].ravel()])
        noise = float(np.std(ring))
        st -= np.median(ring)
        # Cosmic rays / hot pixels are single-pixel spikes that can top the real
        # star peak (common in short exposures); a 3×3 median filter is immune to
        # them, so we take the peak / SNR / centroid from the filtered stamp. The
        # raw stamp is what gets stacked — residual spikes wash out in the median.
        sm = ndimage.median_filter(st, size=3)
        peak = float(sm.max())
        if peak <= 0 or peak > _PSF_SAT_PEAK:
            continue
        if noise <= 0 or peak < _PSF_MIN_PEAK_SNR * noise:
            continue                        # noise-dominated stamp
        cy, cx = ndimage.center_of_mass(np.clip(sm, 0, None))
        if not (np.isfinite(cx) and np.isfinite(cy)):
            continue
        if abs(cy - half) > fwhm or abs(cx - half) > fwhm:
            continue                        # centroid far off — blend/artifact
        st = ndimage.shift(st, (half - cy, half - cx), order=3, mode="nearest")
        st /= peak
        stamps.append(st)
    if len(stamps) < _PSF_MIN_STAMPS:
        return None, 0
    return np.median(np.stack(stamps), axis=0), len(stamps)


def _radial_profile(stamp, half, np, rstep=0.5):
    """Azimuthally-averaged (ring-median) radial profile, peak-normalized."""
    n = stamp.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - half, yy - half)
    edges = np.arange(0.0, half + rstep, rstep)
    r, prof = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (rr >= lo) & (rr < hi)
        if m.any():
            r.append((lo + hi) / 2)
            prof.append(float(np.median(stamp[m])))
    prof = np.array(prof)
    if prof.size and prof.max() > 0:
        prof = prof / prof.max()
    return np.array(r), prof


def _cut_fwhm(profile, np) -> float:
    """FWHM from half-max crossings of a 1D cut through the peak."""
    profile = np.asarray(profile)
    half = profile.max() / 2.0
    above = np.where(profile >= half)[0]
    if len(above) < 2:
        return float("nan")
    lo, hi = int(above[0]), int(above[-1])
    left = (lo - (profile[lo] - half) / (profile[lo] - profile[lo - 1])
            if lo > 0 and profile[lo] != profile[lo - 1] else float(lo))
    right = (hi + (profile[hi] - half) / (profile[hi] - profile[hi + 1])
             if hi < len(profile) - 1 and profile[hi] != profile[hi + 1]
             else float(hi))
    return float(right - left)


def _wcs_sky_axes(wcs_dict: dict):
    """Unit vectors in pixel (x, y) space pointing East (+RA) and North (+Dec)
    at the frame center, from the solved WCS header dict. Returns
    (east_unit, north_unit) or None. Lets us cut the PSF along sky axes so an
    elongation reads directly as RA vs Dec tracking error rather than detector
    x/y (this camera is rotated ~35° + flipped relative to the sky)."""
    if not wcs_dict:
        return None
    try:
        import numpy as np
        from astropy.io import fits
        from astropy.wcs import WCS

        hdr = fits.Header()
        for k, v in wcs_dict.items():
            if v is not None:
                hdr[k] = v
        w = WCS(hdr)
        x0 = float(w.wcs.crpix[0]) - 1.0
        y0 = float(w.wcs.crpix[1]) - 1.0
        ra0, dec0 = (float(c) for c in w.all_pix2world(x0, y0, 0))
        dd = 1.0 / 3600.0  # 1 arcsec step
        xn, yn = (float(c) for c in w.all_world2pix(ra0, dec0 + dd, 0))
        xe, ye = (float(c) for c in w.all_world2pix(
            ra0 + dd / math.cos(math.radians(dec0)), dec0, 0))
        north = np.array([xn - x0, yn - y0])
        east = np.array([xe - x0, ye - y0])
        nn, ne = np.linalg.norm(north), np.linalg.norm(east)
        if not (np.isfinite(nn) and np.isfinite(ne) and nn > 0 and ne > 0):
            return None
        return east / ne, north / nn
    except Exception:
        return None


def _sample_line(stamp, half, unit, np):
    """Sample the stamp along a line through its center in pixel direction
    ``unit`` (x, y), one sample per pixel from -half to +half."""
    from scipy.ndimage import map_coordinates

    t = np.arange(-half, half + 1.0)
    xs = half + t * unit[0]
    ys = half + t * unit[1]
    return map_coordinates(stamp, [ys, xs], order=1, mode="constant", cval=0.0)


def _data_psf_profile(calib: NightCalibration):
    import numpy as np

    if not calib.source_dir:
        return None
    cands = [f for f in calib.frames
             if f.track_mode == "sidereal" and f.fwhm_px and f.zero_point
             and f.batch_id and f.frame_index is not None and f.frame_index >= 0
             and (f.n_stars or 0) >= _PSF_MIN_STARS_FRAME]
    if len(cands) < 2 * _PSF_N_BANDS:
        return None
    try:
        batch_paths = _batch_result_paths(calib.source_dir)
    except (OSError, json.JSONDecodeError):
        return None
    if not batch_paths:
        return None

    fwhms = np.array([f.fwhm_px for f in cands])
    edges = np.percentile(fwhms, np.linspace(0, 100, _PSF_N_BANDS + 1))
    half = _PSF_STAMP_HALF
    bands = []
    for i in range(_PSF_N_BANDS):
        lo, hi = float(edges[i]), float(edges[i + 1])
        in_band = [f for f in cands if lo <= f.fwhm_px <=
                   (hi if i == _PSF_N_BANDS - 1 else hi - 1e-9)]
        if not in_band:
            continue
        in_band.sort(key=lambda f: -f.zero_point)   # clearest first
        picked = None
        for f in in_band[:12]:                       # try a few before giving up
            bp = batch_paths.get(f.batch_id)
            if not bp or not bp.is_file():
                continue
            fd = _sidereal_frame_dict(bp, f.frame_index)
            if not fd:
                continue
            fp = fd.get("original_frame_path")
            stars = (fd.get("starfield") or {}).get("catalog_stars") or []
            if not fp or not Path(fp).is_file() or len(stars) < _PSF_MIN_STARS_FRAME:
                continue
            stamp, n = _psf_stack_stamp(fp, stars, f.fwhm_px, half)
            if stamp is None or stamp.max() <= 0:
                continue
            stamp = stamp / stamp.max()      # peak = 1 for radial & cuts alike
            # Cut along sky axes (RA = East, Dec = North) when the WCS is
            # available, so an elongation reads as a tracking error in RA vs Dec;
            # else fall back to detector x/y.
            sky = _wcs_sky_axes((fd.get("starfield") or {}).get("wcs") or {})
            if sky is not None:
                east_u, north_u = sky
                cut_ra = _sample_line(stamp, half, east_u, np)
                cut_dec = _sample_line(stamp, half, north_u, np)
                axes_kind = "sky"
            else:
                east_u = north_u = None
                cut_ra = stamp[half, :]
                cut_dec = stamp[:, half]
                axes_kind = "pixel"
            fwhm_ra = _cut_fwhm(cut_ra, np)
            fwhm_dec = _cut_fwhm(cut_dec, np)
            # Reject implausible stacks (cosmic rays / settling-trailed short
            # frames) so the band falls through to a clean frame.
            if not (np.isfinite(fwhm_ra) and np.isfinite(fwhm_dec)
                    and 0 < max(fwhm_ra, fwhm_dec) <= _PSF_FWHM_SANITY * f.fwhm_px):
                continue
            r, radial = _radial_profile(stamp, half, np)
            picked = {
                "fwhm_lo": lo, "fwhm_hi": hi, "fwhm_det": float(f.fwhm_px),
                "fwhm_ra": float(fwhm_ra), "fwhm_dec": float(fwhm_dec),
                "axes_kind": axes_kind,
                "n_stars": int(n), "zp": float(f.zero_point),
                "exposure": f.exposure_time,
                "timestamp": f.timestamp.isoformat() if f.timestamp else None,
                "axis": (np.arange(stamp.shape[0]) - half).tolist(),
                "cut_ra": cut_ra.tolist(), "cut_dec": cut_dec.tolist(),
                "r": r.tolist(), "radial": radial.tolist(),
                "stamp2d": np.round(stamp, 4).tolist(),
                "east_unit": (None if east_u is None
                              else [float(east_u[0]), float(east_u[1])]),
                "north_unit": (None if north_u is None
                               else [float(north_u[0]), float(north_u[1])]),
            }
            break
        if picked is None:
            continue
        bands.append(picked)
    if not bands:
        return None
    psc = next((f.pixel_scale_arcsec for f in cands if f.pixel_scale_arcsec), None)
    return {"half": half, "bands": bands, "pixel_scale_arcsec": psc}


def _render_psf_profile(d, meta, output_dir, plt, np) -> Path:
    bands = d["bands"]
    half = d["half"]
    psc = d.get("pixel_scale_arcsec")
    nb = len(bands)
    colors = plt.cm.viridis(np.linspace(0.12, 0.85, nb))
    sky = all(b.get("axes_kind") == "sky" for b in bands)
    a_name, b_name = ("RA", "Dec") if sky else ("x", "y")
    win = min(half, max(12.0, max(
        3.0 * max(b["fwhm_ra"], b["fwhm_dec"]) for b in bands)))

    # Layout: top row = one 2D heatmap per band; bottom row = radial (left half)
    # + 1D cuts (right half). 2*nb columns so the bottom splits evenly.
    fig = plt.figure(figsize=(max(13.0, 4.4 * nb), 10.5))
    gs = fig.add_gridspec(2, 2 * nb, height_ratios=[1.05, 1.0])

    for i, (b, c) in enumerate(zip(bands, colors)):
        ax = fig.add_subplot(gs[0, 2 * i:2 * i + 2])
        stamp = np.asarray(b["stamp2d"])
        ext = [-half, half, -half, half]
        ax.imshow(np.arcsinh(np.clip(stamp, 0, None) / 0.02), origin="lower",
                  extent=ext, cmap="inferno")
        grid = np.linspace(-half, half, stamp.shape[0])
        ax.contour(grid, grid, stamp, levels=[0.5], colors="cyan", linewidths=1.0)
        # N/E arrows so the heatmap is readable in sky orientation.
        if b.get("north_unit") and b.get("east_unit"):
            L = 0.7 * win
            for u, name, col in ((b["north_unit"], "N", "white"),
                                 (b["east_unit"], "E", "deepskyblue")):
                ax.annotate("", xy=(L * u[0], L * u[1]), xytext=(0, 0),
                            arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
                ax.text(L * 1.12 * u[0], L * 1.12 * u[1], name, color=col,
                        fontsize=9, ha="center", va="center")
        ax.set_xlim(-win, win)
        ax.set_ylim(-win, win)
        sec = f", {b['fwhm_det']*psc:.1f}\"" if psc else ""
        ax.set_title(f"FWHM {b['fwhm_lo']:.1f}–{b['fwhm_hi']:.1f}px{sec}\n"
                     f"{b['exposure']:.0f}s, {a_name}/{b_name}="
                     f"{b['fwhm_ra']:.1f}/{b['fwhm_dec']:.1f}px, n={b['n_stars']}",
                     fontsize=9)
        ax.set_xlabel("Δx (px)")
        if i == 0:
            ax.set_ylabel("Δy (px)")

    axr = fig.add_subplot(gs[1, :nb])
    axc = fig.add_subplot(gs[1, nb:])
    for b, c in zip(bands, colors):
        lbl = (f"FWHM {b['fwhm_lo']:.1f}–{b['fwhm_hi']:.1f}px  "
               f"{a_name}/{b_name}={b['fwhm_ra']:.1f}/{b['fwhm_dec']:.1f}px")
        axr.plot(b["r"], np.clip(b["radial"], 1e-4, None), color=c, lw=2, label=lbl)
        axc.plot(b["axis"], b["cut_ra"], color=c, lw=2.0, ls="-")
        axc.plot(b["axis"], b["cut_dec"], color=c, lw=1.5, ls="--")

    axr.set_yscale("log")
    axr.set_ylim(1e-3, 1.3)
    axr.set_xlim(0, half)
    axr.set_xlabel("radius (px)")
    axr.set_ylabel("normalized flux (peak = 1)")
    axr.set_title("azimuthally-averaged radial profile")
    axr.grid(True, which="both", alpha=0.3)
    axr.legend(fontsize=8, loc="upper right")

    axc.axhline(0.5, color="gray", ls=":", lw=1)
    axc.set_xlim(-win, win)
    axc.set_ylim(-0.05, 1.05)
    axc.set_xlabel("Δ from center along sky axis (px)" if sky
                   else "Δ from center (px)")
    axc.set_ylabel("normalized flux")
    axc.set_title(f"{a_name} cut (E–W, solid) & {b_name} cut (N–S, dashed)"
                  if sky else f"{a_name} cut (solid) & {b_name} cut (dashed)")
    axc.grid(True, alpha=0.3)

    scale_txt = f"   plate scale {psc:.2f}\"/px" if psc else ""
    axis_txt = ("RA/Dec → tracking-error direction" if sky else "detector x/y")
    fig.suptitle(f"{meta['night_id']}: empirical PSF by seeing band — 2D stack, "
                 f"radial & {axis_txt} cuts (median of isolated stars){scale_txt}",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save(fig, output_dir / "psf_profile.png")


_PLOT_BUILDERS.update({
    "extinction_curve": (_data_extinction_curve, _render_extinction_curve),
    "limiting_magnitude_hist": (
        _data_limiting_magnitude_hist, _render_limiting_magnitude_hist),
    "alt_az_coverage": (_data_alt_az_coverage, _render_alt_az_coverage),
    "zp_drift": (_data_zp_drift, _render_zp_drift),
    "search_rate": (_data_search_rate, _render_search_rate),
    "slew_model": (_data_slew_model, _render_slew_model),
    "psf_profile": (_data_psf_profile, _render_psf_profile),
})


def _data_snr_vs_exposure(calib: NightCalibration):
    import numpy as np

    band = _clear_sky_zp_band(calib.frames)
    if band is None:
        return None
    zp_mode, zp_sig = band
    ext_k = {filt: fit.k for filt, fit in calib.extinction_per_filter.items()}
    k_default = float(np.median(list(ext_k.values()))) if ext_k else 0.0
    min_meas_snr = 3.0
    by_task: dict[str, list] = {}
    for f in _zp_frames(calib.frames):
        if (f.zero_point is None or not f.exposure_time or not f.stars_mag
                or abs(f.zero_point - zp_mode) > zp_sig or not _moon_ok(f)):
            continue
        k = ext_k.get(f.filter_name or "unknown", k_default)
        corr = (10 ** (0.4 * k * (f.airmass - 1.0))
                if f.airmass is not None else 1.0)
        by_task.setdefault(_frame_task(f), []).extend(
            (f.exposure_time, m, s * corr)
            for m, s, iso in zip(f.stars_mag, f.stars_snr, _isolated_flags(f))
            if s >= min_meas_snr and iso and _snr_consistent(s, m, f)
        )
    order = [t for t in ("coverage", "photometric", "calsats", "other")
             if by_task.get(t)]
    if not order:
        return None
    all_pts = [p for t in order for p in by_task[t]]
    a_exp = np.array([p[0] for p in all_pts])
    a_mag = np.array([p[1] for p in all_pts])
    std_exps = list(range(max(1, math.floor(a_exp.min())),
                          math.ceil(a_exp.max()) + 1))
    bins = list(range(math.floor(a_mag.min()), math.ceil(a_mag.max())))

    def _series(pts):
        e = np.array([p[0] for p in pts])
        m = np.array([p[1] for p in pts])
        s = np.array([p[2] for p in pts])
        out = []
        for lo in bins:
            in_bin = (m >= lo) & (m < lo + 1)
            if in_bin.sum() < 5:
                continue
            xs, meds, e_lo, e_hi = [], [], [], []
            for t in std_exps:
                sel = s[in_bin & (np.abs(e - t) <= 0.5)]
                if len(sel) < 20:
                    continue
                sem = 1.253 * float(np.std(sel)) / math.sqrt(len(sel))
                xs.append(t)
                meds.append(float(np.median(sel)))
                e_lo.append(sem)
                e_hi.append(sem)
            if len(xs) >= 2:
                out.append({"bin": lo, "x": xs, "y": meds,
                            "e_lo": e_lo, "e_hi": e_hi})
        return out

    pooled_pts = [p for t in ("coverage", "photometric")
                  for p in by_task.get(t, [])]
    return {
        "std_exps": std_exps, "bins": bins, "order": order,
        "faceted": {tk: _series(by_task[tk]) for tk in order},
        "pooled": _series(pooled_pts) if pooled_pts else [],
    }


def _render_snr_vs_exposure(d, meta, output_dir, plt, np) -> list:
    from matplotlib.ticker import FuncFormatter, NullFormatter

    std_exps, bins, order = d["std_exps"], d["bins"], d["order"]
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, max(len(bins), 1)))
    color_of = {lo: colors[i] for i, lo in enumerate(bins)}
    paths = []

    fig, axes = plt.subplots(1, len(order), figsize=(5.5 * len(order), 6.5),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, tk in zip(axes, order):
        for ser in d["faceted"].get(tk, []):
            ax.errorbar(ser["x"], ser["y"], yerr=[ser["e_lo"], ser["e_hi"]],
                        fmt="o-", color=color_of[ser["bin"]], alpha=0.8,
                        linewidth=1.5, markersize=5, capsize=3,
                        label=f"{ser['bin'] + 0.5:.1f}")
        ax.set_yscale("log")
        ax.set_xticks(std_exps)
        ax.set_xlabel("Exposure Time [s]")
        ax.set_title(tk)
        ax.grid(True, alpha=0.3, which="both")
    axes[0].set_ylabel("SNR (normalized to airmass = 1)")
    for ax in axes:
        if ax.lines:
            ax.legend(loc="lower right", fontsize=7, title=r"m$_G$", ncol=2)
    fig.suptitle(
        f"{meta['night_id']}: SNR vs exposure by task "
        f"(weather-masked, airmass-normalized, Moon>{_MOON_SEP_MIN_DEG:.0f}°)")
    fig.tight_layout()
    paths.append(_save(fig, output_dir / "snr_vs_exposure_by_task.png"))

    if d["pooled"]:
        figg, axg = plt.subplots(figsize=(10, 7))
        for ser in d["pooled"]:
            axg.errorbar(ser["x"], ser["y"], yerr=[ser["e_lo"], ser["e_hi"]],
                         fmt="o-", color=color_of[ser["bin"]], alpha=0.8,
                         linewidth=1.5, markersize=5, capsize=3,
                         label=f"{ser['bin'] + 0.5:.1f}")
        axg.set_yscale("log")
        axg.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        axg.yaxis.set_minor_formatter(NullFormatter())
        axg.set_ylim(top=axg.get_ylim()[1] * 1.8)
        axg.set_xticks(std_exps)
        axg.set_xlabel("Exposure Time [seconds]")
        axg.set_ylabel("SNR (normalized to airmass = 1)")
        axg.set_title(
            f"{meta['night_id']}: SNR vs exposure  (coverage+photometric, "
            f"weather-masked, airmass-norm, Moon>{_MOON_SEP_MIN_DEG:.0f}°)")
        axg.grid(True, alpha=0.3, which="both")
        axg.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8,
                   title=r"m$_G$", ncol=1)
        paths.append(_save(figg, output_dir / "snr_vs_exposure_by_magnitude.png"))
    return paths


def _data_snr_vs_mag_weathermasked(calib: NightCalibration):
    import numpy as np

    band = _clear_sky_zp_band(calib.frames)
    if band is None:
        return None
    zp_mode, zp_sig = band
    ext_k = {filt: fit.k for filt, fit in calib.extinction_per_filter.items()}
    k_default = float(np.median(list(ext_k.values()))) if ext_k else 0.0
    min_meas_snr = 3.0
    cal_tasks = ("coverage", "photometric")
    mag_pts = []
    for f in _zp_frames(calib.frames):
        if (f.zero_point is None or not f.exposure_time or not f.stars_mag
                or abs(f.zero_point - zp_mode) > zp_sig
                or _frame_task(f) not in cal_tasks or not _moon_ok(f)):
            continue
        kk = ext_k.get(f.filter_name or "unknown", k_default)
        corr = (10 ** (0.4 * kk * (f.airmass - 1.0))
                if f.airmass is not None else 1.0)
        mag_pts.extend(
            (f.exposure_time, m, s * corr)
            for m, s, iso in zip(f.stars_mag, f.stars_snr, _isolated_flags(f))
            if s >= min_meas_snr and iso and _snr_consistent(s, m, f)
        )
    if not mag_pts:
        return None
    exps = np.array([p[0] for p in mag_pts])
    mags = np.array([p[1] for p in mag_pts])
    snrs = np.array([p[2] for p in mag_pts])
    std_exps = list(range(max(1, math.floor(exps.min())),
                          math.ceil(exps.max()) + 1))
    mgrid = np.arange(math.floor(mags.min()), math.ceil(mags.max()), 0.5)
    lines = []
    for t in std_exps:
        sel_t = np.abs(exps - t) <= 0.5
        if sel_t.sum() < 20:
            continue
        xs, ys = [], []
        for mlo in mgrid:
            cell = snrs[sel_t & (mags >= mlo) & (mags < mlo + 0.5)]
            if len(cell) >= 5:
                xs.append(float(mlo + 0.25))
                ys.append(float(np.median(cell)))
        if len(xs) >= 3:
            lines.append({"exp": int(t), "x": xs, "y": ys})
    lims = [f.limiting_magnitude_50 for f in _zp_frames(calib.frames)
            if f.limiting_magnitude_50 is not None and f.zero_point is not None
            and abs(f.zero_point - zp_mode) <= zp_sig
            and _frame_task(f) in cal_tasks and _moon_ok(f)]
    lim50 = None
    if lims:
        lim50 = {"med": float(np.median(lims)),
                 "lo": float(np.percentile(lims, 16)),
                 "hi": float(np.percentile(lims, 84))}
    return {"std_exps": std_exps, "lines": lines, "lim50": lim50,
            "min_meas_snr": min_meas_snr, "zp_mode": zp_mode, "zp_sig": zp_sig}


def _render_snr_vs_mag_weathermasked(d, meta, output_dir, plt, np) -> Path:
    std_exps = d["std_exps"]
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, max(len(std_exps), 1)))
    color_of = {t: cmap[i] for i, t in enumerate(std_exps)}
    for ln in d["lines"]:
        ax.plot(ln["x"], ln["y"], "o-", color=color_of[ln["exp"]], ms=4,
                lw=1.5, alpha=0.85, label=f"{int(ln['exp'])}s")
    lim = d["lim50"]
    if lim is not None:
        ax.axvspan(lim["lo"], lim["hi"], color="red", alpha=0.10)
        ax.axvline(lim["lo"], color="red", ls=":", lw=1, alpha=0.6)
        ax.axvline(lim["hi"], color="red", ls=":", lw=1, alpha=0.6)
        ax.axvline(lim["med"], color="red", ls="--", lw=1.8,
                   label=f"lim50 = {lim['med']:.2f} "
                         f"(16/84: {lim['lo']:.2f}–{lim['hi']:.2f})")
    ax.axhline(d["min_meas_snr"], color="gray", ls=":", lw=1,
               label=f"SNR = {d['min_meas_snr']:.0f}")
    ax.set_yscale("log")
    ax.set_xlabel("Gaia G magnitude")
    ax.set_ylabel("SNR (normalized to airmass = 1)")
    ax.set_title(
        f"{meta['night_id']}: SNR vs magnitude  (coverage+photometric, "
        f"weather-masked ZP {d['zp_mode']:.2f}±{d['zp_sig']:.2f}, "
        f"Moon>{_MOON_SEP_MIN_DEG:.0f}°)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9, title="exposure")
    return _save(fig, output_dir / "snr_vs_mag_weathermasked.png")


_PLOT_BUILDERS.update({
    "snr_vs_exposure": (_data_snr_vs_exposure, _render_snr_vs_exposure),
    "snr_vs_mag_weathermasked": (
        _data_snr_vs_mag_weathermasked, _render_snr_vs_mag_weathermasked),
})


def _data_moon_az_el(calib: NightCalibration):
    import numpy as np

    moon_frames = [f for f in calib.frames
                   if f.moon_sep_deg is not None and f.altitude_deg is not None
                   and f.azimuth_deg is not None]
    if not moon_frames:
        return None
    track = None
    try:
        from astropy.coordinates import AltAz, EarthLocation, get_body
        from astropy.time import Time
        from astropy import units as u
        site = calib.site or {}
        loc = EarthLocation(
            lat=site["latitude"] * u.deg, lon=site["longitude"] * u.deg,
            height=(site.get("altitude_km") or 0.0) * 1000.0 * u.m)
        tss = [f.timestamp for f in moon_frames if f.timestamp]
        t0, t1 = min(tss), max(tss)
        tt = Time([t0 + (t1 - t0) * i / 60 for i in range(61)], scale="utc")
        mtrk = get_body("moon", tt, loc).transform_to(
            AltAz(obstime=tt, location=loc))
        up = mtrk.alt.deg > 0
        track = {"theta": list(np.radians(mtrk.az.deg[up])),
                 "r": list(90 - mtrk.alt.deg[up])}
    except Exception as e:
        logger.debug("moon track overlay skipped: %s", e)
    return {
        "theta": [math.radians(f.azimuth_deg) for f in moon_frames],
        "r": [90 - f.altitude_deg for f in moon_frames],
        "sep": [f.moon_sep_deg for f in moon_frames],
        "track": track,
    }


def _render_moon_az_el(d, meta, output_dir, plt, np) -> Path:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    if d["track"] is not None:
        ax.plot(d["track"]["theta"], d["track"]["r"], "-", color="gold", lw=3,
                label="Moon track", zorder=3)
    sc = ax.scatter(d["theta"], d["r"], c=d["sep"], s=14, cmap="viridis",
                    alpha=0.75, zorder=2)
    ax.set_ylim(0, 90)
    ax.set_yticks([15, 30, 45, 60, 75])
    ax.set_yticklabels([f"{90 - r}°" for r in [15, 30, 45, 60, 75]])
    plt.colorbar(sc, ax=ax, pad=0.10, shrink=0.7, label="Moon separation (°)")
    ax.legend(loc="lower left", fontsize=8)
    mi = meta.get("moon_illumination")
    illum = f"{100 * mi:.0f}% illuminated" if mi is not None else ""
    ax.set_title(f"{meta['night_id']}: pointings + Moon ({illum})", pad=20)
    return _save(fig, output_dir / "moon_az_el.png")


def _data_snr_vs_moon_distance(calib: NightCalibration):
    import numpy as np

    band = _clear_sky_zp_band(calib.frames)
    if band is None or not any(f.moon_sep_deg is not None for f in calib.frames):
        return None
    zp_mode, zp_sig = band
    ext_k = {filt: fit.k for filt, fit in calib.extinction_per_filter.items()}
    k_default = float(np.median(list(ext_k.values()))) if ext_k else 0.0
    min_meas_snr = 3.0
    FAR = 60.0

    def _norm_stars(f):
        kk = ext_k.get(f.filter_name or "unknown", k_default)
        corr = (10 ** (0.4 * kk * (f.airmass - 1.0))
                if f.airmass is not None else 1.0)
        for m, s, iso in zip(f.stars_mag, f.stars_snr, _isolated_flags(f)):
            if iso and m and s and s >= min_meas_snr and _snr_consistent(s, m, f):
                yield round(f.exposure_time), m, s * corr

    masked = [f for f in _zp_frames(calib.frames)
              if f.zero_point is not None and f.exposure_time and f.stars_mag
              and abs(f.zero_point - zp_mode) <= zp_sig
              and f.moon_sep_deg is not None]
    cells: dict[tuple, list] = {}
    for f in masked:
        if f.moon_sep_deg < FAR:
            continue
        for ex, m, s in _norm_stars(f):
            cells.setdefault((ex, round(m * 2) / 2), []).append(s)
    base_cell = {c: float(np.median(v)) for c, v in cells.items() if len(v) >= 5}
    sep_arr, dmag_arr, mag_arr = [], [], []
    for f in masked:
        for ex, m, s in _norm_stars(f):
            b = base_cell.get((ex, round(m * 2) / 2))
            if b and b > 0:
                sep_arr.append(f.moon_sep_deg)
                dmag_arr.append(2.5 * math.log10(s / b))
                mag_arr.append(m)
    near_lims = [f.limiting_magnitude_50 for f in masked
                 if f.moon_sep_deg < 30 and f.limiting_magnitude_50 is not None]
    m_safe = (float(np.median(near_lims)) - 2.5) if near_lims else None
    if len(sep_arr) < 50:
        return None
    sep_arr = np.array(sep_arr)
    dmag_arr = np.array(dmag_arr)
    mag_arr = np.array(mag_arr)
    edges = np.arange(0, math.ceil(sep_arr.max() / 10) * 10 + 10, 10)

    def _series(mask, name, dot_color, dot_alpha, bin_color, fit_color):
        cx, cy, e_lo, e_hi = [], [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            v = dmag_arr[mask & (sep_arr >= lo) & (sep_arr < hi)]
            if len(v) >= 20:
                md = float(np.median(v))
                cx.append((lo + hi) / 2)
                cy.append(md)
                e_lo.append(md - float(np.percentile(v, 16)))
                e_hi.append(float(np.percentile(v, 84)) - md)
        fit = None
        if len(cx) >= 2:
            slope, intercept = np.polyfit(cx, cy, 1)
            fit = {"slope": float(slope), "intercept": float(intercept)}
        return {"name": name, "sep": list(sep_arr[mask]),
                "dmag": list(dmag_arr[mask]),
                "binned": {"x": cx, "y": cy, "e_lo": e_lo, "e_hi": e_hi},
                "fit": fit, "dot_color": dot_color, "dot_alpha": dot_alpha,
                "bin_color": bin_color, "fit_color": fit_color}

    series = [_series(np.ones(len(sep_arr), bool), "all stars",
                      "lightgray", 0.12, "black", "red")]
    if m_safe is not None and (mag_arr < m_safe).sum() >= 50:
        series.append(_series(
            mag_arr < m_safe, f"bright m<{m_safe:.1f} (complete at all sep)",
            "tab:blue", 0.08, "tab:blue", "tab:blue"))
    return {"series": series}


def _render_snr_vs_moon_distance(d, meta, output_dir, plt, np) -> Path:
    fig, ax = plt.subplots(figsize=(10, 7))
    for s in d["series"]:
        ax.scatter(s["sep"], s["dmag"], s=6, alpha=s["dot_alpha"],
                   color=s["dot_color"], zorder=1)
        b = s["binned"]
        if b["x"]:
            ax.errorbar(b["x"], b["y"], yerr=[b["e_lo"], b["e_hi"]], fmt="o",
                        color=s["bin_color"], ms=7, capsize=4, capthick=1.5,
                        elinewidth=1.5, zorder=5)
        if s["fit"] is not None:
            slope, intercept = s["fit"]["slope"], s["fit"]["intercept"]
            lx = np.linspace(min(b["x"]), max(b["x"]), 50)
            ax.plot(lx, slope * lx + intercept, "-", color=s["fit_color"],
                    lw=2, alpha=0.85,
                    label=f"{s['name']}: {-slope * 10:+.2f} mag / 10° closer")
    ax.axhline(0, color="gray", ls=":")
    mi = meta.get("moon_illumination")
    illum = f"{100 * mi:.0f}%" if mi is not None else "?"
    ax.set_xlabel("Moon separation (°)")
    ax.set_ylabel("ΔSNR [mag]  (0 = far-Moon baseline; negative = penalty)")
    ax.set_title(f"{meta['night_id']}: SNR penalty vs Moon distance "
                 f"(Moon {illum}; all-stars vs bright-complete = "
                 f"survivorship check)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, output_dir / "snr_vs_moon_distance.png")


def _usable_cal_frames(calib, zp_mode, zp_sig):
    """Weather-masked calibration-taskset frames with Moon geometry (shared by
    the two fixed-magnitude Moon plots)."""
    return [f for f in _zp_frames(calib.frames)
            if f.zero_point is not None and f.exposure_time and f.stars_mag
            and abs(f.zero_point - zp_mode) <= zp_sig
            and _frame_task(f) in ("coverage", "photometric")
            and f.moon_sep_deg is not None]


def _best_sampled_mag(usable, min_meas_snr):
    mag_tot: dict[int, int] = {}
    for f in usable:
        for m, s, iso in zip(f.stars_mag, f.stars_snr, _isolated_flags(f)):
            if iso and m and s and s >= min_meas_snr:
                mag_tot[int(m)] = mag_tot.get(int(m), 0) + 1
    return max(mag_tot, key=mag_tot.get) if mag_tot else None


def _data_snr_vs_exposure_6s_explained(calib: NightCalibration):
    import numpy as np

    band = _clear_sky_zp_band(calib.frames)
    if band is None or not any(f.moon_sep_deg is not None for f in calib.frames):
        return None
    zp_mode, zp_sig = band
    ext_k = {filt: fit.k for filt, fit in calib.extinction_per_filter.items()}
    k_default = float(np.median(list(ext_k.values()))) if ext_k else 0.0
    min_meas_snr = 3.0
    usable = _usable_cal_frames(calib, zp_mode, zp_sig)
    best_mag = _best_sampled_mag(usable, min_meas_snr)
    if best_mag is None:
        return None
    prog = {"coverage": {}, "photometric": {}}
    for f in usable:
        tk = _frame_task(f)
        if tk not in prog:
            continue
        corr = (10 ** (0.4 * ext_k.get(f.filter_name or "unknown", k_default)
                       * (f.airmass - 1.0)) if f.airmass is not None else 1.0)
        ex = round(f.exposure_time)
        for m, s, iso in zip(f.stars_mag, f.stars_snr, _isolated_flags(f)):
            if (iso and m and s and s >= min_meas_snr
                    and _snr_consistent(s, m, f) and best_mag <= m < best_mag + 1):
                prog[tk].setdefault(ex, []).append(s * corr)
    if not any(len(dd) >= 2 for dd in prog.values()):
        return None
    out_prog = {}
    for tk, dd in prog.items():
        exps = sorted(ex for ex in dd if len(dd[ex]) >= 10)
        if len(exps) < 2:
            continue
        out_prog[tk] = {
            "exps": exps,
            "med": [float(np.median(dd[ex])) for ex in exps],
            "sem": [1.253 * float(np.std(dd[ex])) / math.sqrt(len(dd[ex]))
                    for ex in exps],
        }
    annotation = None
    if prog["coverage"].get(1) and prog["photometric"].get(1):
        c1 = float(np.median(prog["coverage"][1]))
        p1 = float(np.median(prog["photometric"][1]))
        annotation = {"p1": p1, "delta": 2.5 * math.log10(c1 / p1)}
    xticks = sorted({ex for dd in prog.values() for ex in dd
                     if len(dd[ex]) >= 10})
    return {"best_mag": best_mag, "prog": out_prog,
            "annotation": annotation, "xticks": xticks}


def _render_snr_vs_exposure_6s_explained(d, meta, output_dir, plt, np) -> Path:
    pcolor = {"coverage": "tab:blue", "photometric": "tab:orange"}
    fig, ax = plt.subplots(figsize=(10, 7))
    for tk, dd in d["prog"].items():
        exps, med, sem = dd["exps"], dd["med"], dd["sem"]
        ax.errorbar(exps, med, yerr=sem, fmt="o", color=pcolor[tk], ms=9,
                    capsize=4, zorder=5, label=f"{tk} (measured)")
        xr = np.linspace(exps[0], exps[-1], 50)
        ax.plot(xr, med[0] * np.sqrt(xr / exps[0]), "--", color=pcolor[tk],
                alpha=0.7, label=f"{tk}: √t from {exps[0]}s")
    ann = d["annotation"]
    if ann is not None:
        ax.annotate(
            f"photometric is {ann['delta']:+.2f} mag below coverage\n"
            f"at 1s — same offset at 6s/10s.\n"
            f"Whole-program shift (Moon + airmass),\nnot a 6s effect.",
            xy=(1, ann["p1"]), xytext=(0.32, 0.10), textcoords="axes fraction",
            fontsize=9, ha="left",
            arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_yscale("log")
    if d["xticks"]:
        ax.set_xticks(d["xticks"])
    ax.set_xlabel("Exposure Time [seconds]")
    ax.set_ylabel("SNR (normalized to airmass = 1)")
    ax.set_title(
        f"{meta['night_id']}: the pooled 6s 'dip' is a program offset, "
        f"not an exposure effect  (m_G≈{d['best_mag'] + 0.5:.1f})")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, output_dir / "snr_vs_exposure_6s_explained.png")


def _data_snr_vs_moon_fixedmag(calib: NightCalibration):
    import numpy as np

    band = _clear_sky_zp_band(calib.frames)
    if band is None or not any(f.moon_sep_deg is not None for f in calib.frames):
        return None
    zp_mode, zp_sig = band
    ext_k = {filt: fit.k for filt, fit in calib.extinction_per_filter.items()}
    k_default = float(np.median(list(ext_k.values()))) if ext_k else 0.0
    min_meas_snr = 3.0
    usable = _usable_cal_frames(calib, zp_mode, zp_sig)
    best_mag = _best_sampled_mag(usable, min_meas_snr)
    if best_mag is None:
        return None
    target = best_mag + 0.5
    sm, sl = [], []
    for f in usable:
        if f.moon_sep_deg < 60:
            continue
        corr = (10 ** (0.4 * ext_k.get(f.filter_name or "unknown", k_default)
                       * (f.airmass - 1.0)) if f.airmass is not None else 1.0)
        rt = math.sqrt(f.exposure_time)
        for m, s, iso in zip(f.stars_mag, f.stars_snr, _isolated_flags(f)):
            if (iso and m and s and s >= min_meas_snr
                    and target - 2 <= m < target + 2):
                sm.append(m)
                sl.append(math.log10(s * corr / rt))
    slope = float(np.polyfit(sm, sl, 1)[0]) if len(sm) >= 50 else -0.37
    pts = []
    for f in usable:
        corr = (10 ** (0.4 * ext_k.get(f.filter_name or "unknown", k_default)
                       * (f.airmass - 1.0)) if f.airmass is not None else 1.0)
        rt = math.sqrt(f.exposure_time)
        for m, s, iso in zip(f.stars_mag, f.stars_snr, _isolated_flags(f)):
            if (iso and m and s and s >= min_meas_snr
                    and _snr_consistent(s, m, f)
                    and target - 1 <= m < target + 1):
                pts.append((f.moon_sep_deg,
                            (s * corr / rt) * 10 ** (slope * (target - m))))
    if len(pts) < 100:
        return None
    msep = np.array([p[0] for p in pts])
    snrt = np.array([p[1] for p in pts])
    edges = np.arange(0, math.ceil(msep.max() / 10) * 10 + 10, 10)
    cx, cy, cerr = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        v = snrt[(msep >= lo) & (msep < hi)]
        if len(v) >= 20:
            cx.append((lo + hi) / 2)
            cy.append(float(np.median(v)))
            cerr.append(1.253 * float(np.std(v)) / math.sqrt(len(v)))
    return {"target": target, "sep": list(msep), "snrt": list(snrt),
            "binned": {"x": cx, "y": cy, "err": cerr},
            "moon_cut": _MOON_SEP_MIN_DEG}


def _render_snr_vs_moon_fixedmag(d, meta, output_dir, plt, np) -> Path:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(d["sep"], d["snrt"], s=6, alpha=0.10, color="lightgray",
               label="individual stars")
    b = d["binned"]
    if b["x"]:
        ax.errorbar(b["x"], b["y"], yerr=b["err"], fmt="o-", color="black",
                    ms=7, lw=1.5, capsize=4, capthick=1.5, elinewidth=1.5,
                    zorder=5, label="binned median ± SEM")
    ax.axvline(d["moon_cut"], color="red", ls="--", lw=1.5,
               label=f"Moon cut = {d['moon_cut']:.0f}°")
    ax.set_xlabel("Moon separation (°)")
    ax.set_ylabel(r"SNR / $\sqrt{t_{exp}}$ (airmass-normalized)")
    ax.set_title(f"{meta['night_id']}: Moon penalty at m_G={d['target']:.1f} "
                 f"(±1 mag corrected to center, √t-collapsed)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, output_dir / "snr_vs_moon_fixedmag.png")


_PLOT_BUILDERS.update({
    "moon_az_el": (_data_moon_az_el, _render_moon_az_el),
    "snr_vs_moon_distance": (
        _data_snr_vs_moon_distance, _render_snr_vs_moon_distance),
    "snr_vs_exposure_6s_explained": (
        _data_snr_vs_exposure_6s_explained, _render_snr_vs_exposure_6s_explained),
    "snr_vs_moon_fixedmag": (
        _data_snr_vs_moon_fixedmag, _render_snr_vs_moon_fixedmag),
})


def _data_fwhm(calib: NightCalibration):
    import numpy as np

    pts = [(f.timestamp, f.fwhm_px, f.airmass) for f in calib.frames
           if f.fwhm_px is not None and f.timestamp is not None]
    if not pts:
        return None
    fw = [float(p[1]) for p in pts]
    return {
        "t": [p[0].isoformat() for p in pts],
        "fwhm": fw,
        "airmass": [float(p[2]) if p[2] is not None else 0.0 for p in pts],
        "median_fwhm": float(np.median(fw)),
        "p16": float(np.percentile(fw, 16)),
        "p84": float(np.percentile(fw, 84)),
    }


def _render_fwhm(d, meta, output_dir, plt, np) -> Path:
    from datetime import datetime as _dt

    fig, ax = plt.subplots(figsize=(10, 5))
    xs = [_dt.fromisoformat(t) for t in d["t"]]
    cs = d["airmass"]
    sc = ax.scatter(xs, d["fwhm"], c=cs, cmap="viridis", s=12, alpha=0.6)
    ax.axhline(d["median_fwhm"], color="firebrick", ls="--", lw=1.5,
               label=f"median = {d['median_fwhm']:.1f} px "
                     f"(16/84: {d['p16']:.1f}–{d['p84']:.1f})")
    if any(c > 0 for c in cs):
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("airmass")
    ax.set_xlabel("UTC time")
    ax.set_ylabel("PSF FWHM (pixels)")
    ax.set_title(f"{meta['night_id']}: PSF FWHM over the night "
                 f"(sidereal; flat trend = stable optics, ramp = focus drift)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    return _save(fig, output_dir / "fwhm_vs_time.png")


def _data_sky_background(calib: NightCalibration):
    import numpy as np

    rows = []  # (moon_sep, sky_adu, mu_or_nan, altitude)
    for f in calib.frames:
        if f.sky_adu is None or f.moon_sep_deg is None:
            continue
        mu = _sky_mu(f)
        rows.append((f.moon_sep_deg, float(f.sky_adu),
                     float(mu) if mu is not None else float("nan"),
                     float(f.altitude_deg) if f.altitude_deg is not None else 0.0))
    if len(rows) < 10:
        return None
    sep = np.array([r[0] for r in rows])
    adu = np.array([r[1] for r in rows])
    mu = np.array([r[2] for r in rows])
    has_mu = bool(np.isfinite(mu).sum() >= 10)
    yvals = mu if has_mu else adu
    edges = np.arange(0, math.ceil(sep.max() / 10) * 10 + 10, 10)
    cx, cy, e_lo, e_hi = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        v = yvals[(sep >= lo) & (sep < hi)]
        v = v[np.isfinite(v)]
        if len(v) >= 5:
            md = float(np.median(v))
            cx.append((lo + hi) / 2)
            cy.append(md)
            e_lo.append(md - float(np.percentile(v, 16)))
            e_hi.append(float(np.percentile(v, 84)) - md)
    mu_clean = mu[np.isfinite(mu)]
    return {
        "sep": list(sep), "adu": list(adu),
        "mu": [None if not math.isfinite(m) else float(m) for m in mu],
        "alt": [r[3] for r in rows], "has_mu": has_mu,
        "binned": {"x": cx, "y": cy, "e_lo": e_lo, "e_hi": e_hi},
        "median_adu": float(np.median(adu)),
        "median_mu": float(np.median(mu_clean)) if len(mu_clean) else None,
    }


def _render_sky_background(d, meta, output_dir, plt, np) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    sep = np.array(d["sep"])
    alt = np.array(d["alt"])
    use_mu = d["has_mu"]
    yv = np.array([np.nan if m is None else m for m in d["mu"]]) if use_mu \
        else np.array(d["adu"])
    m = np.isfinite(yv)
    sc = ax.scatter(sep[m], yv[m], c=alt[m], cmap="viridis", s=10, alpha=0.4)
    b = d["binned"]
    if b["x"]:
        ax.errorbar(b["x"], b["y"], yerr=[b["e_lo"], b["e_hi"]], fmt="o-",
                    color="black", ms=7, capsize=4, capthick=1.5, elinewidth=1.5,
                    zorder=5, label="binned median ± 16/84%")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("altitude (deg)")
    mi = meta.get("moon_illumination")
    illum = f"Moon {100 * mi:.0f}%" if mi is not None else ""
    ax.set_xlabel("Moon separation (°)")
    if use_mu:
        ax.set_ylabel("sky surface brightness (mag/arcsec² — lower = brighter)")
        ax.invert_yaxis()  # brighter sky (smaller mag) at top
        ax.set_title(f"{meta['night_id']}: sky brightness vs Moon separation "
                     f"({illum}; median {d['median_mu']:.1f} mag/arcsec², "
                     f"{d['median_adu']:.0f} ADU)")
    else:
        ax.set_ylabel("sky background (ADU, flat-fielded, pre-subtraction)")
        ax.set_title(f"{meta['night_id']}: sky background vs Moon separation "
                     f"({illum}; median {d['median_adu']:.0f} ADU)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    return _save(fig, output_dir / "sky_background_vs_moon.png")


_PLOT_BUILDERS.update({
    "fwhm_vs_time": (_data_fwhm, _render_fwhm),
    "sky_background_vs_moon": (_data_sky_background, _render_sky_background),
})


# --- CLI hook -----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Standalone CLI: ``python -m senpai.engine.observability.calibration <night_dir>``.

    The same code is invokable from ``senpai-burr calibrate`` (Phase 3 wiring)."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate per-batch SenpaiRun JSONs into a night calibration."
    )
    parser.add_argument(
        "night_dir",
        help="Processed-night dir (output of `senpai-burr night ...`); "
             "must contain manifest.json.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output dir for calibration JSON + plots (default: <night_dir>/calibration/).",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot rendering (faster; matplotlib not required).",
    )
    parser.add_argument(
        "--from-plot-data", action="store_true",
        help="Skip reprocessing: render plots from an existing "
             "<output>/plot_data.json instead of the batch JSONs.",
    )
    args = parser.parse_args(argv)

    night_dir = Path(args.night_dir)
    out_dir = Path(args.output_dir) if args.output_dir else (night_dir / "calibration")

    if args.from_plot_data:
        plot_calibration(load_plot_data(out_dir / "plot_data.json"), out_dir)
        return 0

    calib = analyze_night(night_dir)
    save_calibration(calib, out_dir)
    if not args.no_plots:
        plot_calibration(calib, out_dir)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
