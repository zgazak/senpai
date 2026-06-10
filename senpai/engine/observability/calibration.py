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
    fwhm_px: float | None = None  # sidereal PSF FWHM (detection_metadata)

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
    fwhm_px = None
    det_meta = (starfield or {}).get("detection_metadata") or {}
    if track_mode != "rate":
        fwhm_px = det_meta.get("pixel_fwhm")

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


def _fit_extinction(frames: list[FramePhoto]) -> dict[str, ExtinctionFit]:
    """Per-filter Bouguer fit ``zero_point = m0 - k * airmass``.

    Requires ≥3 frames with both ZP and airmass in a filter. Standard OLS on
    the line ``y = m0 + slope * x``; the extinction coefficient ``k`` is the
    negated slope so it matches the conventional positive-extinction sign
    (atmosphere makes stars dimmer at higher airmass → ZP decreases with
    airmass → slope < 0 → k = -slope > 0).
    """

    by_filter: dict[str, list[tuple[float, float]]] = {}
    for f in _zp_frames(frames):
        if f.zero_point is None or f.airmass is None:
            continue
        key = f.filter_name or "unknown"
        by_filter.setdefault(key, []).append((f.airmass, f.zero_point))

    def _ols(xs, ys):
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        ssxx = sum((x - mean_x) ** 2 for x in xs)
        ssxy = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
        if ssxx <= 0:
            return None
        slope = ssxy / ssxx
        m0 = mean_y - slope * mean_x
        resid = [ys[i] - (m0 + slope * xs[i]) for i in range(n)]
        ss_res = sum(r * r for r in resid)
        sigma2 = ss_res / max(n - 2, 1)
        slope_err = math.sqrt(sigma2 / ssxx) if sigma2 > 0 else 0.0
        m0_err = math.sqrt(sigma2 * (1 / n + mean_x * mean_x / ssxx)) if sigma2 > 0 else 0.0
        return slope, m0, slope_err, m0_err, math.sqrt(sigma2)

    out: dict[str, ExtinctionFit] = {}
    for filt, pairs in by_filter.items():
        if len(pairs) < 3:
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        if max(xs) - min(xs) < _MIN_AIRMASS_RANGE:
            continue
        # Robust Bouguer fit: cloud only ATTENUATES (drops ZP), so on a
        # non-photometric night cloudy frames sit below the clear-sky line and
        # a plain OLS over-steepens k (conflating cloud with extinction).
        # Iteratively sigma-clip residuals (2.5σ) to reject the cloud dropouts
        # while preserving the airmass slope. A flat ZP cut can't be used here:
        # it would also reject high-airmass clear frames whose ZP is genuinely
        # lower from extinction, flattening k to ~0.
        fit = _ols(xs, ys)
        if fit is None:
            continue
        for _ in range(3):
            slope, m0, _se, _me, sigma = fit
            if sigma <= 0:
                break
            kept = [(x, y) for x, y in zip(xs, ys)
                    if abs(y - (m0 + slope * x)) <= 2.5 * sigma]
            if len(kept) == len(xs) or len(kept) < 3:
                break
            kxs = [p[0] for p in kept]
            if max(kxs) - min(kxs) < _MIN_AIRMASS_RANGE:
                break
            xs, ys = kxs, [p[1] for p in kept]
            refit = _ols(xs, ys)
            if refit is None:
                break
            fit = refit
        slope, m0, slope_err, m0_err, _sigma = fit
        out[filt] = ExtinctionFit(
            filter_name=filt,
            m0=m0, m0_err=m0_err,
            k=-slope, k_err=slope_err,  # extinction coefficient is -slope
            n=len(xs),
            airmass_range=(min(xs), max(xs)),
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


def _data_depth_vs_exposure(calib: NightCalibration):
    pts = [(f.exposure_time, f.limiting_magnitude_50, f.airmass)
           for f in _zp_frames(calib.frames)
           if f.exposure_time and f.limiting_magnitude_50]
    if not pts:
        return None
    return {
        "exposure": [p[0] for p in pts],
        "lim50": [p[1] for p in pts],
        "airmass": [p[2] if p[2] is not None else 0.0 for p in pts],
    }


def _render_depth_vs_exposure(d, meta, output_dir, plt, np) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    cs = d["airmass"]
    sc = ax.scatter(d["exposure"], d["lim50"], c=cs, cmap="viridis", s=24,
                    alpha=0.8, edgecolor="black", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("exposure time (s)")
    ax.set_ylabel("limiting magnitude (50% completeness)")
    ax.set_title(f"{meta['night_id']}: depth vs exposure time")
    ax.grid(True, alpha=0.3)
    if any(c > 0 for c in cs):
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("airmass")
    return _save(fig, output_dir / "depth_vs_exposure.png")


def _data_detection_rate_vs_altitude(calib: NightCalibration):
    by_filter: dict[str, dict] = {}
    for f in calib.frames:
        if not (f.n_stars and f.altitude_deg is not None):
            continue
        rate = f.n_stars / f.exposure_time if f.exposure_time else f.n_stars
        s = by_filter.setdefault(f.filter_name or "unknown", {"alt": [], "rate": []})
        s["alt"].append(f.altitude_deg)
        s["rate"].append(rate)
    return {"by_filter": by_filter} if by_filter else None


def _render_detection_rate_vs_altitude(d, meta, output_dir, plt, np) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for filt in sorted(d["by_filter"]):
        s = d["by_filter"][filt]
        ax.scatter(s["alt"], s["rate"], label=f"{filt} (n={len(s['alt'])})",
                   s=20, alpha=0.7)
    ax.set_xlabel("altitude (deg)")
    ax.set_ylabel("stars detected per second")
    ax.set_yscale("log")
    ax.set_title(f"{meta['night_id']}: detection rate vs altitude")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    return _save(fig, output_dir / "detection_rate_vs_altitude.png")


def _data_extinction_curve(calib: NightCalibration):
    import numpy as np

    ext_min_snr = 20.0
    star_ext = []  # (airmass, per-star ZP)
    for f in _zp_frames(calib.frames):
        if f.airmass is None or not f.stars_zp_offset or not f.exposure_time:
            continue
        texp_term = 2.5 * math.log10(f.exposure_time)
        star_ext.extend(
            (f.airmass, off - texp_term)
            for m, off, snr, iso in zip(f.stars_mag, f.stars_zp_offset,
                                        f.stars_snr, _isolated_flags(f))
            if off is not None and snr >= ext_min_snr and iso
            and _snr_consistent(snr, m, f)
        )
    if len(star_ext) >= 10:
        airmasses = np.array([p[0] for p in star_ext])
        offsets = np.array([p[1] for p in star_ext])
        keep = np.abs(offsets - offsets.mean()) <= 3 * offsets.std()
        airmasses, offsets = airmasses[keep], offsets[keep]
        bin_edges = np.arange(airmasses.min(), airmasses.max() + 0.2, 0.2)
        centers, medians, err_lo, err_hi = [], [], [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            in_bin = offsets[(airmasses >= lo) & (airmasses < hi)]
            if len(in_bin) < 3:
                continue
            med = float(np.median(in_bin))
            centers.append((lo + hi) / 2)
            medians.append(med)
            err_lo.append(med - float(np.percentile(in_bin, 16)))
            err_hi.append(float(np.percentile(in_bin, 84)) - med)
        if len(centers) >= 2:
            slope, intercept = np.polyfit(centers, medians, 1)
            fit_note = "median-binned fit"
        else:
            slope, intercept = np.polyfit(airmasses, offsets, 1)
            fit_note = "per-star fit"
        # Overlay the robust frame-level Bouguer fit (the k reported in
        # night_calibration.json). Per-star ZP and frame ZP share the same
        # m + 2.5·log10(flux/t_exp) scale, so the two lines are directly
        # comparable — showing both keeps the plot consistent with the JSON.
        bfits = list(calib.extinction_per_filter.values())
        bouguer = None
        if bfits:
            bf = max(bfits, key=lambda x: x.n)
            bouguer = {"k": bf.k, "k_err": bf.k_err, "m0": bf.m0, "n": bf.n}
        return {
            "mode": "per_star",
            "airmass": airmasses, "offset": offsets,
            "binned": {"x": centers, "y": medians,
                       "err_lo": err_lo, "err_hi": err_hi},
            "fit": {"slope": float(slope), "intercept": float(intercept),
                    "note": fit_note},
            "bouguer": bouguer,
            "n_stars": int(len(airmasses)), "ext_min_snr": ext_min_snr,
        }
    zp_pts = [(f.airmass, f.zero_point, f.filter_name or "unknown")
              for f in _zp_frames(calib.frames)
              if f.airmass is not None and f.zero_point is not None]
    if not zp_pts:
        return None
    per_filter = {}
    for filt in sorted({p[2] for p in zp_pts}):
        xs = [p[0] for p in zp_pts if p[2] == filt]
        ys = [p[1] for p in zp_pts if p[2] == filt]
        fit = calib.extinction_per_filter.get(filt)
        per_filter[filt] = {
            "airmass": xs, "zp": ys,
            "fit": ({"k": fit.k, "k_err": fit.k_err,
                     "m0": fit.m0, "m0_err": fit.m0_err} if fit else None),
        }
    return {"mode": "frame", "per_filter": per_filter}


def _render_extinction_curve(d, meta, output_dir, plt, np) -> Path:
    fig, ax = plt.subplots(figsize=(10, 7))
    if d["mode"] == "per_star":
        airmasses = np.array(d["airmass"])
        offsets = np.array(d["offset"])
        ax.scatter(airmasses, offsets, alpha=0.3, s=10, color="lightgray",
                   label="Individual stars")
        b = d["binned"]
        if b["x"]:
            ax.errorbar(b["x"], b["y"], yerr=[b["err_lo"], b["err_hi"]], fmt="o",
                        color="black", markersize=7, capsize=4, capthick=1.5,
                        elinewidth=1.5, alpha=0.85,
                        label="Binned data (median ± 1σ percentiles)")
        slope, intercept = d["fit"]["slope"], d["fit"]["intercept"]
        line_x = np.linspace(airmasses.min(), airmasses.max(), 50)
        ax.plot(line_x, slope * line_x + intercept, "r-", linewidth=2, alpha=0.8,
                label=f"per-star fit: k={-slope:.3f} mag/airmass "
                      f"({d['fit']['note']})")
        bg = d.get("bouguer")
        if bg is not None:
            ax.plot(line_x, bg["m0"] - bg["k"] * line_x, color="darkorange",
                    ls="--", linewidth=2, alpha=0.85,
                    label=f"robust frame Bouguer: k={bg['k']:.3f}±{bg['k_err']:.3f} "
                          f"(n={bg['n']})")
        ax.set_ylabel(r"per-star ZP (m$_{cat}$ + 2.5·log$_{10}$(flux/t$_{exp}$)) [mag]")
        ax.set_title(f"{meta['night_id']}: extinction curve ({d['n_stars']} "
                     f"isolated stars, sidereal, SNR≥{d['ext_min_snr']:.0f})")
    else:
        per_filter = d["per_filter"]
        filters = sorted(per_filter)
        cmap = plt.cm.viridis(np.linspace(0, 0.85, max(len(filters), 1)))
        for color, filt in zip(cmap, filters):
            s = per_filter[filt]
            xs, ys = s["airmass"], s["zp"]
            ax.scatter(xs, ys, label=f"{filt} (n={len(xs)})", s=12, alpha=0.6,
                       color=color)
            fit = s["fit"]
            if fit:
                line_x = np.linspace(min(xs), max(xs), 50)
                ax.plot(line_x, fit["m0"] - fit["k"] * line_x, color=color,
                        linewidth=1.5,
                        label=f"  k={fit['k']:.3f}±{fit['k_err']:.3f}, "
                              f"m0={fit['m0']:.3f}±{fit['m0_err']:.3f}")
        ax.set_ylabel("zero point (instrumental → catalog mag)")
        ax.set_title(f"{meta['night_id']}: Bouguer extinction (per-frame)")
    ax.set_xlabel("Airmass")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
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


def _data_search_rate(calib: NightCalibration):
    import numpy as np

    target_snr, min_exposure_s, readout_s, min_meas_snr = 6.0, 0.1, 1.0, 3.0
    star_pts, lim50s = [], []
    for f in _zp_frames(calib.frames):
        if not f.exposure_time or not f.fov_sq_deg or not f.stars_mag:
            continue
        if f.limiting_magnitude_50 is not None:
            lim50s.append(f.limiting_magnitude_50)
        for m, s, iso in zip(f.stars_mag, f.stars_snr, _isolated_flags(f)):
            if not iso:
                continue
            if s < min_meas_snr:
                star_pts.append((m, 0.0))
            elif _snr_consistent(s, m, f):
                t_req = max(f.exposure_time * (target_snr / s) ** 2,
                            min_exposure_s)
                star_pts.append((m, f.fov_sq_deg / (t_req + readout_s) * 3600.0))
    if not star_pts:
        return None
    mags = np.array([p[0] for p in star_pts])
    rates = np.array([p[1] for p in star_pts])
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
        "median_lim50": float(np.median(lim50s)) if lim50s else None,
        "n_stars": len(star_pts), "target_snr": target_snr,
        "min_meas_snr": min_meas_snr,
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
    ax.set_ylabel(f"Search Rate (deg²/hour to reach SNR={d['target_snr']:.0f})")
    ax.set_title(f"{meta['night_id']}: search rate vs magnitude "
                 f"({d['n_stars']} isolated stars, sidereal; measured at "
                 f"SNR≥{d['min_meas_snr']:.0f}, rate scaled to SNR="
                 f"{d['target_snr']:.0f})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    return _save(fig, output_dir / "search_rate.png")


_PLOT_BUILDERS.update({
    "extinction_curve": (_data_extinction_curve, _render_extinction_curve),
    "limiting_magnitude_hist": (
        _data_limiting_magnitude_hist, _render_limiting_magnitude_hist),
    "alt_az_coverage": (_data_alt_az_coverage, _render_alt_az_coverage),
    "zp_drift": (_data_zp_drift, _render_zp_drift),
    "depth_vs_exposure": (_data_depth_vs_exposure, _render_depth_vs_exposure),
    "search_rate": (_data_search_rate, _render_search_rate),
    "detection_rate_vs_altitude": (
        _data_detection_rate_vs_altitude, _render_detection_rate_vs_altitude),
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


_PLOT_BUILDERS.update({
    "fwhm_vs_time": (_data_fwhm, _render_fwhm),
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
