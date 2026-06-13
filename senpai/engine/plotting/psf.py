"""Per-frame empirical PSF panels (gated by ``config.plotting.psfs``).

Two panels, both built by median-stacking many bright, isolated, unsaturated
catalog stars straight from the frame pixels:

* **sidereal** — stars are points; stack them into a 2D PSF and read off the
  radial profile + RA/Dec cuts (so an elongation reads as a tracking error).
* **rate** — stars are streaks; stack oriented (streak-aligned) stamps and read
  off the along-streak and across-streak profiles, with the fitted length×width
  box overlaid.

The stacking is cosmic-ray robust (peak / centroid / SNR taken from a 3x3
median-filtered copy; the raw stamp is what gets stacked, so spikes wash out in
the median). Each panel also drops a small ``.npy`` of the stacked stamp next to
the PNG so the panel can be regenerated later without the raw FITS; replot falls
back to reloading the processed FITS and re-slicing when the .npy is absent.

This module is the shared home for the stacking/profile primitives; the
night-level observability plots can import them from here too.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy import ndimage
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

SAT_PEAK = 40000.0      # raw ADU; reject saturated stars (below the 65535 clip)
MAX_STARS = 200         # stamps to stack
MIN_STAMPS = 15         # min stacked stars for a usable panel
MIN_PEAK_SNR = 20.0     # per-stamp peak/background-noise floor
SIDEREAL_HALF = 30      # sidereal stamp is (2*half+1)^2 px
_GAUSS_W25_OVER_W50 = math.sqrt(math.log(4) / math.log(2))  # = sqrt(2)


# --------------------------------------------------------------------------
# profile primitives (shared with observability.calibration)
# --------------------------------------------------------------------------
def cut_width(profile, level: float = 0.5) -> float:
    """Full width at ``level`` x peak from the outermost interpolated crossings."""
    profile = np.asarray(profile, dtype=float)
    thr = profile.max() * level
    above = np.where(profile >= thr)[0]
    if len(above) < 2:
        return float("nan")
    lo, hi = int(above[0]), int(above[-1])
    left = (lo - (profile[lo] - thr) / (profile[lo] - profile[lo - 1])
            if lo > 0 and profile[lo] != profile[lo - 1] else float(lo))
    right = (hi + (profile[hi] - thr) / (profile[hi] - profile[hi + 1])
             if hi < len(profile) - 1 and profile[hi] != profile[hi + 1]
             else float(hi))
    return float(right - left)


def profile_shape(profile) -> dict:
    """Multi-level widths + Gaussianity ``spike_index`` of a 1D cut.
    ~1 Gaussian, >>1 a narrow core on a broad halo (FWHM then spurious), <1
    flat-top / donut."""
    w50 = cut_width(profile, 0.5)
    w25 = cut_width(profile, 0.25)
    w75 = cut_width(profile, 0.75)
    ok = np.isfinite(w50) and w50 > 0 and np.isfinite(w25)
    idx = float((w25 / w50) / _GAUSS_W25_OVER_W50) if ok else float("nan")
    return {"fwhm": w50, "fwqm": w25, "fw3qm": w75, "spike_index": idx}


def radial_profile(stamp, half, rstep: float = 0.5):
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


def sky_axes(astropy_wcs):
    """Pixel-space (x, y) unit vectors pointing East(+RA) and North(+Dec) at the
    frame center. Returns (east, north) or None."""
    if astropy_wcs is None:
        return None
    try:
        x0 = float(astropy_wcs.wcs.crpix[0]) - 1.0
        y0 = float(astropy_wcs.wcs.crpix[1]) - 1.0
        ra0, dec0 = (float(c) for c in astropy_wcs.all_pix2world(x0, y0, 0))
        dd = 1.0 / 3600.0
        xn, yn = (float(c) for c in astropy_wcs.all_world2pix(ra0, dec0 + dd, 0))
        xe, ye = (float(c) for c in astropy_wcs.all_world2pix(
            ra0 + dd / math.cos(math.radians(dec0)), dec0, 0))
        north = np.array([xn - x0, yn - y0])
        east = np.array([xe - x0, ye - y0])
        nn, ne = np.linalg.norm(north), np.linalg.norm(east)
        if not (np.isfinite(nn) and np.isfinite(ne) and nn > 0 and ne > 0):
            return None
        return east / ne, north / nn
    except Exception:
        return None


def _sample_line(stamp, half, unit):
    from scipy.ndimage import map_coordinates
    t = np.arange(-half, half + 1.0)
    return map_coordinates(stamp, [half + t * unit[1], half + t * unit[0]],
                           order=1, mode="constant", cval=0.0)


# --------------------------------------------------------------------------
# stacking
# --------------------------------------------------------------------------
def _isolated_order(xy, mags, iso_radius):
    """Brightest-first indices of stars with no brighter-or-comparable neighbor
    within ``iso_radius``."""
    tree = cKDTree(xy)
    for i in np.argsort(mags):
        neigh = tree.query_ball_point(xy[i], iso_radius)
        if not any(j != i and mags[j] < mags[i] + 2.0 for j in neigh):
            yield int(i)


def stack_stars(data, stars, fwhm, half=SIDEREAL_HALF, max_stars=MAX_STARS):
    """Median-stacked, peak-normalized point-source PSF (sidereal).

    ``stars`` is a list of (x, y, mag). Returns (stamp, n) or (None, 0)."""
    keep = [(s[0], s[1], s[2] if s[2] is not None else np.inf) for s in stars
            if s[0] is not None and s[1] is not None]
    if len(keep) < 20:
        return None, 0
    xy = np.array([(s[0], s[1]) for s in keep])
    mags = np.array([s[2] for s in keep])
    h, w = data.shape
    n = 2 * half + 1
    stamps = []
    for i in _isolated_order(xy, mags, max(60.0, 2.0 * fwhm)):
        if len(stamps) >= max_stars:
            break
        x, y = xy[i]
        if not (half + 2 < x < w - half - 2 and half + 2 < y < h - half - 2):
            continue
        xi, yi = int(round(x)), int(round(y))
        st = data[yi - half:yi + half + 1, xi - half:xi + half + 1].astype(float)
        if st.shape != (n, n):
            continue
        ring = np.concatenate([st[0:4].ravel(), st[-4:].ravel(),
                               st[:, 0:4].ravel(), st[:, -4:].ravel()])
        noise = float(np.std(ring))
        st = st - np.median(ring)
        sm = ndimage.median_filter(st, size=3)
        peak = float(sm.max())
        if peak <= 0 or peak > SAT_PEAK or noise <= 0 or peak < MIN_PEAK_SNR * noise:
            continue
        cy, cx = ndimage.center_of_mass(np.clip(sm, 0, None))
        if not (np.isfinite(cx) and np.isfinite(cy)):
            continue
        if abs(cy - half) > fwhm or abs(cx - half) > fwhm:
            continue
        st = ndimage.shift(st, (half - cy, half - cx), order=3, mode="nearest")
        st /= peak
        stamps.append(st)
    if len(stamps) < MIN_STAMPS:
        return None, 0
    stamp = np.median(np.stack(stamps), axis=0)
    if stamp.max() > 0:
        stamp = stamp / stamp.max()
    return stamp, len(stamps)


def _oriented_stamp(data, x, y, cos_a, sin_a, half_a, half_p):
    """Sample a streak-aligned stamp (rows = perpendicular, cols = along) at
    (x, y). Returns the float stamp or None if out of bounds."""
    from scipy.ndimage import map_coordinates
    ta = np.arange(-half_a, half_a + 1.0)
    tp = np.arange(-half_p, half_p + 1.0)
    TA, TP = np.meshgrid(ta, tp)
    sx = x + TA * cos_a - TP * sin_a
    sy = y + TA * sin_a + TP * cos_a
    h, w = data.shape
    if (sx.min() < 1 or sx.max() > w - 2 or sy.min() < 1 or sy.max() > h - 2):
        return None
    return map_coordinates(data, [sy.ravel(), sx.ravel()], order=1).reshape(TA.shape)


def stack_streaks(data, stars, fwhm, length, angle_deg, max_stars=MAX_STARS):
    """Median-stacked, peak-normalized streak PSF in streak-aligned coords.

    Each catalog star is a streak; stack oriented stamps centered on the bright
    isolated ones. Returns (stamp[perp, along], half_along, half_perp, n)."""
    half_a = int(min(90, max(8, round(length / 2 + 4 * fwhm))))
    half_p = int(min(40, max(5, round(3 * fwhm))))
    keep = [(s[0], s[1], s[2] if s[2] is not None else np.inf) for s in stars
            if s[0] is not None and s[1] is not None]
    if len(keep) < 20:
        return None, half_a, half_p, 0
    xy = np.array([(s[0], s[1]) for s in keep])
    mags = np.array([s[2] for s in keep])
    cos_a = math.cos(math.radians(angle_deg))
    sin_a = math.sin(math.radians(angle_deg))
    iso = max(60.0, length + 4 * fwhm)
    stamps = []
    for i in _isolated_order(xy, mags, iso):
        if len(stamps) >= max_stars:
            break
        st = _oriented_stamp(data, xy[i][0], xy[i][1], cos_a, sin_a, half_a, half_p)
        if st is None:
            continue
        ring = np.concatenate([st[0:2].ravel(), st[-2:].ravel()])  # perp edges
        noise = float(np.std(ring))
        st = st - np.median(ring)
        sm = ndimage.median_filter(st, size=3)
        peak = float(sm.max())
        if peak <= 0 or peak > SAT_PEAK or noise <= 0 or peak < MIN_PEAK_SNR * noise:
            continue
        # Center perpendicular only (along position varies with where the catalog
        # point falls on the trail; the across profile is what we want centered).
        perp_prof = np.clip(sm, 0, None).sum(axis=1)
        cy = float(ndimage.center_of_mass(perp_prof)[0])
        if not np.isfinite(cy) or abs(cy - half_p) > 2 * fwhm:
            continue
        st = ndimage.shift(st, (half_p - cy, 0.0), order=3, mode="nearest")
        st /= peak
        stamps.append(st)
    if len(stamps) < MIN_STAMPS:
        return None, half_a, half_p, 0
    stamp = np.median(np.stack(stamps), axis=0)
    if stamp.max() > 0:
        stamp = stamp / stamp.max()
    return stamp, half_a, half_p, len(stamps)


# --------------------------------------------------------------------------
# renderers
# --------------------------------------------------------------------------
def _save(fig, png_path):
    FigureCanvasAgg(fig)
    fig.savefig(str(png_path), dpi=130)


def render_sidereal_psf(stamp, n_stars, axes, meta, png_path):
    """Sidereal per-frame PSF panel: 2D heatmap (+contour, N/E) + radial + cuts."""
    half = stamp.shape[0] // 2
    psc = meta.get("pixel_scale_arcsec")
    if axes is not None:
        east, north = axes
        cut_ra = _sample_line(stamp, half, east)
        cut_dec = _sample_line(stamp, half, north)
        a, b = "RA", "Dec"
    else:
        cut_ra, cut_dec = stamp[half, :], stamp[:, half]
        a, b = "x", "y"
    sh_ra, sh_dec = profile_shape(cut_ra), profile_shape(cut_dec)
    r, rad = radial_profile(stamp, half)
    win = min(half, max(10.0, 3.0 * max(sh_ra["fwhm"], sh_dec["fwhm"])))

    fig = Figure(figsize=(13, 4.6))
    ax0, ax1, ax2 = fig.subplots(1, 3)
    grid = np.linspace(-half, half, stamp.shape[0])
    ax0.imshow(np.arcsinh(np.clip(stamp, 0, None) / 0.02), origin="lower",
               extent=[-half, half, -half, half], cmap="inferno")
    ax0.contour(grid, grid, stamp, levels=[0.5], colors="cyan", linewidths=0.9)
    if axes is not None:
        for u, name, col in ((north, "N", "white"), (east, "E", "deepskyblue")):
            L = 0.7 * win
            ax0.annotate("", xy=(L * u[0], L * u[1]), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
            ax0.text(L * 1.13 * u[0], L * 1.13 * u[1], name, color=col, fontsize=9,
                     ha="center", va="center")
    ax0.set_xlim(-win, win)
    ax0.set_ylim(-win, win)
    ax0.set_title("stacked PSF (50% contour)")
    ax0.set_xlabel("Δx (px)")
    ax0.set_ylabel("Δy (px)")

    ax1.plot(r, np.clip(rad, 1e-4, None), color="purple", lw=2)
    ax1.set_yscale("log")
    ax1.set_ylim(1e-3, 1.3)
    ax1.set_xlim(0, half)
    ax1.set_xlabel("radius (px)")
    ax1.set_ylabel("normalized flux")
    ax1.set_title("radial profile")
    ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(np.arange(stamp.shape[0]) - half, cut_ra, color="tab:red", lw=2,
             label=f"{a} FWHM {sh_ra['fwhm']:.1f}px")
    ax2.plot(np.arange(stamp.shape[0]) - half, cut_dec, color="tab:blue", lw=1.6,
             ls="--", label=f"{b} FWHM {sh_dec['fwhm']:.1f}px")
    for lv in (0.25, 0.5, 0.75):
        ax2.axhline(lv, color="gray", ls=":", lw=0.7, alpha=0.6)
    ax2.set_xlim(-win, win)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("Δ from center (px)")
    ax2.set_title(f"{a} (solid) / {b} (dashed) cuts")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    spike = max(sh_ra["spike_index"], sh_dec["spike_index"])
    sec = f", {sh_ra['fwhm'] * psc:.1f}\"" if psc else ""
    n_txt = n_stars if n_stars is not None else "?"
    fig.suptitle(f"frame {meta.get('index', '?')} sidereal PSF — "
                 f"{meta.get('exposure', '?')}s, n={n_txt}, "
                 f"{a}/{b} FWHM={sh_ra['fwhm']:.1f}/{sh_dec['fwhm']:.1f}px{sec}"
                 f"{'  ⚠spike' if spike >= 1.3 else ''}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, png_path)


def render_streak_psf(stamp, half_a, half_p, n_stars, length, fwhm, sky_in_streak,
                      meta, png_path):
    """Rate per-frame streak panel: oriented 2D stamp (+box, contour, N/E) +
    along-streak and across-streak profiles."""
    psc = meta.get("pixel_scale_arcsec")
    along = stamp.sum(axis=0)        # collapse perpendicular
    along = along / along.max() if along.max() > 0 else along
    across = stamp[:, stamp.shape[1] // 2 - 2: stamp.shape[1] // 2 + 3].sum(axis=1)
    across = across / across.max() if across.max() > 0 else across
    sh_across = profile_shape(across)

    fig = Figure(figsize=(13, 4.6))
    ax0, ax1, ax2 = fig.subplots(1, 3)
    ext = [-half_a, half_a, -half_p, half_p]
    ax0.imshow(np.arcsinh(np.clip(stamp, 0, None) / 0.02), origin="lower",
               extent=ext, aspect="auto", cmap="inferno")
    gx = np.linspace(-half_a, half_a, stamp.shape[1])
    gy = np.linspace(-half_p, half_p, stamp.shape[0])
    ax0.contour(gx, gy, stamp, levels=[0.5], colors="cyan", linewidths=0.9)
    # fitted length x width box (streak-aligned frame: along = x, perp = y)
    from matplotlib.patches import Rectangle
    ax0.add_patch(Rectangle((-length / 2, -fwhm / 2), length, fwhm, fill=False,
                            edgecolor="lime", lw=1.4, ls="--"))
    if sky_in_streak is not None:
        east_s, north_s = sky_in_streak
        for u, name, col in ((north_s, "N", "white"), (east_s, "E", "deepskyblue")):
            L = 0.6 * half_p
            ax0.annotate("", xy=(L * u[0], L * u[1]), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="->", color=col, lw=1.3))
            ax0.text(L * 1.2 * u[0], L * 1.2 * u[1], name, color=col, fontsize=9,
                     ha="center", va="center")
    ax0.set_xlabel("along streak (px)")
    ax0.set_ylabel("across (px)")
    ax0.set_title("stacked streak (lime = fitted L×W)")

    ax1.plot(np.arange(stamp.shape[1]) - half_a, along, color="darkorange", lw=2)
    ax1.axvline(-length / 2, color="lime", ls="--", lw=1)
    ax1.axvline(length / 2, color="lime", ls="--", lw=1)
    ax1.set_xlabel("along streak (px)")
    ax1.set_ylabel("normalized flux")
    ax1.set_title(f"along-streak profile (L={length:.1f}px)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(np.arange(stamp.shape[0]) - half_p, across, color="tab:blue", lw=2)
    for lv in (0.25, 0.5, 0.75):
        ax2.axhline(lv, color="gray", ls=":", lw=0.7, alpha=0.6)
    ax2.set_xlim(-half_p, half_p)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("across streak (px)")
    ax2.set_title(f"across-streak profile (FWHM={sh_across['fwhm']:.1f}px)")
    ax2.grid(True, alpha=0.3)

    sec = f", {fwhm * psc:.1f}\"" if psc else ""
    n_txt = n_stars if n_stars is not None else "?"
    fig.suptitle(f"frame {meta.get('index', '?')} rate streak PSF — "
                 f"{meta.get('exposure', '?')}s, n={n_txt}, "
                 f"length={length:.1f}px, width(FWHM)={sh_across['fwhm']:.1f}px{sec}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, png_path)


# --------------------------------------------------------------------------
# high-level entry points (data in -> npy + png out)
# --------------------------------------------------------------------------
def make_sidereal_psf(data, stars, astropy_wcs, fwhm, meta, png_path, npy_path=None):
    stamp, n = stack_stars(data, stars, fwhm)
    if stamp is None:
        logger.info("psf: frame %s too few stars for sidereal PSF", meta.get("index"))
        return False
    if npy_path is not None:
        np.save(str(npy_path), stamp.astype(np.float32))
    render_sidereal_psf(stamp, n, sky_axes(astropy_wcs), meta, png_path)
    return True


def make_streak_psf(data, stars, astropy_wcs, fwhm, length, angle_deg, meta,
                    png_path, npy_path=None):
    stamp, half_a, half_p, n = stack_streaks(data, stars, fwhm, length, angle_deg)
    if stamp is None:
        logger.info("psf: frame %s too few streaks for streak PSF", meta.get("index"))
        return False
    if npy_path is not None:
        np.save(str(npy_path), stamp.astype(np.float32))
    # express N/E in the streak-aligned frame (rotate pixel axes by -angle)
    sis = None
    ax = sky_axes(astropy_wcs)
    if ax is not None:
        ca, sa = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
        rot = lambda u: np.array([ca * u[0] + sa * u[1], -sa * u[0] + ca * u[1]])
        sis = (rot(ax[0]), rot(ax[1]))
    render_streak_psf(stamp, half_a, half_p, n, length, fwhm, sis, meta, png_path)
    return True


# --------------------------------------------------------------------------
# in-memory frame adapters (duck-typed: SiderealFrame / RateTrackFrame)
# --------------------------------------------------------------------------
def _stars(sf):
    return [(s.x, s.y, s.magnitude) for s in (sf.catalog_stars or [])]


def _astropy_wcs(sf):
    try:
        return sf.wcs.to_astropy_wcs() if sf.wcs is not None else None
    except Exception:
        return None


def _plate_scale(astropy_wcs):
    if astropy_wcs is None:
        return None
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales
        return float(np.mean(proj_plane_pixel_scales(astropy_wcs)) * 3600.0)
    except Exception:
        return None


def _exposure(frame):
    fm = getattr(frame, "frame_metadata", None)
    return getattr(fm, "exposure_time_seconds", None) if fm else None


def plot_sidereal_frame(frame, png_path, npy_path=None) -> bool:
    sf = getattr(frame, "starfield", None)
    if sf is None or not sf.catalog_stars:
        return False
    wcs = _astropy_wcs(sf)
    fwhm = (getattr(getattr(frame, "seeing", None), "pixel_fwhm", None)
            or (sf.fwhm_stats.median_fwhm if sf.fwhm_stats else None) or 4.0)
    meta = {"index": frame.index, "exposure": _exposure(frame),
            "pixel_scale_arcsec": _plate_scale(wcs)}
    return make_sidereal_psf(frame.frame.data, _stars(sf), wcs, float(fwhm), meta,
                             png_path, npy_path)


def plot_rate_frame(frame, png_path, npy_path=None) -> bool:
    sf = getattr(frame, "starfield", None)
    st = getattr(frame, "streak", None)
    if sf is None or not sf.catalog_stars or st is None or not st.pixel_length:
        return False
    wcs = _astropy_wcs(sf)
    meta = {"index": frame.index, "exposure": _exposure(frame),
            "pixel_scale_arcsec": _plate_scale(wcs)}
    return make_streak_psf(frame.frame.data, _stars(sf), wcs, float(st.fwhm),
                           float(st.pixel_length), float(st.degree_angle()), meta,
                           png_path, npy_path)


# --------------------------------------------------------------------------
# regenerate a panel from a saved .npy stamp (no raw FITS needed)
# --------------------------------------------------------------------------
def sidereal_from_stamp(stamp, astropy_wcs, meta, png_path):
    render_sidereal_psf(stamp, None, sky_axes(astropy_wcs), meta, png_path)


def streak_from_stamp(stamp, astropy_wcs, fwhm, length, angle_deg, meta, png_path):
    half_p = stamp.shape[0] // 2
    half_a = stamp.shape[1] // 2
    sis = None
    ax = sky_axes(astropy_wcs)
    if ax is not None:
        ca, sa = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
        rot = lambda u: np.array([ca * u[0] + sa * u[1], -sa * u[0] + ca * u[1]])
        sis = (rot(ax[0]), rot(ax[1]))
    render_streak_psf(stamp, half_a, half_p, None, length, fwhm, sis, meta, png_path)
