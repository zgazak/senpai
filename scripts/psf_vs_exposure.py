"""Empirical PSF vs exposure time — stacked 2D profiles from raw frames.

For each requested exposure bucket, pick a representative sidereal frame
(detection FWHM nearest the bucket median, frame ZP >= zp_min to exclude
cloudy frames), then median-stack subpixel-aligned stamps of many bright,
isolated, unsaturated catalog stars from the RAW FITS. One frame per panel so
the (per-pointing) drift vector isn't averaged away across frames.

Usage:
  uv run --python 3.13 python scripts/psf_vs_exposure.py <night_dir> <out.png>
"""

import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy import ndimage
from scipy.spatial import cKDTree

BUCKETS = (1, 3, 5, 10)
STAMP = 81  # px, odd; holds the 10 s smeared core comfortably
HALF = STAMP // 2
ZP_MIN = 26.1  # photometric-ish frames only
# Per-exposure magnitude windows: brighter stars on shorter exposures so
# every panel's stamps carry comparable per-star SNR (the SAT_PEAK guard
# still rejects saturation). Equal mag ranges make the short-exposure
# stacks noise-dominated and bias any width estimate upward.
MAG_RANGES = {1: (10.0, 13.0), 3: (10.5, 13.5), 5: (11.0, 14.0),
              10: (11.5, 14.5)}
SAT_PEAK = 40000.0  # raw ADU; stay well below the 65535 ceiling
ISO_RADIUS = 60.0  # px — no brighter-or-comparable neighbor inside this
MAX_STARS = 250


def collect_frames(night_dir: Path):
    """(texp_bucket, fwhm, zp, path, stars, timestamp) per sidereal frame."""
    from datetime import datetime

    out = []
    for fn in glob.glob(str(night_dir / "batches" / "*" / "senpai_*.json")):
        try:
            run = json.load(open(fn))
        except Exception:
            continue
        for fr in run.get("sidereal_frames", []):
            fmd = fr.get("frame_metadata") or {}
            texp = fmd.get("exposure_time_seconds")
            sf = fr.get("starfield") or {}
            dm = sf.get("detection_metadata") or {}
            ps = fr.get("photometry_summary") or {}
            path = fr.get("original_frame_path")
            ts = fr.get("timestamp")
            if not texp or not path or not dm.get("pixel_fwhm") or not ts:
                continue
            b = round(texp) if texp < 8 else 10
            if b not in BUCKETS or abs(texp - b) > 0.3:
                continue
            zp = ps.get("zero_point")
            if zp is None or zp < ZP_MIN:
                continue
            out.append((b, float(dm["pixel_fwhm"]), float(zp), path,
                        sf.get("catalog_stars") or [],
                        datetime.fromisoformat(ts)))
    return out


def pick_tight_set(frames):
    """One frame per bucket minimizing the total time span — a single
    atmospheric moment, so exposure time is the only variable."""
    by_bucket = {b: [f for f in frames if f[0] == b and len(f[4]) > 300]
                 for b in BUCKETS}
    if any(not v for v in by_bucket.values()):
        missing = [b for b, v in by_bucket.items() if not v]
        raise SystemExit(f"no candidate frames for buckets: {missing}")
    best, best_span = None, None
    for anchor in by_bucket[BUCKETS[-1]]:  # rarest bucket (10 s) anchors
        t0 = anchor[5]
        pick = {BUCKETS[-1]: anchor}
        for b in BUCKETS[:-1]:
            pick[b] = min(by_bucket[b],
                          key=lambda f: abs((f[5] - t0).total_seconds()))
        ts = [pick[b][5] for b in BUCKETS]
        span = (max(ts) - min(ts)).total_seconds()
        if best_span is None or span < best_span:
            best, best_span = pick, span
    print(f"tightest set spans {best_span/60:.1f} min")
    return best


def stack_psf(path, stars, fwhm, mag_range):
    data = fits.getdata(path).astype(np.float64)
    h, w = data.shape

    keep = [s for s in stars
            if s.get("x") is not None and s.get("y") is not None]
    xy = np.array([(s["x"], s["y"]) for s in keep])
    mags = np.array([s.get("magnitude") if s.get("magnitude") is not None
                     else np.inf for s in keep])
    tree = cKDTree(xy)

    stamps = []
    order = np.argsort(mags)  # brightest first (catalog snr/counts are None)
    for i in order:
        if len(stamps) >= MAX_STARS:
            break
        if not (mag_range[0] <= mags[i] <= mag_range[1]):
            continue
        x, y = xy[i]
        if not (HALF + 2 < x < w - HALF - 2 and HALF + 2 < y < h - HALF - 2):
            continue
        # isolation: no neighbor within ISO_RADIUS brighter than mag+2
        neigh = tree.query_ball_point((x, y), ISO_RADIUS)
        if any(j != i and mags[j] < mags[i] + 2.0 for j in neigh):
            continue
        xi, yi = int(round(x)), int(round(y))
        st = data[yi - HALF: yi + HALF + 1, xi - HALF: xi + HALF + 1].copy()
        # local background: median of the stamp's border ring
        ring = np.concatenate([st[0:4].ravel(), st[-4:].ravel(),
                               st[:, 0:4].ravel(), st[:, -4:].ravel()])
        st -= np.median(ring)
        if st.max() > SAT_PEAK or st.max() <= 0:
            continue
        # subpixel align on flux-weighted centroid of the core
        core = np.clip(st, 0, None)
        cy, cx = ndimage.center_of_mass(core)
        if not (np.isfinite(cx) and np.isfinite(cy)):
            continue
        if abs(cy - HALF) > fwhm or abs(cx - HALF) > fwhm:
            continue  # centroid far off — blend/artifact
        st = ndimage.shift(st, (HALF - cy, HALF - cx), order=3, mode="nearest")
        st /= st.max()
        stamps.append(st)
    if len(stamps) < 10:
        return None, 0
    return np.median(np.stack(stamps), axis=0), len(stamps)


def cut_fwhm(profile):
    """FWHM from the half-max crossings of a 1D cut through the peak.
    (Second moments are useless here: stamp-wing noise dominates them.)"""
    half = profile.max() / 2.0
    above = np.where(profile >= half)[0]
    if len(above) < 2:
        return float("nan")
    lo, hi = above[0], above[-1]
    # linear interpolation at both edges
    left = lo - (profile[lo] - half) / (profile[lo] - profile[lo - 1]) \
        if lo > 0 and profile[lo] != profile[lo - 1] else float(lo)
    right = hi + (profile[hi] - half) / (profile[hi] - profile[hi + 1]) \
        if hi < len(profile) - 1 and profile[hi] != profile[hi + 1] else float(hi)
    return right - left


def moment_fwhm(img):
    h = img.shape[0] // 2
    return cut_fwhm(img[h, :]), cut_fwhm(img[:, h])


def main():
    night_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    frames = collect_frames(night_dir)
    print(f"{len(frames)} candidate sidereal frames (ZP>={ZP_MIN})")

    picks = pick_tight_set(frames)
    fig, axes = plt.subplots(2, len(BUCKETS), figsize=(4.2 * len(BUCKETS), 8.6))
    extent = [-HALF, HALF, -HALF, HALF]
    for col, b in enumerate(BUCKETS):
        rep = picks.get(b)
        ax_im, ax_cut = axes[0][col], axes[1][col]
        if rep is None:
            ax_im.set_title(f"{b} s — no frame")
            continue
        _, fwhm, zp, path, stars, when = rep
        psf, n = stack_psf(path, stars, fwhm, MAG_RANGES[b])
        if psf is None:
            ax_im.set_title(f"{b} s — too few stars")
            continue
        fx, fy = moment_fwhm(psf)
        im = ax_im.imshow(np.arcsinh(psf / 0.02), origin="lower",
                          extent=extent, cmap="inferno")
        ax_im.contour(np.linspace(-HALF, HALF, STAMP),
                      np.linspace(-HALF, HALF, STAMP),
                      psf, levels=[0.5], colors="cyan", linewidths=1.0)
        ax_im.set_title(f"{b} s — {when:%H:%M} UTC   ({n} stars)\n"
                        f"FWHM x={fx:.1f}px  y={fy:.1f}px", fontsize=11)
        ax_im.set_xlim(-40, 40); ax_im.set_ylim(-40, 40)
        ax_im.set_xlabel("Δx (px)")
        if col == 0:
            ax_im.set_ylabel("Δy (px)")

        ax_cut.plot(np.arange(STAMP) - HALF, psf[HALF, :], "-",
                    color="tab:red", label="x cut")
        ax_cut.plot(np.arange(STAMP) - HALF, psf[:, HALF], "-",
                    color="tab:blue", label="y cut")
        ax_cut.axhline(0.5, color="gray", ls=":", lw=1)
        ax_cut.set_xlim(-40, 40); ax_cut.set_ylim(-0.05, 1.05)
        ax_cut.set_xlabel("Δ (px)")
        if col == 0:
            ax_cut.set_ylabel("normalized profile")
            ax_cut.legend(loc="upper right", fontsize=9)
        ax_cut.grid(alpha=0.3)
        print(f"{b:>3}s: {path.split('/')[-1]} n={n} det_fwhm={fwhm:.1f} "
              f"stack FWHM x={fx:.1f} y={fy:.1f} zp={zp:.2f}")

    fig.suptitle(
        f"{night_dir.name}: empirical PSF vs exposure time "
        "(median stack of isolated unsaturated stars, single frame per panel; "
        "cyan = 50% contour)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
