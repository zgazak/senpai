"""Offline Gaia queries against a local mirror built by senpai.catalog.gaia_mirror.

Drop-in for senpai.catalog.gaia.query_by_ra_dec_bounds: same signature, same
star-dict shape (ra/dec in radians, mv, magnitudes incl. synthetic Johnson_V /
Sloan_r, source_id, proper motion), so it slots into catalog.runner unchanged and
the in-process sliver cache still wraps it. Reads only the HEALPix tiles whose
bbox (from index.json) overlaps the requested box — sub-second per field.
"""

from __future__ import annotations

import functools
import json
import logging
import os
from typing import Any

import numpy as np

from senpai.catalog.gaia_mirror import MIRROR_DTYPE

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=8)
def _load_index(mirror_dir: str) -> dict:
    with open(os.path.join(mirror_dir, "index.json")) as fh:
        return json.load(fh)


def _ra_subranges(min_ra: float, max_ra: float) -> list[tuple[float, float]]:
    """Normalize the RA box to [0,360), splitting across the 0/360 seam if needed."""
    lo, hi = np.mod(min_ra, 360.0), np.mod(max_ra, 360.0)
    if lo <= hi:
        return [(lo, hi)]
    return [(lo, 360.0), (0.0, hi)]  # wraps 0


def query_by_ra_dec_bounds(
    min_ra: float, max_ra: float, min_dec: float, max_dec: float,
    faint_lim: float | None = None, bright_lim: float | None = None,
    primary_filter: str = "G", *, mirror_dir: str,
) -> list[dict[str, Any]]:
    """Stars from the local mirror within the RA/Dec box and magnitude limits.

    Returns the same dict shape as the online query so it's a transparent
    substitute (see module docstring)."""
    if faint_lim is None:
        faint_lim = 20.0
    if bright_lim is None:
        bright_lim = -32.0

    index = _load_index(mirror_dir)
    ra_ranges = _ra_subranges(min_ra, max_ra)

    # Pick tiles whose bbox overlaps the (possibly seam-split) box.
    chosen = []
    for meta in index["tiles"].values():
        if meta["dec_max"] < min_dec or meta["dec_min"] > max_dec:
            continue
        if any(not (meta["ra_max"] < r0 or meta["ra_min"] > r1) for r0, r1 in ra_ranges):
            chosen.append(meta)
    if not chosen:
        logger.info("Gaia local: no tiles overlap the requested box")
        return []

    parts = [np.fromfile(os.path.join(mirror_dir, m["file"]), dtype=MIRROR_DTYPE) for m in chosen]
    a = np.concatenate(parts) if len(parts) > 1 else parts[0]

    mask = (a["g"] >= bright_lim) & (a["g"] <= faint_lim) & \
           (a["dec"] >= min_dec) & (a["dec"] <= max_dec)
    ra_mask = np.zeros(len(a), dtype=bool)
    for r0, r1 in ra_ranges:
        ra_mask |= (a["ra"] >= r0) & (a["ra"] <= r1)
    a = a[mask & ra_mask]

    # Bound dict-building on ultra-dense (galactic-plane) fields: a single
    # frame there can contain millions of stars (observed 2.57M), and building
    # a dict per row then projecting/isolating them all before the caller's
    # magnitude-stratified max_stars_per_frame cap (applied downstream) peaked
    # ~26 GB and drew the OOM killer. Keep the brightest MAX_LOCAL_ROWS here;
    # the downstream stratified cap subsamples this for completeness. The cut is
    # far fainter than any per-frame cap, so normal fields are untouched.
    MAX_LOCAL_ROWS = 200_000
    n_raw = len(a)
    if n_raw > MAX_LOCAL_ROWS:
        idx = np.argpartition(a["g"], MAX_LOCAL_ROWS)[:MAX_LOCAL_ROWS]
        a = a[idx]

    logger.info(
        "Gaia local: %d stars from %d tiles (box RA[%.3f,%.3f] Dec[%.3f,%.3f] G<=%.1f)%s",
        len(a), len(chosen), min_ra, max_ra, min_dec, max_dec, faint_lim,
        f" [capped from {n_raw} brightest-{MAX_LOCAL_ROWS}]" if n_raw > MAX_LOCAL_ROWS else "",
    )
    return [_to_star(r, primary_filter, faint_lim) for r in a]


def _to_star(row, primary_filter: str, faint_lim: float) -> dict[str, Any]:
    """Build the same star dict as gaia.query_by_ra_dec_bounds (radians, synthetic
    Johnson_V / Sloan_r from BP-RP, proper motion in rad/s)."""
    from senpai.catalog.gaia_transforms import (
        gaia_bp_rp_to_johnson_v, gaia_bp_rp_to_sloan_r,
    )

    g, bp, rp = float(row["g"]), float(row["bp"]), float(row["rp"])
    magnitudes: dict[str, float] = {}
    if np.isfinite(g) and g < 32:
        magnitudes["Gaia_G"] = g
    if np.isfinite(bp) and bp < 32:
        magnitudes["Gaia_BP"] = bp
    if np.isfinite(rp) and rp < 32:
        magnitudes["Gaia_RP"] = rp
    if {"Gaia_G", "Gaia_BP", "Gaia_RP"} <= magnitudes.keys():
        bp_rp = magnitudes["Gaia_BP"] - magnitudes["Gaia_RP"]
        jv = gaia_bp_rp_to_johnson_v(magnitudes["Gaia_G"], bp_rp)
        if jv is not None:
            magnitudes["Johnson_V"] = jv
        sr = gaia_bp_rp_to_sloan_r(magnitudes["Gaia_G"], bp_rp)
        if sr is not None:
            magnitudes["Sloan_r"] = sr

    band = {"G": g, "BP": bp, "RP": rp}.get(primary_filter, g)
    primary_mag = band if (np.isfinite(band) and band < 32) else (
        g if (np.isfinite(g) and g < 32) else faint_lim
    )

    MAS2RAD = 4.84813681109535993589914102358e-9
    YEAR2SEC = 3.1556952e7
    pmra = float(row["pmra"])
    pmdec = float(row["pmdec"])
    ra_pm = pmra * MAS2RAD / YEAR2SEC if np.isfinite(pmra) else 0.0
    dec_pm = pmdec * MAS2RAD / YEAR2SEC if np.isfinite(pmdec) else 0.0

    return {
        "ra": np.radians(float(row["ra"])),
        "dec": np.radians(float(row["dec"])),
        "mv": primary_mag,
        "magnitudes": magnitudes,
        "catalog": "Gaia",
        "source_id": str(int(row["source_id"])),
        "ra_pm": ra_pm,
        "dec_pm": dec_pm,
        "parallax": 0.0,  # not stored in the trimmed mirror
    }
