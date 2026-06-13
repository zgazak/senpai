"""Regenerate per-frame diagnostic plots from a processed batch directory.

The night pipeline writes the heavy data products — the ``*_processed.fits``
frames and the ``senpai_<id>.json`` result — but renders no per-frame plots
inline: on 8120x8120 frames the review/photometry/kernel figures render at
50-150 MB each and dominate wall time (a single kernel plot was measured at
~50 s). This module regenerates any of them after the fact from ONLY the batch
directory's FITS + JSON, so plotting is decoupled from processing and iterating
on a plot never costs a re-run.

It reuses the exact pipeline plot code rather than reimplementing it:

* ``review`` (``final_*``/``raw_*`` overlays + sequence GIFs) -> ``plot_single_frame``
  from the rehydrated ``StarField`` / detections / streak. Pure plot, no recompute.
* ``photometry`` (completeness + limiting-mag curves) -> ``_save_completeness_plot``
  / ``_save_simple_limiting_mag_plot`` straight from the arrays already stored in
  ``photometry_summary``. No recompute.
* ``aperture`` (the per-star aperture overlay) needs the in-memory aperture
  geometry, so it re-runs the *cheap* photometry measurement (astrometry +
  catalog + WCS are skipped — the solved ``StarField``/WCS are rehydrated from
  JSON) with plotting forced on.

The serializable run JSON round-trips back into full ``StarField`` /
``SatelliteListImage`` / ``StreakMetadata`` objects via ``load_senpai_run``, so no
JSON expansion is needed for any of the above.
"""

from __future__ import annotations

import logging
from pathlib import Path

from senpai.core.config import get_config
from senpai.engine.utils.file_io import load_fits_file, load_senpai_run

logger = logging.getLogger(__name__)

# Kinds the caller can request; "all" expands to everything.
ALL_KINDS = ("review", "photometry", "aperture", "psf")


def find_batch_dirs(root: Path) -> list[Path]:
    """Directories under ``root`` (inclusive) holding a run result JSON.

    A batch dir is identified by a ``senpai_*.json`` that is not a
    ``*_summary.json`` — the same artifact ``_run_batch`` writes.
    """
    root = Path(root)
    found: set[Path] = set()
    candidates = [root] if root.is_dir() else []
    candidates += [p.parent for p in root.rglob("senpai_*.json")]
    for d in candidates:
        if _find_result_json(d) is not None:
            found.add(d)
    return sorted(found)


def _find_result_json(batch_dir: Path) -> Path | None:
    matches = [
        p
        for p in sorted(batch_dir.glob("senpai_*.json"))
        if not p.name.endswith("_summary.json")
    ]
    return matches[0] if matches else None


def _resolve_frame_image(batch_dir: Path, processed_path: str | None):
    """Load a frame's processed FITS, preferring the stored path but falling back
    to a same-named file in ``batch_dir`` (so a moved/copied dir still plots)."""
    candidates: list[Path] = []
    if processed_path:
        candidates.append(Path(processed_path))
        candidates.append(batch_dir / Path(processed_path).name)
    for c in candidates:
        if c.exists():
            return load_fits_file(c)
    return None


def _streak_candidate_objs(candidates):
    """The serializable model stores streak_candidates as raw dicts, but
    ``plot_single_frame`` reads them by attribute (``.x``, ``.length_pixels``,
    ...). Wrap each dict so attribute access (and getattr-with-default) works."""
    from types import SimpleNamespace

    if not candidates:
        return None
    out = []
    for c in candidates:
        out.append(SimpleNamespace(**c) if isinstance(c, dict) else c)
    return out or None


def _plot_review(img, frame, out_dir: Path, force: bool) -> list[Path]:
    """final_<idx>.png (overlays) + raw_<idx>.png for one frame."""
    from senpai.engine.plotting.images import plot_single_frame

    written: list[Path] = []
    final_path = out_dir / f"final_{frame.index}.png"
    raw_path = out_dir / f"raw_{frame.index}.png"

    if force or not final_path.exists():
        plot_single_frame(
            img.data,
            starfield=frame.starfield,
            detections=frame.detections,
            streak=getattr(frame, "streak", None),
            streak_candidates=_streak_candidate_objs(frame.streak_candidates),
            output_file=final_path,
        )
        written.append(final_path)
    if force or not raw_path.exists():
        plot_single_frame(img.data, output_file=raw_path)
        written.append(raw_path)
    return written


def _plot_photometry_curves(frame, out_dir: Path, force: bool) -> list[Path]:
    """Completeness + limiting-mag diagnostics from the stored summary arrays."""
    from senpai.engine.photometry.utils import (
        _completeness_limits,
        _save_completeness_plot,
        _save_simple_limiting_mag_plot,
    )

    ps = frame.photometry_summary or {}
    written: list[Path] = []

    comp_mag = ps.get("completeness_mag")
    comp_pct = ps.get("completeness_pct")
    if comp_mag and comp_pct:
        comp_path = out_dir / f"frame_{frame.index}_completeness.png"
        if force or not comp_path.exists():
            target = float(get_config().photometry.limiting_completeness_fraction)
            m_target, m50, m90 = _completeness_limits(comp_mag, comp_pct, target=target)
            _save_completeness_plot(comp_mag, comp_pct, m_target, m50, m90, comp_path)
            written.append(comp_path)

    stars_mag = ps.get("stars_mag")
    stars_snr = ps.get("stars_snr")
    if stars_mag and stars_snr:
        lim_path = out_dir / f"frame_{frame.index}_limiting_mag.png"
        if force or not lim_path.exists():
            limiting = ps.get("limiting_magnitude_50") or ps.get("limiting_magnitude")
            min_snr = float(ps.get("limiting_snr", get_config().photometry.limiting_snr))
            _save_simple_limiting_mag_plot(
                stars_mag, stars_snr, limiting, min_snr, lim_path
            )
            written.append(lim_path)
    return written


# Match the pipeline's per-frame photometry star cap (wcs_helpers).
_MAX_STARS_FOR_APERTURE = 500


def _plot_aperture(img, frame, kind: str, out_dir: Path, force: bool) -> list[Path]:
    """Regenerate the per-star aperture overlay.

    Reconstructs a real ``SiderealFrame``/``RateTrackFrame`` from the rehydrated
    serializable frame + the reloaded processed FITS, then calls the shared
    ``calculate_star_snrs_with_aperture_photometry`` (which emits the overlay when
    ``plotting.photometry`` is on) — the same routine the WCS-refinement path uses
    inline. No astrometry/catalog/WCS recompute: the solved StarField is reused.
    """
    from senpai.engine.models.senpai import RateTrackFrame, SiderealFrame
    from senpai.engine.photometry.utils import (
        calculate_star_snrs_with_aperture_photometry,
    )

    ap_path = out_dir / f"frame_{frame.index}_aperture_photometry_stars.png"
    if not force and ap_path.exists():
        return []
    sf = frame.starfield
    if sf is None or sf.detection_metadata is None or not sf.catalog_stars:
        logger.info(
            "frame %s: no starfield/detection metadata/catalog, skipping aperture plot",
            frame.index,
        )
        return []

    # Brightest-first, capped — mirrors the pipeline's photometry star selection.
    stars = [s for s in sf.catalog_stars if s.x is not None and s.y is not None]
    stars.sort(key=lambda s: (s.magnitude is None, s.magnitude or 0.0))
    stars = stars[:_MAX_STARS_FOR_APERTURE]

    # Serializable timestamps are ISO strings; the full frame model wants datetime.
    from datetime import datetime

    ts = frame.timestamp
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except ValueError:
            ts = datetime.now()

    common = dict(
        starfield=sf,
        detections=frame.detections,
        frame=img,
        index=frame.index,
        timestamp=ts,
        frame_metadata=frame.frame_metadata,
        photometry_summary=frame.photometry_summary,
    )
    if kind == "rate":
        frame_obj = RateTrackFrame(streak=getattr(frame, "streak", None), **common)
    else:
        frame_obj = SiderealFrame(**common)

    cfg = get_config()
    cfg.runtime.output_dir = out_dir
    prev = cfg.plotting.photometry
    cfg.plotting.photometry = True
    try:
        calculate_star_snrs_with_aperture_photometry(frame_obj, stars, plot=True)
    finally:
        cfg.plotting.photometry = prev
    return [ap_path] if ap_path.exists() else []


def _plot_psf(img, frame, mode: str, out_dir: Path, force: bool) -> list[Path]:
    """Per-frame empirical PSF panel. Prefers the saved .npy stamp (cheap, no FITS
    reload); falls back to reloading the processed FITS and re-stacking (which
    also re-writes the .npy), so panels regenerate even if psfs was off at run."""
    from senpai.engine.plotting import psf as P

    suffix = "psf" if mode == "sidereal" else "streak"
    png = out_dir / f"frame_{frame.index}_{suffix}.png"
    npy = out_dir / f"frame_{frame.index}_{suffix}.npy"
    if not force and png.exists():
        return []
    sf = frame.starfield
    if sf is None or not sf.catalog_stars:
        return []
    wcs = P._astropy_wcs(sf)
    meta = {"index": frame.index, "exposure": P._exposure(frame),
            "pixel_scale_arcsec": P._plate_scale(wcs)}
    st = getattr(frame, "streak", None)
    if mode == "rate" and (st is None or not st.pixel_length):
        return []
    try:
        if not force and npy.exists():           # cheap: render from saved stamp
            import numpy as np
            stamp = np.load(npy)
            if mode == "sidereal":
                P.sidereal_from_stamp(stamp, wcs, meta, png)
            else:
                P.streak_from_stamp(stamp, wcs, float(st.fwhm),
                                    float(st.pixel_length), float(st.degree_angle()),
                                    meta, png)
        else:                                    # reload-and-slice (re-stacks)
            data = img.data if img is not None else None
            if data is None:                     # fall back to the original FITS
                op = getattr(frame, "original_frame_path", None)
                if op and Path(op).exists():
                    data = load_fits_file(op).data
            if data is None:
                return []
            if mode == "sidereal":
                fwhm = (getattr(getattr(frame, "seeing", None), "pixel_fwhm", None)
                        or (sf.fwhm_stats.median_fwhm if sf.fwhm_stats else None) or 4.0)
                P.make_sidereal_psf(data, P._stars(sf), wcs, float(fwhm), meta, png, npy)
            else:
                P.make_streak_psf(data, P._stars(sf), wcs, float(st.fwhm),
                                  float(st.pixel_length), float(st.degree_angle()),
                                  meta, png, npy)
    except Exception as e:
        logger.warning("frame %s: PSF panel failed: %s", frame.index, e)
        return []
    return [png] if png.exists() else []


def replot_batch_dir(
    batch_dir: Path,
    kinds: tuple[str, ...] = ALL_KINDS,
    force: bool = False,
    gifs: bool = True,
) -> dict[str, int]:
    """Regenerate the requested plot ``kinds`` for one batch directory.

    Returns a per-kind count of files written.
    """
    from senpai.engine.processing.collect import _write_sequence_gif

    batch_dir = Path(batch_dir)
    result_json = _find_result_json(batch_dir)
    if result_json is None:
        raise FileNotFoundError(f"No senpai_*.json result in {batch_dir}")

    run = load_senpai_run(result_json)
    frames = [(f, "sidereal") for f in run.sidereal_frames]
    frames += [(f, "rate") for f in run.rate_track_frames]

    counts = {k: 0 for k in kinds}
    review_finals: list[Path] = []
    review_raws: list[Path] = []
    review_rate_finals: list[Path] = []

    for frame, mode in sorted(frames, key=lambda fm: fm[0].index):
        img = _resolve_frame_image(batch_dir, frame.processed_frame_path)
        needs_img = "review" in kinds or "aperture" in kinds or "psf" in kinds
        if img is None and needs_img:
            logger.warning(
                "frame %s: processed FITS not found (%s); skipping image plots",
                frame.index, frame.processed_frame_path,
            )

        if "review" in kinds and img is not None:
            written = _plot_review(img, frame, batch_dir, force)
            counts["review"] += len(written)
            # Track for the sequence GIFs regardless of whether they were
            # just (re)written, so the GIF reflects the full sequence.
            final_path = batch_dir / f"final_{frame.index}.png"
            raw_path = batch_dir / f"raw_{frame.index}.png"
            if final_path.exists():
                review_finals.append(final_path)
                if mode == "rate":
                    review_rate_finals.append(final_path)
            if raw_path.exists():
                review_raws.append(raw_path)

        if "photometry" in kinds:
            counts["photometry"] += len(
                _plot_photometry_curves(frame, batch_dir, force)
            )

        if "aperture" in kinds and img is not None:
            counts["aperture"] += len(
                _plot_aperture(img, frame, mode, batch_dir, force)
            )

        if "psf" in kinds:
            counts["psf"] += len(_plot_psf(img, frame, mode, batch_dir, force))

    if gifs and "review" in kinds:
        run_id = run.id
        if review_finals:
            _write_sequence_gif(review_finals, batch_dir / f"{run_id}_sequence.gif")
        if review_rate_finals:
            _write_sequence_gif(
                review_rate_finals, batch_dir / f"{run_id}_sequence_rate.gif"
            )
        if review_raws:
            _write_sequence_gif(review_raws, batch_dir / f"{run_id}_sequence_raw.gif")

    return counts


def replot(
    paths: list[Path],
    kinds: tuple[str, ...] = ALL_KINDS,
    force: bool = False,
    gifs: bool = True,
) -> dict[str, int]:
    """Regenerate plots for every batch dir found under each path.

    Returns aggregate per-kind counts across all batches.
    """
    batch_dirs: list[Path] = []
    for p in paths:
        batch_dirs.extend(find_batch_dirs(Path(p)))
    batch_dirs = sorted(set(batch_dirs))

    if not batch_dirs:
        logger.warning("No batch directories (senpai_*.json) found under: %s", paths)
        return {}

    totals: dict[str, int] = {k: 0 for k in kinds}
    for d in batch_dirs:
        logger.info("Replotting %s (%s)", d.name, ", ".join(kinds))
        try:
            counts = replot_batch_dir(d, kinds=kinds, force=force, gifs=gifs)
        except Exception as e:
            logger.warning("Replot failed for %s: %s", d, e)
            continue
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v
    logger.info(
        "Replot complete: %d batch dir(s), wrote %s",
        len(batch_dirs),
        ", ".join(f"{v} {k}" for k, v in totals.items()),
    )
    return totals
