"""``senpai-burr`` CLI: drive senpai's collect pipeline against burr nights.

Usage::

    python -m senpai.cli.burr night <night_dir> -o <output_dir> [...]
    python -m senpai.cli.burr calibrate <processed_night_dir>
    python -m senpai.cli.burr live <sensor_data_dir> -o <output_dir> [...]
    python -m senpai.cli.burr build-dataset <night_dir> [...] -o <dataset_dir>

The ``night`` sub-command consumes one ``/burr/burr/<Sensor>_<YYYYMMDD>`` tree,
groups its FITS into per-collection ``FrameBatch`` objects via
:class:`~senpai.integrations.burr.BurrNight`, runs each batch through
:func:`senpai.engine.processing.collect.process_senpai_collect`, and writes the
resulting ``SenpaiRun`` JSON beside the night for downstream stages (calibration
post-stage and COCO streak-dataset export).

``live`` and ``build-dataset`` are scaffolds wired to the same adapter — see
their handlers for status.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

from senpai.cli.common import ensure_output_dir, save_run_metadata
from senpai.core.config import get_config, initialize_config
from senpai.core.constants import CONFIG_DIR
from senpai.core.logging import set_log_level
from senpai.integrations.burr import BurrNight, FrameBatch

# Default senpai config tuned to burr's FITS header conventions
# (RA/DEC as float degrees, RA_RATE/DEC_RATE in deg/s, no TRKMODE override).
BURR_DEFAULT_CONFIG = CONFIG_DIR / "burr.yaml"

logger = logging.getLogger(__name__)


# --- shared helpers -----------------------------------------------------------


def _batch_passes_filter(batch: FrameBatch, tasks: list[str] | None) -> bool:
    if not tasks:
        return True
    return batch.task in tasks


def _batch_output_dir(night_root: Path, batch: FrameBatch) -> Path:
    return night_root / "batches" / batch.batch_id


def _batch_already_done(batch_dir: Path, batch_id: str) -> bool:
    """We consider a batch complete if both the full result and the summary
    JSON exist. Anything less means we re-run — partial state is unsafe."""
    return (
        (batch_dir / f"senpai_{batch_id}.json").is_file()
        and (batch_dir / f"senpai_{batch_id}_summary.json").is_file()
    )


@contextmanager
def _tee_logs(log_path: Path):
    """Tee root-logger output into ``log_path`` for the duration of the block,
    so each batch's full senpai processing log is saved beside its outputs
    (WCS solve, frame-to-frame shift solves, photometry — the record needed to
    diagnose per-frame failures). Captures whatever level the root logger is
    set to; set ``config.logging.level: DEBUG`` for the most detail."""

    root = logging.getLogger()
    handler = logging.FileHandler(log_path, mode="w")
    handler.setLevel(logging.NOTSET)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root.addHandler(handler)
    try:
        yield
    finally:
        handler.flush()
        handler.close()
        root.removeHandler(handler)


def _run_batch(batch: FrameBatch, batch_dir: Path) -> dict:
    """Run one batch through senpai.collect and persist its results.

    Returns a small manifest entry (paths + timing) for the night's manifest.
    """

    from senpai.engine.processing.collect import final_plots, process_senpai_collect
    from senpai.engine.utils.file_io import load_fits_files

    batch_dir.mkdir(parents=True, exist_ok=True)
    config = get_config()
    config.runtime.output_dir = batch_dir
    config.runtime.run_id = batch.batch_id

    save_run_metadata(batch_dir, "senpai.cli.burr", config)

    t0 = time.time()
    with _tee_logs(batch_dir / "senpai.log"):
        file_list = load_fits_files([str(p) for p in batch.paths])
        _apply_intended_track_mode_overrides(file_list, batch)
        senpai_run = process_senpai_collect(file_list, id=batch.batch_id)

        result = senpai_run.to_result()
        result_path = batch_dir / f"senpai_{result.id}.json"
        with open(result_path, "w") as f:
            json.dump(result.model_dump(), f, indent=4)

        summary = senpai_run.to_summary()
        summary_path = batch_dir / f"senpai_{summary.id}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary.model_dump(), f, indent=4)

        # final_plots writes the review PNGs (final_<idx>.png, raw_<idx>.png,
        # + per-mode sequence GIFs) when config.plotting.review is True.
        # cli.detect skips this call — that's a separate gap to file upstream.
        # Review plots are diagnostic only; the SenpaiRun JSON above is the
        # batch's real output, so a plotting failure must not fail the batch.
        try:
            final_plots(senpai_run, batch_dir)
        except Exception as e:
            logger.warning(
                "final_plots failed for %s (results already saved): %s",
                batch.batch_id, e,
            )

    elapsed = time.time() - t0
    return {
        "batch_id": batch.batch_id,
        "task": batch.task,
        "num_frames": len(batch.frames),
        "command": batch.command.command if batch.command else None,
        "command_target": batch.command.target_label if batch.command else None,
        "result_path": str(result_path),
        "summary_path": str(summary_path),
        "elapsed_seconds": round(elapsed, 2),
        "completed": senpai_run.completed,
        "error": senpai_run.error_message,
    }


def _apply_intended_track_mode_overrides(file_list, batch: FrameBatch) -> None:
    """Patch in-memory FITS headers to reflect burr's *intended* per-frame
    tracking mode, derived from the command log (calsats) or the filename's
    target token (coverage, photometric_standards, twilight_flats).

    Why: burr writes ``TRKMODE: rate`` for every frame in a multi-frame
    collection, including the f3 leg of a calsat sequence and the AltAzTarget
    sub-exposure of a coverage point — those are *intended sidereal* but the
    header alone would route them to senpai's rate-only WCS path (blind-solve
    on streak centroids), which gives an inferior anchor than a true sidereal
    point-source solve. We set TRKMODE per intent and zero the rate keys for
    sidereal frames so both senpai's classifier paths agree.

    Frames with no intent (UUID-named, unattributed orphans with non-semantic
    targets) are left untouched.
    """

    path_to_intent: dict[str, str | None] = {
        str(r.path): r.intended_tracking_mode for r in batch.frames
    }
    for img in file_list:
        file_path = getattr(img, "file_path", None)
        if not file_path:
            continue
        intent = path_to_intent.get(str(file_path))
        if intent not in ("sidereal", "rate"):
            continue
        header = img.header
        header["TRKMODE"] = intent
        if intent == "sidereal":
            # Burr's residual rates on the sidereal leg (~15"/s for calsat f3)
            # would otherwise push the rate-magnitude fallback back to RATE,
            # so we zero them out — the *intent* is to track sidereal.
            for k in ("RA_RATE", "DEC_RATE", "ALT_RATE", "AZ_RATE"):
                if k in header:
                    header[k] = 0.0
    logger.debug(
        "Applied per-frame TRKMODE overrides for batch %s: %s",
        batch.batch_id,
        {Path(p).name: m for p, m in path_to_intent.items() if m},
    )


def _write_manifest(night_root: Path, night: BurrNight, entries: list[dict]) -> Path:
    """A per-night manifest of what was processed, for the calibration and
    dataset stages to consume without re-walking the data dir.

    Merges with any existing manifest (keyed by ``batch_id``) so running `night`
    repeatedly against the same ``-o`` — e.g. one task at a time, or resuming —
    accumulates batches rather than clobbering the prior run's. New entries
    replace same-id old ones; the union is written back in batch-id order.
    """

    path = night_root / "manifest.json"
    merged: dict[str, dict] = {}
    if path.is_file():
        try:
            prior = json.loads(path.read_text())
            for e in prior.get("batches", []):
                if e.get("batch_id"):
                    merged[e["batch_id"]] = e
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Ignoring unreadable existing manifest %s: %s", path, e)
    for e in entries:
        if e.get("batch_id"):
            merged[e["batch_id"]] = e

    batches = sorted(merged.values(), key=lambda e: e.get("batch_id", ""))
    manifest = {
        "night_id": night.night_id,
        "sensor": night.sensor,
        "date_str": night.date_str,
        "burr_root": str(night.burr_root),
        "data_dir": str(night.data_dir),
        "run_state_path": str(night.run_state_path),
        "site": night.run_state.config.site.model_dump() if night.run_state.config.site else None,
        "n_batches": len(batches),
        "batches": batches,
    }
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(
        "Wrote night manifest: %s (%d batches, %d new this run)",
        path, len(batches), len(entries),
    )
    return path


def _iter_filtered_batches(
    night: BurrNight,
    tasks: list[str] | None,
    limit: int | None,
    max_frames: int | None = None,
    seq_key: str | None = None,
) -> Iterable[FrameBatch]:
    n_yielded = 0
    n_frames = 0
    for batch in night.frame_batches(seq_key=seq_key):
        if not _batch_passes_filter(batch, tasks):
            continue
        # Don't admit a batch that would push us past max_frames; we yield
        # whole batches so the collect pipeline keeps its cross-frame WCS
        # shifts intact.
        if max_frames is not None and n_frames + len(batch.frames) > max_frames:
            if n_yielded == 0:
                # Always yield at least the first qualifying batch even if
                # it overshoots — otherwise --max-frames=10 with a 13-frame
                # leading batch would silently process nothing.
                yield batch
                n_yielded += 1
                n_frames += len(batch.frames)
            return
        yield batch
        n_yielded += 1
        n_frames += len(batch.frames)
        if limit is not None and n_yielded >= limit:
            return
        if max_frames is not None and n_frames >= max_frames:
            return


# --- sub-command: night -------------------------------------------------------


def _resolve_nights(args: argparse.Namespace) -> list[BurrNight]:
    """One or more BurrNights to process. ``--auto-nights`` splits a flat,
    multi-night data dir (burr's "didn't split per night" bug) into one night
    per observing session; otherwise it's the single run_state-delimited night."""

    if not args.auto_nights:
        return [
            BurrNight.from_night_dir(
                args.night_dir, burr_root=args.burr_root, data_dir=args.data_dir
            )
        ]

    if args.data_dir is None:
        raise SystemExit(
            "--auto-nights requires --data-dir (the flat FITS dir to split)."
        )
    run_state_path = Path(args.night_dir) / "metadata" / "run_state.json"
    nights = BurrNight.auto_nights(
        run_state_path, args.data_dir, gap_hours=args.gap_hours
    )
    if args.night:
        nights = [n for n in nights if args.night in n.night_id]
        if not nights:
            raise SystemExit(f"--night {args.night!r} matched no detected night.")
    return nights


def _worker_init(
    config_path: str, log_level: str, detect: bool,
    save_processed_fits: bool = True, detect_streaks: bool = True,
) -> None:
    """Per-worker setup for parallel batch processing. Spawned workers start with
    no initialized config singleton, so each must load it; BLAS is already pinned
    to 1 thread via the env set before the pool was created (inherited at import).
    CLI overrides applied to the parent config must be re-applied here — workers
    re-read the YAML and would otherwise silently drop them."""
    config = initialize_config(Path(config_path))
    set_log_level(log_level)
    config.detection.detect = detect
    config.detection.detect_streaks = detect_streaks
    config.runtime.save_processed_fits = save_processed_fits


def _failed_entry(batch: FrameBatch, exc: Exception) -> dict:
    return {
        "batch_id": batch.batch_id,
        "task": batch.task,
        "num_frames": len(batch.frames),
        "error": str(exc),
        "completed": False,
    }


def _process_night(night: BurrNight, args: argparse.Namespace, output_root: Path) -> int:
    """Run every selected batch of one night; write its manifest. Returns 0 on
    full success, 2 if any batch failed. With --jobs N>1, independent batches run
    in parallel across N worker processes."""

    batches = list(_iter_filtered_batches(
        night, args.task, args.limit, args.max_frames, seq_key=args.seq_key
    ))
    night_root = output_root / night.night_id
    night_root.mkdir(parents=True, exist_ok=True)
    jobs = max(1, getattr(args, "jobs", 1))
    logger.info(
        "BurrNight %s: %d batches to process (tasks=%s, limit=%s, jobs=%d)",
        night.night_id, len(batches), args.task or "all", args.limit, jobs,
    )

    entries: list[dict] = []
    n_done = n_skipped = n_failed = 0

    # Write the manifest incrementally (every MANIFEST_FLUSH_EVERY completed
    # batches, plus once at the end) so calibrate/export can run on partial
    # results mid-run. _write_manifest merges by batch_id, so repeated calls
    # are idempotent and a crash mid-night still leaves a usable manifest.
    MANIFEST_FLUSH_EVERY = 25

    # Resolve skips up front (cheap, parent-side); only the real work is dispatched.
    todo: list[tuple[FrameBatch, Path]] = []
    for batch in batches:
        batch_dir = _batch_output_dir(night_root, batch)
        if args.skip_existing and _batch_already_done(batch_dir, batch.batch_id):
            logger.info("skip (existing): %s", batch.batch_id)
            entries.append({
                "batch_id": batch.batch_id,
                "task": batch.task,
                "num_frames": len(batch.frames),
                "skipped": True,
                "result_path": str(batch_dir / f"senpai_{batch.batch_id}.json"),
                "summary_path": str(batch_dir / f"senpai_{batch.batch_id}_summary.json"),
            })
            n_skipped += 1
        else:
            todo.append((batch, batch_dir))

    n_total = len(todo)
    if jobs == 1 or n_total <= 1:
        for i, (batch, batch_dir) in enumerate(todo, start=1):
            logger.info("[%d/%d] batch %s (%s, %d frames)",
                        i, n_total, batch.batch_id, batch.task, len(batch.frames))
            try:
                entry = _run_batch(batch, batch_dir)
            except Exception as e:
                logger.exception("Batch %s failed: %s", batch.batch_id, e)
                entry = _failed_entry(batch, e)
                n_failed += 1
            else:
                n_done += 1
            entries.append(entry)
            if i % MANIFEST_FLUSH_EVERY == 0:
                _write_manifest(night_root, night, entries)
    else:
        import concurrent.futures
        import multiprocessing as mp

        # Pin BLAS to 1 thread/worker so N processes don't oversubscribe cores
        # (set before the spawn pool so children inherit it at numpy import).
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        ctx = mp.get_context("spawn")
        logger.info("Processing %d batches across %d workers", n_total, jobs)
        # Recycle workers by slicing the work and giving each slice a FRESH
        # pool: long-lived workers accumulate memory (caches, fragmentation)
        # over hundreds of batches, and one worker reaching the OOM killer
        # breaks the pool and fails every queued batch. Deliberately NOT
        # max_tasks_per_child: when all spawn workers hit that limit together
        # (and they do — they start together on uniform work) the executor
        # can fail to respawn any of them and deadlocks with zero children;
        # observed hung at exactly jobs×24 batches. A fresh pool per slice is
        # a few seconds of respawn against many minutes of batch work.
        #
        # 8 batches/worker/cycle: workers grow ~0.5-1 GB per batch (observed
        # 9->22 GB across a dense-field stretch at 24/worker), and a single
        # dense galactic-plane batch adds a 10-20 GB working set on top.
        # At jobs*24 one worker hit 30 GB -> OOM kill -> BrokenProcessPool
        # failed the whole queued slice (59 batches, _full6 v5). Recycling
        # 3x more often bounds the accumulation well under the spike room.
        tasks_per_cycle = jobs * 8
        n = 0
        for lo in range(0, len(todo), tasks_per_cycle):
            chunk = todo[lo:lo + tasks_per_cycle]
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=jobs, mp_context=ctx,
                initializer=_worker_init,
                initargs=(str(args.config), get_config().logging.level,
                          args.detect,
                          get_config().runtime.save_processed_fits,
                          args.detect and not args.no_streaks),
            ) as ex:
                futs = {ex.submit(_run_batch, b, bd): b for (b, bd) in chunk}
                for fut in concurrent.futures.as_completed(futs):
                    n += 1
                    batch = futs[fut]
                    try:
                        entry = fut.result()
                    except Exception as e:
                        logger.exception("Batch %s failed: %s", batch.batch_id, e)
                        entry = _failed_entry(batch, e)
                        n_failed += 1
                    else:
                        n_done += 1
                        logger.info("[%d/%d done] batch %s (%s)",
                                    n, n_total, batch.batch_id, batch.task)
                    entries.append(entry)
                    if n % MANIFEST_FLUSH_EVERY == 0:
                        _write_manifest(night_root, night, entries)

    _write_manifest(night_root, night, entries)
    logger.info(
        "Night %s complete: %d processed, %d skipped, %d failed (out of %d)",
        night.night_id, n_done, n_skipped, n_failed, len(batches),
    )
    return 0 if n_failed == 0 else 2


def cmd_night(args: argparse.Namespace) -> int:
    """Process one (or, with --auto-nights, several) collected burr nights."""

    nights = _resolve_nights(args)

    if args.dry_run:
        from collections import Counter
        if not nights:
            print("No nights resolved (empty data dir?).")
        for night in nights:
            batches = list(
                _iter_filtered_batches(
                    night, args.task, args.limit, args.max_frames, seq_key=args.seq_key
                )
            )
            by_task = Counter(b.task for b in batches)
            win = night.night_window()
            print(f"BurrNight {night.night_id}: {len(batches)} batches")
            print(f"  sensor={night.sensor}  data_dir={night.data_dir}")
            print(f"  window={win[0].isoformat()} .. {win[1].isoformat()}")
            print(f"  by task: {dict(by_task)}")
            for b in batches[:5]:
                mark = "✓" if b.command else "·"
                print(f"  {mark} {b.batch_id}  ({b.task}, {len(b.frames)} frames)")
            if len(batches) > 5:
                print(f"  ... and {len(batches) - 5} more")
        return 0

    from senpai.astrometry import enforce_indices, require_astrometry_install
    from senpai.catalog.runner import enforce_catalog

    output_root = ensure_output_dir(Path(args.output_dir), default_stem="burr_runs")

    config = initialize_config(Path(args.config))
    set_log_level(config.logging.level)
    config.detection.detect = args.detect
    config.detection.detect_streaks = args.detect and not args.no_streaks
    if args.no_processed_fits:
        config.runtime.save_processed_fits = False
    if args.debug_plots:
        config.plotting.debug = True
        config.plotting.review = True

    require_astrometry_install()
    enforce_indices()
    enforce_catalog()

    rc = 0
    for night in nights:
        rc |= _process_night(night, args, output_root)
    return rc


# --- sub-command: flats ---------------------------------------------------


def cmd_flats(args: argparse.Namespace) -> int:
    """Build a per-night master flat from the night's twilight_flats frames.

    Twilight flats are auto-exposed (sky level held roughly constant while
    the exposure time ramps) and untracked, so stars drift between frames
    and the sigma-clipped median rejects them. The result is a photometric
    flat (median = 1.0) written to --output-dir, named so the
    BINNING-matched apply path (``app.calibrations.master_flats_dir`` +
    ``auto_apply_flats``) can pick it up.
    """
    from senpai.engine.utils.flats import _create_master_flat_from_files

    output_dir = Path(args.output_dir)
    rc = 0
    for night in _resolve_nights(args):
        files = sorted(
            frame.path
            for batch in _iter_filtered_batches(night, ["twilight_flats"], None, None)
            for frame in batch.frames
        )
        if len(files) < args.min_frames:
            logger.warning(
                "Night %s: only %d twilight_flats frames (< %d) — skipping",
                night.night_id, len(files), args.min_frames,
            )
            rc = 2
            continue
        out_path = output_dir / f"{night.night_id}_master_flat.fits"
        logger.info(
            "Night %s: building master flat from %d twilight frames -> %s",
            night.night_id, len(files), out_path,
        )
        try:
            _create_master_flat_from_files(
                files,
                output_path=out_path,
                min_median=args.min_median,
                max_median=args.max_median,
                max_counts=args.max_counts,
                max_percentile=99.9,
                min_frames=args.min_frames,
                sigma=3.0,
                maxiters=5,
            )
        except Exception as e:
            logger.exception("Night %s: master flat failed: %s", night.night_id, e)
            rc = 2
    return rc


# --- sub-command: calibrate ---------------------------------------------------


def cmd_nights_summary(args: argparse.Namespace) -> int:
    """Cross-night conditions table (PSF / sky / extinction vs Moon & weather)."""
    from senpai.engine.observability.calibration import summarize_nights

    table = summarize_nights(args.root, csv_path=args.csv)
    print(table)
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Aggregate per-batch SenpaiRun JSONs into a per-night calibration."""

    from senpai.engine.observability.calibration import (
        analyze_night, load_plot_data, plot_calibration, save_calibration,
    )

    night_dir = Path(args.processed_night_dir)
    out_dir = Path(args.output_dir) if args.output_dir else (night_dir / "calibration")

    if getattr(args, "from_plot_data", False):
        # Replot from the saved plotted-data dict — no batch reprocessing.
        plot_calibration(load_plot_data(out_dir / "plot_data.json"), out_dir)
        return 0

    calib = analyze_night(night_dir)
    save_calibration(calib, out_dir)
    if not args.no_plots:
        plot_calibration(calib, out_dir)
    return 0


# --- sub-command: live (scaffold) ---------------------------------------------


def cmd_live(args: argparse.Namespace) -> int:
    """Watch a sensor data dir for new FITS and dispatch as the night unfolds.

    Not yet implemented; the design reuses :func:`_run_batch` once a batch
    becomes complete (matched command + expected frame count, or timeout
    after the last frame's arrival).
    """

    raise NotImplementedError(
        "senpai-burr live is not yet implemented. Use `night` against a "
        "completed night for now; live mode lands in a follow-up commit "
        "once the polling/dispatch loop is shaken out."
    )


# --- sub-command: plots -------------------------------------------------------


def cmd_plots(args: argparse.Namespace) -> int:
    """Regenerate per-frame diagnostic plots from processed batch dirs.

    Reads only each batch's ``*_processed.fits`` + ``senpai_*.json`` (no
    re-processing), so plotting is decoupled from the slow night pipeline and
    iterating on a plot never costs a re-run.
    """
    from senpai.integrations.burr.replot import ALL_KINDS, replot

    config = initialize_config(Path(args.config))
    set_log_level(config.logging.level)

    requested = args.kind or ["review", "photometry"]
    kinds = tuple(ALL_KINDS) if "all" in requested else tuple(dict.fromkeys(requested))
    paths = [Path(p) for p in args.paths]
    totals = replot(paths, kinds=kinds, force=args.force, gifs=not args.no_gifs)
    if not totals:
        print(f"No batch directories found under: {', '.join(str(p) for p in paths)}")
        return 1
    print("Wrote: " + ", ".join(f"{v} {k}" for k, v in totals.items()))
    return 0


# --- sub-command: build-dataset (scaffold) ------------------------------------


def cmd_build_dataset(args: argparse.Namespace) -> int:
    """Aggregate per-night SenpaiRun outputs into a starcsp-ready COCO dataset.

    Steps:

    1. For each processed night dir, load its manifest and each non-skipped
       batch's ``SenpaiRunResult`` JSON.
    2. Feed each ``SenpaiRunResult`` to :class:`SenpaiCocoExporter`, which
       writes per-frame ``*_point_sat.json`` + ``*_line_star.json`` and copies
       the FITS image into one staging dir.
    3. Hand the staging dir to :class:`DatasetSplitter`, which writes
       ``<dataset_dir>/{train,val,test}/`` + ``<dataset_dir>/annotations/
       {points,lines}_{split}.json`` — the format starcsp ingests.
    """

    from senpai.export.coco import SenpaiCocoExporter
    from senpai.export.dataset_split import DatasetSplit, DatasetSplitter
    from senpai.engine.models.senpai import SenpaiRunResult

    output_dir = ensure_output_dir(Path(args.output_dir), default_stem="burr_dataset")
    staging_dir = output_dir / "_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    exporter = SenpaiCocoExporter(
        output_dir=staging_dir,
        write_fits=True,
        write_png=False,
        snr_cut=args.snr_cut,
        max_streak_length=args.max_streak_length,
        process_sidereal=args.include_sidereal,
    )

    n_nights = 0
    n_batches_total = 0
    n_batches_exported = 0
    for night_dir_str in args.night_dirs:
        night_dir = Path(night_dir_str)
        manifest_path = night_dir / "manifest.json"
        if not manifest_path.is_file():
            logger.warning("Skipping %s (no manifest.json)", night_dir)
            continue
        manifest = json.loads(manifest_path.read_text())
        n_nights += 1
        logger.info(
            "Exporting night %s (%d batches)", manifest.get("night_id"),
            len(manifest.get("batches", [])),
        )
        for entry in manifest.get("batches", []):
            n_batches_total += 1
            result_path = entry.get("result_path")
            if not result_path:
                continue
            if not Path(result_path).is_file():
                # A failed batch may have left no JSON behind.
                continue
            try:
                result = SenpaiRunResult.model_validate_json(Path(result_path).read_text())
            except Exception as e:
                logger.warning("Failed to parse %s: %s", result_path, e)
                continue
            try:
                exporter.export_senpai_run(result, collect_id=entry["batch_id"])
                n_batches_exported += 1
            except Exception as e:
                logger.exception(
                    "Exporter failed on batch %s: %s", entry["batch_id"], e,
                )

    n_annotation_files = sum(
        1 for _ in staging_dir.glob("*_point_sat.json")
    ) + sum(
        1 for _ in staging_dir.glob("*_line_star.json")
    )
    logger.info(
        "Exported %d/%d batches across %d nights into %s (%d annotation files)",
        n_batches_exported, n_batches_total, n_nights, staging_dir,
        n_annotation_files,
    )

    if n_annotation_files == 0:
        logger.warning(
            "No annotation files written — staging dir is empty, skipping "
            "split. Check that the input nights have non-skipped, WCS-solved "
            "batches with --detect enabled."
        )
        return 2

    split = DatasetSplit(
        train=args.splits[0], val=args.splits[1], test=args.splits[2],
    )
    splitter = DatasetSplitter(split, random_seed=args.seed)
    splitter.split_coco_dataset(
        input_dir=staging_dir,
        output_dir=output_dir,
        temporal_split=not args.random_split,
        exclude_sidereal_from_lines=True,
    )

    if args.keep_staging:
        logger.info("Staging dir kept at %s", staging_dir)
    else:
        import shutil
        shutil.rmtree(staging_dir)
        logger.info("Removed staging dir")

    return 0


# --- argparse -----------------------------------------------------------------


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "-o", "--output_dir",
        default="burr_runs",
        help="Root output directory (default: burr_runs/). Each night lands in "
             "<output_dir>/<night_id>/.",
    )
    p.add_argument(
        "-c", "--config",
        default=str(BURR_DEFAULT_CONFIG),
        help=f"Senpai config file (default: {BURR_DEFAULT_CONFIG} — overlay "
             f"tuned to burr's FITS header conventions).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="senpai-burr",
        description="Drive senpai against burr-collected nights.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_night = sub.add_parser("night", help="Process one collected night.")
    p_night.add_argument(
        "night_dir",
        help="Path to /burr/burr/<Sensor>_<YYYYMMDD> (the metadata-sidecar dir).",
    )
    p_night.add_argument(
        "--burr-root",
        default=None,
        help="Burr tree root (default: parent of the metadata sidecar's parent).",
    )
    p_night.add_argument(
        "--data-dir",
        default=None,
        help="Override sensor data dir (default: <burr_root>/<sensor>/).",
    )
    p_night.add_argument(
        "--auto-nights",
        action="store_true",
        help="Split a flat multi-night --data-dir into one night per observing "
             "session (gaps > --gap-hours), each written to its own "
             "<output>/<sensor>_<YYYYMMDD>/. Reuses the run_state for site "
             "config only; its command log / lighting window are ignored.",
    )
    p_night.add_argument(
        "--gap-hours",
        type=float,
        default=3.0,
        help="With --auto-nights, the inter-frame gap (hours) that separates "
             "observing nights (default 3.0).",
    )
    p_night.add_argument(
        "--night",
        default=None,
        help="With --auto-nights, only process detected nights whose id "
             "contains this string (e.g. 20260529). Useful for reprocessing a "
             "single night from a multi-night dump.",
    )
    p_night.add_argument(
        "--seq-key",
        default=None,
        help="FITS header keyword that marks one logical collection set (e.g. "
             "BURRSEQ). When set, frames are batched by this header id — the "
             "authoritative split for rate-sidereal work (sidereal anchor + its "
             "rate sub-frames per set) — instead of filename/command heuristics.",
    )
    p_night.add_argument(
        "--task",
        action="append",
        choices=["calsats", "coverage", "photometric_standards", "twilight_flats"],
        help="Only process batches with this task (repeatable).",
    )
    p_night.add_argument(
        "--limit", type=int, default=None,
        help="Stop after N batches (useful for development iteration).",
    )
    p_night.add_argument(
        "--max-frames", type=int, default=None,
        help="Stop once cumulative frame count would exceed this. Whole "
             "batches are kept intact, so the actual frame count can land "
             "slightly under the cap (or at the first batch size if a single "
             "batch exceeds the cap on its own).",
    )
    p_night.add_argument(
        "--no-processed-fits", action="store_true",
        help="Don't write per-frame *_processed.fits (~260 MB/frame on 8k "
             "sensors, ~94%% of a night's output). Calibration/results JSONs "
             "are unaffected; decoupled replotting needs the FITS and won't "
             "work for runs produced with this flag.",
    )
    p_night.add_argument(
        "--skip-existing", action="store_true",
        help="Skip batches whose SenpaiRun JSON already exists. Resumable.",
    )
    p_night.add_argument(
        "-j", "--jobs", type=int, default=1,
        help="Process this many batches in parallel (default 1 = serial). "
             "Batches are independent (own dirs, own astrometry --rm containers, "
             "own Gaia queries), so this scales near-linearly on beefy machines. "
             "Each worker is pinned to 1 BLAS thread to avoid oversubscription; "
             "rule of thumb: -j ~= cores/2 (astrometry + photometry both use CPU). "
             "Note each 66 MP frame is ~260 MB, so watch RAM at high -j.",
    )
    p_night.add_argument(
        "-D", "--detect", action="store_true",
        help="Enable streak / non-star detection (default off; needed for the "
             "starcsp dataset build).",
    )
    p_night.add_argument(
        "--no-streaks", action="store_true",
        help="With -D, run point-source detection only and SKIP streak "
             "detection (detect=True, detect_streaks=False). For calsats the "
             "satellite is a point source in rate frames, so this avoids the "
             "slow sidereal streak detector — ~10x faster per calsat batch.",
    )
    p_night.add_argument(
        "--dry-run", action="store_true",
        help="List the batches that would be processed and exit (no FITS loaded, "
             "no senpai config required).",
    )
    p_night.add_argument(
        "--debug-plots", action="store_true",
        help="Override config to enable senpai's per-frame debug + review "
             "plots (final_<idx>.png, raw_<idx>.png, per-step intermediates, "
             "sequence GIFs). Useful when diagnosing WCS/aperture issues; "
             "slower and produces more output.",
    )
    _add_common_args(p_night)
    p_night.set_defaults(func=cmd_night)

    p_flats = sub.add_parser(
        "flats",
        help="Build a per-night master flat from the night's twilight_flats frames.",
    )
    p_flats.add_argument(
        "night_dir",
        help="Path to /burr/burr/<Sensor>_<YYYYMMDD> (the metadata-sidecar dir).",
    )
    p_flats.add_argument("--burr-root", default=None,
                         help="Burr tree root (default: parent of the metadata sidecar's parent).")
    p_flats.add_argument("--data-dir", default=None,
                         help="Override sensor data dir (default: <burr_root>/<sensor>/).")
    p_flats.add_argument("--auto-nights", action="store_true",
                         help="Split a flat multi-night --data-dir into per-session nights "
                              "(same semantics as `night --auto-nights`).")
    p_flats.add_argument("--gap-hours", type=float, default=3.0,
                         help="With --auto-nights, inter-frame gap (hours) separating nights.")
    p_flats.add_argument("--night", default=None,
                         help="With --auto-nights, only nights whose id contains this string.")
    p_flats.add_argument(
        "-o", "--output-dir", required=True,
        help="Directory for <night_id>_master_flat.fits — point "
             "app.calibrations.master_flats_dir here to enable apply.",
    )
    p_flats.add_argument("--min-frames", type=int, default=10,
                         help="Minimum accepted twilight frames to build (default 10).")
    p_flats.add_argument("--min-median", type=float, default=20000.0,
                         help="Reject frames with median below this (under-exposed probes).")
    p_flats.add_argument("--max-median", type=float, default=60000.0,
                         help="Reject frames with median above this (nonlinear/saturating).")
    p_flats.add_argument("--max-counts", type=float, default=65000.0,
                         help="Reject frames whose 99.9th percentile reaches this (saturated regions).")
    p_flats.set_defaults(func=cmd_flats)

    p_cal = sub.add_parser(
        "calibrate", help="Build per-night photometric calibration from processed batches.",
    )
    p_cal.add_argument(
        "processed_night_dir",
        help="Output dir produced by `senpai-burr night` (must contain manifest.json).",
    )
    p_cal.add_argument(
        "-o", "--output_dir", default=None,
        help="Output dir for calibration JSON + plots (default: <night_dir>/calibration/).",
    )
    p_cal.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot rendering.",
    )
    p_cal.add_argument(
        "--from-plot-data", action="store_true",
        help="Skip reprocessing: render plots from an existing "
             "<output>/plot_data.json instead of the batch JSONs.",
    )
    p_cal.set_defaults(func=cmd_calibrate)

    p_ns = sub.add_parser(
        "nights-summary",
        help="Cross-night conditions table (PSF/sky/extinction vs Moon & weather).")
    p_ns.add_argument(
        "root",
        help="Processed root containing <sensor>_<night>/calibration/"
             "night_calibration.json dirs (e.g. .../_full8).")
    p_ns.add_argument("--csv", default=None, help="Also write the table as CSV.")
    p_ns.set_defaults(func=cmd_nights_summary)

    p_live = sub.add_parser("live", help="(stub) Watch a sensor data dir live.")
    p_live.add_argument("data_dir", help="Sensor data dir, e.g. /burr/Hornet/.")
    p_live.add_argument("--metadata-root", default=None, help="Optional override for /burr/burr.")
    _add_common_args(p_live)
    p_live.set_defaults(func=cmd_live)

    p_plots = sub.add_parser(
        "plots",
        help="Regenerate diagnostic plots from processed batch dirs (FITS+JSON).",
    )
    p_plots.add_argument(
        "paths", nargs="+",
        help="Batch dirs, night dirs, or any parent — every directory holding a "
             "senpai_*.json result is replotted.",
    )
    p_plots.add_argument(
        "--kind", action="append", default=None,
        choices=["all", "review", "photometry", "aperture", "psf"],
        help="Plot kind(s); repeatable. 'review' = final_/raw_ overlays + GIFs, "
             "'photometry' = completeness + limiting-mag curves, 'aperture' = "
             "per-star aperture overlay (re-runs cheap photometry), 'psf' = "
             "per-frame empirical PSF panel (stacked stars / streak). "
             "Default: review + photometry (skips the heavy aperture overlay).",
    )
    p_plots.add_argument(
        "--force", action="store_true",
        help="Overwrite existing plot files (default: skip ones already present).",
    )
    p_plots.add_argument(
        "--no-gifs", action="store_true",
        help="Skip the per-batch review sequence GIFs.",
    )
    p_plots.add_argument(
        "-c", "--config",
        default=str(BURR_DEFAULT_CONFIG),
        help=f"Senpai config file (default: {BURR_DEFAULT_CONFIG}).",
    )
    p_plots.set_defaults(func=cmd_plots)

    p_ds = sub.add_parser(
        "build-dataset", help="Build a starcsp-ready COCO streak dataset.",
    )
    p_ds.add_argument(
        "night_dirs", nargs="+",
        help="One or more processed-night dirs (output of `night`, each "
             "containing manifest.json).",
    )
    p_ds.add_argument(
        "--splits", nargs=3, type=float, default=(0.7, 0.2, 0.1),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test fractions (default 0.7 0.2 0.1).",
    )
    p_ds.add_argument(
        "--random-split", action="store_true",
        help="Random split instead of the default temporal split (later frames "
             "→ test). Use temporal split when assessing generalization to "
             "future data.",
    )
    p_ds.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (for --random-split reproducibility).",
    )
    p_ds.add_argument(
        "--snr-cut", type=float, default=3.0,
        help="Minimum SNR for annotations (default 3.0).",
    )
    p_ds.add_argument(
        "--max-streak-length", type=float, default=None,
        help="Drop frames whose streak length exceeds this (pixels).",
    )
    p_ds.add_argument(
        "--include-sidereal", action="store_true",
        help="Also process sidereal frames (default: rate-track only, which "
             "is what starcsp needs).",
    )
    p_ds.add_argument(
        "--keep-staging", action="store_true",
        help="Keep the intermediate _staging/ dir of per-frame COCO files for "
             "debugging.",
    )
    _add_common_args(p_ds)
    p_ds.set_defaults(func=cmd_build_dataset)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
