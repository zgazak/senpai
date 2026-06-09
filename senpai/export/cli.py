"""Command-line interface for SENPAI data export."""

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from senpai.engine.models.senpai import SenpaiRunResult
from senpai.engine.utils.file_io import load_senpai_run
from senpai.export.coco import SenpaiCocoExporter
from senpai.export.dataset_split import split_coco_dataset

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Ensure thread-safe logging
    logging.getLogger().handlers[0].setFormatter(
        logging.Formatter("%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s")
    )


def export_single_run(
    run_path: str,
    output_dir: str,
    collect_id: Optional[str] = None,
    write_png: bool = False,
    write_fits: bool = True,
    save_annotated_images: bool = False,
    remove_median: bool = False,
    snr_cut: float = 0.5,
    box_size: int = 4,
    streak_box_size: int = 10,
    mask_radius: Optional[float] = None,
    max_streak_length: Optional[float] = None,
    apply_calibrations: bool = True,
    verbose: bool = False,
) -> None:
    """Export a single SENPAI run to COCO format."""
    setup_logging(verbose)

    # Load the SENPAI run
    logger.info(f"Loading SENPAI run from {run_path}")
    senpai_run = load_senpai_run(run_path)

    # Use provided collect_id or extract from run
    if collect_id is None:
        if isinstance(senpai_run, SenpaiRunResult):
            collect_id = senpai_run.metadata.get("observation_id", "unknown")
        else:
            collect_id = "unknown"

    # Create exporter
    exporter = SenpaiCocoExporter(
        output_dir=output_dir,
        write_png=write_png,
        write_fits=write_fits,
        save_annotated_images=save_annotated_images,
        remove_median=remove_median,
        snr_cut=snr_cut,
        box_size=box_size,
        streak_box_size=streak_box_size,
        mask_radius=mask_radius,
        max_streak_length=max_streak_length,
    )

    # Export the run
    logger.info(f"Exporting SENPAI run {collect_id} to {output_dir}")
    exporter.export_senpai_run(senpai_run, collect_id, apply_calibrations, source_path=run_path)
    logger.info("Export completed successfully")


def _export_run_worker(args):
    """Worker function for parallel export processing."""
    (
        run_file,
        senpai_run,
        output_dir,
        write_png,
        write_fits,
        save_annotated_images,
        remove_median,
        snr_cut,
        box_size,
        streak_box_size,
        mask_radius,
        max_streak_length,
        apply_calibrations,
        run_index,
        total_runs,
    ) = args

    # Extract collect_id from filename or run metadata
    if isinstance(senpai_run, SenpaiRunResult):
        # Use observation_id if present, else fallback to run_file.stem
        collect_id = getattr(senpai_run.collect_metadata, "observation_id", None) or run_file.stem
    else:
        collect_id = run_file.stem

    logger.info(f"Exporting run {run_index}/{total_runs}: {collect_id}")

    try:
        # Create exporter for this thread
        exporter = SenpaiCocoExporter(
            output_dir=output_dir,
            write_png=write_png,
            write_fits=write_fits,
            save_annotated_images=save_annotated_images,
            remove_median=remove_median,
            snr_cut=snr_cut,
            box_size=box_size,
            streak_box_size=streak_box_size,
            mask_radius=mask_radius,
            max_streak_length=max_streak_length,
        )

        exporter.export_senpai_run(senpai_run, collect_id, apply_calibrations, source_path=run_file)
        logger.info(f"Successfully exported {collect_id}")
        return True, collect_id, None
    except Exception as e:
        logger.error(f"Failed to export {collect_id}: {e}")
        return False, collect_id, str(e)


def export_folder(
    folder_path: str,
    output_dir: str,
    max_runs: Optional[int] = None,
    workers: int = 1,
    write_png: bool = False,
    write_fits: bool = True,
    save_annotated_images: bool = False,
    remove_median: bool = False,
    snr_cut: float = 0.5,
    box_size: int = 4,
    streak_box_size: int = 10,
    mask_radius: Optional[float] = None,
    max_streak_length: Optional[float] = None,
    apply_calibrations: bool = True,
    verbose: bool = False,
) -> None:
    """Export all SENPAI runs from a folder structure to COCO format."""
    setup_logging(verbose)

    folder_path = Path(folder_path)
    output_dir = Path(output_dir)

    if not folder_path.exists():
        logger.error(f"Folder {folder_path} does not exist")
        return

    # Find all SENPAI run files
    run_files = []
    for pattern in ["*.json", "*.json.gz"]:
        run_files.extend(folder_path.rglob(pattern))

    # Filter out non-SENPAI files and sort
    senpai_runs = []
    for run_file in run_files:
        try:
            # Skip files that are likely not SENPAI runs
            if any(skip in run_file.name.lower() for skip in ["calsat", "metadata", "config"]):
                continue

            # Try to load as SENPAI run
            senpai_run = load_senpai_run(str(run_file))
            senpai_runs.append((run_file, senpai_run))
            logger.debug(f"Found SENPAI run: {run_file}")
        except Exception as e:
            logger.debug(f"Skipping {run_file}: {e}")
            continue

    if not senpai_runs:
        logger.error(f"No valid SENPAI runs found in {folder_path}")
        return

    # Sort by modification time (newest first)
    senpai_runs.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)

    # Limit number of runs if specified
    if max_runs is not None:
        senpai_runs = senpai_runs[:max_runs]
        logger.info(f"Limited to {max_runs} most recent runs")

    logger.info(f"Found {len(senpai_runs)} SENPAI runs to export using {workers} workers")

    # Prepare arguments for worker threads
    worker_args = []
    for i, (run_file, senpai_run) in enumerate(senpai_runs, 1):
        args = (
            run_file,
            senpai_run,
            output_dir,
            write_png,
            write_fits,
            save_annotated_images,
            remove_median,
            snr_cut,
            box_size,
            streak_box_size,
            mask_radius,
            max_streak_length,
            apply_calibrations,
            i,
            len(senpai_runs),
        )
        worker_args.append(args)

    # Process runs in parallel or sequentially
    successful_exports = 0
    failed_exports = 0

    if workers > 1:
        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_args = {executor.submit(_export_run_worker, args): args for args in worker_args}

            # Process completed tasks
            for future in as_completed(future_to_args):
                success, collect_id, error = future.result()
                if success:
                    successful_exports += 1
                else:
                    failed_exports += 1
    else:
        # Sequential processing
        for args in worker_args:
            success, collect_id, error = _export_run_worker(args)
            if success:
                successful_exports += 1
            else:
                failed_exports += 1

    logger.info(f"Export completed. Successfully processed {successful_exports} runs, failed {failed_exports} runs.")


def split_dataset(
    input_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    random_seed: Optional[int] = None,
    image_pattern: str = "*.fits",
    verbose: bool = False,
) -> None:
    """Split a COCO dataset into train/val/test sets."""
    setup_logging(verbose)

    logger.info(f"Splitting dataset from {input_dir} to {output_dir}")
    logger.info(f"Split ratios: {train_ratio:.1%} train, {val_ratio:.1%} val, {test_ratio:.1%} test")

    try:
        splits = split_coco_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
            image_pattern=image_pattern,
        )

        logger.info("Dataset split completed successfully")
        for split_name, image_ids in splits.items():
            logger.info(f"  {split_name}: {len(image_ids)} images")
    except Exception as e:
        logger.error(f"Failed to split dataset: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        raise


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export SENPAI data to COCO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export a single run
  python -m senpai.export.cli single /path/to/run.json /path/to/output

  # Export all runs from a folder (sequential)
  python -m senpai.export.cli folder /path/to/folder /path/to/output --max-runs 10

  # Export all runs from a folder (parallel with 4 workers)
  python -m senpai.export.cli folder /path/to/folder /path/to/output --max-runs 10 --workers 4

  # Export with custom settings
  python -m senpai.export.cli single /path/to/run.json /path/to/output \\
    --write-fits --save-annotated-images --snr-cut 2.0 --verbose

  # Split a COCO dataset into train/val/test sets
  python -m senpai.export.cli split /path/to/coco/dataset /path/to/split/output \\
    --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --random-seed 42
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Export command")

    # Single run export
    single_parser = subparsers.add_parser("single", help="Export a single SENPAI run")
    single_parser.add_argument("run_path", help="Path to SENPAI run file")
    single_parser.add_argument("output_dir", help="Output directory for COCO files")
    single_parser.add_argument("--collect-id", help="Collection ID (defaults to run metadata)")

    # Folder export
    folder_parser = subparsers.add_parser("folder", help="Export all SENPAI runs from a folder")
    folder_parser.add_argument("folder_path", help="Path to folder containing SENPAI runs")
    folder_parser.add_argument("output_dir", help="Output directory for COCO files")
    folder_parser.add_argument("--max-runs", type=int, help="Maximum number of runs to export")
    folder_parser.add_argument(
        "--workers", type=int, default=1, help="Number of workers for parallel export (default: 1)"
    )

    # Dataset split
    split_parser = subparsers.add_parser("split", help="Split a COCO dataset into train/val/test sets")
    split_parser.add_argument("input_dir", help="Input directory containing COCO files")
    split_parser.add_argument("output_dir", help="Output directory for split datasets")
    split_parser.add_argument("--train-ratio", type=float, default=0.7, help="Training set ratio (default: 0.7)")
    split_parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation set ratio (default: 0.2)")
    split_parser.add_argument("--test-ratio", type=float, default=0.1, help="Test set ratio (default: 0.1)")
    split_parser.add_argument("--random-seed", type=int, help="Random seed for reproducible splits")
    split_parser.add_argument(
        "--image-pattern", default="*.fits", help="Pattern to match image files (default: *.fits)"
    )
    split_parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose logging")

    # Common options
    for subparser in [single_parser, folder_parser]:
        subparser.add_argument(
            "--write-png", action="store_true", default=False, help="Save PNG images (default: False)"
        )
        subparser.add_argument(
            "--write-fits", action="store_true", default=True, help="Save FITS images (default: True)"
        )
        subparser.add_argument(
            "--save-annotated-images", action="store_true", default=False, help="Save annotated images (default: False)"
        )
        subparser.add_argument(
            "--remove-median", action="store_true", default=False, help="Remove median from images (default: False)"
        )
        subparser.add_argument("--snr-cut", type=float, default=0.5, help="Minimum SNR for annotations (default: 0.5)")
        subparser.add_argument(
            "--box-size", type=int, default=4, help="Bounding box size for point sources (default: 4)"
        )
        subparser.add_argument(
            "--streak-box-size", type=int, default=10, help="Bounding box size for satellites (default: 10)"
        )
        subparser.add_argument("--mask-radius", type=float, help="Radius to mask around center (pixels)")
        subparser.add_argument(
            "--max-streak-length", type=float, help="Maximum streak length in pixels (default: no limit)"
        )
        subparser.add_argument(
            "--no-calibrations", action="store_true", default=False, help="Skip applying calibrations"
        )
        subparser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose logging")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "single":
            # Common parameters for export commands
            common_params = {
                "write_png": args.write_png,
                "write_fits": args.write_fits,
                "save_annotated_images": args.save_annotated_images,
                "remove_median": args.remove_median,
                "snr_cut": args.snr_cut,
                "box_size": args.box_size,
                "streak_box_size": args.streak_box_size,
                "mask_radius": args.mask_radius,
                "max_streak_length": args.max_streak_length,
                "apply_calibrations": not args.no_calibrations,
                "verbose": args.verbose,
            }

            export_single_run(
                run_path=args.run_path,
                output_dir=args.output_dir,
                collect_id=args.collect_id,
                **common_params,
            )
        elif args.command == "folder":
            # Common parameters for export commands
            common_params = {
                "write_png": args.write_png,
                "write_fits": args.write_fits,
                "save_annotated_images": args.save_annotated_images,
                "remove_median": args.remove_median,
                "snr_cut": args.snr_cut,
                "box_size": args.box_size,
                "streak_box_size": args.streak_box_size,
                "mask_radius": args.mask_radius,
                "max_streak_length": args.max_streak_length,
                "apply_calibrations": not args.no_calibrations,
                "verbose": args.verbose,
            }

            export_folder(
                folder_path=args.folder_path,
                output_dir=args.output_dir,
                max_runs=args.max_runs,
                workers=args.workers,
                **common_params,
            )
        elif args.command == "split":
            split_dataset(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                random_seed=args.random_seed,
                image_pattern=args.image_pattern,
                verbose=args.verbose,
            )
    except KeyboardInterrupt:
        logger.info("Export interrupted by user")
    except Exception as e:
        logger.error(f"Export failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
