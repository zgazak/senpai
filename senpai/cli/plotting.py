"""Regenerate per-frame diagnostic plots from processed batch dirs.

Decoupled from processing: reads only each batch's ``*_processed.fits`` (or the
original FITS) + ``senpai_<id>.json`` — no re-run — so iterating on a plot is
cheap. Works on any SenpaiRun output, not just burr nights.

Usage::

    python -m senpai.cli.plotting <paths...> [--kind review|photometry|aperture|psf|all]
                                  [--force] [--no-gifs] [-c config]

``<paths>`` may be batch dirs, night dirs, or any parent — every directory
holding a ``senpai_*.json`` result is replotted.
"""

import logging
from pathlib import Path

from senpai.core.config import initialize_config
from senpai.core.constants import LOCAL_APP_CONFIG_OVERRIDE
from senpai.core.logging import set_log_level

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from senpai.engine.plotting.replot import ALL_KINDS, replot

    parser = argparse.ArgumentParser(
        description="Regenerate per-frame diagnostic plots from processed batch "
                    "dirs (FITS + senpai_*.json); no reprocessing.",
    )
    parser.add_argument(
        "paths", nargs="+",
        help="Batch dirs, night dirs, or any parent — every directory holding a "
             "senpai_*.json result is replotted.",
    )
    parser.add_argument(
        "--kind", action="append", default=None,
        choices=["all", *ALL_KINDS],
        help="Plot kind(s); repeatable. 'review' = final_/raw_ overlays + GIFs, "
             "'photometry' = completeness + limiting-mag curves, 'aperture' = "
             "per-star aperture overlay (re-runs cheap photometry), 'psf' = "
             "per-frame empirical PSF panel (stacked stars / streak). "
             "Default: review + photometry.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing plot files (default: skip ones already present).",
    )
    parser.add_argument(
        "--no-gifs", action="store_true",
        help="Skip the per-batch review sequence GIFs.",
    )
    parser.add_argument(
        "-c", "--config", default=LOCAL_APP_CONFIG_OVERRIDE,
        help=f"Senpai config file (default: {LOCAL_APP_CONFIG_OVERRIDE}).",
    )
    args = parser.parse_args(argv)

    config = initialize_config(Path(args.config))
    set_log_level(config.logging.level)

    requested = args.kind or ["review", "photometry"]
    kinds = tuple(ALL_KINDS) if "all" in requested else tuple(dict.fromkeys(requested))
    totals = replot([Path(p) for p in args.paths], kinds=kinds,
                    force=args.force, gifs=not args.no_gifs)
    if not totals:
        print(f"No batch directories found under: "
              f"{', '.join(str(p) for p in args.paths)}")
        return 1
    print("Wrote: " + ", ".join(f"{v} {k}" for k, v in totals.items()))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
