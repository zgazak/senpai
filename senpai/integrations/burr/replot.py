"""Back-compat shim. Replot moved to :mod:`senpai.engine.plotting.replot`
(it regenerates per-frame plots from any SenpaiRun batch dir — nothing
burr-specific). Import from there; this re-export is kept so old paths work.
"""

from senpai.engine.plotting.replot import (  # noqa: F401
    ALL_KINDS,
    _streak_candidate_objs,
    find_batch_dirs,
    replot,
    replot_batch_dir,
)
