"""Tests for the standalone batch-dir plot regeneration (senpai-burr plots).

These exercise the no-recompute paths — batch discovery and the photometry-curve
plots rebuilt straight from the stored ``photometry_summary`` arrays — without
needing a full processed FITS or a solved StarField in the fixture.
"""

import json

import numpy as np
import pytest
from astropy.io import fits

from senpai.core.config import initialize_config
from senpai.core.constants import CONFIG_DIR
from senpai.engine.models.senpai import (
    CollectionMetadata,
    SenpaiRunResult,
    SiderealFrameSerializable,
)
from senpai.engine.plotting.replot import (
    _find_result_json,
    find_batch_dirs,
    replot_batch_dir,
)


@pytest.fixture(autouse=True)
def _config():
    """Replot reads completeness/SNR defaults from the global config, which the
    CLI always initializes; mirror that here."""
    initialize_config(CONFIG_DIR / "dao.yaml")


def _photometry_summary():
    mags = list(np.linspace(10, 20, 24))
    # completeness rolls over from 1.0 -> ~0.3 across the range
    pct = list(np.clip(105 - (np.array(mags) - 10) * 9, 30, 100))
    return {
        "completeness_mag": mags,
        "completeness_pct": pct,
        "stars_mag": list(np.linspace(10, 20, 200)),
        "stars_snr": list(np.linspace(200, 2, 200)),
        "limiting_snr": 3.0,
        "limiting_magnitude_50": 18.0,
        "limiting_magnitude": 18.0,
    }


def _write_batch(tmp_path, with_fits=False):
    batch = tmp_path / "DAO-01_20260529_xxx_coverage_3_abc12345"
    batch.mkdir()

    sid = SiderealFrameSerializable(
        index=0,
        timestamp="2026-05-30T02:24:35",
        photometry_summary=_photometry_summary(),
    )
    if with_fits:
        fpath = batch / "f0_processed.fits"
        fits.PrimaryHDU(np.zeros((64, 64), dtype=np.float32)).writeto(fpath)
        sid.processed_frame_path = str(fpath)

    result = SenpaiRunResult(
        id="abc12345",
        num_frames=1,
        collect_metadata=CollectionMetadata(),
        sidereal_frames=[sid],
    )
    rj = batch / "senpai_abc12345.json"
    rj.write_text(json.dumps(result.model_dump(), indent=2))
    # a sibling summary that must NOT be picked as the result json
    (batch / "senpai_abc12345_summary.json").write_text("{}")
    return batch


def test_find_result_json_excludes_summary(tmp_path):
    batch = _write_batch(tmp_path)
    found = _find_result_json(batch)
    assert found is not None
    assert found.name == "senpai_abc12345.json"


def test_find_batch_dirs_discovers_nested(tmp_path):
    batch = _write_batch(tmp_path)
    # discovery from a parent several levels up
    dirs = find_batch_dirs(tmp_path)
    assert batch in dirs
    # and directly on the batch dir itself
    assert find_batch_dirs(batch) == [batch]


def test_replot_photometry_curves_from_json(tmp_path):
    batch = _write_batch(tmp_path)
    counts = replot_batch_dir(batch, kinds=("photometry",), force=False, gifs=False)
    assert counts["photometry"] == 2
    assert (batch / "frame_0_completeness.png").exists()
    assert (batch / "frame_0_limiting_mag.png").exists()


def test_replot_photometry_skips_existing_without_force(tmp_path):
    batch = _write_batch(tmp_path)
    replot_batch_dir(batch, kinds=("photometry",), gifs=False)
    # second pass should write nothing (files already present)
    counts = replot_batch_dir(batch, kinds=("photometry",), force=False, gifs=False)
    assert counts["photometry"] == 0
    # ...but --force rewrites them
    counts = replot_batch_dir(batch, kinds=("photometry",), force=True, gifs=False)
    assert counts["photometry"] == 2


def test_streak_candidate_objs_wraps_dicts():
    """Serializable streak_candidates are dicts; plot_single_frame reads them by
    attribute. The wrapper must expose .x/.length_pixels etc. (regression: the
    full replot failed 6 batches with 'dict' object has no attribute 'x')."""
    from senpai.engine.plotting.replot import _streak_candidate_objs

    assert _streak_candidate_objs(None) is None
    assert _streak_candidate_objs([]) is None
    objs = _streak_candidate_objs(
        [{"x": 7603.4, "y": 3818.5, "angle_deg": 80.0, "length_pixels": 40.0,
          "width_pixels": 12.7}]
    )
    assert objs[0].x == 7603.4
    assert objs[0].length_pixels == 40.0


def test_replot_review_with_dict_candidates(tmp_path):
    """End-to-end review replot on a frame carrying dict streak_candidates."""
    batch = tmp_path / "DAO-01_x_coverage_3_def67890"
    batch.mkdir()
    fpath = batch / "f0_processed.fits"
    fits.PrimaryHDU(np.zeros((48, 48), dtype=np.float32)).writeto(fpath)
    sid = SiderealFrameSerializable(
        index=0,
        timestamp="2026-05-30T02:24:35",
        processed_frame_path=str(fpath),
        streak_candidates=[
            {"x": 24.0, "y": 24.0, "angle_deg": 30.0, "length_pixels": 10.0,
             "width_pixels": 3.0}
        ],
    )
    result = SenpaiRunResult(
        id="def67890", num_frames=1, collect_metadata=CollectionMetadata(),
        sidereal_frames=[sid],
    )
    (batch / "senpai_def67890.json").write_text(json.dumps(result.model_dump()))

    counts = replot_batch_dir(batch, kinds=("review",), gifs=False)
    assert counts["review"] == 2  # final_ + raw_
    assert (batch / "final_0.png").exists()


def test_replot_missing_result_json_raises(tmp_path):
    empty = tmp_path / "not_a_batch"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        replot_batch_dir(empty)
