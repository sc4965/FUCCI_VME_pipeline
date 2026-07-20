"""Tests for stitch_short_gaps -- bridging short btrack gaps using position
alone, found necessary on real data where a condensed mitotic chromatin
object was correctly segmented but failed to link into its own otherwise-
continuous track (confirmed via direct label-at-click checks against the
raw segmentation mask).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__file__.rsplit("/tests/", 1)[0]))

from fucci_vme_pipeline.pipeline import stitch_short_gaps  # noqa: E402


def _run(label, fn):
    fn()
    print(f"[ OK ] {label}")


def test_stitches_single_frame_gap_within_tolerance():
    df = pd.DataFrame(
        {
            "frame": [0, 2, 1],
            "x": [10.0, 14.0, 12.0],
            "y": [10.0, 14.0, 12.0],
            "track_id": [1, 1, np.nan],
            "parent_track_id": [-1, -1, np.nan],
            "lineage_id": [1, 1, np.nan],
        }
    )
    out = stitch_short_gaps(df, max_gap_frames=3, max_stitch_distance_px=10.0)
    assert len(out) == 3
    assert out["track_id"].isna().sum() == 0
    stitched_row = out[out["frame"] == 1].iloc[0]
    assert stitched_row["track_id"] == 1
    assert stitched_row["parent_track_id"] == -1
    assert stitched_row["lineage_id"] == 1


def test_gap_larger_than_max_is_not_stitched():
    df = pd.DataFrame(
        {
            "frame": [0, 5, 2],
            "x": [10.0, 15.0, 12.0],
            "y": [10.0, 15.0, 12.0],
            "track_id": [1, 1, np.nan],
        }
    )
    out = stitch_short_gaps(df, max_gap_frames=3, max_stitch_distance_px=10.0)
    assert len(out) == 2, out  # the untracked row at frame 2 stays dropped
    assert set(out["frame"]) == {0, 5}


def test_untracked_object_too_far_is_not_stitched():
    df = pd.DataFrame(
        {
            "frame": [0, 2, 1],
            "x": [10.0, 14.0, 999.0],  # nowhere near the interpolated (12, 12)
            "y": [10.0, 14.0, 999.0],
            "track_id": [1, 1, np.nan],
        }
    )
    out = stitch_short_gaps(df, max_gap_frames=3, max_stitch_distance_px=10.0)
    assert len(out) == 2
    assert set(out["frame"]) == {0, 2}


def test_no_orphans_leaves_dataframe_unchanged():
    df = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "x": [10.0, 11.0, 12.0],
            "y": [10.0, 11.0, 12.0],
            "track_id": [1, 1, 1],
        }
    )
    out = stitch_short_gaps(df, max_gap_frames=3, max_stitch_distance_px=10.0)
    assert len(out) == 3


def test_two_tracks_each_claim_their_own_nearest_orphan():
    df = pd.DataFrame(
        {
            "frame": [0, 2, 0, 2, 1, 1],
            "x": [0.0, 0.0, 100.0, 100.0, 0.0, 100.0],
            "y": [0.0, 4.0, 0.0, 4.0, 2.5, 2.5],
            "track_id": [1, 1, 2, 2, np.nan, np.nan],
        }
    )
    out = stitch_short_gaps(df, max_gap_frames=3, max_stitch_distance_px=5.0)
    assert len(out) == 6
    assert out["track_id"].isna().sum() == 0
    frame1 = out[out["frame"] == 1].sort_values("x")
    assert list(frame1["track_id"]) == [1, 2]


def test_no_tracks_at_all_returns_empty():
    df = pd.DataFrame({"frame": [0, 1], "x": [1.0, 2.0], "y": [1.0, 2.0], "track_id": [np.nan, np.nan]})
    out = stitch_short_gaps(df, max_gap_frames=3, max_stitch_distance_px=10.0)
    assert len(out) == 0


if __name__ == "__main__":
    _run("stitches a single-frame gap within tolerance", test_stitches_single_frame_gap_within_tolerance)
    _run("gap larger than max is not stitched", test_gap_larger_than_max_is_not_stitched)
    _run("untracked object too far is not stitched", test_untracked_object_too_far_is_not_stitched)
    _run("no orphans leaves dataframe unchanged", test_no_orphans_leaves_dataframe_unchanged)
    _run("two tracks each claim their own nearest orphan", test_two_tracks_each_claim_their_own_nearest_orphan)
    _run("no tracks at all returns empty", test_no_tracks_at_all_returns_empty)
    print("\nAll gap-stitching tests passed.")
