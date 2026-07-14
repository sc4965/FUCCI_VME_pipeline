"""Test for _drop_ambiguous_duplicate_track_frames -- found on real data,
where a track ended up with two genuinely different positions claiming the
same track_id at the same frame (likely a dummy-filled position coexisting
with the real detection for that frame).
"""
from __future__ import annotations

import sys

import pandas as pd

sys.path.insert(0, str(__file__.rsplit("/tests/", 1)[0]))

from fucci_vme_pipeline.pipeline import _drop_ambiguous_duplicate_track_frames  # noqa: E402


def _run(label, fn):
    fn()
    print(f"[ OK ] {label}")


def test_drops_both_rows_of_a_duplicate_track_frame():
    df = pd.DataFrame(
        {
            "track_id": [1, 1, 1, 2, 2],
            "frame": [0, 1, 1, 0, 1],  # track 1 has two entries at frame 1
            "x": [10.0, 20.0, 999.0, 50.0, 51.0],
            "y": [10.0, 20.0, 999.0, 50.0, 51.0],
        }
    )
    out = _drop_ambiguous_duplicate_track_frames(df)
    assert list(out["track_id"]) == [1, 2, 2], out["track_id"].tolist()
    assert list(out["frame"]) == [0, 0, 1], out["frame"].tolist()


def test_no_duplicates_leaves_dataframe_unchanged():
    df = pd.DataFrame(
        {
            "track_id": [1, 1, 2, 2],
            "frame": [0, 1, 0, 1],
            "x": [10.0, 20.0, 50.0, 51.0],
            "y": [10.0, 20.0, 50.0, 51.0],
        }
    )
    out = _drop_ambiguous_duplicate_track_frames(df)
    assert len(out) == len(df)


if __name__ == "__main__":
    _run("drops both rows of a duplicate (track_id, frame) pair", test_drops_both_rows_of_a_duplicate_track_frame)
    _run("no duplicates leaves dataframe unchanged", test_no_duplicates_leaves_dataframe_unchanged)
    print("\nAll dedup tests passed.")
