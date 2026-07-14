"""Tests for tracking.py, run against the real btrack library (not mocked) --
this is exactly what surfaced the division-hypothesis caveat documented in
the module docstring. Only assert what's actually reliable: continuous
tracking of non-dividing/moving cells. Deliberately does NOT assert that a
synthetic division produces a parent/child link, because direct testing
showed the default config does not guarantee that -- it needs validation
against real annotated data, not a synthetic unit test asserting a specific
outcome.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, str(__file__.rsplit("/tests/", 1)[0]))

from fucci_vme_pipeline.config import PipelineConfig  # noqa: E402
from fucci_vme_pipeline.tracking import link_infected_population, track_fucci4_population  # noqa: E402


def _run(label, fn):
    fn()
    print(f"[ OK ] {label}")


def _make_moving_cell_labels(n_frames=5, size=60, start=(20, 20), step=(3, 2)):
    labels = np.zeros((n_frames, size, size), dtype=np.int32)
    for t in range(n_frames):
        cy, cx = start[0] + t * step[0], start[1] + t * step[1]
        labels[t, cy - 5 : cy + 5, cx - 5 : cx + 5] = 1
    return labels


def test_btrack_follows_single_moving_cell_across_frames():
    labels = _make_moving_cell_labels()
    config = PipelineConfig(btrack_max_search_radius_um=50.0)
    df = track_fucci4_population(labels, pixel_size_um=1.0, frame_interval_min=6.0, config=config)

    assert set(df.columns) >= {"frame", "x", "y", "track_id", "parent_track_id", "lineage_id", "generation", "time_min"}
    assert df["track_id"].nunique() == 1, f"expected one continuous track, got {df['track_id'].nunique()}"
    assert sorted(df["frame"]) == [0, 1, 2, 3, 4]
    assert (df["parent_track_id"] == -1).all(), "a founder track with no division should have parent_track_id -1"
    assert df["time_min"].max() == 4 * 6.0


def test_btrack_two_independent_cells_stay_separate():
    n_frames = 4
    size = 80
    labels = np.zeros((n_frames, size, size), dtype=np.int32)
    for t in range(n_frames):
        labels[t, 10:18, 10:18] = 1
        labels[t, 60:68, 60:68] = 2
    config = PipelineConfig(btrack_max_search_radius_um=20.0)
    df = track_fucci4_population(labels, pixel_size_um=1.0, frame_interval_min=6.0, config=config)

    assert df["track_id"].nunique() == 2, f"expected two separate tracks, got {df['track_id'].nunique()}"
    for tid, g in df.groupby("track_id"):
        assert sorted(g["frame"]) == [0, 1, 2, 3]


def test_infected_linker_follows_single_moving_cell():
    centroids_per_frame = [np.array([[20.0 + t * 3, 20.0 + t * 2]]) for t in range(5)]
    df = link_infected_population(
        centroids_per_frame, max_distance_um=50.0, pixel_size_um=1.0, frame_interval_min=6.0
    )
    assert df["track_id"].nunique() == 1
    assert (df["parent_track_id"] == -1).all()
    assert (df["generation"] == 0).all()


def test_infected_linker_gates_out_of_range_jump_as_new_track():
    centroids_per_frame = [
        np.array([[10.0, 10.0]]),
        np.array([[11.0, 11.0]]),
        np.array([[200.0, 200.0]]),  # far outside max_distance -- must not link
    ]
    df = link_infected_population(
        centroids_per_frame, max_distance_um=5.0, pixel_size_um=1.0, frame_interval_min=6.0
    )
    assert df["track_id"].nunique() == 2, "the distant jump should start a new track, not extend the old one"


if __name__ == "__main__":
    _run("btrack follows a single moving (non-dividing) cell", test_btrack_follows_single_moving_cell_across_frames)
    _run("btrack keeps two independent cells as separate tracks", test_btrack_two_independent_cells_stay_separate)
    _run("simple linker follows a single moving cell", test_infected_linker_follows_single_moving_cell)
    _run("simple linker gates out-of-range jumps into a new track", test_infected_linker_gates_out_of_range_jump_as_new_track)
    print("\nAll tracking tests passed.")
