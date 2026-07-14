"""Tests for Stage 7: fate transition table (pure pandas)."""
from __future__ import annotations

import sys

import pandas as pd

sys.path.insert(0, str(__file__.rsplit("/tests/", 1)[0]))

from fucci_vme_pipeline.fate import compute_fate_transition_table  # noqa: E402


def _run(label, fn):
    fn()
    print(f"[ OK ] {label}")


def test_deterministic_transition():
    # every G1 always goes to S one frame later
    df = pd.DataFrame(
        {
            "track_id": [1, 1, 2, 2],
            "frame": [0, 1, 0, 1],
            "cell_cycle_call": ["G1", "S", "G1", "S"],
        }
    )
    table = compute_fate_transition_table(df, state_col="cell_cycle_call", delta_frames=1)
    assert table.loc["G1", "S"] == 1.0


def test_probabilistic_transition_rows_sum_to_one():
    df = pd.DataFrame(
        {
            "track_id": [1, 2, 3, 1, 2, 3],
            "frame": [0, 0, 0, 1, 1, 1],
            "cell_cycle_call": ["G1", "G1", "S", "S", "G1", "S"],
        }
    )
    table = compute_fate_transition_table(df, state_col="cell_cycle_call", delta_frames=1)
    row_sums = table.sum(axis=1)
    for s in row_sums:
        assert abs(s - 1.0) < 1e-9
    assert abs(table.loc["G1", "S"] - 0.5) < 1e-9
    assert abs(table.loc["G1", "G1"] - 0.5) < 1e-9


def test_only_pairs_exactly_delta_frames_apart_are_counted():
    df = pd.DataFrame(
        {
            "track_id": [1, 1, 1],
            "frame": [0, 1, 3],  # gap between frame 1 and 3 -- not delta=1 apart
            "cell_cycle_call": ["G1", "S", "M"],
        }
    )
    table = compute_fate_transition_table(df, state_col="cell_cycle_call", delta_frames=1)
    assert table.loc["G1", "S"] == 1.0
    assert "M" not in table.columns or table["M"].sum() == 0


def test_rejects_nonpositive_delta():
    df = pd.DataFrame({"track_id": [1], "frame": [0], "cell_cycle_call": ["G1"]})
    try:
        compute_fate_transition_table(df, state_col="cell_cycle_call", delta_frames=0)
        raise AssertionError("expected ValueError for delta_frames=0")
    except ValueError:
        pass


if __name__ == "__main__":
    _run("deterministic transition table", test_deterministic_transition)
    _run("probabilistic transition rows sum to 1", test_probabilistic_transition_rows_sum_to_one)
    _run("only pairs exactly delta_frames apart are counted", test_only_pairs_exactly_delta_frames_apart_are_counted)
    _run("rejects non-positive delta_frames", test_rejects_nonpositive_delta)
    print("\nAll fate-transition tests passed.")
