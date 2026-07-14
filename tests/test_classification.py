"""Pure pandas/numpy tests for Stage 4/5 classification logic."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__file__.rsplit("/tests/", 1)[0]))

from fucci_vme_pipeline.classification import (  # noqa: E402
    classify_cell_cycle,
    classify_infection,
    compute_mphase_duration,
    compute_track_coverage,
    filter_by_track_coverage,
    normalize_per_track,
)
from fucci_vme_pipeline.config import PipelineConfig  # noqa: E402


def _run(label, fn):
    fn()
    print(f"[ OK ] {label}")


def test_normalize_per_track_min_max():
    df = pd.DataFrame({"track_id": [1, 1, 1], "cdt1": [10.0, 20.0, 30.0]})
    out = normalize_per_track(df, ["cdt1"])
    assert list(out["cdt1_norm"]) == [0.0, 0.5, 1.0]


def test_normalize_per_track_degenerate_span_falls_back_to_half():
    df = pd.DataFrame({"track_id": [1, 1], "cdt1": [5.0, 5.0]})
    out = normalize_per_track(df, ["cdt1"])
    assert list(out["cdt1_norm"]) == [0.5, 0.5]


def test_classify_cell_cycle_precedence():
    config = PipelineConfig()
    df = pd.DataFrame(
        {
            "cdt1_norm": [0.9, 0.1, 0.1, 0.1],
            "slbp_norm": [0.1, 0.9, 0.1, 0.1],
            "geminin_norm": [0.1, 0.1, 0.9, 0.9],
            "condensation_score": [0.0, 0.0, 0.0, 0.9],
        }
    )
    out = classify_cell_cycle(df, config)
    assert list(out["cell_cycle_call"]) == ["G1", "S", "G2", "M"]


def test_classify_cell_cycle_condensation_overrides_ambiguous_markers():
    # Geminin high + condensed, even if SLBP also happens to read high mid-transition
    config = PipelineConfig()
    df = pd.DataFrame(
        {
            "cdt1_norm": [0.1],
            "slbp_norm": [0.9],
            "geminin_norm": [0.9],
            "condensation_score": [0.9],
        }
    )
    out = classify_cell_cycle(df, config)
    assert list(out["cell_cycle_call"]) == ["M"], "condensation must win over a stale SLBP signal"


def test_mphase_duration_tracks_consecutive_runs_and_resets_at_track_boundary():
    df = pd.DataFrame(
        {
            "track_id": [1, 1, 1, 1, 1, 2, 2],
            "frame": [0, 1, 2, 3, 4, 0, 1],
            "cell_cycle_call": ["G2", "M", "M", "G1", "M", "M", "M"],
            "frame_interval_min": [6.0] * 7,
        }
    )
    out = compute_mphase_duration(df)
    durations = out.set_index(["track_id", "frame"])["m_phase_duration"]

    assert np.isnan(durations[(1, 0)])  # G2, not M
    assert durations[(1, 1)] == 6.0  # first frame of an M run
    assert durations[(1, 2)] == 12.0  # second consecutive M frame
    assert np.isnan(durations[(1, 3)])  # G1 interrupts the run
    assert durations[(1, 4)] == 6.0  # a fresh M run starts here
    # track 2 starts its own M-run from frame 0, must NOT inherit track 1's run state
    assert durations[(2, 0)] == 6.0
    assert durations[(2, 1)] == 12.0


def test_track_coverage_and_filter():
    df = pd.DataFrame(
        {
            "track_id": [1, 1, 1, 1, 1, 2, 2],  # track 1: 5/10 frames, track 2: 2/10
            "frame": [0, 1, 2, 3, 4, 0, 1],
        }
    )
    coverage = compute_track_coverage(df, n_total_frames=10)
    assert coverage[1] == 0.5
    assert coverage[2] == 0.2

    config = PipelineConfig(min_track_coverage=0.4)
    filtered = filter_by_track_coverage(df, config, n_total_frames=10)
    assert set(filtered["track_id"]) == {1}


def test_classify_infection_otsu_separates_bimodal_populations():
    rng = np.random.default_rng(0)
    uninfected = rng.normal(100, 10, 200)
    infected = rng.normal(800, 30, 50)
    df = pd.DataFrame(
        {
            "experiment_id": ["exp1"] * 250,
            "mean_intensity": np.concatenate([uninfected, infected]),
        }
    )
    config = PipelineConfig(infection_gate_method="otsu")
    out = classify_infection(df, config)
    assert out["is_infected"].sum() == 50, out["is_infected"].sum()
    assert out.iloc[:200]["is_infected"].sum() == 0
    assert out.iloc[200:]["is_infected"].sum() == 50


def test_classify_infection_gmm_separates_bimodal_populations():
    rng = np.random.default_rng(1)
    uninfected = rng.normal(100, 10, 200)
    infected = rng.normal(800, 30, 50)
    df = pd.DataFrame(
        {
            "experiment_id": ["exp1"] * 250,
            "mean_intensity": np.concatenate([uninfected, infected]),
        }
    )
    config = PipelineConfig(infection_gate_method="gmm")
    out = classify_infection(df, config)
    assert out.iloc[:200]["is_infected"].sum() == 0
    assert out.iloc[200:]["is_infected"].sum() == 50


def test_classify_infection_is_per_experiment():
    # exp2's whole population is dimmer than exp1's -- must not use a shared global threshold
    df = pd.DataFrame(
        {
            "experiment_id": ["exp1"] * 4 + ["exp2"] * 4,
            "mean_intensity": [100, 105, 900, 910] + [10, 12, 90, 95],
        }
    )
    config = PipelineConfig(infection_gate_method="otsu")
    out = classify_infection(df, config)
    exp1_infected = out[out["experiment_id"] == "exp1"]["is_infected"]
    exp2_infected = out[out["experiment_id"] == "exp2"]["is_infected"]
    assert list(exp1_infected) == [False, False, True, True]
    assert list(exp2_infected) == [False, False, True, True]


if __name__ == "__main__":
    _run("per-track min-max normalization", test_normalize_per_track_min_max)
    _run("degenerate span falls back to 0.5", test_normalize_per_track_degenerate_span_falls_back_to_half)
    _run("cell-cycle precedence G1/S/G2/M", test_classify_cell_cycle_precedence)
    _run("condensation overrides ambiguous markers for M", test_classify_cell_cycle_condensation_overrides_ambiguous_markers)
    _run("m_phase_duration tracks runs and resets at track boundary", test_mphase_duration_tracks_consecutive_runs_and_resets_at_track_boundary)
    _run("track coverage computation and filter", test_track_coverage_and_filter)
    _run("infection classification (Otsu) separates bimodal populations", test_classify_infection_otsu_separates_bimodal_populations)
    _run("infection classification (GMM) separates bimodal populations", test_classify_infection_gmm_separates_bimodal_populations)
    _run("infection classification is per-experiment, not global", test_classify_infection_is_per_experiment)
    print("\nAll classification tests passed.")
