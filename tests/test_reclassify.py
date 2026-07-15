"""Test for reclassify(): re-applying classification thresholds to an
already fully-processed table without touching ingestion/segmentation/
tracking -- the "cheap, adjustable" re-tune path for expensive-to-generate
real data.
"""
from __future__ import annotations

import sys

import pandas as pd

sys.path.insert(0, str(__file__.rsplit("/tests/", 1)[0]))

from fucci_vme_pipeline.config import PipelineConfig  # noqa: E402
from fucci_vme_pipeline.pipeline import reclassify  # noqa: E402


def _run(label, fn):
    fn()
    print(f"[ OK ] {label}")


def _make_processed_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "track_id": [1, 1, 1],
            "frame": [0, 1, 2],
            "frame_interval_min": [6.0, 6.0, 6.0],
            "cdt1_norm": [0.1, 0.1, 0.1],
            "slbp_norm": [0.1, 0.1, 0.1],
            "geminin_norm": [0.9, 0.9, 0.9],
            "condensation_score": [0.5, 0.5, 0.5],
            "is_infected": [False, False, False],
        }
    )


def test_raising_threshold_eliminates_m_calls():
    df = _make_processed_table()
    low_config = PipelineConfig(mitosis_condensation_threshold=0.3)
    high_config = PipelineConfig(mitosis_condensation_threshold=0.9)

    low_out = reclassify(df, low_config)
    high_out = reclassify(df, high_config)

    assert (low_out["cell_cycle_call"] == "M").all(), low_out["cell_cycle_call"].tolist()
    assert not (high_out["cell_cycle_call"] == "M").any(), high_out["cell_cycle_call"].tolist()


def test_infected_rows_stay_not_applicable_regardless_of_threshold():
    df = _make_processed_table()
    df["is_infected"] = True
    config = PipelineConfig(mitosis_condensation_threshold=0.1)  # would trigger M if not masked
    out = reclassify(df, config)
    assert (out["cell_cycle_call"] == "not_applicable").all(), out["cell_cycle_call"].tolist()


def test_does_not_require_raw_intensity_or_population_columns():
    # confirms reclassify only needs already-derived columns, matching what
    # a real saved output CSV actually contains -- no raw cdt1/slbp/geminin,
    # no `population`, no `x`/`y` needed
    df = _make_processed_table()
    assert "cdt1" not in df.columns and "population" not in df.columns
    config = PipelineConfig()
    out = reclassify(df, config)  # should not raise
    assert "cell_cycle_call" in out.columns


if __name__ == "__main__":
    _run("raising the threshold eliminates M calls, lowering keeps them", test_raising_threshold_eliminates_m_calls)
    _run("infected rows stay not_applicable regardless of threshold", test_infected_rows_stay_not_applicable_regardless_of_threshold)
    _run("reclassify only needs already-derived columns", test_does_not_require_raw_intensity_or_population_columns)
    print("\nAll reclassify tests passed.")
