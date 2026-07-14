"""Tests for Stage 6: neighbor/exposure logic (pure scipy/numpy, no real data needed)."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__file__.rsplit("/tests/", 1)[0]))

from fucci_vme_pipeline.config import PipelineConfig  # noqa: E402
from fucci_vme_pipeline.neighbors import compute_delaunay_vme, compute_exposure  # noqa: E402


def _run(label, fn):
    fn()
    print(f"[ OK ] {label}")


def test_exposure_identifies_close_neighbor_and_ignores_far_cell():
    # infected cell at origin; one uninfected cell close (10um), one far (500um)
    df = pd.DataFrame(
        {
            "frame": [0, 0, 0],
            "x": [0.0, 10.0, 500.0],
            "y": [0.0, 0.0, 0.0],
            "is_infected": [True, False, False],
        }
    )
    config = PipelineConfig(neighbor_radius_um=50.0, exposure_max_radius_um=200.0)
    out = compute_exposure(df, config, pixel_size_um=1.0)

    close_row = out[out["x"] == 10.0].iloc[0]
    far_row = out[out["x"] == 500.0].iloc[0]
    infected_row = out[out["x"] == 0.0].iloc[0]

    assert close_row["is_vme_neighbor"] == True  # noqa: E712
    assert close_row["dist_nearest_infected_um"] == 10.0
    assert close_row["exposure_score"] > 0

    assert far_row["is_vme_neighbor"] == False  # noqa: E712
    assert far_row["exposure_score"] == 0.0  # outside exposure_max_radius_um -> no contribution

    assert np.isnan(infected_row["dist_nearest_infected_um"])
    assert infected_row["exposure_score"] == 0.0


def test_exposure_score_sums_multiple_infected_neighbors():
    # two infected cells at distance 10 and 20 from one uninfected cell
    df = pd.DataFrame(
        {
            "frame": [0, 0, 0],
            "x": [0.0, 20.0, 10.0],
            "y": [0.0, 0.0, 0.0],
            "is_infected": [True, True, False],
        }
    )
    config = PipelineConfig(exposure_decay="inverse_square", exposure_max_radius_um=100.0)
    out = compute_exposure(df, config, pixel_size_um=1.0)
    uninfected_row = out[out["x"] == 10.0].iloc[0]

    # uninfected cell at x=10 is distance 10 from infected@0 and distance 10 from infected@20
    expected = 1 / 10.0**2 + 1 / 10.0**2
    assert abs(uninfected_row["exposure_score"] - expected) < 1e-9


def test_exposure_respects_pixel_size_conversion():
    # same as first test but positions in pixels with a non-1 pixel size,
    # so um distances differ from raw pixel differences
    df = pd.DataFrame(
        {
            "frame": [0, 0],
            "x": [0.0, 20.0],  # 20 px apart
            "y": [0.0, 0.0],
            "is_infected": [True, False],
        }
    )
    config = PipelineConfig(neighbor_radius_um=15.0)
    out = compute_exposure(df, config, pixel_size_um=0.5)  # 20px * 0.5 um/px = 10 um
    row = out[out["x"] == 20.0].iloc[0]
    assert abs(row["dist_nearest_infected_um"] - 10.0) < 1e-9
    assert row["is_vme_neighbor"] == True  # noqa: E712 -- 10um < 15um radius


def test_delaunay_vme_flags_adjacent_uninfected_cells():
    # infected cell at center of a small grid; its 4 orthogonal neighbors
    # should be flagged, a distant 5th uninfected cell should not be
    df = pd.DataFrame(
        {
            "frame": [0] * 6,
            "x": [0.0, 10.0, -10.0, 0.0, 0.0, 500.0],
            "y": [0.0, 0.0, 0.0, 10.0, -10.0, 500.0],
            "is_infected": [True, False, False, False, False, False],
        }
    )
    out = compute_delaunay_vme(df, pixel_size_um=1.0)
    near = out[out["x"].isin([10.0, -10.0, 0.0]) & (out["y"] != 0.0) | (out["x"].isin([10.0, -10.0]))]
    far = out[out["x"] == 500.0].iloc[0]
    assert far["is_vme_neighbor_delaunay"] == False  # noqa: E712
    # at least the immediate orthogonal neighbors should be flagged True
    immediate_neighbors = out[(out["x"].isin([10.0, -10.0])) | (out["y"].isin([10.0, -10.0]))]
    assert immediate_neighbors["is_vme_neighbor_delaunay"].all()


def test_delaunay_vme_skips_sparse_frames_without_error():
    df = pd.DataFrame(
        {
            "frame": [0, 0],
            "x": [0.0, 10.0],
            "y": [0.0, 0.0],
            "is_infected": [True, False],
        }
    )
    out = compute_delaunay_vme(df, pixel_size_um=1.0)  # only 2 points, can't triangulate
    assert (out["is_vme_neighbor_delaunay"] == False).all()  # noqa: E712


if __name__ == "__main__":
    _run("exposure identifies close neighbor, ignores far cell", test_exposure_identifies_close_neighbor_and_ignores_far_cell)
    _run("exposure score sums multiple infected neighbors", test_exposure_score_sums_multiple_infected_neighbors)
    _run("exposure respects pixel-to-micron conversion", test_exposure_respects_pixel_size_conversion)
    _run("Delaunay VME flags adjacent uninfected cells", test_delaunay_vme_flags_adjacent_uninfected_cells)
    _run("Delaunay VME skips sparse frames without error", test_delaunay_vme_skips_sparse_frames_without_error)
    print("\nAll neighbor/exposure tests passed.")
