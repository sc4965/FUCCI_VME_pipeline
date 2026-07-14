"""Tests for Stage 8: time-derivatives + PCA/UMAP (real sklearn/umap, no mocks)."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__file__.rsplit("/tests/", 1)[0]))

from fucci_vme_pipeline.config import PipelineConfig  # noqa: E402
from fucci_vme_pipeline.dimensionality import compute_time_derivatives, run_pca_umap  # noqa: E402


def _run(label, fn):
    fn()
    print(f"[ OK ] {label}")


def test_time_derivative_first_frame_is_nan_and_rate_is_correct():
    df = pd.DataFrame(
        {
            "track_id": [1, 1, 1],
            "frame": [0, 1, 2],
            "cdt1": [0.0, 12.0, 24.0],  # +12 units/frame
            "frame_interval_min": [6.0, 6.0, 6.0],
        }
    )
    out = compute_time_derivatives(df, ["cdt1"])
    assert np.isnan(out.iloc[0]["cdt1_deriv"])
    assert abs(out.iloc[1]["cdt1_deriv"] - 2.0) < 1e-9  # 12 units / 6 min = 2 units/min
    assert abs(out.iloc[2]["cdt1_deriv"] - 2.0) < 1e-9


def test_time_derivative_resets_at_track_boundary():
    df = pd.DataFrame(
        {
            "track_id": [1, 1, 2, 2],
            "frame": [0, 1, 0, 1],
            "cdt1": [0.0, 100.0, 0.0, 5.0],
            "frame_interval_min": [6.0] * 4,
        }
    )
    out = compute_time_derivatives(df, ["cdt1"])
    track2_first = out[(out["track_id"] == 2) & (out["frame"] == 0)].iloc[0]
    assert np.isnan(track2_first["cdt1_deriv"]), "track 2 must not inherit track 1's trailing value"


def test_pca_umap_separates_two_distinct_clusters():
    rng = np.random.default_rng(0)
    n = 30
    cluster_a = rng.normal(0, 0.5, size=(n, 4)) + np.array([0, 0, 0, 0])
    cluster_b = rng.normal(0, 0.5, size=(n, 4)) + np.array([20, 20, 20, 20])
    X = np.vstack([cluster_a, cluster_b])
    df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
    config = PipelineConfig(n_pca_components=3, random_state=0)

    out = run_pca_umap(df, ["f1", "f2", "f3", "f4"], config)
    assert {"pca_1", "pca_2", "pca_3", "umap_1", "umap_2"} <= set(out.columns)
    assert not out["umap_1"].isna().any()

    umap_a_centroid = out.iloc[:n][["umap_1", "umap_2"]].mean()
    umap_b_centroid = out.iloc[n:][["umap_1", "umap_2"]].mean()
    dist = np.linalg.norm(umap_a_centroid - umap_b_centroid)
    assert dist > 1.0, f"expected well-separated clusters in UMAP space, got centroid distance {dist}"

    assert len(out.attrs["pca_explained_variance_ratio"]) == 3
    assert sum(out.attrs["pca_explained_variance_ratio"]) > 0.9  # 2 well-separated clusters is a near-1D signal


def test_pca_umap_drops_nan_rows_rather_than_imputing():
    df = pd.DataFrame(
        {
            "f1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
            "f2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    config = PipelineConfig(n_pca_components=1, random_state=0)
    out = run_pca_umap(df, ["f1", "f2"], config)
    assert pd.isna(out.iloc[2]["pca_1"])
    assert not pd.isna(out.iloc[0]["pca_1"])


if __name__ == "__main__":
    _run("time derivative: first frame NaN, rate correct", test_time_derivative_first_frame_is_nan_and_rate_is_correct)
    _run("time derivative resets at track boundary", test_time_derivative_resets_at_track_boundary)
    _run("PCA/UMAP separates two distinct clusters", test_pca_umap_separates_two_distinct_clusters)
    _run("PCA/UMAP drops NaN rows rather than imputing", test_pca_umap_drops_nan_rows_rather_than_imputing)
    print("\nAll dimensionality-reduction tests passed.")
