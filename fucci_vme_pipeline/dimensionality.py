"""Stage 8: PCA -> UMAP dimensionality reduction.

PCA runs first, both as a denoising step and as a check on how much
variance the normalized FUCCI channels alone explain. UMAP then runs on the
PCA output rather than raw features. Feature set is meant to be normalized
intensities + morphology (area, eccentricity) + short local time-derivatives
(see `compute_time_derivatives`) -- intensities alone would mostly just
re-derive the known FUCCI cycle diagram; morphology and derivatives let
arrested/aberrant cells show up as outliers instead of being folded into
that same 1D cycle manifold.
"""
from __future__ import annotations

import pandas as pd

from .config import PipelineConfig


def compute_time_derivatives(
    df: pd.DataFrame,
    channels: list[str],
    track_col: str = "track_id",
    frame_col: str = "frame",
) -> pd.DataFrame:
    """Per-track frame-to-frame rate of change for each channel
    (intensity-units per minute). NaN at each track's first observed frame,
    since there's no prior point to difference against.
    """
    df = df.sort_values([track_col, frame_col]).copy()
    for ch in channels:
        delta_value = df.groupby(track_col)[ch].diff()
        delta_time = df.groupby(track_col)[frame_col].diff() * df["frame_interval_min"]
        df[f"{ch}_deriv"] = delta_value / delta_time
    return df


def run_pca_umap(df: pd.DataFrame, feature_cols: list[str], config: PipelineConfig) -> pd.DataFrame:
    """Adds `pca_1..pca_n` and `umap_1`/`umap_2` columns. Rows with any NaN
    in `feature_cols` (e.g. a track's very first frame, before a derivative
    exists) are dropped from the embedding rather than imputed -- a
    fabricated "rate of change at the first observation" would be
    meaningless, not a reasonable default to invent.
    """
    from sklearn.decomposition import PCA
    import umap

    clean = df.dropna(subset=feature_cols)
    if clean.empty:
        raise ValueError(
            "No rows survive dropping NaNs in feature_cols -- check that "
            "upstream derivative/normalization columns are populated."
        )

    X = clean[feature_cols].to_numpy()
    n_components = min(config.n_pca_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=config.random_state)
    pca_coords = pca.fit_transform(X)

    reducer = umap.UMAP(n_components=2, random_state=config.random_state)
    umap_coords = reducer.fit_transform(pca_coords)

    result = df.copy()
    for i in range(n_components):
        result.loc[clean.index, f"pca_{i + 1}"] = pca_coords[:, i]
    result.loc[clean.index, "umap_1"] = umap_coords[:, 0]
    result.loc[clean.index, "umap_2"] = umap_coords[:, 1]
    result.attrs["pca_explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()

    return result
