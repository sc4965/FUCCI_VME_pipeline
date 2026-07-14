"""Stage 6: neighbor / exposure definition.

Both representations are cheap, adjustable filters computed purely from
per-frame positions (plus infection calls) -- re-runnable in seconds without
touching segmentation/tracking:

- `compute_exposure`: adjustable-radius neighbor flag + continuous exposure
  score (sum of inverse-square or exponential decay over nearby infected
  cells), via `scipy.spatial.cKDTree`.
- `compute_delaunay_vme`: optional alternate adjacency representation via
  `scipy.spatial.Delaunay` -- kept available (cheap: milliseconds/frame at
  realistic cell counts) even though it isn't the primary analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree

from .config import PipelineConfig


def compute_exposure(df: pd.DataFrame, config: PipelineConfig, pixel_size_um: float) -> pd.DataFrame:
    """Per (cell, frame) for uninfected cells: distance to the nearest
    infected cell (um) and a continuous exposure score over all infected
    cells within `config.exposure_max_radius_um`. Infected cells themselves
    get NaN distance / zero exposure by convention (exposure describes
    uninfected cells' proximity to infection, not infected cells to each
    other). Requires `frame`, `x`, `y` (pixels), `is_infected` columns.
    """
    df = df.copy()
    df["dist_nearest_infected_um"] = np.nan
    df["exposure_score"] = 0.0
    df["is_vme_neighbor"] = False

    for _, group in df.groupby("frame"):
        infected_mask = group["is_infected"].to_numpy()
        if not infected_mask.any():
            continue

        infected_xy_um = group.loc[infected_mask, ["x", "y"]].to_numpy() * pixel_size_um
        tree = cKDTree(infected_xy_um)

        uninfected_idx = group.index[~infected_mask]
        if len(uninfected_idx) == 0:
            continue
        uninfected_xy_um = group.loc[uninfected_idx, ["x", "y"]].to_numpy() * pixel_size_um

        dist, _ = tree.query(uninfected_xy_um, k=1)
        df.loc[uninfected_idx, "dist_nearest_infected_um"] = dist
        df.loc[uninfected_idx, "is_vme_neighbor"] = dist <= config.neighbor_radius_um

        neighbor_lists = tree.query_ball_point(uninfected_xy_um, r=config.exposure_max_radius_um)
        exposure = np.zeros(len(uninfected_idx))
        for i, neighbor_indices in enumerate(neighbor_lists):
            if not neighbor_indices:
                continue
            d = np.linalg.norm(infected_xy_um[neighbor_indices] - uninfected_xy_um[i], axis=1)
            d = np.maximum(d, 1e-6)  # guard against div-by-zero for coincident positions
            if config.exposure_decay == "inverse_square":
                exposure[i] = np.sum(1.0 / d**2)
            elif config.exposure_decay == "exponential":
                exposure[i] = np.sum(np.exp(-d / config.exposure_max_radius_um))
            else:
                raise ValueError(f"Unknown exposure_decay: {config.exposure_decay!r}")
        df.loc[uninfected_idx, "exposure_score"] = exposure

    return df


def compute_delaunay_vme(df: pd.DataFrame, pixel_size_um: float) -> pd.DataFrame:
    """Alternate adjacency mode: True for an uninfected cell if it shares a
    Delaunay edge with at least one infected cell in that frame. Frames with
    fewer than 4 cells (or degenerate/collinear point sets) can't form a
    triangulation and are left False, not raised as an error, since sparse
    frames are an expected edge case, not a bug.
    """
    df = df.copy()
    df["is_vme_neighbor_delaunay"] = False

    for _, group in df.groupby("frame"):
        n = len(group)
        if n < 4:
            continue
        xy_um = group[["x", "y"]].to_numpy() * pixel_size_um
        try:
            tri = Delaunay(xy_um)
        except Exception:
            continue

        adjacency = [set() for _ in range(n)]
        for simplex in tri.simplices:
            for a in simplex:
                for b in simplex:
                    if a != b:
                        adjacency[a].add(b)

        is_infected = group["is_infected"].to_numpy()
        neighbor_flag = np.zeros(n, dtype=bool)
        for i in range(n):
            if is_infected[i]:
                continue
            if any(is_infected[j] for j in adjacency[i]):
                neighbor_flag[i] = True

        df.loc[group.index, "is_vme_neighbor_delaunay"] = neighbor_flag

    return df
