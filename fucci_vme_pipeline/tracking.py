"""Stage 3: tracking.

FUCCI-4 population: btrack (division/death-aware Bayesian tracking), since
this population is expected to include real divisions during the 36-48 hpi
imaging window.
Infected population: simple frame-to-frame nearest-centroid linking, since
lytic-infected cells aren't expected to divide -- no need for the heavier
division/death hypothesis machinery there.

*** IMPORTANT, found via direct testing against the real btrack library ***
The vendored `configs/cell_config.json` (btrack's own community-standard
default) does NOT guarantee that a real division produces a parent/child
link in the output, out of the box. In a synthetic test with a mother cell
splitting into two symmetric daughters, btrack's motion-based linking step
greedily continued the mother's track into whichever daughter it matched
first, and registered the other daughter as a brand-new, unrelated track
(`parent_track_id == -1`, no lineage connection at all) -- silently, with no
error. This happens because branch/division hypotheses are only considered
between a track that actually *terminates* and candidate children; if the
motion linker never lets the mother's track terminate (because it
successfully matched one daughter), no division hypothesis is ever
generated for it.

This means: division-detection sensitivity is governed by
`dist_thresh`/`time_thresh`/`lambda_branch` (hypothesis model) and
`accuracy`/`max_lost` (motion model) in `configs/cell_config.json`, and MUST
be validated against real annotated division events (per the validation
plan) before `parent_track_id`/`lineage_id` can be trusted. Do not treat
btrack's division calls as correct until that validation has happened --
this is a real, open task, not a solved problem.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .config import PipelineConfig

CELL_CONFIG_PATH = Path(__file__).parent / "configs" / "cell_config.json"


def track_fucci4_population(
    labels: np.ndarray,
    pixel_size_um: float,
    frame_interval_min: float,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Division/death-aware tracking via btrack.

    `labels` is a (T, Y, X) integer label stack. Returns a tidy dataframe,
    one row per (track, frame): frame, x, y, track_id, parent_track_id
    (-1 for founders, matching this pipeline's convention -- btrack itself
    uses parent == own ID for founders), lineage_id, generation, time_min.
    """
    import btrack
    from btrack.config import load_config
    from btrack.utils import segmentation_to_objects

    objects = segmentation_to_objects(labels, properties=("area",))

    cfg = load_config(CELL_CONFIG_PATH)
    cfg.max_search_radius = config.btrack_max_search_radius_um / pixel_size_um
    height, width = labels.shape[1], labels.shape[2]
    cfg.volume = ((0, width), (0, height))

    tracker = btrack.BayesianTracker()
    tracker.configure(cfg)
    tracker.append(objects)
    tracker.track(step_size=100)
    tracker.optimize()

    rows = []
    for tr in tracker.tracks:
        d = tr.to_dict()
        parent = -1 if tr.parent == tr.ID else tr.parent
        for i, frame in enumerate(d["t"]):
            rows.append(
                {
                    "frame": int(frame),
                    "x": float(d["x"][i]),
                    "y": float(d["y"][i]),
                    "track_id": int(tr.ID),
                    "parent_track_id": int(parent),
                    "lineage_id": int(tr.root),
                    "generation": int(d["generation"]),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["time_min"] = df["frame"] * frame_interval_min
    return df


def link_infected_population(
    centroids_per_frame: list[np.ndarray],
    max_distance_um: float,
    pixel_size_um: float,
    frame_interval_min: float,
) -> pd.DataFrame:
    """Simple frame-to-frame nearest-centroid linking (Hungarian algorithm),
    no division/death handling. `centroids_per_frame[t]` is an (N_t, 2) array
    of (y, x) centroids for frame t. Returns the same tidy schema as
    `track_fucci4_population` for downstream compatibility, with
    `parent_track_id = -1` and `generation = 0` always (no lineage here by
    design -- see module docstring).
    """
    max_distance_px = max_distance_um / pixel_size_um
    next_track_id = 0
    active_tracks: dict[int, np.ndarray] = {}  # track_id -> last known (y, x)
    rows = []

    for frame, centroids in enumerate(centroids_per_frame):
        if centroids.shape[0] == 0:
            active_tracks = {}
            continue

        if not active_tracks:
            for c in centroids:
                active_tracks[next_track_id] = c
                rows.append({"frame": frame, "x": c[1], "y": c[0], "track_id": next_track_id})
                next_track_id += 1
            continue

        active_ids = list(active_tracks.keys())
        prev_positions = np.stack([active_tracks[tid] for tid in active_ids])
        cost = np.linalg.norm(prev_positions[:, None, :] - centroids[None, :, :], axis=-1)
        cost_gated = np.where(cost <= max_distance_px, cost, 1e6)

        row_ind, col_ind = linear_sum_assignment(cost_gated)
        assigned_centroid_idx = set()
        new_active_tracks: dict[int, np.ndarray] = {}

        for r, c in zip(row_ind, col_ind):
            if cost_gated[r, c] >= 1e6:
                continue  # gated out: too far to be the same cell
            tid = active_ids[r]
            new_active_tracks[tid] = centroids[c]
            assigned_centroid_idx.add(c)
            rows.append({"frame": frame, "x": centroids[c][1], "y": centroids[c][0], "track_id": tid})

        for c_idx in range(centroids.shape[0]):
            if c_idx not in assigned_centroid_idx:
                new_active_tracks[next_track_id] = centroids[c_idx]
                rows.append(
                    {"frame": frame, "x": centroids[c_idx][1], "y": centroids[c_idx][0], "track_id": next_track_id}
                )
                next_track_id += 1

        active_tracks = new_active_tracks

    df = pd.DataFrame(rows)
    if not df.empty:
        df["parent_track_id"] = -1
        df["lineage_id"] = df["track_id"]
        df["generation"] = 0
        df["time_min"] = df["frame"] * frame_interval_min
    return df
