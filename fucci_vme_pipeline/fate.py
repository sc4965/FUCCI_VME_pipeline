"""Stage 7: fate / lineage analysis.

Once tracking and classification exist, fate transition probabilities are
just a groupby/crosstab on the base table -- this module is deliberately
thin. `delta_frames` is left as a parameter (not fixed to one window) so any
Δt can be explored without recomputing anything upstream; callers filter
`df` first (by starting state, exposure condition, etc.) before calling, so
this stays a generic Markov-table builder rather than special-casing any one
analysis.
"""
from __future__ import annotations

import pandas as pd


def compute_fate_transition_table(
    df: pd.DataFrame,
    state_col: str,
    delta_frames: int,
    track_col: str = "track_id",
    frame_col: str = "frame",
) -> pd.DataFrame:
    """P(state at t+delta_frames | state at t), as a row-normalized
    transition matrix. Only pairs of rows from the *same track* that are
    exactly `delta_frames` apart are counted -- a gap in a track's frames
    (e.g. from upstream filtering) simply contributes no pair for that gap.
    """
    if delta_frames <= 0:
        raise ValueError(f"delta_frames must be positive, got {delta_frames}")

    base = df[[track_col, frame_col, state_col]].copy()
    shifted = base.copy()
    shifted[frame_col] = shifted[frame_col] - delta_frames

    merged = base.merge(shifted, on=[track_col, frame_col], suffixes=("_t0", "_t1"))
    if merged.empty:
        return pd.DataFrame()

    counts = pd.crosstab(merged[f"{state_col}_t0"], merged[f"{state_col}_t1"])
    probs = counts.div(counts.sum(axis=1), axis=0)
    return probs
