"""Stages 4 & 5: cell-cycle and infection classification.

Operates on a per-(track_id, frame) dataframe that already has, per row:
`cdt1`, `slbp`, `geminin` (raw per-cell mean intensities from Stage 2/3
joins), `condensation_score` (from Stage 2), and `frame_interval_min`.

Cell-cycle phase precedence is deliberately G1 < S < G2 < M (each later
assignment below overwrites earlier ones on overlapping/ambiguous rows) --
this matches the biological progression through the cycle, so an ambiguous
transition frame is called as the more-advanced phase, and a condensed
nucleus is always called M regardless of what the other markers say (per
the design rationale: condensation is the primary M-phase signal, Geminin
only confirms).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig


def normalize_per_track(df: pd.DataFrame, channels: list[str], track_col: str = "track_id") -> pd.DataFrame:
    """Per-cell (per-track) min-max normalization of each channel, using
    only that track's own observed range. Controls for both cell-to-cell
    expression-level variation and photobleaching over the movie, since
    every cell's own reporters are assumed to cycle low-to-high-to-low
    within its observed lifetime.

    Cells whose observed range never actually spans a full cycle (e.g. very
    short tracks, or genuinely quiescent cells) get a degenerate span, and
    fall back to 0.5 (neither high nor low) rather than a division blow-up.
    This is exactly the failure mode `min_track_coverage` filtering (see
    `compute_track_coverage`) is meant to catch downstream.
    """
    df = df.copy()
    for ch in channels:
        norm_col = f"{ch}_norm"
        group = df.groupby(track_col)[ch]
        ch_min = group.transform("min")
        ch_max = group.transform("max")
        span = ch_max - ch_min
        df[norm_col] = np.where(span > 0, (df[ch] - ch_min) / span, 0.5)
    return df


def classify_cell_cycle(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Assigns `cell_cycle_call` per row from normalized marker intensities
    plus `condensation_score`. Requires `cdt1_norm`/`slbp_norm`/`geminin_norm`
    (from `normalize_per_track`) and `condensation_score` columns.
    """
    df = df.copy()
    t = config.phase_gate_threshold
    cdt1_high = df["cdt1_norm"] >= t
    slbp_high = df["slbp_norm"] >= t
    geminin_high = df["geminin_norm"] >= t
    condensed = df["condensation_score"] >= config.mitosis_condensation_threshold

    call = np.full(len(df), "unknown", dtype=object)
    call[(cdt1_high & ~geminin_high).to_numpy()] = "G1"
    call[slbp_high.to_numpy()] = "S"
    call[(geminin_high & ~slbp_high & ~cdt1_high).to_numpy()] = "G2"
    call[(geminin_high & condensed).to_numpy()] = "M"  # highest precedence: condensation decides M

    df["cell_cycle_call"] = call
    return df


def compute_mphase_duration(df: pd.DataFrame, track_col: str = "track_id", frame_col: str = "frame") -> pd.DataFrame:
    """Running elapsed time (minutes) since the current track's present
    high-condensation (M-phase) run began; NaN on non-M rows. Resets at
    track boundaries and whenever a run of M frames is interrupted, so an
    `mitotic_outcome` classification downstream (arrested vs. normal, per
    the placeholder `normal_mphase_reference_min` threshold) can just
    filter on this column's max per track -- no need to re-derive run
    boundaries again there.
    """
    df = df.sort_values([track_col, frame_col]).reset_index(drop=True)
    is_m = df["cell_cycle_call"] == "M"

    track_changed = df[track_col] != df[track_col].shift(1)
    run_break = (~is_m) | track_changed
    run_id = run_break.cumsum()

    consec_count = is_m.groupby(run_id).cumsum()
    duration = pd.Series(np.nan, index=df.index)
    duration[is_m] = consec_count[is_m] * df.loc[is_m, "frame_interval_min"]
    df["m_phase_duration"] = duration
    return df


def compute_track_coverage(df: pd.DataFrame, n_total_frames: int, track_col: str = "track_id") -> pd.Series:
    """Fraction of the movie's frames each track appears in."""
    return df.groupby(track_col).size() / n_total_frames


def filter_by_track_coverage(df: pd.DataFrame, config: PipelineConfig, n_total_frames: int, track_col: str = "track_id") -> pd.DataFrame:
    """Drops rows belonging to tracks observed in fewer than
    `config.min_track_coverage` of the movie's frames -- guards against
    distorted per-track normalization from truncated cycle observation.
    A cheap, adjustable filter: re-run with a different threshold without
    touching tracking/segmentation.
    """
    coverage = compute_track_coverage(df, n_total_frames, track_col=track_col)
    keep_ids = coverage[coverage >= config.min_track_coverage].index
    return df[df[track_col].isin(keep_ids)].copy()


def mask_cell_cycle_for_infected(df: pd.DataFrame) -> pd.DataFrame:
    """Infected cells don't express FUCCI-4 reporters, so their cdt1/slbp/
    geminin channels are pure background -- per-track normalization on
    background noise still produces *some* cell_cycle_call (typically a
    degenerate-span fallback), which is meaningless and would silently
    pollute Stage 7 fate transitions and Stage 8's embedding if left in.
    Requires `classify_infection` to have already run (needs `is_infected`).
    """
    df = df.copy()
    infected = df["is_infected"]
    df.loc[infected, "cell_cycle_call"] = "not_applicable"
    df.loc[infected, "m_phase_duration"] = np.nan
    return df


def _otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """Standard Otsu's method: threshold maximizing between-class variance.

    When classes are well-separated, every bin spanning the empty gap
    between them ties for the maximum variance (weight1/weight2/mean1/mean2
    don't change until a data point is crossed). Naively taking
    `argmax` picks the *first* tied bin, which sits right at the edge of
    the lower cluster -- any real data point jittering slightly above that
    edge (common with real, noisy intensities) gets misclassified. Take the
    midpoint of the tied plateau instead, landing in the middle of the gap.
    """
    hist, bin_edges = np.histogram(values, bins=n_bins)
    hist = hist.astype(np.float64)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    weight1 = np.cumsum(hist)
    weight2 = weight1[-1] - weight1
    cumsum_intensity = np.cumsum(hist * bin_centers)
    total_intensity = cumsum_intensity[-1]

    mean1 = np.divide(cumsum_intensity, weight1, out=np.zeros_like(cumsum_intensity), where=weight1 > 0)
    mean2 = np.divide(
        total_intensity - cumsum_intensity, weight2, out=np.zeros_like(cumsum_intensity), where=weight2 > 0
    )
    inter_class_variance = weight1 * weight2 * (mean1 - mean2) ** 2
    max_var = inter_class_variance.max()
    tied_indices = np.flatnonzero(np.isclose(inter_class_variance, max_var))
    idx = int((tied_indices[0] + tied_indices[-1]) // 2)
    return float(bin_centers[idx])


def classify_infection(
    df: pd.DataFrame,
    config: PipelineConfig,
    intensity_col: str = "mean_intensity",
    experiment_col: str = "experiment_id",
) -> pd.DataFrame:
    """Population-identity infection call, adaptive per experiment (Otsu or
    GMM on the segmented object's intensity) -- the sole authoritative
    infected/uninfected call (Stage 2's coarse area pre-filter is not this).
    """
    df = df.copy()
    df["is_infected"] = False

    for _, group in df.groupby(experiment_col):
        values = group[intensity_col].to_numpy(dtype=np.float64)
        if config.infection_gate_method == "otsu":
            threshold = _otsu_threshold(values)
            is_infected = values >= threshold
        elif config.infection_gate_method == "gmm":
            from sklearn.mixture import GaussianMixture

            log_values = np.log1p(values).reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, random_state=0).fit(log_values)
            component_labels = gmm.predict(log_values)
            infected_component = int(np.argmax(gmm.means_.flatten()))
            is_infected = component_labels == infected_component
        else:
            raise ValueError(f"Unknown infection_gate_method: {config.infection_gate_method!r}")

        df.loc[group.index, "is_infected"] = is_infected

    return df
