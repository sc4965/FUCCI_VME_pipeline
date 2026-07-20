"""Orchestrates all stages into a single tidy per-(cell, frame) table.

CLI usage:
    python -m fucci_vme_pipeline.pipeline --demo --out-csv demo_output.csv
    python -m fucci_vme_pipeline.pipeline \\
        --nuclear-nd2 h1.nd2 --cdt1-nd2 mscarlet3.nd2 \\
        --geminin-nd2 emirfp670.nd2 --slbp-nd2 mtagbfp2.nd2 \\
        --out-csv out.csv

`--max-frames`/`--frame-range` let you smoke-test the whole chain on a slice
of real data before committing GPU time to a full experiment, per the
"expensive, run once" vs. "cheap, adjustable" split in the design.

Merging the (green-channel-derived) object table with the tracker's output:
btrack/`link_infected_population` are called WITHOUT an intensity image, so
their reported (x, y) centroids are the same plain pixel-mean as this
module's own `regionprops_from_labels` -- confirmed by direct testing
against the real btrack library, not assumed. Rows are joined on
(frame, x, y) rounded to 6 decimals as a defensive tolerance. A small
fraction of objects (bounded by `config.track_merge_max_drop_fraction`) may
legitimately have no matching track -- btrack can reject some initial
detections as false positives (likely segmentation noise) -- but a larger
drop, or any row surplus, still raises loudly rather than trusting a
silently broken join. See `_merge_tracks_onto_objects` for the reasoning.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .classification import (
    classify_cell_cycle,
    classify_infection,
    compute_mphase_duration,
    filter_by_track_coverage,
    mask_cell_cycle_for_infected,
    normalize_per_track,
)
from .config import PipelineConfig
from .demo import make_synthetic_movie
from .dimensionality import compute_time_derivatives, run_pca_umap
from .ingestion import ChannelStack, read_nd2_channel, read_tiff_channel
from .neighbors import compute_delaunay_vme, compute_exposure
from .segmentation import mean_intensity_by_label, regionprops_from_labels, run_cellpose_sam
from .tracking import link_infected_population, track_fucci4_population

OTHER_CHANNELS = ("cdt1", "slbp", "geminin")


def build_object_table(
    labels: np.ndarray,
    images: dict[str, np.ndarray],
    config: PipelineConfig,
    frame_interval_min: float,
) -> pd.DataFrame:
    """Per-(frame, object) table: regionprops from the green channel plus
    per-object mean intensity sampled from every other channel using the
    SAME labels (Stage 2's "green locates, other channels characterize").
    """
    rows = []
    for t in range(labels.shape[0]):
        props = regionprops_from_labels(labels[t], images["nuclear_infection"][t], config)
        other_means = {role: mean_intensity_by_label(labels[t], images[role][t]) for role in OTHER_CHANNELS}
        for p in props:
            row = {
                "frame": t,
                "label": p.label,
                "x": p.centroid_x,
                "y": p.centroid_y,
                "area": p.area_px,
                "eccentricity": p.eccentricity,
                "condensation_score": p.condensation_score,
                "mean_intensity": p.mean_intensity,
                "population": p.population,
            }
            for role, means in other_means.items():
                row[role] = means.get(p.label, np.nan)
            rows.append(row)

    df = pd.DataFrame(rows)
    df["frame_interval_min"] = frame_interval_min
    return df


def _labels_for_population(labels: np.ndarray, object_table: pd.DataFrame, population: str) -> np.ndarray:
    """Zeros out objects not in the given coarse population, frame by
    frame, using the exact mask label IDs already in `object_table` -- so
    btrack/the simple linker only ever sees the relevant subset.
    """
    filtered = np.zeros_like(labels)
    for t in range(labels.shape[0]):
        keep_labels = object_table.loc[
            (object_table["frame"] == t) & (object_table["population"] == population), "label"
        ].to_numpy()
        if keep_labels.size == 0:
            continue
        mask = np.isin(labels[t], keep_labels)
        filtered[t][mask] = labels[t][mask]
    return filtered


def _merge_tracks_onto_objects(
    object_subset: pd.DataFrame, track_df: pd.DataFrame, max_drop_fraction: float = 0.02
) -> pd.DataFrame:
    """Merges tracker output back onto the (green-channel-derived) object
    table via (frame, x, y), on the confirmed assumption that btrack/the
    simple linker (called without an intensity image) report the same
    plain pixel-mean centroid as `regionprops_from_labels`.

    Uses a LEFT join (object table on the left), not an inner join: an
    object with no matching track is KEPT, with `track_id`/`parent_track_id`/
    `lineage_id` as NaN, rather than silently discarded. This matters
    because "no matching track" isn't always noise -- confirmed on real
    data via direct label-at-click checks against the raw segmentation
    mask that a genuinely well-segmented condensed mitotic chromatin object
    can fail to link into its own otherwise-continuous track, because
    btrack's appearance/motion hypothesis model doesn't expect a tracked
    cell's appearance to change that sharply. An inner join here would
    delete that object with no trace it ever existed; keeping it as an
    untracked row lets `stitch_short_gaps` (called right after this)
    attempt to bridge it back into its track using position alone. Any
    object still untracked after that stitching attempt is dropped by the
    caller -- same end result as before for objects that really are noise.

    A drop beyond `max_drop_fraction` (now measured as the untracked
    fraction, before stitching) is NOT tolerated -- that would suggest the
    (frame, x, y) join assumption itself has broken, not just ordinary
    false-positive rejection, and the output shouldn't be trusted until
    investigated.
    """
    if track_df.empty:
        raise ValueError("Tracking produced no output for a non-empty object subset.")

    left = object_subset.copy()
    right = track_df.copy()
    left["_x_round"] = left["x"].round(6)
    left["_y_round"] = left["y"].round(6)
    right["_x_round"] = right["x"].round(6)
    right["_y_round"] = right["y"].round(6)

    merged = left.merge(
        right.drop(columns=["x", "y"]),
        on=["frame", "_x_round", "_y_round"],
        how="left",
    ).drop(columns=["_x_round", "_y_round"])

    n_untracked = int(merged["track_id"].isna().sum())
    untracked_fraction = n_untracked / len(merged) if len(merged) else 0.0

    if n_untracked > 0:
        if untracked_fraction > max_drop_fraction:
            raise RuntimeError(
                f"Track merge left {n_untracked}/{len(merged)} objects "
                f"({untracked_fraction:.1%}) untracked, exceeding the "
                f"{max_drop_fraction:.1%} tolerance for expected false-positive "
                "rejection. The (frame, x, y) join assumption may have broken -- "
                "do not trust this output until investigated."
            )
        print(
            f"Note: {n_untracked}/{len(merged)} objects ({untracked_fraction:.2%}) had no "
            "matching track -- kept as untracked rows for stitch_short_gaps to "
            "attempt to recover, rather than dropped outright."
        )
    if len(merged) > len(object_subset):
        raise RuntimeError(
            f"Track merge produced MORE rows ({len(merged)}) than input objects "
            f"({len(object_subset)}) -- a duplicate (frame, x, y) key means the "
            "join can no longer be trusted to be 1:1. Do not trust this output "
            "until investigated."
        )
    return merged


def stitch_short_gaps(
    df: pd.DataFrame, max_gap_frames: int = 3, max_stitch_distance_px: float = 60.0
) -> pd.DataFrame:
    """Bridges short gaps in existing tracks using untracked objects, purely
    by position -- no appearance features involved.

    Motivation: `_merge_tracks_onto_objects` now keeps objects btrack didn't
    link (see its docstring) instead of discarding them. Some of those are
    genuine noise, but some are real cells btrack's appearance/motion
    hypothesis model failed to link into an otherwise-good track, because
    that one frame's appearance changed too sharply (confirmed on real data
    for condensed mitotic chromatin specifically). Since the goal here is
    only to bridge a short gap in a track that's already mostly right, this
    deliberately ignores appearance entirely and asks a much narrower
    question: is there an untracked object sitting close to where this
    track's own recent motion says it should be?

    For each track, for each gap of length 1..max_gap_frames, linearly
    interpolates the expected (x, y) at each missing frame from the track's
    position just before and just after the gap, and claims the closest
    untracked object at that exact frame if it's within
    `max_stitch_distance_px`. Each untracked object can be claimed at most
    once. Anything still untracked afterward is left for the caller to drop,
    same as before this function existed -- this only recovers gaps inside
    tracks btrack mostly got right, not brand-new tracks, and is not a
    substitute for validating btrack's own division/false-positive behavior.
    """
    tracked = df[df["track_id"].notna()].copy()
    orphans = df[df["track_id"].isna()].copy()

    if tracked.empty:
        return tracked  # nothing tracked at all -- surface as empty, don't keep raw untracked rows
    if orphans.empty:
        return df  # nothing to stitch; df already equals tracked here

    lineage_cols = [c for c in ("parent_track_id", "lineage_id", "generation") if c in tracked.columns]
    stitched_rows = []
    claimed_orphan_indices: set = set()

    for track_id, group in tracked.groupby("track_id"):
        frames = sorted(group["frame"].unique())
        for i in range(len(frames) - 1):
            gap_size = frames[i + 1] - frames[i] - 1
            if not (0 < gap_size <= max_gap_frames):
                continue

            before = group[group["frame"] == frames[i]].iloc[0]
            after = group[group["frame"] == frames[i + 1]].iloc[0]
            span = frames[i + 1] - frames[i]

            for missing_frame in range(frames[i] + 1, frames[i + 1]):
                frac = (missing_frame - frames[i]) / span
                expected_x = before["x"] + frac * (after["x"] - before["x"])
                expected_y = before["y"] + frac * (after["y"] - before["y"])

                candidates = orphans[
                    (orphans["frame"] == missing_frame) & (~orphans.index.isin(claimed_orphan_indices))
                ]
                if candidates.empty:
                    continue

                dists = np.sqrt((candidates["x"] - expected_x) ** 2 + (candidates["y"] - expected_y) ** 2)
                best_idx = dists.idxmin()
                if dists[best_idx] <= max_stitch_distance_px:
                    stitched = candidates.loc[best_idx].copy()
                    stitched["track_id"] = track_id
                    for col in lineage_cols:
                        stitched[col] = before[col]
                    stitched_rows.append(stitched)
                    claimed_orphan_indices.add(best_idx)

    n_stitched = len(stitched_rows)
    n_still_untracked = len(orphans) - n_stitched
    if n_stitched > 0:
        print(
            f"Stitched {n_stitched} object(s) into existing track gaps by position alone; "
            f"{n_still_untracked} object(s) remain untracked."
        )
        return pd.concat([tracked, pd.DataFrame(stitched_rows)], ignore_index=True)

    if len(orphans) > 0:
        print(f"No stitchable gaps found; {len(orphans)} object(s) remain untracked.")
    return tracked


def _drop_ambiguous_duplicate_track_frames(df: pd.DataFrame) -> pd.DataFrame:
    """Drops every row belonging to a (track_id, frame) pair that appears
    more than once.

    Found on real data: a track can end up with two genuinely different
    positions claiming the same track_id at the same frame -- most likely
    btrack inserting a dummy (Kalman-predicted) position to bridge a
    perceived gap while the real detection for that same frame also gets
    attributed to the track. Either way, a track whose identity is a
    mashup of two different physical cells poisons everything built on
    top of it (per-track normalization, phase calls, exposure). There's no
    reliable way to tell which of the duplicate entries is "real" from
    here, so both are dropped rather than silently guessing -- printed
    loudly so a large count is investigable rather than invisible.
    """
    dupe_mask = df.duplicated(subset=["track_id", "frame"], keep=False)
    n_dupes = int(dupe_mask.sum())
    if n_dupes > 0:
        affected_tracks = sorted(df.loc[dupe_mask, "track_id"].unique().tolist())
        print(
            f"Note: dropping {n_dupes} row(s) across {len(affected_tracks)} track(s) "
            "with more than one position at the same frame (ambiguous track "
            f"identity): track_ids={affected_tracks[:20]}"
            f"{'...' if len(affected_tracks) > 20 else ''}"
        )
    return df[~dupe_mask].copy()


def _add_dimensionality_reduction(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Stage 8, restricted to the FUCCI-4 (non-infected) population -- an
    infected cell's cdt1/slbp/geminin channels are pure background (already
    masked to `not_applicable` cell-cycle calls by
    `mask_cell_cycle_for_infected`), so including their degenerate
    normalized-channel values in the embedding would silently pollute it
    with cells that have no real cell-cycle signal at all.
    """
    normalized_channels = [f"{ch}_norm" for ch in OTHER_CHANNELS]
    fucci4_subset = df[~df["is_infected"]].copy()
    fucci4_subset = compute_time_derivatives(fucci4_subset, normalized_channels)

    feature_cols = normalized_channels + ["area", "eccentricity"] + [f"{ch}_deriv" for ch in normalized_channels]
    embedded = run_pca_umap(fucci4_subset, feature_cols, config)

    new_cols = [c for c in embedded.columns if c not in df.columns]
    df = df.merge(embedded[["frame", "x", "y"] + new_cols], on=["frame", "x", "y"], how="left")
    return df


def reclassify(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Re-applies only the stages that depend on tunable classification
    thresholds (`mitosis_condensation_threshold`, `phase_gate_threshold`)
    to an already fully-processed output table.

    Skips ingestion, segmentation, tracking, per-track normalization, and
    infection classification entirely -- none of those depend on these
    thresholds, and their outputs (`condensation_score`, `cdt1_norm`/
    `slbp_norm`/`geminin_norm`, `is_infected`) are already saved columns
    in the input CSV. This is the "cheap, adjustable" side of the
    pipeline's architecture: re-tune a threshold and see the result in
    seconds of CPU time, without re-running GPU-bound segmentation or
    tracking again.
    """
    df = classify_cell_cycle(df, config)
    df = compute_mphase_duration(df)
    df = mask_cell_cycle_for_infected(df)
    return df


def run_demo(config: PipelineConfig, max_frames: int | None = None) -> pd.DataFrame:
    """Full pipeline on synthetic data -- no real files, no cellpose/GPU
    needed. The fast sanity check: confirms every stage's glue works before
    ever pointing this at real ND2s.
    """
    labels, images, pixel_size_um, frame_interval_min = make_synthetic_movie()
    if max_frames is not None:
        labels = labels[:max_frames]
        images = {role: arr[:max_frames] for role, arr in images.items()}

    object_table = build_object_table(labels, images, config, frame_interval_min)
    n_total_frames = labels.shape[0]

    fucci4_labels = _labels_for_population(labels, object_table, "nuclear_candidate")
    fucci4_tracks = track_fucci4_population(fucci4_labels, pixel_size_um, frame_interval_min, config)
    fucci4_objects = object_table[object_table["population"] == "nuclear_candidate"]
    fucci4_merged = _merge_tracks_onto_objects(
        fucci4_objects, fucci4_tracks, max_drop_fraction=config.track_merge_max_drop_fraction
    )
    fucci4_merged = stitch_short_gaps(
        fucci4_merged, config.gap_stitch_max_gap_frames, config.gap_stitch_max_distance_px
    )

    infected_objects = object_table[object_table["population"] == "large_candidate"]
    centroids_per_frame = [
        infected_objects[infected_objects["frame"] == t][["y", "x"]].to_numpy() for t in range(n_total_frames)
    ]
    infected_tracks = link_infected_population(
        centroids_per_frame, config.infected_link_max_distance_um, pixel_size_um, frame_interval_min
    )
    infected_merged = (
        _merge_tracks_onto_objects(
            infected_objects, infected_tracks, max_drop_fraction=config.track_merge_max_drop_fraction
        )
        if not infected_objects.empty
        else infected_objects
    )
    if not infected_merged.empty:
        infected_merged = infected_merged[infected_merged["track_id"].notna()]

    df = pd.concat([fucci4_merged, infected_merged], ignore_index=True)
    df = _drop_ambiguous_duplicate_track_frames(df)
    df["experiment_id"] = "demo"

    df = filter_by_track_coverage(df, config, n_total_frames)
    df = normalize_per_track(df, list(OTHER_CHANNELS))
    df = classify_cell_cycle(df, config)
    df = compute_mphase_duration(df)
    df = classify_infection(df, config, intensity_col="mean_intensity")
    df = mask_cell_cycle_for_infected(df)
    df = compute_exposure(df, config, pixel_size_um)
    if config.neighbor_mode == "delaunay":
        df = compute_delaunay_vme(df, pixel_size_um)
    df = _add_dimensionality_reduction(df, config)

    return df


def _load_channels(args: argparse.Namespace, config: PipelineConfig) -> dict[str, ChannelStack]:
    """Loads all four channels from whichever input format was given.
    Exactly one of ND2 or TIFF paths must be supplied -- mixing formats
    isn't supported and mistakenly leaving one set half-filled is a config
    error worth catching immediately rather than silently picking one.
    """
    nd2_paths = [args.nuclear_nd2, args.cdt1_nd2, args.geminin_nd2, args.slbp_nd2]
    tif_paths = [args.nuclear_tif, args.cdt1_tif, args.geminin_tif, args.slbp_tif]
    nd2_given = any(p is not None for p in nd2_paths)
    tif_given = any(p is not None for p in tif_paths)

    if nd2_given and tif_given:
        raise SystemExit("Pass either --*-nd2 flags or --*-tif flags, not both.")

    if nd2_given:
        if any(p is None for p in nd2_paths):
            raise SystemExit("All of --nuclear-nd2/--cdt1-nd2/--geminin-nd2/--slbp-nd2 are required together.")
        return {
            "nuclear_infection": read_nd2_channel(args.nuclear_nd2, config),
            "cdt1": read_nd2_channel(args.cdt1_nd2, config),
            "geminin": read_nd2_channel(args.geminin_nd2, config),
            "slbp": read_nd2_channel(args.slbp_nd2, config),
        }

    if tif_given:
        if any(p is None for p in tif_paths):
            raise SystemExit("All of --nuclear-tif/--cdt1-tif/--geminin-tif/--slbp-tif are required together.")
        return {
            "nuclear_infection": read_tiff_channel(
                args.nuclear_tif, config, pixel_size_um=args.pixel_size_um, frame_interval_min=args.frame_interval_min
            ),
            "cdt1": read_tiff_channel(
                args.cdt1_tif, config, pixel_size_um=args.pixel_size_um, frame_interval_min=args.frame_interval_min
            ),
            "geminin": read_tiff_channel(
                args.geminin_tif, config, pixel_size_um=args.pixel_size_um, frame_interval_min=args.frame_interval_min
            ),
            "slbp": read_tiff_channel(
                args.slbp_tif, config, pixel_size_um=args.pixel_size_um, frame_interval_min=args.frame_interval_min
            ),
        }

    raise SystemExit("Provide --demo, or all four --*-nd2 flags, or all four --*-tif flags.")


def main() -> None:
    parser = argparse.ArgumentParser(description="FUCCI/VME analysis pipeline")
    parser.add_argument("--demo", action="store_true", help="Run on synthetic data, no real files needed")
    parser.add_argument("--nuclear-nd2", type=str, help="mNeonGreen-H1.0 / surface-eGFP channel ND2")
    parser.add_argument("--cdt1-nd2", type=str)
    parser.add_argument("--geminin-nd2", type=str)
    parser.add_argument("--slbp-nd2", type=str)
    parser.add_argument("--nuclear-tif", type=str, help="mNeonGreen-H1.0 / surface-eGFP channel grayscale TIFF")
    parser.add_argument("--cdt1-tif", type=str)
    parser.add_argument("--geminin-tif", type=str)
    parser.add_argument("--slbp-tif", type=str)
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        default=None,
        help="Required for TIFF input unless the file has ImageJ hyperstack calibration metadata",
    )
    parser.add_argument(
        "--frame-interval-min",
        type=float,
        default=None,
        help="Required for TIFF input unless the file has ImageJ hyperstack calibration metadata",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Truncate to the first N frames")
    parser.add_argument(
        "--reclassify-csv",
        type=str,
        help=(
            "Path to an existing pipeline output CSV. Re-applies ONLY the "
            "classification thresholds below to it -- skips ingestion, "
            "segmentation, and tracking entirely, since condensation_score, "
            "geminin_norm etc. are already saved columns. Use this to "
            "re-tune a threshold in seconds without spending GPU time again."
        ),
    )
    parser.add_argument(
        "--mitosis-condensation-threshold",
        type=float,
        default=None,
        help="Overrides config.mitosis_condensation_threshold (PLACEHOLDER default 0.6 -- almost certainly needs tuning per dataset)",
    )
    parser.add_argument(
        "--phase-gate-threshold",
        type=float,
        default=None,
        help="Overrides config.phase_gate_threshold (default 0.5)",
    )
    parser.add_argument("--out-csv", type=str, required=True)
    args = parser.parse_args()

    config = PipelineConfig()
    if args.mitosis_condensation_threshold is not None:
        config.mitosis_condensation_threshold = args.mitosis_condensation_threshold
    if args.phase_gate_threshold is not None:
        config.phase_gate_threshold = args.phase_gate_threshold

    if args.reclassify_csv:
        df = pd.read_csv(args.reclassify_csv)
        df = reclassify(df, config)
    elif args.demo:
        df = run_demo(config, max_frames=args.max_frames)
    else:
        channels = _load_channels(args, config)
        nuclear, cdt1, geminin, slbp = (
            channels["nuclear_infection"],
            channels["cdt1"],
            channels["geminin"],
            channels["slbp"],
        )
        source_stem = Path(nuclear.source_path).stem

        images = {"nuclear_infection": nuclear.data, "cdt1": cdt1.data, "geminin": geminin.data, "slbp": slbp.data}
        if args.max_frames is not None:
            images = {role: arr[: args.max_frames] for role, arr in images.items()}

        labels = run_cellpose_sam(images["nuclear_infection"], config)
        object_table = build_object_table(labels, images, config, nuclear.frame_interval_min)
        n_total_frames = labels.shape[0]

        fucci4_labels = _labels_for_population(labels, object_table, "nuclear_candidate")
        fucci4_tracks = track_fucci4_population(fucci4_labels, nuclear.pixel_size_um, nuclear.frame_interval_min, config)
        fucci4_objects = object_table[object_table["population"] == "nuclear_candidate"]
        fucci4_merged = _merge_tracks_onto_objects(
        fucci4_objects, fucci4_tracks, max_drop_fraction=config.track_merge_max_drop_fraction
    )
        fucci4_merged = stitch_short_gaps(
            fucci4_merged, config.gap_stitch_max_gap_frames, config.gap_stitch_max_distance_px
        )

        infected_objects = object_table[object_table["population"] == "large_candidate"]
        centroids_per_frame = [
            infected_objects[infected_objects["frame"] == t][["y", "x"]].to_numpy() for t in range(n_total_frames)
        ]
        infected_tracks = link_infected_population(
            centroids_per_frame, config.infected_link_max_distance_um, nuclear.pixel_size_um, nuclear.frame_interval_min
        )
        infected_merged = (
            _merge_tracks_onto_objects(
                infected_objects, infected_tracks, max_drop_fraction=config.track_merge_max_drop_fraction
            )
            if not infected_objects.empty
            else infected_objects
        )
        if not infected_merged.empty:
            infected_merged = infected_merged[infected_merged["track_id"].notna()]

        df = pd.concat([fucci4_merged, infected_merged], ignore_index=True)
        df = _drop_ambiguous_duplicate_track_frames(df)
        df["experiment_id"] = source_stem

        df = filter_by_track_coverage(df, config, n_total_frames)
        df = normalize_per_track(df, list(OTHER_CHANNELS))
        df = classify_cell_cycle(df, config)
        df = compute_mphase_duration(df)
        df = classify_infection(df, config, intensity_col="mean_intensity")
        df = mask_cell_cycle_for_infected(df)
        df = compute_exposure(df, config, nuclear.pixel_size_um)
        if config.neighbor_mode == "delaunay":
            df = compute_delaunay_vme(df, nuclear.pixel_size_um)
        df = _add_dimensionality_reduction(df, config)

    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
