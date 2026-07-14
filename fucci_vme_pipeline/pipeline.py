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
(frame, x, y) rounded to 6 decimals as a defensive tolerance; the merge
row-count is asserted to catch any future mismatch loudly rather than
silently dropping/duplicating cells.
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


def _merge_tracks_onto_objects(object_subset: pd.DataFrame, track_df: pd.DataFrame) -> pd.DataFrame:
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
        how="inner",
    ).drop(columns=["_x_round", "_y_round"])

    if len(merged) != len(object_subset):
        raise RuntimeError(
            f"Track merge row count mismatch: {len(object_subset)} objects vs. "
            f"{len(merged)} merged rows. The (frame, x, y) join assumption "
            "(tracker centroids match regionprops centroids exactly) may "
            "have broken -- do not trust this output until investigated."
        )
    return merged


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
    fucci4_merged = _merge_tracks_onto_objects(fucci4_objects, fucci4_tracks)

    infected_objects = object_table[object_table["population"] == "large_candidate"]
    centroids_per_frame = [
        infected_objects[infected_objects["frame"] == t][["y", "x"]].to_numpy() for t in range(n_total_frames)
    ]
    infected_tracks = link_infected_population(
        centroids_per_frame, config.infected_link_max_distance_um, pixel_size_um, frame_interval_min
    )
    infected_merged = _merge_tracks_onto_objects(infected_objects, infected_tracks) if not infected_objects.empty else infected_objects

    df = pd.concat([fucci4_merged, infected_merged], ignore_index=True)
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
    parser.add_argument("--out-csv", type=str, required=True)
    args = parser.parse_args()

    config = PipelineConfig()

    if args.demo:
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
        fucci4_merged = _merge_tracks_onto_objects(fucci4_objects, fucci4_tracks)

        infected_objects = object_table[object_table["population"] == "large_candidate"]
        centroids_per_frame = [
            infected_objects[infected_objects["frame"] == t][["y", "x"]].to_numpy() for t in range(n_total_frames)
        ]
        infected_tracks = link_infected_population(
            centroids_per_frame, config.infected_link_max_distance_um, nuclear.pixel_size_um, nuclear.frame_interval_min
        )
        infected_merged = (
            _merge_tracks_onto_objects(infected_objects, infected_tracks) if not infected_objects.empty else infected_objects
        )

        df = pd.concat([fucci4_merged, infected_merged], ignore_index=True)
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
