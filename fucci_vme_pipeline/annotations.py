"""Matches hand-annotated ground truth (frame, x, y, label) against a
pipeline output table, for building/validating classifiers against real
confirmed examples rather than a hand-tuned heuristic threshold alone.

The matching tolerance matters more than it looks: an annotator marks a
point anywhere within a clicked nucleus, not necessarily its exact
centroid, so `max_distance_px` needs to cover a realistic nucleus radius --
confirmed via direct testing on real data that a too-tight tolerance
(20px) rejected the large majority of correctly-corresponding annotations.
A *much* larger, unexplained mismatch (found via a random-point baseline
comparison -- median real nearest-object distance statistically
indistinguishable from picking uniformly random points in the frame) turned
out to mean the annotations were made against the wrong acquisition
position entirely, not a tolerance problem -- worth checking for that
class of mismatch before assuming a bigger tolerance will fix things.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_annotations(path: str) -> pd.DataFrame:
    """Loads a CSV with columns: frame, x, y, label."""
    df = pd.read_csv(path)
    required = {"frame", "x", "y", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Annotations file missing required columns: {missing}")
    return df


def match_annotations_to_objects(
    annotations: pd.DataFrame,
    object_df: pd.DataFrame,
    max_distance_px: float = 40.0,
    label_to_population: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nearest-neighbor matches each annotation to the closest segmented
    object in the same frame, within `max_distance_px`.

    `label_to_population`, if given, restricts each annotation's candidate
    objects to only its expected population before the nearest-neighbor
    search -- e.g. `{"infected": "large_candidate"}` keeps an `infected`
    annotation from ever matching a nearby nuclear-scale object, and
    (implicitly, since it's just excluded from that population) keeps a
    `mitotic`/`dividing`/`non_mitotic` annotation from matching a nearby
    infected cell's much larger blob. Labels not present in the mapping
    are unrestricted. Matters in a densely-packed real field: found that
    plain nearest-neighbor search (no population restriction) can grab
    the wrong population's object entirely when cells are closer together
    than the matching tolerance needs to be to accommodate real click
    imprecision.

    Returns (matched, unmatched). `matched` has all of `object_df`'s
    columns plus `label` and `match_distance_px`; `unmatched` has the
    original annotation columns for whatever didn't find a close-enough
    object (worth inspecting -- could mean segmentation missed that cell,
    or the annotation/data are mismatched, per the module docstring).
    """
    matched_rows = []
    unmatched_rows = []

    for _, ann in annotations.iterrows():
        frame_objects = object_df[object_df["frame"] == ann["frame"]]
        if label_to_population and ann["label"] in label_to_population:
            frame_objects = frame_objects[frame_objects["population"] == label_to_population[ann["label"]]]
        if frame_objects.empty:
            unmatched_rows.append(ann)
            continue

        dists = np.sqrt((frame_objects["x"] - ann["x"]) ** 2 + (frame_objects["y"] - ann["y"]) ** 2)
        min_idx = dists.idxmin()
        if dists[min_idx] <= max_distance_px:
            matched = frame_objects.loc[min_idx].copy()
            matched["label"] = ann["label"]
            matched["match_distance_px"] = dists[min_idx]
            matched_rows.append(matched)
        else:
            unmatched_rows.append(ann)

    matched = pd.DataFrame(matched_rows).reset_index(drop=True)
    unmatched = pd.DataFrame(unmatched_rows).reset_index(drop=True)
    return matched, unmatched


def nearest_distances(annotations: pd.DataFrame, object_df: pd.DataFrame) -> pd.DataFrame:
    """Per-annotation nearest-object distance regardless of any tolerance
    cutoff -- the right first diagnostic when annotations aren't matching:
    reveals the real distribution (mostly-close-but-outside-a-tight-cutoff
    vs. genuinely nowhere-nearby) instead of a single opaque pass/fail.
    """
    records = []
    for _, ann in annotations.iterrows():
        frame_objects = object_df[object_df["frame"] == ann["frame"]]
        record = dict(ann)
        if frame_objects.empty:
            record["nearest_distance_px"] = np.nan
        else:
            dists = np.sqrt((frame_objects["x"] - ann["x"]) ** 2 + (frame_objects["y"] - ann["y"]) ** 2)
            record["nearest_distance_px"] = dists.min()
        records.append(record)
    return pd.DataFrame(records)


def random_baseline_distance(
    annotations: pd.DataFrame,
    object_df: pd.DataFrame,
    image_size: int,
    seed: int = 0,
) -> float:
    """Median nearest-object distance for uniformly random points in place
    of the real annotation coordinates. If the real annotations' median
    distance is close to this baseline, the annotations are statistically
    indistinguishable from having no real relationship to the segmented
    objects at all -- a sign of a data/position mismatch, not a tolerance
    or segmentation-sensitivity problem. Found this useful in exactly that
    situation on real data (annotations made against the wrong imaging
    position entirely).
    """
    rng = np.random.default_rng(seed)
    dists = []
    for _, ann in annotations.iterrows():
        frame_objects = object_df[object_df["frame"] == ann["frame"]]
        if frame_objects.empty:
            continue
        rx, ry = rng.uniform(0, image_size, size=2)
        d = np.sqrt((frame_objects["x"] - rx) ** 2 + (frame_objects["y"] - ry) ** 2)
        dists.append(d.min())
    return float(np.median(dists)) if dists else float("nan")
