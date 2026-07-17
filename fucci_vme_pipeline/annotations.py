"""Matches hand-annotated ground truth (frame, x, y, label) against a
pipeline output table, for building/validating classifiers against real
confirmed examples rather than a hand-tuned heuristic threshold alone.

The matching tolerance is a real, load-bearing tradeoff, not a minor
detail -- confirmed empirically on real spot2 data by sweeping
`max_distance_px` from 50 to 200px and checking downstream classifier
performance at each:

- Too tight (20-40px) rejects the majority of genuinely correct
  annotations, since an annotator marks a point near a nucleus, not
  necessarily its exact centroid.
- Too loose (150-200px), in a densely-packed real field (~59px average
  inter-cell spacing here), starts silently matching annotations to the
  WRONG neighboring cell instead of the intended one. This showed up as a
  real, measurable effect: the Geminin signal for `mitotic`-labeled
  matches was diluted from a clean ~2x separation from `non_mitotic` at
  60px down to almost no separation at 150-200px, and end-to-end
  classifier precision/recall dropped from 0.85/0.85 (at 80px) to
  0.68/0.65 (at 100px) purely from this contamination.
- 80px was the empirically best balance found for this dataset (cleanest
  signal while still keeping enough examples per class to train on) --
  it is NOT a universal constant, and should be re-validated the same way
  (sweep + check downstream separation/performance) for any new dataset,
  since it depends on real cell density and annotator click precision,
  both of which can vary.

A separate failure mode, much larger and unrelated to tolerance tuning:
a random-point baseline comparison (median real nearest-object distance
statistically indistinguishable from picking uniformly random points in
the frame) revealed annotations made against the wrong acquisition
position entirely. Worth ruling that out first, before tuning tolerance,
if matching looks bad in a new dataset.
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
    max_distance_px: float = 80.0,
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
            # the object's own x/y (above) get overwritten nowhere here, but are
            # easy to mistake for the click itself during visual review -- keep
            # the original annotation coordinates too, under their own names,
            # so show_annotation_match can plot the actual click separately from
            # wherever nearest-neighbor matching landed.
            matched["ann_x"] = ann["x"]
            matched["ann_y"] = ann["y"]
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
