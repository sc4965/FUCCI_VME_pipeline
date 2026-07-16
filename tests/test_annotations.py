"""Tests for annotation-matching logic (pure pandas/numpy)."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__file__.rsplit("/tests/", 1)[0]))

from fucci_vme_pipeline.annotations import (  # noqa: E402
    match_annotations_to_objects,
    nearest_distances,
    random_baseline_distance,
)


def _run(label, fn):
    fn()
    print(f"[ OK ] {label}")


def test_matches_within_tolerance():
    annotations = pd.DataFrame({"frame": [0], "x": [105.0], "y": [95.0], "label": ["mitotic"]})
    object_df = pd.DataFrame({"frame": [0], "x": [100.0], "y": [100.0], "condensation_score": [0.5]})
    matched, unmatched = match_annotations_to_objects(annotations, object_df, max_distance_px=40.0)
    assert len(matched) == 1
    assert len(unmatched) == 0
    assert matched.iloc[0]["label"] == "mitotic"
    assert matched.iloc[0]["condensation_score"] == 0.5


def test_rejects_beyond_tolerance():
    annotations = pd.DataFrame({"frame": [0], "x": [500.0], "y": [500.0], "label": ["mitotic"]})
    object_df = pd.DataFrame({"frame": [0], "x": [100.0], "y": [100.0], "condensation_score": [0.5]})
    matched, unmatched = match_annotations_to_objects(annotations, object_df, max_distance_px=40.0)
    assert len(matched) == 0
    assert len(unmatched) == 1


def test_matches_nearest_of_several_candidates():
    annotations = pd.DataFrame({"frame": [0], "x": [102.0], "y": [98.0], "label": ["mitotic"]})
    object_df = pd.DataFrame(
        {
            "frame": [0, 0, 0],
            "x": [100.0, 300.0, 500.0],
            "y": [100.0, 300.0, 500.0],
            "condensation_score": [0.5, 0.1, 0.9],
        }
    )
    matched, _ = match_annotations_to_objects(annotations, object_df, max_distance_px=40.0)
    assert matched.iloc[0]["condensation_score"] == 0.5  # the close one, not the others


def test_population_restriction_avoids_cross_population_mismatch():
    # a mitotic annotation sits closer to a large infected blob than to the
    # real (slightly farther) nuclear-scale object it actually refers to --
    # without population restriction it would wrongly grab the blob
    annotations = pd.DataFrame({"frame": [0], "x": [100.0], "y": [100.0], "label": ["mitotic"]})
    object_df = pd.DataFrame(
        {
            "frame": [0, 0],
            "x": [105.0, 140.0],
            "y": [105.0, 140.0],
            "population": ["large_candidate", "nuclear_candidate"],
            "condensation_score": [0.01, 0.6],
        }
    )
    label_to_population = {"infected": "large_candidate", "mitotic": "nuclear_candidate", "dividing": "nuclear_candidate", "non_mitotic": "nuclear_candidate"}

    unrestricted, _ = match_annotations_to_objects(annotations, object_df, max_distance_px=100.0)
    assert unrestricted.iloc[0]["population"] == "large_candidate", "sanity: closer object is the blob, not the nucleus"

    restricted, _ = match_annotations_to_objects(
        annotations, object_df, max_distance_px=100.0, label_to_population=label_to_population
    )
    assert restricted.iloc[0]["population"] == "nuclear_candidate"
    assert restricted.iloc[0]["condensation_score"] == 0.6


def test_infected_annotation_restricted_to_large_candidate_population():
    annotations = pd.DataFrame({"frame": [0], "x": [100.0], "y": [100.0], "label": ["infected"]})
    object_df = pd.DataFrame(
        {
            "frame": [0, 0],
            "x": [102.0, 130.0],
            "y": [102.0, 130.0],
            "population": ["nuclear_candidate", "large_candidate"],
        }
    )
    label_to_population = {"infected": "large_candidate"}
    matched, _ = match_annotations_to_objects(
        annotations, object_df, max_distance_px=100.0, label_to_population=label_to_population
    )
    assert matched.iloc[0]["population"] == "large_candidate"


def test_no_objects_in_frame_is_unmatched_not_an_error():
    annotations = pd.DataFrame({"frame": [5], "x": [100.0], "y": [100.0], "label": ["mitotic"]})
    object_df = pd.DataFrame({"frame": [0], "x": [100.0], "y": [100.0], "condensation_score": [0.5]})
    matched, unmatched = match_annotations_to_objects(annotations, object_df)
    assert len(matched) == 0
    assert len(unmatched) == 1


def test_nearest_distances_reports_even_beyond_tolerance():
    annotations = pd.DataFrame({"frame": [0], "x": [500.0], "y": [500.0], "label": ["mitotic"]})
    object_df = pd.DataFrame({"frame": [0], "x": [100.0], "y": [100.0], "condensation_score": [0.5]})
    out = nearest_distances(annotations, object_df)
    expected = np.sqrt(400**2 + 400**2)
    assert abs(out.iloc[0]["nearest_distance_px"] - expected) < 1e-6


def test_random_baseline_is_high_when_real_annotations_are_close():
    # real annotations sit right on top of real objects (near-zero distance);
    # random points in a huge, sparsely-populated frame should be much farther
    rng_objects = pd.DataFrame(
        {"frame": [0] * 5, "x": [100.0, 900.0, 1700.0, 300.0, 1200.0], "y": [100.0, 900.0, 1700.0, 300.0, 1200.0]}
    )
    annotations = pd.DataFrame({"frame": [0] * 5, "x": rng_objects["x"] + 1, "y": rng_objects["y"] + 1, "label": ["mitotic"] * 5})
    baseline = random_baseline_distance(annotations, rng_objects, image_size=2048, seed=0)
    real_median = nearest_distances(annotations, rng_objects)["nearest_distance_px"].median()
    assert baseline > real_median, (baseline, real_median)


if __name__ == "__main__":
    _run("matches within tolerance", test_matches_within_tolerance)
    _run("rejects beyond tolerance", test_rejects_beyond_tolerance)
    _run("matches nearest of several candidates", test_matches_nearest_of_several_candidates)
    _run("population restriction avoids cross-population mismatch", test_population_restriction_avoids_cross_population_mismatch)
    _run("infected annotation restricted to large_candidate population", test_infected_annotation_restricted_to_large_candidate_population)
    _run("no objects in frame is unmatched, not an error", test_no_objects_in_frame_is_unmatched_not_an_error)
    _run("nearest_distances reports even beyond tolerance", test_nearest_distances_reports_even_beyond_tolerance)
    _run("random baseline exceeds real close-match median", test_random_baseline_is_high_when_real_annotations_are_close)
    print("\nAll annotation-matching tests passed.")
