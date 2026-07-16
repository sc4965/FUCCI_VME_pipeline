"""Tests for the classical mitosis classifier (real sklearn, no mocks)."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__file__.rsplit("/tests/", 1)[0]))

from fucci_vme_pipeline.mitosis_classifier import (  # noqa: E402
    prepare_training_data,
    predict,
    train_and_evaluate,
)


def _run(label, fn):
    fn()
    print(f"[ OK ] {label}")


def _make_matched_annotations(n_per_class: int = 15, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mitotic = pd.DataFrame(
        {
            "condensation_score": rng.normal(0.6, 0.05, n_per_class),
            "eccentricity": rng.normal(0.9, 0.05, n_per_class),
            "area": rng.normal(300, 20, n_per_class),
            "mean_intensity": rng.normal(400, 30, n_per_class),
            "geminin": rng.normal(600, 40, n_per_class),  # high: condensation alone isn't enough, Geminin must co-occur
            "label": "mitotic",
        }
    )
    dividing = pd.DataFrame(
        {
            "condensation_score": rng.normal(0.55, 0.05, 3),
            "eccentricity": rng.normal(0.85, 0.05, 3),
            "area": rng.normal(310, 20, 3),
            "mean_intensity": rng.normal(390, 30, 3),
            "geminin": rng.normal(580, 40, 3),
            "label": "dividing",
        }
    )
    non_mitotic = pd.DataFrame(
        {
            "condensation_score": rng.normal(0.1, 0.05, n_per_class),
            "eccentricity": rng.normal(0.2, 0.1, n_per_class),
            "area": rng.normal(500, 40, n_per_class),
            "mean_intensity": rng.normal(150, 20, n_per_class),
            "geminin": rng.normal(200, 40, n_per_class),  # low
            "label": "non_mitotic",
        }
    )
    infected = pd.DataFrame(
        {
            "condensation_score": rng.normal(0.05, 0.02, 5),
            "eccentricity": rng.normal(0.3, 0.1, 5),
            "area": rng.normal(2500, 200, 5),
            "mean_intensity": rng.normal(800, 50, 5),
            "geminin": rng.normal(50, 20, 5),  # infected cells don't express FUCCI-4 reporters at all
            "label": "infected",
        }
    )
    return pd.concat([mitotic, dividing, non_mitotic, infected], ignore_index=True)


def test_prepare_training_data_excludes_infected_and_folds_dividing_into_positive():
    matched = _make_matched_annotations()
    X, y, original_label = prepare_training_data(matched)
    assert "infected" not in original_label.unique()
    assert y[original_label == "dividing"].eq(1).all()
    assert y[original_label == "mitotic"].eq(1).all()
    assert y[original_label == "non_mitotic"].eq(0).all()


def test_train_and_evaluate_separates_clearly_distinct_classes():
    matched = _make_matched_annotations()
    X, y, _ = prepare_training_data(matched)
    result = train_and_evaluate(X, y)
    assert result["precision"] > 0.8, result["precision"]
    assert result["recall"] > 0.8, result["recall"]
    assert result["n_positive"] == 18  # 15 mitotic + 3 dividing
    assert result["n_negative"] == 15


def test_train_and_evaluate_rejects_too_few_examples():
    X = pd.DataFrame({"condensation_score": [0.1, 0.2, 0.3], "eccentricity": [0.1, 0.2, 0.3]})
    y = pd.Series([0, 1, 0])
    try:
        train_and_evaluate(X, y)
        raise AssertionError("expected ValueError for too few examples")
    except ValueError as e:
        assert "too few" in str(e)


def test_train_and_evaluate_rejects_single_class():
    X = pd.DataFrame({"a": range(12)})
    y = pd.Series([1] * 12)
    try:
        train_and_evaluate(X, y)
        raise AssertionError("expected ValueError for single-class data")
    except ValueError as e:
        assert "each class" in str(e)


def test_predict_applies_trained_model_to_new_data():
    matched = _make_matched_annotations()
    X, y, _ = prepare_training_data(matched)
    result = train_and_evaluate(X, y)

    new_mitotic = pd.DataFrame(
        {"condensation_score": [0.6], "eccentricity": [0.9], "area": [300], "mean_intensity": [400], "geminin": [600]}
    )
    new_non_mitotic = pd.DataFrame(
        {"condensation_score": [0.1], "eccentricity": [0.2], "area": [500], "mean_intensity": [150], "geminin": [200]}
    )
    assert predict(result["model"], new_mitotic)[0] == 1
    assert predict(result["model"], new_non_mitotic)[0] == 0


if __name__ == "__main__":
    _run("prepare_training_data excludes infected, folds dividing into positive", test_prepare_training_data_excludes_infected_and_folds_dividing_into_positive)
    _run("train_and_evaluate separates clearly distinct classes", test_train_and_evaluate_separates_clearly_distinct_classes)
    _run("train_and_evaluate rejects too few examples", test_train_and_evaluate_rejects_too_few_examples)
    _run("train_and_evaluate rejects single-class data", test_train_and_evaluate_rejects_single_class)
    _run("predict applies trained model to new data", test_predict_applies_trained_model_to_new_data)
    print("\nAll mitosis classifier tests passed.")
