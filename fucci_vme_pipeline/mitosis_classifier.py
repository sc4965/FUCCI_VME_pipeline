"""Classical-ML mitosis classifier trained on hand annotations, as a
data-driven alternative to the hand-tuned `mitosis_condensation_threshold`
heuristic.

Deliberately NOT a CNN: with an annotated set of dozens (not hundreds+ per
class) of examples, a simple classifier on top of already-computed
segmentation-time features is far more data-efficient than a model that
has to learn its own features from raw pixels. If this ceiling turns out
to be too low -- i.e. these features genuinely don't separate real
mitotic cells from confusable ones -- that's the signal to invest in a
CNN, not before.

`geminin` (raw per-object mean intensity in the 640nm channel) is included
deliberately, not an oversight to fix later: the hand-tuned heuristic
(`classify_cell_cycle`) always required condensed chromatin AND high
Geminin together, never condensation alone -- an earlier version of this
classifier omitted Geminin and, unsurprisingly, `condensation_score` alone
showed almost no separation between mitotic/dividing/non_mitotic on real
annotated data. `geminin` here is the raw per-object intensity (available
directly from segmentation), not the per-track-normalized `geminin_norm`
the heuristic uses -- deliberately, so this classifier doesn't inherit the
tracking-reliability issues found elsewhere in the pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_FEATURE_COLS = ["condensation_score", "eccentricity", "area", "mean_intensity", "geminin"]


def prepare_training_data(
    matched: pd.DataFrame,
    feature_cols: list[str] = DEFAULT_FEATURE_COLS,
    positive_labels: tuple[str, ...] = ("mitotic", "dividing"),
    exclude_labels: tuple[str, ...] = ("infected",),
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Builds (X, y, original_label) from matched annotations.

    `infected` is excluded by default -- it's ground truth for validating
    the separate infection classifier, not for mitosis, per this dataset's
    annotation scheme.

    `dividing` is folded into the positive class alongside `mitotic` by
    default, since a handful of examples isn't enough to train a reliable
    separate third class -- but the original label is returned too, so
    this can be revisited once more `dividing` examples exist.
    """
    usable = matched[~matched["label"].isin(exclude_labels)].copy()
    missing_features = [c for c in feature_cols if c not in usable.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    X = usable[feature_cols].reset_index(drop=True)
    y = usable["label"].isin(positive_labels).astype(int).reset_index(drop=True)
    original_label = usable["label"].reset_index(drop=True)
    return X, y, original_label


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> dict:
    """Trains a logistic regression classifier and evaluates it with
    leave-one-out cross-validation -- appropriate for the small sample
    sizes (dozens, not hundreds+, of examples) these annotation efforts
    realistically produce, where a single train/test split would leave
    too few examples in either half to trust.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, precision_score, recall_score
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(X) < 10:
        raise ValueError(
            f"Only {len(X)} labeled examples -- too few to train or "
            "meaningfully cross-validate a classifier. Gather more "
            "annotations first."
        )
    if y.nunique() < 2:
        raise ValueError("Need at least one example of each class (positive and negative) to train.")

    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
    loo_predictions = cross_val_predict(model, X, y, cv=LeaveOneOut())

    model.fit(X, y)  # final model, fit on all data, for actual use

    cm = confusion_matrix(y, loo_predictions)
    precision = precision_score(y, loo_predictions, zero_division=0)
    recall = recall_score(y, loo_predictions, zero_division=0)

    return {
        "model": model,
        "loo_predictions": loo_predictions,
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "n_examples": len(X),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
    }


def predict(model, df: pd.DataFrame, feature_cols: list[str] = DEFAULT_FEATURE_COLS) -> np.ndarray:
    """Applies a trained model to new rows (e.g. a full pipeline output
    table) with the same feature columns used for training.
    """
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")
    return model.predict(df[feature_cols])
