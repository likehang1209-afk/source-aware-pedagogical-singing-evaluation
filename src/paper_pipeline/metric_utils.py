"""Explicit classification metrics for the SVQTD experiments.

The held-aria folds do not always contain every globally defined class.  These
helpers report both estimands instead of delegating the choice to a library
default:

* ``present`` averages only classes represented in the evaluated ground truth;
* ``global`` averages all dataset-defined classes and assigns zero recall/F1 to
  a class with no evaluated examples.

Model selection uses the present-class estimand.  Manuscript tables should
state the estimand explicitly and accompany fold metrics with class supports.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


TARGETS = [
    "chest_resonance",
    "head_resonance",
    "front_placement",
    "back_placement",
    "open_throat",
    "breathiness",
    "vibrato",
]

CLASS_COUNTS = {
    "chest_resonance": 4,
    "head_resonance": 4,
    "front_placement": 3,
    "back_placement": 3,
    "open_throat": 4,
    "breathiness": 2,
    "vibrato": 3,
}


def _as_integer_vector(values: Iterable[int] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.equal(array, np.floor(array))):
            raise ValueError(f"{name} contains non-integer labels")
        array = array.astype(np.int64)
    return array.astype(np.int64, copy=False)


def confusion_counts(
    y_true: Iterable[int] | np.ndarray,
    y_pred: Iterable[int] | np.ndarray,
    class_count: int,
) -> np.ndarray:
    """Return a fixed-size confusion matrix with rows=true and columns=predicted."""
    truth = _as_integer_vector(y_true, "y_true")
    prediction = _as_integer_vector(y_pred, "y_pred")
    if truth.shape != prediction.shape:
        raise ValueError(f"Label shapes differ: {truth.shape} versus {prediction.shape}")
    if class_count <= 0:
        raise ValueError("class_count must be positive")
    if len(truth) and (
        truth.min() < 0
        or prediction.min() < 0
        or truth.max() >= class_count
        or prediction.max() >= class_count
    ):
        raise ValueError("Observed label falls outside the predefined class range")
    encoded = truth * class_count + prediction
    return np.bincount(encoded, minlength=class_count * class_count).reshape(
        class_count, class_count
    )


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, float | int | list[int]]:
    """Compute explicit present-class and global-class summaries."""
    matrix = np.asarray(confusion, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion must be a square matrix")
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support > 0,
    )
    f1_denominator = 2.0 * true_positive + (predicted - true_positive) + (
        support - true_positive
    )
    f1 = np.divide(
        2.0 * true_positive,
        f1_denominator,
        out=np.zeros_like(true_positive),
        where=f1_denominator > 0,
    )
    present = support > 0
    total = support.sum()
    uar_present = float(recall[present].mean()) if present.any() else float("nan")
    macro_f1_present = float(f1[present].mean()) if present.any() else float("nan")
    uar_global = float(recall.mean())
    macro_f1_global = float(f1.mean())
    accuracy = float(true_positive.sum() / total) if total else float("nan")
    return {
        "uar_present": uar_present,
        "macro_f1_present": macro_f1_present,
        "uar_global": uar_global,
        "macro_f1_global": macro_f1_global,
        "accuracy": accuracy,
        "utility_present": 0.5 * (uar_present + macro_f1_present),
        "utility_global": 0.5 * (uar_global + macro_f1_global),
        "sample_count": int(total),
        "present_class_count": int(present.sum()),
        "global_class_count": int(len(support)),
        "zero_support_classes": np.flatnonzero(~present).astype(int).tolist(),
        "class_support": support.astype(int).tolist(),
        "class_recall": recall.astype(float).tolist(),
        "class_f1": f1.astype(float).tolist(),
    }


def classification_metrics(
    y_true: Iterable[int] | np.ndarray,
    y_pred: Iterable[int] | np.ndarray,
    class_count: int,
) -> dict[str, float | int | list[int]]:
    return metrics_from_confusion(confusion_counts(y_true, y_pred, class_count))


def probability_metrics(
    y_true: Iterable[int] | np.ndarray,
    probability: np.ndarray,
    class_count: int,
) -> dict[str, float | int | list[int]]:
    probability = np.asarray(probability)
    if probability.ndim != 2 or probability.shape[1] != class_count:
        raise ValueError(
            f"Expected probability shape (n, {class_count}), got {probability.shape}"
        )
    return classification_metrics(y_true, probability.argmax(axis=1), class_count)


def primary_metric_view(values: dict[str, float | int | list[int]]) -> dict[str, float]:
    """Return the consistently defined present-class metrics used for selection."""
    return {
        "uar": float(values["uar_present"]),
        "macro_f1": float(values["macro_f1_present"]),
        "accuracy": float(values["accuracy"]),
        "utility": float(values["utility_present"]),
    }
