"""Reproduce segment-, source-, and aria-weighted metrics from OOF scores."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "predictions" / "main_oof_predictions.csv.gz"
OUTPUT = ROOT / "reproduced" / "metric_estimands.csv"
EXPECTED = ROOT / "results" / "metric_estimands.csv"

CLASS_COUNTS = {
    "chest_resonance": 4,
    "head_resonance": 4,
    "front_placement": 3,
    "back_placement": 3,
    "open_throat": 4,
    "breathiness": 2,
    "vibrato": 3,
}

EXPECTED_NAMES = {
    "MuQ multi-task": "MuQ",
    "Class-bias-adjusted MuQ": "Adjusted MuQ",
    "Class-bias-adjusted MERT/MuQ posterior fusion": "Adjusted fusion",
}

ESTIMANDS = {
    "segment_weighted": None,
    "source_equal_weighted": "source_uid",
    "aria_equal_weighted": "aria",
}


def weighted_confusion(
    truth: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    class_count: int,
) -> np.ndarray:
    matrix = np.zeros((class_count, class_count), dtype=np.float64)
    np.add.at(matrix, (truth, prediction), weight)
    return matrix


def metrics(matrix: np.ndarray) -> dict[str, float]:
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support > 0,
    )
    denominator = 2 * true_positive + (predicted - true_positive) + (
        support - true_positive
    )
    f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    return {
        "uar": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "accuracy": float(true_positive.sum() / support.sum()),
    }


def observation_weights(frame: pd.DataFrame, group: str | None) -> np.ndarray:
    if group is None:
        return np.ones(len(frame), dtype=np.float64)
    size = frame.groupby(group)[group].transform("size").to_numpy(dtype=np.float64)
    return 1.0 / size


def reproduce(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, method_frame in frame.groupby("method", sort=True):
        for estimand, equal_unit in ESTIMANDS.items():
            target_rows = []
            for target, target_frame in method_frame.groupby("target", sort=True):
                class_count = CLASS_COUNTS[target]
                weight = observation_weights(target_frame, equal_unit)
                matrix = weighted_confusion(
                    target_frame["y_true"].to_numpy(dtype=np.int64),
                    target_frame["y_pred"].to_numpy(dtype=np.int64),
                    weight,
                    class_count,
                )
                values = metrics(matrix)
                rows.append(
                    {
                        "method": method,
                        "estimand": estimand,
                        "level": "target",
                        "target": target,
                        **values,
                    }
                )
                target_rows.append(values)
            rows.append(
                {
                    "method": method,
                    "estimand": estimand,
                    "level": "target_macro",
                    "target": "ALL",
                    "uar": float(np.mean([row["uar"] for row in target_rows])),
                    "macro_f1": float(
                        np.mean([row["macro_f1"] for row in target_rows])
                    ),
                    "accuracy": float(
                        np.mean([row["accuracy"] for row in target_rows])
                    ),
                }
            )
    return pd.DataFrame(rows)


def verify_expected(reproduced: pd.DataFrame) -> None:
    expected = pd.read_csv(EXPECTED)
    expected = expected[
        (expected["level"] == "target_macro")
        & (expected["scheme"].isin(EXPECTED_NAMES.values()))
    ].copy()
    observed = reproduced[reproduced["level"] == "target_macro"].copy()
    failures = []
    for method, expected_name in EXPECTED_NAMES.items():
        for estimand in ESTIMANDS:
            left = observed[
                (observed["method"] == method)
                & (observed["estimand"] == estimand)
            ].iloc[0]
            right = expected[
                (expected["scheme"] == expected_name)
                & (expected["estimand"] == estimand)
            ].iloc[0]
            for metric in ["uar", "macro_f1", "accuracy"]:
                difference = abs(float(left[metric]) - float(right[metric]))
                if difference > 1e-10:
                    failures.append(
                        {
                            "method": method,
                            "estimand": estimand,
                            "metric": metric,
                            "difference": difference,
                        }
                    )
    if failures:
        raise AssertionError(
            "Locked metric mismatch:\n" + json.dumps(failures, indent=2)
        )


def main() -> None:
    frame = pd.read_csv(PREDICTIONS)
    reproduced = reproduce(frame)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    reproduced.to_csv(OUTPUT, index=False)
    verify_expected(reproduced)
    print(f"OK: reproduced {len(reproduced)} metric rows at {OUTPUT}")


if __name__ == "__main__":
    main()

