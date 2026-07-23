"""Create prespecified equal-weight posterior fusions without label tuning."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metric_utils import CLASS_COUNTS, classification_metrics, primary_metric_view


KEYS = ["segment_id", "source_id", "outer_fold", "target", "y_true"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multitask-predictions", required=True)
    parser.add_argument("--beats-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def normalized_frame(path: str, scheme: str, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame.scheme == scheme].copy()
    if len(frame) == 0:
        raise ValueError(f"Scheme {scheme} not found in {path}")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"Duplicate keys in component {name}")
    frame = frame[KEYS + ["probability"]].rename(
        columns={"probability": f"probability_{name}"}
    )
    return frame


def probability(value: str, class_count: int) -> np.ndarray:
    output = np.asarray(json.loads(value), dtype=np.float64)
    if output.shape != (class_count,):
        raise ValueError(f"Expected {class_count} probabilities, got {output.shape}")
    return output / output.sum()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    components = {
        "mert": normalized_frame(
            args.multitask_predictions, "mert_multitask", "mert"
        ),
        "muq": normalized_frame(
            args.multitask_predictions, "muq_multitask", "muq"
        ),
        "beats": normalized_frame(
            args.beats_predictions, "beats_as2m_multitask", "beats"
        ),
    }
    aligned = None
    for name, frame in components.items():
        aligned = (
            frame
            if aligned is None
            else aligned.merge(frame, on=KEYS, how="inner", validate="one_to_one")
        )
    expected = len(next(iter(components.values())))
    if len(aligned) != expected or any(len(frame) != expected for frame in components.values()):
        raise ValueError("Foundation prediction files are not fully aligned")

    prediction_rows = []
    metric_rows = []
    combinations = [
        combination
        for size in [2, 3]
        for combination in itertools.combinations(components, size)
    ]
    for names in combinations:
        scheme = "_".join(names) + "_multitask_uniform_posterior"
        for target, target_frame in aligned.groupby("target", sort=False):
            class_count = CLASS_COUNTS[target]
            matrices = [
                np.stack(
                    [
                        probability(value, class_count)
                        for value in target_frame[f"probability_{name}"]
                    ]
                )
                for name in names
            ]
            fused = np.mean(matrices, axis=0)
            prediction = fused.argmax(axis=1)
            truth = target_frame.y_true.to_numpy(np.int64)
            values = classification_metrics(truth, prediction, class_count)
            metric_rows.append(
                {
                    "scheme": scheme,
                    "target": target,
                    **primary_metric_view(values),
                    "uar_global": values["uar_global"],
                    "macro_f1_global": values["macro_f1_global"],
                    "accuracy": values["accuracy"],
                }
            )
            for row, row_probability, row_prediction in zip(
                target_frame.itertuples(index=False), fused, prediction
            ):
                prediction_rows.append(
                    {
                        "segment_id": row.segment_id,
                        "source_id": row.source_id,
                        "outer_fold": int(row.outer_fold),
                        "scheme": scheme,
                        "target": target,
                        "y_true": int(row.y_true),
                        "y_pred": int(row_prediction),
                        "probability": json.dumps(row_probability.tolist()),
                    }
                )
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(output / "predictions_private.csv", index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "pooled_per_target_metrics.csv", index=False)
    metrics.groupby("scheme")[[
        "uar_global", "macro_f1_global", "accuracy"
    ]].mean().reset_index().to_csv(output / "summary.csv", index=False)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "protocol": "fixed_equal_weight_outer_test_posterior_fusions",
                "components": {
                    "mert": "mert_multitask",
                    "muq": "muq_multitask",
                    "beats": "beats_as2m_multitask",
                },
                "combinations": [list(value) for value in combinations],
                "weights": "uniform within every listed combination",
                "label_tuning": False,
                "selection_warning": "all prespecified pair and triple combinations must be reported",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pd.read_csv(output / "summary.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
