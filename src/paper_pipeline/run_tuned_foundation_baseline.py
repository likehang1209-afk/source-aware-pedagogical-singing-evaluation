"""Tuned frozen-representation probes for an externally extracted encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_tuned_strong_baselines import evaluate_representation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--fold-roles", required=True)
    parser.add_argument("--feature-metadata", required=True)
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--representation-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--statistic-count", type=int, default=5)
    parser.add_argument("--c-grid", default="0.001,0.01,0.1,1,10")
    parser.add_argument("--jobs", type=int, default=14)
    return parser.parse_args()


def aligned_features(
    metadata: pd.DataFrame, feature_metadata_path: str, feature_file: str
) -> np.ndarray:
    feature_metadata = pd.read_csv(feature_metadata_path).reset_index(drop=True)
    if feature_metadata.segment_id.duplicated().any():
        raise ValueError("External feature metadata contains duplicate segment IDs")
    row_lookup = pd.Series(
        feature_metadata.index.to_numpy(), index=feature_metadata.segment_id.astype(str)
    )
    missing = metadata.loc[
        ~metadata.segment_id.astype(str).isin(row_lookup.index), "segment_id"
    ]
    if len(missing):
        raise ValueError(f"External features lack {len(missing)} requested segments")
    source_lookup = pd.Series(
        feature_metadata.source_id.astype(str).to_numpy(),
        index=feature_metadata.segment_id.astype(str),
    )
    observed_sources = source_lookup.loc[metadata.segment_id.astype(str)].to_numpy()
    if not np.array_equal(observed_sources, metadata.source_id.astype(str).to_numpy()):
        raise ValueError("Source IDs disagree after external-feature alignment")
    values = np.load(feature_file, mmap_mode="r")
    if len(values) != len(feature_metadata):
        raise ValueError("External feature matrix and metadata row counts differ")
    indices = row_lookup.loc[metadata.segment_id.astype(str)].to_numpy()
    aligned = np.asarray(values[indices], dtype=np.float32)
    if not np.isfinite(aligned).all():
        raise ValueError("External feature matrix contains non-finite values")
    return aligned


def append_selected_pooling(
    representation: str,
    fold: pd.DataFrame,
    indices: dict[str, np.ndarray],
    fold_probabilities: dict[str, list[np.ndarray]],
    metric_rows: list[dict],
    prediction_rows: list[dict],
    selection_rows: list[dict],
) -> None:
    from metric_utils import CLASS_COUNTS, TARGETS
    from run_tuned_strong_baselines import metric_values

    mean_scheme = f"{representation}_mean_tuned_probe"
    stats_scheme = f"{representation}_five_stats_tuned_probe"
    selected_scheme = f"{representation}_inner_selected_pooling_probe"
    outer_fold = int(fold.outer_fold.iloc[0])
    test_frame = fold.loc[
        indices["outer_test"], ["segment_id", "source_id"]
    ].reset_index(drop=True)
    for task, target in enumerate(TARGETS):
        candidates = [
            row
            for row in selection_rows
            if row["outer_fold"] == outer_fold
            and row["target"] == target
            and row["scheme"] in {mean_scheme, stats_scheme}
        ]
        chosen = max(
            candidates,
            key=lambda row: (row["inner_utility"], row["scheme"] == mean_scheme),
        )
        probability = fold_probabilities[chosen["scheme"]][task]
        truth = fold.loc[indices["outer_test"], target].to_numpy(np.int64)
        prediction = probability.argmax(axis=1)
        metric_rows.append(
            {
                "outer_fold": outer_fold,
                "scheme": selected_scheme,
                "target": target,
                **metric_values(truth, prediction, CLASS_COUNTS[target]),
            }
        )
        selection_rows.append(
            {
                "outer_fold": outer_fold,
                "scheme": selected_scheme,
                "target": target,
                "algorithm": "derived_from_selected_representation",
                "parameters": chosen["scheme"],
                "inner_utility": chosen["inner_utility"],
            }
        )
        for row_index, row_probability in enumerate(probability):
            prediction_rows.append(
                {
                    "outer_fold": outer_fold,
                    "scheme": selected_scheme,
                    "segment_id": test_frame.loc[row_index, "segment_id"],
                    "source_id": test_frame.loc[row_index, "source_id"],
                    "target": target,
                    "y_true": int(truth[row_index]),
                    "y_pred": int(prediction[row_index]),
                    "probability": json.dumps(row_probability.tolist()),
                }
            )


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata).reset_index(drop=True)
    roles = pd.read_csv(args.fold_roles)
    values = aligned_features(
        metadata, args.feature_metadata, args.feature_file
    )
    if values.shape[1] % args.statistic_count:
        raise ValueError("Feature width is not divisible by statistic_count")
    embedding_width = values.shape[1] // args.statistic_count
    mean_values = values[:, :embedding_width]
    c_grid = [float(value) for value in args.c_grid.split(",")]
    candidates = [
        (algorithm, (c_value,))
        for algorithm in ["logistic", "linear_svm"]
        for c_value in c_grid
    ]
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    selection_rows: list[dict] = []

    for outer_fold in sorted(roles.outer_fold.unique()):
        fold_roles = roles.loc[
            roles.outer_fold == outer_fold, ["outer_fold", "segment_id", "role"]
        ]
        fold = metadata.merge(
            fold_roles, on="segment_id", how="inner", validate="one_to_one"
        )
        row_lookup = pd.Series(metadata.index.to_numpy(), index=metadata.segment_id)
        fold_rows = row_lookup.loc[fold.segment_id].to_numpy()
        fold_values = values[fold_rows]
        fold_mean = mean_values[fold_rows]
        indices = {
            role: fold.index[fold.role == role].to_numpy()
            for role in ["inner_train", "inner_dev", "outer_test"]
        }
        fold_probabilities = {}
        for scheme, representation_values in [
            (f"{args.representation_name}_mean_tuned_probe", fold_mean),
            (f"{args.representation_name}_five_stats_tuned_probe", fold_values),
        ]:
            metrics, predictions, selections, probabilities = evaluate_representation(
                scheme,
                representation_values,
                fold,
                indices,
                candidates,
                args.jobs,
            )
            metric_rows.extend(metrics)
            prediction_rows.extend(predictions)
            selection_rows.extend(selections)
            fold_probabilities[scheme] = probabilities
        append_selected_pooling(
            args.representation_name,
            fold,
            indices,
            fold_probabilities,
            metric_rows,
            prediction_rows,
            selection_rows,
        )
        pd.DataFrame(metric_rows).to_csv(
            output / "per_fold_target_metrics.partial.csv", index=False
        )
        pd.DataFrame(prediction_rows).to_csv(
            output / "predictions_private.partial.csv", index=False
        )
        pd.DataFrame(selection_rows).to_csv(
            output / "inner_selection.partial.csv", index=False
        )
        print(f"completed {args.representation_name} probe fold {outer_fold}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "per_fold_target_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / "predictions_private.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output / "inner_selection.csv", index=False)
    summary = (
        metrics.groupby("scheme")[[
            "uar_present", "macro_f1_present", "uar_global", "macro_f1_global", "accuracy"
        ]]
        .mean()
        .reset_index()
    )
    summary.to_csv(output / "summary.csv", index=False)
    manifest = {
        "protocol": "strict_leave_one_aria_out_inner_source_disjoint_tuned_single_task_probe",
        "representation": args.representation_name,
        "feature_file": args.feature_file,
        "feature_metadata": args.feature_metadata,
        "aligned_rows": len(metadata),
        "feature_width": int(values.shape[1]),
        "embedding_width": int(embedding_width),
        "statistics": args.statistic_count,
        "selection_partition": "inner_dev",
        "fit_partition": "inner_train",
        "test_used_for_selection": False,
        "candidates": candidates,
        "hyperparameters": vars(args),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
