"""Strict leave-one-aria-out baselines from openSMILE functionals.

The outer aria is never used for feature cleaning, model selection, or fitting.
Missing and non-finite feature values are imputed with inner-training medians.
eGeMAPS is evaluated with linear and RBF classifiers; the high-dimensional
ComParE representation uses linear classifiers to keep the comparison stable.
"""

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
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scheme", required=True)
    parser.add_argument(
        "--classifier-profile",
        choices=["linear", "linear_rbf"],
        default="linear_rbf",
    )
    parser.add_argument("--c-grid", default="0.001,0.01,0.1,1,10")
    parser.add_argument("--rbf-c-grid", default="0.1,1,10,100")
    parser.add_argument("--rbf-gamma-grid", default="scale,0.01,0.1")
    parser.add_argument("--jobs", type=int, default=14)
    return parser.parse_args()


def candidates_from_args(args: argparse.Namespace) -> list[tuple[str, tuple]]:
    c_grid = [float(value) for value in args.c_grid.split(",")]
    candidates = [
        (algorithm, (c_value,))
        for algorithm in ["logistic", "linear_svm"]
        for c_value in c_grid
    ]
    if args.classifier_profile == "linear_rbf":
        rbf_c_grid = [float(value) for value in args.rbf_c_grid.split(",")]
        rbf_gamma_grid = [
            value if value == "scale" else float(value)
            for value in args.rbf_gamma_grid.split(",")
        ]
        candidates.extend(
            ("rbf_svm", (c_value, gamma))
            for c_value in rbf_c_grid
            for gamma in rbf_gamma_grid
        )
    return candidates


def align_features(metadata: pd.DataFrame, feature_dir: Path) -> np.ndarray:
    feature_metadata = pd.read_csv(feature_dir / "metadata.csv")
    if len(feature_metadata) != len(metadata):
        raise ValueError("openSMILE and experiment metadata have different row counts")
    expected = metadata.segment_id.astype(str).to_numpy()
    observed = feature_metadata.segment_id.astype(str).to_numpy()
    if not np.array_equal(expected, observed):
        raise ValueError("openSMILE features are not in experiment-metadata order")
    completed_path = feature_dir / "completed.npy"
    if completed_path.exists():
        completed = np.load(completed_path).astype(bool)
        if not completed.all():
            raise ValueError(
                f"openSMILE extraction incomplete: {int((~completed).sum())} rows missing"
            )
    values = np.asarray(np.load(feature_dir / "features.npy", mmap_mode="r"))
    if values.shape[0] != len(metadata):
        raise ValueError("openSMILE feature matrix has an invalid first dimension")
    return values.astype(np.float32, copy=False)


def train_only_impute(
    values: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    cleaned = np.asarray(values, dtype=np.float32).copy()
    cleaned[~np.isfinite(cleaned)] = np.nan
    train_values = cleaned[train_indices]
    with np.errstate(all="ignore"):
        medians = np.nanmedian(train_values, axis=0)
    all_missing = ~np.isfinite(medians)
    medians[all_missing] = 0.0
    missing = np.isnan(cleaned)
    if missing.any():
        rows, columns = np.where(missing)
        cleaned[rows, columns] = medians[columns]
    return cleaned, {
        "feature_count": int(cleaned.shape[1]),
        "all_missing_train_features": int(all_missing.sum()),
        "imputed_values": int(missing.sum()),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata).reset_index(drop=True)
    roles = pd.read_csv(args.fold_roles)
    raw_values = align_features(metadata, Path(args.feature_dir))
    candidates = candidates_from_args(args)

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    selection_rows: list[dict] = []
    cleaning_rows: list[dict] = []

    for outer_fold in sorted(roles.outer_fold.unique()):
        fold_roles = roles.loc[
            roles.outer_fold == outer_fold, ["outer_fold", "segment_id", "role"]
        ]
        fold = metadata.merge(
            fold_roles, on="segment_id", how="inner", validate="one_to_one"
        )
        indices = {
            role: fold.index[fold.role == role].to_numpy()
            for role in ["inner_train", "inner_dev", "outer_test"]
        }
        # The merge preserves metadata order, but indexing raw_values by fold
        # positions would be fragile if that behavior changed. Align explicitly.
        row_lookup = pd.Series(metadata.index.to_numpy(), index=metadata.segment_id)
        fold_values = raw_values[row_lookup.loc[fold.segment_id].to_numpy()]
        values, cleaning = train_only_impute(fold_values, indices["inner_train"])
        cleaning_rows.append({"outer_fold": int(outer_fold), **cleaning})
        metrics, predictions, selections, _ = evaluate_representation(
            args.scheme, values, fold, indices, candidates, args.jobs
        )
        metric_rows.extend(metrics)
        prediction_rows.extend(predictions)
        selection_rows.extend(selections)
        pd.DataFrame(metric_rows).to_csv(
            output / "per_fold_target_metrics.partial.csv", index=False
        )
        pd.DataFrame(prediction_rows).to_csv(
            output / "predictions_private.partial.csv", index=False
        )
        pd.DataFrame(selection_rows).to_csv(
            output / "inner_selection.partial.csv", index=False
        )
        print(f"completed openSMILE baseline fold {outer_fold}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "per_fold_target_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        output / "predictions_private.csv", index=False
    )
    pd.DataFrame(selection_rows).to_csv(output / "inner_selection.csv", index=False)
    pd.DataFrame(cleaning_rows).to_csv(output / "cleaning_audit.csv", index=False)
    summary = (
        metrics.groupby("scheme")[[
            "uar_present",
            "macro_f1_present",
            "uar_global",
            "macro_f1_global",
            "accuracy",
        ]]
        .mean()
        .reset_index()
    )
    summary.to_csv(output / "summary.csv", index=False)
    feature_manifest_path = Path(args.feature_dir) / "manifest.json"
    feature_manifest = (
        json.loads(feature_manifest_path.read_text(encoding="utf-8"))
        if feature_manifest_path.exists()
        else None
    )
    manifest = {
        "protocol": "strict_leave_one_aria_out_with_inner_source_disjoint_tuning",
        "selection_partition": "inner_dev",
        "fit_partition": "inner_train",
        "test_used_for_selection": False,
        "scheme": args.scheme,
        "classifier_profile": args.classifier_profile,
        "candidates": candidates,
        "feature_manifest": feature_manifest,
        "nonfinite_policy": "inner-training median imputation per outer fold",
        "metric_policy": "present-class utility for inner selection; pooled OOF global-class metrics downstream",
        "hyperparameters": vars(args),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
