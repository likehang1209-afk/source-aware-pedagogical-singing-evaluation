"""Tuned SSL and acoustic baselines for strict leave-one-aria-out evaluation.

Every hyperparameter decision is made on the inner development partition.  The
held aria is evaluated once.  This script deliberately keeps the probes
single-task so that they provide a strong, fair reference for the controlled
multi-task-head ablation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from descriptor_utils import validate_descriptor_columns
from metric_utils import CLASS_COUNTS, TARGETS, classification_metrics, primary_metric_view


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--fold-roles", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--layer-selection", required=True)
    parser.add_argument("--pedagogy-features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--c-grid", default="0.001,0.01,0.1,1,10")
    parser.add_argument("--rbf-c-grid", default="0.1,1,10,100")
    parser.add_argument("--rbf-gamma-grid", default="scale,0.01,0.1")
    parser.add_argument("--jobs", type=int, default=14)
    return parser.parse_args()


def class_weights(labels: np.ndarray) -> dict[int, float]:
    classes, counts = np.unique(labels, return_counts=True)
    values = np.sqrt(counts.sum() / counts.astype(float))
    values /= values.mean()
    return {int(label): float(weight) for label, weight in zip(classes, values)}


def metric_values(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> dict:
    explicit = classification_metrics(y_true, y_pred, class_count)
    return {
        **primary_metric_view(explicit),
        "uar_present": float(explicit["uar_present"]),
        "macro_f1_present": float(explicit["macro_f1_present"]),
        "uar_global": float(explicit["uar_global"]),
        "macro_f1_global": float(explicit["macro_f1_global"]),
        "present_class_count": int(explicit["present_class_count"]),
        "global_class_count": int(explicit["global_class_count"]),
    }


def residualize_against_log_f0(
    values: np.ndarray, f0_hz: np.ndarray, train_indices: np.ndarray
) -> np.ndarray:
    """Remove train-estimated linear log-F0 effects without using test labels."""
    log_f0 = np.log(np.maximum(np.asarray(f0_hz, dtype=np.float64), 1.0))
    design = np.column_stack([np.ones(len(log_f0)), log_f0])
    coefficients, _, _, _ = np.linalg.lstsq(
        design[train_indices],
        np.asarray(values[train_indices], dtype=np.float64),
        rcond=None,
    )
    return (values - design @ coefficients).astype(np.float32)


def softmax(values: np.ndarray) -> np.ndarray:
    values = values - values.max(axis=1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(axis=1, keepdims=True)


def fixed_probability(
    model, features: np.ndarray, class_count: int, algorithm: str
) -> np.ndarray:
    if algorithm == "logistic":
        raw = model.predict_proba(features)
    else:
        decision = np.asarray(model.decision_function(features))
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])
        raw = softmax(decision)
    probability = np.zeros((len(features), class_count), dtype=np.float32)
    probability[:, np.asarray(model.classes_, dtype=int)] = raw
    row_sum = probability.sum(axis=1, keepdims=True)
    probability /= np.maximum(row_sum, 1e-12)
    return probability


def fit_candidate(
    algorithm: str,
    parameter: tuple,
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    class_count: int,
):
    unique = np.unique(train_y)
    if len(unique) == 1:
        probability = np.zeros((len(eval_x), class_count), dtype=np.float32)
        probability[:, int(unique[0])] = 1.0
        return None, probability

    weights = class_weights(train_y)
    if algorithm == "logistic":
        (c_value,) = parameter
        model = LogisticRegression(
            C=c_value,
            class_weight=weights,
            max_iter=2000,
            solver="lbfgs",
            tol=1e-4,
        )
    elif algorithm == "linear_svm":
        (c_value,) = parameter
        model = LinearSVC(
            C=c_value,
            class_weight=weights,
            dual=train_x.shape[0] <= train_x.shape[1],
            max_iter=50000,
            tol=1e-4,
        )
    elif algorithm == "rbf_svm":
        c_value, gamma = parameter
        model = SVC(
            C=c_value,
            gamma=gamma,
            class_weight=weights,
            decision_function_shape="ovr",
            probability=False,
            cache_size=2048,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    model.fit(train_x, train_y)
    return model, fixed_probability(model, eval_x, class_count, algorithm)


def tune_target(
    train_x: np.ndarray,
    train_y: np.ndarray,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    class_count: int,
    candidates: list[tuple[str, tuple]],
) -> dict:
    rows = []
    for algorithm, parameter in candidates:
        _, probability = fit_candidate(
            algorithm, parameter, train_x, train_y, dev_x, class_count
        )
        metrics = metric_values(dev_y, probability.argmax(axis=1), class_count)
        complexity_rank = {"logistic": 0, "linear_svm": 1, "rbf_svm": 2}[algorithm]
        c_value = float(parameter[0])
        rows.append(
            {
                "algorithm": algorithm,
                "parameter": parameter,
                "probability": probability,
                "utility": float(metrics["utility"]),
                "tie_break": (-complexity_rank, -abs(np.log10(c_value))),
            }
        )
    return max(rows, key=lambda row: (row["utility"], *row["tie_break"]))


def evaluate_representation(
    scheme: str,
    values: np.ndarray,
    fold: pd.DataFrame,
    indices: dict[str, np.ndarray],
    candidates: list[tuple[str, tuple]],
    jobs: int,
) -> tuple[list[dict], list[dict], list[dict], list[np.ndarray]]:
    scaler = StandardScaler().fit(values[indices["inner_train"]])
    train_x = scaler.transform(values[indices["inner_train"]]).astype(np.float32)
    dev_x = scaler.transform(values[indices["inner_dev"]]).astype(np.float32)
    test_x = scaler.transform(values[indices["outer_test"]]).astype(np.float32)
    train_y = fold.loc[indices["inner_train"], TARGETS].to_numpy(np.int64)
    dev_y = fold.loc[indices["inner_dev"], TARGETS].to_numpy(np.int64)
    test_y = fold.loc[indices["outer_test"], TARGETS].to_numpy(np.int64)

    selections = Parallel(n_jobs=jobs, prefer="threads")(
        delayed(tune_target)(
            train_x,
            train_y[:, task],
            dev_x,
            dev_y[:, task],
            CLASS_COUNTS[target],
            candidates,
        )
        for task, target in enumerate(TARGETS)
    )
    fitted = Parallel(n_jobs=jobs, prefer="threads")(
        delayed(fit_candidate)(
            selected["algorithm"],
            selected["parameter"],
            train_x,
            train_y[:, task],
            test_x,
            CLASS_COUNTS[target],
        )
        for task, (target, selected) in enumerate(zip(TARGETS, selections))
    )
    test_probabilities = [item[1] for item in fitted]
    metric_rows = []
    prediction_rows = []
    selection_rows = []
    test_frame = fold.loc[
        indices["outer_test"], ["segment_id", "source_id"]
    ].reset_index(drop=True)
    outer_fold = int(fold.outer_fold.iloc[0])
    for task, target in enumerate(TARGETS):
        selected = selections[task]
        probability = test_probabilities[task]
        prediction = probability.argmax(axis=1)
        metric_rows.append(
            {
                "outer_fold": outer_fold,
                "scheme": scheme,
                "target": target,
                **metric_values(test_y[:, task], prediction, CLASS_COUNTS[target]),
            }
        )
        selection_rows.append(
            {
                "outer_fold": outer_fold,
                "scheme": scheme,
                "target": target,
                "algorithm": selected["algorithm"],
                "parameters": json.dumps(selected["parameter"]),
                "inner_utility": selected["utility"],
            }
        )
        for row_index, row_probability in enumerate(probability):
            prediction_rows.append(
                {
                    "outer_fold": outer_fold,
                    "scheme": scheme,
                    "segment_id": test_frame.loc[row_index, "segment_id"],
                    "source_id": test_frame.loc[row_index, "source_id"],
                    "target": target,
                    "y_true": int(test_y[row_index, task]),
                    "y_pred": int(prediction[row_index]),
                    "probability": json.dumps(row_probability.tolist()),
                }
            )
    return metric_rows, prediction_rows, selection_rows, test_probabilities


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata)
    roles = pd.read_csv(args.fold_roles)
    layer_selection = pd.read_csv(args.layer_selection)
    descriptors = pd.read_csv(args.pedagogy_features).drop_duplicates("segment_id", keep="first")
    descriptor_validation = validate_descriptor_columns(descriptors, args.sample_rate)
    descriptor_columns = descriptor_validation["retained_features"]
    aligned_descriptors = metadata[["segment_id"]].merge(
        descriptors[["segment_id", *descriptor_columns]],
        on="segment_id",
        how="left",
        validate="one_to_one",
    )
    if aligned_descriptors[descriptor_columns].isna().any().any():
        raise ValueError("Missing acoustic descriptors after segment alignment")
    descriptor_values = aligned_descriptors[descriptor_columns].to_numpy(np.float32)
    no_level_columns = [
        column
        for column in descriptor_columns
        if column not in {"ped_rms", "ped_duration_sec"}
    ]
    no_level_values = aligned_descriptors[no_level_columns].to_numpy(np.float32)
    direct_f0_columns = {
        "ped_f0_hz_mean",
        "ped_f0_hz_std",
        "ped_f0_hz_median",
        "ped_f0_hz_iqr",
        "ped_f0_voiced_frames",
        "ped_f0_slope_abs_mean",
    }
    f0_residual_columns = [
        column for column in no_level_columns if column not in direct_f0_columns
    ]
    f0_residual_base = aligned_descriptors[f0_residual_columns].to_numpy(np.float32)
    f0_reference = aligned_descriptors["ped_f0_hz_mean"].to_numpy(np.float64)

    c_grid = [float(value) for value in args.c_grid.split(",")]
    rbf_c_grid = [float(value) for value in args.rbf_c_grid.split(",")]
    rbf_gamma_grid = [
        value if value == "scale" else float(value)
        for value in args.rbf_gamma_grid.split(",")
    ]
    linear_candidates = [
        (algorithm, (c_value,))
        for algorithm in ["logistic", "linear_svm"]
        for c_value in c_grid
    ]
    acoustic_candidates = [
        *linear_candidates,
        *[
            ("rbf_svm", (c_value, gamma))
            for c_value in rbf_c_grid
            for gamma in rbf_gamma_grid
        ],
    ]

    feature_cache: dict[tuple[str, int], np.ndarray] = {}
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
        indices = {
            role: fold.index[fold.role == role].to_numpy()
            for role in ["inner_train", "inner_dev", "outer_test"]
        }
        representation_specs = []
        for encoder in ["mert", "muq"]:
            selected = layer_selection[
                (layer_selection.outer_fold == outer_fold)
                & (layer_selection.encoder == encoder)
                & layer_selection.selected
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"Fold {outer_fold} has {len(selected)} selected {encoder} layers"
                )
            layer = int(selected.iloc[0].layer)
            key = (encoder, layer)
            if key not in feature_cache:
                feature_cache[key] = np.load(
                    Path(args.feature_dir) / f"{encoder}_layer_{layer:02d}.npy",
                    mmap_mode="r",
                )
            five_stats = feature_cache[key]
            width = five_stats.shape[1] // 5
            representation_specs.extend(
                [
                    (f"{encoder}_mean_tuned_probe", five_stats[:, :width], linear_candidates),
                    (f"{encoder}_five_stats_tuned_probe", five_stats, linear_candidates),
                ]
            )
        representation_specs.extend(
            [
                ("acoustic55_tuned_probe", descriptor_values, acoustic_candidates),
                (
                    "acoustic_no_level_duration_tuned_probe",
                    no_level_values,
                    acoustic_candidates,
                ),
                (
                    "acoustic_f0_residualized_no_level_tuned_probe",
                    residualize_against_log_f0(
                        f0_residual_base,
                        f0_reference,
                        indices["inner_train"],
                    ),
                    acoustic_candidates,
                ),
            ]
        )

        fold_probabilities: dict[str, list[np.ndarray]] = {}
        for scheme, values, candidates in representation_specs:
            metrics, predictions, selections, probabilities = evaluate_representation(
                scheme, values, fold, indices, candidates, args.jobs
            )
            metric_rows.extend(metrics)
            prediction_rows.extend(predictions)
            selection_rows.extend(selections)
            fold_probabilities[scheme] = probabilities

        # Select the MERT pooling representation per target using inner-dev only.
        # The two source models above are retained separately, so this derived
        # scheme does not hide whether mean or five-statistic pooling was chosen.
        for task, target in enumerate(TARGETS):
            candidates = [
                row
                for row in selection_rows
                if row["outer_fold"] == int(outer_fold)
                and row["target"] == target
                and row["scheme"]
                in {"mert_mean_tuned_probe", "mert_five_stats_tuned_probe"}
            ]
            chosen = max(
                candidates,
                key=lambda row: (
                    row["inner_utility"],
                    row["scheme"] == "mert_mean_tuned_probe",
                ),
            )
            probability = fold_probabilities[chosen["scheme"]][task]
            test_y = fold.loc[indices["outer_test"], target].to_numpy(np.int64)
            prediction = probability.argmax(axis=1)
            metric_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "scheme": "mert_inner_selected_pooling_probe",
                    "target": target,
                    **metric_values(test_y, prediction, CLASS_COUNTS[target]),
                }
            )
            selection_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "scheme": "mert_inner_selected_pooling_probe",
                    "target": target,
                    "algorithm": "derived_from_selected_representation",
                    "parameters": chosen["scheme"],
                    "inner_utility": chosen["inner_utility"],
                }
            )
            test_frame = fold.loc[
                indices["outer_test"], ["segment_id", "source_id"]
            ].reset_index(drop=True)
            for row_index, row_probability in enumerate(probability):
                prediction_rows.append(
                    {
                        "outer_fold": int(outer_fold),
                        "scheme": "mert_inner_selected_pooling_probe",
                        "segment_id": test_frame.loc[row_index, "segment_id"],
                        "source_id": test_frame.loc[row_index, "source_id"],
                        "target": target,
                        "y_true": int(test_y[row_index]),
                        "y_pred": int(prediction[row_index]),
                        "probability": json.dumps(row_probability.tolist()),
                    }
                )

        pd.DataFrame(metric_rows).to_csv(output / "per_fold_target_metrics.partial.csv", index=False)
        pd.DataFrame(prediction_rows).to_csv(output / "predictions_private.partial.csv", index=False)
        pd.DataFrame(selection_rows).to_csv(output / "inner_selection.partial.csv", index=False)
        print(f"completed tuned-baseline fold {outer_fold}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "per_fold_target_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / "predictions_private.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output / "inner_selection.csv", index=False)
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
    manifest = {
        "protocol": "strict_leave_one_aria_out_inner_source_disjoint_tuned_single_task_probes",
        "selection_partition": "inner_dev",
        "fit_partition": "inner_train",
        "test_used_for_selection": False,
        "linear_candidates": linear_candidates,
        "acoustic_candidates": acoustic_candidates,
        "descriptor_validation": descriptor_validation,
        "descriptor_profiles": {
            "acoustic55": descriptor_columns,
            "acoustic_no_level_duration": no_level_columns,
            "acoustic_f0_residualized_no_level": f0_residual_columns,
        },
        "f0_residualization": {
            "reference": "natural log of ped_f0_hz_mean",
            "fit_partition": "inner_train separately in every outer fold",
            "direct_f0_columns_removed": sorted(direct_f0_columns),
        },
        "metric_policy": "present-class utility for inner selection; pooled OOF global-class metrics are computed downstream",
        "hyperparameters": vars(args),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
