"""Post-hoc analyses requested in the July 2026 external review.

This script intentionally operates only on locked out-of-fold predictions.  It
does not tune a new model or reconstruct unavailable inner-development scores.
The outputs therefore separate analyses that can be answered from the locked
predictions from experiments that require a new nested run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metric_utils import CLASS_COUNTS, TARGETS, confusion_counts, metrics_from_confusion


SCHEME_LABELS = {
    "mert_multitask": "MERT",
    "mert_multitask_inner_prior_adjusted": "MERT + class-bias adjustment",
    "muq_multitask": "MuQ",
    "muq_multitask_inner_prior_adjusted": "MuQ + class-bias adjustment",
    "mert_muq_multitask": "MERT-MuQ feature concatenation",
    "mert_muq_multitask_inner_prior_adjusted": (
        "MERT-MuQ feature concatenation + class-bias adjustment"
    ),
    "mert_muq_multitask_posterior_0p5": "MERT-MuQ equal posterior fusion",
    "mert_muq_multitask_posterior_0p5_inner_prior_adjusted": (
        "MERT-MuQ equal posterior fusion + class-bias adjustment"
    ),
}

PRIMARY_SCHEMES = [
    "muq_multitask",
    "muq_multitask_inner_prior_adjusted",
    "mert_muq_multitask_posterior_0p5",
    "mert_muq_multitask_posterior_0p5_inner_prior_adjusted",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def discover_project() -> Path:
    candidates = sorted(Path.cwd().glob("IEEE_Access*20260719"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            "Run from the vocal-paper workspace or pass explicit input/output paths"
        )
    return candidates[0]


def parse_scores(value: str | list[float], class_count: int) -> np.ndarray:
    values = json.loads(value) if isinstance(value, str) else value
    score = np.asarray(values, dtype=np.float64)
    if score.shape != (class_count,):
        raise ValueError(f"Expected {class_count} scores, received {score.shape}")
    total = score.sum()
    if not np.isfinite(score).all() or total <= 0:
        raise ValueError("Scores must be finite and have positive mass")
    return score / total


def average_precision(binary_truth: np.ndarray, score: np.ndarray) -> float:
    """Compute non-interpolated average precision without external dependencies."""
    truth = np.asarray(binary_truth, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    positives = int(truth.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    sorted_truth = truth[order]
    cumulative_true = np.cumsum(sorted_truth)
    ranks = np.arange(1, len(sorted_truth) + 1)
    precision_at_rank = cumulative_true / ranks
    return float(precision_at_rank[sorted_truth == 1].sum() / positives)


def aggregate_metrics(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    for target in TARGETS:
        subset = frame[frame.target == target]
        values = metrics_from_confusion(
            confusion_counts(subset.y_true, subset.y_pred, CLASS_COUNTS[target])
        )
        rows.append(
            {
                "target": target,
                "uar": values["uar_global"],
                "macro_f1": values["macro_f1_global"],
                "accuracy": values["accuracy"],
                "utility": values["utility_global"],
            }
        )
    per_target = pd.DataFrame(rows)
    summary = {
        metric: float(per_target[metric].mean())
        for metric in ["uar", "macro_f1", "accuracy", "utility"]
    }
    return per_target, summary


def adjustment_matrix(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_frames = []
    summary_rows = []
    for scheme, label in SCHEME_LABELS.items():
        frame = predictions[predictions.scheme == scheme]
        if frame.empty:
            continue
        per_target, summary = aggregate_metrics(frame)
        per_target.insert(0, "method", label)
        per_target.insert(0, "scheme", scheme)
        target_frames.append(per_target)
        summary_rows.append({"scheme": scheme, "method": label, **summary})
    return pd.DataFrame(summary_rows), pd.concat(target_frames, ignore_index=True)


def class_diagnostics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    confusion_rows = []
    for scheme in PRIMARY_SCHEMES:
        scheme_frame = predictions[predictions.scheme == scheme]
        if scheme_frame.empty:
            continue
        for target in TARGETS:
            frame = scheme_frame[scheme_frame.target == target]
            class_count = CLASS_COUNTS[target]
            truth = frame.y_true.to_numpy(dtype=np.int64)
            prediction = frame.y_pred.to_numpy(dtype=np.int64)
            scores = np.vstack(
                [parse_scores(value, class_count) for value in frame.probability]
            )
            matrix = confusion_counts(truth, prediction, class_count)
            total = matrix.sum()
            for true_class in range(class_count):
                for predicted_class in range(class_count):
                    confusion_rows.append(
                        {
                            "scheme": scheme,
                            "method": SCHEME_LABELS[scheme],
                            "target": target,
                            "true_class": true_class,
                            "predicted_class": predicted_class,
                            "count": int(matrix[true_class, predicted_class]),
                        }
                    )
            for class_index in range(class_count):
                tp = int(matrix[class_index, class_index])
                fn = int(matrix[class_index, :].sum() - tp)
                fp = int(matrix[:, class_index].sum() - tp)
                tn = int(total - tp - fn - fp)
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1 = (
                    2.0 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0
                )
                fpr = fp / (fp + tn) if fp + tn else 0.0
                binary_truth = (truth == class_index).astype(np.int64)
                pr_auc = average_precision(binary_truth, scores[:, class_index])
                rows.append(
                    {
                        "scheme": scheme,
                        "method": SCHEME_LABELS[scheme],
                        "target": target,
                        "class": class_index,
                        "support": int(tp + fn),
                        "predicted_count": int(tp + fp),
                        "true_prevalence": float((tp + fn) / total),
                        "predicted_prevalence": float((tp + fp) / total),
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "false_positive_rate": fpr,
                        "one_vs_rest_pr_auc": float(pr_auc),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(confusion_rows)


def source_confusions(
    frame: pd.DataFrame, clusters: list[tuple[int, str]]
) -> np.ndarray:
    cluster_index = {cluster: index for index, cluster in enumerate(clusters)}
    target_index = {target: index for index, target in enumerate(TARGETS)}
    matrices = np.zeros((len(clusters), len(TARGETS), 4, 4), dtype=np.int64)
    for row in frame.itertuples(index=False):
        cluster = (int(row.outer_fold), str(row.source_id))
        matrices[
            cluster_index[cluster],
            target_index[row.target],
            int(row.y_true),
            int(row.y_pred),
        ] += 1
    return matrices


def summaries_from_matrices(matrices: np.ndarray) -> tuple[dict, list[dict]]:
    per_target = []
    for target_index, target in enumerate(TARGETS):
        count = CLASS_COUNTS[target]
        values = metrics_from_confusion(matrices[target_index, :count, :count])
        per_target.append(
            {
                "target": target,
                "uar": float(values["uar_global"]),
                "macro_f1": float(values["macro_f1_global"]),
                "accuracy": float(values["accuracy"]),
                "utility": float(values["utility_global"]),
            }
        )
    summary = {
        metric: float(np.mean([row[metric] for row in per_target]))
        for metric in ["uar", "macro_f1", "accuracy", "utility"]
    }
    return summary, per_target


def multiplicity_global(
    clusters: list[tuple[int, str]], rng: np.random.Generator
) -> np.ndarray:
    return rng.multinomial(len(clusters), np.full(len(clusters), 1.0 / len(clusters)))


def multiplicity_stratified(
    clusters: list[tuple[int, str]], rng: np.random.Generator
) -> np.ndarray:
    multiplicity = np.zeros(len(clusters), dtype=np.int64)
    folds = np.asarray([fold for fold, _ in clusters], dtype=np.int64)
    for fold in np.unique(folds):
        indices = np.flatnonzero(folds == fold)
        multiplicity[indices] = rng.multinomial(
            len(indices), np.full(len(indices), 1.0 / len(indices))
        )
    return multiplicity


def empirical_two_sided_p(values: np.ndarray) -> float:
    count = len(values)
    lower = (np.count_nonzero(values <= 0) + 1) / (count + 1)
    upper = (np.count_nonzero(values >= 0) + 1) / (count + 1)
    return float(min(1.0, 2.0 * min(lower, upper)))


def adjust_pvalues(values: pd.Series, method: str) -> np.ndarray:
    p = values.to_numpy(dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order]
    count = len(p)
    if method == "bh":
        adjusted_ranked = ranked * count / np.arange(1, count + 1)
        adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    elif method == "holm":
        adjusted_ranked = ranked * (count - np.arange(count))
        adjusted_ranked = np.maximum.accumulate(adjusted_ranked)
    else:
        raise ValueError(method)
    adjusted = np.empty(count, dtype=np.float64)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def paired_bootstrap(
    predictions: pd.DataFrame,
    candidate: str,
    reference: str,
    design: str,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["outer_fold", "segment_id", "source_id", "target", "y_true"]
    left = predictions[predictions.scheme == candidate][keys + ["y_pred"]].rename(
        columns={"y_pred": "candidate_prediction"}
    )
    right = predictions[predictions.scheme == reference][keys + ["y_pred"]].rename(
        columns={"y_pred": "reference_prediction"}
    )
    paired = left.merge(right, on=keys, validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise ValueError("Candidate and reference predictions are not fully aligned")
    clusters = sorted(
        {(int(row.outer_fold), str(row.source_id)) for row in paired.itertuples()}
    )
    candidate_matrices = source_confusions(
        paired.rename(columns={"candidate_prediction": "y_pred"}), clusters
    )
    reference_matrices = source_confusions(
        paired.rename(columns={"reference_prediction": "y_pred"}), clusters
    )
    candidate_point, candidate_targets = summaries_from_matrices(
        candidate_matrices.sum(axis=0)
    )
    reference_point, reference_targets = summaries_from_matrices(
        reference_matrices.sum(axis=0)
    )
    rng = np.random.default_rng(seed)
    overall_values = {metric: [] for metric in ["uar", "macro_f1", "accuracy", "utility"]}
    target_values = {
        target: {metric: [] for metric in overall_values} for target in TARGETS
    }
    replicate_rows = []
    for replicate in range(replicates):
        if design == "global_source_cluster":
            multiplicity = multiplicity_global(clusters, rng)
        elif design == "aria_stratified_source_cluster":
            multiplicity = multiplicity_stratified(clusters, rng)
        else:
            raise ValueError(design)
        candidate_sum = np.tensordot(multiplicity, candidate_matrices, axes=(0, 0))
        reference_sum = np.tensordot(multiplicity, reference_matrices, axes=(0, 0))
        candidate_summary, candidate_target = summaries_from_matrices(candidate_sum)
        reference_summary, reference_target = summaries_from_matrices(reference_sum)
        row = {"design": design, "replicate": replicate}
        for metric in overall_values:
            difference = candidate_summary[metric] - reference_summary[metric]
            overall_values[metric].append(difference)
            row[f"difference_{metric}"] = difference
        for index, target in enumerate(TARGETS):
            for metric in overall_values:
                target_values[target][metric].append(
                    candidate_target[index][metric] - reference_target[index][metric]
                )
        replicate_rows.append(row)
    overall_rows = []
    for metric, raw_values in overall_values.items():
        values = np.asarray(raw_values)
        overall_rows.append(
            {
                "design": design,
                "candidate": candidate,
                "reference": reference,
                "metric": metric,
                "candidate_point": candidate_point[metric],
                "reference_point": reference_point[metric],
                "difference_point": candidate_point[metric] - reference_point[metric],
                "ci_2_5": float(np.quantile(values, 0.025)),
                "ci_97_5": float(np.quantile(values, 0.975)),
                "p_two_sided_empirical": empirical_two_sided_p(values),
                "probability_difference_gt_zero": float(np.mean(values > 0)),
                "source_clusters": len(clusters),
                "paired_decisions": len(paired),
            }
        )
    per_target_rows = []
    for index, target in enumerate(TARGETS):
        for metric in overall_values:
            values = np.asarray(target_values[target][metric])
            per_target_rows.append(
                {
                    "design": design,
                    "candidate": candidate,
                    "reference": reference,
                    "target": target,
                    "metric": metric,
                    "candidate_point": candidate_targets[index][metric],
                    "reference_point": reference_targets[index][metric],
                    "difference_point": (
                        candidate_targets[index][metric]
                        - reference_targets[index][metric]
                    ),
                    "ci_2_5": float(np.quantile(values, 0.025)),
                    "ci_97_5": float(np.quantile(values, 0.975)),
                    "p_two_sided_empirical": empirical_two_sided_p(values),
                }
            )
    per_target = pd.DataFrame(per_target_rows)
    per_target["p_bh_within_metric"] = per_target.groupby("metric", group_keys=False)[
        "p_two_sided_empirical"
    ].transform(lambda series: adjust_pvalues(series, "bh"))
    per_target["p_holm_within_metric"] = per_target.groupby("metric", group_keys=False)[
        "p_two_sided_empirical"
    ].transform(lambda series: adjust_pvalues(series, "holm"))
    return pd.DataFrame(overall_rows), per_target, pd.DataFrame(replicate_rows)


def no_information_baselines(
    metadata: pd.DataFrame,
    roles: pd.DataFrame,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build fold-local majority and stochastic null predictions.

    Class frequencies are estimated only from the non-test portion of each
    outer fold.  Random baselines are summarized over reproducible repeated
    draws instead of reporting a favorable single draw.
    """
    metadata = metadata[["segment_id", "source_id", *TARGETS]].copy()
    metadata["segment_id"] = metadata.segment_id.astype(str)
    metadata["source_id"] = metadata.source_id.astype(str)
    roles = roles.copy()
    roles["segment_id"] = roles.segment_id.astype(str)
    roles["source_id"] = roles.source_id.astype(str)
    fold_data = []
    for fold in sorted(roles.outer_fold.unique()):
        fold_roles = roles[roles.outer_fold == fold]
        joined = fold_roles.merge(
            metadata, on=["segment_id", "source_id"], validate="one_to_one"
        )
        train = joined[joined.role != "outer_test"]
        test = joined[joined.role == "outer_test"]
        if train.empty or test.empty:
            raise ValueError(f"Fold {fold} lacks train or outer-test observations")
        fold_data.append((int(fold), train, test))

    majority_rows = []
    for fold, train, test in fold_data:
        for target in TARGETS:
            counts = (
                train[target]
                .value_counts()
                .reindex(range(CLASS_COUNTS[target]), fill_value=0)
            )
            majority = int(counts.idxmax())
            majority_rows.append(
                pd.DataFrame(
                    {
                        "outer_fold": fold,
                        "target": target,
                        "y_true": test[target].to_numpy(dtype=np.int64),
                        "y_pred": majority,
                    }
                )
            )
    majority_frame = pd.concat(majority_rows, ignore_index=True)
    _, majority_summary = aggregate_metrics(majority_frame)
    replicate_summaries = [
        {"method": "fold-local majority", "replicate": 0, **majority_summary}
    ]

    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        generated = {"uniform random": [], "fold-local stratified random": []}
        for fold, train, test in fold_data:
            for target in TARGETS:
                count = CLASS_COUNTS[target]
                size = len(test)
                truth = test[target].to_numpy(dtype=np.int64)
                train_counts = (
                    train[target]
                    .value_counts()
                    .reindex(range(count), fill_value=0)
                    .to_numpy(dtype=np.float64)
                )
                distribution = train_counts / train_counts.sum()
                generated["uniform random"].append(
                    pd.DataFrame(
                        {
                            "outer_fold": fold,
                            "target": target,
                            "y_true": truth,
                            "y_pred": rng.integers(0, count, size=size),
                        }
                    )
                )
                generated["fold-local stratified random"].append(
                    pd.DataFrame(
                        {
                            "outer_fold": fold,
                            "target": target,
                            "y_true": truth,
                            "y_pred": rng.choice(count, size=size, p=distribution),
                        }
                    )
                )
        for method, frames in generated.items():
            _, summary = aggregate_metrics(pd.concat(frames, ignore_index=True))
            replicate_summaries.append(
                {"method": method, "replicate": replicate, **summary}
            )
    replicate_frame = pd.DataFrame(replicate_summaries)
    rows = []
    for method, frame in replicate_frame.groupby("method"):
        for metric in ["uar", "macro_f1", "accuracy", "utility"]:
            values = frame[metric].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "q_2_5": float(np.quantile(values, 0.025)),
                    "q_97_5": float(np.quantile(values, 0.975)),
                    "replicates": len(values),
                }
            )
    return pd.DataFrame(rows), replicate_frame


def main() -> None:
    args = parse_args()
    project = discover_project()
    default_root = (
        project
        / "review_audit"
        / "gpu_pipeline_final_20260719"
        / "gpu_pipeline_v1"
    )
    prediction_path = args.predictions or (
        default_root / "02c_prior_adjusted_multitask" / "predictions_private.csv"
    )
    output = args.output or (
        project / "review_audit" / "new_reviewer_analysis_20260722"
    )
    output.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(prediction_path)
    predictions["scheme"] = predictions.scheme.astype(str)
    predictions["source_id"] = predictions.source_id.astype(str)

    summary, per_target = adjustment_matrix(predictions)
    summary.to_csv(output / "class_bias_adjustment_method_matrix.csv", index=False)
    per_target.to_csv(output / "class_bias_adjustment_per_target.csv", index=False)

    class_metrics, confusions = class_diagnostics(predictions)
    class_metrics.to_csv(output / "minority_class_diagnostics.csv", index=False)
    confusions.to_csv(output / "confusion_matrices_long.csv", index=False)

    candidate = "mert_muq_multitask_posterior_0p5_inner_prior_adjusted"
    reference = "muq_multitask_inner_prior_adjusted"
    overall_frames = []
    target_frames = []
    replicate_frames = []
    for index, design in enumerate(
        ["global_source_cluster", "aria_stratified_source_cluster"]
    ):
        overall, targets, replicate_frame = paired_bootstrap(
            predictions,
            candidate,
            reference,
            design,
            args.replicates,
            args.seed + index,
        )
        overall_frames.append(overall)
        target_frames.append(targets)
        replicate_frames.append(replicate_frame)
    bootstrap_overall = pd.concat(overall_frames, ignore_index=True)
    bootstrap_targets = pd.concat(target_frames, ignore_index=True)
    bootstrap_replicates = pd.concat(replicate_frames, ignore_index=True)
    bootstrap_overall.to_csv(
        output / "adjusted_fusion_vs_adjusted_muq_bootstrap.csv", index=False
    )
    bootstrap_targets.to_csv(
        output / "adjusted_fusion_vs_adjusted_muq_per_target_fdr.csv", index=False
    )
    bootstrap_replicates.to_csv(
        output / "adjusted_fusion_vs_adjusted_muq_bootstrap_replicates.csv", index=False
    )

    adjustment_comparisons = [
        ("mert_multitask_inner_prior_adjusted", "mert_multitask"),
        ("muq_multitask_inner_prior_adjusted", "muq_multitask"),
        (
            "mert_muq_multitask_inner_prior_adjusted",
            "mert_muq_multitask",
        ),
        (
            "mert_muq_multitask_posterior_0p5_inner_prior_adjusted",
            "mert_muq_multitask_posterior_0p5",
        ),
    ]
    effect_overall = []
    effect_targets = []
    for comparison_index, (effect_candidate, effect_reference) in enumerate(
        adjustment_comparisons
    ):
        for design_index, design in enumerate(
            ["global_source_cluster", "aria_stratified_source_cluster"]
        ):
            effect_summary, effect_per_target, _ = paired_bootstrap(
                predictions,
                effect_candidate,
                effect_reference,
                design,
                args.replicates,
                args.seed + 1000 + comparison_index * 10 + design_index,
            )
            effect_overall.append(effect_summary)
            effect_targets.append(effect_per_target)
    pd.concat(effect_overall, ignore_index=True).to_csv(
        output / "class_bias_adjustment_effect_bootstraps.csv", index=False
    )
    pd.concat(effect_targets, ignore_index=True).to_csv(
        output / "class_bias_adjustment_effect_per_target_fdr.csv", index=False
    )

    selection_path = (
        default_root
        / "02c_prior_adjusted_multitask"
        / "prior_adjustment_selection.csv"
    )
    selection = pd.read_csv(selection_path)
    selection.to_csv(output / "selected_class_bias_tau_by_fold_target.csv", index=False)
    stability = (
        selection.groupby(["source_scheme", "target", "selected_tau"])
        .size()
        .rename("selection_count")
        .reset_index()
    )
    stability.to_csv(output / "selected_tau_frequencies.csv", index=False)

    metadata_path = Path.cwd() / "cloud_backup_20260718" / "aria_loao" / "metadata_private.csv"
    roles_path = (
        Path.cwd()
        / "cloud_backup_20260718"
        / "aria_loao"
        / "aria_nested_split_roles_private.csv"
    )
    baseline_summary, baseline_replicates = no_information_baselines(
        pd.read_csv(metadata_path),
        pd.read_csv(roles_path),
        replicates=200,
        seed=args.seed + 100,
    )
    baseline_summary.to_csv(output / "no_information_baselines.csv", index=False)
    baseline_replicates.to_csv(
        output / "no_information_baseline_replicates.csv", index=False
    )

    manifest = {
        "prediction_source": str(prediction_path),
        "created_from_locked_oof_predictions": True,
        "new_model_training": False,
        "replicates_per_bootstrap_design": args.replicates,
        "random_seed": args.seed,
        "no_information_random_replicates": 200,
        "probability_semantics": (
            "Adjusted rows contain normalized class-adjusted decision scores; "
            "they are not described as calibrated posterior probabilities."
        ),
        "unresolved_without_nested_rerun": [
            "BEATs class-bias adjustment selected on inner development data",
            "utility-weight sensitivity and Pareto analysis",
            "repeated grouped inner-split selection stability",
            "training-loss controls such as unweighted CE and Balanced Softmax",
            "multi-seed fold-local acoustic permutation controls",
            "full label-permutation null pipeline",
        ],
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print("\nAdjusted fusion versus adjusted MuQ")
    print(bootstrap_overall.to_string(index=False))


if __name__ == "__main__":
    main()
