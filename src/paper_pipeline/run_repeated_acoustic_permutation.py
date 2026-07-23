"""Repeated fold-local acoustic permutation null for the MERT descriptor branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from descriptor_utils import validate_descriptor_columns
from metric_utils import CLASS_COUNTS, TARGETS
from run_aria_leave_one_out_multitask import metric_values, standardize
from run_aria_leave_one_out_pedagogy import train_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--fold-roles", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--layer-selection", required=True)
    parser.add_argument("--pedagogy-features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--model-seed", type=int, default=17)
    parser.add_argument("--permutation-count", type=int, default=20)
    parser.add_argument("--permutation-seed", type=int, default=20260722)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--mert-projection", type=int, default=96)
    parser.add_argument("--pedagogy-projection", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.35)
    return parser.parse_args()


def permute_within_source(
    values: np.ndarray,
    source_ids: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, int]:
    """Permute rows within recording sources, retaining source/repertoire context."""
    output = values.copy()
    movable_rows = 0
    singleton_rows = 0
    for source in np.unique(source_ids):
        indices = np.flatnonzero(source_ids == source)
        if len(indices) <= 1:
            singleton_rows += len(indices)
            continue
        output[indices] = values[indices[rng.permutation(len(indices))]]
        movable_rows += len(indices)
    return output, movable_rows, singleton_rows


def aggregate_prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (permutation, target), frame in predictions.groupby(
        ["permutation", "target"]
    ):
        values = metric_values(
            frame.y_true.to_numpy(dtype=np.int64),
            frame.y_pred.to_numpy(dtype=np.int64),
            CLASS_COUNTS[target],
        )
        rows.append(
            {
                "permutation": int(permutation),
                "target": target,
                **values,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = pd.read_csv(args.metadata).reset_index(drop=True)
    roles = pd.read_csv(args.fold_roles)
    selection = pd.read_csv(args.layer_selection)
    descriptors = pd.read_csv(args.pedagogy_features).drop_duplicates(
        "segment_id", keep="first"
    )
    validation = validate_descriptor_columns(descriptors, args.sample_rate)
    feature_columns = validation["retained_features"]
    aligned = metadata[["segment_id"]].merge(
        descriptors[["segment_id", *feature_columns]],
        on="segment_id",
        how="left",
        validate="one_to_one",
    )
    if aligned[feature_columns].isna().any().any():
        raise ValueError("Missing descriptors after segment alignment")
    descriptor_values = aligned[feature_columns].to_numpy(dtype=np.float32)
    class_counts = [CLASS_COUNTS[target] for target in TARGETS]
    mert_cache: dict[int, np.ndarray] = {}
    prediction_rows = []
    run_rows = []
    permutation_rows = []

    for permutation in range(args.permutation_count):
        for outer_fold in sorted(roles.outer_fold.unique()):
            fold_roles = roles.loc[
                roles.outer_fold == outer_fold, ["segment_id", "role"]
            ]
            fold = metadata.merge(
                fold_roles, on="segment_id", how="inner", validate="one_to_one"
            )
            indices = {
                role: fold.index[fold.role == role].to_numpy()
                for role in ["inner_train", "inner_dev", "outer_test"]
            }
            selected = selection[
                (selection.outer_fold == outer_fold)
                & (selection.encoder == "mert")
                & selection.selected
            ]
            layer = int(selected.iloc[0].layer)
            if layer not in mert_cache:
                mert_cache[layer] = np.load(
                    Path(args.feature_dir) / f"mert_layer_{layer:02d}.npy"
                ).astype(np.float32)
            train_mert, dev_mert, test_mert = standardize(
                mert_cache[layer][indices["inner_train"]],
                mert_cache[layer][indices["inner_dev"]],
                mert_cache[layer][indices["outer_test"]],
            )
            train_desc, dev_desc, test_desc = standardize(
                descriptor_values[indices["inner_train"]],
                descriptor_values[indices["inner_dev"]],
                descriptor_values[indices["outer_test"]],
            )
            fold_seed = (
                args.permutation_seed
                + permutation * 1009
                + int(outer_fold) * 9176
            )
            rng = np.random.default_rng(fold_seed)
            source_ids = {
                role: fold.loc[indices[role], "source_id"].astype(str).to_numpy()
                for role in indices
            }
            permuted_train, train_movable, train_singletons = permute_within_source(
                train_desc, source_ids["inner_train"], rng
            )
            permuted_dev, dev_movable, dev_singletons = permute_within_source(
                dev_desc, source_ids["inner_dev"], rng
            )
            permuted_test, test_movable, test_singletons = permute_within_source(
                test_desc, source_ids["outer_test"], rng
            )
            arrays = {
                "train_mert": train_mert,
                "dev_mert": dev_mert,
                "test_mert": test_mert,
                "train_pedagogy": permuted_train,
                "dev_pedagogy": permuted_dev,
                "test_pedagogy": permuted_test,
            }
            labels = {
                "train": fold.loc[indices["inner_train"], TARGETS].to_numpy(np.int64),
                "dev": fold.loc[indices["inner_dev"], TARGETS].to_numpy(np.int64),
                "test": fold.loc[indices["outer_test"], TARGETS].to_numpy(np.int64),
            }
            _, test_probability, run = train_seed(
                "mert_pedagogy_multitask",
                args.model_seed,
                arrays,
                labels,
                class_counts,
                args,
                device,
            )
            run_rows.append(
                {
                    "permutation": permutation,
                    "outer_fold": int(outer_fold),
                    "permutation_seed": fold_seed,
                    **run,
                }
            )
            permutation_rows.append(
                {
                    "permutation": permutation,
                    "outer_fold": int(outer_fold),
                    "train_movable": train_movable,
                    "train_singletons": train_singletons,
                    "dev_movable": dev_movable,
                    "dev_singletons": dev_singletons,
                    "test_movable": test_movable,
                    "test_singletons": test_singletons,
                }
            )
            test_rows = fold.loc[
                indices["outer_test"], ["segment_id", "source_id"]
            ].reset_index(drop=True)
            for task, target in enumerate(TARGETS):
                truth = labels["test"][:, task]
                prediction = test_probability[task].argmax(axis=1)
                for row_index, probability in enumerate(test_probability[task]):
                    prediction_rows.append(
                        {
                            "permutation": permutation,
                            "outer_fold": int(outer_fold),
                            "segment_id": test_rows.loc[row_index, "segment_id"],
                            "source_id": test_rows.loc[row_index, "source_id"],
                            "target": target,
                            "y_true": int(truth[row_index]),
                            "y_pred": int(prediction[row_index]),
                            "score": json.dumps(probability.tolist()),
                        }
                    )
            pd.DataFrame(prediction_rows).to_csv(
                output / "permutation_predictions_private.partial.csv", index=False
            )
            print(
                f"completed permutation {permutation} fold {outer_fold}", flush=True
            )

    predictions = pd.DataFrame(prediction_rows)
    metrics = aggregate_prediction_metrics(predictions)
    summaries = (
        metrics.groupby("permutation")[["uar", "macro_f1", "accuracy", "utility"]]
        .mean()
        .reset_index()
    )
    predictions.to_csv(output / "permutation_predictions_private.csv", index=False)
    metrics.to_csv(output / "permutation_per_target_metrics.csv", index=False)
    summaries.to_csv(output / "permutation_method_summaries.csv", index=False)
    pd.DataFrame(run_rows).to_csv(output / "permutation_runs.csv", index=False)
    pd.DataFrame(permutation_rows).to_csv(
        output / "permutation_group_audit.csv", index=False
    )
    manifest = {
        "null": "descriptor rows permuted independently in train/dev/test within source_id",
        "repertoire_control": "source_id is nested in held-aria assignment",
        "permutation_count": args.permutation_count,
        "permutation_seed": args.permutation_seed,
        "model_seed": args.model_seed,
        "outer_test_used_for_selection": False,
        "device": str(device),
        "hyperparameters": vars(args),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summaries.describe().to_string(), flush=True)


if __name__ == "__main__":
    main()
