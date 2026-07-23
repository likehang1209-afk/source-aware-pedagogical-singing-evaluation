"""Select MERT and MuQ layers with a multi-task inner-development criterion.

One fixed seed is used for layer selection.  The resulting layer table is then
consumed by the full three-seed experiments.  Candidate test results are saved
only as a descriptive sensitivity analysis and never enter selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from metric_utils import CLASS_COUNTS, TARGETS
from run_aria_leave_one_out_multitask import (
    mean_utility,
    metric_values,
    standardize,
    train_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--fold-roles", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--encoders", default="mert,muq")
    parser.add_argument("--layers", default="3,6,9,12")
    parser.add_argument("--selection-seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--projection-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--inner-repeat", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    encoders = [value.strip() for value in args.encoders.split(",") if value.strip()]
    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    unknown = sorted(set(encoders).difference({"mert", "muq"}))
    if unknown:
        raise ValueError(f"Unsupported encoders: {unknown}")

    metadata = pd.read_csv(args.metadata)
    roles = pd.read_csv(args.fold_roles)
    if "inner_repeat" in roles.columns:
        if args.inner_repeat is None:
            raise ValueError("--inner-repeat is required for repeated role files")
        roles = roles[roles.inner_repeat == args.inner_repeat].copy()
        if roles.empty:
            raise ValueError(f"No rows found for inner repeat {args.inner_repeat}")
    labels_all = metadata[TARGETS].to_numpy(np.int64)
    class_counts = [CLASS_COUNTS[target] for target in TARGETS]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache: dict[tuple[str, int], np.ndarray] = {}
    candidate_rows = []
    metric_rows = []
    prediction_rows = []
    selection_rows = []

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
        labels = {
            role: fold.loc[indices[index_role], TARGETS].to_numpy(np.int64)
            for role, index_role in [
                ("train", "inner_train"),
                ("dev", "inner_dev"),
                ("test", "outer_test"),
            ]
        }
        test_frame = fold.loc[
            indices["outer_test"], ["segment_id", "source_id"]
        ].reset_index(drop=True)

        for encoder in encoders:
            layer_runs = []
            for layer in layers:
                key = (encoder, layer)
                if key not in cache:
                    cache[key] = np.load(
                        Path(args.feature_dir) / f"{encoder}_layer_{layer:02d}.npy",
                        mmap_mode="r",
                    )
                values = cache[key]
                train, dev, test = standardize(
                    values[indices["inner_train"]],
                    values[indices["inner_dev"]],
                    values[indices["outer_test"]],
                )
                # train_seed expects both branches, but the unused branch is
                # never instantiated in an encoder-specific mode.
                dummy = {
                    "train": np.zeros((len(train), 1), dtype=np.float32),
                    "dev": np.zeros((len(dev), 1), dtype=np.float32),
                    "test": np.zeros((len(test), 1), dtype=np.float32),
                }
                if encoder == "mert":
                    arrays = {
                        "train_mert": train,
                        "dev_mert": dev,
                        "test_mert": test,
                        "train_muq": dummy["train"],
                        "dev_muq": dummy["dev"],
                        "test_muq": dummy["test"],
                    }
                    mode = "mert_multitask"
                else:
                    arrays = {
                        "train_mert": dummy["train"],
                        "dev_mert": dummy["dev"],
                        "test_mert": dummy["test"],
                        "train_muq": train,
                        "dev_muq": dev,
                        "test_muq": test,
                    }
                    mode = "muq_multitask"
                dev_probability, test_probability, run = train_seed(
                    mode,
                    args.selection_seed,
                    arrays,
                    labels,
                    class_counts,
                    args,
                    device,
                )
                dev_utility = mean_utility(labels["dev"], dev_probability)
                layer_runs.append((dev_utility, layer, dev_probability, test_probability, run))
                candidate_rows.append(
                    {
                        "outer_fold": int(outer_fold),
                        "encoder": encoder,
                        "layer": layer,
                        "selection_seed": args.selection_seed,
                        "inner_utility": dev_utility,
                        **run,
                    }
                )
                for task, target in enumerate(TARGETS):
                    y_true = labels["test"][:, task]
                    probability = test_probability[task]
                    y_pred = probability.argmax(axis=1)
                    metric_rows.append(
                        {
                            "outer_fold": int(outer_fold),
                            "encoder": encoder,
                            "layer": layer,
                            "target": target,
                            **metric_values(y_true, y_pred, class_counts[task]),
                        }
                    )
                    for row_index, row_probability in enumerate(probability):
                        prediction_rows.append(
                            {
                                "outer_fold": int(outer_fold),
                                "encoder": encoder,
                                "layer": layer,
                                "segment_id": test_frame.loc[row_index, "segment_id"],
                                "source_id": test_frame.loc[row_index, "source_id"],
                                "target": target,
                                "y_true": int(y_true[row_index]),
                                "y_pred": int(y_pred[row_index]),
                                "probability": json.dumps(row_probability.tolist()),
                            }
                        )
            selected = max(layer_runs, key=lambda item: (item[0], -item[1]))
            for dev_utility, layer, _, _, _ in layer_runs:
                selection_rows.append(
                    {
                        "outer_fold": int(outer_fold),
                        "encoder": encoder,
                        "layer": layer,
                        "inner_utility": dev_utility,
                        "selected": layer == selected[1],
                        "selection_model": f"{encoder}_multitask",
                        "selection_seed": args.selection_seed,
                    }
                )
        pd.DataFrame(candidate_rows).to_csv(
            output / "candidate_runs.partial.csv", index=False
        )
        pd.DataFrame(metric_rows).to_csv(
            output / "candidate_test_metrics.partial.csv", index=False
        )
        pd.DataFrame(prediction_rows).to_csv(
            output / "candidate_test_predictions_private.partial.csv", index=False
        )
        pd.DataFrame(selection_rows).to_csv(
            output / "layer_selection.partial.csv", index=False
        )
        print(f"completed multi-task layer-selection fold {outer_fold}", flush=True)

    pd.DataFrame(candidate_rows).to_csv(output / "candidate_runs.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output / "candidate_test_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        output / "candidate_test_predictions_private.csv", index=False
    )
    pd.DataFrame(selection_rows).to_csv(output / "layer_selection.csv", index=False)
    manifest = {
        "protocol": "inner-development multi-task layer selection under strict leave-one-aria-out evaluation",
        "selection_seed": args.selection_seed,
        "encoders": encoders,
        "candidate_layers": layers,
        "selection_partition": "inner_dev",
        "outer_test_used_for_selection": False,
        "candidate_test_results_role": "descriptive layer sensitivity only",
        "device": str(device),
        "hyperparameters": vars(args),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        pd.DataFrame(selection_rows)
        .loc[lambda frame: frame.selected]
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
