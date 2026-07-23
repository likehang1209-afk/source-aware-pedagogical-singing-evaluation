"""Strict leave-one-aria-out ablation for pedagogy-informed descriptors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_aria_leave_one_out_multitask import (
    TARGETS,
    class_weights,
    mean_utility,
    metric_values,
    seed_everything,
    standardize,
)
from metric_utils import CLASS_COUNTS
from descriptor_utils import validate_descriptor_columns


TEMPERATURES = [0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]


def temperature_scale(probability: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probability, 1e-8, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def negative_log_likelihood(y_true: np.ndarray, probability: np.ndarray) -> float:
    return float(-np.log(np.clip(probability[np.arange(len(y_true)), y_true], 1e-8, 1.0)).mean())


def calibration_metrics(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> dict[str, float]:
    confidence = probability.max(axis=1)
    prediction = probability.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if mask.any():
            ece += mask.mean() * abs((prediction[mask] == y_true[mask]).mean() - confidence[mask].mean())
    one_hot = np.eye(probability.shape[1], dtype=np.float32)[y_true]
    return {
        "nll": negative_log_likelihood(y_true, probability),
        "brier": float(np.mean(np.sum((probability - one_hot) ** 2, axis=1))),
        "ece": float(ece),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--fold-roles", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--layer-selection", required=True)
    parser.add_argument("--pedagogy-features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seeds", default="17,29,43")
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


class PedagogyHead(nn.Module):
    def __init__(
        self,
        mert_dim: int,
        pedagogy_dim: int,
        mode: str,
        args: argparse.Namespace,
        class_counts: list[int],
    ) -> None:
        super().__init__()
        self.mode = mode
        self.mert = None
        if mode in {"mert_multitask", "mert_pedagogy_multitask"}:
            self.mert = nn.Sequential(
                nn.Linear(mert_dim, args.mert_projection),
                nn.LayerNorm(args.mert_projection),
                nn.SiLU(),
                nn.Dropout(args.dropout),
            )
        self.pedagogy = None
        if mode in {"pedagogy_multitask", "mert_pedagogy_multitask"}:
            self.pedagogy = nn.Sequential(
                nn.Linear(pedagogy_dim, args.pedagogy_projection),
                nn.LayerNorm(args.pedagogy_projection),
                nn.SiLU(),
                nn.Dropout(args.dropout),
            )
        input_dim = args.pedagogy_projection if self.pedagogy is not None else 0
        if self.mert is not None:
            input_dim += args.mert_projection
        self.shared = nn.Sequential(
            nn.Linear(input_dim, args.hidden_dim),
            nn.LayerNorm(args.hidden_dim),
            nn.SiLU(),
            nn.Dropout(args.dropout),
        )
        self.heads = nn.ModuleList([nn.Linear(args.hidden_dim, count) for count in class_counts])

    def forward(self, mert: torch.Tensor, pedagogy: torch.Tensor) -> list[torch.Tensor]:
        branches = []
        if self.mert is not None:
            branches.append(self.mert(mert))
        if self.pedagogy is not None:
            branches.append(self.pedagogy(pedagogy))
        hidden = self.shared(torch.cat(branches, dim=1))
        return [head(hidden) for head in self.heads]


@torch.no_grad()
def predict(
    model: nn.Module,
    mert: np.ndarray,
    pedagogy: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(mert), torch.from_numpy(pedagogy)),
        batch_size=batch_size,
        shuffle=False,
    )
    chunks = None
    for mert_batch, pedagogy_batch in loader:
        logits = model(mert_batch.to(device), pedagogy_batch.to(device))
        probabilities = [torch.softmax(value, dim=1).cpu().numpy() for value in logits]
        if chunks is None:
            chunks = [[] for _ in probabilities]
        for task, probability in enumerate(probabilities):
            chunks[task].append(probability)
    assert chunks is not None
    return [np.concatenate(value, axis=0) for value in chunks]


def train_seed(
    mode: str,
    seed: int,
    arrays: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    class_counts: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, float | int]]:
    seed_everything(seed)
    model = PedagogyHead(
        arrays["train_mert"].shape[1],
        arrays["train_pedagogy"].shape[1],
        mode,
        args,
        class_counts,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    losses = [nn.CrossEntropyLoss(weight=w) for w in class_weights(labels["train"], class_counts, device)]
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(arrays["train_mert"]),
            torch.from_numpy(arrays["train_pedagogy"]),
            torch.from_numpy(labels["train"]).long(),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_state = None
    best_score = -np.inf
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for mert_batch, pedagogy_batch, target_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(mert_batch.to(device), pedagogy_batch.to(device))
            target_batch = target_batch.to(device)
            loss = torch.stack([losses[t](logits[t], target_batch[:, t]) for t in range(len(TARGETS))]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        dev = predict(
            model,
            arrays["dev_mert"],
            arrays["dev_pedagogy"],
            args.batch_size,
            device,
        )
        score = mean_utility(labels["dev"], dev)
        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    dev_probability = predict(
        model,
        arrays["dev_mert"],
        arrays["dev_pedagogy"],
        args.batch_size,
        device,
    )
    test_probability = predict(
        model,
        arrays["test_mert"],
        arrays["test_pedagogy"],
        args.batch_size,
        device,
    )
    return dev_probability, test_probability, {
        "seed": seed,
        "best_epoch": best_epoch,
        "inner_dev_utility": best_score,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = pd.read_csv(args.metadata)
    roles = pd.read_csv(args.fold_roles)
    selection = pd.read_csv(args.layer_selection)
    pedagogy = pd.read_csv(args.pedagogy_features).drop_duplicates("segment_id", keep="first")
    descriptor_validation = validate_descriptor_columns(pedagogy, args.sample_rate)
    feature_columns = descriptor_validation["retained_features"]
    no_level_columns = [
        column
        for column in feature_columns
        if column not in {"ped_rms", "ped_duration_sec"}
    ]
    (output / "descriptor_validation.json").write_text(
        json.dumps(descriptor_validation, indent=2), encoding="utf-8"
    )
    aligned = metadata[["segment_id"]].merge(
        pedagogy[["segment_id", *feature_columns]], on="segment_id", how="left", validate="one_to_one"
    )
    if aligned[feature_columns].isna().any().any():
        raise ValueError("Missing pedagogy features after segment alignment")
    pedagogy_all = aligned[feature_columns].to_numpy(np.float32)
    no_level_indices = [feature_columns.index(column) for column in no_level_columns]
    pedagogy_no_level_all = pedagogy_all[:, no_level_indices]
    class_counts = [CLASS_COUNTS[target] for target in TARGETS]
    mert_cache = {}
    metric_rows = []
    prediction_rows = []
    run_rows = []
    seed_metric_rows = []
    seed_prediction_rows = []
    calibration_rows = []
    experiment_specs = [
        ("mert_multitask", "mert_multitask", "real"),
        ("pedagogy_multitask", "pedagogy_multitask", "real"),
        (
            "pedagogy_no_level_duration_multitask",
            "pedagogy_multitask",
            "no_level",
        ),
        ("mert_pedagogy_multitask", "mert_pedagogy_multitask", "real"),
        (
            "mert_pedagogy_no_level_duration_multitask",
            "mert_pedagogy_multitask",
            "no_level",
        ),
        (
            "mert_permuted_pedagogy_multitask",
            "mert_pedagogy_multitask",
            "permuted",
        ),
    ]
    scheme_names = [specification[0] for specification in experiment_specs]
    for outer_fold in sorted(roles.outer_fold.unique()):
        fold_roles = roles.loc[roles.outer_fold == outer_fold, ["segment_id", "role"]]
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
            mert_cache[layer] = np.load(Path(args.feature_dir) / f"mert_layer_{layer:02d}.npy").astype(np.float32)
        mert_all = mert_cache[layer]
        train_mert, dev_mert, test_mert = standardize(
            mert_all[indices["inner_train"]],
            mert_all[indices["inner_dev"]],
            mert_all[indices["outer_test"]],
        )
        train_ped, dev_ped, test_ped = standardize(
            pedagogy_all[indices["inner_train"]],
            pedagogy_all[indices["inner_dev"]],
            pedagogy_all[indices["outer_test"]],
        )
        train_ped_no_level, dev_ped_no_level, test_ped_no_level = standardize(
            pedagogy_no_level_all[indices["inner_train"]],
            pedagogy_no_level_all[indices["inner_dev"]],
            pedagogy_no_level_all[indices["outer_test"]],
        )
        arrays = {
            "train_mert": train_mert,
            "dev_mert": dev_mert,
            "test_mert": test_mert,
            "train_pedagogy": train_ped,
            "dev_pedagogy": dev_ped,
            "test_pedagogy": test_ped,
        }
        no_level_arrays = {
            **arrays,
            "train_pedagogy": train_ped_no_level,
            "dev_pedagogy": dev_ped_no_level,
            "test_pedagogy": test_ped_no_level,
        }
        permutation_rng = np.random.default_rng(20260719 + int(outer_fold))
        permuted_arrays = {
            **arrays,
            "train_pedagogy": train_ped[permutation_rng.permutation(len(train_ped))],
            "dev_pedagogy": dev_ped[permutation_rng.permutation(len(dev_ped))],
            "test_pedagogy": test_ped[permutation_rng.permutation(len(test_ped))],
        }
        labels = {
            "train": fold.loc[indices["inner_train"], TARGETS].to_numpy(np.int64),
            "dev": fold.loc[indices["inner_dev"], TARGETS].to_numpy(np.int64),
            "test": fold.loc[indices["outer_test"], TARGETS].to_numpy(np.int64),
        }
        test_frame = fold.loc[indices["outer_test"], ["segment_id", "source_id"]].reset_index(drop=True)
        fold_dev_probabilities = {}
        fold_test_probabilities = {}
        for scheme, mode, feature_source in experiment_specs:
            if feature_source == "real":
                experiment_arrays = arrays
            elif feature_source == "no_level":
                experiment_arrays = no_level_arrays
            else:
                experiment_arrays = permuted_arrays
            seed_dev_probabilities = []
            seed_test_probabilities = []
            for seed in seeds:
                dev_probabilities, test_probabilities, run = train_seed(
                    mode, seed, experiment_arrays, labels, class_counts, args, device
                )
                seed_dev_probabilities.append(dev_probabilities)
                seed_test_probabilities.append(test_probabilities)
                run_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "scheme": scheme,
                        "feature_source": feature_source,
                        **run,
                    }
                )
                for task, target in enumerate(TARGETS):
                    y_true = labels["test"][:, task]
                    probability = test_probabilities[task]
                    y_pred = probability.argmax(axis=1)
                    seed_metric_rows.append(
                        {
                            "outer_fold": outer_fold,
                            "scheme": scheme,
                            "seed": seed,
                            "target": target,
                            **metric_values(y_true, y_pred, class_counts[task]),
                        }
                    )
                    for row_index, row_probability in enumerate(probability):
                        seed_prediction_rows.append(
                            {
                                "outer_fold": outer_fold,
                                "scheme": scheme,
                                "seed": seed,
                                "segment_id": test_frame.loc[row_index, "segment_id"],
                                "source_id": test_frame.loc[row_index, "source_id"],
                                "target": target,
                                "y_true": int(y_true[row_index]),
                                "y_pred": int(y_pred[row_index]),
                                "probability": json.dumps(row_probability.tolist()),
                            }
                        )
            dev_ensemble = [
                np.mean([run[task] for run in seed_dev_probabilities], axis=0)
                for task in range(len(TARGETS))
            ]
            test_ensemble = [
                np.mean([run[task] for run in seed_test_probabilities], axis=0)
                for task in range(len(TARGETS))
            ]
            fold_dev_probabilities[scheme] = dev_ensemble
            fold_test_probabilities[scheme] = test_ensemble
            for task, target in enumerate(TARGETS):
                y_true = labels["test"][:, task]
                y_pred = test_ensemble[task].argmax(axis=1)
                values = metric_values(y_true, y_pred, class_counts[task])
                metric_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "scheme": scheme,
                        "target": target,
                        **values,
                        "utility": 0.5 * (values["uar"] + values["macro_f1"]),
                    }
                )
                for row_index, probability in enumerate(test_ensemble[task]):
                    prediction_rows.append(
                        {
                            "outer_fold": outer_fold,
                            "scheme": scheme,
                            "segment_id": test_frame.loc[row_index, "segment_id"],
                            "source_id": test_frame.loc[row_index, "source_id"],
                            "target": target,
                            "y_true": int(y_true[row_index]),
                            "y_pred": int(y_pred[row_index]),
                            "probability": json.dumps(probability.tolist()),
                        }
                    )

        # Select the acoustic branch separately for each target using inner-dev
        # utility only. Ties retain the simpler MERT-only expert.
        selected_test_probabilities = []
        selection_rows = []
        for task, target in enumerate(TARGETS):
            candidates = {}
            for mode in [
                "mert_multitask",
                "mert_pedagogy_multitask",
                "mert_pedagogy_no_level_duration_multitask",
            ]:
                dev_prediction = fold_dev_probabilities[mode][task].argmax(axis=1)
                values = metric_values(
                    labels["dev"][:, task], dev_prediction, class_counts[task]
                )
                candidates[mode] = 0.5 * (values["uar"] + values["macro_f1"])
            selected_mode = max(candidates, key=lambda key: (candidates[key], key == "mert_multitask"))
            selected_test_probabilities.append(fold_test_probabilities[selected_mode][task])
            selection_rows.append(
                {
                    "outer_fold": outer_fold,
                    "target": target,
                    "selected_scheme": selected_mode,
                    "mert_inner_utility": candidates["mert_multitask"],
                    "mert_pedagogy_inner_utility": candidates["mert_pedagogy_multitask"],
                    "mert_pedagogy_no_level_duration_inner_utility": candidates[
                        "mert_pedagogy_no_level_duration_multitask"
                    ],
                }
            )
        for task, target in enumerate(TARGETS):
            probability_matrix = selected_test_probabilities[task]
            y_true = labels["test"][:, task]
            y_pred = probability_matrix.argmax(axis=1)
            values = metric_values(y_true, y_pred, class_counts[task])
            metric_rows.append(
                {
                    "outer_fold": outer_fold,
                    "scheme": "inner_selected_pedagogy",
                    "target": target,
                    **values,
                    "utility": 0.5 * (values["uar"] + values["macro_f1"]),
                }
            )
            for row_index, probability in enumerate(probability_matrix):
                prediction_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "scheme": "inner_selected_pedagogy",
                        "segment_id": test_frame.loc[row_index, "segment_id"],
                        "source_id": test_frame.loc[row_index, "source_id"],
                        "target": target,
                        "y_true": int(y_true[row_index]),
                        "y_pred": int(y_pred[row_index]),
                        "probability": json.dumps(probability.tolist()),
                    }
                )
        fold_dev_probabilities["inner_selected_pedagogy"] = [
            fold_dev_probabilities[row["selected_scheme"]][task]
            for task, row in enumerate(selection_rows)
        ]
        fold_test_probabilities["inner_selected_pedagogy"] = selected_test_probabilities

        for scheme in [*scheme_names, "inner_selected_pedagogy"]:
            for task, target in enumerate(TARGETS):
                dev_probability = fold_dev_probabilities[scheme][task]
                dev_y = labels["dev"][:, task]
                temperature = min(
                    TEMPERATURES,
                    key=lambda value: negative_log_likelihood(
                        dev_y, temperature_scale(dev_probability, value)
                    ),
                )
                raw_probability = fold_test_probabilities[scheme][task]
                calibrated_probability = temperature_scale(raw_probability, temperature)
                y_true = labels["test"][:, task]
                raw_values = calibration_metrics(y_true, raw_probability)
                calibrated_values = calibration_metrics(y_true, calibrated_probability)
                calibration_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "scheme": scheme,
                        "target": target,
                        "temperature": temperature,
                        **{f"raw_{key}": value for key, value in raw_values.items()},
                        **{f"calibrated_{key}": value for key, value in calibrated_values.items()},
                    }
                )
                y_pred = calibrated_probability.argmax(axis=1)
                for row_index, probability in enumerate(calibrated_probability):
                    prediction_rows.append(
                        {
                            "outer_fold": outer_fold,
                            "scheme": f"{scheme}_calibrated",
                            "segment_id": test_frame.loc[row_index, "segment_id"],
                            "source_id": test_frame.loc[row_index, "source_id"],
                            "target": target,
                            "y_true": int(y_true[row_index]),
                            "y_pred": int(y_pred[row_index]),
                            "probability": json.dumps(probability.tolist()),
                        }
                    )
        selection_path = output / "inner_target_selection.partial.csv"
        existing_selection = []
        if selection_path.exists():
            existing_selection = pd.read_csv(selection_path).to_dict("records")
            existing_selection = [row for row in existing_selection if row["outer_fold"] != outer_fold]
        pd.DataFrame([*existing_selection, *selection_rows]).to_csv(selection_path, index=False)
        pd.DataFrame(metric_rows).to_csv(output / "per_aria_target_metrics.partial.csv", index=False)
        pd.DataFrame(prediction_rows).to_csv(output / "predictions_private.partial.csv", index=False)
        pd.DataFrame(run_rows).to_csv(output / "seed_runs.partial.csv", index=False)
        pd.DataFrame(seed_metric_rows).to_csv(
            output / "seed_metrics.partial.csv", index=False
        )
        pd.DataFrame(seed_prediction_rows).to_csv(
            output / "seed_predictions_private.partial.csv", index=False
        )
        pd.DataFrame(calibration_rows).to_csv(output / "calibration_metrics.partial.csv", index=False)
        print(f"completed aria fold {outer_fold}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    summary = metrics.groupby("scheme")[["uar", "macro_f1", "accuracy", "utility"]].mean().reset_index()
    metrics.to_csv(output / "per_aria_target_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / "predictions_private.csv", index=False)
    pd.DataFrame(run_rows).to_csv(output / "seed_runs.csv", index=False)
    pd.DataFrame(seed_metric_rows).to_csv(output / "seed_metrics.csv", index=False)
    pd.DataFrame(seed_prediction_rows).to_csv(
        output / "seed_predictions_private.csv", index=False
    )
    pd.DataFrame(calibration_rows).to_csv(output / "calibration_metrics.csv", index=False)
    partial_selection = output / "inner_target_selection.partial.csv"
    if partial_selection.exists():
        pd.read_csv(partial_selection).to_csv(output / "inner_target_selection.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "protocol": "leave_one_aria_out_inner_source_disjoint_pedagogy_ablation",
                "device": str(device),
                "seeds": seeds,
                "class_counts": CLASS_COUNTS,
                "metric_policy": "present-class UAR/F1 for fold selection; global-class variants also saved",
                "pedagogy_feature_count": len(feature_columns),
                "pedagogy_features": feature_columns,
                "pedagogy_no_level_duration_feature_count": len(no_level_columns),
                "pedagogy_no_level_duration_features": no_level_columns,
                "descriptor_validation": descriptor_validation,
                "negative_control": {
                    "scheme": "mert_permuted_pedagogy_multitask",
                    "procedure": "independent deterministic row permutation within inner-train, inner-dev, and outer-test",
                    "permutation_seed_base": 20260719,
                },
                "hyperparameters": vars(args),
                "test_used_for_selection": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
