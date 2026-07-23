"""Controlled MERT-head ablations under the strict leave-one-aria-out protocol.

The experiment separates four factors that were confounded in the historical
linear-versus-multi-task comparison: nonlinearity, task sharing, parameter
count, and seed ensembling.  It also retrains the shared model after removing
the mixed Dataset Breathy/Roughness target.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from metric_utils import CLASS_COUNTS, TARGETS, classification_metrics, primary_metric_view


ARCHITECTURES = [
    "shared_linear_lowrank",
    "independent_mlp_matched",
    "independent_mlp_full",
    "shared_mlp_7target",
    "shared_mlp_6target",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--fold-roles", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--layer-selection", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--architectures", default=",".join(ARCHITECTURES))
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--projection-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.35)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-5] = 1.0
    return tuple(((value - mean) / std).astype(np.float32) for value in (train, *others))


def class_weights(
    labels: np.ndarray, target_names: list[str], device: torch.device
) -> list[torch.Tensor]:
    result = []
    for task, target in enumerate(target_names):
        count = CLASS_COUNTS[target]
        frequency = np.bincount(labels[:, task], minlength=count).astype(np.float32)
        weight = np.sqrt(frequency.sum() / np.maximum(frequency, 1.0))
        weight /= weight.mean()
        result.append(torch.tensor(weight, dtype=torch.float32, device=device))
    return result


def metric_values(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> dict:
    explicit = classification_metrics(y_true, y_pred, CLASS_COUNTS[target])
    return {
        **primary_metric_view(explicit),
        "uar_present": float(explicit["uar_present"]),
        "macro_f1_present": float(explicit["macro_f1_present"]),
        "uar_global": float(explicit["uar_global"]),
        "macro_f1_global": float(explicit["macro_f1_global"]),
        "present_class_count": int(explicit["present_class_count"]),
        "global_class_count": int(explicit["global_class_count"]),
    }


class SharedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        target_names: list[str],
        projection_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.shared = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, CLASS_COUNTS[target]) for target in target_names]
        )

    def forward(self, features: torch.Tensor) -> list[torch.Tensor]:
        hidden = self.shared(self.projector(features))
        return [head(hidden) for head in self.heads]


class SharedLinearLowRank(nn.Module):
    def __init__(self, input_dim: int, target_names: list[str], rank: int) -> None:
        super().__init__()
        self.projector = nn.Linear(input_dim, rank)
        self.heads = nn.ModuleList(
            [nn.Linear(rank, CLASS_COUNTS[target]) for target in target_names]
        )

    def forward(self, features: torch.Tensor) -> list[torch.Tensor]:
        shared = self.projector(features)
        return [head(shared) for head in self.heads]


class SingleTaskMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        class_count: int,
        projection_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, class_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class IndependentMLPs(nn.Module):
    def __init__(
        self,
        input_dim: int,
        target_names: list[str],
        projection_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.networks = nn.ModuleList(
            [
                SingleTaskMLP(
                    input_dim,
                    CLASS_COUNTS[target],
                    projection_dim,
                    hidden_dim,
                    dropout,
                )
                for target in target_names
            ]
        )

    def forward(self, features: torch.Tensor) -> list[torch.Tensor]:
        return [network(features) for network in self.networks]


def architecture_targets(name: str) -> list[str]:
    if name == "shared_mlp_6target":
        return [target for target in TARGETS if target != "breathiness"]
    return list(TARGETS)


def build_model(
    architecture: str,
    input_dim: int,
    target_names: list[str],
    args: argparse.Namespace,
) -> nn.Module:
    if architecture.startswith("shared_mlp"):
        return SharedMLP(
            input_dim,
            target_names,
            args.projection_dim,
            args.hidden_dim,
            args.dropout,
        )
    if architecture == "shared_linear_lowrank":
        return SharedLinearLowRank(input_dim, target_names, args.projection_dim)
    if architecture == "independent_mlp_full":
        return IndependentMLPs(
            input_dim,
            target_names,
            args.projection_dim,
            args.hidden_dim,
            args.dropout,
        )
    if architecture == "independent_mlp_matched":
        projection = max(2, args.projection_dim // len(target_names))
        hidden = max(4, args.hidden_dim // len(target_names))
        return IndependentMLPs(
            input_dim,
            target_names,
            projection,
            hidden,
            args.dropout,
        )
    raise ValueError(f"Unknown architecture: {architecture}")


@torch.no_grad()
def predict(
    model: nn.Module, features: np.ndarray, batch_size: int, device: torch.device
) -> list[np.ndarray]:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)), batch_size=batch_size, shuffle=False
    )
    chunks = None
    for (batch,) in loader:
        probabilities = [
            torch.softmax(logits, dim=1).cpu().numpy()
            for logits in model(batch.to(device))
        ]
        if chunks is None:
            chunks = [[] for _ in probabilities]
        for task, probability in enumerate(probabilities):
            chunks[task].append(probability)
    assert chunks is not None
    return [np.concatenate(values, axis=0) for values in chunks]


def mean_utility(
    labels: np.ndarray, probabilities: list[np.ndarray], target_names: list[str]
) -> float:
    values = []
    for task, target in enumerate(target_names):
        score = metric_values(labels[:, task], probabilities[task].argmax(axis=1), target)
        values.append(score["utility"])
    return float(np.mean(values))


def train_seed(
    architecture: str,
    seed: int,
    arrays: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    target_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    seed_everything(seed)
    model = build_model(architecture, arrays["train"].shape[1], target_names, args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    losses = [
        nn.CrossEntropyLoss(weight=weight)
        for weight in class_weights(labels["train"], target_names, device)
    ]
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(arrays["train"]), torch.from_numpy(labels["train"]).long()
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
        for feature_batch, target_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(feature_batch.to(device))
            target_batch = target_batch.to(device)
            loss = torch.stack(
                [losses[task](logits[task], target_batch[:, task]) for task in range(len(target_names))]
            ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        dev_probability = predict(model, arrays["dev"], args.batch_size, device)
        score = mean_utility(labels["dev"], dev_probability, target_names)
        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return (
        predict(model, arrays["dev"], args.batch_size, device),
        predict(model, arrays["test"], args.batch_size, device),
        {
            "seed": seed,
            "best_epoch": best_epoch,
            "inner_dev_utility": best_score,
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        },
    )


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    architectures = [value.strip() for value in args.architectures.split(",") if value.strip()]
    unknown = sorted(set(architectures).difference(ARCHITECTURES))
    if unknown:
        raise ValueError(f"Unknown architectures: {unknown}")
    seeds = [int(value) for value in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = pd.read_csv(args.metadata)
    roles = pd.read_csv(args.fold_roles)
    selection = pd.read_csv(args.layer_selection)
    cache: dict[int, np.ndarray] = {}
    ensemble_metric_rows = []
    ensemble_prediction_rows = []
    seed_metric_rows = []
    seed_prediction_rows = []
    run_rows = []

    for outer_fold in sorted(roles.outer_fold.unique()):
        fold_roles = roles.loc[roles.outer_fold == outer_fold, ["segment_id", "role"]]
        fold = metadata.merge(fold_roles, on="segment_id", how="inner", validate="one_to_one")
        indices = {
            role: fold.index[fold.role == role].to_numpy()
            for role in ["inner_train", "inner_dev", "outer_test"]
        }
        chosen = selection[
            (selection.outer_fold == outer_fold)
            & (selection.encoder == "mert")
            & selection.selected
        ]
        if len(chosen) != 1:
            raise ValueError(f"Fold {outer_fold} has {len(chosen)} selected MERT layers")
        layer = int(chosen.iloc[0].layer)
        if layer not in cache:
            cache[layer] = np.load(Path(args.feature_dir) / f"mert_layer_{layer:02d}.npy").astype(
                np.float32
            )
        features = cache[layer]
        train, dev, test = standardize(
            features[indices["inner_train"]],
            features[indices["inner_dev"]],
            features[indices["outer_test"]],
        )
        arrays = {"train": train, "dev": dev, "test": test}
        test_frame = fold.loc[
            indices["outer_test"], ["segment_id", "source_id"]
        ].reset_index(drop=True)

        for architecture in architectures:
            target_names = architecture_targets(architecture)
            target_columns = [TARGETS.index(target) for target in target_names]
            labels = {
                role: fold.loc[indices[index_role], target_names].to_numpy(np.int64)
                for role, index_role in [
                    ("train", "inner_train"),
                    ("dev", "inner_dev"),
                    ("test", "outer_test"),
                ]
            }
            seed_tests = []
            for seed in seeds:
                _, test_probability, run = train_seed(
                    architecture,
                    seed,
                    arrays,
                    labels,
                    target_names,
                    args,
                    device,
                )
                seed_tests.append(test_probability)
                run_rows.append(
                    {
                        "outer_fold": int(outer_fold),
                        "scheme": architecture,
                        "targets": json.dumps(target_names),
                        "selected_mert_layer": layer,
                        **run,
                    }
                )
                for task, target in enumerate(target_names):
                    y_true = labels["test"][:, task]
                    y_pred = test_probability[task].argmax(axis=1)
                    seed_metric_rows.append(
                        {
                            "outer_fold": int(outer_fold),
                            "scheme": architecture,
                            "seed": seed,
                            "target": target,
                            **metric_values(y_true, y_pred, target),
                        }
                    )
                    for row_index, probability in enumerate(test_probability[task]):
                        seed_prediction_rows.append(
                            {
                                "outer_fold": int(outer_fold),
                                "scheme": architecture,
                                "seed": seed,
                                "segment_id": test_frame.loc[row_index, "segment_id"],
                                "source_id": test_frame.loc[row_index, "source_id"],
                                "target": target,
                                "y_true": int(y_true[row_index]),
                                "y_pred": int(y_pred[row_index]),
                                "probability": json.dumps(probability.tolist()),
                            }
                        )
            ensemble = [
                np.mean([seed_probability[task] for seed_probability in seed_tests], axis=0)
                for task in range(len(target_names))
            ]
            for task, target in enumerate(target_names):
                y_true = labels["test"][:, task]
                y_pred = ensemble[task].argmax(axis=1)
                ensemble_metric_rows.append(
                    {
                        "outer_fold": int(outer_fold),
                        "scheme": architecture,
                        "target": target,
                        **metric_values(y_true, y_pred, target),
                    }
                )
                for row_index, probability in enumerate(ensemble[task]):
                    ensemble_prediction_rows.append(
                        {
                            "outer_fold": int(outer_fold),
                            "scheme": architecture,
                            "segment_id": test_frame.loc[row_index, "segment_id"],
                            "source_id": test_frame.loc[row_index, "source_id"],
                            "target": target,
                            "y_true": int(y_true[row_index]),
                            "y_pred": int(y_pred[row_index]),
                            "probability": json.dumps(probability.tolist()),
                        }
                    )

        pd.DataFrame(run_rows).to_csv(output / "seed_runs.partial.csv", index=False)
        pd.DataFrame(seed_metric_rows).to_csv(output / "seed_metrics.partial.csv", index=False)
        pd.DataFrame(seed_prediction_rows).to_csv(
            output / "seed_predictions_private.partial.csv", index=False
        )
        pd.DataFrame(ensemble_metric_rows).to_csv(
            output / "ensemble_metrics.partial.csv", index=False
        )
        pd.DataFrame(ensemble_prediction_rows).to_csv(
            output / "ensemble_predictions_private.partial.csv", index=False
        )
        print(f"completed controlled-head fold {outer_fold}", flush=True)

    runs = pd.DataFrame(run_rows)
    seed_metrics = pd.DataFrame(seed_metric_rows)
    ensemble_metrics = pd.DataFrame(ensemble_metric_rows)
    runs.to_csv(output / "seed_runs.csv", index=False)
    seed_metrics.to_csv(output / "seed_metrics.csv", index=False)
    pd.DataFrame(seed_prediction_rows).to_csv(
        output / "seed_predictions_private.csv", index=False
    )
    ensemble_metrics.to_csv(output / "ensemble_metrics.csv", index=False)
    pd.DataFrame(ensemble_prediction_rows).to_csv(
        output / "ensemble_predictions_private.csv", index=False
    )
    seed_metrics.groupby(["scheme", "seed"])[
        ["uar_present", "macro_f1_present", "accuracy"]
    ].mean().reset_index().to_csv(output / "seed_summary.csv", index=False)
    ensemble_metrics.groupby("scheme")[[
        "uar_present",
        "macro_f1_present",
        "uar_global",
        "macro_f1_global",
        "accuracy",
    ]].mean().reset_index().to_csv(output / "ensemble_summary.csv", index=False)
    manifest = {
        "protocol": "strict_leave_one_aria_out_controlled_head_ablation",
        "architectures": architectures,
        "seeds": seeds,
        "device": str(device),
        "class_counts": CLASS_COUNTS,
        "metric_policy": "present-class UAR/F1 for model selection; global-class variants also saved",
        "mixed_target_excluded_in": "shared_mlp_6target",
        "outer_test_used_for_selection": False,
        "hyperparameters": vars(args),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print((output / "ensemble_summary.csv").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
