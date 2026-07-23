"""Leave-one-aria-out evaluation of compact multi-task foundation-feature heads.

Layer choices are inherited from inner-development-only linear selection. Model
selection and early stopping use the inner development partition; the held aria
is evaluated once after seed ensembling.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from metric_utils import (
    CLASS_COUNTS,
    TARGETS,
    classification_metrics,
    primary_metric_view,
)

TEMPERATURES = [0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
FUSION_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
PRIOR_ADJUSTMENT_TAUS = [0.0, 0.25, 0.5, 0.75, 1.0]
UTILITY_UAR_WEIGHTS = [1.0, 0.75, 0.5, 0.25, 0.0]


def temperature_scale(probability: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probability, 1e-8, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def select_temperature(y_true: np.ndarray, probability: np.ndarray) -> float:
    candidates = []
    for temperature in TEMPERATURES:
        scaled = temperature_scale(probability, temperature)
        nll = -np.log(np.clip(scaled[np.arange(len(y_true)), y_true], 1e-8, 1.0)).mean()
        candidates.append((float(nll), abs(temperature - 1.0), temperature))
    return float(min(candidates)[2])


def adjust_for_training_prior(
    probability: np.ndarray,
    train_labels: np.ndarray,
    class_count: int,
    tau: float,
) -> np.ndarray:
    """Apply Laplace-smoothed prior correction without test-label access."""
    counts = np.bincount(train_labels, minlength=class_count).astype(np.float64) + 1.0
    prior = counts / counts.sum()
    adjusted = probability / np.power(prior.reshape(1, -1), tau)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--fold-roles", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--layer-selection", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--projection-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument(
        "--loss-mode",
        choices=["sqrt_weighted_ce", "unweighted_ce", "balanced_softmax"],
        default="sqrt_weighted_ce",
    )
    parser.add_argument("--selection-uar-weight", type=float, default=0.5)
    parser.add_argument(
        "--inner-repeat",
        type=int,
        help="Select one repeated inner split when fold-role input contains inner_repeat",
    )
    parser.add_argument(
        "--modes",
        default="mert_multitask,muq_multitask,mert_muq_multitask",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CompactMultiTaskHead(nn.Module):
    def __init__(
        self,
        mert_dim: int,
        muq_dim: int,
        mode: str,
        projection_dim: int,
        hidden_dim: int,
        dropout: float,
        class_counts: list[int],
    ) -> None:
        super().__init__()
        self.mode = mode
        self.mert_projector = None
        self.muq_projector = None
        branch_count = 0
        if mode in {"mert_multitask", "mert_muq_multitask"}:
            self.mert_projector = nn.Sequential(
                nn.Linear(mert_dim, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            branch_count += 1
        if mode in {"muq_multitask", "mert_muq_multitask"}:
            self.muq_projector = nn.Sequential(
                nn.Linear(muq_dim, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            branch_count += 1
        self.shared = nn.Sequential(
            nn.Linear(branch_count * projection_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, count) for count in class_counts])

    def forward(self, mert: torch.Tensor, muq: torch.Tensor) -> list[torch.Tensor]:
        branches = []
        if self.mert_projector is not None:
            branches.append(self.mert_projector(mert))
        if self.muq_projector is not None:
            branches.append(self.muq_projector(muq))
        shared = self.shared(torch.cat(branches, dim=1))
        return [head(shared) for head in self.heads]


def metric_values(
    y_true: np.ndarray, y_pred: np.ndarray, class_count: int
) -> dict[str, float | int]:
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


def weighted_selection_score(
    values: dict[str, float | int], uar_weight: float
) -> float:
    return float(
        uar_weight * float(values["uar"])
        + (1.0 - uar_weight) * float(values["macro_f1"])
    )


def mean_utility(
    y_true: np.ndarray,
    probabilities: list[np.ndarray],
    uar_weight: float = 0.5,
) -> float:
    values = []
    for task, probability in enumerate(probabilities):
        result = metric_values(
            y_true[:, task],
            probability.argmax(axis=1),
            CLASS_COUNTS[TARGETS[task]],
        )
        values.append(weighted_selection_score(result, uar_weight))
    return float(np.mean(values))


@torch.no_grad()
def predict(
    model: nn.Module,
    mert: np.ndarray,
    muq: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(mert), torch.from_numpy(muq)),
        batch_size=batch_size,
        shuffle=False,
    )
    chunks: list[list[np.ndarray]] | None = None
    for mert_batch, muq_batch in loader:
        logits = model(mert_batch.to(device), muq_batch.to(device))
        batch_probabilities = [torch.softmax(value, dim=1).cpu().numpy() for value in logits]
        if chunks is None:
            chunks = [[] for _ in batch_probabilities]
        for task, probability in enumerate(batch_probabilities):
            chunks[task].append(probability)
    assert chunks is not None
    return [np.concatenate(task_chunks, axis=0) for task_chunks in chunks]


def standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-5] = 1.0
    return tuple(((array - mean) / std).astype(np.float32) for array in (train, *others))


def class_weights(labels: np.ndarray, class_counts: list[int], device: torch.device) -> list[torch.Tensor]:
    weights = []
    for task, count in enumerate(class_counts):
        frequencies = np.bincount(labels[:, task], minlength=count).astype(np.float32)
        value = np.sqrt(frequencies.sum() / np.maximum(frequencies, 1.0))
        value /= value.mean()
        weights.append(torch.tensor(value, dtype=torch.float32, device=device))
    return weights


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
    model = CompactMultiTaskHead(
        arrays["train_mert"].shape[1],
        arrays["train_muq"].shape[1],
        mode,
        args.projection_dim,
        args.hidden_dim,
        args.dropout,
        class_counts,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_mode = getattr(args, "loss_mode", "sqrt_weighted_ce")
    weighted_losses = [
        nn.CrossEntropyLoss(weight=w)
        for w in class_weights(labels["train"], class_counts, device)
    ]
    log_class_counts = []
    for task, count in enumerate(class_counts):
        frequencies = np.bincount(labels["train"][:, task], minlength=count).astype(
            np.float32
        )
        log_class_counts.append(
            torch.log(torch.tensor(np.maximum(frequencies, 1.0), device=device))
        )
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(arrays["train_mert"]),
            torch.from_numpy(arrays["train_muq"]),
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
        for mert_batch, muq_batch, target_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(mert_batch.to(device), muq_batch.to(device))
            target_batch = target_batch.to(device)
            task_losses = []
            for task in range(len(TARGETS)):
                if loss_mode == "sqrt_weighted_ce":
                    value = weighted_losses[task](logits[task], target_batch[:, task])
                elif loss_mode == "unweighted_ce":
                    value = F.cross_entropy(logits[task], target_batch[:, task])
                elif loss_mode == "balanced_softmax":
                    value = F.cross_entropy(
                        logits[task] + log_class_counts[task].reshape(1, -1),
                        target_batch[:, task],
                    )
                else:
                    raise ValueError(loss_mode)
                task_losses.append(value)
            loss = torch.stack(task_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        dev_probability = predict(
            model,
            arrays["dev_mert"],
            arrays["dev_muq"],
            args.batch_size,
            device,
        )
        score = mean_utility(
            labels["dev"],
            dev_probability,
            getattr(args, "selection_uar_weight", 0.5),
        )
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
        arrays["dev_muq"],
        args.batch_size,
        device,
    )
    test_probability = predict(
        model,
        arrays["test_mert"],
        arrays["test_muq"],
        args.batch_size,
        device,
    )
    metadata = {
        "seed": seed,
        "best_epoch": best_epoch,
        "inner_dev_utility": best_score,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "loss_mode": loss_mode,
        "selection_uar_weight": float(getattr(args, "selection_uar_weight", 0.5)),
    }
    return dev_probability, test_probability, metadata


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = pd.read_csv(args.metadata)
    roles = pd.read_csv(args.fold_roles)
    if "inner_repeat" in roles.columns:
        if args.inner_repeat is None:
            raise ValueError("--inner-repeat is required for repeated role files")
        roles = roles[roles.inner_repeat == args.inner_repeat].copy()
        if roles.empty:
            raise ValueError(f"No rows found for inner repeat {args.inner_repeat}")
    selection = pd.read_csv(args.layer_selection)
    class_counts = [CLASS_COUNTS[target] for target in TARGETS]
    feature_cache: dict[tuple[str, int], np.ndarray] = {}

    def feature(encoder: str, layer: int) -> np.ndarray:
        key = (encoder, layer)
        if key not in feature_cache:
            path = Path(args.feature_dir) / f"{encoder}_layer_{layer:02d}.npy"
            feature_cache[key] = np.load(path).astype(np.float32)
        return feature_cache[key]

    metric_rows = []
    prediction_rows = []
    run_rows = []
    seed_metric_rows = []
    seed_prediction_rows = []
    fusion_rows = []
    prior_adjustment_rows = []
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    valid_modes = {"mert_multitask", "muq_multitask", "mert_muq_multitask"}
    unknown_modes = sorted(set(modes).difference(valid_modes))
    if unknown_modes:
        raise ValueError(f"Unsupported modes: {unknown_modes}")
    if not {"mert_multitask", "muq_multitask"}.issubset(modes):
        raise ValueError(
            "Current fusion-sensitivity output requires both mert_multitask and "
            "muq_multitask; omit only mert_muq_multitask when reducing runtime"
        )
    for outer_fold in sorted(roles["outer_fold"].unique()):
        fold_roles = roles.loc[roles.outer_fold == outer_fold, ["segment_id", "role"]]
        fold = metadata.merge(
            fold_roles, on="segment_id", how="inner", validate="one_to_one"
        )
        indices = {
            role: fold.index[fold.role == role].to_numpy()
            for role in ["inner_train", "inner_dev", "outer_test"]
        }
        selected = selection[(selection.outer_fold == outer_fold) & selection.selected]
        layers = {row.encoder: int(row.layer) for row in selected.itertuples()}
        mert_all = feature("mert", layers["mert"])
        muq_all = feature("muq", layers["muq"])
        # Metadata was used to create every feature array, so rows have identical order.
        train_mert, dev_mert, test_mert = standardize(
            mert_all[indices["inner_train"]],
            mert_all[indices["inner_dev"]],
            mert_all[indices["outer_test"]],
        )
        train_muq, dev_muq, test_muq = standardize(
            muq_all[indices["inner_train"]],
            muq_all[indices["inner_dev"]],
            muq_all[indices["outer_test"]],
        )
        arrays = {
            "train_mert": train_mert,
            "dev_mert": dev_mert,
            "test_mert": test_mert,
            "train_muq": train_muq,
            "dev_muq": dev_muq,
            "test_muq": test_muq,
        }
        labels = {
            "train": fold.loc[indices["inner_train"], TARGETS].to_numpy(np.int64),
            "dev": fold.loc[indices["inner_dev"], TARGETS].to_numpy(np.int64),
            "test": fold.loc[indices["outer_test"], TARGETS].to_numpy(np.int64),
        }
        test_frame = fold.loc[indices["outer_test"], ["segment_id", "source_id"]].reset_index(drop=True)
        fold_dev_ensembles = {}
        fold_test_ensembles = {}
        for mode in modes:
            seed_dev_probabilities = []
            seed_test_probabilities = []
            for seed in seeds:
                dev_probabilities, test_probabilities, run_metadata = train_seed(
                    mode, seed, arrays, labels, class_counts, args, device
                )
                seed_dev_probabilities.append(dev_probabilities)
                seed_test_probabilities.append(test_probabilities)
                run_rows.append({"outer_fold": outer_fold, "scheme": mode, **run_metadata})
                for task, target in enumerate(TARGETS):
                    y_true = labels["test"][:, task]
                    y_pred = test_probabilities[task].argmax(axis=1)
                    seed_metric_rows.append(
                        {
                            "outer_fold": outer_fold,
                            "scheme": mode,
                            "seed": seed,
                            "target": target,
                            **metric_values(y_true, y_pred, class_counts[task]),
                        }
                    )
                    for row_index, probability in enumerate(test_probabilities[task]):
                        seed_prediction_rows.append(
                            {
                                "outer_fold": outer_fold,
                                "scheme": mode,
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
                np.mean([run[task] for run in seed_test_probabilities], axis=0)
                for task in range(len(TARGETS))
            ]
            dev_ensemble = [
                np.mean([run[task] for run in seed_dev_probabilities], axis=0)
                for task in range(len(TARGETS))
            ]
            fold_dev_ensembles[mode] = dev_ensemble
            fold_test_ensembles[mode] = ensemble
            for task, target in enumerate(TARGETS):
                y_true = labels["test"][:, task]
                y_pred = ensemble[task].argmax(axis=1)
                values = metric_values(y_true, y_pred, class_counts[task])
                metric_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "scheme": mode,
                        "target": target,
                        **values,
                        "utility": 0.5 * (values["uar"] + values["macro_f1"]),
                    }
                )
                for row_index, probability in enumerate(ensemble[task]):
                    prediction_rows.append(
                        {
                            "outer_fold": outer_fold,
                            "scheme": mode,
                            "segment_id": test_frame.loc[row_index, "segment_id"],
                            "source_id": test_frame.loc[row_index, "source_id"],
                            "target": target,
                            "y_true": int(y_true[row_index]),
                            "y_pred": int(y_pred[row_index]),
                            "probability": json.dumps(probability.tolist()),
                        }
                    )

        mert_dev = fold_dev_ensembles["mert_multitask"]
        muq_dev = fold_dev_ensembles["muq_multitask"]
        mert_test = fold_test_ensembles["mert_multitask"]
        muq_test = fold_test_ensembles["muq_multitask"]
        fusion_schemes = {
            "mert_muq_multitask_posterior_0p5": [
                0.5 * (mert_test[task] + muq_test[task])
                for task in range(len(TARGETS))
            ]
        }
        # Select class-bias strength separately for each target using only the
        # inner-development partition. Five UAR/F1 preferences are retained as
        # a sensitivity curve; tau=0 is preferred on exact ties.
        for source_mode in modes:
            for uar_weight in UTILITY_UAR_WEIGHTS:
                adjusted_test = []
                for task, target in enumerate(TARGETS):
                    candidates = []
                    for tau in PRIOR_ADJUSTMENT_TAUS:
                        development = adjust_for_training_prior(
                            fold_dev_ensembles[source_mode][task],
                            labels["train"][:, task],
                            class_counts[task],
                            tau,
                        )
                        values = metric_values(
                            labels["dev"][:, task],
                            development.argmax(axis=1),
                            class_counts[task],
                        )
                        score = weighted_selection_score(values, uar_weight)
                        candidates.append((score, -tau, tau, values))
                    selected = max(candidates, key=lambda row: (row[0], row[1]))
                    selected_tau = float(selected[2])
                    adjusted_test.append(
                        adjust_for_training_prior(
                            fold_test_ensembles[source_mode][task],
                            labels["train"][:, task],
                            class_counts[task],
                            selected_tau,
                        )
                    )
                    prior_adjustment_rows.append(
                        {
                            "outer_fold": int(outer_fold),
                            "source_scheme": source_mode,
                            "target": target,
                            "selection_uar_weight": uar_weight,
                            "selection_macro_f1_weight": 1.0 - uar_weight,
                            "selected_tau": selected_tau,
                            "inner_selection_score": float(selected[0]),
                            "inner_uar": float(selected[3]["uar"]),
                            "inner_macro_f1": float(selected[3]["macro_f1"]),
                        }
                    )
                if uar_weight == 0.5:
                    scheme = f"{source_mode}_inner_prior_adjusted"
                else:
                    tag = str(uar_weight).replace(".", "p")
                    scheme = f"{source_mode}_inner_class_bias_uarw_{tag}"
                fusion_schemes[scheme] = adjusted_test

        raw_fusion_dev = [
            0.5 * (mert_dev[task] + muq_dev[task])
            for task in range(len(TARGETS))
        ]
        raw_fusion_test = fusion_schemes["mert_muq_multitask_posterior_0p5"]
        for uar_weight in UTILITY_UAR_WEIGHTS:
            adjusted_fusion_test = []
            for task, target in enumerate(TARGETS):
                candidates = []
                for tau in PRIOR_ADJUSTMENT_TAUS:
                    development = adjust_for_training_prior(
                        raw_fusion_dev[task],
                        labels["train"][:, task],
                        class_counts[task],
                        tau,
                    )
                    values = metric_values(
                        labels["dev"][:, task],
                        development.argmax(axis=1),
                        class_counts[task],
                    )
                    score = weighted_selection_score(values, uar_weight)
                    candidates.append((score, -tau, tau, values))
                selected = max(candidates, key=lambda row: (row[0], row[1]))
                selected_tau = float(selected[2])
                adjusted_fusion_test.append(
                    adjust_for_training_prior(
                        raw_fusion_test[task],
                        labels["train"][:, task],
                        class_counts[task],
                        selected_tau,
                    )
                )
                prior_adjustment_rows.append(
                    {
                        "outer_fold": int(outer_fold),
                        "source_scheme": "mert_muq_multitask_posterior_0p5",
                        "target": target,
                        "selection_uar_weight": uar_weight,
                        "selection_macro_f1_weight": 1.0 - uar_weight,
                        "selected_tau": selected_tau,
                        "inner_selection_score": float(selected[0]),
                        "inner_uar": float(selected[3]["uar"]),
                        "inner_macro_f1": float(selected[3]["macro_f1"]),
                    }
                )
            if uar_weight == 0.5:
                scheme = "mert_muq_multitask_posterior_0p5_inner_prior_adjusted"
            else:
                tag = str(uar_weight).replace(".", "p")
                scheme = f"mert_muq_multitask_posterior_0p5_inner_class_bias_uarw_{tag}"
            fusion_schemes[scheme] = adjusted_fusion_test
        calibrated_mert_dev = []
        calibrated_muq_dev = []
        calibrated_mert_test = []
        calibrated_muq_test = []
        temperatures = []
        for task, target in enumerate(TARGETS):
            mert_temperature = select_temperature(labels["dev"][:, task], mert_dev[task])
            muq_temperature = select_temperature(labels["dev"][:, task], muq_dev[task])
            temperatures.append((mert_temperature, muq_temperature))
            calibrated_mert_dev.append(temperature_scale(mert_dev[task], mert_temperature))
            calibrated_muq_dev.append(temperature_scale(muq_dev[task], muq_temperature))
            calibrated_mert_test.append(temperature_scale(mert_test[task], mert_temperature))
            calibrated_muq_test.append(temperature_scale(muq_test[task], muq_temperature))
        fusion_schemes["mert_muq_multitask_calibrated_0p5"] = [
            0.5 * (calibrated_mert_test[task] + calibrated_muq_test[task])
            for task in range(len(TARGETS))
        ]
        selected_fusion = []
        for task, target in enumerate(TARGETS):
            candidates = []
            for weight in FUSION_WEIGHTS:
                dev_probability = (
                    weight * calibrated_mert_dev[task]
                    + (1.0 - weight) * calibrated_muq_dev[task]
                )
                values = metric_values(
                    labels["dev"][:, task],
                    dev_probability.argmax(axis=1),
                    class_counts[task],
                )
                candidates.append(
                    (values["utility"], -abs(weight - 0.5), weight)
                )
            selected_weight = float(max(candidates, key=lambda row: (row[0], row[1]))[2])
            selected_fusion.append(
                selected_weight * calibrated_mert_test[task]
                + (1.0 - selected_weight) * calibrated_muq_test[task]
            )
            fusion_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "target": target,
                    "mert_temperature": temperatures[task][0],
                    "muq_temperature": temperatures[task][1],
                    "selected_mert_weight": selected_weight,
                    "selected_muq_weight": 1.0 - selected_weight,
                    "inner_utility": float(
                        max(candidates, key=lambda row: (row[0], row[1]))[0]
                    ),
                }
            )
        fusion_schemes["mert_muq_multitask_calibrated_inner_weighted"] = selected_fusion

        for scheme, probabilities in fusion_schemes.items():
            for task, target in enumerate(TARGETS):
                y_true = labels["test"][:, task]
                y_pred = probabilities[task].argmax(axis=1)
                values = metric_values(y_true, y_pred, class_counts[task])
                metric_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "scheme": scheme,
                        "target": target,
                        **values,
                    }
                )
                for row_index, probability in enumerate(probabilities[task]):
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
        pd.DataFrame(metric_rows).to_csv(output / "per_aria_target_metrics.partial.csv", index=False)
        pd.DataFrame(prediction_rows).to_csv(output / "predictions_private.partial.csv", index=False)
        pd.DataFrame(run_rows).to_csv(output / "seed_runs.partial.csv", index=False)
        pd.DataFrame(seed_metric_rows).to_csv(output / "seed_metrics.partial.csv", index=False)
        pd.DataFrame(seed_prediction_rows).to_csv(
            output / "seed_predictions_private.partial.csv", index=False
        )
        pd.DataFrame(fusion_rows).to_csv(output / "posterior_fusion_selection.partial.csv", index=False)
        pd.DataFrame(prior_adjustment_rows).to_csv(
            output / "prior_adjustment_selection.partial.csv", index=False
        )
        print(f"completed aria fold {outer_fold}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    summary = (
        metrics.groupby("scheme")[["uar", "macro_f1", "accuracy", "utility"]]
        .mean()
        .reset_index()
    )
    metrics.to_csv(output / "per_aria_target_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / "predictions_private.csv", index=False)
    pd.DataFrame(run_rows).to_csv(output / "seed_runs.csv", index=False)
    pd.DataFrame(seed_metric_rows).to_csv(output / "seed_metrics.csv", index=False)
    pd.DataFrame(seed_prediction_rows).to_csv(
        output / "seed_predictions_private.csv", index=False
    )
    pd.DataFrame(fusion_rows).to_csv(output / "posterior_fusion_selection.csv", index=False)
    pd.DataFrame(prior_adjustment_rows).to_csv(
        output / "prior_adjustment_selection.csv", index=False
    )
    summary.to_csv(output / "summary.csv", index=False)
    manifest = {
        "protocol": "leave_one_aria_out_inner_source_disjoint_multitask",
        "device": str(device),
        "seeds": seeds,
        "targets": TARGETS,
        "class_counts": CLASS_COUNTS,
        "metric_policy": "present-class UAR/F1 for fold selection; global-class variants also saved",
        "posterior_fusion": {
            "weights": FUSION_WEIGHTS,
            "temperatures": TEMPERATURES,
            "selection_partition": "inner_dev",
        },
        "class_bias_adjustment": {
            "taus": PRIOR_ADJUSTMENT_TAUS,
            "selection_uar_weights": UTILITY_UAR_WEIGHTS,
            "selection_partition": "inner_dev",
            "class_prior_partition": "inner_train",
            "smoothing": "add one per class",
            "tie_break": "prefer smaller tau",
            "output_semantics": "normalized class-adjusted decision score, not calibrated posterior probability",
        },
        "hyperparameters": vars(args),
        "test_used_for_selection": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
