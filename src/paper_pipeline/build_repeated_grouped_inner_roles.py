"""Create repeated source-grouped inner splits while holding outer arias fixed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metric_utils import CLASS_COUNTS, TARGETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--base-roles", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def distribution_vector(frame: pd.DataFrame) -> np.ndarray:
    values = []
    for target in TARGETS:
        counts = (
            frame[target]
            .value_counts()
            .reindex(range(CLASS_COUNTS[target]), fill_value=0)
            .to_numpy(dtype=np.float64)
        )
        values.extend((counts / max(counts.sum(), 1.0)).tolist())
    return np.asarray(values, dtype=np.float64)


def missing_class_count(frame: pd.DataFrame, reference: pd.DataFrame) -> int:
    missing = 0
    for target in TARGETS:
        reference_classes = set(reference[target].astype(int).unique())
        observed = set(frame[target].astype(int).unique())
        missing += len(reference_classes - observed)
    return missing


def support_rows(
    fold: int, repeat: int, role: str, frame: pd.DataFrame
) -> list[dict]:
    rows = []
    for target in TARGETS:
        counts = (
            frame[target]
            .value_counts()
            .reindex(range(CLASS_COUNTS[target]), fill_value=0)
        )
        for class_index, count in counts.items():
            rows.append(
                {
                    "outer_fold": fold,
                    "inner_repeat": repeat,
                    "role": role,
                    "target": target,
                    "class": int(class_index),
                    "count": int(count),
                    "source_count": int(frame.source_id.nunique()),
                    "segment_count": len(frame),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata).reset_index(drop=True)
    base_roles = pd.read_csv(args.base_roles)
    metadata["segment_id"] = metadata.segment_id.astype(str)
    metadata["source_id"] = metadata.source_id.astype(str)
    base_roles["segment_id"] = base_roles.segment_id.astype(str)
    base_roles["source_id"] = base_roles.source_id.astype(str)
    rng = np.random.default_rng(args.seed)
    role_rows = []
    diagnostic_rows = []
    support = []

    for fold in sorted(base_roles.outer_fold.unique()):
        fold_roles = base_roles[base_roles.outer_fold == fold]
        fold_data = fold_roles.merge(
            metadata, on=["segment_id", "source_id"], validate="one_to_one"
        )
        outer_test = fold_data[fold_data.role == "outer_test"]
        outer_train = fold_data[fold_data.role != "outer_test"]
        original_dev = fold_data[fold_data.role == "inner_dev"]
        source_ids = np.asarray(sorted(outer_train.source_id.unique()), dtype=object)
        dev_source_count = int(original_dev.source_id.nunique())
        target_dev_fraction = len(original_dev) / len(outer_train)
        full_distribution = distribution_vector(outer_train)
        used: set[tuple[str, ...]] = set()

        for repeat in range(args.repeats):
            best = None
            for _ in range(args.candidates):
                chosen = tuple(
                    sorted(
                        rng.choice(source_ids, size=dev_source_count, replace=False).tolist()
                    )
                )
                if chosen in used:
                    continue
                is_dev = outer_train.source_id.isin(chosen)
                dev = outer_train[is_dev]
                train = outer_train[~is_dev]
                distribution_error = float(
                    np.mean(np.abs(distribution_vector(dev) - full_distribution))
                )
                size_error = abs(len(dev) / len(outer_train) - target_dev_fraction)
                dev_missing = missing_class_count(dev, outer_train)
                train_missing = missing_class_count(train, outer_train)
                score = (
                    distribution_error
                    + 0.5 * size_error
                    + 0.25 * dev_missing
                    + 1.0 * train_missing
                )
                candidate = (
                    score,
                    dev_missing + train_missing,
                    size_error,
                    chosen,
                    dev,
                    train,
                )
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
            if best is None:
                raise RuntimeError(f"No unique split found for fold {fold}, repeat {repeat}")
            score, total_missing, size_error, chosen, dev, train = best
            used.add(chosen)
            assignments = pd.concat(
                [
                    train.assign(role="inner_train"),
                    dev.assign(role="inner_dev"),
                    outer_test.assign(role="outer_test"),
                ],
                ignore_index=True,
            )
            for row in assignments.itertuples(index=False):
                role_rows.append(
                    {
                        "outer_fold": int(fold),
                        "inner_repeat": repeat,
                        "segment_id": str(row.segment_id),
                        "source_id": str(row.source_id),
                        "role": row.role,
                    }
                )
            diagnostic_rows.append(
                {
                    "outer_fold": int(fold),
                    "inner_repeat": repeat,
                    "split_score": float(score),
                    "missing_class_penalty_count": int(total_missing),
                    "dev_fraction_error": float(size_error),
                    "train_sources": int(train.source_id.nunique()),
                    "dev_sources": int(dev.source_id.nunique()),
                    "test_sources": int(outer_test.source_id.nunique()),
                    "train_segments": len(train),
                    "dev_segments": len(dev),
                    "test_segments": len(outer_test),
                    "dev_source_ids": json.dumps(chosen),
                }
            )
            support.extend(support_rows(int(fold), repeat, "inner_train", train))
            support.extend(support_rows(int(fold), repeat, "inner_dev", dev))
            support.extend(support_rows(int(fold), repeat, "outer_test", outer_test))

    roles = pd.DataFrame(role_rows)
    duplicates = roles.duplicated(
        ["outer_fold", "inner_repeat", "segment_id"], keep=False
    )
    if duplicates.any():
        raise ValueError("Repeated role construction created duplicate assignments")
    roles.to_csv(args.output_dir / "repeated_inner_roles_private.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(
        args.output_dir / "repeated_inner_split_diagnostics.csv", index=False
    )
    pd.DataFrame(support).to_csv(
        args.output_dir / "repeated_inner_class_support.csv", index=False
    )
    manifest = {
        "outer_test_policy": "identical to base leave-one-aria-out roles",
        "inner_group": "source_id",
        "repeats_per_outer_fold": args.repeats,
        "candidate_group_splits_per_repeat": args.candidates,
        "objective": (
            "match full outer-train class proportions and original dev size; "
            "penalize missing train/dev classes"
        ),
        "random_seed": args.seed,
        "base_roles": str(args.base_roles),
        "metadata": str(args.metadata),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(diagnostic_rows).to_string(index=False))


if __name__ == "__main__":
    main()
