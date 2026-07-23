"""Build deterministic inner/outer roles for leave-one-aria-out evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


TARGETS = [
    "chest_resonance",
    "head_resonance",
    "front_placement",
    "back_placement",
    "open_throat",
    "breathiness",
    "vibrato",
]


def distribution(frame: pd.DataFrame, target: str, classes: list[int]) -> np.ndarray:
    counts = frame[target].value_counts().reindex(classes, fill_value=0).to_numpy(dtype=float)
    return counts / max(1.0, counts.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata)
    assignments = pd.read_csv(args.assignments)
    data = metadata.merge(assignments, on=["segment_id", "source_id"], how="inner", validate="one_to_one")
    role_rows = []
    summaries = []

    for fold in sorted(data["outer_fold"].unique()):
        outer_test = data[data["outer_fold"].eq(fold)]
        outer_train = data[~data["outer_fold"].eq(fold)].copy()
        groups = outer_train["source_id"].to_numpy()
        candidates = GroupShuffleSplit(
            n_splits=args.candidates,
            test_size=0.2,
            random_state=args.seed + int(fold),
        )
        best = None
        for train_index, dev_index in candidates.split(outer_train, groups=groups):
            inner_train = outer_train.iloc[train_index]
            inner_dev = outer_train.iloc[dev_index]
            score = abs(inner_dev["source_id"].nunique() / outer_train["source_id"].nunique() - 0.2)
            valid = True
            for target in TARGETS:
                classes = sorted(outer_train[target].unique())
                if set(classes) - set(inner_train[target].unique()) or set(classes) - set(inner_dev[target].unique()):
                    valid = False
                    break
                reference = distribution(outer_train, target, classes)
                score += np.abs(distribution(inner_train, target, classes) - reference).mean()
                score += np.abs(distribution(inner_dev, target, classes) - reference).mean()
            if valid and (best is None or score < best[0]):
                best = (float(score), set(inner_dev["source_id"]))
        if best is None:
            raise RuntimeError(f"No valid inner split found for aria fold {fold}")
        dev_sources = best[1]
        for row in data.itertuples(index=False):
            role = "outer_test" if row.outer_fold == fold else (
                "inner_dev" if row.source_id in dev_sources else "inner_train"
            )
            role_rows.append(
                {
                    "outer_fold": int(fold),
                    "segment_id": row.segment_id,
                    "source_id": row.source_id,
                    "role": role,
                }
            )
        summaries.append(
            {
                "outer_fold": int(fold),
                "held_aria": outer_test["aria"].iloc[0],
                "outer_test_sources": int(outer_test["source_id"].nunique()),
                "outer_test_excerpts": int(len(outer_test)),
                "inner_dev_sources": int(len(dev_sources)),
                "inner_objective": best[0],
            }
        )

    roles = pd.DataFrame(role_rows)
    roles.to_csv(output / "aria_nested_split_roles_private.csv", index=False)
    assignments.to_csv(output / "aria_outer_fold_assignments_private.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "aria_fold_summary.csv", index=False)
    manifest = {
        "protocol": "leave one recovered aria out; source-disjoint inner development within remaining arias",
        "mapped_excerpts": int(len(data)),
        "mapped_sources": int(data["source_id"].nunique()),
        "outer_folds": int(data["outer_fold"].nunique()),
        "candidate_inner_splits_per_fold": args.candidates,
        "seed": args.seed,
        "private_fields": ["source_id", "aria"],
    }
    (output / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
