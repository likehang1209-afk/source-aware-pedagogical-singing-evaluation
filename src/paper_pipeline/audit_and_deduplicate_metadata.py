"""Audit cross-partition duplicate segment IDs before grouped evaluation.

Exact duplicate IDs with identical target labels are retained once according
to an explicit split priority.  Any duplicate ID with conflicting target labels
is excluded in full because no label can be preferred without external evidence.
The audit table preserves every decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from metric_utils import TARGETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-priority", default="train,val,validation,dev,test")
    return parser.parse_args()


def deduplicate(
    metadata: pd.DataFrame, split_priority: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"segment_id", "split", *TARGETS}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata lacks columns: {sorted(missing)}")
    priority = {name.lower(): index for index, name in enumerate(split_priority)}
    keep_indices = []
    audit_rows = []
    for segment_id, group in metadata.groupby("segment_id", sort=False, dropna=False):
        group = group.copy()
        if len(group) == 1:
            keep_indices.append(group.index[0])
            continue
        unique_labels = group[TARGETS].drop_duplicates()
        label_conflict = len(unique_labels) > 1
        group["_priority"] = group["split"].astype(str).str.lower().map(priority).fillna(
            len(priority)
        )
        group = group.sort_values(["_priority", "split"], kind="stable")
        kept_index = None if label_conflict else int(group.index[0])
        if kept_index is not None:
            keep_indices.append(kept_index)
        for index, row in group.iterrows():
            audit_rows.append(
                {
                    "segment_id": segment_id,
                    "original_row": int(index),
                    "split": row["split"],
                    "segment_path": row.get("segment_path", row.get("audio_relpath", "")),
                    "label_conflict": bool(label_conflict),
                    "action": (
                        "exclude_conflicting_group"
                        if label_conflict
                        else ("retain" if index == kept_index else "remove_exact_duplicate")
                    ),
                    "labels": json.dumps(
                        {target: int(row[target]) for target in TARGETS}, sort_keys=True
                    ),
                }
            )
    cleaned = metadata.loc[sorted(keep_indices)].reset_index(drop=True)
    if cleaned.segment_id.duplicated().any():
        raise AssertionError("Deduplication left repeated segment IDs")
    return cleaned, pd.DataFrame(audit_rows)


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata)
    split_priority = [value.strip() for value in args.split_priority.split(",") if value.strip()]
    cleaned, audit = deduplicate(metadata, split_priority)
    cleaned.to_csv(output / "metadata_deduplicated.csv", index=False)
    audit.to_csv(output / "duplicate_resolution_audit.csv", index=False)
    manifest = {
        "input_rows": int(len(metadata)),
        "output_rows": int(len(cleaned)),
        "duplicate_ids": int(audit.segment_id.nunique()) if len(audit) else 0,
        "conflicting_duplicate_ids": int(
            audit.loc[audit.label_conflict, "segment_id"].nunique()
        )
        if len(audit)
        else 0,
        "split_priority": split_priority,
        "policy": "retain one identical-label duplicate; exclude every row of a conflicting-label duplicate",
    }
    (output / "duplicate_resolution_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
