"""Summarize rare-class support across six folds and five inner repetitions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data_manifests" / "inner_class_support_full.csv"
EXPECTED = ROOT / "data_manifests" / "inner_class_support_summary.csv"
OUTPUT = ROOT / "reproduced" / "inner_class_support_summary.csv"


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[frame["role"].isin(["inner_train", "inner_dev"])]
    return (
        selected.groupby(["role", "target", "class"], as_index=False)["count"]
        .agg(
            minimum="min",
            median="median",
            maximum="max",
            zero_support_cells=lambda values: int((values == 0).sum()),
            evaluated_cells="size",
        )
        .sort_values(["role", "target", "class"])
        .reset_index(drop=True)
    )


def main() -> None:
    summary = summarize(pd.read_csv(INPUT))
    expected = pd.read_csv(EXPECTED).sort_values(
        ["role", "target", "class"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(summary, expected, check_dtype=False)
    if int(summary["zero_support_cells"].sum()) != 0:
        raise AssertionError("At least one inner train/development class is absent")
    dev_minimum = int(summary.loc[summary["role"] == "inner_dev", "minimum"].min())
    if dev_minimum != 2:
        raise AssertionError(f"Expected rarest inner-dev support 2, got {dev_minimum}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT, index=False)
    print(
        "OK: all inner partitions contain every class; "
        f"minimum inner-development support={dev_minimum}"
    )


if __name__ == "__main__":
    main()

