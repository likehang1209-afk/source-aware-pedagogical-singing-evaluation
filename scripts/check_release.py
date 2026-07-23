"""Check release structure, identifiers, probabilities, and file hashes."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".txt",
    ".json",
    ".yml",
    ".yaml",
    ".cff",
    ".gitignore",
}
FORBIDDEN_TEXT = [
    re.compile(r"https?://(?:www\.)?(?:youtube\.com|youtu\.be)", re.I),
    re.compile(r"connect\.cqa\d*\.seetacloud\.com", re.I),
    re.compile(r"ssh\s+-p\s+\d+\s+root@", re.I),
    re.compile(r"[A-Za-z]:\\(?:Backup|Users|人声分离)", re.I),
    re.compile(r"/root/autodl", re.I),
]


def verify_manifest() -> None:
    manifest_path = ROOT / "release_manifest_sha256.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = ROOT / item["path"]
        if not path.exists():
            raise AssertionError(f"Manifest file missing: {item['path']}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise AssertionError(f"Hash mismatch: {item['path']}")


def verify_text() -> None:
    findings = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.name in {"LICENSE", "requirements.txt", "requirements-training.txt"}:
            is_text = True
        else:
            is_text = path.suffix.lower() in TEXT_SUFFIXES
        if not is_text:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in FORBIDDEN_TEXT:
            if pattern.search(text):
                findings.append(f"{path.relative_to(ROOT)}: {pattern.pattern}")
    if findings:
        raise AssertionError("Sensitive text found:\n" + "\n".join(findings))


def verify_manifests_and_predictions() -> None:
    cohort = pd.read_csv(ROOT / "data_manifests" / "cohort_pseudonymized.csv.gz")
    if cohort["segment_uid"].nunique() != 3456:
        raise AssertionError("Unexpected cohort segment count")
    if cohort["source_uid"].nunique() != 227:
        raise AssertionError("Unexpected cohort source count")
    if cohort["aria"].nunique() != 6:
        raise AssertionError("Unexpected cohort aria count")
    if set(cohort.columns) & {"segment_id", "source_id", "segment_path", "url"}:
        raise AssertionError("Private identifier column present in cohort")

    predictions = pd.read_csv(
        ROOT / "predictions" / "main_oof_predictions.csv.gz"
    )
    expected_rows = predictions["method"].nunique() * 3456 * 7
    if len(predictions) != expected_rows:
        raise AssertionError(
            f"Prediction rows {len(predictions)} do not match {expected_rows}"
        )
    keys = ["method", "segment_uid", "target"]
    if predictions.duplicated(keys).any():
        raise AssertionError("Duplicate method-segment-target prediction")
    if set(predictions.columns) & {"segment_id", "source_id", "segment_path", "url"}:
        raise AssertionError("Private identifier column present in predictions")

    for value in predictions["probability"]:
        vector = np.asarray(ast.literal_eval(value), dtype=np.float64)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise AssertionError("Invalid probability vector")
        if not np.isclose(vector.sum(), 1.0, atol=1e-6):
            raise AssertionError("Probability vector does not sum to one")


def main() -> None:
    verify_manifest()
    verify_text()
    verify_manifests_and_predictions()
    print("OK: release manifest, privacy scan, cohort, and OOF predictions pass")


if __name__ == "__main__":
    main()
