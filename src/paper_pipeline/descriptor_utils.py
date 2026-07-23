"""Validation helpers for the explicit SVQTD acoustic descriptor table."""

from __future__ import annotations

import re

import pandas as pd


def validate_descriptor_columns(
    frame: pd.DataFrame, sample_rate: int, prefix: str = "ped_"
) -> dict:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    candidate_columns = [column for column in frame.columns if column.startswith(prefix)]
    nyquist = sample_rate / 2.0
    above_nyquist = []
    for column in candidate_columns:
        match = re.search(r"_(\d+)_(\d+)_ratio", column)
        if match and float(match.group(1)) >= nyquist:
            above_nyquist.append(column)
    constant_columns = [
        column
        for column in candidate_columns
        if pd.to_numeric(frame[column], errors="coerce").nunique(dropna=True) <= 1
    ]
    excluded_columns = sorted(set(above_nyquist).union(constant_columns))
    retained_features = [
        column for column in candidate_columns if column not in excluded_columns
    ]
    if not retained_features:
        raise ValueError("No usable descriptors remain after validation")
    return {
        "sample_rate_hz": sample_rate,
        "nyquist_hz": nyquist,
        "candidate_feature_count": len(candidate_columns),
        "retained_feature_count": len(retained_features),
        "retained_features": retained_features,
        "excluded_above_nyquist": above_nyquist,
        "excluded_constant": constant_columns,
    }
