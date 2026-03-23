from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd


@dataclass
class DataSpec:
    timestamp_col: str = "Year_Mnth_Day"
    id_col: str = "basin_id"
    target_col: str = "QObs(mm/d)"
    drop_unnamed_first_col: bool = True


def load_and_prepare_csv(csv_path: str, spec: DataSpec) -> pd.DataFrame:
    """Load CSV and apply required cleaning for the benchmark."""
    df = pd.read_csv(csv_path)

    if spec.drop_unnamed_first_col and len(df.columns) > 0:
        first_col = str(df.columns[0]).strip().lower()
        if first_col.startswith("unnamed") or first_col == "":
            df = df.drop(columns=[df.columns[0]])

    df[spec.timestamp_col] = pd.to_datetime(df[spec.timestamp_col], errors="coerce")
    df = df.dropna(subset=[spec.timestamp_col, spec.id_col, spec.target_col]).copy()
    df = df.sort_values([spec.id_col, spec.timestamp_col]).reset_index(drop=True)

    # Forward fill all numeric columns within each basin to keep timeline continuity.
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        df[numeric_cols] = df.groupby(spec.id_col)[numeric_cols].ffill()
        df = df.dropna(subset=[spec.target_col]).copy()

    return df


def chronological_split_by_id(
    df: pd.DataFrame,
    id_col: str,
    timestamp_col: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Dict[str, pd.DataFrame]:
    """Split each basin chronologically, then concatenate global splits."""
    if round(train_ratio + val_ratio + test_ratio, 10) != 1.0:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    train_parts = []
    val_parts = []
    test_parts = []

    for _, group in df.groupby(id_col, sort=False):
        g = group.sort_values(timestamp_col)
        n = len(g)
        if n < 3:
            continue

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        # Guarantee at least one row in each split when possible.
        train_end = max(train_end, 1)
        val_end = max(val_end, train_end + 1)
        val_end = min(val_end, n - 1)

        train_parts.append(g.iloc[:train_end])
        val_parts.append(g.iloc[train_end:val_end])
        test_parts.append(g.iloc[val_end:])

    return {
        "train": pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(),
        "val": pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame(),
        "test": pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(),
    }


def to_chronos_df(
    df: pd.DataFrame,
    id_col: str,
    timestamp_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Convert data to Chronos expected long format."""
    out = df[[id_col, timestamp_col, target_col]].copy()
    out = out.rename(columns={id_col: "id", timestamp_col: "timestamp", target_col: "target"})
    out["id"] = out["id"].astype(str)
    out = out.sort_values(["id", "timestamp"]).reset_index(drop=True)
    return out


def build_context_and_test(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build context_df and test_df for Chronos rolling evaluation.

    context_df contains all rows available before test start (train + val).
    test_df contains rows where targets are evaluated.
    """
    context_df = pd.concat([train_df, val_df], ignore_index=True)
    context_df = context_df.sort_values(["id", "timestamp"]).reset_index(drop=True)
    test_df = test_df.sort_values(["id", "timestamp"]).reset_index(drop=True)
    return context_df, test_df
