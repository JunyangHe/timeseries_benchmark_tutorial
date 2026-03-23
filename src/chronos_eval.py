from __future__ import annotations

from typing import List

import pandas as pd
from tqdm.auto import tqdm


def _extract_point_forecast(pred_df: pd.DataFrame) -> float:
    """Extract a single point forecast from Chronos prediction output."""
    if pred_df.empty:
        raise ValueError("Chronos returned an empty prediction dataframe.")

    candidate_cols = ["0.5", "median", "mean", "prediction", "target"]
    for col in candidate_cols:
        if col in pred_df.columns:
            return float(pred_df.iloc[0][col])

    # Fallback: first numeric non-id/time column.
    skip_cols = {"id", "item_id", "timestamp", "ds"}
    numeric_cols = [
        c for c in pred_df.columns if c not in skip_cols and pd.api.types.is_numeric_dtype(pred_df[c])
    ]
    if not numeric_cols:
        raise ValueError(f"Could not infer prediction column from {list(pred_df.columns)}")
    return float(pred_df.iloc[0][numeric_cols[0]])


def rolling_one_step_predictions(
    pipeline,
    context_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lookback_window: int = 30,
    quantile_levels: List[float] | None = None,
) -> pd.DataFrame:
    """
    Run rolling one-step predictions per basin with observed-history updates.

    Returns a dataframe with columns:
    - id
    - timestamp
    - actual
    - predicted
    """
    if quantile_levels is None:
        quantile_levels = [0.5]

    context_df = context_df.sort_values(["id", "timestamp"]).reset_index(drop=True)
    test_df = test_df.sort_values(["id", "timestamp"]).reset_index(drop=True)

    hist_by_id = {k: g.copy() for k, g in context_df.groupby("id", sort=False)}
    out_rows = []

    # Iterate per basin and roll forward using actual past observations.
    for basin_id, g_test in tqdm(test_df.groupby("id", sort=False), desc="Rolling inference"):
        history = hist_by_id.get(basin_id, pd.DataFrame(columns=["id", "timestamp", "target"]))
        history = history.sort_values("timestamp").reset_index(drop=True)
        if history.empty:
            continue

        for _, row in g_test.iterrows():
            context_slice = history.tail(lookback_window)
            pred_df = pipeline.predict_df(
                context_slice,
                prediction_length=1,
                quantile_levels=quantile_levels,
                id_column="id",
                timestamp_column="timestamp",
                target="target",
            )
            y_hat = _extract_point_forecast(pred_df)

            out_rows.append(
                {
                    "id": basin_id,
                    "timestamp": row["timestamp"],
                    "actual": float(row["target"]),
                    "predicted": y_hat,
                }
            )

            # Append actual observation so next one-step forecast uses true history.
            history = pd.concat(
                [
                    history,
                    pd.DataFrame(
                        [{"id": basin_id, "timestamp": row["timestamp"], "target": row["target"]}]
                    ),
                ],
                ignore_index=True,
            )

    return pd.DataFrame(out_rows)
