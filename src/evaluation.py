"""Model and trading performance evaluation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def forecast_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Compute standard regression and directional forecasting metrics."""
    actual_aligned, predicted_aligned = actual.align(predicted, join="inner")
    errors = predicted_aligned - actual_aligned
    rmse = float(np.sqrt(mean_squared_error(actual_aligned, predicted_aligned)))
    mae = float(mean_absolute_error(actual_aligned, predicted_aligned))
    mape = float(np.mean(np.abs(errors / actual_aligned.replace(0, np.nan))) * 100)
    direction_actual = np.sign(actual_aligned.diff().fillna(0))
    direction_pred = np.sign(predicted_aligned.diff().fillna(0))
    directional_accuracy = float((direction_actual == direction_pred).mean() * 100)
    bias = float(errors.mean())

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "directional_accuracy": directional_accuracy,
        "r2": float(r2_score(actual_aligned, predicted_aligned)),
        "forecast_bias": bias,
    }


def rank_models(results: dict[str, dict[str, float]], metric: str = "rmse") -> pd.DataFrame:
    """Rank model performance by a selected metric."""
    frame = pd.DataFrame(results).T
    ascending = metric not in {"directional_accuracy", "r2"}
    return frame.sort_values(metric, ascending=ascending)
