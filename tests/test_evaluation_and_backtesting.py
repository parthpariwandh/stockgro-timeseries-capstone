"""Focused smoke tests for evaluation metrics and trading backtest output."""

from __future__ import annotations

import unittest

import pandas as pd

from src.backtesting import generate_signals, run_backtest
from src.evaluation import forecast_metrics


class EvaluationAndBacktestingTests(unittest.TestCase):
    def test_forecast_metrics_keys(self) -> None:
        idx = pd.date_range("2025-01-01", periods=6, freq="B")
        actual = pd.Series([100, 101, 102, 103, 104, 105], index=idx)
        pred = pd.Series([100, 100.5, 102.5, 102.8, 104.1, 104.9], index=idx)
        metrics = forecast_metrics(actual, pred)
        expected_keys = {
            "rmse",
            "mae",
            "mape",
            "directional_accuracy",
            "r2",
            "forecast_bias",
        }
        self.assertEqual(set(metrics.keys()), expected_keys)

    def test_backtest_outputs_portfolio_value(self) -> None:
        idx = pd.date_range("2025-01-01", periods=8, freq="B")
        close = pd.Series([100, 101, 103, 102, 104, 106, 105, 107], index=idx)
        forecast = pd.Series([101, 102, 104, 101, 105, 107, 104, 108], index=idx)
        signals = generate_signals(close, forecast)
        results = run_backtest(close, signals)
        self.assertIn("portfolio_value", results.columns)
        self.assertGreater(float(results["portfolio_value"].iloc[-1]), 0.0)


if __name__ == "__main__":
    unittest.main()
