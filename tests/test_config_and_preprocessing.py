"""Focused smoke tests for config loading and preprocessing behavior."""

from __future__ import annotations

import unittest

import pandas as pd

from src.config import load_config
from src.preprocessing import add_features, clean_ohlcv, time_split


class ConfigAndPreprocessingTests(unittest.TestCase):
    def test_load_config_contains_required_horizons(self) -> None:
        config = load_config()
        self.assertEqual(config.forecasting.live_horizon_days, 2)
        self.assertEqual(config.forecasting.validation_horizon_days, 5)
        self.assertEqual(config.data.test_months, 6)

    def test_preprocessing_feature_pipeline(self) -> None:
        idx = pd.date_range("2021-01-01", periods=300, freq="B")
        df = pd.DataFrame(
            {
                "Open": range(300),
                "High": range(1, 301),
                "Low": range(300),
                "Close": range(300),
                "Adj Close": range(300),
                "Volume": [1000] * 300,
            },
            index=idx,
        )
        cleaned = clean_ohlcv(df)
        featured = add_features(cleaned)
        train, test = time_split(featured, test_months=6)
        self.assertFalse(featured.empty)
        self.assertGreater(len(test), 0)
        self.assertGreater(len(train), 0)


if __name__ == "__main__":
    unittest.main()
