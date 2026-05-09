"""Data preprocessing and feature engineering for time-series modelling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Clean an OHLCV frame with ordered datetime index and imputed values."""
    data = df.copy()
    data = data.sort_index()
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    data = data.ffill().bfill()
    return data


def remove_outliers_iqr(series: pd.Series, whisker_width: float = 1.5) -> pd.Series:
    """Clip outliers via IQR bounds to reduce noise spikes."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - whisker_width * iqr
    upper = q3 + whisker_width * iqr
    return series.clip(lower=lower, upper=upper)


def add_features(df: pd.DataFrame, close_col: str = "Close") -> pd.DataFrame:
    """Add lag, momentum, volatility, moving-average, and return features."""
    data = df.copy()
    close = data[close_col]
    data["log_return"] = np.log(close / close.shift(1))
    data["return_1d"] = close.pct_change(1)
    data["lag_1"] = close.shift(1)
    data["lag_2"] = close.shift(2)
    data["sma_5"] = close.rolling(5).mean()
    data["sma_20"] = close.rolling(20).mean()
    data["volatility_20"] = data["return_1d"].rolling(20).std()
    data["momentum_5"] = close - close.shift(5)
    return data.dropna()


def scale_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Scale train and test features with a train-fitted MinMax scaler."""
    scaler = MinMaxScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train), index=train.index, columns=train.columns
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test), index=test.index, columns=test.columns
    )
    return train_scaled, test_scaled, scaler


def time_split(df: pd.DataFrame, test_months: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by reserving the final N months for testing/validation."""
    split_cutoff = df.index.max() - pd.DateOffset(months=test_months)
    train = df[df.index < split_cutoff]
    test = df[df.index >= split_cutoff]
    return train, test
