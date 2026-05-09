"""Stationarity diagnostics and differencing helpers."""

from __future__ import annotations

import pandas as pd

try:
    from statsmodels.tsa.stattools import adfuller, kpss
except Exception:  # pragma: no cover - optional dependency fallback
    adfuller = None
    kpss = None


def adf_test(series: pd.Series) -> dict[str, float | bool | str]:
    """Run ADF test and return the key statistics."""
    if adfuller is None:
        raise ImportError("statsmodels is required for adf_test")
    statistic, pvalue, *_ = adfuller(series.dropna())
    return {
        "test": "ADF",
        "statistic": float(statistic),
        "p_value": float(pvalue),
        "is_stationary": bool(pvalue < 0.05),
    }


def kpss_test(series: pd.Series) -> dict[str, float | bool | str]:
    """Run KPSS test and return the key statistics."""
    if kpss is None:
        raise ImportError("statsmodels is required for kpss_test")
    statistic, pvalue, *_ = kpss(series.dropna(), regression="c", nlags="auto")
    return {
        "test": "KPSS",
        "statistic": float(statistic),
        "p_value": float(pvalue),
        "is_stationary": bool(pvalue >= 0.05),
    }


def difference_until_stationary(series: pd.Series, max_diff: int = 2) -> tuple[pd.Series, int]:
    """Apply iterative differencing until ADF signals stationarity or max diff is hit."""
    transformed = series.copy()
    if adfuller is None:
        return transformed.dropna(), 0

    for d in range(max_diff + 1):
        pvalue = adfuller(transformed.dropna())[1]
        if pvalue < 0.05:
            return transformed.dropna(), d
        transformed = transformed.diff().dropna()
    return transformed.dropna(), max_diff
