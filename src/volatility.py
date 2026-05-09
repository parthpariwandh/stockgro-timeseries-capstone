"""Volatility modelling and risk analytics functions."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from arch import arch_model
except Exception:  # pragma: no cover
    arch_model = None


def compute_log_returns(close: pd.Series) -> pd.Series:
    """Compute continuously compounded returns."""
    return np.log(close / close.shift(1)).dropna()


def rolling_volatility(returns: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """Compute rolling standard deviation and optionally annualize it."""
    vol = returns.rolling(window).std()
    return vol * np.sqrt(252) if annualize else vol


def fit_garch_11(returns: pd.Series):
    """Fit a GARCH(1,1) process for volatility forecasting."""
    if arch_model is None:
        raise ImportError("arch is required for GARCH modelling")
    model = arch_model(returns.dropna() * 100, p=1, q=1, vol="GARCH")
    return model.fit(disp="off")


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio."""
    excess = returns - risk_free_rate / 252
    return float(np.sqrt(252) * excess.mean() / excess.std(ddof=1))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sortino ratio."""
    excess = returns - risk_free_rate / 252
    downside = excess[excess < 0].std(ddof=1)
    return float(np.sqrt(252) * excess.mean() / downside)


def max_drawdown(cumulative_returns: pd.Series) -> float:
    """Maximum observed drawdown in cumulative return curve."""
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    return float(drawdown.min())


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Value at Risk estimate."""
    percentile = (1 - confidence) * 100
    return float(np.percentile(returns.dropna(), percentile))
