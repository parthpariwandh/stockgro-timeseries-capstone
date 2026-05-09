"""Portfolio construction and optimization strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _portfolio_stats(weights: np.ndarray, returns: pd.DataFrame) -> tuple[float, float, float]:
    expected = float(np.sum(returns.mean() * weights) * 252)
    covariance = returns.cov() * 252
    volatility = float(np.sqrt(weights.T @ covariance @ weights))
    sharpe = expected / volatility if volatility > 0 else 0.0
    return expected, volatility, sharpe


def optimize_max_sharpe(returns: pd.DataFrame) -> pd.Series:
    """Mean-variance optimization for max Sharpe portfolio."""
    n_assets = returns.shape[1]
    init = np.array([1 / n_assets] * n_assets)

    def objective(weights: np.ndarray) -> float:
        return -_portfolio_stats(weights, returns)[2]

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = tuple((0, 1) for _ in range(n_assets))
    result = minimize(objective, init, method="SLSQP", bounds=bounds, constraints=constraints)
    return pd.Series(result.x, index=returns.columns, name="max_sharpe_weights")


def forecast_guided_allocation(expected_returns: pd.Series) -> pd.Series:
    """Allocate proportionally to positive forecast alpha scores."""
    clipped = expected_returns.clip(lower=0)
    if clipped.sum() == 0:
        return pd.Series(1 / len(clipped), index=clipped.index)
    return (clipped / clipped.sum()).rename("forecast_guided")


def volatility_aware_diversification(volatility: pd.Series) -> pd.Series:
    """Inverse-volatility risk allocation."""
    inv_vol = 1 / volatility.replace(0, np.nan)
    weights = inv_vol / inv_vol.sum()
    return weights.fillna(0).rename("volatility_aware")


def momentum_risk_balanced(momentum: pd.Series, volatility: pd.Series) -> pd.Series:
    """Momentum score adjusted by volatility for balanced allocation."""
    score = momentum / volatility.replace(0, np.nan)
    score = score.clip(lower=0)
    if score.sum() == 0:
        return pd.Series(1 / len(score), index=score.index)
    return (score / score.sum()).fillna(0).rename("momentum_risk_balanced")
