"""Visualization utilities for publication-quality financial charts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_candlestick(df: pd.DataFrame, title: str) -> go.Figure:
    """Build interactive candlestick chart."""
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
            )
        ]
    )
    fig.update_layout(title=title, template="plotly_white")
    return fig


def plot_forecast(actual: pd.Series, predicted: pd.Series, title: str) -> plt.Figure:
    """Create forecast vs actual line chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    actual.plot(ax=ax, label="Actual", linewidth=2)
    predicted.plot(ax=ax, label="Forecast", linestyle="--")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(returns: pd.DataFrame, title: str = "Correlation Heatmap") -> go.Figure:
    """Create interactive correlation heatmap."""
    corr = returns.corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1, title=title)
    fig.update_layout(template="plotly_white")
    return fig
