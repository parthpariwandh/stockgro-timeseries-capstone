"""Trading signal generation and portfolio backtesting workflow."""

from __future__ import annotations

import pandas as pd


def generate_signals(close: pd.Series, forecast: pd.Series, threshold: float = 0.005) -> pd.Series:
    """Generate Buy/Sell/Hold signals from forecasted return thresholds."""
    aligned_close, aligned_forecast = close.align(forecast, join="inner")
    forecast_ret = (aligned_forecast - aligned_close) / aligned_close
    signals = pd.Series("HOLD", index=forecast_ret.index)
    signals[forecast_ret > threshold] = "BUY"
    signals[forecast_ret < -threshold] = "SELL"
    return signals


def run_backtest(
    close: pd.Series,
    signals: pd.Series,
    initial_cash: float = 1_000_000.0,
    stop_loss: float = 0.08,
) -> pd.DataFrame:
    """Run a simple discrete-time backtest with stop-loss and cash accounting."""
    position = 0
    cash = initial_cash
    entry_price = 0.0
    records: list[dict[str, float | int | str]] = []

    for timestamp, price in close.items():
        signal = signals.reindex([timestamp]).fillna("HOLD").iloc[0]
        if position > 0 and price <= entry_price * (1 - stop_loss):
            signal = "SELL"

        if signal == "BUY" and cash >= price:
            qty = int(cash // price)
            if qty > 0:
                position += qty
                cash -= qty * float(price)
                entry_price = float(price)
        elif signal == "SELL" and position > 0:
            cash += position * float(price)
            position = 0
            entry_price = 0.0

        portfolio_value = cash + position * float(price)
        records.append(
            {
                "date": timestamp,
                "signal": signal,
                "price": float(price),
                "position": position,
                "cash": float(cash),
                "portfolio_value": float(portfolio_value),
            }
        )

    return pd.DataFrame(records).set_index("date")
