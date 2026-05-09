"""Forecasting models and walk-forward helpers for market time series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from prophet import Prophet
except Exception:  # pragma: no cover
    Prophet = None

try:
    from pmdarima import auto_arima
except Exception:  # pragma: no cover
    auto_arima = None

try:
    from statsmodels.tsa.api import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:  # pragma: no cover
    ExponentialSmoothing = None
    SARIMAX = None

try:
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential
except Exception:  # pragma: no cover
    Sequential = None


@dataclass
class ForecastOutput:
    """Container for model forecasts and confidence interval bounds."""

    prediction: pd.Series
    lower_ci: pd.Series | None = None
    upper_ci: pd.Series | None = None


def persistence_forecast(series: pd.Series, horizon: int) -> ForecastOutput:
    """Baseline persistence forecast using the latest observed value."""
    last_value = float(series.iloc[-1])
    index = pd.RangeIndex(start=1, stop=horizon + 1, name="step")
    pred = pd.Series([last_value] * horizon, index=index, name="persistence")
    return ForecastOutput(prediction=pred)


def arima_forecast(series: pd.Series, horizon: int) -> ForecastOutput:
    """ARIMA forecast with optional auto_arima tuning and CI extraction."""
    if SARIMAX is None:
        raise ImportError("statsmodels is required for arima_forecast")

    order = (1, 1, 1)
    if auto_arima is not None:
        model_hint = auto_arima(series, seasonal=False, suppress_warnings=True)
        order = model_hint.order

    model = SARIMAX(series, order=order, enforce_stationarity=False)
    fitted = model.fit(disp=False)
    pred_obj = fitted.get_forecast(steps=horizon)
    ci = pred_obj.conf_int()
    return ForecastOutput(
        prediction=pred_obj.predicted_mean.rename("arima"),
        lower_ci=ci.iloc[:, 0].rename("lower_ci"),
        upper_ci=ci.iloc[:, 1].rename("upper_ci"),
    )


def sarima_forecast(series: pd.Series, horizon: int, seasonal_period: int = 5) -> ForecastOutput:
    """SARIMA forecast for seasonality-aware short-horizon forecasts."""
    if SARIMAX is None:
        raise ImportError("statsmodels is required for sarima_forecast")

    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, seasonal_period),
        enforce_stationarity=False,
    )
    fitted = model.fit(disp=False)
    pred_obj = fitted.get_forecast(steps=horizon)
    ci = pred_obj.conf_int()
    return ForecastOutput(
        prediction=pred_obj.predicted_mean.rename("sarima"),
        lower_ci=ci.iloc[:, 0].rename("lower_ci"),
        upper_ci=ci.iloc[:, 1].rename("upper_ci"),
    )


def holt_winters_forecast(series: pd.Series, horizon: int) -> ForecastOutput:
    """Holt-Winters trend-seasonality based forecast."""
    if ExponentialSmoothing is None:
        raise ImportError("statsmodels is required for holt_winters_forecast")

    model = ExponentialSmoothing(series, trend="add", seasonal=None)
    fit = model.fit(optimized=True)
    pred = fit.forecast(horizon).rename("holt_winters")
    return ForecastOutput(prediction=pred)


def prophet_forecast(series: pd.Series, horizon: int) -> ForecastOutput:
    """Facebook Prophet forecast with confidence intervals."""
    if Prophet is None:
        raise ImportError("prophet is required for prophet_forecast")

    frame = pd.DataFrame({"ds": series.index, "y": series.values})
    model = Prophet(daily_seasonality=False)
    model.fit(frame)
    future = model.make_future_dataframe(periods=horizon, freq="B")
    forecast = model.predict(future).tail(horizon)
    return ForecastOutput(
        prediction=pd.Series(forecast["yhat"].values, index=forecast["ds"], name="prophet"),
        lower_ci=pd.Series(forecast["yhat_lower"].values, index=forecast["ds"], name="lower_ci"),
        upper_ci=pd.Series(forecast["yhat_upper"].values, index=forecast["ds"], name="upper_ci"),
    )


def lstm_forecast(
    train_array: np.ndarray,
    horizon: int,
    sequence_length: int = 30,
    epochs: int = 20,
) -> np.ndarray:
    """Stacked LSTM with dropout and early stopping for sequence forecasting."""
    if Sequential is None:
        raise ImportError("tensorflow is required for lstm_forecast")

    x_train, y_train = [], []
    for i in range(sequence_length, len(train_array)):
        x_train.append(train_array[i - sequence_length : i])
        y_train.append(train_array[i])
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        validation_split=0.2,
        verbose=0,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
    )

    sequence = train_array[-sequence_length:]
    preds = []
    for _ in range(horizon):
        pred = model.predict(sequence[np.newaxis, :, :], verbose=0)
        preds.append(float(pred[0, 0]))
        sequence = np.vstack([sequence[1:], pred])
    return np.array(preds)


def ensemble_mean(forecasts: dict[str, pd.Series]) -> pd.Series:
    """Compute a simple ensemble mean across aligned model predictions."""
    frame = pd.concat(forecasts.values(), axis=1)
    return frame.mean(axis=1).rename("ensemble")
