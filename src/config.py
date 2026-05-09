"""Configuration loading and strongly-typed config objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    """Data configuration values."""

    tickers: list[str]
    start_date: str
    end_date: str
    test_months: int


@dataclass(frozen=True)
class ForecastConfig:
    """Forecasting configuration values."""

    live_horizon_days: int
    validation_horizon_days: int


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    project_name: str
    random_seed: int
    data: DataConfig
    forecasting: ForecastConfig


DEFAULT_CONFIG_PATH = Path("config/config.yaml")


def load_yaml(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load and validate project configuration."""
    raw = load_yaml(path)
    data = DataConfig(**raw["data"])
    forecasting = ForecastConfig(**raw["forecasting"])
    return AppConfig(
        project_name=raw["project_name"],
        random_seed=int(raw["random_seed"]),
        data=data,
        forecasting=forecasting,
    )
