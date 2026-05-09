"""Main CLI entrypoint for capstone pipeline orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import load_config
from src.data_fetch import DataFetcher
from src.utils import set_global_seed, setup_logger


def main() -> None:
    """Run initial pipeline setup and data collection."""
    logger = setup_logger("main")
    config = load_config()
    set_global_seed(config.random_seed)

    logger.info("Starting project: %s", config.project_name)
    fetcher = DataFetcher(out_dir=Path("data/raw"))
    result = fetcher.fetch(
        tickers=config.data.tickers,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
    )

    summary = {
        "fetched_tickers": sorted(result.data.keys()),
        "invalid_tickers": result.invalid_tickers,
        "test_months_reserved": config.data.test_months,
        "live_horizon_days": config.forecasting.live_horizon_days,
        "validation_horizon_days": config.forecasting.validation_horizon_days,
    }
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    Path("outputs/reports/run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    logger.info("Run summary saved to outputs/reports/run_summary.json")


if __name__ == "__main__":
    main()
