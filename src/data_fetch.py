"""Institutional-style market data ingestion from Yahoo Finance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import sleep

import pandas as pd
import yfinance as yf

from src.utils import ensure_dir, setup_logger


@dataclass
class FetchResult:
    """Represents the fetch operation outputs."""

    data: dict[str, pd.DataFrame]
    invalid_tickers: list[str]


class DataFetcher:
    """Fetch OHLCV data from Yahoo Finance with retries and validation."""

    def __init__(self, out_dir: str | Path = "data/raw", retries: int = 3) -> None:
        self.out_dir = ensure_dir(out_dir)
        self.retries = retries
        self.logger = setup_logger(__name__)

    def _download_once(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        frame = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=False,
        )
        if frame.empty:
            raise ValueError(f"Ticker {ticker} returned no data")

        frame = frame[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
        frame.index.name = "Date"
        return frame

    def fetch(self, tickers: list[str], start_date: str, end_date: str) -> FetchResult:
        """Fetch and persist OHLCV frames for all requested tickers."""
        data: dict[str, pd.DataFrame] = {}
        invalid_tickers: list[str] = []

        for idx, ticker in enumerate(tickers, start=1):
            self.logger.info("[%s/%s] Fetching %s", idx, len(tickers), ticker)
            success = False
            for attempt in range(1, self.retries + 1):
                try:
                    frame = self._download_once(ticker, start_date, end_date)
                    frame = frame.ffill().bfill()
                    file_path = self.out_dir / f"{ticker.replace('.', '_')}.csv"
                    frame.to_csv(file_path)
                    data[ticker] = frame
                    success = True
                    break
                except Exception as exc:
                    self.logger.warning(
                        "Attempt %s failed for %s: %s", attempt, ticker, exc
                    )
                    sleep(1)

            if not success:
                self.logger.error("Invalid or unavailable ticker: %s", ticker)
                invalid_tickers.append(ticker)

        return FetchResult(data=data, invalid_tickers=invalid_tickers)
