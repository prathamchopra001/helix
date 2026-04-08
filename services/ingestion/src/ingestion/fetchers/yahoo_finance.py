"""Yahoo Finance OHLCV fetcher with exponential backoff retry."""

import time

import pandas as pd
import yfinance as yf

from shared.logging import get_logger

log = get_logger(__name__)


def fetch_ohlcv(
    ticker: str,
    period: str = "1d",
    interval: str = "1d",
    max_retries: int = 3,
    correlation_id: str = "",
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a single ticker from Yahoo Finance.

    Retries up to max_retries times with exponential backoff.
    Raises on permanent failure.
    """
    bound_log = log.bind(ticker=ticker, correlation_id=correlation_id)

    for attempt in range(1, max_retries + 1):
        try:
            bound_log.info("fetching_ohlcv", attempt=attempt, period=period)
            obj = yf.Ticker(ticker)
            df = obj.history(period=period, interval=interval, auto_adjust=True)

            if df.empty:
                raise ValueError(f"Yahoo Finance returned empty DataFrame for {ticker}")

            df = df.reset_index()
            df.columns = [str(c).lower() for c in df.columns]
            df["ticker"] = ticker

            # Keep only needed columns; drop dividends / stock splits if present
            keep = ["date", "open", "high", "low", "close", "volume", "ticker"]
            df = df[[c for c in keep if c in df.columns]]

            bound_log.info("fetched_ohlcv", rows=len(df))
            return df

        except Exception as exc:
            wait = 2 ** (attempt - 1)
            if attempt == max_retries:
                bound_log.error(
                    "fetch_failed_permanently",
                    error=str(exc),
                    attempts=max_retries,
                )
                raise
            bound_log.warning(
                "fetch_failed_retrying",
                attempt=attempt,
                error=str(exc),
                retry_in_seconds=wait,
            )
            time.sleep(wait)

    raise RuntimeError("unreachable")  # satisfies type checkers
