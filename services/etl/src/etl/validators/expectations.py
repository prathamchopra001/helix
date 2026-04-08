"""
Data quality validation using Great Expectations.

What this does:
  Runs a suite of checks on the raw OHLCV DataFrame BEFORE any transformation.
  If any check fails, it raises DataQualityError, which causes the Airflow task
  (and therefore the DAG run) to fail with a clear message.

Why this order matters:
  Validating BEFORE transforming means we never silently propagate bad data into
  the feature store. A single null close price would corrupt every z-score and
  indicator downstream. Catching it here gives a clear error instead of silent
  NaN pollution.

Checks:
  - No nulls in ticker, timestamp, open, high, low, close, volume
  - volume > 0 (zero volume means the exchange had no trades — bad data)
  - close > 0 (sanity check — price can't be negative or zero)
  - timestamp is unique per ticker (no duplicate rows)
"""
import pandas as pd

from shared.logging import get_logger

log = get_logger(__name__)


class DataQualityError(Exception):
    """Raised when the data quality checks fail. Fails the Airflow task."""


def validate_ohlcv(df: pd.DataFrame, ticker: str, correlation_id: str = "") -> None:
    """
    Validate raw OHLCV data for a single ticker.

    Raises DataQualityError with a descriptive message if any check fails.
    """
    failures: list[str] = []

    required_cols = ["ticker", "timestamp", "open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            failures.append(f"missing column: {col}")
        elif df[col].isnull().any():
            null_count = int(df[col].isnull().sum())
            failures.append(f"nulls in {col}: {null_count} rows")

    if "volume" in df.columns and not df["volume"].isnull().any():
        zero_vol = int((df["volume"] <= 0).sum())
        if zero_vol > 0:
            failures.append(f"volume <= 0: {zero_vol} rows")

    if "close" in df.columns and not df["close"].isnull().any():
        bad_close = int((df["close"] <= 0).sum())
        if bad_close > 0:
            failures.append(f"close <= 0: {bad_close} rows")

    if "timestamp" in df.columns:
        dup_count = int(df["timestamp"].duplicated().sum())
        if dup_count > 0:
            failures.append(f"duplicate timestamps: {dup_count} rows")

    if failures:
        msg = f"Data quality checks failed for {ticker}: {'; '.join(failures)}"
        log.error(
            "data_quality_failed",
            ticker=ticker,
            failures=failures,
            correlation_id=correlation_id,
        )
        raise DataQualityError(msg)

    log.info(
        "data_quality_passed",
        ticker=ticker,
        rows=len(df),
        correlation_id=correlation_id,
    )
