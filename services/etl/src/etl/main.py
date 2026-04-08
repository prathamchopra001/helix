"""
ETL orchestration — ties all transformers together for one or more tickers.

What run_etl() does end to end:
  1. Reads raw OHLCV data from postgres for the given tickers
  2. Validates each ticker's data quality (fails fast if bad)
  3. Applies technical indicators (RSI, MACD, Bollinger Bands, ATR, OBV)
  4. Applies rolling stats (z-scores, log returns, volume ratios)
  5. Generates anomaly labels
  6. Writes feature vectors to features.feature_vectors

Callable from Airflow PythonOperator or directly from a script.
"""
import os

import pandas as pd
import psycopg2
import psycopg2.extras

from etl.feature_store import upsert_features
from etl.transformers.label_generator import add_anomaly_label
from etl.transformers.rolling_stats import add_rolling_stats
from etl.transformers.technical_indicators import add_technical_indicators
from etl.validators.expectations import validate_ohlcv
from shared.logging import get_logger

log = get_logger(__name__)

_FETCH_SQL = """
    SELECT ticker, timestamp, open, high, low, close, volume
    FROM raw.ohlcv_data
    WHERE ticker = %s
    ORDER BY timestamp ASC
"""


def _get_dsn() -> str:
    """Build a psycopg2 DSN from individual POSTGRES_* env vars."""
    return (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ['POSTGRES_HOST']}:{os.environ.get('POSTGRES_PORT', '5432')}"
        f"/{os.environ['POSTGRES_DB']}"
    )


def _fetch_raw(ticker: str, dsn: str) -> pd.DataFrame:
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_FETCH_SQL, (ticker,))
            rows = cur.fetchall()
    finally:
        conn.close()
    df = pd.DataFrame(rows)
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


def run_etl(
    tickers: list[str] | None = None,
    correlation_id: str = "",
) -> dict[str, int]:
    """
    Run the full ETL pipeline for the given tickers.

    Returns a dict of {ticker: rows_written}.
    If tickers is None, reads ETL_TICKERS env var (comma-separated).
    """
    dsn = _get_dsn()

    if tickers is None:
        raw = os.getenv("ETL_TICKERS", "AAPL,MSFT,GOOGL,TSLA,SPY,QQQ,BTC-USD,ETH-USD,NVDA,AMZN,META")
        tickers = [t.strip() for t in raw.split(",")]

    results: dict[str, int] = {}

    for ticker in tickers:
        log.info("etl_ticker_start", ticker=ticker, correlation_id=correlation_id)
        try:
            df = _fetch_raw(ticker, dsn)
            if df.empty:
                log.warning("etl_no_raw_data", ticker=ticker, correlation_id=correlation_id)
                results[ticker] = 0
                continue

            validate_ohlcv(df, ticker=ticker, correlation_id=correlation_id)
            df = add_technical_indicators(df, correlation_id=correlation_id)
            df = add_rolling_stats(df, correlation_id=correlation_id)
            df = add_anomaly_label(df, correlation_id=correlation_id)

            rows = upsert_features(df, dsn=dsn, correlation_id=correlation_id)
            results[ticker] = rows

            log.info("etl_ticker_done", ticker=ticker, rows=rows, correlation_id=correlation_id)

        except Exception as exc:
            log.error(
                "etl_ticker_failed",
                ticker=ticker,
                error=str(exc),
                correlation_id=correlation_id,
            )
            results[ticker] = 0

    total = sum(results.values())
    log.info("etl_complete", total_rows=total, tickers=tickers, correlation_id=correlation_id)
    return results
