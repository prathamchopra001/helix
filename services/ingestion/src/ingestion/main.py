"""
Ingestion entrypoint — callable from Airflow PythonOperator or CLI.

Usage:
    python -m ingestion.main              # fetch 1d for all default tickers
    python -m ingestion.main --period 2y  # full backfill
"""

import os
import uuid
from datetime import UTC, datetime

from ingestion.fetchers.yahoo_finance import fetch_ohlcv
from ingestion.writers.minio_writer import archive_ohlcv
from ingestion.writers.postgres_writer import upsert_ohlcv
from shared.config import BaseConfig
from shared.logging import bind_request_context, configure_logging, get_logger
from shared.storage import StorageClient

log = get_logger(__name__)


def _get_dsn() -> str:
    """Build a psycopg2 DSN from individual POSTGRES_* env vars."""
    return (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ['POSTGRES_HOST']}:{os.environ.get('POSTGRES_PORT', '5432')}"
        f"/{os.environ['POSTGRES_DB']}"
    )


DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "TSLA",
    "SPY",
    "QQQ",
    "BTC-USD",
    "ETH-USD",
    "NVDA",
    "AMZN",
    "META",
]


class IngestionConfig(BaseConfig):
    ingestion_tickers: str = ",".join(DEFAULT_TICKERS)

    @property
    def tickers(self) -> list[str]:
        return [t.strip() for t in self.ingestion_tickers.split(",")]


def run_ingestion(
    tickers: list[str] | None = None,
    period: str = "1d",
    correlation_id: str | None = None,
) -> dict[str, int]:
    """
    Fetch OHLCV data and write to PostgreSQL + MinIO for each ticker.

    Returns: dict mapping ticker → rows upserted (0 on failure for that ticker).
    Failures per-ticker are logged and swallowed so one bad ticker doesn't
    abort the rest of the run.
    """
    configure_logging("ingestion")
    cfg = IngestionConfig()
    correlation_id = correlation_id or str(uuid.uuid4())
    bind_request_context(correlation_id)

    tickers = tickers or cfg.tickers
    date_label = datetime.now(UTC).strftime("%Y-%m-%d")

    storage = StorageClient(
        endpoint=cfg.minio_endpoint,
        access_key=cfg.minio_access_key,
        secret_key=cfg.minio_secret_key,
        secure=cfg.minio_secure,
    )
    storage.ensure_bucket(cfg.minio_bucket_raw)

    results: dict[str, int] = {}

    for ticker in tickers:
        try:
            df = fetch_ohlcv(ticker, period=period, correlation_id=correlation_id)
            rows = upsert_ohlcv(df, _get_dsn(), correlation_id=correlation_id)
            archive_ohlcv(
                df,
                storage,
                cfg.minio_bucket_raw,
                ticker=ticker,
                date_label=date_label,
                correlation_id=correlation_id,
            )
            results[ticker] = rows
        except Exception as exc:
            log.error("ticker_ingestion_failed", ticker=ticker, error=str(exc))
            results[ticker] = 0

    total = sum(results.values())
    log.info("ingestion_complete", total_rows=total, tickers=list(results.keys()))
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default="1d")
    parser.add_argument("--tickers", default=None, help="Comma-separated list")
    args = parser.parse_args()

    ticker_list = [t.strip() for t in args.tickers.split(",")] if args.tickers else None
    run_ingestion(tickers=ticker_list, period=args.period)
