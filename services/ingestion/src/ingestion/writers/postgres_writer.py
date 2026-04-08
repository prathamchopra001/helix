"""PostgreSQL writer — upserts OHLCV rows into raw.ohlcv_data via psycopg2."""

from datetime import UTC, datetime

import pandas as pd
import psycopg2
import psycopg2.extras

from shared.logging import get_logger

log = get_logger(__name__)

_UPSERT_SQL = """
    INSERT INTO raw.ohlcv_data
        (ticker, timestamp, open, high, low, close, volume, ingested_at)
    VALUES %s
    ON CONFLICT (ticker, timestamp) DO UPDATE SET
        open        = EXCLUDED.open,
        high        = EXCLUDED.high,
        low         = EXCLUDED.low,
        close       = EXCLUDED.close,
        volume      = EXCLUDED.volume,
        ingested_at = EXCLUDED.ingested_at
"""


def _to_psycopg2_dsn(database_url: str) -> str:
    """Convert a SQLAlchemy-style URL to a plain psycopg2 DSN."""
    dsn = database_url
    for prefix in ("postgresql+asyncpg://", "postgresql+psycopg2://"):
        dsn = dsn.replace(prefix, "postgresql://")
    # Strip query params (e.g. ?ssl=disable — psycopg2 doesn't need them for local)
    return dsn.split("?")[0]


def upsert_ohlcv(
    df: pd.DataFrame,
    database_url: str,
    correlation_id: str = "",
) -> int:
    """
    Upsert OHLCV rows from a DataFrame into raw.ohlcv_data.

    Uses psycopg2 execute_values for efficient batch inserts.
    Safe to call multiple times — ON CONFLICT updates existing rows.
    Returns number of rows processed.
    """
    now = datetime.now(UTC)
    records: list[tuple] = []

    for _, row in df.iterrows():
        ts = row.get("date") or row.get("datetime") or row.get("timestamp")
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()

        records.append(
            (
                str(row["ticker"]),
                ts,
                float(row.get("open", 0.0)),
                float(row.get("high", 0.0)),
                float(row.get("low", 0.0)),
                float(row.get("close", 0.0)),
                int(row.get("volume", 0)),
                now,
            )
        )

    if not records:
        log.warning("upsert_skipped_empty_df", correlation_id=correlation_id)
        return 0

    dsn = _to_psycopg2_dsn(database_url)
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, _UPSERT_SQL, records)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    log.info("upserted_ohlcv", rows=len(records), correlation_id=correlation_id)
    return len(records)
