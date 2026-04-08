"""MinIO writer — archives raw OHLCV CSVs. Idempotent: overwrites the same key."""
from datetime import datetime, timezone

import pandas as pd

from shared.logging import get_logger
from shared.storage import StorageClient

log = get_logger(__name__)


def archive_ohlcv(
    df: pd.DataFrame,
    storage: StorageClient,
    bucket: str,
    ticker: str,
    date_label: str = "",
    correlation_id: str = "",
) -> str:
    """
    Upload a DataFrame as CSV to MinIO at ohlcv/{ticker}/{date_label}.csv.

    Overwrites if the object already exists (idempotent).
    Returns the object key.
    """
    if not date_label:
        date_label = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    key = f"ohlcv/{ticker}/{date_label}.csv"
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    storage.upload_bytes(bucket, key, csv_bytes, content_type="text/csv")
    log.info(
        "archived_ohlcv",
        bucket=bucket,
        key=key,
        rows=len(df),
        correlation_id=correlation_id,
    )
    return key
