"""
Write engineered feature vectors to features.feature_vectors.

What this does:
  Takes a fully transformed DataFrame (with indicators, rolling stats, and labels)
  and writes it to PostgreSQL. Each row becomes one feature vector: the ticker,
  the date, all numeric features packed into a JSONB dict, the binary label, and
  a train/val/test split assignment.

Split assignment (time-based, never random):
  The data is sorted by timestamp. The earliest 70% of rows go to 'train', the
  next 15% to 'val', the last 15% to 'test'. This mirrors real-world deployment
  — you train on the past, validate on more-recent past, test on the most recent
  data. Random splitting would leak future information into training.

Why JSONB for features:
  The feature set will grow across phases (Phase 4 adds indicators, Phase 8 adds
  drift signals). JSONB lets us add new features without Alembic migrations every
  time. The model loader reads the JSONB dict and reconstructs the feature tensor.

Upsert semantics:
  ON CONFLICT (ticker, timestamp) DO UPDATE — safe to re-run if ETL is triggered
  multiple times on the same day.
"""
import json
from datetime import datetime, timezone

import pandas as pd
import psycopg2
import psycopg2.extras

from shared.logging import get_logger

log = get_logger(__name__)

_FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "rsi_14",
    "macd", "macd_signal", "macd_histogram",
    "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct",
    "atr_14",
    "obv",
    "log_return",
    "z_score_5d", "z_score_10d", "z_score_20d", "z_score_60d",
    "volume_ratio_5d", "volume_ratio_10d", "volume_ratio_20d",
    "day_gap",
    # Momentum and volatility features added for v10+ models
    "return_5d", "return_20d",   # multi-day price momentum
    "hl_range",                   # intraday high/low spread
    "overnight_gap",              # open vs prev close (news events)
]

_UPSERT_SQL = """
    INSERT INTO features.feature_vectors
        (ticker, timestamp, features, label, split, created_at)
    VALUES %s
    ON CONFLICT (ticker, timestamp) DO UPDATE SET
        features   = EXCLUDED.features,
        label      = EXCLUDED.label,
        split      = EXCLUDED.split,
        created_at = EXCLUDED.created_at
"""


def _assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Assign train/val/test split based on timestamp order (never random)."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    df["split"] = "test"
    df.loc[:train_end - 1, "split"] = "train"
    df.loc[train_end:val_end - 1, "split"] = "val"
    return df


def upsert_features(
    df: pd.DataFrame,
    dsn: str,
    correlation_id: str = "",
) -> int:
    """
    Upsert feature vectors for a single ticker into features.feature_vectors.

    Returns number of rows written.
    """
    df = _assign_splits(df)
    now = datetime.now(timezone.utc)

    available_cols = [c for c in _FEATURE_COLS if c in df.columns]
    records: list[tuple] = []

    for _, row in df.iterrows():
        features_dict = {col: float(row[col]) for col in available_cols}
        records.append((
            str(row["ticker"]),
            row["timestamp"],
            json.dumps(features_dict),
            int(row["label"]),
            str(row["split"]),
            now,
        ))

    if not records:
        log.warning("feature_upsert_skipped_empty", correlation_id=correlation_id)
        return 0

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

    split_counts = df["split"].value_counts().to_dict()
    anomaly_pct = round(df["label"].mean() * 100, 2)

    log.info(
        "features_upserted",
        rows=len(records),
        splits=split_counts,
        anomaly_pct=anomaly_pct,
        correlation_id=correlation_id,
    )
    return len(records)
