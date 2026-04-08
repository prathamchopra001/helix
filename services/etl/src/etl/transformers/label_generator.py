"""
Generate binary anomaly labels for each row.

An anomaly is defined as a day where BOTH conditions hold:
  1. The 20-day z-score of close price exceeds the threshold (default ±2.5)
     — price moved unusually far from its recent average
  2. The 5-day volume ratio exceeds the threshold (default 2.0)
     — volume was at least 2x the recent average

Both conditions must be true simultaneously. A big price move on normal volume
could just be a news event. Unusual volume on a flat day could be rebalancing.
The combination is the signal.

Thresholds are env-configurable so they can be tuned without changing code.
"""

import os

import pandas as pd

from shared.logging import get_logger

log = get_logger(__name__)

_Z_THRESHOLD = float(os.getenv("ANOMALY_Z_SCORE_THRESHOLD", "2.5"))
_VOL_THRESHOLD = float(os.getenv("ANOMALY_VOLUME_RATIO_THRESHOLD", "2.0"))


def add_anomaly_label(df: pd.DataFrame, correlation_id: str = "") -> pd.DataFrame:
    """
    Add a binary 'label' column: 1 = anomaly, 0 = normal.

    Requires columns: z_score_20d, volume_ratio_5d (produced by rolling_stats).
    """
    df = df.copy()

    price_anomaly = df["z_score_20d"].abs() > _Z_THRESHOLD
    volume_spike = df["volume_ratio_5d"] > _VOL_THRESHOLD
    df["label"] = (price_anomaly | volume_spike).astype(int)

    anomaly_count = df["label"].sum()
    anomaly_pct = anomaly_count / len(df) * 100

    log.info(
        "labels_generated",
        total_rows=len(df),
        anomaly_rows=int(anomaly_count),
        anomaly_pct=round(anomaly_pct, 2),
        z_threshold=_Z_THRESHOLD,
        vol_threshold=_VOL_THRESHOLD,
        correlation_id=correlation_id,
    )
    return df
