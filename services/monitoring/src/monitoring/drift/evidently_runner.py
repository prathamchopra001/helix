"""
Data drift detection using Population Stability Index (PSI).

PSI measures how much the distribution of a feature has shifted between
training (reference) and recent inference inputs (current).

PSI interpretation:
  < 0.1  — no significant drift, model is stable
  0.1-0.2 — moderate drift, worth monitoring
  > 0.2  — significant drift, consider retraining

PSI formula (per feature):
  PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))

Why not use Evidently?
  Evidently 0.7+ pulls in dynaconf and litestar which conflict with
  Airflow's Celery worker hostname resolution at import time. Manual PSI
  is simple numpy and gives us exactly what we need without the 500MB dep.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import psycopg2

from shared.logging import get_logger

log = get_logger(__name__)

N_BINS = 10
MIN_CURRENT_ROWS = 30


def _compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = N_BINS) -> float:
    """
    Compute PSI between reference and current distributions.
    Handles edge cases: empty arrays, zero buckets, single-value columns.
    """
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    # Use reference percentiles as bin edges so buckets are evenly populated
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(reference, percentiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0  # no variance — no drift possible

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    eps = 1e-6
    ref_pct = np.maximum(ref_counts / len(reference), eps)
    cur_pct = np.maximum(cur_counts / len(current), eps)

    return round(float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))), 6)


def _fetch_reference(dsn: str) -> pd.DataFrame:
    """Fetch up to 1000 random training rows from feature_vectors."""
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT features FROM features.feature_vectors WHERE split = 'train' ORDER BY RANDOM() LIMIT 1000"
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame()

    records = []
    for (feat_json,) in rows:
        feat = feat_json if isinstance(feat_json, dict) else json.loads(feat_json)
        records.append({k: float(v) for k, v in feat.items()})
    return pd.DataFrame(records)


def _fetch_current(dsn: str) -> pd.DataFrame:
    """Fetch features for predictions made in the last 7 days."""
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT fv.features
                FROM predictions.inference_log p
                JOIN features.feature_vectors fv
                  ON p.ticker = fv.ticker
                 AND DATE(p.created_at) = DATE(fv.timestamp)
                WHERE p.created_at >= NOW() - INTERVAL '7 days'
                LIMIT 2000
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame()

    records = []
    for (feat_json,) in rows:
        feat = feat_json if isinstance(feat_json, dict) else json.loads(feat_json)
        records.append({k: float(v) for k, v in feat.items()})
    return pd.DataFrame(records)


def _store_report(dsn: str, psi_scores: dict[str, float], triggered_retraining: bool) -> None:
    report = {
        "psi_scores": psi_scores,
        "n_features_drifted": sum(1 for v in psi_scores.values() if v > 0.2),
    }
    conn = psycopg2.connect(dsn)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO monitoring.drift_reports (report, triggered_retraining) VALUES (%s, %s)",
                (json.dumps(report), triggered_retraining),
            )
    finally:
        conn.close()


def compute_drift(dsn: str, correlation_id: str = "") -> dict[str, float]:
    """
    Compute PSI for every feature. Returns {feature_name: psi_score}.
    Returns empty dict if not enough current data.
    """
    log.info("drift_detection_start", correlation_id=correlation_id)

    reference_df = _fetch_reference(dsn)
    if reference_df.empty:
        log.warning("drift_no_reference_data", correlation_id=correlation_id)
        return {}

    current_df = _fetch_current(dsn)
    if len(current_df) < MIN_CURRENT_ROWS:
        log.info(
            "drift_insufficient_current_data",
            rows=len(current_df),
            min_required=MIN_CURRENT_ROWS,
            correlation_id=correlation_id,
        )
        _store_report(dsn, {}, triggered_retraining=False)
        return {}

    common_cols = sorted(set(reference_df.columns) & set(current_df.columns))
    if not common_cols:
        log.warning("drift_no_common_columns", correlation_id=correlation_id)
        return {}

    psi_scores: dict[str, float] = {}
    for col in common_cols:
        ref_vals = reference_df[col].dropna().values
        cur_vals = current_df[col].dropna().values
        psi_scores[col] = _compute_psi(ref_vals, cur_vals)

    drifted = {k: v for k, v in psi_scores.items() if v > 0.2}
    log.info(
        "drift_detection_complete",
        total_features=len(psi_scores),
        drifted_features=len(drifted),
        max_psi=round(max(psi_scores.values(), default=0.0), 4),
        correlation_id=correlation_id,
    )

    _store_report(dsn, psi_scores, triggered_retraining=False)
    return psi_scores
