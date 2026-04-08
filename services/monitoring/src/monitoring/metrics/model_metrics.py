"""
Model KPI computation.

Joins predictions.inference_log with features.feature_vectors on
(ticker, timestamp) to get ground truth labels for predictions.

For each window (7d, 30d):
  - Fetch predictions made in that window
  - Match to ground truth label from feature_vectors
  - Compute F1, precision, recall, AUC
  - Write to monitoring.model_metrics
"""
from __future__ import annotations

import psycopg2
import psycopg2.extras
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from shared.logging import get_logger

log = get_logger(__name__)

_JOIN_SQL = """
    SELECT p.score, p.label AS predicted, fv.label AS actual
    FROM predictions.inference_log p
    JOIN features.feature_vectors fv
        ON p.ticker = fv.ticker
        AND DATE(p.created_at) = DATE(fv.timestamp)
    WHERE p.created_at >= NOW() - INTERVAL '{window_days} days'
"""

_UPSERT_METRIC_SQL = """
    INSERT INTO monitoring.model_metrics
        (metric_name, metric_value, window_days, window_start, window_end,
         model_version, computed_at)
    VALUES (%s, %s, %s, NOW() - INTERVAL '{window_days} days', NOW(), %s, NOW())
"""


def compute_model_metrics(
    dsn: str,
    window_days: int = 7,
    correlation_id: str = "",
) -> dict[str, float]:
    """
    Compute rolling F1/precision/recall/AUC for the given window.

    Returns a dict of metric_name → value, or an empty dict if there
    is insufficient matched data (< 10 rows).
    """
    log.info(
        "model_metrics_start",
        window_days=window_days,
        correlation_id=correlation_id,
    )

    conn = psycopg2.connect(dsn)
    try:
        sql = _JOIN_SQL.format(window_days=window_days)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()

        if len(rows) < 10:
            log.warning(
                "model_metrics_insufficient_data",
                matched_rows=len(rows),
                required=10,
                window_days=window_days,
                correlation_id=correlation_id,
            )
            return {}

        scores = [float(r["score"]) for r in rows]
        predicted = [int(r["predicted"]) for r in rows]
        actual = [int(r["actual"]) for r in rows]

        f1 = float(f1_score(actual, predicted, zero_division=0))
        precision = float(precision_score(actual, predicted, zero_division=0))
        recall = float(recall_score(actual, predicted, zero_division=0))

        # AUC requires both classes present in actual labels
        unique_labels = set(actual)
        if len(unique_labels) >= 2:
            auc = float(roc_auc_score(actual, scores))
        else:
            auc = 0.0
            log.warning(
                "model_metrics_auc_skipped",
                reason="only_one_class_in_actuals",
                window_days=window_days,
                correlation_id=correlation_id,
            )

        result: dict[str, float] = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "auc": auc,
        }

        # Determine the model version from the most recent prediction in the window
        model_version = "unknown"
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT model_version
                FROM predictions.inference_log
                WHERE created_at >= NOW() - INTERVAL %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (f"{window_days} days",),
            )
            version_row = cur.fetchone()
            if version_row:
                model_version = version_row["model_version"]

        upsert_sql = _UPSERT_METRIC_SQL.format(window_days=window_days)
        with conn.cursor() as cur:
            for metric_name, metric_value in result.items():
                cur.execute(
                    upsert_sql,
                    (metric_name, metric_value, window_days, model_version),
                )
        conn.commit()

        log.info(
            "model_metrics_complete",
            window_days=window_days,
            matched_rows=len(rows),
            f1=f1,
            precision=precision,
            recall=recall,
            auc=auc,
            model_version=model_version,
            correlation_id=correlation_id,
        )

        return result

    except psycopg2.Error as exc:
        log.error(
            "model_metrics_db_error",
            error=str(exc),
            window_days=window_days,
            correlation_id=correlation_id,
        )
        raise
    finally:
        conn.close()
