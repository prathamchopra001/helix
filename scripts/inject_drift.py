"""
Synthetic drift injection script for testing the monitoring pipeline.

Usage:
  # Step 1 — inject enough predictions so drift monitor has >= 30 current rows
  python scripts/inject_drift.py --mode predictions

  # Step 2 — additionally shift feature distributions to simulate real drift
  python scripts/inject_drift.py --mode drift --drift-factor 2.5

  # Step 3 — trigger the monitoring DAG manually (or wait for @daily)
  python scripts/inject_drift.py --mode trigger

How it works:
  The monitoring drift query joins predictions.inference_log with
  features.feature_vectors on (ticker, date). This script inserts
  fake prediction rows backdated to match existing feature_vector timestamps.
  In --mode drift, it also inserts new feature_vectors with shifted
  distributions to guarantee PSI > 0.2 for most features.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import timedelta

import psycopg2
import requests


def _get_dsn() -> str:
    return (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ.get('POSTGRES_HOST', 'localhost')}:{os.environ.get('POSTGRES_PORT', '5432')}"
        f"/{os.environ['POSTGRES_DB']}"
    )


def inject_predictions(dsn: str) -> int:
    """
    Insert fake predictions for every feature_vector row in the last 7 days.
    Sets created_at = fv.timestamp so the drift monitor's join resolves.
    Returns the number of rows inserted.
    """
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ticker, timestamp
                FROM features.feature_vectors
                WHERE timestamp >= NOW() - INTERVAL '7 days'
                ORDER BY timestamp DESC
            """)
            rows = cur.fetchall()

        if not rows:
            print("No feature_vectors found in last 7 days — run seed_data.sh first")
            return 0

        inserted = 0
        with conn:
            with conn.cursor() as cur:
                for ticker, ts in rows:
                    cur.execute(
                        """
                        INSERT INTO predictions.inference_log
                            (request_id, ticker, timestamp, features, score, label,
                             model_version, backend, latency_ms, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (request_id) DO NOTHING
                        """,
                        (
                            str(uuid.uuid4()),
                            ticker,
                            ts,
                            json.dumps({}),
                            0.72 if ticker in {"TSLA", "BTC-USD", "ETH-USD"} else 0.25,
                            1 if ticker in {"TSLA", "BTC-USD", "ETH-USD"} else 0,
                            "synthetic",
                            "pytorch",
                            10.5,
                            ts,  # backdate created_at to match fv.timestamp date
                        ),
                    )
                    inserted += 1
        print(f"Inserted {inserted} synthetic predictions")
        return inserted
    finally:
        conn.close()


def inject_drifted_features(dsn: str, drift_factor: float = 2.5) -> int:
    """
    Clone the most recent feature_vectors and scale all numeric features
    by drift_factor, inserting them with today's timestamp.
    This guarantees PSI > 0.2 for most features.
    """
    import numpy as np
    from datetime import datetime, timezone

    conn = psycopg2.connect(dsn)
    try:
        # Grab one representative sample per ticker from training data
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ticker, features
                FROM features.feature_vectors
                WHERE split = 'train'
                ORDER BY RANDOM()
                LIMIT 200
            """)
            source_rows = cur.fetchall()

        if not source_rows:
            print("No training feature_vectors found")
            return 0

        now = datetime.now(tz=timezone.utc)
        inserted = 0

        with conn:
            with conn.cursor() as cur:
                for ticker, feat_json in source_rows:
                    feat = feat_json if isinstance(feat_json, dict) else json.loads(feat_json)
                    # Shift features: multiply by drift_factor + add Gaussian noise
                    drifted = {}
                    for k, v in feat.items():
                        val = float(v) * drift_factor
                        val += np.random.normal(0, abs(val) * 0.1 + 1e-6)
                        drifted[k] = round(val, 8)

                    cur.execute(
                        """
                        INSERT INTO features.feature_vectors
                            (ticker, timestamp, features, label, split)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                        """,
                        (ticker, now, json.dumps(drifted), 1, 'val'),
                    )
                    inserted += 1

                # Also insert matching predictions for today
                for ticker, _ in source_rows[:50]:
                    cur.execute(
                        """
                        INSERT INTO predictions.inference_log
                            (request_id, ticker, timestamp, features, score, label,
                             model_version, backend, latency_ms, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (request_id) DO NOTHING
                        """,
                        (
                            str(uuid.uuid4()), ticker, now, json.dumps({}),
                            0.85, 1, "synthetic", "pytorch", 12.0, now,
                        ),
                    )

        print(f"Inserted {inserted} drifted feature vectors (factor={drift_factor})")
        return inserted
    finally:
        conn.close()


def trigger_monitoring_dag() -> None:
    """
    Trigger monitoring_dag via docker compose exec (avoids REST auth complexity).
    Airflow's default API auth is session-based, so CLI is more reliable.
    """
    import subprocess
    result = subprocess.run(
        [
            "docker", "compose", "exec", "airflow-webserver",
            "airflow", "dags", "trigger", "monitoring_dag",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("Triggered monitoring_dag successfully")
        print("Monitor at: http://localhost/airflow/dags/monitoring_dag/grid")
    else:
        # Also try unpause first in case it's paused
        subprocess.run(
            ["docker", "compose", "exec", "airflow-webserver",
             "airflow", "dags", "unpause", "monitoring_dag"],
            capture_output=True,
        )
        result2 = subprocess.run(
            ["docker", "compose", "exec", "airflow-webserver",
             "airflow", "dags", "trigger", "monitoring_dag"],
            capture_output=True,
            text=True,
        )
        if result2.returncode == 0:
            print("Triggered monitoring_dag successfully")
        else:
            print(f"Trigger failed: {result2.stderr[:200]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject synthetic drift for monitoring tests")
    parser.add_argument(
        "--mode",
        choices=["predictions", "drift", "trigger", "all"],
        default="all",
        help=(
            "predictions: insert fake prediction logs only\n"
            "drift: insert predictions + drifted feature vectors\n"
            "trigger: trigger monitoring DAG via Airflow API\n"
            "all: predictions + drift + trigger"
        ),
    )
    parser.add_argument(
        "--drift-factor",
        type=float,
        default=2.5,
        help="Scale factor for drifted feature values (default: 2.5)",
    )
    args = parser.parse_args()

    required_env = ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"]
    missing = [v for v in required_env if not os.environ.get(v)]
    if missing:
        print(f"Missing env vars: {', '.join(missing)}")
        print("Run: export $(grep -v '^#' .env | xargs)  before running this script")
        sys.exit(1)

    dsn = _get_dsn()

    if args.mode in ("predictions", "all"):
        inject_predictions(dsn)

    if args.mode in ("drift", "all"):
        inject_drifted_features(dsn, args.drift_factor)

    if args.mode in ("trigger", "all"):
        trigger_monitoring_dag()

    print("\nDone. Check Airflow at http://localhost:8080 and Grafana at http://localhost:3000")


if __name__ == "__main__":
    main()
