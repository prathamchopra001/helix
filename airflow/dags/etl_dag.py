"""
ETL DAG — runs daily, reads raw OHLCV and writes feature vectors.

What this DAG does:
  Single task that calls run_etl() for all configured tickers. Airflow handles
  scheduling, retries, and failure alerting. The task fails (marks red) if any
  ticker wrote 0 rows — that means either the raw data is missing or the quality
  gate rejected it.

Schedule: @daily (runs at midnight UTC)
SLA: 2 hours — if the task hasn't finished by then, Airflow sends an SLA miss alert
Max active runs: 1 — prevents two ETL runs overlapping and double-writing features
"""
from __future__ import annotations

from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago


def _run_etl(**context: dict) -> None:
    import sys
    sys.path.insert(0, "/opt/helix/shared/src")
    sys.path.insert(0, "/opt/helix/etl/src")

    from etl.main import run_etl

    from shared.logging import configure_logging

    correlation_id = context["run_id"]
    configure_logging("etl")

    results = run_etl(correlation_id=correlation_id)

    failed = [ticker for ticker, rows in results.items() if rows == 0]
    if failed:
        raise RuntimeError(f"ETL produced 0 rows for tickers: {failed}")


with DAG(
    dag_id="etl_dag",
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "sla": timedelta(hours=2),
    },
    tags=["etl", "features"],
) as dag:

    PythonOperator(
        task_id="run_etl",
        python_callable=_run_etl,
    )
