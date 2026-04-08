"""
Ingestion DAG — fetches OHLCV data from Yahoo Finance hourly.

Schedule: @hourly
SLA:      30 minutes
Max runs: 1 concurrent run at a time (MAX_ACTIVE_RUNS_PER_DAG=1 set globally)
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    "owner": "helix",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}


def _ingest_ohlcv(**context: dict) -> None:
    """PythonOperator callable — runs ingestion for the current hour."""
    from ingestion.main import run_ingestion

    correlation_id = context["run_id"]
    results = run_ingestion(period="1d", correlation_id=correlation_id)

    failed = [t for t, rows in results.items() if rows == 0]
    if failed:
        raise RuntimeError(f"Ingestion failed for tickers: {failed}")


with DAG(
    dag_id="ingestion_dag",
    default_args=DEFAULT_ARGS,
    description="Hourly OHLCV ingestion from Yahoo Finance → PostgreSQL + MinIO",
    schedule="@hourly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["helix", "ingestion"],
) as dag:
    ingest = PythonOperator(
        task_id="ingest_ohlcv",
        python_callable=_ingest_ohlcv,
        sla=timedelta(minutes=30),
    )
