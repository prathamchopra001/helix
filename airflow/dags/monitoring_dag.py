"""
Monitoring DAG — runs daily.

Tasks:
  1. run_monitoring: drift detection + KPI computation
  2. maybe_trigger_retrain: if should_retrain=True, triggers retraining_dag

Uses XCom to pass should_retrain between tasks.
"""
from __future__ import annotations

from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.dates import days_ago


def _run_monitoring(**context: dict) -> None:
    import sys
    sys.path.insert(0, "/opt/helix/shared/src")
    sys.path.insert(0, "/opt/helix/monitoring/src")

    from monitoring.main import run_monitoring
    from shared.logging import configure_logging

    configure_logging("monitoring")
    correlation_id = context["run_id"]

    result = run_monitoring(correlation_id=correlation_id)

    # Push result to XCom for the downstream task
    context["task_instance"].xcom_push(key="monitoring_result", value=result)
    context["task_instance"].xcom_push(key="should_retrain", value=result["should_retrain"])


def _maybe_trigger_retrain(**context: dict) -> None:
    """
    Check the should_retrain flag from XCom.

    If True, trigger the retraining_dag via TriggerDagRunOperator logic.
    This function acts as a gate; the actual trigger operator only fires
    when this function returns without raising (Airflow will handle it).
    """
    should_retrain: bool = context["task_instance"].xcom_pull(
        task_ids="run_monitoring", key="should_retrain"
    )

    import sys
    sys.path.insert(0, "/opt/helix/shared/src")

    from shared.logging import get_logger
    log = get_logger(__name__)

    if should_retrain:
        log.info(
            "retrain_trigger_decision",
            decision="trigger",
            correlation_id=context.get("run_id", ""),
        )
    else:
        log.info(
            "retrain_trigger_decision",
            decision="skip",
            correlation_id=context.get("run_id", ""),
        )

    # Raise SkipException if retraining is not needed so the trigger task is bypassed
    from airflow.exceptions import AirflowSkipException
    if not should_retrain:
        raise AirflowSkipException("No significant drift detected — skipping retraining.")


with DAG(
    dag_id="monitoring_dag",
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["monitoring", "drift"],
) as dag:

    run_monitoring_task = PythonOperator(
        task_id="run_monitoring",
        python_callable=_run_monitoring,
    )

    check_retrain_task = PythonOperator(
        task_id="check_retrain",
        python_callable=_maybe_trigger_retrain,
    )

    trigger_retraining_task = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="retraining_dag",
        wait_for_completion=False,
        reset_dag_run=True,
        conf={"triggered_by": "monitoring_dag", "reason": "drift_detected"},
    )

    run_monitoring_task >> check_retrain_task >> trigger_retraining_task
