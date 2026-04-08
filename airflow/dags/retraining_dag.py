"""
Retraining DAG — trains a new model and promotes it if it beats the current one.

What this DAG does:
  1. Runs the full training pipeline (loads features, trains model, logs to MLflow)
  2. Compares the new model's val F1 against the current Production model's F1
  3. Promotes to Production ONLY if new F1 > current F1 + 0.02 (a real improvement,
     not just noise)
  4. If no Production model exists yet, promotes unconditionally

Schedule: @weekly + triggerable on demand by monitoring_dag when drift is detected
Max active runs: 1 — never run two training jobs simultaneously
"""
from __future__ import annotations

from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

PROMOTION_THRESHOLD = 0.02


def _train(**context: dict) -> None:
    import sys
    sys.path.insert(0, "/opt/helix/shared/src")
    sys.path.insert(0, "/opt/helix/training/src")

    from shared.logging import configure_logging
    from training.main import run_training

    configure_logging("training")
    correlation_id = context["run_id"]

    result = run_training(correlation_id=correlation_id)

    # Push result to XCom so the promote task can read it
    context["task_instance"].xcom_push(key="train_result", value=result)


def _export_onnx(**context: dict) -> None:
    """
    Export the promoted model to ONNX and store in MinIO.
    Only runs if promote_model actually promoted (checks Production stage).
    Skips silently if the new version wasn't promoted.
    """
    import sys
    sys.path.insert(0, "/opt/helix/shared/src")
    sys.path.insert(0, "/opt/helix/training/src")

    from mlflow.tracking import MlflowClient

    result = context["task_instance"].xcom_pull(
        task_ids="train_model", key="train_result"
    )
    model_version = result["model_version"]
    model_name = "helix_anomaly_detector"

    client = MlflowClient(tracking_uri="http://mlflow:5000")
    mv = client.get_model_version(model_name, model_version)

    if mv.current_stage != "Production":
        print(f"Version {model_version} is in {mv.current_stage} — skipping ONNX export")
        return

    from training.exporters.onnx_exporter import export_to_onnx
    key = export_to_onnx(model_version)
    print(f"ONNX export complete for version {model_version}: {key}")


def _promote(**context: dict) -> None:
    """
    Promote the newly trained model to Production if it beats the current one.

    Reads the training result from XCom, checks the current Production model's
    F1 in MLflow, and transitions the new version if the improvement is real.
    """
    from mlflow.tracking import MlflowClient

    result = context["task_instance"].xcom_pull(
        task_ids="train_model", key="train_result"
    )

    model_name = "helix_anomaly_detector"
    new_version: str = result["model_version"]

    client = MlflowClient(tracking_uri="http://mlflow:5000")

    # Prefer the threshold-tuned F1 over the raw 0.5-threshold F1.
    # threshold_val_f1 is logged when _find_best_threshold() runs in the trainer.
    # Falls back to best_val_f1 for models trained before threshold optimisation.
    new_run = client.get_run(result["mlflow_run_id"])
    new_f1: float = float(
        new_run.data.metrics.get("threshold_val_f1")
        or result["best_val_f1"]
    )

    # Find the current Production model's comparable F1 (also prefer tuned)
    current_f1 = 0.0
    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if prod_versions:
            prod_run = client.get_run(prod_versions[0].run_id)
            current_f1 = float(
                prod_run.data.metrics.get("threshold_val_f1")
                or prod_run.data.metrics.get("best_val_f1", 0.0)
            )
    except Exception:
        current_f1 = 0.0

    improvement = new_f1 - current_f1

    if improvement >= PROMOTION_THRESHOLD or current_f1 == 0.0:
        # Transition new version to Production
        client.transition_model_version_stage(
            name=model_name,
            version=new_version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(
            f"Promoted version {new_version} to Production "
            f"(F1: {current_f1:.4f} → {new_f1:.4f}, +{improvement:.4f})"
        )
    else:
        print(
            f"Version {new_version} stays in Staging "
            f"(F1: {new_f1:.4f} vs Production: {current_f1:.4f}, "
            f"delta {improvement:+.4f} < required +{PROMOTION_THRESHOLD})"
        )


with DAG(
    dag_id="retraining_dag",
    schedule_interval="@weekly",
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
    },
    tags=["training", "mlflow"],
) as dag:

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=_train,
    )

    promote_task = PythonOperator(
        task_id="promote_model",
        python_callable=_promote,
    )

    export_task = PythonOperator(
        task_id="export_onnx",
        python_callable=_export_onnx,
    )

    train_task >> promote_task >> export_task
