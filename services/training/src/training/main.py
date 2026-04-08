"""
Training entry point — orchestrates data loading, windowing, and training.

What run_training() does:
  1. Loads all feature vectors from features.feature_vectors
  2. Builds 60-day sliding windows for train and val splits
  3. Fits a StandardScaler on train data only, saves it to MinIO
  4. Calls the trainer which runs the training loop and logs to MLflow
  5. Returns the MLflow run ID, model version, and best val F1

Callable from the Airflow retraining_dag or directly from a script.
"""

from shared.logging import get_logger
from training.datasets.time_series_dataset import (
    build_windows,
    load_features_df,
    save_scaler,
)
from training.trainers.anomaly_trainer import train

log = get_logger(__name__)


def run_training(
    hyperparams: dict | None = None,
    correlation_id: str = "",
) -> dict:
    """
    Run the full training pipeline.

    Returns dict with mlflow_run_id, model_version, best_val_f1.
    Raises RuntimeError if feature store is empty or no val windows exist.
    """
    log.info("run_training_start", correlation_id=correlation_id)

    df = load_features_df()
    if df.empty:
        raise RuntimeError("Feature store is empty — run ETL DAG first.")

    X, y, splits, scaler = build_windows(df, fit_scaler=True)

    X_train = X[splits == "train"]
    y_train = y[splits == "train"]
    X_val = X[splits == "val"]
    y_val = y[splits == "val"]

    log.info(
        "splits_ready",
        train_windows=len(X_train),
        val_windows=len(X_val),
        train_anomalies=int(y_train.sum()),
        val_anomalies=int(y_val.sum()),
        correlation_id=correlation_id,
    )

    if len(X_val) == 0:
        raise RuntimeError("No validation windows — need more data.")

    result = train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        hyperparams=hyperparams,
        correlation_id=correlation_id,
    )

    save_scaler(scaler, model_version=result["model_version"])

    log.info(
        "run_training_complete",
        model_version=result["model_version"],
        best_val_f1=result["best_val_f1"],
        correlation_id=correlation_id,
    )
    return result
