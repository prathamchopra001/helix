"""
Monitoring service entry point.

Orchestrates drift detection and KPI computation. Called by monitoring_dag.
"""

from __future__ import annotations

from monitoring.config import MonitoringConfig
from monitoring.drift.evidently_runner import compute_drift
from monitoring.metrics.model_metrics import compute_model_metrics
from shared.logging import configure_logging, get_logger

configure_logging("monitoring")
log = get_logger(__name__)


def run_monitoring(correlation_id: str = "") -> dict:
    """Called by monitoring_dag. Returns drift PSI dict and metrics dict."""
    cfg = MonitoringConfig()
    dsn = cfg.dsn

    # 1. Compute drift
    psi_scores = compute_drift(dsn, correlation_id=correlation_id)

    # 2. Compute model KPIs for 7d and 30d windows
    metrics_7d = compute_model_metrics(dsn, window_days=7, correlation_id=correlation_id)
    metrics_30d = compute_model_metrics(dsn, window_days=30, correlation_id=correlation_id)

    # 3. Determine if retraining should be triggered
    drifted_features = [f for f, psi in psi_scores.items() if psi > cfg.drift_psi_threshold]
    should_retrain = len(drifted_features) >= cfg.drift_min_features

    log.info(
        "monitoring_complete",
        drifted_features=len(drifted_features),
        should_retrain=should_retrain,
        val_f1_7d=metrics_7d.get("f1"),
        correlation_id=correlation_id,
    )

    return {
        "psi_scores": psi_scores,
        "drifted_features": drifted_features,
        "should_retrain": should_retrain,
        "metrics_7d": metrics_7d,
        "metrics_30d": metrics_30d,
    }
