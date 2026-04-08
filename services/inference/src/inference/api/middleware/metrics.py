"""
Prometheus metrics for the inference service.

Counters and histograms are module-level singletons — Prometheus client
manages the registry internally. Import this module anywhere to access them.
"""
from prometheus_client import Counter, Gauge, Histogram

# Count of predictions by outcome (anomaly=1 or normal=0)
PREDICTIONS_TOTAL = Counter(
    "helix_predictions_total",
    "Total predictions made",
    ["outcome", "backend"],  # outcome: anomaly|normal, backend: tensorrt|onnx|pytorch
)

# Inference latency (end-to-end including DB fetch + model run)
INFERENCE_LATENCY = Histogram(
    "helix_inference_latency_seconds",
    "End-to-end inference latency",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# Model preprocessing latency (DB fetch + scaler)
PREPROCESS_LATENCY = Histogram(
    "helix_preprocess_latency_seconds",
    "Preprocessing latency (DB fetch + scaling)",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

# Current active model version (label) and backend
ACTIVE_MODEL_VERSION = Gauge(
    "helix_active_model_version",
    "Currently loaded model version",
    ["version", "backend"],
)

# Auth failures
AUTH_FAILURES = Counter(
    "helix_auth_failures_total",
    "Total authentication failures",
    ["reason"],  # reason: missing_key|invalid_key
)

# Errors during inference
INFERENCE_ERRORS = Counter(
    "helix_inference_errors_total",
    "Total inference errors",
    ["error_type"],
)
