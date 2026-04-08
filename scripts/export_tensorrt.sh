#!/usr/bin/env bash
# Export the current Production model to ONNX then TensorRT.
#
# Usage (from project root):
#   make export-trt
#   OR
#   docker compose --profile gpu run --rm trt-exporter
#
# What this does:
#   1. Finds the Production model version in MLflow
#   2. Exports PyTorch → ONNX (with validation)
#   3. Compiles ONNX → TensorRT engine (FP16, batch profiles 1/8/32)
#   4. Saves engine + build_metadata.json to MinIO
#   5. Prints the MinIO paths for the inference service to use
set -euo pipefail

MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://mlflow:5000}"
export MLFLOW_TRACKING_URI

echo "==> TensorRT Export Pipeline"
echo "    MLflow: $MLFLOW_TRACKING_URI"
echo "    MinIO:  ${MINIO_ENDPOINT:-minio:9000}"
echo ""

python3 - <<'EOF'
import os
import sys

sys.path.insert(0, "/opt/helix/shared/src")
sys.path.insert(0, "/opt/helix/training/src")

import mlflow
from mlflow.tracking import MlflowClient

from shared.logging import configure_logging
from training.exporters.onnx_exporter import export_to_onnx
from training.exporters.tensorrt_exporter import export_to_tensorrt

configure_logging("trt-exporter")

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = MlflowClient()

# Find current Production version
prod_versions = client.get_latest_versions("helix_anomaly_detector", stages=["Production"])
if not prod_versions:
    print("ERROR: No Production model found in MLflow registry.")
    print("       Run retraining_dag first to train and promote a model.")
    sys.exit(1)

model_version = prod_versions[0].version
print(f"==> Found Production model: version {model_version}")

# Step 1: PyTorch → ONNX
print("==> Exporting to ONNX...")
onnx_key = export_to_onnx(model_version=model_version, correlation_id="export")
print(f"    ONNX saved: {onnx_key}")

# Step 2: ONNX → TensorRT
print("==> Building TensorRT engine (this takes 1-3 minutes)...")
result = export_to_tensorrt(
    model_version=model_version,
    onnx_minio_key=onnx_key,
    correlation_id="export",
)
print(f"    Engine saved: {result['engine_key']}")
print(f"    Metadata saved: {result['metadata_key']}")

print("")
print("Export complete.")
print(f"  ONNX:     models/{onnx_key}")
print(f"  Engine:   models/{result['engine_key']}")
print(f"  Metadata: models/{result['metadata_key']}")
EOF
