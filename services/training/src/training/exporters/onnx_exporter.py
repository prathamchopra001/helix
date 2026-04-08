"""
Export the Production PyTorch model to ONNX format.

Why ONNX?
  ONNX (Open Neural Network Exchange) is a standard format that acts as the
  bridge between PyTorch and TensorRT. PyTorch can't talk directly to TensorRT
  — you export to ONNX first, then TensorRT compiles that ONNX into an
  optimized engine for your specific GPU.

Dynamic axes:
  We export with a dynamic batch dimension so the same ONNX file works for
  batch size 1 (single request) through 32 (batched inference). The sequence
  length (60) is fixed. Feature count is inferred from the model's LSTM
  input_size so it automatically adapts when features are added.

Validation:
  After export we run the same dummy input through both the PyTorch model and
  the ONNX runtime and compare outputs. If they differ by more than 1e-4, the
  export failed silently — we catch that here rather than discovering it in
  production.

Output: saves to MinIO at models/onnx/{model_version}/model.onnx
"""

import io
import os
import tempfile

import mlflow.pytorch
import numpy as np
import onnxruntime as ort
import torch

from shared.logging import get_logger
from shared.storage import get_minio_client

log = get_logger(__name__)

INPUT_NAME = "input"
OUTPUT_NAME = "output"
SEQ_LEN = 60
VALIDATION_TOLERANCE = 1e-4


def export_to_onnx(model_version: str, correlation_id: str = "") -> str:
    """
    Load Production model from MLflow, export to ONNX, validate, upload to MinIO.

    Returns the MinIO key of the uploaded ONNX file.
    """
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)

    log.info(
        "loading_model_from_mlflow", model_version=model_version, correlation_id=correlation_id
    )
    model = mlflow.pytorch.load_model(f"models:/helix_anomaly_detector/{model_version}")
    model.eval()

    # Infer feature count from the model's LSTM input size — handles 25 or 29+ features
    n_features: int = model.lstm.input_size
    dummy_input = torch.randn(1, SEQ_LEN, n_features)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name

    log.info("exporting_to_onnx", onnx_path=onnx_path, correlation_id=correlation_id)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={
            INPUT_NAME: {0: "batch_size"},  # batch dimension is dynamic
            OUTPUT_NAME: {0: "batch_size"},
        },
        dynamo=False,  # Force legacy exporter — always produces a single .onnx file.
        # The new dynamo exporter (PyTorch 2.5+) splits weights into a
        # separate .onnx.data file which ORT can't load from bytes alone.
    )

    # Validate: run both models on the same input, compare outputs
    log.info("validating_onnx", correlation_id=correlation_id)
    with torch.no_grad():
        torch_output = model(dummy_input).numpy()

    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_output = ort_session.run([OUTPUT_NAME], {INPUT_NAME: dummy_input.numpy()})[0]

    max_diff = float(np.abs(torch_output - onnx_output).max())
    if max_diff > VALIDATION_TOLERANCE:
        raise ValueError(
            f"ONNX validation failed: max output difference {max_diff:.6f} > {VALIDATION_TOLERANCE}"
        )

    log.info("onnx_validation_passed", max_diff=round(max_diff, 8), correlation_id=correlation_id)

    # Upload to MinIO
    bucket = os.environ.get("MINIO_BUCKET_MODELS", "models")
    key = f"onnx/{model_version}/model.onnx"
    with open(onnx_path, "rb") as f:
        data = f.read()

    client = get_minio_client()
    client.put_object(bucket, key, io.BytesIO(data), length=len(data))

    os.unlink(onnx_path)
    log.info(
        "onnx_uploaded",
        bucket=bucket,
        key=key,
        size_mb=round(len(data) / 1e6, 2),
        correlation_id=correlation_id,
    )
    return key
