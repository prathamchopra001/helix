"""
Three-tier model loader: TensorRT → ONNX → PyTorch fallback.

Why three tiers?
  TensorRT is fastest (GPU, FP16) but requires a GPU and a compiled engine.
  ONNX Runtime is fast on CPU and needs no GPU.
  PyTorch is the fallback of last resort — always works, slower.

Hot reload:
  A background thread polls MLflow every 60s. When the Production model
  version changes, it downloads the new artifacts and swaps them in without
  restarting the container. In-flight requests complete before the swap.

Thread safety:
  _model_lock protects the global _active_backend / _active_version.
  Reads don't need the lock (Python GIL protects reference swaps on CPython).
"""
import io
import os
import pickle
import tempfile
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import mlflow
import numpy as np
import onnxruntime as ort
import torch
from mlflow.tracking import MlflowClient

from shared.logging import get_logger
from shared.storage import get_minio_client
from inference.config import settings

log = get_logger(__name__)


class Backend(str, Enum):
    TENSORRT = "tensorrt"
    ONNX = "onnx"
    PYTORCH = "pytorch"


@dataclass
class LoadedModel:
    version: str
    backend: Backend
    # One of these will be set depending on backend
    trt_context: object = None       # tensorrt.IExecutionContext
    ort_session: object = None       # onnxruntime.InferenceSession
    torch_model: object = None       # nn.Module
    scaler: object = None            # StandardScaler
    threshold: float = 0.5          # classification threshold optimised on val set


# Global active model — swapped atomically by the reload thread
_active: Optional[LoadedModel] = None
_model_lock = threading.Lock()


def _load_threshold(model_version: str) -> float:
    """
    Fetch the classification threshold that was optimised during training.
    Stored as an MLflow run param 'classification_threshold'.
    Falls back to 0.5 if not found (older models without threshold tuning).
    """
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = MlflowClient()
        versions = client.get_model_version("helix_anomaly_detector", model_version)
        versions = [versions] if versions else []
        if not versions:
            return 0.5
        run_id = versions[0].run_id
        run = client.get_run(run_id)
        param_val = run.data.params.get("classification_threshold")
        if param_val is not None:
            threshold = float(param_val)
            log.info("threshold_loaded", model_version=model_version, threshold=threshold)
            return threshold
    except Exception as exc:
        log.warning("threshold_load_failed", model_version=model_version, error=str(exc))
    return 0.5


def _load_scaler(model_version: str) -> object:
    """Download the StandardScaler from MinIO that was saved during training."""
    client = get_minio_client()
    bucket = settings.minio_bucket_models
    key = f"scalers/{model_version}/scaler.pkl"
    try:
        data = client.get_object(bucket, key).read()
        scaler = pickle.loads(data)
        log.info("scaler_loaded", model_version=model_version, key=key)
        return scaler
    except Exception as exc:
        log.warning("scaler_load_failed", model_version=model_version, error=str(exc))
        return None


def _try_load_tensorrt(model_version: str) -> Optional[LoadedModel]:
    """
    Attempt to load TRT engine from MinIO.
    Returns None if GPU unavailable, engine missing, or import fails.
    """
    try:
        import tensorrt as trt  # only available in GPU container
    except ImportError:
        log.info("tensorrt_not_available", reason="tensorrt package not installed")
        return None

    if not torch.cuda.is_available():
        log.info("tensorrt_skipped", reason="no CUDA device")
        return None

    client = get_minio_client()
    bucket = settings.minio_bucket_models
    engine_key = f"trt/{model_version}/engine.trt"

    try:
        engine_bytes = client.get_object(bucket, engine_key).read()
    except Exception as exc:
        log.info("trt_engine_not_found", key=engine_key, error=str(exc))
        return None

    try:
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        context = engine.create_execution_context()

        scaler = _load_scaler(model_version)
        threshold = _load_threshold(model_version)
        log.info("tensorrt_loaded", model_version=model_version, engine_key=engine_key)
        return LoadedModel(
            version=model_version,
            backend=Backend.TENSORRT,
            trt_context=context,
            scaler=scaler,
            threshold=threshold,
        )
    except Exception as exc:
        log.warning("tensorrt_load_failed", error=str(exc))
        return None


def _try_load_onnx(model_version: str) -> Optional[LoadedModel]:
    """Download ONNX from MinIO and create an ORT session."""
    client = get_minio_client()
    bucket = settings.minio_bucket_models
    onnx_key = f"onnx/{model_version}/model.onnx"

    try:
        onnx_bytes = client.get_object(bucket, onnx_key).read()
    except Exception as exc:
        log.info("onnx_not_found", key=onnx_key, error=str(exc))
        return None

    try:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
            if torch.cuda.is_available() else ["CPUExecutionProvider"]
        session = ort.InferenceSession(onnx_bytes, providers=providers)
        scaler = _load_scaler(model_version)
        threshold = _load_threshold(model_version)
        log.info("onnx_loaded", model_version=model_version, providers=providers)
        return LoadedModel(
            version=model_version,
            backend=Backend.ONNX,
            ort_session=session,
            scaler=scaler,
            threshold=threshold,
        )
    except Exception as exc:
        log.warning("onnx_load_failed", error=str(exc))
        return None


def _try_load_pytorch(model_version: str) -> Optional[LoadedModel]:
    """Load PyTorch model directly from MLflow as last resort."""
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        model = mlflow.pytorch.load_model(
            f"models:/helix_anomaly_detector/{model_version}"
        )
        model.eval()
        scaler = _load_scaler(model_version)
        # Prefer threshold stored on the model object (set during training)
        threshold = getattr(model, "threshold", None) or _load_threshold(model_version)
        log.info("pytorch_loaded", model_version=model_version, threshold=threshold)
        return LoadedModel(
            version=model_version,
            backend=Backend.PYTORCH,
            torch_model=model,
            scaler=scaler,
            threshold=threshold,
        )
    except Exception as exc:
        log.error("pytorch_load_failed", error=str(exc))
        return None


def load_model(model_version: str) -> Optional[LoadedModel]:
    """Try TRT → ONNX → PyTorch in order, return first that works."""
    log.info("loading_model", model_version=model_version)

    loaded = _try_load_tensorrt(model_version)
    if loaded:
        return loaded

    loaded = _try_load_onnx(model_version)
    if loaded:
        return loaded

    return _try_load_pytorch(model_version)


def get_production_version() -> Optional[str]:
    """Query MLflow for the current Production model version."""
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = MlflowClient()
        versions = client.get_latest_versions(
            "helix_anomaly_detector", stages=["Production"]
        )
        if versions:
            return versions[0].version
    except Exception as exc:
        log.warning("mlflow_version_check_failed", error=str(exc))
    return None


def _reload_loop() -> None:
    """Background thread: poll MLflow every N seconds, hot-swap if version changed."""
    global _active
    while True:
        time.sleep(settings.model_poll_interval)
        try:
            version = get_production_version()
            if version is None:
                continue
            current_version = _active.version if _active else None
            if version == current_version:
                continue

            log.info("new_production_version_detected", version=version, current=current_version)
            new_model = load_model(version)
            if new_model:
                with _model_lock:
                    _active = new_model
                log.info(
                    "model_hot_reloaded",
                    version=version,
                    backend=new_model.backend,
                )
        except Exception as exc:
            log.error("reload_loop_error", error=str(exc))


def startup_load() -> None:
    """Called once at app startup to load the current Production model."""
    global _active
    version = get_production_version()
    if version is None:
        log.warning("no_production_model_found", note="inference will return 503 until a model is promoted")
        return

    model = load_model(version)
    if model:
        _active = model
        log.info("startup_model_loaded", version=version, backend=model.backend)
    else:
        log.error("startup_model_load_failed", version=version)

    # Start background reload thread
    t = threading.Thread(target=_reload_loop, daemon=True)
    t.start()
    log.info("model_reload_thread_started", interval_seconds=settings.model_poll_interval)


def get_active_model() -> Optional[LoadedModel]:
    return _active
