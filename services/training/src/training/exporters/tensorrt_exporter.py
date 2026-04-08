"""
Convert an ONNX model to a TensorRT engine optimized for your specific GPU.

Why TensorRT?
  TensorRT takes the ONNX graph, fuses layers, picks the fastest CUDA kernels
  for your exact GPU, and compiles everything into a single binary engine.
  The result is typically 5-15x faster than PyTorch on GPU for inference.

FP16 precision:
  We build with FP16 enabled. The RTX 4060 has dedicated Tensor Cores for FP16
  that are 2x faster than FP32. The accuracy loss is negligible for our use
  case (anomaly probability output changes by <0.5%).

Optimization profiles:
  TensorRT needs to know the range of batch sizes it will see at runtime so it
  can pre-compile the best kernels for each. We define:
    min = (1, 60, 25)   — single request
    opt = (8, 60, 25)   — typical batch in production
    max = (32, 60, 25)  — largest batch we'll send

  TRT optimizes hardest for the "opt" shape. min and max just define the
  valid range.

Engine portability warning:
  The compiled engine is SPECIFIC to this GPU, driver, and TRT version.
  It will not run on a different GPU or after a driver update. This is why
  we save build_metadata.json alongside it — so we always know what hardware
  it requires.

Output: saves to MinIO at
  models/trt/{model_version}/engine.trt
  models/trt/{model_version}/build_metadata.json
"""
import io
import json
import os
import platform
import tempfile

import tensorrt as trt
import torch

from shared.logging import get_logger
from shared.storage import get_minio_client

log = get_logger(__name__)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
SEQ_LEN = 60
N_FEATURES = 25


def _get_build_metadata(model_version: str) -> dict:
    """Capture everything needed to reproduce or validate this engine."""
    return {
        "model_version": model_version,
        "tensorrt_version": trt.__version__,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "driver_version": "595.79",
        "platform": platform.platform(),
        "precision": "fp16",
        "batch_min": 1,
        "batch_opt": 8,
        "batch_max": 32,
        "seq_len": SEQ_LEN,
        "n_features": N_FEATURES,
    }


def build_engine_from_onnx(onnx_path: str) -> bytes:
    """
    Parse ONNX and compile a TensorRT engine with FP16 + optimization profiles.

    Returns the serialized engine as bytes.
    """
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    # FP16 — uses Tensor Cores on RTX 4060
    config.set_flag(trt.BuilderFlag.FP16)

    # Memory pool: 2GB workspace for TRT to use during engine building
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parsing failed: {errors}")

    # Optimization profile — defines valid batch size range
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input",
        min=(1, SEQ_LEN, N_FEATURES),   # smallest batch
        opt=(8, SEQ_LEN, N_FEATURES),   # TRT optimizes hardest for this
        max=(32, SEQ_LEN, N_FEATURES),  # largest batch
    )
    config.add_optimization_profile(profile)

    log.info("building_trt_engine", note="this takes 1-3 minutes on first build")
    engine_bytes = builder.build_serialized_network(network, config)

    if engine_bytes is None:
        raise RuntimeError("TensorRT engine build failed — check GPU memory and ONNX compatibility")

    return bytes(engine_bytes)


def export_to_tensorrt(
    model_version: str,
    onnx_minio_key: str,
    correlation_id: str = "",
) -> dict[str, str]:
    """
    Download ONNX from MinIO, build TRT engine, upload engine + metadata.

    Returns dict with 'engine_key' and 'metadata_key' MinIO paths.
    """
    bucket = os.environ.get("MINIO_BUCKET_MODELS", "models")
    client = get_minio_client()

    # Download ONNX from MinIO
    log.info("downloading_onnx", key=onnx_minio_key, correlation_id=correlation_id)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
        onnx_data = client.get_object(bucket, onnx_minio_key).read()
        f.write(onnx_data)

    try:
        log.info("building_engine", model_version=model_version, correlation_id=correlation_id)
        engine_bytes = build_engine_from_onnx(onnx_path)
    finally:
        os.unlink(onnx_path)

    # Upload engine
    engine_key = f"trt/{model_version}/engine.trt"
    client.put_object(
        bucket, engine_key,
        io.BytesIO(engine_bytes),
        length=len(engine_bytes),
    )

    # Upload build metadata
    metadata = _get_build_metadata(model_version)
    metadata_bytes = json.dumps(metadata, indent=2).encode()
    metadata_key = f"trt/{model_version}/build_metadata.json"
    client.put_object(
        bucket, metadata_key,
        io.BytesIO(metadata_bytes),
        length=len(metadata_bytes),
    )

    size_mb = round(len(engine_bytes) / 1e6, 2)
    log.info(
        "trt_export_complete",
        engine_key=engine_key,
        engine_size_mb=size_mb,
        metadata_key=metadata_key,
        correlation_id=correlation_id,
    )
    return {"engine_key": engine_key, "metadata_key": metadata_key}
