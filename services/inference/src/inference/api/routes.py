"""
FastAPI routes for the inference service.

POST /predict        — single ticker prediction
POST /predict/batch  — multiple tickers in one request
GET  /health         — liveness probe (is the process alive?)
GET  /ready          — readiness probe (is a model loaded?)
GET  /metrics        — Prometheus metrics
"""
import time
import uuid

import numpy as np
import psycopg2
from fastapi import APIRouter, HTTPException, Request
from inference.api.middleware.metrics import (
    ACTIVE_MODEL_VERSION,
    INFERENCE_ERRORS,
    INFERENCE_LATENCY,
    PREDICTIONS_TOTAL,
    PREPROCESS_LATENCY,
)
from inference.core.model_loader import Backend, get_active_model
from inference.core.preprocessor import PreprocessingError, preprocess
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from shared.logging import get_logger

log = get_logger(__name__)
router = APIRouter()


# ── Request / Response models ──────────────────────────────────────────────────

class PredictRequest(BaseModel):
    ticker: str = Field(..., description="Ticker symbol, e.g. AAPL")


class PredictResponse(BaseModel):
    ticker: str
    score: float = Field(..., description="Anomaly probability [0, 1]")
    label: int = Field(..., description="1 = anomaly, 0 = normal (threshold: 0.5)")
    model_version: str
    backend: str
    latency_ms: float
    request_id: str


class BatchPredictRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1, max_length=20)


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    errors: dict[str, str] = Field(default_factory=dict)


# ── Inference helper ───────────────────────────────────────────────────────────

def _run_inference(input_array: np.ndarray, model) -> float:
    """
    Run the model on preprocessed input. Returns scalar anomaly probability.
    input_array shape: (1, 60, 25)
    """
    if model.backend == Backend.TENSORRT:
        import torch

        context = model.trt_context

        # Bind input/output buffers
        input_tensor = input_array  # already (1, 60, 25) float32
        output = np.empty((1, 1), dtype=np.float32)

        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda

        d_input = cuda.mem_alloc(input_tensor.nbytes)
        d_output = cuda.mem_alloc(output.nbytes)

        cuda.memcpy_htod(d_input, input_tensor)
        context.execute_v2([int(d_input), int(d_output)])
        cuda.memcpy_dtoh(output, d_output)

        return float(output[0, 0])

    elif model.backend == Backend.ONNX:
        result = model.ort_session.run(
            ["output"], {"input": input_array}
        )
        return float(result[0][0, 0])

    else:  # PyTorch
        import torch
        tensor = torch.from_numpy(input_array)
        with torch.no_grad():
            out = model.torch_model(tensor)
        return float(out[0, 0])


def _log_prediction(
    ticker: str,
    score: float,
    label: int,
    model_version: str,
    backend: str,
    latency_ms: float,
    request_id: str,
    dsn: str,
) -> None:
    """Write prediction to predictions.inference_log. Fire-and-forget — errors logged, not raised."""
    import json as _json
    try:
        conn = psycopg2.connect(dsn)
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                    INSERT INTO predictions.inference_log
                        (request_id, ticker, timestamp, features, score, label,
                         model_version, backend, latency_ms)
                    VALUES (%s, %s, NOW(), %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (request_id) DO NOTHING
                    """,
                (request_id, ticker, _json.dumps({}), score, label,
                 model_version, backend, latency_ms),
            )
        conn.close()
    except Exception as exc:
        log.warning("prediction_log_failed", request_id=request_id, error=str(exc))


def _predict_one(ticker: str, request_id: str, correlation_id: str) -> PredictResponse:
    """Core prediction logic for a single ticker."""
    from inference.config import settings

    model = get_active_model()
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded yet — try again shortly")

    t_start = time.perf_counter()

    # Preprocess
    t_pre = time.perf_counter()
    try:
        input_array = preprocess(ticker, model, correlation_id=correlation_id)
    except PreprocessingError as exc:
        INFERENCE_ERRORS.labels(error_type="preprocessing").inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    PREPROCESS_LATENCY.observe(time.perf_counter() - t_pre)

    # Inference
    try:
        score = _run_inference(input_array, model)
    except Exception as exc:
        INFERENCE_ERRORS.labels(error_type="model_run").inc()
        log.error("inference_failed", ticker=ticker, error=str(exc), correlation_id=correlation_id)
        raise HTTPException(status_code=500, detail="Inference failed") from exc

    latency_ms = (time.perf_counter() - t_start) * 1000
    label = 1 if score >= model.threshold else 0
    outcome = "anomaly" if label == 1 else "normal"

    PREDICTIONS_TOTAL.labels(outcome=outcome, backend=model.backend.value).inc()
    INFERENCE_LATENCY.observe(latency_ms / 1000)
    ACTIVE_MODEL_VERSION.labels(version=model.version, backend=model.backend.value).set(1)

    log.info(
        "prediction_made",
        ticker=ticker,
        score=round(score, 4),
        label=label,
        model_version=model.version,
        backend=model.backend.value,
        latency_ms=round(latency_ms, 2),
        correlation_id=correlation_id,
        request_id=request_id,
    )

    _log_prediction(ticker, score, label, model.version, model.backend.value, latency_ms, request_id, settings.dsn)

    return PredictResponse(
        ticker=ticker,
        score=round(score, 6),
        label=label,
        model_version=model.version,
        backend=model.backend.value,
        latency_ms=round(latency_ms, 2),
        request_id=request_id,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request):
    correlation_id = request.headers.get("X-Correlation-ID", "")
    request_id = str(uuid.uuid4())
    return _predict_one(body.ticker.upper(), request_id, correlation_id)


@router.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(body: BatchPredictRequest, request: Request):
    correlation_id = request.headers.get("X-Correlation-ID", "")
    results = []
    errors = {}

    for ticker in body.tickers:
        ticker = ticker.upper()
        request_id = str(uuid.uuid4())
        try:
            result = _predict_one(ticker, request_id, correlation_id)
            results.append(result)
        except HTTPException as exc:
            errors[ticker] = exc.detail
        except Exception as exc:
            errors[ticker] = str(exc)

    return BatchPredictResponse(results=results, errors=errors)


@router.get("/health")
def health():
    """Liveness probe — always returns 200 if the process is running."""
    return {"status": "ok"}


@router.get("/ready")
def ready():
    """Readiness probe — returns 503 if no model is loaded yet."""
    model = get_active_model()
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    return {
        "status": "ready",
        "model_version": model.version,
        "backend": model.backend.value,
    }


@router.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
