"""
FastAPI application entry point for the Helix inference service.

Startup sequence:
  1. Validate config (Pydantic raises on missing vars)
  2. Load current Production model from MLflow (TRT → ONNX → PyTorch fallback)
  3. Start background thread to poll for new model versions
  4. Mount auth middleware (all routes except /health /ready /metrics)
  5. Mount rate limiter (100 req/min on /predict, 10 req/min on /predict/batch)
  6. Register routes
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from inference.api.middleware.auth import AuthMiddleware
from inference.api.routes import router
from inference.core.model_loader import startup_load
from shared.logging import configure_logging, get_logger

configure_logging("inference")
log = get_logger(__name__)

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("inference_service_starting")
    startup_load()
    log.info("inference_service_ready")
    yield
    log.info("inference_service_stopping")


app = FastAPI(
    title="Helix Inference API",
    version="0.1.0",
    lifespan=lifespan,
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Auth (must be added after exception handlers)
app.add_middleware(AuthMiddleware)

# Routes
app.include_router(router)
