# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Helix** — end-to-end ML platform for financial market anomaly detection. Binary classification on 60-bar OHLCV sliding windows. See `IMPLEMENTATION_PLAN.md` for the full plan, phase checklist, and progress tracker. See `FINDINGS.md` (gitignored) for error log and discovered gotchas.

## Common Commands

```bash
# Bring up all CPU services
make up

# Bring up with GPU/TensorRT profile
make up-gpu

# Backfill 2 years of market data
make seed

# Trigger model training manually
make train

# Export TensorRT engine for current Production model
make export-trt

# Run all tests
make test

# Tear everything down including volumes
make clean
```

Individual test commands (once test suite exists):
```bash
# Run unit tests only
pytest tests/unit/ -v

# Run a single test file
pytest tests/unit/test_transformers.py -v

# Run integration tests (requires Docker — uses testcontainers)
pytest tests/integration/ -v

# Run with coverage
pytest --cov=services --cov-report=term-missing
```

Schema migrations:
```bash
# Apply all pending migrations
alembic upgrade head

# Create a new migration
alembic revision --autogenerate -m "description"

# Check current migration state
alembic current
```

## Architecture

### Service Topology

All services run in Docker Compose. Nginx is the single entrypoint at `:80`, routing by path prefix to each service. There is no direct service-to-service HTTP outside of this.

```
Nginx (:80)
  /mlflow   → MLflow (:5000)
  /airflow  → Airflow webserver (:8080)
  /grafana  → Grafana (:3000)
  /api      → Inference API (:8000)
  /minio    → MinIO console (:9001)
```

Backing services: PostgreSQL (primary store), MinIO (object store — model artifacts, raw CSVs), Redis (Airflow Celery broker).

### Data Flow

```
yfinance → ingestion service → postgres.raw.ohlcv_data + minio/raw-data/
         → [Airflow etl_dag @daily]
         → postgres.features.feature_vectors (40+ features, anomaly label)
         → [Airflow retraining_dag @weekly + on-demand]
         → PyTorch training → MLflow registry → ONNX → TensorRT engine
         → FastAPI inference service (TRT → ONNX → PyTorch fallback)
         → postgres.predictions.inference_log
         → [Airflow monitoring_dag @daily]
         → Evidently drift detection + model KPIs → Prometheus → Grafana
```

### Shared Package

`shared/src/shared/` is an internal Python package installed into every service. It provides:
- `logging.py` — `structlog` JSON logger factory; always call this, never configure logging directly
- `db.py` — SQLAlchemy async engine + session factory; use `get_session()` context manager
- `storage.py` — MinIO client wrapper; all artifact reads/writes go through this
- `config.py` — `Pydantic BaseSettings` base class; all service configs extend this

### Database

4 PostgreSQL schemas — never mix concerns across schemas:
- `raw` — ingested OHLCV data as-is from yfinance
- `features` — engineered feature vectors + anomaly labels + train/val/test split assignment
- `predictions` — inference log (every prediction request stored here)
- `monitoring` — model KPIs, drift reports, pipeline failure events

All schema changes go through Alembic. Never write raw DDL.

### Inference Model Loading

`services/inference/src/core/model_loader.py` implements a three-tier fallback:
1. TensorRT `.engine` file (GPU profile only) — loaded from MinIO
2. ONNX Runtime — loaded from MinIO
3. PyTorch `model.pt` — loaded from MinIO

Backend is controlled by `INFERENCE_BACKEND` env var. The loader polls MLflow Model Registry every 60s for a new "Production" model and hot-reloads without restarting the container.

### TensorRT

- Export path: PyTorch → ONNX (`onnx_exporter.py`) → TensorRT engine (`tensorrt_exporter.py`)
- Pinned versions: CUDA 12.1, TensorRT 8.6.1, PyTorch 2.1.0+cu121, ONNX 1.14.0
- Engines are GPU/driver-specific — always store `build_metadata.json` alongside `.engine` in MinIO
- Host `nvcc` version is irrelevant — CUDA runs inside Docker via NVIDIA Container Toolkit
- If engine load fails, inference service falls back to ONNX silently (never crashes)

### Airflow DAG Dependencies

DAGs must not use `ExternalTaskSensor` for cross-DAG dependencies — use `TriggerDagRunOperator` or Airflow Datasets instead (sensors hold worker slots). `MAX_ACTIVE_RUNS_PER_DAG=1` is set globally. Every task must be idempotent.

## Code Standards

- All Python must pass `mypy --strict`. No bare `Any`.
- No bare `except` — catch specific exceptions, log with context.
- Every service uses `shared.logging` — never configure `structlog` or `logging` directly in a service.
- Every log line must include `correlation_id` (injected by Nginx, propagated via request context).
- All Dockerfiles are multi-stage builds. Runtime images must not contain build tools.
- Config values come from env vars validated by Pydantic `BaseSettings` on startup — if a required var is missing, the service must fail fast with a clear message, not silently use a default.
- Anomaly thresholds (`ANOMALY_Z_SCORE_THRESHOLD`, `ANOMALY_VOLUME_RATIO_THRESHOLD`) and drift thresholds (`DRIFT_PSI_THRESHOLD`) are env vars — never hardcode them.
