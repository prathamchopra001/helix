# Helix

End-to-end ML platform for financial market anomaly detection. Detects unusual price and volume behavior on equity and crypto tickers using a PyTorch LSTM model trained on 60-bar sliding windows of engineered features.

```
yfinance → ingestion → feature engineering → training → MLflow → ONNX → FastAPI
                                                                        ↓
                                          Grafana ← Prometheus ← monitoring (Evidently drift)
```

---

## Quickstart

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env — change passwords, generate AIRFLOW_FERNET_KEY (see .env.example)

# 2. Bring up all services
make up

# 3. Bootstrap infrastructure (MinIO buckets, DB migrations)
make bootstrap

# 4. Backfill 2 years of market data
make seed

# 5. Trigger the ETL and training pipeline in Airflow
# Open http://localhost/airflow → unpause etl_dag, retraining_dag → trigger manually
```

After training completes (~10 min), the model is automatically registered in MLflow, promoted to Production if F1 improves, and the ONNX artifact is exported to MinIO. The inference service hot-reloads within 60 seconds.

**Test a prediction:**
```bash
curl -X POST http://localhost/api/predict \
  -H "X-API-Key: dev-key-1" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```

---

## Architecture

### Service Topology

Nginx at `:80` is the single entrypoint, routing by path prefix:

| Path | Service | Port |
|------|---------|------|
| `/api/` | Inference API (FastAPI) | 8000 |
| `/mlflow/` | MLflow UI + Model Registry | 5000 |
| `/airflow/` | Airflow Webserver | 8080 |
| `/grafana/` | Grafana Dashboards | 3000 |
| — | MinIO Console (direct) | 9001 |
| — | Prometheus (internal) | 9090 |

### Data Flow

```
Yahoo Finance
    └─► ingestion service (@hourly)
            ├─► postgres.raw.ohlcv_data
            └─► minio/raw-data/ (CSV archive)
                    │
            [Airflow: etl_dag @daily]
                    ▼
        Technical indicators (RSI, MACD, Bollinger Bands, ATR, OBV)
        Rolling z-scores (5d/10d/20d/60d), volume ratios
        Momentum features: return_5d, return_20d, hl_range, overnight_gap
        Anomaly labels: |z_score_20d| > 2.0 OR volume_ratio_5d > 1.5
                    │
            postgres.features.feature_vectors  (29 features, train/val/test split)
                    │
            [Airflow: retraining_dag @weekly + drift-triggered]
                    ▼
        LSTM (128 hidden, 3 layers) + FocalLoss + threshold optimization
        MLflow experiment tracking → model registry (Staging → Production)
        ONNX export → minio/models/onnx/{version}/model.onnx
                    │
            FastAPI inference service (ONNX backend, hot-reload every 60s)
                    │
            postgres.predictions.inference_log
                    │
            [Airflow: monitoring_dag @daily]
                    ▼
        Evidently drift detection (PSI per feature)
        Rolling F1/precision/recall (7d + 30d)
        Prometheus metrics → Grafana dashboards
        PSI > 0.2 on ≥3 features → triggers retraining_dag
```

### Model Architecture

```
Input (batch, 60 days, 29 features)
    ↓
LSTM (128 hidden, 3 layers, dropout=0.3) — reads sequence day-by-day
    ↓
Take last timestep
    ↓
Dropout(0.4) → Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→1) → Sigmoid
    ↓
Output: anomaly probability [0, 1]
```

**Inference backend** falls back automatically: TensorRT (GPU) → ONNX Runtime → PyTorch.

**Current Production model (v14):** threshold_val_f1=0.4776 at threshold=0.55, trained with FocalLoss (γ=2) on 29 features, 9105 training windows across 11 tickers.

---

## Services

| Service | Purpose |
|---------|---------|
| `ingestion` | Fetches OHLCV from Yahoo Finance, upserts to `raw.ohlcv_data` and MinIO |
| `etl` | Engineers 29 features, labels anomalies, writes to `features.feature_vectors` |
| `training` | Trains LSTM model, logs to MLflow, exports ONNX |
| `inference` | FastAPI serving predictions with API key auth and rate limiting |
| `monitoring` | Runs Evidently drift detection and rolling KPI computation |
| `shared` | Internal Python package: logging, DB connection pool, MinIO client, config |

---

## Inference API

All endpoints require `X-API-Key` header. Keys are comma-separated in `API_KEYS` env var.

### `POST /predict`
```bash
curl -X POST http://localhost/api/predict \
  -H "X-API-Key: dev-key-1" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```
```json
{
  "ticker": "AAPL",
  "score": 0.556707,
  "label": 1,
  "model_version": "14",
  "backend": "onnx",
  "latency_ms": 58.22,
  "request_id": "5316e803-f26d-4ced-9ef3-c0ca8d102aee"
}
```

`label=1` means anomaly. The threshold is model-version-specific (stored in MLflow params, loaded at startup).

### `POST /predict/batch`
```bash
curl -X POST http://localhost/api/predict/batch \
  -H "X-API-Key: dev-key-1" \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "MSFT", "BTC-USD"]}'
```
```json
{
  "results": [
    {"ticker": "AAPL", "score": 0.557, "label": 1, "latency_ms": 21.0},
    {"ticker": "MSFT", "score": 0.338, "label": 0, "latency_ms": 19.3}
  ],
  "errors": {}
}
```

### `GET /health` / `GET /ready`
```bash
curl http://localhost/api/health   # liveness — always 200 if process is alive
curl http://localhost/api/ready    # readiness — 503 until model is loaded
```

### `GET /metrics`
Prometheus metrics endpoint (predictions counter, latency histogram, active model version gauge).

**Rate limits:** 100 req/min per key on `/predict`, 10 req/min on `/predict/batch`.

---

## Airflow DAGs

| DAG | Schedule | Purpose |
|-----|----------|---------|
| `ingestion_dag` | `@hourly` | Fetch OHLCV from Yahoo Finance for all tickers |
| `etl_dag` | `@daily` | Feature engineering + anomaly labeling |
| `retraining_dag` | `@weekly` + on-demand | Train → evaluate → promote → export ONNX |
| `monitoring_dag` | `@daily` | Drift detection + KPI computation; triggers retraining if PSI > 0.2 on ≥3 features |

`retraining_dag` promotes a new model to Production only if its threshold-tuned val F1 exceeds the current Production model's by more than 0.02.

---

## Database Schema

Four PostgreSQL schemas — concerns never mix across them:

| Schema | Table | Purpose |
|--------|-------|---------|
| `raw` | `ohlcv_data` | Raw OHLCV from Yahoo Finance |
| `features` | `feature_vectors` | 29 engineered features + label + train/val/test split (JSONB) |
| `predictions` | `inference_log` | Every prediction request with score, latency, model version |
| `monitoring` | `model_metrics` | Rolling F1/precision/recall by window |
| `monitoring` | `drift_reports` | Evidently PSI reports (JSON) |
| `monitoring` | `pipeline_events` | DAG failure events |

All timestamps are indexed with BRIN. All schema changes go through Alembic — never raw DDL.

---

## Make Commands

```bash
make up           # Start all CPU services
make up-gpu       # Start with GPU/TensorRT profile
make down         # Stop containers (keeps volumes)
make clean        # Stop containers and delete all volumes
make bootstrap    # Create MinIO buckets, run Alembic migrations
make seed         # Backfill 2 years of OHLCV for 11 tickers
make train        # Manually trigger a training run
make export-trt   # Export current Production model to TensorRT engine
make test         # Run test suite with coverage
make logs         # Tail all container logs
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

```bash
# PostgreSQL
POSTGRES_USER=helix
POSTGRES_PASSWORD=<strong-password>
POSTGRES_DB=helix

# MinIO
MINIO_ACCESS_KEY=<access-key>
MINIO_SECRET_KEY=<secret-key>

# Airflow — generate Fernet key with:
# python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
AIRFLOW_FERNET_KEY=<generated-key>
AIRFLOW__WEBSERVER__SECRET_KEY=<random-secret>

# Inference API — comma-separated list of valid keys
API_KEYS=dev-key-1,dev-key-2

# Anomaly detection thresholds
ANOMALY_Z_SCORE_THRESHOLD=2.0
ANOMALY_VOLUME_RATIO_THRESHOLD=1.5

# Drift detection
DRIFT_PSI_THRESHOLD=0.2
DRIFT_FEATURE_COUNT_THRESHOLD=3
```

---

## Dashboards

Open Grafana at `http://localhost/grafana` (default: admin/admin).

| Dashboard | Contents |
|-----------|---------|
| **Model KPIs** | F1/precision/recall trend (7d + 30d), score distribution, prediction volume, model version timeline |
| **Data Drift** | PSI heatmap per feature, drift alert history, retraining trigger events |
| **Infrastructure** | CPU/memory/disk per container, PostgreSQL connections |
| **Logs** | Live log viewer from Loki — filter by service, level, correlation_id |

---

## Observability

Every log line is structured JSON with `service`, `level`, `timestamp`, and `correlation_id`. Nginx generates a `X-Correlation-ID` on each request and propagates it through all downstream calls. The Loki dashboard can trace a single request end-to-end by correlation ID.

Prometheus scrapes the inference `/metrics` endpoint and the monitoring service exporter. Alertmanager is configured for:
- F1 (7d) drops > 10% vs baseline
- Inference error rate > 1%
- PSI > 0.2 on 3+ features
- GPU utilization > 90%, disk > 80%

---

## TensorRT (GPU)

TensorRT engines are GPU/driver-specific. To export for your GPU:

```bash
make up-gpu       # Starts with gpu profile (requires NVIDIA Container Toolkit)
make export-trt   # Exports current Production model to engine.trt + build_metadata.json
```

The inference service falls back to ONNX Runtime silently if no engine is found, so TRT is optional. Always store `build_metadata.json` alongside `.engine` — it records CUDA/driver/TRT versions for reproducibility.

---

## Tickers

Default set (configurable via `ETL_TICKERS` env var):

```
AAPL, MSFT, GOOGL, TSLA, SPY, QQQ, BTC-USD, ETH-USD, NVDA, AMZN, META
```

---

## Development

```bash
# Install pre-commit hooks
pre-commit install

# Run linting and type checks
ruff check .
mypy services/ shared/

# Unit tests only (no Docker required)
pytest tests/unit/ -v

# Integration tests (spins up real Postgres + MinIO via testcontainers)
pytest tests/integration/ -v

# All tests with coverage
make test
```

CI runs on every push: lint → typecheck → unit tests → integration tests → Docker image build → Trivy security scan.

---

## Known Gotchas

- **Yahoo Finance anti-bot:** requires `curl_cffi` — already pinned in `services/ingestion/pyproject.toml`
- **Airflow DATABASE_URL:** use individual `POSTGRES_*` vars inside containers, not a single `DATABASE_URL` — the composed URL gets mangled
- **psycopg2 + NUMERIC columns:** cast to `float` immediately after fetching — `ta` library can't handle `decimal.Decimal`
- **ONNX export:** always pass `dynamo=False` to `torch.onnx.export()` — the new dynamo exporter (PyTorch 2.5+) creates a separate `.onnx.data` file that ORT can't load from bytes
- **TensorRT engines** are not portable across GPU architectures — always rebuild on the target machine
