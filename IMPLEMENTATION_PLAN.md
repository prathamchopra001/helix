# Helix — Implementation Plan

> End-to-end ML platform for financial market anomaly detection — featuring automated ETL pipelines,
> PyTorch + TensorRT inference, MLflow experiment tracking, Airflow orchestration, and full
> observability via Prometheus, Grafana, and Loki. Built with production standards: structured
> logging, CI/CD, drift detection, and auto-retraining.

---

## Domain

**Financial market anomaly detection** — detect unusual price/volume behavior on equity and crypto
tickers using Yahoo Finance as the free data source.

Why this domain:

- Periodic data updates justify automated retraining
- Rich analytics surface (correlations, volatility regimes, drift analysis)
- LSTM+Transformer model benefits strongly from TensorRT optimization
- Observable ground truth (actual price moves) enables real precision/recall tracking

**Model task**: Binary classification — anomaly vs normal — on a 60-bar sliding window of OHLCV

+ technical indicator features. Anomaly label: `|z_score_20| > 2.5 AND volume_ratio_5d > 2.0`.

---

## Tech Stack

| Layer               | Technology                                                             |
| ------------------- | ---------------------------------------------------------------------- |
| ML framework        | PyTorch 2.x — LSTM + Transformer hybrid                               |
| GPU inference       | TensorRT 8.6 via PyTorch → ONNX → TRT path                           |
| Experiment tracking | MLflow 2.x (PostgreSQL backend + MinIO artifact store)                 |
| Orchestration       | Apache Airflow 2.7+ (CeleryExecutor, Redis broker)                     |
| Inference API       | FastAPI + Uvicorn (async ASGI)                                         |
| Primary database    | PostgreSQL 15 (4 schemas: raw, features, predictions, monitoring)      |
| Object storage      | MinIO (S3-compatible local storage)                                    |
| Metrics             | Prometheus + Alertmanager                                              |
| Dashboards          | Grafana — 4 dashboards (model KPIs, data drift, infrastructure, logs) |
| Log aggregation     | Loki + Promtail                                                        |
| Drift detection     | Evidently AI                                                           |
| Feature engineering | Pandas +`ta` library (technical indicators)                          |
| Data validation     | Great Expectations                                                     |
| Schema migrations   | Alembic — no manual SQL ever                                          |
| Structured logging  | `structlog` JSON + correlation IDs across all services               |
| Reverse proxy       | Nginx — single entrypoint, path-based routing                         |
| API security        | API key auth via `X-API-Key` header                                  |
| Rate limiting       | `slowapi` on inference endpoints                                     |
| Code quality        | `ruff` (lint + format) + `mypy --strict`                           |
| Pre-commit          | ruff, mypy, detect-secrets                                             |
| CI/CD               | GitHub Actions: lint → test → build → push images                   |
| Containers          | Docker Compose v2 (profiles: default, gpu, dev, minimal)               |
| Notebooks           | JupyterLab                                                             |
| Data source         | `yfinance` — Yahoo Finance, free, no auth                           |

---

## Project Structure

```
helix/
├── .env.example / .env                 # never committed; validated on startup
├── .pre-commit-config.yaml             # ruff, mypy, detect-secrets
├── .github/workflows/
│   ├── ci.yml                          # lint → typecheck → test → build images
│   └── cd.yml                          # push images to registry on main merge
├── docker-compose.yml                  # base (CPU) services with resource limits
├── docker-compose.gpu.yml              # GPU override — TensorRT inference
├── docker-compose.dev.yml              # dev overrides — hot reload, JupyterLab
├── Makefile                            # make up / up-gpu / seed / train / test / clean
├── pyproject.toml                      # ruff + mypy config, shared dev deps
│
├── infrastructure/
│   ├── nginx/nginx.conf                # reverse proxy, TLS-ready
│   ├── postgres/
│   │   ├── init/01_init_schemas.sql    # schemas + tables + BRIN indexes
│   │   └── postgresql.conf            # tuned: shared_buffers, work_mem
│   ├── grafana/
│   │   ├── provisioning/datasources/  # Prometheus, PostgreSQL, Loki auto-configured
│   │   └── dashboards/                # model_kpis.json, data_drift.json,
│   │                                  # infra.json, logs.json
│   ├── prometheus/
│   │   ├── prometheus.yml
│   │   └── alerts/
│   │       ├── model_alerts.yml       # F1 drop > 10%, error rate > 1%
│   │       └── infra_alerts.yml       # GPU > 90%, disk > 80%, DB conns > 80%
│   ├── alertmanager/alertmanager.yml
│   ├── loki/loki-config.yml
│   ├── promtail/promtail-config.yml   # tails Docker logs → Loki
│   └── airflow/airflow.cfg
│
├── shared/                             # internal Python package used by all services
│   ├── pyproject.toml
│   └── src/shared/
│       ├── logging.py                  # structlog JSON + correlation ID
│       ├── db.py                       # SQLAlchemy async engine + pool
│       ├── storage.py                  # MinIO/S3 client wrapper
│       └── config.py                   # Pydantic BaseSettings base
│
├── services/
│   ├── ingestion/                      # yfinance → postgres.raw + minio
│   ├── etl/                            # raw → features + labels
│   ├── training/                       # PyTorch → MLflow → ONNX → TensorRT
│   ├── inference/                      # FastAPI: auth, rate limit, TRT/ONNX/PyTorch
│   └── monitoring/                     # Evidently drift + KPIs + Prometheus export
│
├── airflow/dags/
│   ├── ingestion_dag.py                # @hourly, SLA 30min
│   ├── etl_dag.py                      # @daily, GE failure = DAG failure
│   ├── retraining_dag.py               # @weekly + triggered, F1 promotion gate
│   └── monitoring_dag.py              # @daily, triggers retraining on drift
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_prototyping.ipynb
│   ├── 04_tensorrt_benchmarks.ipynb
│   └── 05_drift_analysis.ipynb
│
├── migrations/                         # Alembic versions
├── tests/
│   ├── conftest.py                     # testcontainers fixtures
│   ├── unit/
│   └── integration/
└── scripts/
    ├── bootstrap.sh
    ├── seed_data.sh
    └── export_tensorrt.sh
```

---

## Database Schema

**4 PostgreSQL schemas:**

| Schema          | Table               | Key Columns                                                      |
| --------------- | ------------------- | ---------------------------------------------------------------- |
| `raw`         | `ohlcv_data`      | ticker, timestamp, open, high, low, close, volume, ingested_at   |
| `features`    | `feature_vectors` | ticker, timestamp, features JSONB, label, split (train/val/test) |
| `predictions` | `inference_log`   | request_id, ticker, score, label, model_version, latency_ms      |
| `monitoring`  | `model_metrics`   | metric_name, value, window_start, window_end, model_version      |
| `monitoring`  | `drift_reports`   | report JSONB, triggered_retraining BOOL, created_at              |
| `monitoring`  | `pipeline_events` | dag_id, task_id, status, message, created_at                     |

BRIN indexes on all `timestamp` columns. Alembic manages all schema changes.

---

## Data Flow

```
yfinance API
    └─► ingestion service ──► postgres.raw.ohlcv_data
                          └─► minio/raw-data/ (CSV archive)
                                      │
                         [Airflow: etl_dag — @daily]
                                      ▼
              Technical indicators: RSI, MACD, Bollinger Bands, ATR, OBV
              Rolling stats: z-scores (5/10/20/60 windows), log returns
              Anomaly labels: |z_score_20| > 2.5 AND volume_ratio_5d > 2.0
                                      │
                         postgres.features.feature_vectors
                                      │
                    [Airflow: retraining_dag — @weekly + on-demand]
                                      ▼
              PyTorch LSTM+Transformer training
              MLflow: params, metrics, confusion matrix artifact
              ONNX export → TensorRT engine (FP16, batch profiles 1/8/32)
              MLflow Model Registry: Staging → Production (if F1 improves > 0.02)
                                      │
                         FastAPI inference service
                         TRT engine → ONNX Runtime → PyTorch (fallback chain)
                         Hot-reload: polls MLflow registry every 60s
                                      │
                         postgres.predictions.inference_log
                                      │
                    [Airflow: monitoring_dag — @daily]
                                      ▼
              Evidently: PSI per feature (ref = training window, curr = last 7d)
              Model KPIs: rolling F1/precision/recall/AUC (7d + 30d windows)
              Prometheus metrics export
              Trigger retraining if PSI > 0.2 on ≥3 features
                                      │
                         Grafana dashboards (4 panels)
```

---

## Production Standards

Applied across every phase — not deferred to a hardening step:

- **Structured logging** — `structlog` JSON in every service. Every log line has `service`, `correlation_id`, `level`, `timestamp`. Nginx generates `X-Correlation-ID` if absent; propagated through all downstream calls.
- **Type safety** — all Python passes `mypy --strict`. No untyped `Any` without explicit comment justification.
- **Error handling** — no bare `except`. Exceptions caught at service boundaries, logged with context, returned as structured responses.
- **Idempotency** — every DAG task, DB write, and MinIO upload is safe to re-run without side effects.
- **Resource limits** — every Docker container has `mem_limit` and `cpus` set.
- **No secrets in code** — `.env` only; validated on startup by Pydantic `BaseSettings`; `detect-secrets` pre-commit hook blocks accidental commits.
- **Database discipline** — all schema changes via Alembic migrations. No raw DDL outside initial bootstrap. SQLAlchemy async engine with connection pool.
- **Multi-stage Docker builds** — every `Dockerfile` uses build + runtime stages. Images < 1GB where possible.

---

## Implementation Phases

### Phase 1 — Infrastructure Foundation

**Goal:** All backing services running. No application code yet.

- [x] `shared/` package: `logging.py`, `db.py`, `storage.py`, `config.py`
- [x] `docker-compose.yml`: postgres, minio, redis — healthchecks, mem_limit, restart policy
- [x] `infrastructure/postgres/init/01_init_schemas.sql` — all schemas, tables, BRIN indexes
- [x] `infrastructure/postgres/postgresql.conf` — tuned config
- [x] Alembic setup + initial migration
- [x] MLflow server (postgres backend + MinIO artifact store)
- [x] `infrastructure/nginx/nginx.conf` — path-prefix routing to all services
- [x] `scripts/bootstrap.sh` — idempotent MinIO bucket creation + migrations
- [x] `.env.example` — all variables documented

**Exit criteria:** `make up` → all health checks green. Nginx at :80 routes to MLflow and MinIO UIs. Bootstrap runs clean twice.

---

### Phase 2 — Observability Stack

**Goal:** See everything before writing application code.

- [ ] Add Prometheus + Alertmanager + Loki + Promtail + Grafana to `docker-compose.yml`
- [ ] `infrastructure/prometheus/alerts/infra_alerts.yml`
- [ ] `infrastructure/alertmanager/alertmanager.yml`
- [ ] Promtail config — tail all Docker container logs → Loki
- [ ] Grafana: auto-provision Prometheus, PostgreSQL, Loki datasources
- [ ] `infrastructure/grafana/dashboards/infra.json` — CPU/memory/disk per container
- [ ] `infrastructure/grafana/dashboards/logs.json` — live Loki log viewer (filter by service, level, correlation_id)

**Exit criteria:** Grafana at :3000 shows infra metrics and live container logs.

---

### Phase 3 — Data Ingestion

**Goal:** Historical market data flowing into PostgreSQL via Airflow.

- [ ] `services/ingestion/src/fetchers/yahoo_finance.py` — retry, structured logging, upsert
- [ ] `services/ingestion/src/writers/postgres_writer.py` — async SQLAlchemy, ON CONFLICT DO UPDATE
- [ ] `services/ingestion/src/writers/minio_writer.py` — idempotent CSV archive
- [ ] `airflow/dags/ingestion_dag.py` — @hourly, SLA 30min, on_failure_callback
- [ ] Airflow services in docker-compose (webserver, scheduler, worker, airflow-init)
- [ ] `scripts/seed_data.sh` — 2 years daily OHLCV for: AAPL, MSFT, GOOGL, TSLA, SPY, QQQ, BTC-USD, ETH-USD

**Exit criteria:** `raw.ohlcv_data` > 5000 rows. Ingestion DAG logs visible in Loki/Grafana.

---

### Phase 4 — ETL and Feature Engineering

**Goal:** Raw OHLCV → engineered features + anomaly labels in feature store.

- [ ] `services/etl/src/transformers/technical_indicators.py` — RSI(14), MACD, BB(20), ATR(14), OBV
- [ ] `services/etl/src/transformers/rolling_stats.py` — z-scores, log returns, inter-day gaps
- [ ] `services/etl/src/transformers/label_generator.py` — configurable anomaly thresholds
- [ ] `services/etl/src/validators/expectations.py` — Great Expectations suite
- [ ] `services/etl/src/feature_store.py` — time-based train/val/test splits
- [ ] `airflow/dags/etl_dag.py` — @daily, GE failure = DAG failure (not warning)
- [ ] `notebooks/01_eda.ipynb` + `notebooks/02_feature_engineering.ipynb`

**Exit criteria:** Feature store populated. Class balance ~3-5% anomaly confirmed. EDA complete.

---

### Phase 5 — Model Training + MLflow

**Goal:** Working PyTorch model with full experiment tracking.

- [ ] `services/training/src/models/lstm_transformer.py` — LSTM(128, 2L) → TransformerEncoder(4 heads) → Linear → Sigmoid
- [ ] `services/training/src/datasets/time_series_dataset.py` — sliding window (60 bars), StandardScaler saved to MinIO
- [ ] `services/training/src/trainers/anomaly_trainer.py` — weighted BCELoss, MLflow autolog, early stopping on val F1, gradient clipping
- [ ] `airflow/dags/retraining_dag.py` — @weekly + triggered, promotion gate F1 > 0.02 improvement
- [ ] `notebooks/03_model_prototyping.ipynb`

**Exit criteria:** MLflow experiment run visible. Model at "Staging" in registry. Val F1 > 0.5. Scaler artifact in MinIO.

---

### Phase 6 — TensorRT Export Pipeline

**Goal:** PyTorch → ONNX → TensorRT with benchmarked speedup.

- [ ] `services/training/src/exporters/onnx_exporter.py` — dynamic axes, numerical equivalence check
- [ ] `services/training/src/exporters/tensorrt_exporter.py` — FP16, batch profiles 1/8/32, build_metadata.json
- [ ] `scripts/export_tensorrt.sh`
- [ ] `notebooks/04_tensorrt_benchmarks.ipynb` — latency table across all backends

**Exit criteria:** `.engine` + `build_metadata.json` in MinIO. ≥3x speedup vs PyTorch CPU verified.

---

### Phase 7 — FastAPI Inference Service

**Goal:** Production-grade inference API with auth, rate limiting, and metrics.

- [ ] `services/inference/src/core/model_loader.py` — TRT → ONNX → PyTorch fallback, MLflow hot-reload (60s poll)
- [ ] `services/inference/src/core/preprocessor.py` — async DB fetch, StandardScaler, correlation ID
- [ ] `services/inference/src/api/middleware/auth.py` — X-API-Key validation
- [ ] `services/inference/src/api/middleware/rate_limit.py` — slowapi: 100 req/min predict, 10 req/min batch
- [ ] `services/inference/src/api/middleware/metrics.py` — Prometheus counters, histograms, gauges
- [ ] Routes: POST /predict, POST /predict/batch, GET /health, GET /ready, GET /metrics
- [ ] `Dockerfile` (CPU, multi-stage) + `Dockerfile.tensorrt` (GPU+TRT, multi-stage)
- [ ] Integration tests: pytest-asyncio + FastAPI TestClient

**Exit criteria:** Auth works. Rate limiting verified. `/metrics` scraped by Prometheus.

---

### Phase 8 — Model Observability Dashboards

**Goal:** Drift detection, model KPIs, and alerting fully operational.

- [ ] `services/monitoring/src/drift/evidently_runner.py` — PSI per feature → Prometheus + drift_reports table
- [ ] `services/monitoring/src/metrics/model_metrics.py` — rolling F1/precision/recall/AUC (7d + 30d)
- [ ] `services/monitoring/src/metrics/prometheus_exporter.py`
- [ ] `infrastructure/prometheus/alerts/model_alerts.yml`
- [ ] `infrastructure/grafana/dashboards/model_kpis.json`
- [ ] `infrastructure/grafana/dashboards/data_drift.json`
- [ ] `airflow/dags/monitoring_dag.py` — @daily, TriggerDagRunOperator on drift threshold

**Exit criteria:** All 4 Grafana dashboards live with data. Alertmanager fires a test alert successfully.

---

### Phase 9 — CI/CD and Code Quality

**Goal:** Automated quality gates and image publishing on every push.

- [ ] `pyproject.toml` — ruff + mypy config
- [ ] `.pre-commit-config.yaml` — ruff, mypy, detect-secrets
- [ ] `.github/workflows/ci.yml` — lint, typecheck, test (testcontainers), build, trivy security scan
- [ ] `.github/workflows/cd.yml` — push images to registry tagged with git SHA

**Exit criteria:** Green CI on clean PR. Images in registry with commit SHA tag.

---

### Phase 10 — End-to-End Verification and Docs

**Goal:** Reproducible from zero, load tested, fully documented.

- [ ] Full teardown + rebuild: `docker-compose down -v && make up && make seed` → fully working
- [ ] Drift simulation: inject synthetic outliers → monitoring_dag detects → retraining triggered
- [ ] Load test: `locust` at 50 RPS → p99 latency < 50ms, rate limiting verified
- [ ] `notebooks/05_drift_analysis.ipynb`
- [ ] `README.md` — Mermaid architecture diagram, 3-command quickstart, env var reference, TensorRT gotchas

**Exit criteria:** `make test` passes with coverage report. README complete.

---

## Key Risks and Mitigations

| Risk                                      | Mitigation                                                                                                       |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| TRT engine is GPU/driver version-specific | Pin exact versions in Dockerfiles; always fall back to ONNX CPU; store `build_metadata.json` with every engine |
| Airflow inter-DAG deadlock                | `TriggerDagRunOperator` not sensors; `MAX_ACTIVE_RUNS_PER_DAG=1`; all tasks idempotent                       |
| Class imbalance (~3-5% anomaly rate)      | Weighted BCELoss with `pos_weight`; evaluate on F1 not accuracy; include random baseline in MLflow             |
| Docker service startup ordering           | `healthcheck` + `condition: service_healthy`; dedicated `airflow-init` container                           |
| GPU OOM (TRT + JupyterLab on same GPU)    | `CUDA_VISIBLE_DEVICES=0` in TRT container; max batch 32 in TRT profiles; `shm_size: 2gb`                     |
| Secrets leaking into git                  | `detect-secrets` pre-commit; `.env` gitignored; CI credential scan; Pydantic validates on startup            |
| Log correlation gaps                      | Nginx injects `X-Correlation-ID`; structlog context propagates it; Loki dashboard filters by it                |

---

## Progress Tracker

| Phase                                     | Status         |
| ----------------------------------------- | -------------- |
| Phase 1 — Infrastructure Foundation      | ✅ Complete     |
| Phase 2 — Observability Stack            | ⬜ Not started |
| Phase 3 — Data Ingestion                 | ⬜ Not started |
| Phase 4 — ETL and Feature Engineering    | ⬜ Not started |
| Phase 5 — Model Training + MLflow        | ⬜ Not started |
| Phase 6 — TensorRT Export Pipeline       | ⬜ Not started |
| Phase 7 — FastAPI Inference Service      | ⬜ Not started |
| Phase 8 — Model Observability Dashboards | ⬜ Not started |
| Phase 9 — CI/CD and Code Quality         | ⬜ Not started |
| Phase 10 — End-to-End Verification       | ⬜ Not started |
