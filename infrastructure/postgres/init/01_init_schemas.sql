-- Helix — PostgreSQL Schema Initialization
-- Runs automatically on first container start via docker-entrypoint-initdb.d
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Schemas
-- -----------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS predictions;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- -----------------------------------------------------------------------------
-- raw.ohlcv_data
-- Stores raw OHLCV data exactly as received from yfinance. No transformations.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw.ohlcv_data (
    id            BIGSERIAL PRIMARY KEY,
    ticker        VARCHAR(20)      NOT NULL,
    timestamp     TIMESTAMPTZ      NOT NULL,
    open          NUMERIC(18, 6)   NOT NULL,
    high          NUMERIC(18, 6)   NOT NULL,
    low           NUMERIC(18, 6)   NOT NULL,
    close         NUMERIC(18, 6)   NOT NULL,
    volume        BIGINT           NOT NULL,
    ingested_at   TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_ohlcv_ticker_timestamp UNIQUE (ticker, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker ON raw.ohlcv_data (ticker);
CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp ON raw.ohlcv_data USING BRIN
(timestamp);

-- -----------------------------------------------------------------------------
-- features.feature_vectors
-- Engineered features + anomaly labels produced by the ETL service.
-- features column holds the full feature dict as JSONB for schema flexibility.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS features.feature_vectors (
    id            BIGSERIAL PRIMARY KEY,
    ticker        VARCHAR(20)      NOT NULL,
    timestamp     TIMESTAMPTZ      NOT NULL,
    features      JSONB            NOT NULL,
    label         SMALLINT         NOT NULL CHECK (label IN (0, 1)),
    split         VARCHAR(10)      NOT NULL CHECK (split IN ('train', 'val',
'test')),
    created_at    TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_features_ticker_timestamp UNIQUE (ticker, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_features_ticker ON features.feature_vectors (ticker);
CREATE INDEX IF NOT EXISTS idx_features_timestamp ON features.feature_vectors USING
BRIN (timestamp);
CREATE INDEX IF NOT EXISTS idx_features_split ON features.feature_vectors (split);
CREATE INDEX IF NOT EXISTS idx_features_label ON features.feature_vectors (label);

-- -----------------------------------------------------------------------------
-- predictions.inference_log
-- Every prediction request is logged here for KPI computation and drift analysis.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions.inference_log (
    id              BIGSERIAL PRIMARY KEY,
    request_id      UUID             NOT NULL UNIQUE,
    ticker          VARCHAR(20)      NOT NULL,
    timestamp       TIMESTAMPTZ      NOT NULL,
    features        JSONB            NOT NULL,
    score           NUMERIC(8, 6)    NOT NULL CHECK (score >= 0 AND score <= 1),
    label           SMALLINT         NOT NULL CHECK (label IN (0, 1)),
    model_version   VARCHAR(50)      NOT NULL,
    backend         VARCHAR(20)      NOT NULL,
    latency_ms      NUMERIC(10, 3)   NOT NULL,
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_ticker ON predictions.inference_log
(ticker);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions.inference_log
USING BRIN (timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions.inference_log
 (model_version);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions.inference_log
USING BRIN (created_at);

-- -----------------------------------------------------------------------------
-- monitoring.model_metrics
-- Rolling KPIs computed by the monitoring service (F1, precision, recall, AUC).
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS monitoring.model_metrics (
    id              BIGSERIAL PRIMARY KEY,
    metric_name     VARCHAR(50)      NOT NULL,
    metric_value    NUMERIC(10, 6)   NOT NULL,
    window_days     SMALLINT         NOT NULL,
    window_start    TIMESTAMPTZ      NOT NULL,
    window_end      TIMESTAMPTZ      NOT NULL,
    model_version   VARCHAR(50)      NOT NULL,
    computed_at     TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_metrics_name ON monitoring.model_metrics
(metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_computed_at ON monitoring.model_metrics USING
BRIN (computed_at);
CREATE INDEX IF NOT EXISTS idx_metrics_model_version ON monitoring.model_metrics
(model_version);

-- -----------------------------------------------------------------------------
-- monitoring.drift_reports
-- Evidently drift report output per run. triggered_retraining flags whether
-- this report caused a retraining DAG to be triggered.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS monitoring.drift_reports (
    id                    BIGSERIAL PRIMARY KEY,
    report                JSONB            NOT NULL,
    triggered_retraining  BOOLEAN          NOT NULL DEFAULT FALSE,
    created_at            TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_drift_created_at ON monitoring.drift_reports USING
BRIN (created_at);

-- -----------------------------------------------------------------------------
-- monitoring.pipeline_events
-- DAG failure callbacks write here. Used for alerting and audit trail.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS monitoring.pipeline_events (
    id          BIGSERIAL PRIMARY KEY,
    dag_id      VARCHAR(100)     NOT NULL,
    task_id     VARCHAR(100),
    status      VARCHAR(20)      NOT NULL,
    message     TEXT,
    created_at  TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_events_dag_id ON monitoring.pipeline_events (dag_id);
CREATE INDEX IF NOT EXISTS idx_events_created_at ON monitoring.pipeline_events USING
BRIN (created_at);