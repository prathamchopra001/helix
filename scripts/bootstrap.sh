#!/usr/bin/env bash
# =============================================================================
# Helix — Bootstrap Script
# Idempotent: safe to run multiple times. Run after `make up`.
# Creates MinIO buckets and stamps Alembic baseline.
# =============================================================================
set -euo pipefail
# Load .env
if [ ! -f .env ]; then
  echo "ERROR: .env file not found. Copy .env.example to .env and fill in values."
  exit 1
fi
set -a
source .env
set +a
echo "==> Waiting for PostgreSQL to be ready..."
until docker compose exec -T postgres pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" > /dev/null 2>&1; do
  sleep 2
done
echo "    PostgreSQL is ready."
echo "==> Waiting for MinIO to be ready..."
until curl -sf "http://localhost:9000/minio/health/live" > /dev/null 2>&1; do
  sleep 2
done
echo "    MinIO is ready."
echo "==> Creating MinIO buckets..."
docker compose exec -T minio mc alias set local http://localhost:9000 "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY" > /dev/null 2>&1 || true
for bucket in "$MINIO_BUCKET_RAW" "$MINIO_BUCKET_MODELS" "$MINIO_BUCKET_MLFLOW"; do
  if docker compose exec -T minio mc ls "local/$bucket" > /dev/null 2>&1; then
    echo "    Bucket '$bucket' already exists — skipping."
  else
    docker compose exec -T minio mc mb "local/$bucket"
    echo "    Bucket '$bucket' created."
  fi
done
echo "==> Creating Airflow database..."
docker compose exec -T postgres psql -U "$POSTGRES_USER" -c "CREATE DATABASE airflow OWNER $POSTGRES_USER;" 2>/dev/null \
  && echo "    Airflow database created." \
  || echo "    Airflow database already exists — skipping."

echo "==> Stamping Alembic baseline..."
alembic upgrade head
echo "    Alembic migrations applied."
echo ""
echo "Bootstrap complete."
echo ""
echo "  MLflow:   http://localhost/mlflow"
echo "  MinIO:    http://localhost/minio"
echo "  Grafana:  http://localhost/grafana  (Phase 2)"
echo "  Airflow:  http://localhost/airflow  (Phase 3)"