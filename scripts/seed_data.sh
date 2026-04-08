#!/usr/bin/env bash
# =============================================================================
# Helix — Seed Script
# Backfills 2 years of daily OHLCV data for all default tickers.
# Run once after bootstrap. Safe to re-run — upsert semantics.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  echo "ERROR: .env not found."
  exit 1
fi

set -a; source .env; set +a

GATEWAY_IP=$(ip route show default | awk '/default/ {print $3}')

# Override endpoints: seed runs outside Docker, so use host-accessible addresses
export MINIO_ENDPOINT="localhost:9000"
export MINIO_SECURE="false"
export DATABASE_URL="postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${GATEWAY_IP}:5432/${POSTGRES_DB}?ssl=disable"
export POSTGRES_HOST="${GATEWAY_IP}"

echo "==> Seeding 5 years of OHLCV data"
echo "    Tickers: AAPL MSFT GOOGL TSLA SPY QQQ BTC-USD ETH-USD NVDA AMZN META"
echo "    DB:      ${GATEWAY_IP}:5432/${POSTGRES_DB}"
echo "    MinIO:   ${MINIO_ENDPOINT}"
echo ""

python3 - <<'PYEOF'
import sys
sys.path.insert(0, "shared/src")
sys.path.insert(0, "services/ingestion/src")

from shared.logging import configure_logging
from ingestion.main import run_ingestion

configure_logging("seed")

results = run_ingestion(
    tickers=["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ", "BTC-USD", "ETH-USD", "NVDA", "AMZN", "META"],
    period="5y",
    correlation_id="seed-extended",
)

total = sum(results.values())
print(f"\nDone — {total} total rows across {len(results)} tickers:")
for ticker, rows in sorted(results.items()):
    status = "OK" if rows > 0 else "FAILED"
    print(f"  [{status}] {ticker}: {rows} rows")

if any(r == 0 for r in results.values()):
    sys.exit(1)
PYEOF

echo ""
echo "==> Row count in raw.ohlcv_data:"
PGPASSWORD="${POSTGRES_PASSWORD}" psql \
  -h "${GATEWAY_IP}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
  -c "SELECT ticker, COUNT(*) AS rows, MIN(timestamp)::date AS earliest, MAX(timestamp)::date AS latest FROM raw.ohlcv_data GROUP BY ticker ORDER BY ticker;"
