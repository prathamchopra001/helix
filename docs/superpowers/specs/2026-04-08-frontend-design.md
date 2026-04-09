# Helix Frontend — Design Spec
**Date:** 2026-04-08
**Status:** Approved

---

## Overview

A Streamlit demo/showcase frontend for the Helix anomaly detection platform. Three pages: run predictions against live tickers, browse prediction history, and inspect model health. Runs as a Docker service on the internal Compose network, connecting directly to the inference container and PostgreSQL.

---

## Architecture

### New service: `services/frontend/`

```
services/frontend/
├── Dockerfile
├── requirements.txt
└── src/
    └── app/
        ├── main.py              # entry point, shared config, sidebar nav
        ├── pages/
        │   ├── 1_predict.py     # Page 1 — Predict
        │   ├── 2_history.py     # Page 2 — History
        │   └── 3_health.py      # Page 3 — Model Health
        └── db.py                # psycopg2 helper using POSTGRES_* env vars
```

### Docker Compose addition

```yaml
frontend:
  build: services/frontend
  environment:
    INFERENCE_URL: http://inference:8000
    POSTGRES_HOST: postgres
    POSTGRES_PORT: 5432
    POSTGRES_DB: helix
    POSTGRES_USER: helix
    POSTGRES_PASSWORD: helix
  depends_on: [inference, postgres]
  ports:
    - "8501:8501"
```

### Nginx route

```nginx
location /frontend/ {
    proxy_pass http://frontend:8501/;
}
```

No changes to any existing service.

---

## Pages

### Page 1 — Predict

- Ticker text input (e.g. `AAPL`) + "Run Prediction" button
- Calls `POST http://inference:8000/predict` with `{"ticker": "<input>"}`
- Result card:
  - Large badge: `ANOMALY` (red) or `NORMAL` (green)
  - Anomaly score as a progress bar (0.0 → 1.0)
  - Three metric columns: latency (ms), model version, backend (TRT / ONNX / PyTorch)
- Error states:
  - 503 model not loaded → amber warning: "Model loading, try again in a moment"
  - API unreachable → red banner: "Inference service unavailable"
  - Unknown/invalid ticker → red banner with API error detail

### Page 2 — History

- Queries `predictions.inference_log` — last 100 rows ordered by `created_at DESC`
- Summary row: predictions today, anomaly rate (%), avg latency (ms)
- Line chart: anomaly score over time, points coloured red (anomaly) / green (normal) — Plotly
- Data table with ticker text filter input + label dropdown (All / Anomaly / Normal): columns are ticker, score, label, model version, backend, timestamp
- Empty state: friendly message when table has no rows (fresh stack)

### Page 3 — Model Health

Three sections:

1. **Current model** — version, backend, loaded-at timestamp (from `GET /health`)
2. **KPIs** — latest F1, precision, recall, AUC from `monitoring.model_metrics` — metric cards
3. **Drift** — last row from `monitoring.drift_reports`: PSI score per feature (bar chart), retraining-triggered flag, timestamp

Empty states for all three sections when tables have no data yet.

---

## Visual Style

Dark terminal + finance theme.

### Streamlit theme (`.streamlit/config.toml`)

```toml
[theme]
base = "dark"
primaryColor = "#f0b429"
backgroundColor = "#0d1117"
secondaryBackgroundColor = "#161b22"
textColor = "#e6edf3"
font = "monospace"
```

### Colour conventions (consistent across all pages)

| Meaning | Colour |
|---|---|
| Anomaly | `#ef4444` (red) |
| Normal | `#22c55e` (green) |
| Accent / headings | `#f0b429` (amber) |
| Muted text | `#8b949e` (grey) |

### Sidebar

- "HELIX" logo text in amber
- Navigation links to all 3 pages
- Status pill: current model version + backend — green if loaded, red if unavailable
- Refreshes on each page load (no polling)

---

## Data Connections

| Page | Source | Method |
|---|---|---|
| Predict | `inference:8000/predict` | `requests.post()` |
| History | `predictions.inference_log` | psycopg2 direct |
| Health — model info | `inference:8000/health` | `requests.get()` |
| Health — KPIs | `monitoring.model_metrics` | psycopg2 direct |
| Health — drift | `monitoring.drift_reports` | psycopg2 direct |

`db.py` is a minimal psycopg2 helper (~20 lines). No dependency on the `shared` package — the frontend is a standalone service.

---

## Error Handling

- All HTTP calls and DB queries wrapped in `try/except`
- User-facing errors via `st.error()` (red banner) or `st.warning()` (amber)
- DB connection failure shows the specific error message (useful for local dev)
- Empty tables show a friendly empty-state message, never a broken chart
- No retries — Streamlit's "rerun" button serves as the manual retry

---

## Testing

No automated tests. The frontend is a demo/showcase — UI testing overhead is not justified. Smoke-tested manually: `docker compose up` → open `/frontend`, run a prediction, verify all three pages load with data.

---

## Out of Scope

- Authentication (the inference API already handles API key auth at the Nginx level)
- Triggering retraining or managing tickers from the UI
- Real-time WebSocket updates / auto-refresh
- Mobile responsiveness
