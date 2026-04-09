# services/frontend/src/app/pages/2_Health.py
"""Model Health page — current model, KPIs, and drift status."""
from __future__ import annotations

import json
import os

import plotly.graph_objects as go
import requests
import streamlit as st

from db import query

st.set_page_config(page_title="Helix · Health", page_icon="⬡", layout="wide")

INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://localhost:8000")
API_KEY = os.environ.get("INFERENCE_API_KEY", "dev-key")
_HEADERS = {"X-API-Key": API_KEY}


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            "<h2 style='color:#f0b429;letter-spacing:0.15em'>⬡ HELIX</h2>",
            unsafe_allow_html=True,
        )
        st.caption("Market Anomaly Detection")
        st.divider()
        try:
            r = requests.get(f"{INFERENCE_URL}/ready", headers=_HEADERS, timeout=3)
            if r.ok:
                data = r.json()
                st.success(f"**{data.get('model_version', '—')}** · {data.get('backend', '—')}")
            else:
                st.warning("Model loading…")
        except requests.exceptions.RequestException:
            st.error("Inference unreachable")
        st.divider()


_render_sidebar()

st.title("Model Health")

# ── Section 1: Current model ──────────────────────────────────────────────────

st.subheader("Current model")
try:
    r = requests.get(f"{INFERENCE_URL}/ready", headers=_HEADERS, timeout=3)
    if r.ok:
        data = r.json()
        c1, c2 = st.columns(2)
        c1.metric("Version", data.get("model_version", "—"))
        c2.metric("Backend", data.get("backend", "—"))
    else:
        st.warning("No model loaded yet.")
except requests.exceptions.RequestException as exc:
    st.error(f"Could not reach inference service: {exc}")

st.divider()


# ── Section 2: KPIs ───────────────────────────────────────────────────────────

st.subheader("KPIs")
try:
    kpi_rows = query(
        """
        SELECT DISTINCT ON (metric_name)
            metric_name, metric_value, window_days, model_version, computed_at
        FROM monitoring.model_metrics
        ORDER BY metric_name, computed_at DESC
        """
    )
except Exception as exc:
    st.error(f"Database error: {exc}")
    kpi_rows = []

if not kpi_rows:
    st.info("No KPI data yet — the monitoring DAG hasn't run.")
else:
    kpi_map = {r["metric_name"]: float(r["metric_value"]) for r in kpi_rows}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("F1", f"{kpi_map.get('f1', 0):.4f}")
    c2.metric("Precision", f"{kpi_map.get('precision', 0):.4f}")
    c3.metric("Recall", f"{kpi_map.get('recall', 0):.4f}")
    c4.metric("AUC-ROC", f"{kpi_map.get('auc_roc', 0):.4f}")
    st.caption(
        f"Window: {kpi_rows[0]['window_days']}d · "
        f"Model: {kpi_rows[0]['model_version']} · "
        f"Computed: {kpi_rows[0]['computed_at']}"
    )

st.divider()


# ── Section 3: Drift ──────────────────────────────────────────────────────────

st.subheader("Drift")
try:
    drift_rows = query(
        """
        SELECT report, triggered_retraining, created_at
        FROM monitoring.drift_reports
        ORDER BY created_at DESC
        LIMIT 1
        """
    )
except Exception as exc:
    st.error(f"Database error: {exc}")
    drift_rows = []

if not drift_rows:
    st.info("No drift reports yet — the monitoring DAG hasn't run.")
else:
    row = drift_rows[0]
    report = row["report"] if isinstance(row["report"], dict) else json.loads(row["report"])
    psi_scores: dict[str, float] = report.get("psi_scores", {})
    n_drifted: int = report.get("n_features_drifted", 0)

    col_meta1, col_meta2, col_meta3 = st.columns(3)
    col_meta1.metric("Features drifted", n_drifted)
    col_meta2.metric(
        "Retraining triggered",
        "YES" if row["triggered_retraining"] else "NO",
    )
    col_meta3.metric("Report time", str(row["created_at"])[:19])

    if psi_scores:
        features = sorted(psi_scores.keys())
        values = [psi_scores[f] for f in features]
        colours = ["#ef4444" if v > 0.2 else "#22c55e" for v in values]

        fig = go.Figure(
            go.Bar(
                x=features,
                y=values,
                marker_color=colours,
                name="PSI",
            )
        )
        fig.add_hline(
            y=0.2,
            line_dash="dash",
            line_color="#f0b429",
            annotation_text="Drift threshold (0.2)",
            annotation_position="top right",
        )
        fig.update_layout(
            title="PSI per feature (red = drifted)",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font_family="monospace",
            template="plotly_dark",
            xaxis_tickangle=-45,
            yaxis_title="PSI score",
        )
        st.plotly_chart(fig, use_container_width=True)
