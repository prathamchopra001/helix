# services/frontend/src/app/pages/1_History.py
"""History page — browse recent predictions from the inference log."""

from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import psycopg2
import requests
import streamlit as st
from db import query

st.set_page_config(page_title="Helix · History", page_icon="⬡", layout="wide")

INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://localhost:8000")
API_KEY = os.environ["INFERENCE_API_KEY"]
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


# ── Data fetch ────────────────────────────────────────────────────────────────

st.title("History")
st.caption("Last 100 predictions from the inference log.")

try:
    rows = query(
        """
        SELECT ticker, score, label, model_version, backend,
               latency_ms, created_at
        FROM predictions.inference_log
        ORDER BY created_at DESC
        LIMIT 100
        """
    )
except psycopg2.Error as exc:
    st.error(f"Database error: {exc}")
    st.stop()

if not rows:
    st.info("No predictions yet — run some predictions on the Predict page first.")
    st.stop()

df = pd.DataFrame(rows)
df["created_at"] = pd.to_datetime(df["created_at"])
df["score"] = df["score"].astype(float)
df["latency_ms"] = df["latency_ms"].astype(float)


# ── Summary metrics ───────────────────────────────────────────────────────────

today = pd.Timestamp.now(tz="UTC").date()
today_df = df[df["created_at"].dt.date == today]

c1, c2, c3 = st.columns(3)
c1.metric("Predictions today", len(today_df))
anomaly_rate = (df["label"] == 1).mean() * 100
c2.metric("Anomaly rate (all)", f"{anomaly_rate:.1f}%")
c3.metric("Avg latency (all)", f"{df['latency_ms'].mean():.1f} ms")

st.divider()


# ── Score chart ───────────────────────────────────────────────────────────────

chart_df = df.sort_values("created_at")
chart_df["outcome"] = chart_df["label"].map({1: "Anomaly", 0: "Normal"})

fig = px.scatter(
    chart_df,
    x="created_at",
    y="score",
    color="outcome",
    color_discrete_map={"Anomaly": "#ef4444", "Normal": "#22c55e"},
    template="plotly_dark",
    title="Anomaly score over time",
    labels={"created_at": "Time", "score": "Anomaly score", "outcome": ""},
)
fig.update_layout(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font_family="monospace",
    yaxis={"range": [0, 1]},
)
st.plotly_chart(fig, use_container_width=True)

st.divider()


# ── Filtered table ────────────────────────────────────────────────────────────

col_filter, col_label = st.columns([3, 1])
ticker_filter = col_filter.text_input("Filter by ticker", placeholder="AAPL")
label_filter = col_label.selectbox("Label", ["All", "Anomaly", "Normal"])

display_df = df.copy()
if ticker_filter.strip():
    display_df = display_df[
        display_df["ticker"].str.contains(ticker_filter.strip().upper(), case=False)
    ]
if label_filter == "Anomaly":
    display_df = display_df[display_df["label"] == 1]
elif label_filter == "Normal":
    display_df = display_df[display_df["label"] == 0]

display_df["label"] = display_df["label"].map({1: "ANOMALY", 0: "NORMAL"})
display_df = display_df.rename(
    columns={
        "ticker": "Ticker",
        "score": "Score",
        "label": "Label",
        "model_version": "Model version",
        "backend": "Backend",
        "latency_ms": "Latency (ms)",
        "created_at": "Timestamp",
    }
)
st.dataframe(display_df, use_container_width=True, hide_index=True)
