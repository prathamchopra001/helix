# services/frontend/src/app/main.py
"""Predict page — submit a ticker, get an anomaly prediction."""
from __future__ import annotations

import os

import requests
import streamlit as st

st.set_page_config(
    page_title="Helix · Predict",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
                version = data.get("model_version", "—")
                backend = data.get("backend", "—")
                st.success(f"**{version}** · {backend}")
            else:
                st.warning("Model loading…")
        except requests.exceptions.RequestException:
            st.error("Inference unreachable")
        st.divider()


_render_sidebar()


# ── Page body ─────────────────────────────────────────────────────────────────

st.title("Predict")
st.caption("Run a live anomaly prediction against the loaded model.")

ticker_input = st.text_input(
    "Ticker symbol", placeholder="AAPL", max_chars=20, label_visibility="visible"
)
run = st.button("Run Prediction", type="primary", disabled=not ticker_input.strip())

if run and ticker_input.strip():
    ticker = ticker_input.strip().upper()
    with st.spinner(f"Running inference for **{ticker}**…"):
        try:
            resp = requests.post(
                f"{INFERENCE_URL}/predict",
                json={"ticker": ticker},
                headers=_HEADERS,
                timeout=10,
            )
        except requests.exceptions.ConnectionError:
            st.error("Inference service unreachable — is the stack running?")
            st.stop()

    def _detail(r: requests.Response) -> str:
        try:
            return r.json().get("detail", r.text)
        except Exception:
            return r.text

    if resp.status_code == 503:
        st.warning("Model is loading — try again in a moment.")
        st.stop()
    if resp.status_code == 422:
        st.error(f"Preprocessing failed: {_detail(resp)}")
        st.stop()
    if not resp.ok:
        st.error(f"Error {resp.status_code}: {_detail(resp)}")
        st.stop()

    data = resp.json()
    label: int = data["label"]
    score: float = data["score"]

    # ── Result card ───────────────────────────────────────────────────────────
    st.divider()
    colour = "#ef4444" if label == 1 else "#22c55e"
    text = "ANOMALY" if label == 1 else "NORMAL"
    st.markdown(
        f"""<div style="background:{colour};padding:20px 32px;border-radius:10px;
        font-size:2.2rem;font-weight:700;letter-spacing:0.12em;color:#fff;
        text-align:center;font-family:monospace">{text}</div>""",
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.progress(score, text=f"Anomaly score: **{score:.6f}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Latency", f"{data['latency_ms']:.1f} ms")
    c2.metric("Model version", data.get("model_version", "—"))
    c3.metric("Backend", data.get("backend", "—"))
