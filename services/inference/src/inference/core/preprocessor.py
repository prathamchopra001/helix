"""
Preprocessor: given a ticker symbol, fetch the last 60 OHLCV bars from
postgres.features.feature_vectors and apply the StandardScaler that was
saved during training.

Why 60 bars?
  The model was trained on sliding windows of 60 trading days. To predict
  whether today's market behavior is anomalous, we need the 60 most recent
  feature vectors for that ticker.

Why the scaler from training?
  The model was trained on scaled features. At inference time we must apply
  the exact same scaling transform — using the scaler fit on the training
  split — or the input distribution won't match what the model expects.
"""
import json

import numpy as np
import psycopg2

from inference.config import settings
from inference.core.model_loader import LoadedModel
from shared.logging import get_logger

log = get_logger(__name__)

SEQ_LEN = 60
MIN_FEATURES = 10  # sanity floor — catches empty/corrupt rows, not exact count


class PreprocessingError(Exception):
    pass


def fetch_feature_window(ticker: str, correlation_id: str = "") -> np.ndarray:
    """
    Fetch the 60 most recent feature vectors for ticker from features schema.
    Returns shape (60, N_FEATURES) float32 array.
    Feature count is dynamic — determined by what's in the DB.
    Raises PreprocessingError if not enough data.
    """
    query = """
        SELECT features
        FROM features.feature_vectors
        WHERE ticker = %s
        ORDER BY timestamp DESC
        LIMIT %s
    """
    try:
        conn = psycopg2.connect(settings.dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(query, (ticker, SEQ_LEN))
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as exc:
        raise PreprocessingError(f"DB fetch failed for {ticker}: {exc}") from exc

    if len(rows) < SEQ_LEN:
        raise PreprocessingError(
            f"Not enough data for {ticker}: need {SEQ_LEN} bars, got {len(rows)}"
        )

    # rows are newest-first — reverse to get chronological order
    rows = list(reversed(rows))

    # Each row[0] is a JSONB dict of feature_name → value
    vectors = []
    for (features_json,) in rows:
        features = json.loads(features_json) if isinstance(features_json, str) else features_json
        # Sort keys to guarantee consistent feature order (matches training)
        values = [float(features[k]) for k in sorted(features.keys())]
        vectors.append(values)

    arr = np.array(vectors, dtype=np.float32)  # shape (60, n_features)

    if arr.shape[1] < MIN_FEATURES:
        raise PreprocessingError(
            f"Too few features for {ticker}: got {arr.shape[1]}, minimum {MIN_FEATURES}"
        )

    log.info(
        "features_fetched",
        ticker=ticker,
        shape=list(arr.shape),
        correlation_id=correlation_id,
    )
    return arr


def preprocess(
    ticker: str,
    model: LoadedModel,
    correlation_id: str = "",
) -> np.ndarray:
    """
    Fetch features and apply scaler. Returns shape (1, 60, n_features) float32 ready for inference.
    """
    window = fetch_feature_window(ticker, correlation_id=correlation_id)

    if model.scaler is not None:
        # Scaler expects (n_samples, n_features) — reshape, scale, reshape back
        seq_len, n_features = window.shape
        flat = window.reshape(-1, n_features)
        flat = model.scaler.transform(flat).astype(np.float32)
        window = flat.reshape(seq_len, n_features)
    else:
        log.warning("no_scaler_available", ticker=ticker, note="using raw unscaled features")

    # Add batch dimension: (60, 25) → (1, 60, 25)
    return window[np.newaxis, ...]
