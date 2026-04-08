"""
Dataset that loads feature vectors from PostgreSQL and creates sliding windows.

What a "sliding window" means:
  The model needs context — it can't predict anomalies from a single day.
  We feed it 60 consecutive days at a time. For a ticker with 410 rows,
  we get 410 - 60 = 350 windows. Each window's label is the label of its
  LAST day (the day we're predicting about).

  Day:  1  2  3 ... 59  60  ← window 1, label = label of day 60
  Day:  2  3  4 ... 60  61  ← window 2, label = label of day 61
  ...

Scaling:
  StandardScaler is fit ONLY on the train split. Val and test are transformed
  using the train scaler. This prevents data leakage — the model must never
  "see" the scale of future data during training.
  The fitted scaler is serialized to MinIO so the inference service can use
  the exact same scaling at prediction time.

Split handling:
  Rows in features.feature_vectors have a 'split' column (train/val/test).
  Windows that contain rows from multiple splits are assigned to the split
  of their last day. This keeps the split assignment clean.
"""
import io
import json
import os
import pickle

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from shared.logging import get_logger
from shared.storage import get_minio_client

log = get_logger(__name__)

WINDOW_SIZE = 60

# FEATURE_COLS is derived dynamically from the DB at load time (see load_features_df).
# A sorted list of all keys found in the features JSONB column is used — this means
# adding new features to the ETL pipeline automatically flows through to training
# without any changes here. The only constraint is that inference must use the same
# sorted-key ordering, which the preprocessor already does.
FEATURE_COLS: list[str] = []  # populated by load_features_df()

_FETCH_SQL = """
    SELECT ticker, timestamp, features, label, split
    FROM features.feature_vectors
    ORDER BY ticker, timestamp ASC
"""


def _get_dsn() -> str:
    return (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ['POSTGRES_HOST']}:{os.environ.get('POSTGRES_PORT', '5432')}"
        f"/{os.environ['POSTGRES_DB']}"
    )


def load_features_df() -> pd.DataFrame:
    """
    Load all feature vectors from the DB into a DataFrame.
    Feature columns are derived from the sorted JSON keys of the first row —
    adding features to the ETL automatically flows through to training.
    """
    global FEATURE_COLS

    conn = psycopg2.connect(_get_dsn())
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_FETCH_SQL)
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        raise RuntimeError("No feature vectors found in the database")

    # Derive feature columns from the first row — sorted for reproducibility
    first_feat = json.loads(rows[0]["features"]) if isinstance(rows[0]["features"], str) else rows[0]["features"]
    FEATURE_COLS = sorted(first_feat.keys())
    log.info("feature_cols_detected", n_features=len(FEATURE_COLS), cols=FEATURE_COLS)

    records = []
    for row in rows:
        feat = json.loads(row["features"]) if isinstance(row["features"], str) else row["features"]
        record = {
            "ticker": row["ticker"],
            "timestamp": row["timestamp"],
            "label": int(row["label"]),
            "split": row["split"],
        }
        for col in FEATURE_COLS:
            record[col] = float(feat.get(col, float("nan")))
        records.append(record)

    df = pd.DataFrame(records)
    log.info("features_loaded", rows=len(df), n_features=len(FEATURE_COLS))
    return df


def build_windows(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Build sliding windows from the feature DataFrame.

    Returns:
        X: (n_windows, WINDOW_SIZE, n_features) float32 array
        y: (n_windows,) int array of labels
        splits: (n_windows,) array of split names
        scaler: fitted StandardScaler
    """
    if scaler is None:
        scaler = StandardScaler()

    # Fit scaler on train rows only — never on val/test
    if fit_scaler:
        train_df = df[df["split"] == "train"]
        scaler.fit(train_df[FEATURE_COLS].values)

    # Scale all features using the train-fitted scaler
    df = df.copy()
    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS].values).astype(np.float32)

    X_list, y_list, split_list = [], [], []

    for ticker in df["ticker"].unique():
        tdf = df[df["ticker"] == ticker].sort_values("timestamp").reset_index(drop=True)
        features = tdf[FEATURE_COLS].values.astype(np.float32)
        labels = tdf["label"].values
        splits = tdf["split"].values

        for i in range(WINDOW_SIZE, len(tdf)):
            window = features[i - WINDOW_SIZE:i]
            label = labels[i]
            split = splits[i]
            X_list.append(window)
            y_list.append(label)
            split_list.append(split)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    splits_arr = np.array(split_list)

    log.info("windows_built", total=len(X), anomalies=int(y.sum()))
    return X, y, splits_arr, scaler


def save_scaler(scaler: StandardScaler, model_version: str) -> str:
    """Serialize scaler to MinIO. Returns the MinIO key."""
    bucket = os.environ.get("MINIO_BUCKET_MODELS", "models")
    key = f"scalers/{model_version}/scaler.pkl"
    data = pickle.dumps(scaler)
    client = get_minio_client()
    client.put_object(bucket, key, io.BytesIO(data), length=len(data))
    log.info("scaler_saved", bucket=bucket, key=key)
    return key


class AnomalyWindowDataset(Dataset):
    """PyTorch Dataset wrapping pre-built windows."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        import torch
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx]
