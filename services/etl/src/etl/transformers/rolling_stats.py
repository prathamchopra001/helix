"""
Compute rolling statistical features for a single ticker's OHLCV DataFrame.

What this adds and why the model needs it:
- Log returns:      The % change in price expressed as ln(today/yesterday).
                    Better than raw % change because it's symmetric and additive.
- Z-scores:         How many standard deviations away from the rolling mean is
                    today's close? A z-score of ±2.5 is unusual. The model uses
                    multiple windows (5/10/20/60 days) to capture short and long
                    term unusualness.
- Volume ratios:    Today's volume divided by the N-day average. A ratio of 3.0
                    means 3x normal volume — a classic anomaly signal.
- Gap feature:      How many calendar days since the last trading day. Normally 1
                    (weekdays) or 3 (Monday after weekend). Larger gaps = halts,
                    holidays, or exchange issues.
- return_5d/20d:    Cumulative 5-day and 20-day price return. Captures short-term
                    momentum and monthly trend. Anomalies often follow or coincide
                    with sharp multi-day moves.
- hl_range:         (high - low) / close — intraday price range as a fraction of
                    price. Spikes on high-volatility / crisis days.
- overnight_gap:    (open - prev_close) / prev_close — how much the price gapped
                    overnight. Large gaps signal news events between sessions.
"""
import numpy as np
import pandas as pd

from shared.logging import get_logger

log = get_logger(__name__)

_Z_WINDOWS = [5, 10, 20, 60]
_VOL_WINDOWS = [5, 10, 20]


def add_rolling_stats(df: pd.DataFrame, correlation_id: str = "") -> pd.DataFrame:
    """
    Add log returns, z-scores, volume ratios, and gap features to df.

    Input df must have columns: timestamp (datetime index or column),
    close, volume. Returns a new DataFrame with additional columns.
    NaN rows from rolling windows are dropped.
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    # Log returns — daily percentage move expressed in log space
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Z-scores of close price over multiple lookback windows
    for w in _Z_WINDOWS:
        roll = df["close"].rolling(w)
        df[f"z_score_{w}d"] = (df["close"] - roll.mean()) / roll.std()

    # Volume ratios — today's volume vs N-day average
    for w in _VOL_WINDOWS:
        avg = df["volume"].rolling(w).mean()
        df[f"volume_ratio_{w}d"] = df["volume"] / avg

    # Inter-day gap — calendar days since previous trading day
    ts = pd.to_datetime(df["timestamp"])
    df["day_gap"] = ts.diff().dt.days.fillna(1).astype(int)

    # Multi-day momentum — cumulative returns over 5 and 20 days
    df["return_5d"] = df["close"].pct_change(5)
    df["return_20d"] = df["close"].pct_change(20)

    # Intraday range — high/low spread as fraction of close price
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]

    # Overnight gap — today's open vs yesterday's close (news / after-hours events)
    df["overnight_gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    before = len(df)
    df = df.dropna()
    dropped = before - len(df)

    log.info(
        "rolling_stats_computed",
        rows_in=before,
        rows_out=len(df),
        nan_rows_dropped=dropped,
        correlation_id=correlation_id,
    )
    return df
