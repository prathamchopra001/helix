"""Compute technical indicators for a single ticker's OHLCV DataFrame."""
import pandas as pd
import ta

from shared.logging import get_logger

log = get_logger(__name__)


def add_technical_indicators(df: pd.DataFrame, correlation_id: str = "") -> pd.DataFrame:
    """
    Add RSI, MACD, Bollinger Bands, ATR, and OBV columns to df.

    Input df must have columns: open, high, low, close, volume.
    Returns a new DataFrame with additional indicator columns.
    NaN rows (from lookback periods) are dropped.
    """
    df = df.copy()

    # RSI — momentum oscillator, 14-period lookback
    df["rsi_14"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()

    # MACD — trend/momentum, standard 12/26/9 params
    macd = ta.trend.MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_histogram"] = macd.macd_diff()

    # Bollinger Bands — volatility envelope around 20-day SMA
    bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()   # (upper - lower) / middle
    df["bb_pct"] = bb.bollinger_pband()     # where price sits within the band (0=lower, 1=upper)

    # ATR — average daily price range (volatility)
    df["atr_14"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()

    # OBV — cumulative volume direction
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(
        close=df["close"], volume=df["volume"]
    ).on_balance_volume()

    before = len(df)
    df = df.dropna()
    dropped = before - len(df)

    log.info(
        "indicators_computed",
        rows_in=before,
        rows_out=len(df),
        nan_rows_dropped=dropped,
        correlation_id=correlation_id,
    )
    return df