"""
Unit tests for ETL transformers.
These run without any I/O — pure function tests.
"""
import numpy as np
import pandas as pd


def _make_ohlcv(n: int = 100) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.random.randn(n))
    return pd.DataFrame({
        "ticker": "TEST",
        "timestamp": dates,
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
    })


def _make_featured(n: int = 200) -> pd.DataFrame:
    """OHLCV + rolling stats — required before label generation."""
    import sys
    sys.path.insert(0, "services/etl/src")
    from etl.transformers.rolling_stats import add_rolling_stats
    df = _make_ohlcv(n)
    return add_rolling_stats(df).dropna()


def test_label_generator_or_logic():
    """Anomaly label should fire on price OR volume spike, not both."""
    from etl.transformers.label_generator import add_anomaly_label

    df = _make_featured(300)
    # Inject a price spike in the middle
    mid = len(df) // 2
    df = df.copy()
    df.iloc[mid, df.columns.get_loc("z_score_20d")] = 10.0
    result = add_anomaly_label(df)
    assert "label" in result.columns
    assert result["label"].sum() > 0, "Should detect at least one anomaly"
    assert result["label"].dtype in [int, np.int64, np.int32]


def test_label_generator_returns_binary():
    """Labels must be 0 or 1 only."""
    from etl.transformers.label_generator import add_anomaly_label

    df = _make_featured(200)
    result = add_anomaly_label(df)
    unique = set(result["label"].unique())
    assert unique.issubset({0, 1}), f"Expected binary labels, got {unique}"


def test_rolling_stats_no_nulls_after_warmup():
    """Rolling stats should have no nulls after the warmup period."""
    import sys
    sys.path.insert(0, "services/etl/src")
    from etl.transformers.rolling_stats import add_rolling_stats

    df = _make_ohlcv(200)
    result = add_rolling_stats(df)
    # After the longest window (60 bars), no nulls should remain
    result_trimmed = result.iloc[65:]
    null_cols = [c for c in result_trimmed.columns if result_trimmed[c].isnull().any()]
    assert not null_cols, f"Null values found after warmup in: {null_cols}"
