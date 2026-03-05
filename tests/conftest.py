"""Test fixtures."""

import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def sample_daily_turnovers():
    """Sample daily turnover data for testing aggregation."""
    dates = pd.date_range("2024-01-15", "2024-01-26", freq="B")
    rows = []
    for dt in dates:
        for cls in ["shares", "bonds", "futures"]:
            rows.append({
                "trade_date": dt.strftime("%Y-%m-%d"),
                "engine": "stock" if cls != "futures" else "futures",
                "market": cls if cls != "futures" else "forts",
                "instrument_class": cls,
                "value_rub": np.random.uniform(50000, 200000),
                "num_trades": np.random.randint(100000, 1000000),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_index_history():
    """Sample IMOEX daily OHLC for testing analytics."""
    dates = pd.date_range("2024-01-01", "2024-03-31", freq="B")
    n = len(dates)
    close = 3000 + np.cumsum(np.random.randn(n) * 20)
    high = close + np.abs(np.random.randn(n)) * 15
    low = close - np.abs(np.random.randn(n)) * 15

    df = pd.DataFrame({
        "trade_date": dates.strftime("%Y-%m-%d"),
        "ticker": "IMOEX",
        "open_val": close + np.random.randn(n) * 5,
        "high_val": high,
        "low_val": low,
        "close_val": close,
        "capitalization": 5e12 + np.cumsum(np.random.randn(n) * 1e10),
    })
    return df
