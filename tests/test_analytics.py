"""Tests for analytics modules."""

import numpy as np
import pandas as pd
from analytics.trend import compute_adx, weekly_trend_strength, weekly_trend_direction
from analytics.volatility import realized_volatility


def test_compute_adx(sample_index_history):
    """ADX should return values in 0-100 range."""
    adx = compute_adx(sample_index_history)
    valid = adx.dropna()

    assert len(valid) > 0
    assert (valid >= 0).all()
    assert (valid <= 100).all()


def test_weekly_trend_strength(sample_index_history):
    """Weekly ADX should produce factor rows."""
    result = weekly_trend_strength(sample_index_history)

    assert not result.empty
    assert "week_start" in result.columns
    assert "factor_name" in result.columns
    assert "value" in result.columns
    assert (result["factor_name"] == "trend_strength").all()


def test_weekly_trend_direction(sample_index_history):
    """Trend direction should be positive or negative percentage."""
    result = weekly_trend_direction(sample_index_history)

    assert not result.empty
    assert (result["factor_name"] == "trend_direction").all()
    # Direction values should be reasonable (not huge)
    assert (result["value"].abs() < 50).all()


def test_realized_volatility(sample_index_history):
    """Realized volatility should be positive and reasonable."""
    result = realized_volatility(sample_index_history)

    if not result.empty:
        assert (result["factor_name"] == "volatility").all()
        assert (result["value"] > 0).all()
        # Annualized vol should be < 200% for reasonable data
        assert (result["value"] < 200).all()


def test_empty_inputs():
    """Empty dataframes should not crash analytics."""
    empty = pd.DataFrame()

    assert weekly_trend_strength(empty).empty
    assert weekly_trend_direction(empty).empty
    assert realized_volatility(empty).empty
