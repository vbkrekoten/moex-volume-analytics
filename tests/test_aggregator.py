"""Tests for weekly aggregation logic."""

import pandas as pd
from data_pipeline.aggregator import to_weekly_volumes, forward_fill_monthly_to_weekly


def test_to_weekly_volumes(sample_daily_turnovers):
    """Weekly aggregation should sum daily values by instrument class."""
    weekly = to_weekly_volumes(sample_daily_turnovers)

    assert not weekly.empty
    assert "week_start" in weekly.columns
    assert "instrument_class" in weekly.columns
    assert "total_value" in weekly.columns
    assert "avg_daily" in weekly.columns
    assert "trading_days" in weekly.columns

    # Each week should have <=5 trading days
    assert (weekly["trading_days"] <= 5).all()
    # Total should be >= avg_daily (since trading_days >= 1)
    assert (weekly["total_value"] >= weekly["avg_daily"]).all()


def test_to_weekly_volumes_empty():
    """Empty input should return empty output."""
    result = to_weekly_volumes(pd.DataFrame())
    assert result.empty


def test_forward_fill_monthly():
    """Monthly data should be forward-filled to weekly dates."""
    monthly = pd.DataFrame({
        "period_date": ["2024-01-01", "2024-02-01", "2024-03-01"],
        "indicator": ["CPI_YOY", "CPI_YOY", "CPI_YOY"],
        "value": [7.5, 7.7, 7.6],
    })
    weekly_dates = [
        "2024-01-08", "2024-01-15", "2024-01-22", "2024-01-29",
        "2024-02-05", "2024-02-12", "2024-02-19", "2024-02-26",
    ]

    result = forward_fill_monthly_to_weekly(monthly, weekly_dates)
    assert not result.empty
    assert len(result) == len(weekly_dates)

    # January weeks should have Jan CPI value
    jan_rows = result[result["week_start"] < "2024-02-01"]
    assert (jan_rows["value"] == 7.5).all()

    # February weeks should have Feb CPI value
    feb_rows = result[result["week_start"] >= "2024-02-05"]
    assert (feb_rows["value"] == 7.7).all()
