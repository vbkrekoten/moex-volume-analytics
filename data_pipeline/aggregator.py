"""Aggregate daily data into weekly time series."""

import pandas as pd
import numpy as np


def to_weekly_volumes(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily turnovers to ISO weeks (Monday start).
    Input columns: trade_date, instrument_class, value_rub, num_trades.
    Output columns: week_start, instrument_class, total_value, avg_daily, trading_days.
    """
    if daily_df.empty:
        return pd.DataFrame()

    df = daily_df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    # ISO week start (Monday)
    df["week_start"] = df["trade_date"] - pd.to_timedelta(
        df["trade_date"].dt.dayofweek, unit="D"
    )

    weekly = (
        df.groupby(["week_start", "instrument_class"])
        .agg(
            total_value=("value_rub", "sum"),
            avg_daily=("value_rub", "mean"),
            trading_days=("value_rub", "count"),
        )
        .reset_index()
    )

    weekly["week_start"] = weekly["week_start"].dt.strftime("%Y-%m-%d")
    return weekly


def to_weekly_last(daily_df: pd.DataFrame, date_col: str,
                   value_col: str, group_col: str | None = None) -> pd.DataFrame:
    """
    Sample daily data to weekly by taking the last value of each week.
    Used for indices, exchange rates, etc.
    """
    if daily_df.empty:
        return pd.DataFrame()

    df = daily_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["week_start"] = df[date_col] - pd.to_timedelta(
        df[date_col].dt.dayofweek, unit="D"
    )

    group_keys = ["week_start"]
    if group_col:
        group_keys.append(group_col)

    # Take the last observation per week (latest trading day)
    df = df.sort_values(date_col)
    weekly = df.groupby(group_keys).last().reset_index()

    weekly["week_start"] = weekly["week_start"].dt.strftime("%Y-%m-%d")
    return weekly


def forward_fill_monthly_to_weekly(monthly_df: pd.DataFrame,
                                   weekly_dates: list[str]) -> pd.DataFrame:
    """
    Forward-fill monthly indicators to weekly frequency.
    Input: monthly_df with columns (period_date, indicator, value).
    Output: DataFrame with (week_start, factor_name, value) for each indicator.
    """
    if monthly_df.empty or not weekly_dates:
        return pd.DataFrame()

    weekly_dt = pd.to_datetime(weekly_dates)
    results = []

    for indicator in monthly_df["indicator"].unique():
        subset = monthly_df[monthly_df["indicator"] == indicator].copy()
        subset["period_date"] = pd.to_datetime(subset["period_date"])
        subset = subset.sort_values("period_date").set_index("period_date")

        # Create daily index and forward-fill
        daily_idx = pd.date_range(
            start=min(subset.index.min(), weekly_dt.min()),
            end=weekly_dt.max(),
            freq="D",
        )
        daily = subset["value"].reindex(daily_idx, method="ffill")

        # Sample at weekly dates
        for wk in weekly_dt:
            if wk in daily.index and pd.notna(daily[wk]):
                results.append({
                    "week_start": wk.strftime("%Y-%m-%d"),
                    "factor_name": indicator.lower(),
                    "value": float(daily[wk]),
                })

    return pd.DataFrame(results) if results else pd.DataFrame()
