"""Compute all weekly factor values from raw data."""

import pandas as pd

from analytics.volatility import weekly_rvi, realized_volatility
from analytics.trend import weekly_trend_strength, weekly_trend_direction


def compute_index_factors(index_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute index-derived factors from daily index data:
    - volatility (RVI or realized vol)
    - trend_strength (ADX)
    - trend_direction (EMA slope)
    - index_level (IMOEX close)
    - market_cap (IMOEX capitalization)
    """
    frames = []

    # Volatility: prefer RVI, fallback to realized vol
    rvi = weekly_rvi(index_df)
    if rvi.empty:
        rvi = realized_volatility(index_df)
    frames.append(rvi)

    # Trend strength (ADX)
    adx = weekly_trend_strength(index_df)
    frames.append(adx)

    # Trend direction
    direction = weekly_trend_direction(index_df)
    frames.append(direction)

    # Index level: IMOEX weekly close
    imoex = index_df[index_df["ticker"] == "IMOEX"].copy()
    if not imoex.empty:
        imoex["trade_date"] = pd.to_datetime(imoex["trade_date"])
        imoex["week_start"] = imoex["trade_date"] - pd.to_timedelta(
            imoex["trade_date"].dt.dayofweek, unit="D"
        )
        idx_weekly = (
            imoex.sort_values("trade_date")
            .groupby("week_start")["close_val"]
            .last()
            .reset_index()
        )
        idx_weekly.columns = ["week_start", "value"]
        idx_weekly["factor_name"] = "index_level"
        idx_weekly["week_start"] = idx_weekly["week_start"].dt.strftime("%Y-%m-%d")
        frames.append(idx_weekly[["week_start", "factor_name", "value"]])

    # Market capitalization
    if not imoex.empty:
        cap_df = imoex.dropna(subset=["capitalization"])
        if not cap_df.empty:
            cap_weekly = (
                cap_df.sort_values("trade_date")
                .groupby("week_start")["capitalization"]
                .last()
                .reset_index()
            )
            cap_weekly.columns = ["week_start", "value"]
            # Convert to trillions
            cap_weekly["value"] = cap_weekly["value"] / 1e12
            cap_weekly["factor_name"] = "market_cap"
            cap_weekly["week_start"] = cap_weekly["week_start"].dt.strftime("%Y-%m-%d")
            frames.append(cap_weekly[["week_start", "factor_name", "value"]])

    result = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    return result


def compute_currency_factors(currency_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weekly USD/RUB and CNY/RUB from daily CBR rates.
    """
    if currency_df.empty:
        return pd.DataFrame()

    df = currency_df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["week_start"] = df["trade_date"] - pd.to_timedelta(
        df["trade_date"].dt.dayofweek, unit="D"
    )

    frames = []
    for pair in df["pair"].unique():
        subset = df[df["pair"] == pair].sort_values("trade_date")
        weekly = subset.groupby("week_start")["rate"].last().reset_index()
        weekly.columns = ["week_start", "value"]
        weekly["factor_name"] = pair.lower().replace("/", "_")
        weekly["week_start"] = weekly["week_start"].dt.strftime("%Y-%m-%d")
        frames.append(weekly[["week_start", "factor_name", "value"]])

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
