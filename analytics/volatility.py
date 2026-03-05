"""Volatility computation: RVI (implied) and realized volatility from IMOEX."""

import numpy as np
import pandas as pd


def weekly_rvi(index_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract weekly RVI (Russian Volatility Index) close values.
    Input: index_df with columns (trade_date, ticker, close_val).
    Output: DataFrame with (week_start, factor_name='volatility', value).
    """
    if index_df.empty or "ticker" not in index_df.columns:
        return pd.DataFrame()
    rvi = index_df[index_df["ticker"] == "RVI"].copy()
    if rvi.empty:
        return pd.DataFrame()

    rvi["trade_date"] = pd.to_datetime(rvi["trade_date"])
    rvi["week_start"] = rvi["trade_date"] - pd.to_timedelta(
        rvi["trade_date"].dt.dayofweek, unit="D"
    )

    # Last close per week
    weekly = (
        rvi.sort_values("trade_date")
        .groupby("week_start")["close_val"]
        .last()
        .reset_index()
    )
    weekly.columns = ["week_start", "value"]
    weekly["factor_name"] = "volatility"
    weekly["week_start"] = weekly["week_start"].dt.strftime("%Y-%m-%d")

    return weekly[["week_start", "factor_name", "value"]]


def realized_volatility(index_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute annualized realized volatility from IMOEX daily closes.
    Used as fallback when RVI is not available.
    """
    if index_df.empty or "ticker" not in index_df.columns:
        return pd.DataFrame()
    imoex = index_df[index_df["ticker"] == "IMOEX"].copy()
    if imoex.empty:
        return pd.DataFrame()

    imoex["trade_date"] = pd.to_datetime(imoex["trade_date"])
    imoex = imoex.sort_values("trade_date")

    # Log returns
    imoex["log_ret"] = np.log(imoex["close_val"] / imoex["close_val"].shift(1))

    # Rolling realized volatility (annualized)
    imoex["rv"] = imoex["log_ret"].rolling(window).std() * np.sqrt(252) * 100

    imoex["week_start"] = imoex["trade_date"] - pd.to_timedelta(
        imoex["trade_date"].dt.dayofweek, unit="D"
    )

    # Last value per week
    weekly = (
        imoex.dropna(subset=["rv"])
        .sort_values("trade_date")
        .groupby("week_start")["rv"]
        .last()
        .reset_index()
    )
    weekly.columns = ["week_start", "value"]
    weekly["factor_name"] = "volatility"
    weekly["week_start"] = weekly["week_start"].dt.strftime("%Y-%m-%d")

    return weekly[["week_start", "factor_name", "value"]]
