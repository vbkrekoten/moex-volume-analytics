"""Trend strength (ADX) and trend direction from IMOEX OHLC data."""

import numpy as np
import pandas as pd


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Average Directional Index from OHLC data.
    ADX > 25 = strong trend, ADX < 20 = weak/no trend.
    """
    high = df["high_val"].astype(float)
    low = df["low_val"].astype(float)
    close = df["close_val"].astype(float)

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # +DM and -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    # Wilder's smoothing (EMA with alpha = 1/period)
    alpha = 1 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)

    # DX and ADX
    di_sum = plus_di + minus_di
    di_sum = di_sum.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return adx


def weekly_trend_strength(index_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weekly ADX (trend strength) from daily IMOEX OHLC.
    Returns last ADX value per week.
    """
    if index_df.empty or "ticker" not in index_df.columns:
        return pd.DataFrame()
    imoex = index_df[index_df["ticker"] == "IMOEX"].copy()
    if imoex.empty:
        return pd.DataFrame()

    imoex["trade_date"] = pd.to_datetime(imoex["trade_date"])
    imoex = imoex.sort_values("trade_date").reset_index(drop=True)

    # Need OHLC; skip rows with missing data
    imoex = imoex.dropna(subset=["high_val", "low_val", "close_val"])
    if len(imoex) < 30:
        return pd.DataFrame()

    imoex["adx"] = compute_adx(imoex)

    imoex["week_start"] = imoex["trade_date"] - pd.to_timedelta(
        imoex["trade_date"].dt.dayofweek, unit="D"
    )

    weekly = (
        imoex.dropna(subset=["adx"])
        .sort_values("trade_date")
        .groupby("week_start")["adx"]
        .last()
        .reset_index()
    )
    weekly.columns = ["week_start", "value"]
    weekly["factor_name"] = "trend_strength"
    weekly["week_start"] = weekly["week_start"].dt.strftime("%Y-%m-%d")

    return weekly[["week_start", "factor_name", "value"]]


def weekly_trend_direction(index_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weekly trend direction as 4-week EMA slope of IMOEX close.
    Positive = uptrend, negative = downtrend.
    """
    if index_df.empty or "ticker" not in index_df.columns:
        return pd.DataFrame()
    imoex = index_df[index_df["ticker"] == "IMOEX"].copy()
    if imoex.empty:
        return pd.DataFrame()

    imoex["trade_date"] = pd.to_datetime(imoex["trade_date"])
    imoex = imoex.sort_values("trade_date")

    imoex["week_start"] = imoex["trade_date"] - pd.to_timedelta(
        imoex["trade_date"].dt.dayofweek, unit="D"
    )

    # Weekly close
    weekly_close = (
        imoex.groupby("week_start")["close_val"]
        .last()
        .reset_index()
    )
    weekly_close = weekly_close.sort_values("week_start")

    # 4-week EMA and its percentage change
    weekly_close["ema4"] = weekly_close["close_val"].ewm(span=4).mean()
    weekly_close["direction"] = weekly_close["ema4"].pct_change() * 100

    result = weekly_close.dropna(subset=["direction"]).copy()
    result["factor_name"] = "trend_direction"
    result["week_start"] = result["week_start"].dt.strftime("%Y-%m-%d")

    return result[["week_start", "factor_name"]].assign(
        value=result["direction"].values
    )
