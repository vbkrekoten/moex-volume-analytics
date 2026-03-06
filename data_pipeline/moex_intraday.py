"""Fetch intraday candle data from MOEX ISS API for Realized Volatility.

Uses 10-minute candles for the IMOEX index to compute daily RV.
"""

import logging
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

CANDLES_URL = (
    "https://iss.moex.com/iss/engines/stock/markets/index"
    "/securities/IMOEX/candles.json"
)

PAGE_SIZE = 500


def _fetch_candles_page(
    date_from: str,
    date_to: str,
    interval: int = 10,
    start: int = 0,
) -> list[list]:
    """Fetch a single page of candle data."""
    params = {
        "from": date_from,
        "till": date_to,
        "interval": interval,
        "start": start,
        "iss.meta": "off",
        "iss.only": "candles",
    }
    resp = requests.get(CANDLES_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("candles", {}).get("data", [])


def fetch_intraday_candles(
    date_from: date,
    date_to: date,
    interval: int = 10,
    delay: float = 0.05,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Fetch 10-minute intraday candles for IMOEX from MOEX ISS API.

    Returns DataFrame with columns:
        open, close, high, low, value, volume, begin, end

    Fetches month-by-month to keep pagination manageable.
    """
    all_rows: list[list] = []

    # Generate month boundaries
    months = []
    current = date_from.replace(day=1)
    while current <= date_to:
        month_end = (current + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        month_end = min(month_end, date_to)
        months.append((current, month_end))
        current = (current + timedelta(days=32)).replace(day=1)

    total_months = len(months)

    for m_idx, (m_start, m_end) in enumerate(months):
        start_str = m_start.strftime("%Y-%m-%d")
        end_str = m_end.strftime("%Y-%m-%d")
        offset = 0

        while True:
            try:
                rows = _fetch_candles_page(start_str, end_str, interval, offset)
            except requests.RequestException as e:
                logger.warning(
                    "Failed to fetch candles for %s-%s (offset %d): %s",
                    start_str, end_str, offset, e,
                )
                break

            if not rows:
                break

            all_rows.extend(rows)
            if len(rows) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
            time.sleep(delay)

        if progress_callback:
            progress_callback((m_idx + 1) / total_months)

        time.sleep(delay)

    if not all_rows:
        logger.warning("No intraday candles fetched")
        return pd.DataFrame()

    # Columns: open, close, high, low, value, volume, begin, end
    df = pd.DataFrame(
        all_rows,
        columns=["open", "close", "high", "low", "value", "volume", "begin", "end"],
    )
    df["begin"] = pd.to_datetime(df["begin"])
    df["end"] = pd.to_datetime(df["end"])
    df["trade_date"] = df["begin"].dt.date

    for col in ["open", "close", "high", "low", "value"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(
        "Fetched %d intraday candles (%s to %s)",
        len(df),
        df["trade_date"].min(),
        df["trade_date"].max(),
    )

    return df


def compute_realized_volatility(
    candles_df: pd.DataFrame,
    annualize: bool = True,
) -> pd.Series:
    """
    Compute daily Realized Volatility from intraday candle data.

    RV_t = sqrt( sum(r_{t,i}^2) )

    where r_{t,i} = ln(close_i / close_{i-1}) are intraday log returns.

    If annualize=True, multiply by sqrt(252) * 100 to get annualized %.

    Returns a Series indexed by trade_date with daily RV values.
    """
    if candles_df.empty:
        return pd.Series(dtype=float)

    df = candles_df.sort_values("begin").copy()

    # Compute log returns within each day
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Mark first candle of each day (its return is cross-day, exclude it)
    df["is_first"] = df["trade_date"] != df["trade_date"].shift(1)
    df.loc[df["is_first"], "log_return"] = np.nan

    # Squared returns
    df["r_squared"] = df["log_return"] ** 2

    # Sum of squared returns per day
    daily_rv_sq = df.groupby("trade_date")["r_squared"].sum()

    # RV = sqrt(sum of squared returns)
    daily_rv = np.sqrt(daily_rv_sq)

    if annualize:
        daily_rv = daily_rv * np.sqrt(252) * 100

    # Convert index to datetime
    daily_rv.index = pd.to_datetime(daily_rv.index)
    daily_rv.name = "realized_volatility"

    logger.info(
        "Computed RV for %d trading days (mean=%.2f%%, median=%.2f%%)",
        len(daily_rv),
        daily_rv.mean(),
        daily_rv.median(),
    )

    return daily_rv
