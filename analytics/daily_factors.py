"""Compute all daily factor values from raw data sources.

Produces a (trade_date, factor_name, value) long-format DataFrame
suitable for upsert into vol_daily_factors table.
"""

import logging

import numpy as np
import pandas as pd

from analytics.trend import compute_adx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: build a continuous trading-day calendar from turnovers
# ---------------------------------------------------------------------------

def _trading_days(turnovers_df: pd.DataFrame) -> pd.DatetimeIndex:
    """Extract sorted unique trading dates from turnovers."""
    dates = pd.to_datetime(turnovers_df["trade_date"])
    return pd.DatetimeIndex(sorted(dates.unique()))


def _to_long(series: pd.Series, factor_name: str) -> pd.DataFrame:
    """Convert a date-indexed Series into long-format rows."""
    df = series.dropna().reset_index()
    df.columns = ["trade_date", "value"]
    df["factor_name"] = factor_name
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
    return df[["trade_date", "factor_name", "value"]]


# ---------------------------------------------------------------------------
# Individual factor computations
# ---------------------------------------------------------------------------

def _compute_volatility_from_intraday(rv_series: pd.Series | None) -> pd.DataFrame:
    """Use pre-computed Realized Volatility from intraday candles.

    RV_t = sqrt( sum(r_{t,i}^2) ) * sqrt(252) * 100

    where r_{t,i} are intraday log returns from 10-min candles.
    If rv_series is None, returns empty DataFrame.
    """
    if rv_series is None or rv_series.empty:
        return pd.DataFrame()
    return _to_long(rv_series, "volatility")


def _compute_adx_daily(imoex: pd.DataFrame) -> pd.DataFrame:
    """Daily ADX (14-period) trend strength."""
    if len(imoex) < 30:
        return pd.DataFrame()
    df = imoex.sort_values("trade_date").reset_index(drop=True)
    df = df.dropna(subset=["high_val", "low_val", "close_val"])
    adx = compute_adx(df)
    result = pd.Series(adx.values, index=pd.to_datetime(df["trade_date"].values))
    return _to_long(result, "trend_strength")


def _compute_trend_direction(imoex: pd.DataFrame, span: int = 20) -> pd.DataFrame:
    """Trend direction: 20-day EMA slope of IMOEX close (pct change * 100)."""
    if imoex.empty:
        return pd.DataFrame()
    s = imoex.set_index("trade_date")["close_val"].sort_index()
    ema = s.ewm(span=span).mean()
    direction = ema.pct_change(fill_method=None) * 100
    return _to_long(direction, "trend_direction")


def _compute_index_level(imoex: pd.DataFrame) -> pd.DataFrame:
    """Daily IMOEX close level."""
    if imoex.empty:
        return pd.DataFrame()
    s = imoex.set_index("trade_date")["close_val"].sort_index()
    return _to_long(s, "index_level")


def _compute_market_cap(imoex: pd.DataFrame) -> pd.DataFrame:
    """IMOEX capitalization in trillions RUB."""
    if imoex.empty:
        return pd.DataFrame()
    cap = imoex.dropna(subset=["capitalization"]).copy()
    if cap.empty:
        return pd.DataFrame()
    s = cap.set_index("trade_date")["capitalization"].sort_index() / 1e12
    return _to_long(s, "market_cap")


def _compute_imoex_return(imoex: pd.DataFrame) -> pd.DataFrame:
    """Daily IMOEX return in percent."""
    if imoex.empty:
        return pd.DataFrame()
    s = imoex.set_index("trade_date")["close_val"].sort_index()
    ret = s.pct_change(fill_method=None) * 100
    return _to_long(ret, "imoex_return")


def _compute_rgbi_return(index_df: pd.DataFrame) -> pd.DataFrame:
    """Daily RGBI return in percent."""
    rgbi = index_df[index_df["ticker"] == "RGBI"].copy()
    if rgbi.empty:
        return pd.DataFrame()
    rgbi["trade_date"] = pd.to_datetime(rgbi["trade_date"])
    s = rgbi.set_index("trade_date")["close_val"].sort_index()
    ret = s.pct_change(fill_method=None) * 100
    return _to_long(ret, "rgbi_return")


def _compute_rvi(index_df: pd.DataFrame) -> pd.DataFrame:
    """Daily RVI (implied volatility index) close level.

    RVI is the Russian Volatility Index calculated by MOEX from IMOEX options.
    Analogous to VIX for the US market. Reflects market expectations of
    30-day forward volatility.
    """
    rvi = index_df[index_df["ticker"] == "RVI"].copy()
    if rvi.empty:
        return pd.DataFrame()
    rvi["trade_date"] = pd.to_datetime(rvi["trade_date"])
    s = rvi.set_index("trade_date")["close_val"].sort_index()
    s = pd.to_numeric(s, errors="coerce").dropna()
    return _to_long(s, "rvi")


def _compute_currency_factors(currency_df: pd.DataFrame,
                              cal: pd.DatetimeIndex) -> pd.DataFrame:
    """USD/RUB and CNY/RUB daily rates, forward-filled to trading days."""
    if currency_df.empty:
        return pd.DataFrame()
    frames = []
    df = currency_df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for pair in df["pair"].unique():
        sub = df[df["pair"] == pair].set_index("trade_date")["rate"].sort_index()
        sub = sub.reindex(cal).ffill()
        fname = pair.lower().replace("/", "_")
        frames.append(_to_long(sub, fname))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _compute_brent(brent_df: pd.DataFrame,
                   cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily Brent close, forward-filled to trading days."""
    if brent_df.empty:
        return pd.DataFrame()
    s = brent_df.set_index("trade_date")["close"].sort_index()
    s = s.reindex(cal).ffill()
    return _to_long(s, "brent")


def _compute_macro_daily(macro_df: pd.DataFrame,
                         cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward-fill macro indicators (key_rate, m2, cpi_yoy, cpi_index) to daily."""
    if macro_df.empty:
        return pd.DataFrame()
    frames = []
    df = macro_df.copy()
    df["period_date"] = pd.to_datetime(df["period_date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Indicator → factor name mapping
    ind_map = {
        "KEY_RATE": "key_rate",
        "M2": "m2",
        "CPI_YOY": "cpi_yoy",
        "CPI_INDEX": "cpi_index",
    }

    for ind, fname in ind_map.items():
        sub = df[df["indicator"] == ind].set_index("period_date")["value"].sort_index()
        sub = sub[~sub.index.duplicated(keep="last")]
        if sub.empty:
            continue
        # Reindex to daily calendar and forward-fill
        combined = sub.reindex(sub.index.union(cal)).sort_index().ffill()
        combined = combined.reindex(cal).dropna()
        frames.append(_to_long(combined, fname))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _compute_real_rate(macro_df: pd.DataFrame,
                       cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Real interest rate = key_rate - cpi_yoy, forward-filled daily."""
    if macro_df.empty:
        return pd.DataFrame()
    df = macro_df.copy()
    df["period_date"] = pd.to_datetime(df["period_date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    kr = df[df["indicator"] == "KEY_RATE"].set_index("period_date")["value"].sort_index()
    cpi = df[df["indicator"] == "CPI_YOY"].set_index("period_date")["value"].sort_index()

    if kr.empty or cpi.empty:
        return pd.DataFrame()

    kr = kr[~kr.index.duplicated(keep="last")]
    cpi = cpi[~cpi.index.duplicated(keep="last")]

    # Forward-fill both to daily, then subtract
    combined_idx = kr.index.union(cpi.index).union(cal)
    kr_daily = kr.reindex(combined_idx).sort_index().ffill().reindex(cal)
    cpi_daily = cpi.reindex(combined_idx).sort_index().ffill().reindex(cal)
    real = kr_daily - cpi_daily
    return _to_long(real, "real_rate")


def _compute_volume_momentum(turnovers_df: pd.DataFrame,
                             window: int = 5) -> pd.DataFrame:
    """5-day rolling mean of total daily turnover, lagged by 1 day."""
    if turnovers_df.empty:
        return pd.DataFrame()
    df = turnovers_df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["value_rub"] = pd.to_numeric(df["value_rub"], errors="coerce")
    daily_total = df.groupby("trade_date")["value_rub"].sum().sort_index()
    # Rolling mean, then lag by 1 day (shift forward)
    momentum = daily_total.rolling(window).mean().shift(1)
    # Normalize to billions for readability
    momentum = momentum / 1e9
    return _to_long(momentum, "volume_momentum")


def _compute_num_trades(turnovers_df: pd.DataFrame) -> pd.DataFrame:
    """Total number of trades per day (across all classes)."""
    if turnovers_df.empty:
        return pd.DataFrame()
    df = turnovers_df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["num_trades"] = pd.to_numeric(df["num_trades"], errors="coerce")
    daily_total = df.groupby("trade_date")["num_trades"].sum().sort_index()
    # Normalize to thousands
    daily_total = daily_total / 1e3
    return _to_long(daily_total, "num_trades")


def _compute_hh_savings(savings_df: pd.DataFrame | None,
                         macro_df: pd.DataFrame,
                         cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Household savings in trillions RUB, forward-filled to trading days.

    Uses HH_SAVINGS_TOTAL from savings_df (all components incl. escrow).
    Falls back to HH_SAVINGS_TOTAL or HH_DEPOSITS in macro_df.
    """
    df = None
    if savings_df is not None and not savings_df.empty:
        # Use total from the structured savings data
        total = savings_df[savings_df["indicator"] == "HH_SAVINGS_TOTAL"]
        if not total.empty:
            df = total.copy()

    if (df is None or df.empty) and not macro_df.empty:
        # Fallback: try HH_SAVINGS_TOTAL or HH_DEPOSITS in vol_macro
        for ind in ("HH_SAVINGS_TOTAL", "HH_DEPOSITS"):
            hh = macro_df[macro_df["indicator"] == ind].copy()
            if not hh.empty:
                df = hh
                break

    if df is None or df.empty:
        return pd.DataFrame()

    df["period_date"] = pd.to_datetime(df["period_date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    # Convert from billions to trillions
    s = df.set_index("period_date")["value"].sort_index() / 1000
    s = s[~s.index.duplicated(keep="last")]
    # Reindex to daily calendar and forward-fill
    combined = s.reindex(s.index.union(cal)).sort_index().ffill()
    combined = combined.reindex(cal).dropna()
    return _to_long(combined, "hh_savings")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_all_daily_factors(
    index_df: pd.DataFrame,
    currency_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    brent_df: pd.DataFrame,
    turnovers_df: pd.DataFrame,
    savings_df: pd.DataFrame | None = None,
    rv_series: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Compute all daily factors from raw data sources.

    Returns DataFrame with (trade_date, factor_name, value) rows
    ready for upsert into vol_daily_factors.
    """
    frames: list[pd.DataFrame] = []

    # Trading day calendar
    cal = _trading_days(turnovers_df)
    if cal.empty:
        logger.warning("No trading days found in turnovers data")
        return pd.DataFrame()

    # Prepare IMOEX subset
    imoex = pd.DataFrame()
    if not index_df.empty:
        imoex = index_df[index_df["ticker"] == "IMOEX"].copy()
        imoex["trade_date"] = pd.to_datetime(imoex["trade_date"])
        imoex = imoex.sort_values("trade_date").drop_duplicates("trade_date", keep="last")

    # Compute each factor with error isolation
    factor_computations = [
        ("volatility", lambda: _compute_volatility_from_intraday(rv_series)),
        ("trend_strength", lambda: _compute_adx_daily(imoex)),
        ("trend_direction", lambda: _compute_trend_direction(imoex)),
        ("index_level", lambda: _compute_index_level(imoex)),
        ("market_cap", lambda: _compute_market_cap(imoex)),
        ("imoex_return", lambda: _compute_imoex_return(imoex)),
        ("rgbi_return", lambda: _compute_rgbi_return(index_df)),
        ("rvi", lambda: _compute_rvi(index_df)),
        ("usd_rub/cny_rub", lambda: _compute_currency_factors(currency_df, cal)),
        ("brent", lambda: _compute_brent(brent_df, cal)),
        ("macro", lambda: _compute_macro_daily(macro_df, cal)),
        ("real_rate", lambda: _compute_real_rate(macro_df, cal)),
        ("volume_momentum", lambda: _compute_volume_momentum(turnovers_df)),
        ("num_trades", lambda: _compute_num_trades(turnovers_df)),
        ("hh_savings", lambda: _compute_hh_savings(savings_df, macro_df, cal)),
    ]

    for name, fn in factor_computations:
        try:
            result = fn()
            if result is not None and not result.empty:
                frames.append(result)
                logger.info("Factor '%s': %d rows", name, len(result))
        except Exception as e:
            logger.warning("Failed to compute factor '%s': %s", name, e)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
