"""MOEX Volume Analytics Dashboard — main Streamlit entry point."""

import os
import sys

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Load .env for local development
load_dotenv()

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from data_pipeline.db import get_client
from ui.styles import load_custom_css
from ui.sidebar import render_sidebar
from ui.overview_section import render_overview_section
from ui.analysis_section import render_analysis_section
from ui.monthly_data_section import render_monthly_data_section
from ui.data_tab import render_data_section
from ui.forecast_section import render_forecast_section

st.set_page_config(
    page_title="MOEX Volume Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
load_custom_css()

# --- Title ---
st.markdown(
    '<h1 style="margin-bottom:0;">Аналитика торговых оборотов MOEX</h1>',
    unsafe_allow_html=True,
)
st.caption(
    "Зависимости торговых оборотов по классам инструментов "
    "от волатильности, тренда, макроэкономических факторов и других показателей"
)


# --- Sidebar ---
params = render_sidebar()


# --- Load data with caching ---
def _fetch_all(table: str, date_filter: dict | None = None) -> list[dict]:
    """Fetch all rows from a Supabase table, paginating past 1000-row default."""
    client = get_client()
    all_data: list[dict] = []
    offset = 0
    page_size = 1000
    while True:
        query = client.table(table).select("*")
        if date_filter:
            date_col = date_filter.get("col", "trade_date")
            if date_filter.get("gte"):
                query = query.gte(date_col, date_filter["gte"])
            if date_filter.get("lte"):
                query = query.lte(date_col, date_filter["lte"])
        resp = query.range(offset, offset + page_size - 1).execute()
        if not resp.data:
            break
        all_data.extend(resp.data)
        if len(resp.data) < page_size:
            break
        offset += page_size
    return all_data


@st.cache_data(ttl=3600)
def load_daily_turnovers() -> pd.DataFrame:
    """Load daily turnovers, aggregated by instrument_class."""
    data = _fetch_all("vol_daily_turnovers")
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["value_rub"] = pd.to_numeric(df["value_rub"], errors="coerce")
    df["num_trades"] = pd.to_numeric(df["num_trades"], errors="coerce")
    # Aggregate sub-markets to instrument_class level
    agg = (
        df.groupby(["trade_date", "instrument_class"])
        .agg(value_rub=("value_rub", "sum"), num_trades=("num_trades", "sum"))
        .reset_index()
    )
    return agg


@st.cache_data(ttl=3600)
def load_daily_factors() -> pd.DataFrame:
    """Load pre-computed daily factors."""
    data = _fetch_all("vol_daily_factors")
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


# Load data (always daily — weekly is aggregated on-the-fly)
daily_vol = load_daily_turnovers()
daily_factors = load_daily_factors()


# --- Aggregate to weekly if needed ---
def _to_weekly_vol(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily turnovers to weekly (sum per ISO week)."""
    if df.empty:
        return df
    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    out["week_start"] = (
        out["trade_date"] - pd.to_timedelta(out["trade_date"].dt.dayofweek, unit="D")
    )
    result = (
        out.groupby(["week_start", "instrument_class"])
        .agg(value_rub=("value_rub", "sum"), num_trades=("num_trades", "sum"))
        .reset_index()
    )
    result["trade_date"] = result["week_start"].dt.strftime("%Y-%m-%d")
    return result.drop(columns=["week_start"])


def _to_weekly_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily factors to weekly (last value per ISO week)."""
    if df.empty:
        return df
    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    out["week_start"] = (
        out["trade_date"] - pd.to_timedelta(out["trade_date"].dt.dayofweek, unit="D")
    )
    out = out.sort_values("trade_date")
    result = out.groupby(["week_start", "factor_name"]).agg(value=("value", "last")).reset_index()
    result["trade_date"] = result["week_start"].dt.strftime("%Y-%m-%d")
    return result[["trade_date", "factor_name", "value"]]


# Apply frequency selection
if params.get("frequency") == "weekly":
    active_vol = _to_weekly_vol(daily_vol)
    active_factors = _to_weekly_factors(daily_factors)
else:
    active_vol = daily_vol
    active_factors = daily_factors


# --- Scrollable sections ---

# Section 1: Overview
render_overview_section(active_vol, active_factors, params)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# Section 2: Factor Analysis
render_analysis_section(active_vol, active_factors, params)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# Section 3: Monthly Data
render_monthly_data_section(active_vol, active_factors, params)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# Section 4: Forecast (always uses daily data for HAR-RV model)
render_forecast_section(daily_vol, daily_factors, params, fetch_func=_fetch_all)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# Section 5: Data Health
render_data_section()
