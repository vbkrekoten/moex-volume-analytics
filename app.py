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
from ui.sidebar import render_sidebar
from ui.overview_tab import render_overview
from ui.factors_tab import render_factors
from ui.correlation_tab import render_correlations
from ui.regression_tab import render_regression
from ui.data_tab import render_data_tab

st.set_page_config(
    page_title="MOEX Volume Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Аналитика торговых объёмов Московской биржи")
st.caption(
    "Зависимости объёмов торгов по классам инструментов "
    "от волатильности, тренда, макроэкономических факторов и других показателей"
)


# --- Sidebar ---
params = render_sidebar()


# --- Load data with caching ---
def _fetch_all(table: str) -> list[dict]:
    """Fetch all rows from a Supabase table, paginating past the 1000-row default."""
    client = get_client()
    all_data: list[dict] = []
    offset = 0
    page_size = 1000
    while True:
        resp = client.table(table).select("*").range(offset, offset + page_size - 1).execute()
        if not resp.data:
            break
        all_data.extend(resp.data)
        if len(resp.data) < page_size:
            break
        offset += page_size
    return all_data


@st.cache_data(ttl=3600)
def load_weekly_volumes():
    """Load pre-aggregated weekly volumes from Supabase."""
    data = _fetch_all("vol_weekly_volumes")
    if data:
        df = pd.DataFrame(data)
        for col in ["total_value", "avg_daily"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_weekly_factors():
    """Load pre-computed weekly factors from Supabase."""
    data = _fetch_all("vol_weekly_factors")
    if data:
        df = pd.DataFrame(data)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df
    return pd.DataFrame()


weekly_vol = load_weekly_volumes()
weekly_factors = load_weekly_factors()


# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Обзор объёмов",
    "Факторы",
    "Корреляции",
    "Регрессия",
    "Данные",
])

with tab1:
    render_overview(weekly_vol, params)

with tab2:
    render_factors(weekly_vol, weekly_factors, params)

with tab3:
    render_correlations(weekly_vol, weekly_factors, params)

with tab4:
    render_regression(weekly_vol, weekly_factors, params)

with tab5:
    render_data_tab()
