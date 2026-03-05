"""Tab 1: Volume overview — KPI cards and stacked area chart."""

import streamlit as st
import pandas as pd

from ui.charts import stacked_area_chart
from ui.sidebar import CLASS_MAP_REVERSE


def render_overview(weekly_vol: pd.DataFrame, params: dict):
    """Render the volume overview tab."""
    if weekly_vol.empty:
        st.warning("Нет данных по объёмам. Загрузите данные на вкладке «Данные».")
        return

    # Filter by selected classes and date range
    mask = (
        weekly_vol["instrument_class"].isin(params["classes_en"])
        & (weekly_vol["week_start"] >= params["date_from"])
        & (weekly_vol["week_start"] <= params["date_to"])
    )
    filtered = weekly_vol[mask].copy()

    if filtered.empty:
        st.info("Нет данных для выбранных параметров.")
        return

    # --- KPI cards ---
    latest_week = filtered["week_start"].max()
    prev_week = filtered[filtered["week_start"] < latest_week]["week_start"].max()

    latest_data = filtered[filtered["week_start"] == latest_week]
    total_latest = latest_data["total_value"].sum()

    delta = None
    if pd.notna(prev_week):
        prev_data = filtered[filtered["week_start"] == prev_week]
        total_prev = prev_data["total_value"].sum()
        if total_prev > 0:
            delta = f"{(total_latest / total_prev - 1) * 100:+.1f}%"

    # Most active class this week
    if not latest_data.empty:
        most_active = latest_data.loc[latest_data["total_value"].idxmax(), "instrument_class"]
        most_active_ru = CLASS_MAP_REVERSE.get(most_active, most_active)
    else:
        most_active_ru = "—"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Объём за неделю", f"{total_latest / 1e6:,.0f} трлн ₽", delta)
    col2.metric("Средний дневной", f"{total_latest / 5 / 1e6:,.1f} трлн ₽")
    col3.metric("Самый активный", most_active_ru)
    col4.metric(
        "Классов",
        len(filtered["instrument_class"].unique()),
    )

    # --- Chart ---
    st.markdown("---")
    pct_mode = st.toggle("Показать в % от общего", value=False)
    fig = stacked_area_chart(filtered, pct_mode=pct_mode)
    st.plotly_chart(fig, use_container_width=True)

    # --- Data table ---
    with st.expander("Таблица данных"):
        pivot = filtered.pivot_table(
            index="week_start", columns="instrument_class",
            values="total_value", fill_value=0,
        ).sort_index(ascending=False)
        pivot.columns = [CLASS_MAP_REVERSE.get(c, c) for c in pivot.columns]
        st.dataframe(pivot.style.format("{:,.0f}"), use_container_width=True)
