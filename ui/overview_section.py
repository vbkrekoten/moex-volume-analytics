"""Section 1: Turnover overview with KPI cards and combined chart."""

import streamlit as st
import pandas as pd

from ui.charts import combined_turnover_factor_chart, stacked_area_chart
from ui.sidebar import CLASS_MAP_REVERSE, FACTORS


def render_overview_section(
    daily_vol: pd.DataFrame,
    daily_factors: pd.DataFrame,
    params: dict,
):
    """Render the overview section with turnovers + factor overlay."""
    st.markdown(
        '<div class="section-header">'
        '<h2>Обзор торговых оборотов</h2>'
        '<p>Дневные обороты по классам инструментов MOEX с 2018 года</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if daily_vol.empty:
        st.warning("Нет данных по оборотам. Загрузите данные в разделе «Данные».")
        return

    # Filter by classes and dates
    filtered = daily_vol[
        daily_vol["instrument_class"].isin(params["classes_en"])
        & (daily_vol["trade_date"] >= params["date_from"])
        & (daily_vol["trade_date"] <= params["date_to"])
    ].copy()

    if filtered.empty:
        st.info("Нет данных для выбранных параметров.")
        return

    # --- KPI cards ---
    latest_date = filtered["trade_date"].max()
    latest_data = filtered[filtered["trade_date"] == latest_date]
    total_latest = latest_data["value_rub"].sum()

    # Previous trading day
    prev_dates = filtered[filtered["trade_date"] < latest_date]["trade_date"].unique()
    delta = None
    if len(prev_dates) > 0:
        prev_date = sorted(prev_dates)[-1]
        prev_data = filtered[filtered["trade_date"] == prev_date]
        total_prev = prev_data["value_rub"].sum()
        if total_prev > 0:
            delta = f"{(total_latest / total_prev - 1) * 100:+.1f}%"

    # Most active class
    if not latest_data.empty:
        top_class = latest_data.loc[latest_data["value_rub"].idxmax(), "instrument_class"]
        top_class_ru = CLASS_MAP_REVERSE.get(top_class, top_class)
    else:
        top_class_ru = "—"

    # Total period turnover
    total_period = filtered["value_rub"].sum()

    col1, col2, col3, col4 = st.columns(4)
    # value_rub is in millions of RUB (MOEX ISS API default)
    date_label = pd.to_datetime(latest_date).strftime("%d.%m.%Y")
    col1.metric(
        f"Оборот за {date_label}",
        f"{total_latest / 1e3:,.0f} млрд ₽",
        delta,
    )
    d_from_lbl = pd.to_datetime(params["date_from"]).strftime("%m.%Y")
    d_to_lbl = pd.to_datetime(params["date_to"]).strftime("%m.%Y")
    col2.metric(
        f"Оборот {d_from_lbl} – {d_to_lbl}",
        f"{total_period / 1e6:,.1f} трлн ₽",
    )
    col3.metric("Самый активный", top_class_ru)
    col4.metric(
        "Классов",
        len(filtered["instrument_class"].unique()),
    )

    # --- Combined chart: turnovers + factors ---
    st.markdown("")

    # Factor overlay selector (inline)
    c1, c2 = st.columns([3, 1])
    with c2:
        pct_mode = st.toggle("Доли %", value=False, key="overview_pct")
    with c1:
        overlay_factors = st.multiselect(
            "Наложить факторы",
            options=list(FACTORS.keys()),
            default=params.get("factors", [])[:3],
            format_func=lambda x: FACTORS[x],
            key="overview_factors",
        )

    # Filter factors by date range
    if not daily_factors.empty:
        filt_factors = daily_factors[
            (daily_factors["trade_date"] >= params["date_from"])
            & (daily_factors["trade_date"] <= params["date_to"])
        ]
    else:
        filt_factors = pd.DataFrame()

    if pct_mode:
        fig = stacked_area_chart(filtered, pct_mode=True)
    elif overlay_factors:
        fig = combined_turnover_factor_chart(
            filtered, filt_factors, overlay_factors,
        )
    else:
        fig = stacked_area_chart(filtered, pct_mode=False)

    st.plotly_chart(fig, use_container_width=True)
