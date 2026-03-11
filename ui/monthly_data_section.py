"""Section: Monthly numerical data view for turnovers and factors."""

import io

import streamlit as st
import pandas as pd

from ui.sidebar import CLASS_MAP_REVERSE, FACTORS


def render_monthly_data_section(
    daily_vol: pd.DataFrame,
    daily_factors: pd.DataFrame,
    params: dict,
):
    """Render monthly aggregated tables for turnovers and factors."""
    st.markdown(
        '<div class="section-header">'
        '<h2>Помесячные данные</h2>'
        '<p>Числовые значения оборотов и факторов в месячном разрезе</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if daily_vol.empty and daily_factors.empty:
        st.info("Нет данных для отображения.")
        return

    # --- Turnovers table ---
    turnovers_monthly = _build_turnovers_monthly(daily_vol, params)
    factors_monthly = _build_factors_monthly(daily_factors, params)

    tab_vol, tab_fac = st.tabs(["Обороты", "Факторы"])

    with tab_vol:
        if turnovers_monthly.empty:
            st.info("Нет данных по оборотам для выбранных параметров.")
        else:
            st.markdown("###### Обороты по классам инструментов (сумма за месяц, млн руб.)")
            st.dataframe(
                turnovers_monthly.style.format("{:,.0f}", na_rep="—"),
                use_container_width=True,
                height=min(600, 40 + len(turnovers_monthly) * 35),
            )

    with tab_fac:
        if factors_monthly.empty:
            st.info("Нет данных по факторам для выбранных параметров.")
        else:
            st.markdown("###### Среднее значение фактора за месяц")
            st.dataframe(
                factors_monthly.style.format("{:,.2f}", na_rep="—"),
                use_container_width=True,
                height=min(600, 40 + len(factors_monthly) * 35),
            )

    # CSV download combining both tables
    if not turnovers_monthly.empty or not factors_monthly.empty:
        csv_data = _build_csv(turnovers_monthly, factors_monthly)
        st.download_button(
            label="📥 Скачать CSV",
            data=csv_data,
            file_name="moex_monthly_data.csv",
            mime="text/csv",
        )


def _build_turnovers_monthly(daily_vol: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Aggregate daily turnovers to monthly sums, pivoted wide."""
    if daily_vol.empty:
        return pd.DataFrame()

    filtered = daily_vol[
        daily_vol["instrument_class"].isin(params["classes_en"])
        & (daily_vol["trade_date"] >= params["date_from"])
        & (daily_vol["trade_date"] <= params["date_to"])
    ].copy()
    if filtered.empty:
        return pd.DataFrame()

    filtered["trade_date"] = pd.to_datetime(filtered["trade_date"])
    filtered["month"] = filtered["trade_date"].dt.to_period("M")

    monthly = (
        filtered.groupby(["month", "instrument_class"])["value_rub"]
        .sum()
        .reset_index()
    )
    pivot = monthly.pivot_table(
        index="month",
        columns="instrument_class",
        values="value_rub",
        aggfunc="sum",
    ).sort_index()

    # Rename columns to Russian
    pivot.columns = [CLASS_MAP_REVERSE.get(c, c) for c in pivot.columns]
    # Add total column
    pivot["Итого"] = pivot.sum(axis=1)
    # Convert period index to string for display
    pivot.index = pivot.index.astype(str)
    pivot.index.name = "Месяц"
    return pivot


def _build_factors_monthly(daily_factors: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Aggregate daily factors to monthly means, pivoted wide."""
    if daily_factors.empty:
        return pd.DataFrame()

    filtered = daily_factors[
        daily_factors["factor_name"].isin(params["factors"])
        & (daily_factors["trade_date"] >= params["date_from"])
        & (daily_factors["trade_date"] <= params["date_to"])
    ].copy()
    if filtered.empty:
        return pd.DataFrame()

    filtered["trade_date"] = pd.to_datetime(filtered["trade_date"])
    filtered["month"] = filtered["trade_date"].dt.to_period("M")

    monthly = (
        filtered.groupby(["month", "factor_name"])["value"]
        .mean()
        .reset_index()
    )
    pivot = monthly.pivot_table(
        index="month",
        columns="factor_name",
        values="value",
        aggfunc="mean",
    ).sort_index()

    # Rename columns to Russian
    pivot.columns = [FACTORS.get(c, c) for c in pivot.columns]
    pivot.index = pivot.index.astype(str)
    pivot.index.name = "Месяц"
    return pivot


def _build_csv(turnovers: pd.DataFrame, factors: pd.DataFrame) -> str:
    """Combine turnovers and factors into a single CSV string."""
    buf = io.StringIO()
    if not turnovers.empty:
        buf.write("# Обороты (сумма за месяц, млн руб.)\n")
        turnovers.to_csv(buf)
        buf.write("\n")
    if not factors.empty:
        buf.write("# Факторы (среднее за месяц)\n")
        factors.to_csv(buf)
    return buf.getvalue()
