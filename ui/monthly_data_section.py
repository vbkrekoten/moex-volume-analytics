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
    """Render monthly aggregated table combining turnovers and factors."""
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

    turnovers_monthly = _build_turnovers_monthly(daily_vol, params)
    factors_monthly = _build_factors_monthly(daily_factors, params)

    if turnovers_monthly.empty and factors_monthly.empty:
        st.info("Нет данных для выбранных параметров.")
        return

    # Merge into one table: turnovers first, then factors
    combined = _merge_tables(turnovers_monthly, factors_monthly)

    if combined.empty:
        st.info("Нет данных для выбранных параметров.")
        return

    # Reverse sort so latest month is on top
    combined = combined.iloc[::-1]

    # Build per-column format: turnovers as integers, factors as 2-decimal
    turnover_cols = set(turnovers_monthly.columns) if not turnovers_monthly.empty else set()
    fmt = {}
    for col in combined.columns:
        if col in turnover_cols:
            fmt[col] = "{:,.0f}"
        else:
            fmt[col] = "{:,.2f}"

    st.markdown(
        "###### Обороты (сумма, млн руб.) и факторы (среднее за месяц)"
    )
    st.dataframe(
        combined.style.format(fmt, na_rep="—"),
        use_container_width=True,
        height=600,
    )

    # CSV download
    csv_data = _build_csv(combined)
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
    pivot["Итого оборот"] = pivot.sum(axis=1)
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


def _merge_tables(turnovers: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Merge turnovers and factors into one table by month index."""
    if turnovers.empty and factors.empty:
        return pd.DataFrame()
    if turnovers.empty:
        return factors
    if factors.empty:
        return turnovers
    return pd.concat([turnovers, factors], axis=1)


def _build_csv(combined: pd.DataFrame) -> str:
    """Export combined table to CSV string."""
    buf = io.StringIO()
    combined.to_csv(buf)
    return buf.getvalue()
