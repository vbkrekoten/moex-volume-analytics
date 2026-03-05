"""Tab 3: Correlation analysis — heatmap, rolling correlation, scatter plots."""

import streamlit as st
import pandas as pd

from analytics.correlation import compute_correlation_matrix, rolling_correlation
from ui.charts import correlation_heatmap, rolling_corr_chart, scatter_with_regression
from ui.sidebar import CLASS_MAP_REVERSE, FACTORS


def render_correlations(weekly_vol: pd.DataFrame, weekly_factors: pd.DataFrame, params: dict):
    """Render the correlation analysis tab."""
    if weekly_vol.empty or weekly_factors.empty:
        st.warning("Нет данных. Загрузите данные на вкладке «Данные».")
        return

    if not params["factors"]:
        st.info("Выберите хотя бы один фактор в боковой панели.")
        return

    # Filter data
    vol_mask = (
        weekly_vol["instrument_class"].isin(params["classes_en"])
        & (weekly_vol["week_start"] >= params["date_from"])
        & (weekly_vol["week_start"] <= params["date_to"])
    )
    vol_filtered = weekly_vol[vol_mask].copy()

    fac_mask = (
        weekly_factors["factor_name"].isin(params["factors"])
        & (weekly_factors["week_start"] >= params["date_from"])
        & (weekly_factors["week_start"] <= params["date_to"])
    )
    fac_filtered = weekly_factors[fac_mask].copy()

    # Create pivot tables
    vol_pivot = vol_filtered.pivot_table(
        index="week_start", columns="instrument_class",
        values="total_value", fill_value=0,
    )
    vol_pivot.index = pd.to_datetime(vol_pivot.index)

    fac_pivot = fac_filtered.pivot_table(
        index="week_start", columns="factor_name",
        values="value",
    )
    fac_pivot.index = pd.to_datetime(fac_pivot.index)

    # --- Controls ---
    col1, col2 = st.columns(2)
    method = col1.radio("Метод корреляции", ["Pearson", "Spearman"], horizontal=True)
    window = col2.select_slider(
        "Окно скользящей корреляции (недели)",
        options=[12, 26, 52],
        value=26,
    )

    # --- Correlation heatmap ---
    st.subheader("Матрица корреляций (изменения нед/нед)")
    corr_matrix = compute_correlation_matrix(
        vol_pivot, fac_pivot, method=method.lower()
    )
    fig = correlation_heatmap(corr_matrix)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Rolling correlation ---
    st.subheader("Скользящая корреляция")

    available_classes = vol_filtered["instrument_class"].unique()
    class_options = [CLASS_MAP_REVERSE.get(c, c) for c in available_classes]
    class_map_local = dict(zip(class_options, available_classes))

    rc_col1, rc_col2 = st.columns(2)
    rc_class_ru = rc_col1.selectbox(
        "Класс инструментов", class_options, key="rc_class"
    )
    rc_factor = rc_col2.selectbox(
        "Фактор",
        params["factors"],
        format_func=lambda x: FACTORS.get(x, x),
        key="rc_factor",
    )

    rc_class = class_map_local[rc_class_ru]

    if rc_class in vol_pivot.columns and rc_factor in fac_pivot.columns:
        rolling_df = rolling_correlation(
            vol_pivot[rc_class], fac_pivot[rc_factor],
            window=window, method=method.lower(),
        )
        label = f"{rc_class_ru} ↔ {FACTORS.get(rc_factor, rc_factor)}"
        fig = rolling_corr_chart(rolling_df, label)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Scatter plots ---
    st.subheader("Диаграммы рассеяния")

    for factor_name in params["factors"]:
        if factor_name not in fac_pivot.columns:
            continue
        if rc_class not in vol_pivot.columns:
            continue

        # Use percentage changes
        vol_chg = vol_pivot[rc_class].pct_change().dropna()
        fac_chg = fac_pivot[factor_name].pct_change().dropna()
        common = vol_chg.index.intersection(fac_chg.index)

        if len(common) < 10:
            continue

        fig = scatter_with_regression(
            fac_chg[common], vol_chg[common],
            x_label=f"Δ {FACTORS.get(factor_name, factor_name)} (%)",
            y_label=f"Δ Объём {rc_class_ru} (%)",
        )
        st.plotly_chart(fig, use_container_width=True)
