"""Tab 4: OLS regression analysis — factor decomposition."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from analytics.regression import factor_regression
from ui.charts import coefficient_bar_chart, COLORS, DARK_LAYOUT
from ui.sidebar import CLASS_MAP_REVERSE, FACTORS


def render_regression(weekly_vol: pd.DataFrame, weekly_factors: pd.DataFrame, params: dict):
    """Render the regression analysis tab."""
    if weekly_vol.empty or weekly_factors.empty:
        st.warning("Нет данных. Загрузите данные на вкладке «Данные».")
        return

    if len(params["factors"]) < 1:
        st.info("Выберите хотя бы один фактор в боковой панели.")
        return

    # Filter
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

    # Pivots
    vol_pivot = vol_filtered.pivot_table(
        index="week_start", columns="instrument_class",
        values="total_value", fill_value=0,
    )
    vol_pivot.index = pd.to_datetime(vol_pivot.index)

    fac_pivot = fac_filtered.pivot_table(
        index="week_start", columns="factor_name", values="value",
    )
    fac_pivot.index = pd.to_datetime(fac_pivot.index)

    # Select class
    available_classes = vol_filtered["instrument_class"].unique()
    class_options = [CLASS_MAP_REVERSE.get(c, c) for c in available_classes]
    class_map_local = dict(zip(class_options, available_classes))

    selected_class_ru = st.selectbox(
        "Класс инструментов для регрессии",
        class_options,
        key="reg_class",
    )
    selected_class = class_map_local[selected_class_ru]

    if selected_class not in vol_pivot.columns:
        st.warning("Нет данных для выбранного класса.")
        return

    # Run regression
    vol_series = vol_pivot[selected_class]
    result = factor_regression(vol_series, fac_pivot)

    if result["n_obs"] == 0:
        st.warning("Недостаточно данных для регрессии (минимум 20 наблюдений).")
        return

    # --- KPI metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("R²", f"{result['r2']:.4f}")
    col2.metric("R² adj.", f"{result['adj_r2']:.4f}")
    col3.metric("Наблюдений", result["n_obs"])

    st.markdown("---")

    # --- Coefficient bar chart ---
    st.subheader("Значимость факторов")
    st.caption("Стандартизированные коэффициенты OLS. Жёлтые — p < 0.05 (значимые).")

    fig = coefficient_bar_chart(result["coefficients"], result["pvalues"])
    st.plotly_chart(fig, use_container_width=True)

    # --- Coefficient table ---
    st.subheader("Таблица коэффициентов")
    coef_df = pd.DataFrame({
        "Фактор": [FACTORS.get(k, k) for k in result["coefficients"].keys()],
        "Коэффициент": list(result["coefficients"].values()),
        "p-значение": [result["pvalues"].get(k, None) for k in result["coefficients"].keys()],
        "Значимость": [
            "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            for p in (result["pvalues"].get(k, 1) for k in result["coefficients"].keys())
        ],
    })
    st.dataframe(coef_df, use_container_width=True, hide_index=True)

    # --- Residual plot ---
    st.markdown("---")
    st.subheader("Остатки модели")

    residuals = result["residuals"]
    if not residuals.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=residuals.index, y=residuals.values,
            mode="lines",
            line=dict(color=COLORS[0], width=1.5),
            name="Остатки",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="#555")
        fig.update_layout(**DARK_LAYOUT, yaxis_title="Остатки (Δ объём)")
        st.plotly_chart(fig, use_container_width=True)
