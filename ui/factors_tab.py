"""Tab 2: Factor analysis — dual-axis volume vs factor charts."""

import streamlit as st
import pandas as pd

from ui.charts import dual_axis_chart, COLORS, DARK_LAYOUT
from ui.sidebar import CLASS_MAP_REVERSE, FACTORS
import plotly.graph_objects as go


def render_factors(weekly_vol: pd.DataFrame, weekly_factors: pd.DataFrame, params: dict):
    """Render the factors analysis tab."""
    if weekly_vol.empty or weekly_factors.empty:
        st.warning("Нет данных. Загрузите данные на вкладке «Данные».")
        return

    if not params["factors"]:
        st.info("Выберите хотя бы один фактор в боковой панели.")
        return

    # Prepare volume data
    vol_mask = (
        weekly_vol["instrument_class"].isin(params["classes_en"])
        & (weekly_vol["week_start"] >= params["date_from"])
        & (weekly_vol["week_start"] <= params["date_to"])
    )
    vol_filtered = weekly_vol[vol_mask].copy()

    # Prepare factor data
    fac_mask = (
        weekly_factors["factor_name"].isin(params["factors"])
        & (weekly_factors["week_start"] >= params["date_from"])
        & (weekly_factors["week_start"] <= params["date_to"])
    )
    fac_filtered = weekly_factors[fac_mask].copy()

    if vol_filtered.empty or fac_filtered.empty:
        st.info("Нет данных для выбранных параметров.")
        return

    # Select which instrument class to show
    available_classes = vol_filtered["instrument_class"].unique()
    class_options = [CLASS_MAP_REVERSE.get(c, c) for c in available_classes]
    class_map_local = dict(zip(class_options, available_classes))

    selected_class_ru = st.selectbox(
        "Класс инструментов для анализа",
        class_options,
        key="factor_class_select",
    )
    selected_class = class_map_local[selected_class_ru]

    # Volume series for selected class
    vol_series = (
        vol_filtered[vol_filtered["instrument_class"] == selected_class]
        .set_index("week_start")["total_value"]
        .sort_index()
    )
    vol_series.index = pd.to_datetime(vol_series.index)

    # --- Individual factor charts ---
    st.subheader(f"Объём ({selected_class_ru}) vs факторы")

    for factor_name in params["factors"]:
        factor_data = fac_filtered[fac_filtered["factor_name"] == factor_name]
        if factor_data.empty:
            continue

        fac_series = factor_data.set_index("week_start")["value"].sort_index()
        fac_series.index = pd.to_datetime(fac_series.index)

        fig = dual_axis_chart(vol_series, fac_series, selected_class_ru, factor_name)
        st.plotly_chart(fig, use_container_width=True)

    # --- Normalized overlay ---
    st.markdown("---")
    st.subheader("Нормализованные факторы (0–100)")

    fig = go.Figure()

    # Normalize volume to 0-100
    if not vol_series.empty:
        v_min, v_max = vol_series.min(), vol_series.max()
        if v_max > v_min:
            vol_norm = (vol_series - v_min) / (v_max - v_min) * 100
            fig.add_trace(go.Scatter(
                x=vol_norm.index, y=vol_norm.values,
                name=f"Объём ({selected_class_ru})",
                line=dict(color=COLORS[0], width=3),
            ))

    for i, factor_name in enumerate(params["factors"]):
        factor_data = fac_filtered[fac_filtered["factor_name"] == factor_name]
        if factor_data.empty:
            continue
        fac_series = factor_data.set_index("week_start")["value"].sort_index()
        fac_series.index = pd.to_datetime(fac_series.index)

        f_min, f_max = fac_series.min(), fac_series.max()
        if f_max > f_min:
            fac_norm = (fac_series - f_min) / (f_max - f_min) * 100
            fig.add_trace(go.Scatter(
                x=fac_norm.index, y=fac_norm.values,
                name=FACTORS.get(factor_name, factor_name),
                line=dict(color=COLORS[(i + 1) % len(COLORS)], width=2),
            ))

    fig.update_layout(**DARK_LAYOUT, yaxis_title="Нормализованное значение (0–100)")
    st.plotly_chart(fig, use_container_width=True)
