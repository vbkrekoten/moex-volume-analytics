"""Section 2: Unified factor analysis — validation cards, heatmap, explorer."""

import streamlit as st
import pandas as pd
import numpy as np

from ui.sidebar import FACTORS, FACTOR_DESCRIPTIONS, TERM_DESCRIPTIONS, CLASS_MAP_REVERSE
from ui.charts import (
    correlation_heatmap,
    scatter_with_regression,
    rolling_corr_chart,
    coefficient_bar_chart,
    residuals_chart,
    rolling_r2_chart,
)
from analytics.correlation import compute_correlation_matrix, rolling_correlation
from analytics.regression import factor_regression, rolling_regression_r2
from analytics.validation import compute_factor_stability
from analytics.ai_commentary import (
    generate_factor_summary_commentary,
    generate_correlation_commentary,
    generate_regression_commentary,
)


def _render_ai_commentary(text: str):
    """Render AI-generated commentary in a styled glass card."""
    if not text:
        return
    st.markdown(
        f'<div class="glass-card" style="border-left: 3px solid #00d4ff;">'
        f'<div style="font-size: 0.7rem; color: #6b7280; margin-bottom: 0.4rem;">'
        f'🤖 AI-комментарий</div>'
        f'<div style="font-size: 0.88rem; color: #d1d5db; line-height: 1.55;">'
        f'{text}</div></div>',
        unsafe_allow_html=True,
    )


def _term_tooltip(term_key: str, label: str | None = None):
    """Render an inline tooltip (popover) for an analytical term."""
    desc = TERM_DESCRIPTIONS.get(term_key, {})
    if not desc:
        return
    display = label or desc.get("name", term_key)
    with st.popover(f"ℹ {display}", use_container_width=True):
        st.markdown(f"**{desc.get('name', display)}**")
        st.markdown(desc.get("description", ""))
        if desc.get("formula"):
            st.markdown(
                f'<div class="formula-block">{desc["formula"]}</div>',
                unsafe_allow_html=True,
            )


def _render_term_tooltips_row(term_keys: list[str]):
    """Render a row of analytical term tooltips."""
    n_cols = min(5, len(term_keys))
    if n_cols == 0:
        return
    cols = st.columns(n_cols)
    for i, key in enumerate(term_keys):
        desc = TERM_DESCRIPTIONS.get(key, {})
        if not desc:
            continue
        with cols[i % n_cols]:
            _term_tooltip(key)


def _prepare_volumes_wide(daily_vol: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Pivot daily volumes to wide format (date index, class columns)."""
    filtered = daily_vol[
        daily_vol["instrument_class"].isin(params["classes_en"])
        & (daily_vol["trade_date"] >= params["date_from"])
        & (daily_vol["trade_date"] <= params["date_to"])
    ].copy()
    if filtered.empty:
        return pd.DataFrame()
    filtered["trade_date"] = pd.to_datetime(filtered["trade_date"])
    pivot = filtered.pivot_table(
        index="trade_date",
        columns="instrument_class",
        values="value_rub",
        aggfunc="sum",
        fill_value=0,
    ).sort_index()
    return pivot


def _prepare_factors_wide(daily_factors: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Pivot daily factors to wide format (date index, factor columns)."""
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
    pivot = filtered.pivot_table(
        index="trade_date",
        columns="factor_name",
        values="value",
        aggfunc="last",
    ).sort_index()
    return pivot


def render_analysis_section(
    daily_vol: pd.DataFrame,
    daily_factors: pd.DataFrame,
    params: dict,
):
    """Render the unified factor analysis section."""
    is_weekly = params.get("frequency") == "weekly"
    freq_label = "недельным" if is_weekly else "дневным"
    st.markdown(
        '<div class="section-header">'
        '<h2>Факторный анализ</h2>'
        f'<p>Оценка влияния макро- и рыночных факторов на торговые обороты (по {freq_label} данным)</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if daily_vol.empty or daily_factors.empty:
        st.info("Загрузите данные в разделе «Данные» для анализа факторов.")
        return

    vol_wide = _prepare_volumes_wide(daily_vol, params)
    fac_wide = _prepare_factors_wide(daily_factors, params)

    if vol_wide.empty or fac_wide.empty:
        st.info("Недостаточно данных для выбранных параметров.")
        return

    # --- 1. Factor summary panel ---
    _render_factor_summary(vol_wide, fac_wide, params)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # --- 2. Correlation heatmap ---
    _render_correlation_block(vol_wide, fac_wide, params)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # --- 3. Regression + Factor explorer ---
    _render_regression_block(vol_wide, fac_wide, params)


def _render_factor_summary(
    vol_wide: pd.DataFrame,
    fac_wide: pd.DataFrame,
    params: dict,
):
    """Compact factor validation cards with traffic lights."""
    # Select instrument class for validation
    class_options = list(vol_wide.columns)
    class_labels = [CLASS_MAP_REVERSE.get(c, c) for c in class_options]

    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        sel_idx = st.selectbox(
            "Класс инструмента",
            range(len(class_options)),
            format_func=lambda i: class_labels[i],
            key="validation_class",
        )
    selected_class = class_options[sel_idx]

    with col_info:
        st.caption(
            "Карточки показывают силу связи каждого фактора с оборотами выбранного класса. "
            "Цвет: зелёный (p<0.01), жёлтый (p<0.05), "
            "оранжевый (p<0.10), серый (незначим)."
        )
        # Term tooltips for factor cards
        _render_term_tooltips_row(["r_squared", "p_value", "oos_r2", "stability"])

    # Factor tooltips row
    st.markdown("##### Описание факторов")
    _render_factor_tooltips(params["factors"])

    vol_series = vol_wide[selected_class]

    # Compute stability metrics
    stab_window = 24 if params.get("frequency") == "weekly" else 120
    with st.spinner("Вычисление метрик..."):
        stability_df = compute_factor_stability(vol_series, fac_wide, window=stab_window)

    if stability_df.empty:
        st.info("Недостаточно данных для расчёта метрик стабильности.")
        return

    # Render factor cards in rows of 5
    factors_list = stability_df.to_dict("records")
    n_cols = min(5, len(factors_list))
    rows_needed = (len(factors_list) + n_cols - 1) // n_cols

    for row_i in range(rows_needed):
        cols = st.columns(n_cols)
        for col_i in range(n_cols):
            idx = row_i * n_cols + col_i
            if idx >= len(factors_list):
                break
            finfo = factors_list[idx]
            with cols[col_i]:
                _render_single_factor_card(finfo)

    # AI commentary for factor summary
    class_label = CLASS_MAP_REVERSE.get(selected_class, selected_class)
    commentary = generate_factor_summary_commentary(
        factors_list, class_label, FACTORS,
    )
    _render_ai_commentary(commentary)


def _render_single_factor_card(finfo: dict):
    """Render a single factor validation card."""
    fname = finfo["factor"]
    label = FACTORS.get(fname, fname)
    r2 = finfo["r2"]
    pval = finfo["pvalue"]
    status = finfo["status"]
    oos = finfo["oos_r2"]

    # Color based on status
    color_map = {"green": "#51cf66", "yellow": "#ffa94d", "red": "#ff6b6b"}
    color = color_map.get(status, "#6b7280")
    status_emoji = {"green": "●", "yellow": "●", "red": "●"}.get(status, "●")

    # Significance stars
    if pval < 0.001:
        stars = "***"
    elif pval < 0.01:
        stars = "**"
    elif pval < 0.05:
        stars = "*"
    else:
        stars = ""

    st.markdown(f"""
    <div class="factor-card">
        <div class="factor-name">{label}</div>
        <div class="factor-value" style="color: {color};">{r2:.4f}</div>
        <div class="factor-detail">
            R² | p={pval:.3f} {stars}
        </div>
        <div class="factor-detail">
            OOS R²: {oos:.4f} |
            <span style="color: {color};">{status_emoji}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_factor_tooltips(selected_factors: list[str]):
    """Render factor tooltips using st.popover."""
    # Show in rows of 5
    n_cols = min(5, len(selected_factors))
    if n_cols == 0:
        return

    cols = st.columns(n_cols)
    for i, fname in enumerate(selected_factors):
        desc = FACTOR_DESCRIPTIONS.get(fname, {})
        label = FACTORS.get(fname, fname)
        with cols[i % n_cols]:
            with st.popover(f"ℹ {label}", use_container_width=True):
                st.markdown(f"**{desc.get('name', label)}**")
                st.markdown(desc.get("description", "Описание отсутствует."))
                if desc.get("formula"):
                    st.markdown(f'<div class="formula-block">{desc["formula"]}</div>',
                                unsafe_allow_html=True)
                if desc.get("unit"):
                    st.caption(f"Единица: {desc['unit']}")


def _render_correlation_block(
    vol_wide: pd.DataFrame,
    fac_wide: pd.DataFrame,
    params: dict,
):
    """Correlation heatmap + rolling correlation explorer."""
    st.markdown("##### Корреляция оборотов с факторами")

    # Business-friendly explanation
    st.markdown(
        '<div class="glass-card" style="border-left: 3px solid #00d4ff; padding: 0.8rem 1rem;">'
        '<div style="font-size: 0.92rem; color: #d1d5db; line-height: 1.6;">'
        '<b>Как читать корреляцию:</b><br>'
        '• <span style="color:#51cf66;"><b>+1</b></span> — показатели растут вместе<br>'
        '• <span style="color:#ff6b6b;"><b>−1</b></span> — движутся в противоположных направлениях<br>'
        '• <span style="color:#6b7280;"><b>0</b></span> — связи нет'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # Controls: method + period presets
    c1, c2 = st.columns([1, 1])
    with c1:
        method = st.radio(
            "Метод", ["Spearman", "Pearson"],
            horizontal=True, key="corr_method",
            help="Spearman — устойчив к выбросам, Pearson — классический линейный",
        )
    with c2:
        period_label = st.radio(
            "Период скользящей корреляции",
            ["3 месяца", "6 месяцев", "1 год"],
            horizontal=True,
            key="corr_period",
        )
        if params.get("frequency") == "weekly":
            window = {"3 месяца": 13, "6 месяцев": 26, "1 год": 52}[period_label]
        else:
            window = {"3 месяца": 63, "6 месяцев": 126, "1 год": 252}[period_label]

    # Heatmap
    corr_matrix = compute_correlation_matrix(
        vol_wide, fac_wide,
        method=method.lower(),
        use_changes=True,
    )
    if not corr_matrix.empty:
        fig = correlation_heatmap(corr_matrix)
        st.plotly_chart(fig, use_container_width=True)

        # AI commentary for correlation
        corr_data = corr_matrix.to_dict()
        commentary = generate_correlation_commentary(
            corr_data, method, CLASS_MAP_REVERSE, FACTORS,
        )
        _render_ai_commentary(commentary)
    else:
        st.info("Недостаточно данных для матрицы корреляций.")

    # Rolling correlation explorer
    st.markdown("##### Скользящая корреляция")
    st.markdown(
        '<div class="glass-card" style="border-left: 3px solid #ffa94d; padding: 0.8rem 1rem;">'
        '<div style="font-size: 0.85rem; color: #d1d5db; line-height: 1.55;">'
        'Показывает <b>стабильность связи</b> между оборотом и фактором во времени. '
        'Если линия «прыгает» — связь ненадёжная и на неё нельзя опираться в прогнозах.'
        '</div></div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        vol_class = st.selectbox(
            "Класс",
            vol_wide.columns.tolist(),
            format_func=lambda x: CLASS_MAP_REVERSE.get(x, x),
            key="rolling_class",
        )
    with c2:
        fac_name = st.selectbox(
            "Фактор",
            fac_wide.columns.tolist(),
            format_func=lambda x: FACTORS.get(x, x),
            key="rolling_factor",
        )

    if vol_class and fac_name:
        rolling_df = rolling_correlation(
            vol_wide[vol_class],
            fac_wide[fac_name],
            window=window,
            method=method.lower(),
        )
        label = f"{CLASS_MAP_REVERSE.get(vol_class, vol_class)} ~ {FACTORS.get(fac_name, fac_name)}"

        c_chart, c_scatter = st.columns(2)
        with c_chart:
            fig = rolling_corr_chart(rolling_df, label)
            st.plotly_chart(fig, use_container_width=True)
        with c_scatter:
            # Scatter plot
            v_changes = vol_wide[vol_class].pct_change(fill_method=None).dropna()
            f_changes = fac_wide[fac_name].pct_change(fill_method=None).dropna()
            common = v_changes.index.intersection(f_changes.index)
            if len(common) > 20:
                fig = scatter_with_regression(
                    f_changes.loc[common],
                    v_changes.loc[common],
                    FACTORS.get(fac_name, fac_name) + " (Δ%)",
                    CLASS_MAP_REVERSE.get(vol_class, vol_class) + " (Δ%)",
                )
                st.plotly_chart(fig, use_container_width=True)


def _render_regression_block(
    vol_wide: pd.DataFrame,
    fac_wide: pd.DataFrame,
    params: dict,
):
    """Multi-factor regression analysis."""
    st.markdown("##### Многофакторная регрессия")

    # Term tooltips for regression section
    _render_term_tooltips_row(["regression", "r_squared", "adj_r_squared", "std_coefficient"])
    _render_term_tooltips_row(["p_value", "residuals", "rolling_r2"])

    class_options = vol_wide.columns.tolist()
    sel_class = st.selectbox(
        "Класс инструмента для регрессии",
        class_options,
        format_func=lambda x: CLASS_MAP_REVERSE.get(x, x),
        key="reg_class",
    )

    if not sel_class or fac_wide.empty:
        return

    vol_series = vol_wide[sel_class]
    result = factor_regression(vol_series, fac_wide, use_changes=True, min_obs=30)

    if result["n_obs"] == 0:
        st.info("Недостаточно наблюдений для регрессии.")
        return

    # KPI row
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{result['r2']:.4f}")
    c2.metric("R² adj", f"{result['adj_r2']:.4f}")
    c3.metric("Наблюдений", f"{result['n_obs']:,}")

    # Coefficient bar chart + residuals side by side
    c_coef, c_resid = st.columns([3, 2])
    with c_coef:
        st.markdown("###### Значимость факторов")
        st.caption(
            "Зелёный (p<0.01), Жёлтый (p<0.05), Оранжевый (p<0.10), Серый (незначим)"
        )
        fig = coefficient_bar_chart(result["coefficients"], result["pvalues"])
        st.plotly_chart(fig, use_container_width=True)
    with c_resid:
        st.markdown("###### Остатки модели")
        fig = residuals_chart(result["residuals"])
        st.plotly_chart(fig, use_container_width=True)

    # AI commentary for regression
    reg_class_label = CLASS_MAP_REVERSE.get(sel_class, sel_class)
    commentary = generate_regression_commentary(
        result["r2"], result["adj_r2"], result["n_obs"],
        result["coefficients"], result["pvalues"],
        reg_class_label, FACTORS,
    )
    _render_ai_commentary(commentary)

    # Rolling R² stability
    with st.expander("Стабильность модели (Rolling R²)", expanded=False):
        reg_window = 24 if params.get("frequency") == "weekly" else 120
        rolling_df = rolling_regression_r2(vol_series, fac_wide, window=reg_window)
        if not rolling_df.empty:
            label = CLASS_MAP_REVERSE.get(sel_class, sel_class)
            fig = rolling_r2_chart(rolling_df, f"Rolling R² ({label})")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Недостаточно данных для расчёта скользящего R².")

    # Coefficient table
    with st.expander("Таблица коэффициентов", expanded=False):
        if result["coefficients"]:
            rows = []
            for k in result["coefficients"]:
                pv = result["pvalues"].get(k, 1.0)
                if pv < 0.001:
                    sig = "***"
                elif pv < 0.01:
                    sig = "**"
                elif pv < 0.05:
                    sig = "*"
                else:
                    sig = ""
                rows.append({
                    "Фактор": FACTORS.get(k, k),
                    "Коэффициент": result["coefficients"][k],
                    "p-value": pv,
                    "Значимость": sig,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
