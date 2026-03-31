"""Section: Household savings — full structure from CBR quarterly balance sheet."""

from __future__ import annotations

import io

import streamlit as st
import pandas as pd

from data_pipeline.cbr_household_assets import INDICATOR_LABELS
from ui.savings_charts import (
    MAIN_COMPONENTS,
    savings_stacked_area,
    savings_donut,
    deposits_stacked_area,
    equities_stacked_area,
    cash_stacked_area,
)


# All indicators we want to load from vol_macro
_ALL_INDICATORS = list(INDICATOR_LABELS.keys())


def render_savings_section(params: dict, fetch_func) -> None:
    """Render the household savings tab."""
    st.markdown(
        '<div class="section-header">'
        '<h2>Сбережения домохозяйств</h2>'
        '<p>Структура финансовых активов населения по данным ЦБ РФ (помесячные данные)</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    raw = _load_data(params)
    if raw.empty:
        st.info("Нет данных по финансовым активам домохозяйств. Запустите пайплайн.")
        return

    # Pivot: index=period_date, columns=indicator
    pivot = raw.pivot_table(
        index="period_date", columns="indicator", values="value", aggfunc="last",
    ).sort_index()

    # --- KPI cards ---
    _render_kpi(pivot)

    # --- Main structure: stacked area + donut ---
    st.markdown("##### Структура финансовых активов")
    col_area, col_donut = st.columns([3, 2])

    with col_area:
        fig = savings_stacked_area(pivot)
        st.plotly_chart(fig, use_container_width=True)

    with col_donut:
        latest = _get_latest(pivot, list(MAIN_COMPONENTS.keys()))
        if latest:
            fig = savings_donut(latest)
            st.plotly_chart(fig, use_container_width=True)

    # --- Sub-breakdowns ---
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    sub1, sub2, sub3 = st.columns(3)
    with sub1:
        st.markdown("###### Депозиты")
        fig = deposits_stacked_area(pivot)
        st.plotly_chart(fig, use_container_width=True)
    with sub2:
        st.markdown("###### Акции и участие в капитале")
        fig = equities_stacked_area(pivot)
        st.plotly_chart(fig, use_container_width=True)
    with sub3:
        st.markdown("###### Наличные")
        fig = cash_stacked_area(pivot)
        st.plotly_chart(fig, use_container_width=True)

    # --- Monthly table ---
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    _render_table(pivot)

    # --- Methodology ---
    _render_methodology()


def _load_data(params: dict) -> pd.DataFrame:
    """Load all household asset indicators from vol_macro."""
    try:
        from data_pipeline.db import get_client
        client = get_client()

        # Supabase doesn't support large IN lists well, so fetch all HH_* indicators
        all_rows = []
        page_size = 1000
        offset = 0
        while True:
            resp = (
                client.table("vol_macro")
                .select("period_date,indicator,value")
                .like("indicator", "HH_%")
                .gte("period_date", params["date_from"])
                .lte("period_date", params["date_to"])
                .order("period_date")
                .range(offset, offset + page_size - 1)
                .execute()
            )
            if resp.data:
                all_rows.extend(resp.data)
            if not resp.data or len(resp.data) < page_size:
                break
            offset += page_size

        if all_rows:
            return pd.DataFrame(all_rows)
    except Exception:
        pass
    return pd.DataFrame()


def _get_latest(pivot: pd.DataFrame, indicators: list[str]) -> dict[str, float]:
    """Get latest non-null value for each indicator."""
    result = {}
    for ind in indicators:
        if ind in pivot.columns:
            s = pivot[ind].dropna()
            if not s.empty:
                result[ind] = s.iloc[-1]
    return result


def _render_kpi(pivot: pd.DataFrame) -> None:
    """Render hierarchical KPI: total on top, components below."""

    # Find latest date where key indicators have real data (not NaN/0)
    key_col = "HH_DEPOSITS_TOTAL"  # always present if data exists
    if key_col not in pivot.columns:
        key_col = pivot.columns[0]
    valid = pivot[key_col].dropna()
    valid = valid[valid > 0]
    if len(valid) < 2:
        st.info("Недостаточно данных для отображения.")
        return
    latest_date = valid.index[-1]
    prev_date = valid.index[-2]

    # Format dates for display
    latest_str = pd.Timestamp(latest_date).strftime("%d.%m.%Y")
    prev_str = pd.Timestamp(prev_date).strftime("%d.%m.%Y")

    def _get_vals(indicator):
        if indicator not in pivot.columns:
            return None, None, None
        latest_val = pivot.loc[latest_date, indicator] if pd.notna(pivot.loc[latest_date].get(indicator)) else None
        prev_val = pivot.loc[prev_date, indicator] if pd.notna(pivot.loc[prev_date].get(indicator)) else None
        if latest_val is not None and prev_val is not None:
            return float(latest_val), float(latest_val) - float(prev_val), float(prev_val)
        elif latest_val is not None:
            return float(latest_val), None, None
        return None, None, None

    def _card_html(label, value_bln, delta_bln=None, prev_bln=None, color="#f0b429", big=False):
        val_trln = value_bln / 1000
        size = "1.8rem" if big else "1.2rem"
        label_size = "0.85rem" if big else "0.7rem"
        delta_html = ""
        if delta_bln is not None and prev_bln:
            d_trln = delta_bln / 1000
            pct = delta_bln / prev_bln * 100
            arrow = "▲" if d_trln >= 0 else "▼"
            d_color = "#51cf66" if d_trln >= 0 else "#ff6b6b"
            delta_html = (
                f'<div style="font-size:0.65rem;color:{d_color};margin-top:2px;">'
                f'{arrow} {d_trln:+,.1f} трлн ({pct:+.1f}%)</div>'
            )
        return (
            f'<div style="background:rgba(17,24,39,0.5);border:1px solid rgba(255,255,255,0.06);'
            f'border-left:3px solid {color};border-radius:8px;padding:0.6rem 0.8rem;'
            f'margin-bottom:0.4rem;">'
            f'<div style="font-size:{label_size};color:#9ca3af;margin-bottom:2px;">{label}</div>'
            f'<div style="font-size:{size};font-weight:700;color:#e5e7eb;">'
            f'{val_trln:,.1f} <span style="font-size:0.7rem;color:#6b7280;">трлн ₽</span></div>'
            f'{delta_html}</div>'
        )

    # Compute total from main components
    main_keys = list(MAIN_COMPONENTS.keys())
    total_val = 0
    total_prev = 0
    for k in main_keys:
        v, d, p = _get_vals(k)
        if v is not None:
            total_val += v
            if p is not None:
                total_prev += p
    total_delta = total_val - total_prev if total_prev else None

    # Date label (one place for both dates)
    st.markdown(
        f'<div style="font-size:0.8rem;color:#6b7280;margin-bottom:0.5rem;">'
        f'Данные на <b style="color:#9ca3af;">{latest_str}</b>, '
        f'изменение к <b style="color:#9ca3af;">{prev_str}</b></div>',
        unsafe_allow_html=True,
    )

    # Row 1: Total (full width, large)
    st.markdown(
        _card_html("АКТИВЫ ДОМОХОЗЯЙСТВ (ВСЕГО)",
                   total_val, total_delta, total_prev,
                   color="#f0b429", big=True),
        unsafe_allow_html=True,
    )

    # Row 2-3: Components
    components = [
        ("HH_DEPOSITS_TOTAL", "Депозиты", "#f0b429"),
        ("HH_EQUITIES_TOTAL", "Акции и паи ИФ*", "#cc5de8"),
        ("HH_CASH_TOTAL", "Наличные", "#ffa94d"),
        ("HH_INSURANCE_PENSION", "Страховые / пенсионные", "#20c997"),
        ("HH_ESCROW_CBR", "Эскроу", "#ff6b6b"),
        ("HH_BONDS", "Облигации", "#51cf66"),
        ("HH_BROKERAGE", "Брокерские счета", "#74c0fc"),
    ]

    for row_items in [components[:4], components[4:]]:
        cols = st.columns(4)
        for i, (ind, label, color) in enumerate(row_items):
            v, d, p = _get_vals(ind)
            if v is not None:
                with cols[i]:
                    st.markdown(_card_html(label, v, d, p, color=color), unsafe_allow_html=True)


def _render_table(pivot: pd.DataFrame) -> None:
    """Render quarterly data table with all components."""
    st.markdown("##### Помесячные данные (млрд ₽)")

    # Rename columns to Russian labels
    rename = {}
    for ind, label in INDICATOR_LABELS.items():
        if ind in pivot.columns:
            rename[ind] = label

    display = pivot.rename(columns=rename)
    display = display[[c for c in display.columns if c in rename.values()]]
    display = display.iloc[::-1]  # latest on top
    display.index.name = "Дата"

    fmt = {c: "{:,.0f}" for c in display.columns}
    st.dataframe(
        display.style.format(fmt, na_rep="—"),
        use_container_width=True,
        height=500,
    )

    # CSV download
    buf = io.StringIO()
    display.to_csv(buf)
    st.download_button(
        label="Скачать CSV",
        data=buf.getvalue(),
        file_name="moex_hh_assets.csv",
        mime="text/csv",
    )


def _render_methodology() -> None:
    """Methodology expander."""
    with st.expander("Методология и источники", expanded=False):
        st.markdown("""
**Источник:** ЦБ РФ — «Финансовые активы и обязательства д/х по отдельным финансовым инструментам»

**Файл:** [households_bm.xlsx](https://cbr.ru/vfs/statistics/households/households_bm.xlsx)

**Периодичность:** Ежемесячно (с января 2018)

| Категория | Описание |
|-----------|----------|
| **Наличная валюта** | Нац. валюта (обязательство ЦБ) + иностранная валюта |
| **Депозиты** | Переводные (текущие), срочные (вкл. валютные и драгметаллы), в банках-нерезидентах |
| **Брокерские счета** | Средства на счетах у профучастников рынка ЦБ |
| **Облигации** | Облигации, векселя, сберегательные сертификаты |
| **Акции и паи ИФ*** | Котируемые/некотируемые акции, паи ПИФ, иностранные акции |
| **Страховые/пенсионные** | Резервы по страхованию жизни, пенсионные резервы и накопления |
| **Эскроу** | Средства физлиц по ДДУ и сделкам купли-продажи недвижимости |

*Помесячные данные по акциям содержат ограниченный набор — некотируемые акции нерезидентов не включены (полный набор в квартальной публикации: ~60 трлн vs ~31 трлн).
""")
