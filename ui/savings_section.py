"""Section: Household savings structure and dynamics."""

from __future__ import annotations

import io

import streamlit as st
import pandas as pd

from ui.savings_charts import (
    COMPONENTS,
    COMPONENT_ORDER,
    savings_stacked_area,
    savings_donut,
    savings_component_lines,
)


# Indicators to load from vol_macro
_SAVINGS_INDICATORS = [
    "HH_SAVINGS_TOTAL",
    "HH_DEPOSITS_NO_ESCROW",
    "HH_ESCROW",
    "HH_TERM_DEPOSITS",
    "HH_SAVINGS_ACCOUNTS",
    "HH_CURRENT_ACCOUNTS",
    "HH_FX_DEPOSITS",
    "HH_CASH_M0",
]


def render_savings_section(params: dict, fetch_func) -> None:
    """Render the household savings tab."""
    st.markdown(
        '<div class="section-header">'
        '<h2>Сбережения домохозяйств</h2>'
        '<p>Структура и динамика финансовых активов населения по данным ЦБ РФ</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Load data
    raw = _load_savings_data(fetch_func, params)
    if raw.empty:
        st.info("Нет данных по сбережениям. Запустите пайплайн.")
        return

    # Pivot: index=period_date, columns=indicator
    pivot = raw.pivot_table(
        index="period_date", columns="indicator", values="value", aggfunc="last",
    ).sort_index()

    # --- KPI cards ---
    _render_kpi(pivot)

    # --- Charts ---
    col_area, col_donut = st.columns([3, 2])

    with col_area:
        st.markdown("##### Динамика структуры сбережений")
        fig = savings_stacked_area(pivot)
        st.plotly_chart(fig, use_container_width=True)

    with col_donut:
        st.markdown("##### Текущая структура")
        latest = {}
        for ind in COMPONENT_ORDER:
            if ind in pivot.columns:
                s = pivot[ind].dropna()
                if not s.empty:
                    latest[ind] = s.iloc[-1]
        if latest:
            fig = savings_donut(latest)
            st.plotly_chart(fig, use_container_width=True)

    # --- Individual component lines ---
    with st.expander("Динамика каждого компонента", expanded=False):
        fig = savings_component_lines(pivot)
        st.plotly_chart(fig, use_container_width=True)

    # --- Monthly table ---
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    _render_monthly_table(pivot)

    # --- Methodology ---
    _render_methodology()


def _load_savings_data(fetch_func, params: dict) -> pd.DataFrame:
    """Load savings indicators from vol_macro via Supabase."""
    try:
        from data_pipeline.db import get_client
        client = get_client()
        resp = (
            client.table("vol_macro")
            .select("period_date,indicator,value")
            .in_("indicator", _SAVINGS_INDICATORS)
            .gte("period_date", params["date_from"])
            .lte("period_date", params["date_to"])
            .order("period_date")
            .execute()
        )
        if resp.data:
            return pd.DataFrame(resp.data)
    except Exception:
        pass
    return pd.DataFrame()


def _render_kpi(pivot: pd.DataFrame) -> None:
    """Render top-level KPI cards."""
    # Total savings = deposits total + cash M0
    total_col = "HH_SAVINGS_TOTAL"
    cash_col = "HH_CASH_M0"

    cols = st.columns(4)

    # 1. Total deposits
    if total_col in pivot.columns:
        s = pivot[total_col].dropna()
        if len(s) >= 2:
            latest = s.iloc[-1]
            prev = s.iloc[-2]
            delta_abs = latest - prev
            delta_pct = (delta_abs / prev * 100) if prev else 0
            cols[0].metric(
                "Депозиты (всего)",
                f"{latest / 1000:,.1f} трлн ₽",
                f"{delta_abs / 1000:+,.1f} трлн ({delta_pct:+.1f}%)",
            )
        elif len(s) == 1:
            cols[0].metric("Депозиты (всего)", f"{s.iloc[-1] / 1000:,.1f} трлн ₽")

    # 2. Cash M0
    if cash_col in pivot.columns:
        s = pivot[cash_col].dropna()
        if len(s) >= 1:
            cols[1].metric(
                "Наличные (M0)",
                f"{s.iloc[-1] / 1000:,.1f} трлн ₽",
            )

    # 3. Grand total
    grand = 0
    for c in [total_col, cash_col]:
        if c in pivot.columns:
            s = pivot[c].dropna()
            if not s.empty:
                grand += s.iloc[-1]
    if grand > 0:
        cols[2].metric("Всего сбережений", f"{grand / 1000:,.1f} трлн ₽")

    # 4. Share of deposits vs cash
    if total_col in pivot.columns and cash_col in pivot.columns:
        dep = pivot[total_col].dropna()
        cash = pivot[cash_col].dropna()
        if not dep.empty and not cash.empty:
            dep_val = dep.iloc[-1]
            cash_val = cash.iloc[-1]
            total = dep_val + cash_val
            if total > 0:
                dep_pct = dep_val / total * 100
                cols[3].metric("Доля депозитов", f"{dep_pct:.0f}%")


def _render_monthly_table(pivot: pd.DataFrame) -> None:
    """Render monthly data table with all components."""
    st.markdown("##### Помесячные данные (млрд ₽)")

    # Rename columns to Russian
    rename = {}
    for ind, (label, _) in COMPONENTS.items():
        if ind in pivot.columns:
            rename[ind] = label
    # Add totals
    if "HH_SAVINGS_TOTAL" in pivot.columns:
        rename["HH_SAVINGS_TOTAL"] = "Депозиты (всего)"
    if "HH_DEPOSITS_NO_ESCROW" in pivot.columns:
        rename["HH_DEPOSITS_NO_ESCROW"] = "Депозиты (без эскроу)"

    display = pivot.rename(columns=rename)
    # Keep only renamed columns
    display = display[[c for c in display.columns if c in rename.values()]]
    display = display.iloc[::-1]  # latest on top
    display.index.name = "Месяц"

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
        file_name="moex_hh_savings.csv",
        mime="text/csv",
    )


def _render_methodology() -> None:
    """Methodology expander."""
    with st.expander("Методология и источники", expanded=False):
        st.markdown("""
**Источники данных ЦБ РФ:**

| Компонент | Файл | Периодичность |
|-----------|------|---------------|
| Депозиты (всего, эскроу) | 02_01_Funds_all.xlsx | Ежемесячно |
| Срочные, накопительные, текущие, валютные | monetary_agg.xlsx | Ежемесячно |
| Наличные M0 | monetary_agg.xlsx | Ежемесячно |

**Ограничения:**
- Ценные бумаги (акции, облигации, паи фондов) доступны только квартально
  и не включены в помесячный ряд.
- Драгметаллы учитываются ЦБ в составе «Депозиты в инвалюте и драгметаллах»
  (HH_FX_DEPOSITS), отдельно не выделяются.
- Страховые и пенсионные резервы доступны только квартально в агрегированном виде.
- Наличные M0 — это весь кэш в обращении (включая юрлица), доля домохозяйств
  оценивается в ~70-80% по методологии ЦБ.

**Все значения в млрд ₽** (для графиков переведены в трлн ₽).
""")
