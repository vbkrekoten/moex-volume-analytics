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
        '<p>Структура финансовых активов населения по данным ЦБ РФ (квартальные балансы)</p>'
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
    """Render top-level KPI cards."""
    cols = st.columns(5)

    def _kpi(col_idx, indicator, label):
        if indicator in pivot.columns:
            s = pivot[indicator].dropna()
            if len(s) >= 2:
                latest, prev = s.iloc[-1], s.iloc[-2]
                delta = latest - prev
                pct = (delta / prev * 100) if prev else 0
                cols[col_idx].metric(
                    label, f"{latest / 1000:,.1f} трлн ₽",
                    f"{delta / 1000:+,.1f} ({pct:+.1f}%)",
                )
            elif len(s) == 1:
                cols[col_idx].metric(label, f"{s.iloc[-1] / 1000:,.1f} трлн ₽")

    _kpi(0, "HH_ASSETS_TOTAL", "Активы (всего)")
    _kpi(1, "HH_DEPOSITS_TOTAL", "Депозиты")
    _kpi(2, "HH_EQUITIES_TOTAL", "Акции и фонды")
    _kpi(3, "HH_CASH_TOTAL", "Наличные")
    _kpi(4, "HH_INSURANCE_PENSION", "Страх./пенс.")


def _render_table(pivot: pd.DataFrame) -> None:
    """Render quarterly data table with all components."""
    st.markdown("##### Квартальные данные (млрд ₽)")

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
**Источник:** ЦБ РФ — «Финансовые активы и обязательства сектора Домашние хозяйства»

**Файл:** [households_b.xlsx](https://cbr.ru/vfs/statistics/households/households_b.xlsx)

**Периодичность:** Квартальные данные (с Q1 2018)

**Структура активов по методологии СНС:**

| Категория | Описание |
|-----------|----------|
| **Наличная валюта** | Авуары в национальной и иностранной валюте. Наличная нац. валюта = обязательство ЦБ, наличная инвалюта = обязательство остального мира |
| **Депозиты** | Переводные (текущие счета, карты), срочные (в руб. и инвалюте, вкл. драгметаллы), в банках-нерезидентах |
| **Брокерские счета** | Средства на счетах у профучастников рынка ЦБ |
| **Долговые ЦБ** | Облигации, векселя, сберегательные сертификаты, депозитарные расписки на облигации |
| **Акции и участие** | Котируемые/некотируемые акции, паи ПИФ, депозитарные расписки, прочее участие в капитале |
| **Страховые/пенсионные** | Резервы по страхованию жизни, иному страхованию, пенсионные резервы и накопления |
| **Эскроу** | Средства физлиц по ДДУ и сделкам купли-продажи недвижимости |

**Ограничения:**
- Данные квартальные (не помесячные)
- Некотируемые акции и «прочее участие в капитале» включают оценочные значения
- Депозиты в банках-нерезидентах могут включать средства за пределами РФ
""")
