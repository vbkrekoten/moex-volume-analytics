"""Section 3: Data health — freshness table and refresh button."""

import streamlit as st
import pandas as pd

from data_pipeline.db import get_client, max_date, row_count


DATA_SOURCES = [
    ("vol_daily_turnovers", "Обороты (MOEX ISS)", "trade_date"),
    ("vol_index_history", "Индексы (IMOEX, RGBI)", "trade_date"),
    ("vol_currency_rates", "Курсы валют (ЦБР)", "trade_date"),
    ("vol_macro", "Макро (ставка, М2, ИПЦ)", "period_date"),
    ("vol_daily_factors", "Дневные факторы (15 шт.)", "trade_date"),
    ("vol_weekly_volumes", "Недельные обороты", "week_start"),
    ("vol_weekly_factors", "Недельные факторы", "week_start"),
]


def render_data_section():
    """Render the data health and refresh section."""
    st.markdown(
        '<div class="section-header">'
        '<h2>Состояние данных</h2>'
        '<p>Актуальность источников и обновление пайплайна</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    client = get_client()

    # Build freshness table
    rows = []
    for table, label, date_col in DATA_SOURCES:
        try:
            last = max_date(client, table, date_col)
            count = row_count(client, table)
        except Exception:
            last = None
            count = 0
        rows.append({
            "Источник": label,
            "Последняя дата": last or "—",
            "Записей": f"{count:,}",
            "Статус": "✅" if last else "⚠️",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Refresh button
    st.caption(
        "Загрузит новые данные из MOEX ISS и ЦБР, "
        "пересчитает дневные и недельные агрегаты. "
        "Первоначальная загрузка (с 2018 г.) может занять 10–15 минут."
    )

    if st.button("Обновить данные", type="primary", key="refresh_btn"):
        from data_pipeline.run_pipeline import run_full_pipeline

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def on_progress(stage: str, pct: float):
            progress_bar.progress(min(pct, 1.0))
            status_text.text(stage)

        try:
            run_full_pipeline(progress_callback=on_progress)
            st.success("Данные успешно обновлены!")
            st.rerun()
        except Exception as e:
            st.error(f"Ошибка при обновлении: {e}")
