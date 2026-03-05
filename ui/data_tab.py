"""Tab 5: Data health — freshness table and refresh button."""

import streamlit as st
import pandas as pd

from data_pipeline.db import get_client, max_date, row_count


DATA_SOURCES = [
    ("vol_daily_turnovers", "Обороты (MOEX ISS)", "trade_date"),
    ("vol_index_history", "Индексы (IMOEX, RGBI, RVI)", "trade_date"),
    ("vol_currency_rates", "Курсы валют (ЦБР)", "trade_date"),
    ("vol_macro", "Макроданные (ключевая ставка, М2, ИПЦ)", "period_date"),
    ("vol_weekly_volumes", "Недельные объёмы (агрегат)", "week_start"),
    ("vol_weekly_factors", "Недельные факторы (агрегат)", "week_start"),
]


def render_data_tab():
    """Render the data health and refresh tab."""
    st.subheader("Состояние данных")

    client = get_client()

    # Build freshness table
    rows = []
    for table, label, date_col in DATA_SOURCES:
        last = max_date(client, table, date_col)
        count = row_count(client, table)
        rows.append({
            "Источник": label,
            "Таблица": table,
            "Последняя дата": last or "—",
            "Записей": count,
            "Статус": "✅" if last else "⚠️ Пусто",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- Refresh button ---
    st.subheader("Обновление данных")
    st.caption(
        "Загрузит новые данные из MOEX ISS и ЦБР, "
        "пересчитает недельные агрегаты и факторы. "
        "Первоначальная загрузка (с 2018 г.) может занять 10–15 минут."
    )

    if st.button("🔄 Обновить данные", type="primary"):
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
