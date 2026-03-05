"""Sidebar controls: instrument class, period, factor selectors."""

from datetime import date, datetime

import streamlit as st

ALL_CLASSES = ["Акции", "Облигации", "Фьючерсы", "Опционы", "Валюта", "РЕПО"]
CLASS_MAP = {
    "Акции": "shares",
    "Облигации": "bonds",
    "Фьючерсы": "futures",
    "Опционы": "options",
    "Валюта": "currency",
    "РЕПО": "repo",
}
CLASS_MAP_REVERSE = {v: k for k, v in CLASS_MAP.items()}

FACTORS = {
    "volatility": "Волатильность (RVI)",
    "trend_strength": "Сила тренда (ADX)",
    "trend_direction": "Направление тренда",
    "index_level": "Уровень IMOEX",
    "market_cap": "Капитализация рынка",
    "cpi_yoy": "Инфляция ИПЦ (г/г)",
    "usd_rub": "Курс USD/RUB",
    "cny_rub": "Курс CNY/RUB",
    "m2": "Денежная масса М2",
    "key_rate": "Ключевая ставка",
}


def render_sidebar() -> dict:
    """Render sidebar controls and return selected parameters."""
    st.sidebar.header("Инструменты")
    select_all = st.sidebar.checkbox("Все классы", value=True)
    if select_all:
        selected_classes = ALL_CLASSES
    else:
        selected_classes = st.sidebar.multiselect(
            "Классы инструментов",
            ALL_CLASSES,
            default=["Акции"],
        )

    st.sidebar.header("Период")
    col1, col2 = st.sidebar.columns(2)
    date_from = col1.date_input("С", value=date(2018, 1, 1), min_value=date(2018, 1, 1))
    date_to = col2.date_input("По", value=date.today())

    st.sidebar.header("Факторы")
    selected_factors = []
    default_on = {"volatility", "index_level", "usd_rub"}
    for key, label in FACTORS.items():
        if st.sidebar.checkbox(label, value=(key in default_on), key=f"factor_{key}"):
            selected_factors.append(key)

    return {
        "classes_ru": selected_classes,
        "classes_en": [CLASS_MAP[c] for c in selected_classes],
        "date_from": date_from.strftime("%Y-%m-%d"),
        "date_to": date_to.strftime("%Y-%m-%d"),
        "factors": selected_factors,
    }
