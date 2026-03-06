"""Sidebar controls: instrument class, period, factor selectors."""

from datetime import date

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

# Factor labels (Russian)
FACTORS = {
    "volatility": "Волатильность",
    "trend_strength": "Сила тренда (ADX)",
    "trend_direction": "Направление тренда",
    "index_level": "Уровень IMOEX",
    "market_cap": "Капитализация",
    "imoex_return": "Доходность IMOEX",
    "rgbi_return": "Доходность RGBI",
    "brent": "Нефть Brent",
    "usd_rub": "Курс USD/RUB",
    "cny_rub": "Курс CNY/RUB",
    "key_rate": "Ключевая ставка",
    "m2": "Денежная масса М2",
    "cpi_yoy": "Инфляция ИПЦ (г/г)",
    "real_rate": "Реальная ставка",
    "volume_momentum": "Импульс оборотов",
}

# Detailed factor descriptions with formulas (for tooltips/popovers)
FACTOR_DESCRIPTIONS = {
    "volatility": {
        "name": "Реализованная волатильность",
        "description": "Историческая волатильность индекса IMOEX, рассчитанная по дневным логарифмическим доходностям за 20 торговых дней, аннуализированная.",
        "formula": "sigma = std(ln(P_t / P_{t-1}), 20d) * sqrt(252) * 100",
        "unit": "%",
    },
    "trend_strength": {
        "name": "Сила тренда (ADX)",
        "description": "Average Directional Index — индикатор силы тренда. >25 — сильный тренд, <20 — слабый/боковик.",
        "formula": "ADX = EMA(DX, 14), DX = |+DI - -DI| / (+DI + -DI) * 100",
        "unit": "пункты",
    },
    "trend_direction": {
        "name": "Направление тренда",
        "description": "Изменение 20-дневной экспоненциальной скользящей средней IMOEX. Положительное = рост, отрицательное = падение.",
        "formula": "delta_EMA = (EMA20_t / EMA20_{t-1} - 1) * 100",
        "unit": "%",
    },
    "index_level": {
        "name": "Уровень IMOEX",
        "description": "Значение закрытия индекса Московской биржи — основного индикатора российского фондового рынка.",
        "formula": "Цена закрытия IMOEX",
        "unit": "пункты",
    },
    "market_cap": {
        "name": "Капитализация рынка",
        "description": "Суммарная капитализация компаний, входящих в индекс IMOEX.",
        "formula": "CAPITALIZATION / 10^12",
        "unit": "трлн руб.",
    },
    "imoex_return": {
        "name": "Дневная доходность IMOEX",
        "description": "Процентное изменение индекса IMOEX за один торговый день.",
        "formula": "(P_t / P_{t-1} - 1) * 100",
        "unit": "%",
    },
    "rgbi_return": {
        "name": "Дневная доходность RGBI",
        "description": "Процентное изменение индекса государственных облигаций RGBI. Индикатор настроений на рынке ОФЗ.",
        "formula": "(P_t / P_{t-1} - 1) * 100",
        "unit": "%",
    },
    "brent": {
        "name": "Нефть Brent",
        "description": "Цена нефти марки Brent — ключевой макрофактор для российского рынка и бюджета.",
        "formula": "Цена закрытия, USD/баррель",
        "unit": "USD",
    },
    "usd_rub": {
        "name": "Курс USD/RUB",
        "description": "Официальный курс доллара США к российскому рублю по данным ЦБ РФ.",
        "formula": "Курс ЦБ РФ",
        "unit": "руб.",
    },
    "cny_rub": {
        "name": "Курс CNY/RUB",
        "description": "Курс китайского юаня к рублю. Отражает торговые связи с Китаем.",
        "formula": "Курс ЦБ РФ",
        "unit": "руб.",
    },
    "key_rate": {
        "name": "Ключевая ставка ЦБ РФ",
        "description": "Основная процентная ставка Банка России. Определяет стоимость кредитования в экономике.",
        "formula": "Ступенчатая функция (изменяется на заседаниях ЦБ)",
        "unit": "%",
    },
    "m2": {
        "name": "Денежная масса М2",
        "description": "Широкая денежная масса — наличные + безналичные средства. Индикатор ликвидности в экономике.",
        "formula": "Агрегат M2 / 10^12",
        "unit": "трлн руб.",
    },
    "cpi_yoy": {
        "name": "Инфляция ИПЦ (год к году)",
        "description": "Индекс потребительских цен — основной показатель инфляции.",
        "formula": "(CPI_t / CPI_{t-12} - 1) * 100",
        "unit": "%",
    },
    "real_rate": {
        "name": "Реальная процентная ставка",
        "description": "Разница между ключевой ставкой и инфляцией. Показывает реальную стоимость денег.",
        "formula": "KEY_RATE - CPI_YOY",
        "unit": "п.п.",
    },
    "volume_momentum": {
        "name": "Импульс оборотов",
        "description": "Скользящее среднее совокупного дневного оборота за 5 дней с лагом в 1 день. Показывает недавнюю торговую активность.",
        "formula": "SMA(Total_Turnover, 5).shift(1) / 10^9",
        "unit": "млрд руб.",
    },
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
    default_on = {"volatility", "index_level", "usd_rub", "brent", "key_rate"}
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
