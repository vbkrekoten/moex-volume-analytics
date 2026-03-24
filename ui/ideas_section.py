"""Section: Investment Ideas Impact Analysis."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from analytics.event_study import aggregate_event_study
from ui.ideas_charts import (
    event_study_bar_chart,
    impact_distribution_chart,
    significance_pie_chart,
    idea_detail_chart,
    timeline_scatter,
)


def render_ideas_section(params: dict, fetch_func=None):
    """Render the investment ideas impact analysis section.

    Args:
        params: sidebar params (date_from, date_to, etc.)
        fetch_func: callable(table, columns) for fetching from Supabase.
    """
    st.markdown(
        '<div class="section-header">'
        '<h2>Анализ влияния инвестидей на обороты</h2>'
        '<p>Event study: оценка аномального объёма торгов после публикации инвестиционных идей</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if fetch_func is None:
        st.info("Функция загрузки данных не задана.")
        return

    # Load data
    ideas_df, impact_df, history_df = _load_ideas_data(fetch_func, params)

    if impact_df.empty:
        st.info(
            "Нет рассчитанных данных о влиянии инвестидей. "
            "Запустите пайплайн: `python -m data_pipeline.idea_pipeline`"
        )
        return

    # Filter by date range
    impact_df = _filter_by_dates(impact_df, params)

    if impact_df.empty:
        st.info("Нет данных за выбранный период.")
        return

    # Prepare aggregated results
    results_list = _impact_df_to_results(impact_df)
    agg = aggregate_event_study(results_list)

    # --- Glossary ---
    _render_glossary()

    # --- Block 1: KPI cards ---
    _render_kpi_cards(agg)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # --- Block 2: Event study chart + significance ---
    _render_event_study_block(agg)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # --- Block 3: Distribution + Top tickers ---
    _render_distribution_block(impact_df, agg)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # --- Block 4: Timeline scatter ---
    _render_timeline(impact_df)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # --- Block 5: Detail explorer ---
    _render_detail_explorer(impact_df, history_df)

    # --- Block 6: Methodology ---
    _render_methodology()


def _render_glossary():
    """Expandable glossary of terms used in this section."""
    with st.expander("📖 Глоссарий терминов", expanded=False):
        st.markdown("""
| Термин | Расшифровка | Формула / пояснение |
|--------|-------------|---------------------|
| **AV Ratio** (Abnormal Volume Ratio) | Коэффициент аномального объёма | `V(t) / V̄` — отношение фактического оборота к среднему за «нормальный» период. Значение 2.0x означает, что оборот был в 2 раза выше нормы |
| **Пиковый AV** | Максимальный AV Ratio в окне события | Наибольший всплеск оборота среди дней [-1, +5] вокруг публикации идеи |
| **NAV** (Normalized Abnormal Volume) | Нормализованный аномальный объём | `(V(t) − V̄) / σ` — аномальный объём в стандартных отклонениях. Используется для статистической проверки значимости |
| **CAV** (Cumulative Abnormal Volume) | Кумулятивный аномальный объём | `Σ (AV(t) − 1)` за окно [-1, +5] — суммарный избыточный оборот за всё окно события. CAV = +2.0 означает, что за 7 дней суммарно «лишних» оборотов набралось на 2 нормальных дня |
| **Значимость (p<0.05)** | Статистическая значимость | Идея признаётся значимой, если |NAV| > 1.96 хотя бы в один день окна — т.е. аномалия не объясняется обычной волатильностью с вероятностью 95% |
| **d=0, d+1, d-1** | Дни относительно публикации | d=0 — день выхода идеи, d-1 — день до, d+1 — день после и т.д. |
| **Estimation Window** | Окно оценки «нормы» | [-120, -6] торговых дней до события — период, по которому считается средний оборот V̄ и стандартное отклонение σ |
| **Event Window** | Окно события | [-1, +5] дней — период измерения аномального оборота |
""")


@st.cache_data(ttl=3600)
def _load_ideas_data(_fetch_func, params: dict):
    """Load ideas, impact results, and security history from Supabase."""
    ideas_df = pd.DataFrame(
        _fetch_func("investment_ideas", "ticker,idea_date")
    )
    impact_df = pd.DataFrame(
        _fetch_func(
            "idea_impact_results",
            "ticker,idea_date,est_mean_volume,est_std_volume,est_days,"
            "cav,peak_av_ratio,peak_av_day,peak_nav,is_significant,"
            "av_ratio_by_day,nav_by_day,volume_by_day",
        )
    )
    history_df = pd.DataFrame(
        _fetch_func("idea_security_history", "trade_date,ticker,value_rub")
    )
    return ideas_df, impact_df, history_df


def _filter_by_dates(impact_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Filter impact results by sidebar date range."""
    if impact_df.empty:
        return impact_df
    df = impact_df.copy()
    df["idea_date"] = pd.to_datetime(df["idea_date"])
    date_from = pd.Timestamp(params.get("date_from", "2020-01-01"))
    date_to = pd.Timestamp(params.get("date_to", "2030-01-01"))
    return df[(df["idea_date"] >= date_from) & (df["idea_date"] <= date_to)]


def _impact_df_to_results(impact_df: pd.DataFrame) -> list[dict]:
    """Convert impact DataFrame to list of dicts for aggregate_event_study."""
    results = []
    for _, row in impact_df.iterrows():
        av_by_day = row.get("av_ratio_by_day", {})
        nav_by_day = row.get("nav_by_day", {})
        if isinstance(av_by_day, str):
            import json
            av_by_day = json.loads(av_by_day)
        if isinstance(nav_by_day, str):
            import json
            nav_by_day = json.loads(nav_by_day)

        results.append({
            "ticker": row["ticker"],
            "idea_date": str(row["idea_date"]),
            "cav": row.get("cav", 0),
            "peak_av_ratio": row.get("peak_av_ratio", 1),
            "peak_av_day": row.get("peak_av_day", 0),
            "peak_nav": row.get("peak_nav", 0),
            "is_significant": row.get("is_significant", False),
            "av_ratio_by_day": av_by_day if isinstance(av_by_day, dict) else {},
            "nav_by_day": nav_by_day if isinstance(nav_by_day, dict) else {},
            "est_mean_volume": row.get("est_mean_volume", 0),
            "est_std_volume": row.get("est_std_volume", 0),
        })
    return results


def _render_kpi_cards(agg: dict):
    """Block 1: KPI cards."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Всего идей", f"{agg.get('n_total', 0)}",
        help="Количество инвестиционных идей за выбранный период",
    )
    c2.metric(
        "Значимых (p<0.05)",
        f"{agg.get('n_significant', 0)} ({agg.get('pct_significant', 0):.0f}%)",
        help="Идеи, после которых аномальный объём (NAV) превысил 1.96σ — т.е. всплеск оборота статистически не случаен с вероятностью 95%",
    )
    c3.metric(
        "Средний CAV", f"{agg.get('mean_cav', 0):+.2f}",
        help="Cumulative Abnormal Volume — средний кумулятивный избыточный оборот за окно [-1, +5] дней. Положительное значение = идеи в среднем увеличивают обороты",
    )
    c4.metric(
        "Медиана пикового AV",
        f"{agg.get('median_peak_av_ratio', 1):.2f}x",
        help="Медианное значение пикового AV Ratio — во сколько раз оборот превышал норму в самый активный день. Например, 1.50x = оборот был на 50% выше нормы",
    )


def _render_event_study_block(agg: dict):
    """Block 2: Event study bar chart + explanation."""
    st.markdown("##### Средний аномальный объём по дням окна")
    st.markdown(
        '<div class="glass-card" style="border-left: 3px solid #00d4ff; padding: 0.8rem 1rem;">'
        '<div style="font-size: 0.85rem; color: #d1d5db; line-height: 1.55;">'
        '<b>Как читать:</b> столбики показывают, во сколько раз средний оборот '
        'по всем идеям отличался от нормы в каждый день окна. '
        '<b>d=0</b> — день публикации идеи. Усики — 95% доверительный интервал. '
        'Значение > 1.0 означает аномально высокий оборот.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    col_chart, col_pie = st.columns([3, 1])
    with col_chart:
        fig = event_study_bar_chart(agg.get("aav_by_day", {}))
        st.plotly_chart(fig, use_container_width=True)
    with col_pie:
        fig = significance_pie_chart(
            agg.get("n_significant", 0),
            agg.get("n_total", 1),
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_distribution_block(impact_df: pd.DataFrame, agg: dict):
    """Block 3: Distribution histogram + top tickers table."""
    col_hist, col_table = st.columns([1, 1])

    with col_hist:
        st.markdown("##### Распределение пиковых AV Ratio")
        peak_ratios = impact_df["peak_av_ratio"].dropna().tolist()
        fig = impact_distribution_chart(peak_ratios)
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown("##### Топ-тикеры по отклику")
        ticker_summary = agg.get("ticker_summary", [])
        if ticker_summary:
            top_df = pd.DataFrame(ticker_summary[:20])
            top_df.columns = ["Тикер", "Идей", "Сред. CAV", "Сред. пик AV", "% значимых"]
            st.dataframe(
                top_df.style.format({
                    "Сред. CAV": "{:+.3f}",
                    "Сред. пик AV": "{:.2f}x",
                    "% значимых": "{:.0f}%",
                }),
                use_container_width=True,
                hide_index=True,
                height=360,
            )


def _render_timeline(impact_df: pd.DataFrame):
    """Block 4: Timeline scatter of all ideas."""
    st.markdown("##### Timeline инвестидей")
    st.markdown(
        '<div class="glass-card" style="border-left: 3px solid #ffa94d; padding: 0.8rem 1rem;">'
        '<div style="font-size: 0.85rem; color: #d1d5db; line-height: 1.55;">'
        'Каждая точка — одна инвестидея. Размер пропорционален кумулятивному '
        'аномальному объёму (CAV). Зелёные — статистически значимые.'
        '</div></div>',
        unsafe_allow_html=True,
    )
    ideas_data = []
    for _, row in impact_df.iterrows():
        ideas_data.append({
            "idea_date": str(row["idea_date"])[:10],
            "peak_av_ratio": row.get("peak_av_ratio", 1),
            "cav": row.get("cav", 0),
            "ticker": row["ticker"],
            "is_significant": row.get("is_significant", False),
        })
    fig = timeline_scatter(ideas_data)
    st.plotly_chart(fig, use_container_width=True)


def _render_detail_explorer(impact_df: pd.DataFrame, history_df: pd.DataFrame):
    """Block 5: Single idea detail explorer."""
    with st.expander("Детальный анализ конкретной идеи", expanded=False):
        if impact_df.empty:
            st.info("Нет данных.")
            return

        # Build selection options
        options = []
        for _, row in impact_df.iterrows():
            date_str = str(row["idea_date"])[:10]
            av = row.get("peak_av_ratio", 1)
            sig = "★" if row.get("is_significant") else ""
            options.append(f"{row['ticker']} — {date_str} (AV: {av:.2f}x) {sig}")

        selected = st.selectbox(
            "Выберите идею", options, key="idea_detail_select",
        )
        if not selected:
            return

        idx = options.index(selected)
        row = impact_df.iloc[idx]
        ticker = row["ticker"]
        idea_date = str(row["idea_date"])[:10]

        # Get volume history for this ticker
        if history_df.empty:
            st.info("Нет истории торгов для этого тикера.")
            return

        ticker_hist = history_df[history_df["ticker"] == ticker].copy()
        if ticker_hist.empty:
            st.info(f"Нет истории для {ticker}.")
            return

        ticker_hist["trade_date"] = pd.to_datetime(ticker_hist["trade_date"])
        ticker_hist = ticker_hist.sort_values("trade_date")

        # Show ~30 days before and ~10 after
        idea_dt = pd.Timestamp(idea_date)
        mask = (
            (ticker_hist["trade_date"] >= idea_dt - pd.Timedelta(days=45))
            & (ticker_hist["trade_date"] <= idea_dt + pd.Timedelta(days=15))
        )
        window_hist = ticker_hist[mask]

        av_by_day = row.get("av_ratio_by_day", {})
        if isinstance(av_by_day, str):
            import json
            av_by_day = json.loads(av_by_day)

        vol_by_day = row.get("volume_by_day", {})
        if isinstance(vol_by_day, str):
            import json
            vol_by_day = json.loads(vol_by_day)

        vol_data = {
            "dates": window_hist["trade_date"].dt.strftime("%Y-%m-%d").tolist(),
            "values": window_hist["value_rub"].tolist(),
            "event_dates": vol_by_day.get("dates", []) if isinstance(vol_by_day, dict) else [],
            "event_values": vol_by_day.get("values", []) if isinstance(vol_by_day, dict) else [],
        }

        fig = idea_detail_chart(
            vol_data,
            row.get("est_mean_volume", 0),
            av_by_day if isinstance(av_by_day, dict) else {},
            ticker,
            idea_date,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Metrics row
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric(
            "CAV", f"{row.get('cav', 0):+.3f}",
            help="Cumulative Abnormal Volume — сумма (AV−1) за окно [-1,+5]. Показывает общий избыточный оборот",
        )
        mc2.metric(
            "Пик AV", f"{row.get('peak_av_ratio', 1):.2f}x (d{row.get('peak_av_day', 0):+d})",
            help="Максимальное отношение оборота к норме и день, когда оно достигнуто (d=0 — день идеи)",
        )
        mc3.metric(
            "Пик NAV", f"{row.get('peak_nav', 0):+.2f}σ",
            help="Пиковый нормализованный аномальный объём в стандартных отклонениях. |NAV|>1.96 = статистически значимо",
        )
        mc4.metric(
            "Значимость", "Да ★" if row.get("is_significant") else "Нет",
            help="Статистически значимый всплеск оборота (|NAV| > 1.96, p < 0.05) хотя бы в один день окна",
        )


def _render_methodology():
    """Block 7: Methodology explanation."""
    with st.expander("Методология", expanded=False):
        st.markdown("""
**Event Study для торговых объёмов**

Для каждой инвестиционной идеи (публикации) оценивается аномальный объём торгов:

1. **Estimation Window** [-120, -6] торговых дней — определяет «нормальный» уровень оборота
   (среднее V̄ и стандартное отклонение σ).

2. **Event Window** [-1, +5] дней — период измерения эффекта
   (день до публикации + 5 дней после).

3. **Abnormal Volume (AV Ratio)** = V(t) / V̄
   — во сколько раз фактический оборот отличается от нормы.

4. **Normalized AV (NAV)** = (V(t) − V̄) / σ
   — аномальный объём в стандартных отклонениях (для статистических тестов).

5. **Cumulative AV (CAV)** = Σ (AV(t) − 1) за окно [-1, +5]
   — кумулятивный избыточный оборот.

6. **Значимость**: идея считается значимой, если |NAV| > 1.96
   хотя бы в один день окна (p < 0.05).

**Ограничения:**
- Одновременные события (несколько идей в один день) могут искажать оценку.
- Macro-шоки в estimation window влияют на оценку «нормы».
- Модель предполагает постоянство среднего объёма в estimation window.
""")
