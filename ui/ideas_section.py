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

    # --- Inline date and source filters ---
    from datetime import date
    impact_df["idea_date"] = pd.to_datetime(impact_df["idea_date"])
    min_date = impact_df["idea_date"].min().date()
    max_date = impact_df["idea_date"].max().date()

    fc1, fc2 = st.columns(2)
    with fc1:
        idea_date_from = st.date_input(
            "Идеи с", value=min_date, min_value=min_date, max_value=max_date,
            key="idea_date_from",
        )
    with fc2:
        idea_date_to = st.date_input(
            "Идеи по", value=max_date, min_value=min_date, max_value=max_date,
            key="idea_date_to",
        )

    impact_df = _filter_by_dates(
        impact_df,
        str(idea_date_from),
        str(idea_date_to),
    )

    if impact_df.empty:
        st.info("Нет данных за выбранный период.")
        return

    # --- Source filter (inline) ---
    all_sources = sorted(impact_df["source"].dropna().unique().tolist()) if "source" in impact_df.columns else []
    if all_sources:
        selected_sources = st.multiselect(
            "Фильтр по источнику (аналитику)",
            options=all_sources,
            default=all_sources,
            key="idea_source_filter",
        )
        if selected_sources and len(selected_sources) < len(all_sources):
            impact_df = impact_df[
                impact_df["source"].isin(selected_sources) | impact_df["source"].isna()
            ]

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

    # --- Block 3.5: Source analytics ---
    _render_source_block(impact_df, agg)

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
| **Пиковый AV** | Максимальный AV Ratio в окне события | Наибольший всплеск оборота среди дней [0, +3] вокруг публикации идеи |
| **NAV** (Normalized Abnormal Volume) | Нормализованный аномальный объём | `(V(t) − V̄) / σ` — аномальный объём в стандартных отклонениях |
| **CAV** (Cumulative Abnormal Volume) | Кумулятивный аномальный объём | `Σ (AV(t) − 1)` за окно [0, +3] — суммарный избыточный оборот за окно события |
| **ΔCAV** | Скорректированный CAV | `CAV_факт − медиана(CAV_плацебо)` — избыточный оборот за вычетом «фонового шума». Placebo-тест прогоняет 500 случайных псевдо-событий для того же тикера и считает CAV для каждого. ΔCAV показывает, насколько реальный CAV выше типичного случайного |
| **Placebo p-value** | Эмпирический p-value | Доля случайных псевдо-событий, у которых CAV ≥ наблюдаемого. p < 0.05 = реальный CAV находится в топ-5% случайного распределения — маловероятно случайность |
| **Значимость** | Статистическая значимость | Идея признаётся значимой, если placebo p-value < 0.05, т.е. наблюдаемый аномальный объём попадает в верхние 5% эмпирического распределения. Это не доказывает причинно-следственную связь, но показывает, что всплеск необычен |
| **d=0, d+1** | Дни относительно публикации | d=0 — день выхода идеи, d+1 — следующий торговый день и т.д. |
| **Estimation Window** | Окно оценки «нормы» | [-120, -6] торговых дней до события — период для расчёта среднего V̄ и стандартного отклонения σ. Выходные и праздники исключены |
| **Event Window** | Окно события | [0, +3] торговых дней — день публикации + 3 дня после |
""")


@st.cache_data(ttl=3600)
def _load_ideas_data(_fetch_func, params: dict):
    """Load ideas, impact results, and security history from Supabase."""
    ideas_df = pd.DataFrame(
        _fetch_func("investment_ideas", "ticker,idea_date,source")
    )
    impact_df = pd.DataFrame(
        _fetch_func(
            "idea_impact_results",
            "ticker,idea_date,source,est_mean_volume,est_std_volume,est_days,"
            "cav,delta_cav,placebo_pvalue,placebo_percentile,placebo_median_cav,"
            "peak_av_ratio,peak_av_day,peak_nav,is_significant,"
            "av_ratio_by_day,nav_by_day,volume_by_day",
        )
    )
    history_df = pd.DataFrame(
        _fetch_func("idea_security_history", "trade_date,ticker,value_rub")
    )
    return ideas_df, impact_df, history_df


def _filter_by_dates(impact_df: pd.DataFrame, date_from: str, date_to: str) -> pd.DataFrame:
    """Filter impact results by date range."""
    if impact_df.empty:
        return impact_df
    df = impact_df.copy()
    df["idea_date"] = pd.to_datetime(df["idea_date"])
    return df[
        (df["idea_date"] >= pd.Timestamp(date_from))
        & (df["idea_date"] <= pd.Timestamp(date_to))
    ]


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
            "source": row.get("source"),
            "cav": row.get("cav", 0),
            "delta_cav": row.get("delta_cav", row.get("cav", 0)),
            "placebo_pvalue": row.get("placebo_pvalue"),
            "placebo_percentile": row.get("placebo_percentile"),
            "placebo_median_cav": row.get("placebo_median_cav"),
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
    """Block 1: KPI cards — net effect prominently on top, then details."""
    # --- Hero KPI: net turnover effect ---
    net_all = agg.get("net_turnover_effect_rub", 0)
    net_sig = agg.get("net_turnover_effect_sig_rub", 0)

    # Format in billions if large enough, else millions
    def _fmt_rub(val):
        if abs(val) >= 1e9:
            return f"{val / 1e9:+,.1f} млрд ₽"
        return f"{val / 1e6:+,.0f} млн ₽"

    hero1, hero2 = st.columns(2)
    hero1.metric(
        "Суммарный чистый эффект на обороты (все идеи)",
        _fmt_rub(net_all),
        help=(
            "Сумма ΔCAV × средний дневной оборот по каждой идее. "
            "ΔCAV = CAV_факт − медиана(CAV_плацебо). "
            "Учитываются ВСЕ идеи (и положительные, и отрицательные). "
            "Это оценка суммарного избыточного оборота сверх фонового шума за окно [0, +3] дней"
        ),
    )
    hero2.metric(
        "Чистый эффект (только значимые, p<0.05)",
        _fmt_rub(net_sig),
        help=(
            "То же, но только для идей с placebo p-value < 0.05 — "
            "где аномальный объём маловероятно случаен"
        ),
    )

    st.markdown("")

    # --- Detail KPI row ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Всего идей", f"{agg.get('n_total', 0)}",
        help="Количество инвестиционных идей за выбранный период",
    )
    c2.metric(
        "Значимых (placebo p<0.05)",
        f"{agg.get('n_significant', 0)} ({agg.get('pct_significant', 0):.0f}%)",
        help="Идеи, у которых CAV попал в верхние 5% эмпирического плацебо-распределения",
    )
    c3.metric(
        "Средний ΔCAV", f"{agg.get('mean_delta_cav', 0):+.2f}",
        help="Скорректированный CAV = CAV_факт − медиана(CAV_плацебо). Может быть отрицательным",
    )
    c4.metric(
        "Медиана пикового AV",
        f"{agg.get('median_peak_av_ratio', 1):.2f}x",
        help="Медианное значение пикового AV Ratio — во сколько раз оборот превышал норму в самый активный день",
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
            top_df.columns = ["Тикер", "Идей", "Сред. ΔCAV", "Сред. CAV", "Сред. пик AV", "% значимых"]
            st.dataframe(
                top_df.style.format({
                    "Сред. ΔCAV": "{:+.3f}",
                    "Сред. CAV": "{:+.3f}",
                    "Сред. пик AV": "{:.2f}x",
                    "% значимых": "{:.0f}%",
                }),
                use_container_width=True,
                hide_index=True,
                height=360,
            )


def _render_source_block(impact_df: pd.DataFrame, agg: dict):
    """Block 3.5: Source (analyst) analytics."""
    st.markdown("##### Аналитика по источникам идей")
    st.markdown(
        '<div class="glass-card" style="border-left: 3px solid #74c0fc; padding: 0.8rem 1rem;">'
        '<div style="font-size: 0.85rem; color: #d1d5db; line-height: 1.55;">'
        'Сравнение аналитиков/брокеров по влиянию их инвестидей на торговые обороты. '
        'Средний CAV показывает, насколько в среднем вырос кумулятивный оборот после публикации идеи.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    source_summary = agg.get("source_summary", [])
    if not source_summary:
        st.info("Нет данных по источникам.")
        return

    col_table, col_chart = st.columns([1, 1])

    with col_table:
        st.markdown("###### Топ источники по среднему CAV")
        src_df = pd.DataFrame(source_summary)
        src_df.columns = ["Источник", "Идей", "Сред. ΔCAV", "Сред. CAV", "Сред. пик AV", "% значимых"]
        st.dataframe(
            src_df.style.format({
                "Сред. ΔCAV": "{:+.3f}",
                "Сред. CAV": "{:+.3f}",
                "Сред. пик AV": "{:.2f}x",
                "% значимых": "{:.0f}%",
            }),
            use_container_width=True,
            hide_index=True,
            height=500,
        )

    with col_chart:
        from ui.ideas_charts import source_comparison_chart
        fig = source_comparison_chart(source_summary)
        st.plotly_chart(fig, use_container_width=True)


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
            "source": row.get("source", ""),
            "is_significant": row.get("is_significant", False),
        })
    fig = timeline_scatter(ideas_data)
    st.plotly_chart(fig, use_container_width=True)


@st.fragment
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
            src = row.get("source", "") or ""
            src_label = f" [{src}]" if src else ""
            options.append(f"{row['ticker']} — {date_str}{src_label} (AV: {av:.2f}x) {sig}")

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

        # Source label
        source_val = row.get("source", "") or "Неизвестно"
        st.markdown(f"**Источник:** {source_val}")

        # Metrics row
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        dcav = row.get("delta_cav", row.get("cav", 0))
        mc1.metric(
            "ΔCAV", f"{dcav:+.3f}" if dcav is not None else "—",
            help="Скорректированный CAV = CAV_факт − медиана(CAV_плацебо). Избыточный оборот сверх фонового шума",
        )
        mc2.metric(
            "CAV (сырой)", f"{row.get('cav', 0):+.3f}",
            help="Cumulative Abnormal Volume без плацебо-коррекции",
        )
        mc3.metric(
            "Пик AV", f"{row.get('peak_av_ratio', 1):.2f}x (d{row.get('peak_av_day', 0):+d})",
            help="Максимальное отношение оборота к норме и день, когда оно достигнуто (d=0 — день идеи)",
        )
        pval = row.get("placebo_pvalue")
        pval_str = f"{pval:.3f}" if pval is not None else "—"
        mc4.metric(
            "Placebo p-value", pval_str,
            help="Эмпирический p-value: доля случайных псевдо-событий с CAV ≥ наблюдаемого. p<0.05 = маловероятно случайность",
        )
        mc5.metric(
            "Значимость", "Да ★" if row.get("is_significant") else "Нет",
            help="Placebo p-value < 0.05 — аномальный объём в верхних 5% случайного распределения",
        )


def _render_methodology():
    """Block 7: Methodology explanation."""
    with st.expander("Методология", expanded=False):
        st.markdown("""
**Event Study для торговых объёмов (v2 — с плацебо-тестом)**

Для каждой инвестиционной идеи оценивается аномальный объём торгов:

1. **Estimation Window** [-120, -6] торговых дней — определяет «нормальный» уровень оборота
   (среднее V̄ и стандартное отклонение σ). Выходные и праздники исключены.

2. **Event Window** [0, +3] торговых дней — день публикации + 3 дня после.

3. **Abnormal Volume (AV Ratio)** = V(t) / V̄
   — во сколько раз фактический оборот отличается от нормы.

4. **Cumulative AV (CAV)** = Σ (AV(t) − 1) за окно [0, +3].

5. **Плацебо-тест** (ключевое отличие от v1): для каждого тикера прогоняется 500 случайных
   псевдо-событий по тому же временному ряду. Для каждого считается CAV. Это даёт
   эмпирическое распределение CAV «без идеи».

6. **ΔCAV** = CAV_факт − медиана(CAV_плацебо) — избыточный аномальный объём за вычетом
   фонового шума. Положительный ΔCAV ≠ доказанный эффект, но показывает, что всплеск
   необычнее среднего.

7. **Placebo p-value** = доля псевдо-событий с CAV ≥ наблюдаемого. p < 0.05 означает,
   что CAV попал в верхние 5% случайного распределения.

**Важные ограничения:**
- Event study показывает **корреляцию, не каузальность**. Значимый ΔCAV не доказывает,
  что именно идея вызвала рост оборотов.
- Аномальные объёмы случаются и без идей (новости, дивиденды, экспирации и т.д.).
- Отрицательный ΔCAV также показывается — это честная оценка, а не только позитивные случаи.
- Одновременные события (несколько идей в один день) могут искажать оценку.
""")
