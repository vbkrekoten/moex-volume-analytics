"""Section 5: Volatility & turnover forecast (HAR-RV model)."""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st
import pandas as pd
import numpy as np

from ui.sidebar import FACTORS, CLASS_MAP_REVERSE
from ui.forecast_charts import (
    fan_chart,
    turnover_scenario_chart,
    diagnostics_by_horizon_chart,
    har_coefficients_chart,
)
from analytics.forecasting import (
    HARRVModel,
    aggregate_forecast_monthly,
    estimate_turnover_regression,
    forecast_turnovers,
)


# ---------------------------------------------------------------------------
# Scenario presets
# ---------------------------------------------------------------------------

SCENARIO_PRESETS = {
    "Базовый": {
        "description": "Текущие значения макро-параметров сохраняются на прогнозном горизонте",
        "key_rate": None,  # will be filled with latest actual
        "brent": None,
        "usd_rub": None,
    },
    "Оптимистичный": {
        "description": "Смягчение ДКП, рост нефти, укрепление рубля",
        "key_rate": -3.0,  # delta from current
        "brent": +10.0,
        "usd_rub": -5.0,
    },
    "Пессимистичный": {
        "description": "Ужесточение ДКП, падение нефти, ослабление рубля",
        "key_rate": +3.0,
        "brent": -10.0,
        "usd_rub": +8.0,
    },
}


def _get_latest_factor(daily_factors: pd.DataFrame, factor_name: str) -> float | None:
    """Get the latest value of a factor from daily_factors."""
    if daily_factors.empty:
        return None
    fdf = daily_factors[daily_factors["factor_name"] == factor_name].copy()
    if fdf.empty:
        return None
    fdf["trade_date"] = pd.to_datetime(fdf["trade_date"])
    fdf = fdf.sort_values("trade_date")
    val = fdf["value"].iloc[-1]
    return float(val) if pd.notna(val) else None


def _load_forecasts(fetch_func) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load latest forecast and diagnostics from Supabase."""
    # Fetch latest forecast (volatility, base scenario)
    fc_data = fetch_func(
        "vol_forecasts",
        date_filter={"col": "forecast_date", "gte": (date.today() - timedelta(days=7)).isoformat()},
    )
    fc_df = pd.DataFrame(fc_data) if fc_data else pd.DataFrame()

    diag_data = fetch_func(
        "vol_forecast_diagnostics",
        date_filter={"col": "forecast_date", "gte": (date.today() - timedelta(days=7)).isoformat()},
    )
    diag_df = pd.DataFrame(diag_data) if diag_data else pd.DataFrame()

    return fc_df, diag_df


def render_forecast_section(
    daily_vol: pd.DataFrame,
    daily_factors: pd.DataFrame,
    params: dict,
    fetch_func=None,
):
    """Render the forecast section of the dashboard."""
    st.markdown(
        '<h2 style="margin-top:0;">📈 Прогноз волатильности и оборотов</h2>',
        unsafe_allow_html=True,
    )
    st.caption("HAR-RV модель • 6-месячный горизонт • Обновляется ежедневно")

    # --- Load pre-computed forecasts ---
    fc_df = pd.DataFrame()
    diag_df = pd.DataFrame()

    if fetch_func is not None:
        fc_df, diag_df = _load_forecasts(fetch_func)

    # Filter to latest volatility forecast (base scenario)
    vol_fc = pd.DataFrame()
    if not fc_df.empty:
        fc_df["forecast_date"] = pd.to_datetime(fc_df["forecast_date"])
        latest_date = fc_df["forecast_date"].max()
        vol_fc = fc_df[
            (fc_df["forecast_date"] == latest_date) &
            (fc_df["target_name"] == "volatility") &
            (fc_df["model_name"] == "har_rv") &
            (fc_df["scenario"] == "base")
        ].copy()
        for col in ["value", "ci_lower_80", "ci_upper_80", "ci_lower_95", "ci_upper_95"]:
            if col in vol_fc.columns:
                vol_fc[col] = pd.to_numeric(vol_fc[col], errors="coerce")

    # If no pre-computed forecast, try to compute on-the-fly
    if vol_fc.empty and not daily_factors.empty:
        vol_fc = _compute_forecast_inline(daily_factors)

    # --- Prepare RV history ---
    rv_history = pd.DataFrame()
    rvi_history = pd.DataFrame()
    if not daily_factors.empty:
        fdf = daily_factors.copy()
        fdf["trade_date"] = pd.to_datetime(fdf["trade_date"])
        fdf["value"] = pd.to_numeric(fdf["value"], errors="coerce")

        rv_data = fdf[fdf["factor_name"] == "volatility"].dropna(subset=["value"])
        if not rv_data.empty:
            rv_history = rv_data[["trade_date", "value"]].sort_values("trade_date")

        rvi_data = fdf[fdf["factor_name"] == "rvi"].dropna(subset=["value"])
        if not rvi_data.empty:
            rvi_history = rvi_data[["trade_date", "value"]].sort_values("trade_date")

    # --- KPI Cards ---
    _render_kpi_cards(rv_history, vol_fc, diag_df)

    # --- Fan chart ---
    st.plotly_chart(
        fan_chart(rv_history, vol_fc, rvi_history, history_months=12),
        use_container_width=True,
    )

    # --- Business explanation ---
    st.markdown(
        '<div class="glass-card" style="border-left: 3px solid #f0b429;">'
        '<div style="font-size: 0.88rem; color: #d1d5db; line-height: 1.55;">'
        '<b>Как читать график:</b> сплошная линия — историческая волатильность, '
        'пунктир — прогноз. Цветные полосы — доверительные интервалы: '
        'тёмная полоса = 80% вероятность (4 из 5 раз факт попадёт внутрь), '
        'светлая = 95%. Чем дальше горизонт, тем шире интервал — '
        'это нормально для финансовых прогнозов.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # --- Monthly forecast table ---
    if not vol_fc.empty:
        _render_monthly_table(vol_fc)

    # --- Turnover scenario analysis ---
    _render_turnover_scenarios(daily_vol, daily_factors, vol_fc)

    # --- Model diagnostics ---
    _render_diagnostics(diag_df)

    # --- Methodology ---
    _render_methodology()


def _compute_forecast_inline(daily_factors: pd.DataFrame) -> pd.DataFrame:
    """Compute forecast on-the-fly from daily factors (fallback when no DB forecast)."""
    fdf = daily_factors.copy()
    fdf["trade_date"] = pd.to_datetime(fdf["trade_date"])
    fdf["value"] = pd.to_numeric(fdf["value"], errors="coerce")

    rv_df = fdf[fdf["factor_name"] == "volatility"].dropna(subset=["value"])
    if rv_df.empty or len(rv_df) < 252:
        return pd.DataFrame()

    rv_series = rv_df.set_index("trade_date")["value"].sort_index()
    rv_series = rv_series[~rv_series.index.duplicated(keep="last")]

    try:
        model = HARRVModel()
        model.fit(rv_series, min_obs=252)
        return model.forecast(rv_series, horizon=126)
    except Exception:
        return pd.DataFrame()


def _render_kpi_cards(
    rv_history: pd.DataFrame,
    vol_fc: pd.DataFrame,
    diag_df: pd.DataFrame,
):
    """Render KPI cards row."""
    cols = st.columns(4)

    # Current RV
    with cols[0]:
        if not rv_history.empty:
            current_rv = float(rv_history["value"].iloc[-1])
            avg_rv = float(rv_history["value"].mean())
            delta = f"{current_rv - avg_rv:+.1f} vs среднее"
            st.metric("Текущая волатильность (RV)", f"{current_rv:.1f}%", delta)
        else:
            st.metric("Текущая RV", "—")

    # 6-month forecast
    with cols[1]:
        if not vol_fc.empty:
            avg_fc = float(vol_fc["value"].mean())
            ci80_lo = float(vol_fc["ci_lower_80"].mean()) if "ci_lower_80" in vol_fc.columns else None
            ci80_hi = float(vol_fc["ci_upper_80"].mean()) if "ci_upper_80" in vol_fc.columns else None
            ci_str = f" ({ci80_lo:.0f}–{ci80_hi:.0f}%)" if ci80_lo and ci80_hi else ""
            st.metric("Прогноз RV (6 мес. ср.)", f"{avg_fc:.1f}%{ci_str}")
        else:
            st.metric("Прогноз RV (6 мес.)", "—")

    # Model RMSE
    with cols[2]:
        rmse_val = "—"
        if not diag_df.empty:
            rmse_rows = diag_df[
                (diag_df["metric_name"] == "rmse") &
                (diag_df["eval_window"] == "22d")
            ]
            if not rmse_rows.empty:
                rmse_val = f'{float(rmse_rows["metric_value"].iloc[0]):.1f}%'
        st.metric("RMSE модели (1 мес.)", rmse_val)

    # Last forecast date
    with cols[3]:
        if not vol_fc.empty and "forecast_date" in vol_fc.columns:
            fc_date = pd.to_datetime(vol_fc["forecast_date"].iloc[0])
            st.metric("Дата прогноза", fc_date.strftime("%d.%m.%Y"))
        else:
            st.metric("Дата прогноза", date.today().strftime("%d.%m.%Y"))


def _render_monthly_table(vol_fc: pd.DataFrame):
    """Render monthly forecast table."""
    monthly = aggregate_forecast_monthly(vol_fc)
    if monthly.empty:
        return

    st.markdown("#### Помесячный прогноз волатильности")

    display = monthly[["month_label", "value"]].copy()
    display.columns = ["Месяц", "Прогноз RV, %"]

    if "ci_lower_80" in monthly.columns and "ci_upper_80" in monthly.columns:
        display["80% интервал"] = monthly.apply(
            lambda r: f"{r['ci_lower_80']:.1f} – {r['ci_upper_80']:.1f}%", axis=1
        )
    if "ci_lower_95" in monthly.columns and "ci_upper_95" in monthly.columns:
        display["95% интервал"] = monthly.apply(
            lambda r: f"{r['ci_lower_95']:.1f} – {r['ci_upper_95']:.1f}%", axis=1
        )

    st.dataframe(display, hide_index=True, use_container_width=True)


def _render_turnover_scenarios(
    daily_vol: pd.DataFrame,
    daily_factors: pd.DataFrame,
    vol_fc: pd.DataFrame,
):
    """Render interactive turnover scenario analysis."""
    with st.expander("💰 Прогноз оборотов — сценарный анализ", expanded=False):
        if daily_vol.empty or daily_factors.empty or vol_fc.empty:
            st.info("Недостаточно данных для прогноза оборотов")
            return

        # Get current macro values for slider defaults
        current_key_rate = _get_latest_factor(daily_factors, "key_rate") or 21.0
        current_brent = _get_latest_factor(daily_factors, "brent") or 65.0
        current_usd_rub = _get_latest_factor(daily_factors, "usd_rub") or 88.0
        current_index = _get_latest_factor(daily_factors, "index_level") or 2800.0

        # Scenario presets
        st.markdown("##### Макроэкономические допущения")
        preset_col, _ = st.columns([1, 2])
        with preset_col:
            preset = st.radio(
                "Пресет сценария",
                list(SCENARIO_PRESETS.keys()),
                horizontal=True,
                label_visibility="collapsed",
            )

        preset_info = SCENARIO_PRESETS[preset]
        st.caption(f"*{preset_info['description']}*")

        # Calculate preset values
        if preset == "Базовый":
            default_rate = current_key_rate
            default_brent = current_brent
            default_usd = current_usd_rub
        else:
            default_rate = current_key_rate + (preset_info["key_rate"] or 0)
            default_brent = current_brent + (preset_info["brent"] or 0)
            default_usd = current_usd_rub + (preset_info["usd_rub"] or 0)

        # Interactive sliders
        sl1, sl2, sl3 = st.columns(3)
        with sl1:
            sc_key_rate = st.slider(
                "Ключевая ставка, %",
                min_value=5.0, max_value=30.0,
                value=round(default_rate, 1),
                step=0.5,
                key=f"fc_key_rate_{preset}",
            )
        with sl2:
            sc_brent = st.slider(
                "Brent, $/барр.",
                min_value=30.0, max_value=120.0,
                value=round(default_brent, 1),
                step=1.0,
                key=f"fc_brent_{preset}",
            )
        with sl3:
            sc_usd_rub = st.slider(
                "USD/RUB, ₽",
                min_value=60.0, max_value=120.0,
                value=round(default_usd, 1),
                step=0.5,
                key=f"fc_usd_rub_{preset}",
            )

        # Estimate regression models (cached)
        reg_models = _cached_regression(daily_vol, daily_factors)

        if not reg_models:
            st.warning("Недостаточно данных для регрессионной модели оборотов")
            return

        # Average forecasted volatility
        avg_vol_fc = float(vol_fc["value"].mean()) if not vol_fc.empty else 20.0

        # Predict turnovers
        macro_params = {
            "key_rate": sc_key_rate,
            "brent": sc_brent,
            "usd_rub": sc_usd_rub,
            "index_level": current_index,
        }

        predictions = forecast_turnovers(avg_vol_fc, macro_params, reg_models)

        # Current ADTV for comparison
        current_adtv = _compute_current_adtv(daily_vol)

        # Display chart
        if predictions:
            st.plotly_chart(
                turnover_scenario_chart(predictions, current_adtv),
                use_container_width=True,
            )

            # Summary table
            rows = []
            for cls in sorted(predictions.keys()):
                pred_bln = predictions[cls] / 1e3
                curr_bln = current_adtv.get(cls, 0) / 1e3
                change = ((pred_bln / curr_bln - 1) * 100) if curr_bln > 0 else 0
                rows.append({
                    "Класс": CLASS_MAP_REVERSE.get(cls, cls),
                    "Текущий ADTV, млрд ₽": f"{curr_bln:,.1f}",
                    "Прогноз ADTV, млрд ₽": f"{pred_bln:,.1f}",
                    "Изменение": f"{change:+.1f}%",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _cached_regression(daily_vol: pd.DataFrame, daily_factors: pd.DataFrame):
    """Estimate turnover regression models."""
    return estimate_turnover_regression(daily_vol, daily_factors)


def _compute_current_adtv(daily_vol: pd.DataFrame, window: int = 30) -> dict[str, float]:
    """Compute current 30-day ADTV per instrument class."""
    if daily_vol.empty:
        return {}
    vol = daily_vol.copy()
    vol["trade_date"] = pd.to_datetime(vol["trade_date"])
    pivot = vol.pivot_table(
        index="trade_date", columns="instrument_class",
        values="value_rub", aggfunc="sum", fill_value=0,
    ).sort_index()
    if pivot.empty:
        return {}
    adtv = pivot.tail(window).mean()
    return adtv.to_dict()


def _render_diagnostics(diag_df: pd.DataFrame):
    """Render model diagnostics expander."""
    with st.expander("🔬 Диагностика модели", expanded=False):
        if diag_df.empty:
            st.info("Диагностика будет доступна после первого запуска пайплайна прогнозирования")
            return

        diag_df = diag_df.copy()
        diag_df["metric_value"] = pd.to_numeric(diag_df["metric_value"], errors="coerce")

        # HAR-RV coefficients
        coef_rows = diag_df[diag_df["metric_name"].str.startswith("coef_")]
        if not coef_rows.empty:
            st.markdown("##### Коэффициенты HAR-RV модели")
            coefficients = {}
            for _, row in coef_rows.iterrows():
                name = row["metric_name"].replace("coef_", "")
                coefficients[name] = float(row["metric_value"])
            st.plotly_chart(
                har_coefficients_chart(coefficients),
                use_container_width=True,
            )

            st.markdown(
                '<div class="glass-card" style="font-size: 0.85rem; color: #9ca3af;">'
                '<b>Интерпретация:</b> β_d — влияние вчерашней волатильности, '
                'β_w — влияние средней за неделю, β_m — за месяц. '
                'Чем больше β_m относительно β_d, тем более «инертна» волатильность.'
                '</div>',
                unsafe_allow_html=True,
            )

        # RMSE by horizon
        rmse_rows = diag_df[
            (diag_df["metric_name"] == "rmse") &
            (diag_df["eval_window"] != "overall")
        ]
        if not rmse_rows.empty:
            st.markdown("##### RMSE по горизонтам прогноза")
            per_horizon = {}
            for _, row in rmse_rows.iterrows():
                window_str = row["eval_window"]
                # Parse "1d", "5d", "22d", etc.
                try:
                    h = int(window_str.replace("d", ""))
                    per_horizon[h] = {"rmse": float(row["metric_value"])}
                except (ValueError, AttributeError):
                    pass
            if per_horizon:
                st.plotly_chart(
                    diagnostics_by_horizon_chart(per_horizon),
                    use_container_width=True,
                )

        # R²
        r2_rows = diag_df[diag_df["metric_name"] == "r2"]
        if not r2_rows.empty:
            r2_val = float(r2_rows["metric_value"].iloc[0])
            st.metric("R² модели (in-sample)", f"{r2_val:.4f}")


def _render_methodology():
    """Render methodology expander."""
    with st.expander("📖 Методология прогнозирования", expanded=False):
        st.markdown("""
##### Модель HAR-RV

**HAR-RV** (Heterogeneous Autoregressive model of Realized Volatility) —
модель прогнозирования реализованной волатильности, основанная на идее
гетерогенных участников рынка с разными горизонтами:

- **β_d** — краткосрочные трейдеры (дневной компонент)
- **β_w** — среднесрочные инвесторы (недельный, 5 дней)
- **β_m** — долгосрочные фонды (месячный, 22 дня)

**Формула:**
```
RV_{t+1} = β₀ + β_d·RV_t + β_w·mean(RV_{t-4:t}) + β_m·mean(RV_{t-21:t}) + ε
```

##### Прогноз оборотов

Используется двухстадийный подход:
1. **Стадия 1**: HAR-RV прогнозирует волатильность на 6 месяцев
2. **Стадия 2**: OLS-регрессия связывает обороты с волатильностью и макро-факторами

Пользователь может менять макро-допущения (ставка, нефть, курс) и видеть
влияние на прогнозируемые обороты в реальном времени.

##### Ограничения

⚠️ **Модель наиболее надёжна на горизонте 1-2 месяца.** На 6 месяцев
доверительный интервал существенно расширяется — это отражает реальную
неопределённость финансовых рынков.

⚠️ **Модель не предсказывает «чёрных лебедей»** — экстремальные события,
которых не было в обучающей выборке (2018–сегодня).

⚠️ **Прогноз оборотов имеет двойную неопределённость**: неточность
прогноза волатильности + погрешность регрессионной модели.
""")
