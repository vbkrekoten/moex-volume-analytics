"""Plotly chart builders for the forecast section."""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd
import numpy as np

from ui.charts import DARK_LAYOUT, COLORS, _hex_to_rgba
from ui.sidebar import CLASS_MAP_REVERSE


# ---------------------------------------------------------------------------
# 1. Fan chart: historical RV + forecast with expanding CI bands
# ---------------------------------------------------------------------------

def fan_chart(
    rv_history: pd.DataFrame,
    forecast: pd.DataFrame,
    rvi_history: pd.DataFrame | None = None,
    history_months: int = 12,
) -> go.Figure:
    """Fan chart: historical RV (solid) + forecast (dashed) with CI bands.

    Args:
        rv_history: DataFrame with columns (trade_date, value) — daily RV history.
        forecast: DataFrame with columns (target_date, value, ci_lower_80, ...).
        rvi_history: optional RVI overlay.
        history_months: months of history to show before forecast.
    """
    fig = go.Figure()

    if rv_history.empty and forecast.empty:
        fig.update_layout(**DARK_LAYOUT, title="Нет данных для прогноза")
        return fig

    # Prepare history
    if not rv_history.empty:
        hist = rv_history.copy()
        hist["trade_date"] = pd.to_datetime(hist["trade_date"])
        hist = hist.sort_values("trade_date")

        # Filter to last N months
        if history_months and len(hist) > 0:
            cutoff = hist["trade_date"].max() - pd.DateOffset(months=history_months)
            hist = hist[hist["trade_date"] >= cutoff]

        fig.add_trace(go.Scatter(
            x=hist["trade_date"],
            y=hist["value"],
            name="Реализованная волатильность (RV)",
            line=dict(color="#f0b429", width=2),
            hovertemplate="%{y:.1f}%<extra>RV</extra>",
        ))

    # Prepare forecast
    if not forecast.empty:
        fc = forecast.copy()
        fc["target_date"] = pd.to_datetime(fc["target_date"])
        fc = fc.sort_values("target_date")

        # Connect history to forecast with a bridge point
        if not rv_history.empty and len(hist) > 0:
            bridge_date = hist["trade_date"].iloc[-1]
            bridge_val = float(hist["value"].iloc[-1])
            # Add bridge point to forecast traces
            bridge = pd.DataFrame([{
                "target_date": bridge_date,
                "value": bridge_val,
                "ci_lower_80": bridge_val,
                "ci_upper_80": bridge_val,
                "ci_lower_95": bridge_val,
                "ci_upper_95": bridge_val,
            }])
            fc = pd.concat([bridge, fc], ignore_index=True)

        # 95% CI band
        if "ci_lower_95" in fc.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([fc["target_date"], fc["target_date"][::-1]]),
                y=pd.concat([fc["ci_upper_95"], fc["ci_lower_95"][::-1]]),
                fill="toself",
                fillcolor="rgba(240,180,41,0.08)",
                line=dict(color="rgba(0,0,0,0)"),
                name="95% доверительный интервал",
                hoverinfo="skip",
                showlegend=True,
            ))

        # 80% CI band
        if "ci_lower_80" in fc.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([fc["target_date"], fc["target_date"][::-1]]),
                y=pd.concat([fc["ci_upper_80"], fc["ci_lower_80"][::-1]]),
                fill="toself",
                fillcolor="rgba(240,180,41,0.18)",
                line=dict(color="rgba(0,0,0,0)"),
                name="80% доверительный интервал",
                hoverinfo="skip",
                showlegend=True,
            ))

        # Point forecast line
        fig.add_trace(go.Scatter(
            x=fc["target_date"],
            y=fc["value"],
            name="Прогноз RV",
            line=dict(color="#f0b429", width=2.5, dash="dash"),
            hovertemplate="%{y:.1f}%<extra>Прогноз</extra>",
        ))

    # Optional: RVI overlay
    if rvi_history is not None and not rvi_history.empty:
        rvi = rvi_history.copy()
        rvi["trade_date"] = pd.to_datetime(rvi["trade_date"])
        rvi = rvi.sort_values("trade_date")
        if history_months and len(rvi) > 0:
            cutoff = rvi["trade_date"].max() - pd.DateOffset(months=history_months)
            rvi = rvi[rvi["trade_date"] >= cutoff]

        fig.add_trace(go.Scatter(
            x=rvi["trade_date"],
            y=rvi["value"],
            name="RVI (подразумеваемая)",
            line=dict(color="#00d4ff", width=1.5, dash="dot"),
            hovertemplate="%{y:.1f}<extra>RVI</extra>",
            yaxis="y2",
        ))

    layout = {
        **DARK_LAYOUT,
        "height": 450,
        "hovermode": "x unified",
        "yaxis": dict(
            title="RV, % (годовых)",
            titlefont=dict(color="#f0b429"),
            tickfont=dict(color="#f0b429"),
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.06)",
        ),
        "legend": dict(
            bgcolor="rgba(17,24,39,0.7)",
            bordercolor="rgba(240,180,41,0.1)",
            borderwidth=1,
            font=dict(size=10),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    }

    if rvi_history is not None and not rvi_history.empty:
        layout["yaxis2"] = dict(
            title="RVI, пункты",
            titlefont=dict(color="#00d4ff", size=11),
            tickfont=dict(color="#00d4ff", size=10),
            overlaying="y",
            side="right",
            showgrid=False,
        )

    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# 2. Turnover scenario grouped bar chart
# ---------------------------------------------------------------------------

def turnover_scenario_chart(
    predictions: dict[str, float],
    current_adtv: dict[str, float] | None = None,
) -> go.Figure:
    """Grouped bar chart: predicted ADTV per instrument class vs current.

    Args:
        predictions: {instrument_class: predicted_daily_mln_rub}
        current_adtv: {instrument_class: current_30d_adtv_mln_rub}
    """
    fig = go.Figure()

    classes = sorted(predictions.keys())
    labels = [CLASS_MAP_REVERSE.get(c, c) for c in classes]
    pred_vals = [predictions[c] / 1e3 for c in classes]  # mln -> mlrd

    if current_adtv:
        curr_vals = [current_adtv.get(c, 0) / 1e3 for c in classes]
        fig.add_trace(go.Bar(
            x=labels, y=curr_vals,
            name="Текущий ADTV",
            marker_color="rgba(240,180,41,0.6)",
            hovertemplate="%{y:,.1f} млрд ₽<extra>Текущий</extra>",
        ))

    fig.add_trace(go.Bar(
        x=labels, y=pred_vals,
        name="Прогноз ADTV",
        marker_color="rgba(0,212,255,0.7)",
        hovertemplate="%{y:,.1f} млрд ₽<extra>Прогноз</extra>",
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        height=350,
        barmode="group",
        yaxis_title="ADTV, млрд ₽",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            bgcolor="rgba(17,24,39,0.7)",
            font=dict(size=10),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Backtest diagnostics by horizon
# ---------------------------------------------------------------------------

def diagnostics_by_horizon_chart(
    per_horizon: dict[int, dict[str, float]],
) -> go.Figure:
    """Bar chart: RMSE by forecast horizon."""
    if not per_horizon:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, title="Нет данных бэктеста")
        return fig

    horizons = sorted(per_horizon.keys())
    labels_map = {1: "1д", 5: "1нед", 22: "1мес", 63: "3мес", 126: "6мес"}
    labels = [labels_map.get(h, f"{h}д") for h in horizons]
    rmse_vals = [per_horizon[h].get("rmse", 0) for h in horizons]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=rmse_vals,
        marker_color=[
            "#51cf66" if v < 5 else "#f0b429" if v < 10 else "#ff6b6b"
            for v in rmse_vals
        ],
        text=[f"{v:.1f}%" for v in rmse_vals],
        textposition="outside",
        textfont=dict(size=11, color="#e5e7eb"),
        hovertemplate="%{y:.2f}%<extra>RMSE</extra>",
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        height=300,
        yaxis_title="RMSE, %",
        xaxis_title="Горизонт прогноза",
    )
    return fig


# ---------------------------------------------------------------------------
# 4. HAR-RV coefficients chart
# ---------------------------------------------------------------------------

def har_coefficients_chart(coefficients: dict[str, float]) -> go.Figure:
    """Horizontal bar chart of HAR-RV model coefficients."""
    labels_map = {
        "const": "Константа (β₀)",
        "rv_d": "Дневная RV (β_d)",
        "rv_w": "Недельная RV (β_w)",
        "rv_m": "Месячная RV (β_m)",
    }

    names = [labels_map.get(k, k) for k in coefficients]
    values = list(coefficients.values())
    colors = ["#74c0fc" if v >= 0 else "#ff6b6b" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition="outside",
        textfont=dict(size=11, color="#e5e7eb"),
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        height=220,
        xaxis_title="Значение коэффициента",
    )
    return fig
