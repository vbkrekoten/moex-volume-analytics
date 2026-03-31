"""Plotly chart builders for the household savings section."""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd

from ui.charts import DARK_LAYOUT

# Component display config: indicator → (Russian label, color)
COMPONENTS = {
    "HH_TERM_DEPOSITS": ("Срочные вклады", "#f0b429"),
    "HH_SAVINGS_ACCOUNTS": ("Накопительные счета", "#74c0fc"),
    "HH_CURRENT_ACCOUNTS": ("Текущие счета", "#51cf66"),
    "HH_FX_DEPOSITS": ("Валютные депозиты", "#cc5de8"),
    "HH_ESCROW": ("Эскроу", "#ff6b6b"),
    "HH_CASH_M0": ("Наличные (M0)", "#ffa94d"),
}

COMPONENT_ORDER = list(COMPONENTS.keys())


def savings_stacked_area(df: pd.DataFrame) -> go.Figure:
    """Stacked area chart of savings components over time.

    Args:
        df: pivoted DataFrame — index=period_date, columns=indicator names, values=billions RUB.
    """
    fig = go.Figure()

    for ind in COMPONENT_ORDER:
        if ind not in df.columns:
            continue
        label, color = COMPONENTS[ind]
        fig.add_trace(go.Scatter(
            x=df.index, y=df[ind] / 1000,  # billions → trillions
            name=label,
            mode="lines",
            stackgroup="one",
            line=dict(width=0.5, color=color),
            fillcolor=color.replace(")", ",0.6)").replace("rgb", "rgba") if "rgb" in color else color + "99",
            hovertemplate=f"{label}: %{{y:,.1f}} трлн ₽<extra></extra>",
        ))

    layout = {**DARK_LAYOUT}
    layout["height"] = 450
    layout["hovermode"] = "x unified"
    layout["yaxis"] = dict(
        title="трлн ₽",
        gridcolor="rgba(255,255,255,0.04)",
    )
    layout["legend"] = dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="left", x=0,
        bgcolor="rgba(17,24,39,0.7)",
        font=dict(size=10),
    )
    fig.update_layout(**layout)
    return fig


def savings_donut(latest: dict[str, float]) -> go.Figure:
    """Donut chart of current savings structure.

    Args:
        latest: {indicator: value_in_billions} for the most recent month.
    """
    labels = []
    values = []
    colors = []
    for ind in COMPONENT_ORDER:
        val = latest.get(ind, 0)
        if val <= 0:
            continue
        lbl, color = COMPONENTS[ind]
        labels.append(lbl)
        values.append(val / 1000)  # billions → trillions
        colors.append(color)

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker=dict(colors=colors),
        textinfo="label+percent",
        textfont=dict(size=11, color="#e5e7eb"),
        hovertemplate="%{label}: %{value:,.1f} трлн ₽ (%{percent})<extra></extra>",
    ))

    total = sum(values)
    layout = {**DARK_LAYOUT}
    layout["height"] = 400
    layout["showlegend"] = False
    layout["annotations"] = [dict(
        text=f"<b>{total:,.1f}</b><br>трлн ₽",
        x=0.5, y=0.5, font=dict(size=16, color="#f0b429"),
        showarrow=False,
    )]
    fig.update_layout(**layout)
    return fig


def savings_component_lines(df: pd.DataFrame) -> go.Figure:
    """Individual line chart for each savings component.

    Args:
        df: pivoted DataFrame — index=period_date, columns=indicator names.
    """
    fig = go.Figure()

    for ind in COMPONENT_ORDER:
        if ind not in df.columns:
            continue
        label, color = COMPONENTS[ind]
        fig.add_trace(go.Scatter(
            x=df.index, y=df[ind] / 1000,
            name=label,
            line=dict(color=color, width=2),
            hovertemplate=f"{label}: %{{y:,.1f}} трлн ₽<extra></extra>",
        ))

    layout = {**DARK_LAYOUT}
    layout["height"] = 400
    layout["hovermode"] = "x unified"
    layout["yaxis"] = dict(
        title="трлн ₽",
        gridcolor="rgba(255,255,255,0.04)",
    )
    layout["legend"] = dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="left", x=0,
        bgcolor="rgba(17,24,39,0.7)",
        font=dict(size=10),
    )
    fig.update_layout(**layout)
    return fig
