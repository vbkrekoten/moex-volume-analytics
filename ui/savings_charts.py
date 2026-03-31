"""Plotly chart builders for the household savings section."""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd

from ui.charts import DARK_LAYOUT

# Main components for stacked area (top-level breakdown of HH_ASSETS_TOTAL)
MAIN_COMPONENTS = {
    "HH_CASH_TOTAL": ("Наличные", "#ffa94d"),
    "HH_DEPOSITS_TOTAL": ("Депозиты", "#f0b429"),
    "HH_BROKERAGE": ("Брокерские счета", "#74c0fc"),
    "HH_BONDS": ("Облигации", "#51cf66"),
    "HH_EQUITIES_TOTAL": ("Акции и участие в капитале", "#cc5de8"),
    "HH_INSURANCE_PENSION": ("Страховые/пенсионные", "#20c997"),
    "HH_ESCROW_CBR": ("Эскроу", "#ff6b6b"),
    "HH_LOANS_ISSUED": ("Займы выданные", "#868e96"),
    "HH_RECEIVABLES": ("Дебиторская задолженность", "#495057"),
}

# Deposit sub-components
DEPOSIT_COMPONENTS = {
    "HH_DEPOSITS_DEMAND": ("Переводные (текущие)", "#74c0fc"),
    "HH_DEPOSITS_TERM": ("Срочные (вкл. валютные)", "#f0b429"),
    "HH_DEPOSITS_NONRESIDENT": ("В банках-нерезидентах", "#cc5de8"),
}

# Equity sub-components
EQUITY_COMPONENTS = {
    "HH_EQUITIES_LISTED": ("Котируемые акции", "#51cf66"),
    "HH_EQUITIES_UNLISTED": ("Некотируемые акции", "#868e96"),
    "HH_FUNDS": ("Паи инвест. фондов", "#74c0fc"),
    "HH_EQUITIES_FOREIGN": ("Иностранные акции", "#cc5de8"),
}

# Cash sub-components
CASH_COMPONENTS = {
    "HH_CASH_RUB": ("Наличные рубли", "#f0b429"),
    "HH_CASH_FX": ("Наличная инвалюта", "#cc5de8"),
}

MAIN_ORDER = list(MAIN_COMPONENTS.keys())


def _build_stacked(df: pd.DataFrame, components: dict) -> go.Figure:
    """Generic stacked area chart builder."""
    fig = go.Figure()
    for ind, (label, color) in components.items():
        if ind not in df.columns:
            continue
        series = df[ind].dropna()
        if series.empty:
            continue
        fig.add_trace(go.Scatter(
            x=series.index, y=series / 1000,
            name=label, mode="lines", stackgroup="one",
            line=dict(width=0.5, color=color),
            hovertemplate=f"{label}: %{{y:,.1f}} трлн ₽<extra></extra>",
        ))

    layout = {**DARK_LAYOUT}
    layout["height"] = 450
    layout["hovermode"] = "x unified"
    layout["yaxis"] = dict(title="трлн ₽", gridcolor="rgba(255,255,255,0.04)")
    layout["legend"] = dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
        bgcolor="rgba(17,24,39,0.7)", font=dict(size=10),
    )
    fig.update_layout(**layout)
    return fig


def savings_stacked_area(df: pd.DataFrame) -> go.Figure:
    """Stacked area of main asset categories."""
    return _build_stacked(df, MAIN_COMPONENTS)


def deposits_stacked_area(df: pd.DataFrame) -> go.Figure:
    """Stacked area of deposit sub-components."""
    return _build_stacked(df, DEPOSIT_COMPONENTS)


def equities_stacked_area(df: pd.DataFrame) -> go.Figure:
    """Stacked area of equity sub-components."""
    return _build_stacked(df, EQUITY_COMPONENTS)


def cash_stacked_area(df: pd.DataFrame) -> go.Figure:
    """Stacked area of cash sub-components."""
    return _build_stacked(df, CASH_COMPONENTS)


def savings_donut(latest: dict[str, float]) -> go.Figure:
    """Donut chart of current asset structure."""
    labels, values, colors = [], [], []
    for ind in MAIN_ORDER:
        val = latest.get(ind, 0)
        if val <= 0:
            continue
        lbl, color = MAIN_COMPONENTS[ind]
        labels.append(lbl)
        values.append(val / 1000)
        colors.append(color)

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.5,
        marker=dict(colors=colors),
        textinfo="label+percent",
        textfont=dict(size=10, color="#e5e7eb"),
        hovertemplate="%{label}: %{value:,.1f} трлн ₽ (%{percent})<extra></extra>",
    ))

    total = sum(values)
    layout = {**DARK_LAYOUT}
    layout["height"] = 400
    layout["showlegend"] = False
    layout["annotations"] = [dict(
        text=f"<b>{total:,.1f}</b><br>трлн ₽",
        x=0.5, y=0.5, font=dict(size=16, color="#f0b429"), showarrow=False,
    )]
    fig.update_layout(**layout)
    return fig
