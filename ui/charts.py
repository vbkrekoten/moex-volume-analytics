"""Reusable Plotly chart builders for the dashboard."""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from ui.sidebar import CLASS_MAP_REVERSE, FACTORS

# Color palette matching the dark theme
COLORS = [
    "#f0b429",  # gold
    "#00d4ff",  # cyan
    "#ff6b6b",  # red
    "#51cf66",  # green
    "#cc5de8",  # purple
    "#ffa94d",  # orange
    "#74c0fc",  # light blue
    "#f06595",  # pink
    "#a9e34b",  # lime
    "#e599f7",  # lavender
]

DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0a0e17",
    plot_bgcolor="#111827",
    font=dict(color="#e5e7eb"),
    legend=dict(bgcolor="rgba(17,24,39,0.8)"),
    margin=dict(l=60, r=40, t=40, b=40),
)


def stacked_area_chart(weekly_vol: pd.DataFrame, pct_mode: bool = False) -> go.Figure:
    """Stacked area chart of weekly volumes by instrument class."""
    if weekly_vol.empty:
        return go.Figure().update_layout(**DARK_LAYOUT, title="Нет данных")

    pivot = weekly_vol.pivot_table(
        index="week_start", columns="instrument_class",
        values="total_value", fill_value=0,
    )
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()

    # Rename columns to Russian
    pivot.columns = [CLASS_MAP_REVERSE.get(c, c) for c in pivot.columns]

    if pct_mode:
        row_sums = pivot.sum(axis=1)
        pivot = pivot.div(row_sums, axis=0) * 100

    fig = go.Figure()
    for i, col in enumerate(pivot.columns):
        fig.add_trace(go.Scatter(
            x=pivot.index, y=pivot[col],
            name=col,
            stackgroup="one",
            fillcolor=COLORS[i % len(COLORS)] + "80",
            line=dict(color=COLORS[i % len(COLORS)], width=0.5),
        ))

    y_title = "Доля, %" if pct_mode else "Объём, млн руб."
    fig.update_layout(
        **DARK_LAYOUT,
        yaxis_title=y_title,
        xaxis_title="",
        hovermode="x unified",
    )
    return fig


def dual_axis_chart(
    vol_series: pd.Series,
    factor_series: pd.Series,
    vol_name: str,
    factor_name: str,
) -> go.Figure:
    """Dual-axis line chart: volume (left) vs factor (right)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=vol_series.index, y=vol_series.values,
            name=vol_name, line=dict(color=COLORS[0], width=2),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=factor_series.index, y=factor_series.values,
            name=FACTORS.get(factor_name, factor_name),
            line=dict(color=COLORS[1], width=2),
        ),
        secondary_y=True,
    )

    fig.update_layout(**DARK_LAYOUT, hovermode="x unified")
    fig.update_yaxes(title_text=f"Объём ({vol_name})", secondary_y=False)
    fig.update_yaxes(
        title_text=FACTORS.get(factor_name, factor_name), secondary_y=True,
    )
    return fig


def correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Heatmap of correlation matrix."""
    if corr_matrix.empty:
        return go.Figure().update_layout(**DARK_LAYOUT, title="Нет данных")

    # Rename rows to Russian
    index_labels = [CLASS_MAP_REVERSE.get(c, c) for c in corr_matrix.index]
    col_labels = [FACTORS.get(c, c) for c in corr_matrix.columns]

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values.astype(float),
        x=col_labels,
        y=index_labels,
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr_matrix.values.astype(float).round(2),
        texttemplate="%{text:.2f}",
        textfont=dict(size=11),
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        height=max(300, len(index_labels) * 60 + 100),
    )
    return fig


def scatter_with_regression(
    x: pd.Series, y: pd.Series,
    x_label: str, y_label: str,
) -> go.Figure:
    """Scatter plot with OLS regression line."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(color=COLORS[0], size=5, opacity=0.6),
        name="Данные",
    ))

    # Regression line
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() > 5:
        xf = x[mask].values.astype(float)
        yf = y[mask].values.astype(float)
        coeffs = np.polyfit(xf, yf, 1)
        x_line = np.linspace(xf.min(), xf.max(), 100)
        y_line = np.polyval(coeffs, x_line)
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines",
            line=dict(color=COLORS[1], dash="dash", width=2),
            name=f"R = {np.corrcoef(xf, yf)[0,1]:.3f}",
        ))

    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title=x_label,
        yaxis_title=y_label,
    )
    return fig


def rolling_corr_chart(rolling_df: pd.DataFrame, label: str) -> go.Figure:
    """Line chart of rolling correlation over time."""
    if rolling_df.empty:
        return go.Figure().update_layout(**DARK_LAYOUT, title="Нет данных")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_df["week_start"],
        y=rolling_df["correlation"],
        mode="lines",
        line=dict(color=COLORS[0], width=2),
        name=label,
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#555")

    fig.update_layout(
        **DARK_LAYOUT,
        yaxis_title="Корреляция",
        yaxis_range=[-1, 1],
    )
    return fig


def coefficient_bar_chart(coefficients: dict, pvalues: dict) -> go.Figure:
    """Horizontal bar chart of standardized regression coefficients."""
    if not coefficients:
        return go.Figure().update_layout(**DARK_LAYOUT, title="Нет данных")

    names = [FACTORS.get(k, k) for k in coefficients.keys()]
    values = list(coefficients.values())
    pvs = [pvalues.get(k, 1.0) for k in coefficients.keys()]

    # Color by significance
    colors = [COLORS[0] if p < 0.05 else "#555" for p in pvs]

    fig = go.Figure(go.Bar(
        x=values, y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" + ("*" if p < 0.05 else "") for v, p in zip(values, pvs)],
        textposition="outside",
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title="Стандартизированный коэффициент",
        height=max(300, len(names) * 40 + 100),
    )
    return fig
