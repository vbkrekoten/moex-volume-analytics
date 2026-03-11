"""Reusable Plotly chart builders for the dashboard."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from ui.sidebar import CLASS_MAP_REVERSE, FACTORS

# Color palette
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

# Glassmorphism-compatible dark layout
DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(10, 14, 23, 0.0)",
    plot_bgcolor="rgba(17, 24, 39, 0.35)",
    font=dict(color="#e5e7eb", family="Inter, -apple-system, sans-serif", size=12),
    legend=dict(
        bgcolor="rgba(17,24,39,0.6)",
        bordercolor="rgba(240,180,41,0.1)",
        borderwidth=1,
        font=dict(size=11),
    ),
    margin=dict(l=50, r=30, t=45, b=35),
    hoverlabel=dict(
        bgcolor="rgba(17,24,39,0.92)",
        bordercolor="rgba(240,180,41,0.25)",
        font_color="#e5e7eb",
        font_size=12,
    ),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.06)",
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.06)",
    ),
)


def _hex_to_rgba(hex_color: str, alpha: float = 0.5) -> str:
    """Convert hex color (#RRGGBB) to rgba() string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _empty_fig(msg: str = "Нет данных") -> go.Figure:
    """Return an empty figure with a message."""
    fig = go.Figure()
    fig.update_layout(**DARK_LAYOUT, title=msg)
    return fig


# ---------------------------------------------------------------------------
# 1. Stacked area chart (turnovers overview)
# ---------------------------------------------------------------------------

def stacked_area_chart(
    daily_vol: pd.DataFrame,
    pct_mode: bool = False,
    date_col: str = "trade_date",
    value_col: str = "value_rub",
    class_col: str = "instrument_class",
) -> go.Figure:
    """Stacked area chart of turnovers by instrument class."""
    if daily_vol.empty:
        return _empty_fig()

    pivot = daily_vol.pivot_table(
        index=date_col, columns=class_col,
        values=value_col, aggfunc="sum", fill_value=0,
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
        color = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(
            x=pivot.index, y=pivot[col],
            name=col,
            stackgroup="one",
            line=dict(color=color, width=0.5),
            fillcolor=_hex_to_rgba(color, 0.55),
            hovertemplate="%{y:,.0f}<extra>%{fullData.name}</extra>",
        ))

    y_title = "Доля, %" if pct_mode else "Оборот, млн руб."
    fig.update_layout(
        **DARK_LAYOUT,
        yaxis_title=y_title,
        xaxis_title="",
        hovermode="x unified",
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Combined turnover + factor overlay chart
# ---------------------------------------------------------------------------

def combined_turnover_factor_chart(
    daily_vol: pd.DataFrame,
    daily_factors: pd.DataFrame,
    selected_factors: list[str],
    date_col: str = "trade_date",
    value_col: str = "value_rub",
    class_col: str = "instrument_class",
) -> go.Figure:
    """Stacked area of turnovers with factor lines on independent Y-axes."""
    has_factors = bool(selected_factors) and not daily_factors.empty
    n_factors = len(selected_factors) if has_factors else 0

    fig = go.Figure()

    # --- Stacked area: turnovers (primary y-axis) ---
    if not daily_vol.empty:
        pivot = daily_vol.pivot_table(
            index=date_col, columns=class_col,
            values=value_col, aggfunc="sum", fill_value=0,
        )
        pivot.index = pd.to_datetime(pivot.index)
        pivot = pivot.sort_index()
        pivot.columns = [CLASS_MAP_REVERSE.get(c, c) for c in pivot.columns]

        for i, col in enumerate(pivot.columns):
            color = COLORS[i % len(COLORS)]
            fig.add_trace(go.Scatter(
                x=pivot.index, y=pivot[col],
                name=col,
                stackgroup="one",
                line=dict(color=color, width=0.5),
                fillcolor=_hex_to_rgba(color, 0.4),
                hovertemplate="%{y:,.0f}<extra>%{fullData.name}</extra>",
                legendgroup="turnovers",
                legendgrouptitle_text="Обороты",
                yaxis="y",
            ))

    # --- Factor lines on independent Y-axes ---
    factor_colors = ["#00d4ff", "#ff6b6b", "#51cf66", "#cc5de8", "#ffa94d"]
    # Shrink x-axis domain to make room for factor axes
    # Each side gets axes; allocate 0.06 per axis on each side
    n_right = (n_factors + 1) // 2  # axes on right
    n_left = n_factors // 2         # axes on left
    domain_left = 0.06 * n_left
    domain_right = 1 - 0.06 * n_right

    if has_factors:
        for i, fname in enumerate(selected_factors):
            fdata = daily_factors[daily_factors["factor_name"] == fname].copy()
            if fdata.empty:
                continue
            fdata[date_col] = pd.to_datetime(fdata[date_col])
            fdata = fdata.sort_values(date_col)
            color = factor_colors[i % len(factor_colors)]
            label = FACTORS.get(fname, fname)
            axis_idx = i + 2  # y2, y3, y4...
            yaxis_name = f"y{axis_idx}"

            fig.add_trace(go.Scatter(
                x=fdata[date_col], y=fdata["value"],
                name=label,
                line=dict(color=color, width=2, dash="dot"),
                hovertemplate="%{y:.2f}<extra>%{fullData.name}</extra>",
                legendgroup="factors",
                legendgrouptitle_text="Факторы",
                yaxis=yaxis_name,
            ))

    # --- Build layout ---
    layout = {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(10, 14, 23, 0.0)",
        "plot_bgcolor": "rgba(17, 24, 39, 0.35)",
        "font": dict(color="#e5e7eb", family="Inter, -apple-system, sans-serif", size=12),
        "hovermode": "x unified",
        "height": 480,
        "hoverlabel": dict(
            bgcolor="rgba(17,24,39,0.92)",
            bordercolor="rgba(240,180,41,0.25)",
            font_color="#e5e7eb",
            font_size=12,
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
        "margin": dict(l=50, r=50, t=45, b=35),
        # Primary y-axis: turnovers
        "yaxis": dict(
            title="Оборот, млн руб.",
            titlefont=dict(color="#f0b429"),
            tickfont=dict(color="#f0b429"),
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.06)",
            side="left",
        ),
        # X-axis with domain shrunk for factor axes
        "xaxis": dict(
            domain=[domain_left, domain_right] if n_factors > 0 else [0, 1],
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.06)",
        ),
    }

    # Add independent Y-axes for each factor
    if has_factors:
        for i in range(n_factors):
            axis_idx = i + 2
            color = factor_colors[i % len(factor_colors)]
            label = FACTORS.get(selected_factors[i], selected_factors[i])

            # Alternate sides: even index → right, odd → left
            if i % 2 == 0:
                # Right side
                side = "right"
                right_pos = i // 2  # 0, 1, 2...
                position = domain_right + 0.06 * right_pos
            else:
                # Left side
                side = "left"
                left_pos = (i - 1) // 2  # 0, 1, 2...
                position = domain_left - 0.06 * left_pos - 0.06

            axis_key = f"yaxis{axis_idx}"
            layout[axis_key] = dict(
                title=label,
                titlefont=dict(color=color, size=11),
                tickfont=dict(color=color, size=10),
                overlaying="y",
                side=side,
                anchor="free",
                position=max(0, min(1, position)),
                showgrid=False,
            )

    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# 3. Correlation heatmap
# ---------------------------------------------------------------------------

def correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Heatmap of correlation matrix with improved styling."""
    if corr_matrix.empty:
        return _empty_fig()

    index_labels = [CLASS_MAP_REVERSE.get(c, c) for c in corr_matrix.index]
    col_labels = [FACTORS.get(c, c) for c in corr_matrix.columns]

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values.astype(float),
        x=col_labels,
        y=index_labels,
        colorscale=[
            [0, "#ff6b6b"],
            [0.5, "#1f2937"],
            [1, "#51cf66"],
        ],
        zmid=0, zmin=-1, zmax=1,
        text=corr_matrix.values.astype(float).round(2),
        texttemplate="%{text:.2f}",
        textfont=dict(size=11, color="#e5e7eb"),
        colorbar=dict(
            title="Корр.",
            titleside="right",
            tickvals=[-1, -0.5, 0, 0.5, 1],
        ),
    ))

    layout = {
        **DARK_LAYOUT,
        "height": max(280, len(index_labels) * 55 + 80),
        "xaxis": {**DARK_LAYOUT.get("xaxis", {}), "tickangle": -45, "tickfont": dict(size=10)},
    }
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# 4. Scatter with regression line
# ---------------------------------------------------------------------------

def scatter_with_regression(
    x: pd.Series, y: pd.Series,
    x_label: str, y_label: str,
) -> go.Figure:
    """Scatter plot with OLS regression line."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(color=COLORS[0], size=4, opacity=0.5),
        name="Данные",
    ))

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
        height=350,
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Rolling correlation chart
# ---------------------------------------------------------------------------

def rolling_corr_chart(rolling_df: pd.DataFrame, label: str) -> go.Figure:
    """Line chart of rolling correlation over time."""
    if rolling_df.empty:
        return _empty_fig()

    date_col = "date" if "date" in rolling_df.columns else "week_start"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_df[date_col],
        y=rolling_df["correlation"],
        mode="lines",
        line=dict(color=COLORS[0], width=2),
        fill="tozeroy",
        fillcolor=_hex_to_rgba(COLORS[0], 0.1),
        name=label,
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.15)")

    fig.update_layout(
        **DARK_LAYOUT,
        yaxis_title="Корреляция",
        yaxis_range=[-1, 1],
        height=300,
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Coefficient bar chart (regression)
# ---------------------------------------------------------------------------

def coefficient_bar_chart(coefficients: dict, pvalues: dict) -> go.Figure:
    """Horizontal bar chart of standardized regression coefficients."""
    if not coefficients:
        return _empty_fig()

    # Sort by absolute value
    sorted_keys = sorted(coefficients.keys(), key=lambda k: abs(coefficients[k]))
    names = [FACTORS.get(k, k) for k in sorted_keys]
    values = [coefficients[k] for k in sorted_keys]
    pvs = [pvalues.get(k, 1.0) for k in sorted_keys]

    colors = []
    for p in pvs:
        if p < 0.01:
            colors.append("#51cf66")
        elif p < 0.05:
            colors.append("#f0b429")
        elif p < 0.10:
            colors.append("#ffa94d")
        else:
            colors.append("#4b5563")

    stars = []
    for p in pvs:
        if p < 0.001:
            stars.append("***")
        elif p < 0.01:
            stars.append("**")
        elif p < 0.05:
            stars.append("*")
        else:
            stars.append("")

    fig = go.Figure(go.Bar(
        x=values, y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f} {s}" for v, s in zip(values, stars)],
        textposition="outside",
        textfont=dict(size=11),
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title="Стандартизированный коэффициент",
        height=max(250, len(names) * 35 + 80),
    )
    return fig


# ---------------------------------------------------------------------------
# 7. Residuals chart
# ---------------------------------------------------------------------------

def residuals_chart(residuals: pd.Series) -> go.Figure:
    """Residuals line chart with zero reference."""
    if residuals.empty:
        return _empty_fig()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=residuals.index,
        y=residuals.values,
        mode="lines",
        line=dict(color=COLORS[0], width=1.5),
        fill="tozeroy",
        fillcolor=_hex_to_rgba(COLORS[0], 0.08),
        name="Остатки",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.15)")

    fig.update_layout(
        **DARK_LAYOUT,
        yaxis_title="Остаток",
        height=250,
    )
    return fig


# ---------------------------------------------------------------------------
# 8. Rolling R2 chart
# ---------------------------------------------------------------------------

def rolling_r2_chart(rolling_df: pd.DataFrame, label: str = "") -> go.Figure:
    """Line chart showing rolling R2 over time."""
    if rolling_df.empty:
        return _empty_fig()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_df["date"],
        y=rolling_df["r2"],
        mode="lines",
        line=dict(color="#00d4ff", width=2),
        fill="tozeroy",
        fillcolor=_hex_to_rgba("#00d4ff", 0.08),
        name=label or "R2",
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        yaxis_title="R2",
        yaxis_range=[0, 1],
        height=250,
    )
    return fig


# ---------------------------------------------------------------------------
# 9. Dual-axis chart (single volume vs single factor)
# ---------------------------------------------------------------------------

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

    fig.update_layout(**DARK_LAYOUT, hovermode="x unified", height=350)
    fig.update_yaxes(title_text=f"Оборот ({vol_name})", secondary_y=False)
    fig.update_yaxes(
        title_text=FACTORS.get(factor_name, factor_name), secondary_y=True,
    )
    return fig
