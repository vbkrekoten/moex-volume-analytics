"""Plotly chart builders for the investment ideas impact section."""

from __future__ import annotations

import plotly.graph_objects as go
import numpy as np
import pandas as pd

from ui.charts import DARK_LAYOUT, COLORS


def event_study_bar_chart(aav_by_day: dict) -> go.Figure:
    """Bar chart of Average Abnormal Volume by day offset with 95% CI.

    Args:
        aav_by_day: {day_offset: {mean_av_ratio, ci_lower, ci_upper, n}}
    """
    fig = go.Figure()
    if not aav_by_day:
        fig.update_layout(**DARK_LAYOUT, title="Нет данных")
        return fig

    days = sorted(aav_by_day.keys())
    means = [aav_by_day[d]["mean_av_ratio"] for d in days]
    ci_lo = [aav_by_day[d]["ci_lower"] for d in days]
    ci_hi = [aav_by_day[d]["ci_upper"] for d in days]
    labels = [f"d{d:+d}" for d in days]

    # Color bars: event day (d=0) in gold, others in cyan
    colors = ["#f0b429" if d == 0 else "#74c0fc" for d in days]

    fig.add_trace(go.Bar(
        x=labels,
        y=means,
        marker_color=colors,
        text=[f"{v:.2f}x" for v in means],
        textposition="outside",
        textfont=dict(size=11, color="#e5e7eb"),
        hovertemplate="День %{x}<br>Сред. AV: %{y:.3f}x<extra></extra>",
        error_y=dict(
            type="data",
            symmetric=False,
            array=[ci_hi[i] - means[i] for i in range(len(means))],
            arrayminus=[means[i] - ci_lo[i] for i in range(len(means))],
            color="rgba(255,255,255,0.3)",
            thickness=1.5,
        ),
    ))

    # Reference line at 1.0 (no abnormal volume)
    fig.add_hline(
        y=1.0, line_dash="dash", line_color="rgba(255,255,255,0.3)",
        annotation_text="Норма (1.0x)",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color="#6b7280"),
    )

    layout = {**DARK_LAYOUT}
    layout["legend"] = dict(
        bgcolor="rgba(17,24,39,0.7)", font=dict(size=10),
    )
    fig.update_layout(
        **layout,
        height=380,
        yaxis_title="Средний AV Ratio",
        xaxis_title="День относительно публикации",
    )
    return fig


def impact_distribution_chart(peak_ratios: list[float]) -> go.Figure:
    """Histogram of peak AV ratios across all ideas."""
    fig = go.Figure()
    if not peak_ratios:
        fig.update_layout(**DARK_LAYOUT, title="Нет данных")
        return fig

    arr = np.array(peak_ratios)
    bins = [0, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, max(10.0, arr.max() + 1)]
    bin_labels = ["<1x", "1-1.25x", "1.25-1.5x", "1.5-2x", "2-3x", "3-5x", "5x+"]
    counts = np.histogram(arr, bins=bins)[0]

    bar_colors = [
        "#6b7280", "#74c0fc", "#51cf66", "#f0b429",
        "#ffa94d", "#ff6b6b", "#cc5de8",
    ]

    fig.add_trace(go.Bar(
        x=bin_labels,
        y=counts,
        marker_color=bar_colors[:len(counts)],
        text=[str(int(c)) for c in counts],
        textposition="outside",
        textfont=dict(size=11, color="#e5e7eb"),
        hovertemplate="Диапазон: %{x}<br>Идей: %{y}<extra></extra>",
    ))

    layout = {**DARK_LAYOUT}
    layout["margin"] = dict(l=50, r=30, t=30, b=55)
    fig.update_layout(
        **layout,
        height=380,
        yaxis_title="Количество идей",
        xaxis_title="Пиковый AV Ratio",
        yaxis_range=[0, max(counts) * 1.2],
    )
    return fig


def significance_pie_chart(n_significant: int, n_total: int) -> go.Figure:
    """Pie chart: significant vs not significant ideas."""
    fig = go.Figure()
    n_not = n_total - n_significant
    fig.add_trace(go.Pie(
        labels=["Значимые (p<0.05)", "Незначимые"],
        values=[n_significant, n_not],
        marker=dict(colors=["#51cf66", "#6b7280"]),
        textinfo="label+percent",
        textfont=dict(size=12),
        hole=0.4,
        hovertemplate="%{label}: %{value} идей (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        height=320,
        showlegend=False,
    )
    return fig


def idea_detail_chart(
    volume_series: dict,
    est_mean: float,
    av_ratio_by_day: dict,
    ticker: str,
    idea_date: str,
) -> go.Figure:
    """Detailed chart for a single idea: volume timeline with zones.

    Args:
        volume_series: {dates: [...], values: [...]} — full history around event
        est_mean: estimation window mean volume
        av_ratio_by_day: {day_offsets: [...], values: [...]}
        ticker: ticker symbol
        idea_date: event date string
    """
    fig = go.Figure()

    dates = volume_series.get("dates", [])
    values = volume_series.get("values", [])

    if not dates:
        fig.update_layout(**DARK_LAYOUT, title="Нет данных")
        return fig

    # Volume bars
    fig.add_trace(go.Bar(
        x=dates,
        y=[v / 1e9 for v in values],
        marker_color="rgba(116,192,252,0.5)",
        name="Оборот",
        hovertemplate="%{x}<br>%{y:.2f} млрд ₽<extra></extra>",
    ))

    # Estimation mean line
    fig.add_hline(
        y=est_mean / 1e9, line_dash="dash", line_color="#f0b429",
        annotation_text=f"Норма: {est_mean/1e9:.1f} млрд ₽",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#f0b429"),
    )

    # Event date vertical line (use shape+annotation to avoid Plotly sum() bug)
    idea_ts = pd.Timestamp(idea_date)
    fig.add_shape(
        type="line", x0=idea_ts, x1=idea_ts, y0=0, y1=1,
        yref="paper", line=dict(dash="dash", color="#ff6b6b", width=1.5),
    )
    fig.add_annotation(
        x=idea_ts, y=1, yref="paper",
        text=f"{ticker} — идея",
        showarrow=False, font=dict(size=10, color="#ff6b6b"),
        xanchor="left", yanchor="bottom",
    )

    # Annotate AV ratios on event window days
    offsets = av_ratio_by_day.get("day_offsets", [])
    av_values = av_ratio_by_day.get("values", [])
    vol_by_day = volume_series.get("event_dates", [])
    vol_by_day_vals = volume_series.get("event_values", [])

    for i, (off, av) in enumerate(zip(offsets, av_values)):
        if i < len(vol_by_day) and av > 1.1:
            fig.add_annotation(
                x=vol_by_day[i],
                y=vol_by_day_vals[i] / 1e9 if i < len(vol_by_day_vals) else 0,
                text=f"{av:.1f}x",
                showarrow=True,
                arrowhead=2,
                arrowsize=0.5,
                arrowcolor="#f0b429",
                font=dict(size=9, color="#f0b429"),
                ax=0, ay=-25,
            )

    layout = {**DARK_LAYOUT}
    layout["legend"] = dict(
        bgcolor="rgba(17,24,39,0.7)", font=dict(size=10),
        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
    )
    fig.update_layout(
        **layout,
        height=400,
        yaxis_title="Оборот, млрд ₽",
        xaxis_title="Дата",
        title=f"{ticker} — оборот вокруг инвестидеи ({idea_date})",
    )
    return fig


def timeline_scatter(ideas_data: list[dict]) -> go.Figure:
    """Scatter plot: all ideas on timeline.

    Args:
        ideas_data: list of dicts with idea_date, peak_av_ratio, cav, ticker, is_significant
    """
    fig = go.Figure()
    if not ideas_data:
        fig.update_layout(**DARK_LAYOUT, title="Нет данных")
        return fig

    sig = [d for d in ideas_data if d.get("is_significant")]
    nsig = [d for d in ideas_data if not d.get("is_significant")]

    for group, label, color, symbol in [
        (sig, "Значимые", "#51cf66", "circle"),
        (nsig, "Незначимые", "rgba(107,114,128,0.5)", "circle-open"),
    ]:
        if not group:
            continue
        fig.add_trace(go.Scatter(
            x=[d["idea_date"] for d in group],
            y=[d["peak_av_ratio"] for d in group],
            mode="markers",
            name=label,
            marker=dict(
                color=color,
                size=[max(5, min(20, abs(d.get("cav", 1)) * 5)) for d in group],
                symbol=symbol,
                line=dict(width=1, color="rgba(255,255,255,0.3)"),
            ),
            text=[d["ticker"] for d in group],
            customdata=[[d.get("source", "")] for d in group],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Источник: %{customdata[0]}<br>"
                "Дата: %{x}<br>"
                "Пик AV: %{y:.2f}x<br>"
                "<extra></extra>"
            ),
        ))

    fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(255,255,255,0.2)")

    layout = {**DARK_LAYOUT}
    layout["legend"] = dict(
        bgcolor="rgba(17,24,39,0.7)", font=dict(size=10),
        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
    )
    fig.update_layout(
        **layout,
        height=400,
        yaxis_title="Пиковый AV Ratio",
        xaxis_title="Дата публикации идеи",
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Source (analyst) comparison bar chart
# ---------------------------------------------------------------------------

def source_comparison_chart(source_summary: list[dict]) -> go.Figure:
    """Horizontal bar chart comparing sources (analysts) by avg CAV.

    Args:
        source_summary: list of dicts with source, n_ideas, avg_cav, avg_peak_av, pct_significant
    """
    fig = go.Figure()
    if not source_summary:
        fig.update_layout(**DARK_LAYOUT, title="Нет данных")
        return fig

    # Sort by n_ideas descending, take top 20
    data = sorted(source_summary, key=lambda x: x["n_ideas"], reverse=True)[:20]
    sources = [d["source"] for d in data]
    cavs = [d["avg_cav"] for d in data]
    n_ideas = [d["n_ideas"] for d in data]
    pct_sig = [d["pct_significant"] for d in data]

    # Color by avg_cav: positive = green, negative = red
    colors = ["#51cf66" if c > 0 else "#ff6b6b" for c in cavs]

    fig.add_trace(go.Bar(
        y=sources, x=cavs, orientation="h",
        name="Сред. CAV",
        marker_color=colors,
        text=[f"{c:+.2f} ({n} идей, {p:.0f}% знач.)" for c, n, p in zip(cavs, n_ideas, pct_sig)],
        textposition="auto",
        textfont=dict(size=10),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Сред. CAV: %{x:+.3f}<br>"
            "<extra></extra>"
        ),
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")

    layout = {**DARK_LAYOUT}
    layout.pop("legend", None)
    fig.update_layout(
        **layout,
        height=max(300, len(data) * 28),
        xaxis_title="Средний CAV",
        title="Топ-20 источников по количеству идей",
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    return fig
