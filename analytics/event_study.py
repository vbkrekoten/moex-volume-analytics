"""Event study analytics for measuring abnormal volume around investment ideas.

Methodology (v2 — with placebo test):
1. Estimation window [-120, -6] trading days — baseline mean & std (weekends excluded).
2. Event window [0, +3] trading days — period of measurement.
3. Placebo test: run N=500 random pseudo-events for the same ticker to build
   an empirical distribution of CAV under the null hypothesis.
4. ΔCAV = CAV_event − median(CAV_placebo) — excess abnormal volume above baseline noise.
5. Significance: placebo-based percentile rank (p-value) instead of parametric 1.96σ.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_N_PLACEBO = 500  # number of random pseudo-events for placebo test


def compute_abnormal_volume(
    volume_series: pd.Series,
    event_date: str | pd.Timestamp,
    est_window: tuple[int, int] = (-120, -6),
    evt_window: tuple[int, int] = (0, 3),
    run_placebo: bool = True,
    n_placebo: int = _N_PLACEBO,
) -> dict | None:
    """Compute Abnormal Volume metrics for a single event.

    Args:
        volume_series: Series indexed by trade_date (datetime) with daily volumes.
            Must contain ONLY trading days (no weekends / zero-volume rows).
        event_date: the date of the investment idea publication.
        est_window: (start, end) offsets in trading days for estimation period.
        evt_window: (start, end) offsets in trading days for event period.
        run_placebo: if True, run placebo test for empirical p-value.
        n_placebo: number of random pseudo-events for placebo test.

    Returns:
        dict with AV ratio, NAV, CAV, ΔCAV, placebo percentile, day-by-day arrays.
        None if insufficient data.
    """
    event_date = pd.Timestamp(event_date)
    vs = volume_series.sort_index()

    # Filter out zero-volume days (weekends, holidays that slipped through)
    vs = vs[vs > 0]
    if vs.empty:
        return None

    # Find event date position in the trading calendar
    dates = vs.index
    if event_date not in dates:
        mask = dates >= event_date
        if not mask.any():
            return None
        event_date = dates[mask][0]

    event_idx = dates.get_loc(event_date)

    # Extract estimation window
    est_start = event_idx + est_window[0]
    est_end = event_idx + est_window[1]
    if est_start < 0:
        est_start = 0
    if est_end <= est_start:
        return None

    est_values = vs.iloc[est_start:est_end + 1].values

    if len(est_values) < 30:
        return None

    est_mean = float(np.mean(est_values))
    est_std = float(np.std(est_values, ddof=1))

    if est_mean <= 0 or est_std <= 0:
        return None

    # Compute event-window metrics
    result = _compute_window_metrics(vs, dates, event_idx, evt_window, est_mean, est_std)
    if result is None:
        return None

    day_offsets, av_ratios, navs, volumes, trade_dates = result
    cav = sum(r - 1.0 for r in av_ratios)

    # Peak metrics
    peak_idx = int(np.argmax(av_ratios))
    peak_av_ratio = av_ratios[peak_idx]
    peak_av_day = day_offsets[peak_idx]
    peak_nav = navs[peak_idx]

    # --- Placebo test ---
    placebo_cavs = []
    delta_cav = cav
    placebo_pvalue = None
    placebo_percentile = None

    if run_placebo and len(vs) > abs(est_window[0]) + evt_window[1] + 30:
        placebo_cavs = _run_placebo(
            vs, est_window, evt_window, event_idx, n_placebo,
        )
        if placebo_cavs:
            placebo_median = float(np.median(placebo_cavs))
            delta_cav = cav - placebo_median
            # Empirical p-value: fraction of placebo CAVs >= observed CAV
            n_ge = sum(1 for pc in placebo_cavs if pc >= cav)
            placebo_pvalue = round(n_ge / len(placebo_cavs), 4)
            placebo_percentile = round(
                (1.0 - n_ge / len(placebo_cavs)) * 100, 1
            )

    # Significance based on placebo test (top 5% of placebo distribution)
    is_significant = placebo_pvalue is not None and placebo_pvalue < 0.05

    return {
        "ticker": None,  # filled by caller
        "idea_date": str(event_date.date()),
        "est_mean_volume": round(est_mean, 2),
        "est_std_volume": round(est_std, 2),
        "est_days": len(est_values),
        "cav": round(cav, 4),
        "delta_cav": round(delta_cav, 4),
        "placebo_pvalue": placebo_pvalue,
        "placebo_percentile": placebo_percentile,
        "placebo_median_cav": round(float(np.median(placebo_cavs)), 4) if placebo_cavs else None,
        "peak_av_ratio": round(peak_av_ratio, 4),
        "peak_av_day": peak_av_day,
        "peak_nav": round(peak_nav, 4),
        "is_significant": is_significant,
        "av_ratio_by_day": {"day_offsets": day_offsets, "values": av_ratios},
        "nav_by_day": {"day_offsets": day_offsets, "values": navs},
        "volume_by_day": {
            "day_offsets": day_offsets,
            "values": volumes,
            "dates": trade_dates,
        },
    }


def _compute_window_metrics(
    vs: pd.Series,
    dates: pd.DatetimeIndex,
    event_idx: int,
    evt_window: tuple[int, int],
    est_mean: float,
    est_std: float,
) -> tuple | None:
    """Compute AV ratios and NAVs for an event window."""
    day_offsets = []
    av_ratios = []
    navs = []
    volumes = []
    trade_dates = []

    for offset in range(evt_window[0], evt_window[1] + 1):
        idx = event_idx + offset
        if idx < 0 or idx >= len(vs):
            continue
        vol = float(vs.iloc[idx])
        av_ratio = vol / est_mean
        nav = (vol - est_mean) / est_std

        day_offsets.append(offset)
        av_ratios.append(round(av_ratio, 4))
        navs.append(round(nav, 4))
        volumes.append(round(vol, 2))
        trade_dates.append(str(dates[idx].date()))

    if not av_ratios:
        return None
    return day_offsets, av_ratios, navs, volumes, trade_dates


def _run_placebo(
    vs: pd.Series,
    est_window: tuple[int, int],
    evt_window: tuple[int, int],
    real_event_idx: int,
    n_placebo: int,
) -> list[float]:
    """Run placebo (random pseudo-event) test.

    Pick n_placebo random positions in the time series (excluding the real event
    and its neighborhood), compute CAV for each, return the distribution.
    """
    n = len(vs)
    min_idx = abs(est_window[0]) + 5
    max_idx = n - evt_window[1] - 5
    if max_idx <= min_idx:
        return []

    # Exclude zone around real event to avoid contamination
    exclusion = set(range(
        max(0, real_event_idx + est_window[0]),
        min(n, real_event_idx + evt_window[1] + 5),
    ))

    candidates = [i for i in range(min_idx, max_idx) if i not in exclusion]
    if len(candidates) < 20:
        return []

    rng = np.random.default_rng(42)
    chosen = rng.choice(candidates, size=min(n_placebo, len(candidates)), replace=len(candidates) < n_placebo)

    placebo_cavs = []
    dates = vs.index

    for idx in chosen:
        # Estimation window for this pseudo-event
        e_start = idx + est_window[0]
        e_end = idx + est_window[1]
        if e_start < 0:
            continue
        est_vals = vs.iloc[e_start:e_end + 1].values
        est_vals = est_vals[est_vals > 0]
        if len(est_vals) < 30:
            continue

        e_mean = float(np.mean(est_vals))
        if e_mean <= 0:
            continue

        # Event window for pseudo-event
        cav = 0.0
        any_day = False
        for offset in range(evt_window[0], evt_window[1] + 1):
            i = idx + offset
            if i < 0 or i >= n:
                continue
            vol = float(vs.iloc[i])
            cav += (vol / e_mean) - 1.0
            any_day = True

        if any_day:
            placebo_cavs.append(cav)

    return placebo_cavs


def aggregate_event_study(results: list[dict]) -> dict:
    """Cross-sectional aggregation of event study results.

    Returns dict with aggregate metrics using ΔCAV (placebo-adjusted).
    """
    if not results:
        return {}

    # Collect day-by-day ratios across all events
    all_offsets = set()
    for r in results:
        for d in r["av_ratio_by_day"]["day_offsets"]:
            all_offsets.add(d)
    day_range = sorted(all_offsets)

    aav_by_day = {}
    for d in day_range:
        vals = []
        for r in results:
            offsets = r["av_ratio_by_day"]["day_offsets"]
            ratios = r["av_ratio_by_day"]["values"]
            if d in offsets:
                idx = offsets.index(d)
                vals.append(ratios[idx])
        if vals:
            arr = np.array(vals)
            mean_av = float(np.mean(arr))
            std_av = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0
            n = len(arr)
            ci95 = 1.96 * std_av / np.sqrt(n) if n > 1 else 0
            aav_by_day[d] = {
                "mean_av_ratio": round(mean_av, 4),
                "std": round(std_av, 4),
                "ci_lower": round(mean_av - ci95, 4),
                "ci_upper": round(mean_av + ci95, 4),
                "n": n,
            }

    n_total = len(results)
    n_significant = sum(1 for r in results if r.get("is_significant"))
    cavs = [r["cav"] for r in results]
    delta_cavs = [r.get("delta_cav", r["cav"]) for r in results]
    peak_ratios = [r["peak_av_ratio"] for r in results]
    pvalues = [r.get("placebo_pvalue") for r in results if r.get("placebo_pvalue") is not None]

    # Per-ticker summary
    ticker_data: dict[str, list] = {}
    for r in results:
        t = r["ticker"]
        if t not in ticker_data:
            ticker_data[t] = []
        ticker_data[t].append(r)

    ticker_summary = []
    for t, recs in ticker_data.items():
        ticker_summary.append({
            "ticker": t,
            "n_ideas": len(recs),
            "avg_delta_cav": round(float(np.mean([r.get("delta_cav", r["cav"]) for r in recs])), 4),
            "avg_cav": round(float(np.mean([r["cav"] for r in recs])), 4),
            "avg_peak_av": round(float(np.mean([r["peak_av_ratio"] for r in recs])), 4),
            "pct_significant": round(
                sum(1 for r in recs if r.get("is_significant")) / len(recs) * 100, 1
            ),
        })
    ticker_summary.sort(key=lambda x: x["avg_delta_cav"], reverse=True)

    return {
        "n_total": n_total,
        "n_significant": n_significant,
        "pct_significant": round(n_significant / n_total * 100, 1) if n_total else 0,
        "mean_cav": round(float(np.mean(cavs)), 4),
        "median_cav": round(float(np.median(cavs)), 4),
        "mean_delta_cav": round(float(np.mean(delta_cavs)), 4),
        "median_delta_cav": round(float(np.median(delta_cavs)), 4),
        "mean_peak_av_ratio": round(float(np.mean(peak_ratios)), 4),
        "median_peak_av_ratio": round(float(np.median(peak_ratios)), 4),
        "mean_placebo_pvalue": round(float(np.mean(pvalues)), 4) if pvalues else None,
        "aav_by_day": aav_by_day,
        "ticker_summary": ticker_summary,
        "source_summary": _build_source_summary(results),
    }


def _build_source_summary(results: list[dict]) -> list[dict]:
    """Group results by source (analyst) and compute per-source metrics."""
    source_data: dict[str, list] = {}
    for r in results:
        s = r.get("source") or "Неизвестно"
        source_data.setdefault(s, []).append(r)

    summary = []
    for s, recs in source_data.items():
        summary.append({
            "source": s,
            "n_ideas": len(recs),
            "avg_delta_cav": round(float(np.mean([r.get("delta_cav", r["cav"]) for r in recs])), 4),
            "avg_cav": round(float(np.mean([r["cav"] for r in recs])), 4),
            "avg_peak_av": round(float(np.mean([r["peak_av_ratio"] for r in recs])), 4),
            "pct_significant": round(
                sum(1 for r in recs if r.get("is_significant")) / len(recs) * 100, 1
            ),
        })
    summary.sort(key=lambda x: x["avg_delta_cav"], reverse=True)
    return summary
