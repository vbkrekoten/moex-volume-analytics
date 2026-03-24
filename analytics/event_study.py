"""Event study analytics for measuring investment idea impact on trading volume."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_abnormal_volume(
    volume_series: pd.Series,
    event_date: str | pd.Timestamp,
    est_window: tuple[int, int] = (-120, -6),
    evt_window: tuple[int, int] = (-1, 5),
) -> dict | None:
    """Compute Abnormal Volume metrics for a single event.

    Args:
        volume_series: Series indexed by trade_date (datetime) with daily volumes.
        event_date: the date of the investment idea publication.
        est_window: (start, end) offsets in trading days for estimation period.
        evt_window: (start, end) offsets in trading days for event period.

    Returns:
        dict with AV ratio, NAV, CAV, peak metrics, day-by-day arrays.
        None if insufficient data.
    """
    event_date = pd.Timestamp(event_date)
    vs = volume_series.sort_index()

    # Find event date position in the trading calendar
    dates = vs.index
    if event_date not in dates:
        # Find nearest trading day on or after event_date
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
    est_values = est_values[est_values > 0]  # filter zero-volume days

    if len(est_values) < 30:  # minimum observations
        return None

    est_mean = float(np.mean(est_values))
    est_std = float(np.std(est_values, ddof=1))

    if est_mean <= 0 or est_std <= 0:
        return None

    # Extract event window
    evt_start = event_idx + evt_window[0]
    evt_end = event_idx + evt_window[1]
    evt_start = max(evt_start, 0)
    evt_end = min(evt_end, len(vs) - 1)

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

    # Cumulative Abnormal Volume (sum of excess ratios)
    cav = sum(r - 1.0 for r in av_ratios)

    # Peak metrics
    peak_idx = int(np.argmax(av_ratios))
    peak_av_ratio = av_ratios[peak_idx]
    peak_av_day = day_offsets[peak_idx]
    peak_nav = navs[peak_idx]
    is_significant = abs(peak_nav) > 1.96

    return {
        "ticker": None,  # to be filled by caller
        "idea_date": str(event_date.date()),
        "est_mean_volume": round(est_mean, 2),
        "est_std_volume": round(est_std, 2),
        "est_days": len(est_values),
        "cav": round(cav, 4),
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


def aggregate_event_study(results: list[dict]) -> dict:
    """Cross-sectional aggregation of event study results.

    Args:
        results: list of dicts from compute_abnormal_volume (non-None).

    Returns:
        dict with aggregate metrics: AAV by day, CAAV, % significant, top tickers.
    """
    if not results:
        return {}

    # Collect day-by-day ratios across all events
    all_offsets = set()
    for r in results:
        for d in r["av_ratio_by_day"]["day_offsets"]:
            all_offsets.add(d)
    day_range = sorted(all_offsets)

    # Build matrix: rows = events, cols = day offsets
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

    # Aggregate metrics
    n_total = len(results)
    n_significant = sum(1 for r in results if r["is_significant"])
    cavs = [r["cav"] for r in results]
    peak_ratios = [r["peak_av_ratio"] for r in results]

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
            "avg_cav": round(float(np.mean([r["cav"] for r in recs])), 4),
            "avg_peak_av": round(float(np.mean([r["peak_av_ratio"] for r in recs])), 4),
            "pct_significant": round(
                sum(1 for r in recs if r["is_significant"]) / len(recs) * 100, 1
            ),
        })
    ticker_summary.sort(key=lambda x: x["avg_cav"], reverse=True)

    return {
        "n_total": n_total,
        "n_significant": n_significant,
        "pct_significant": round(n_significant / n_total * 100, 1) if n_total else 0,
        "mean_cav": round(float(np.mean(cavs)), 4),
        "median_cav": round(float(np.median(cavs)), 4),
        "mean_peak_av_ratio": round(float(np.mean(peak_ratios)), 4),
        "median_peak_av_ratio": round(float(np.median(peak_ratios)), 4),
        "aav_by_day": aav_by_day,
        "ticker_summary": ticker_summary,
    }
