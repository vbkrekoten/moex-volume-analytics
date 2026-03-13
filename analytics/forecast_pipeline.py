"""Forecast pipeline step: fit HAR-RV, generate forecasts, store in Supabase."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from analytics.forecasting import (
    HARRVModel,
    FORECAST_HORIZON,
)
from data_pipeline.db import upsert_rows

logger = logging.getLogger(__name__)


def run_forecast_pipeline(
    client,
    daily_factors_df: pd.DataFrame,
    progress_callback=None,
) -> None:
    """Generate volatility forecasts and store them.

    Steps:
        1. Extract RV series from daily_factors
        2. Fit HAR-RV model
        3. Generate 126-day forecast with CIs
        4. Walk-forward backtest (diagnostics)
        5. Upsert results to vol_forecasts & vol_forecast_diagnostics
    """

    def _log(msg: str) -> None:
        logger.info(msg)
        if progress_callback:
            progress_callback(msg, 0.90)

    # --- 1. Extract RV series ---
    if daily_factors_df.empty:
        _log("No daily factors data — skipping forecast pipeline")
        return

    fdf = daily_factors_df.copy()
    fdf["trade_date"] = pd.to_datetime(fdf["trade_date"])
    fdf["value"] = pd.to_numeric(fdf["value"], errors="coerce")

    rv_df = fdf[fdf["factor_name"] == "volatility"].dropna(subset=["value"])
    if rv_df.empty:
        _log("No RV data found — skipping forecast pipeline")
        return

    rv_series = rv_df.set_index("trade_date")["value"].sort_index()
    rv_series = rv_series[~rv_series.index.duplicated(keep="last")]

    if len(rv_series) < 252:
        _log(f"Only {len(rv_series)} RV observations — need at least 252, skipping")
        return

    # --- 2. Fit HAR-RV ---
    _log("Fitting HAR-RV model...")
    model = HARRVModel()
    try:
        fit_info = model.fit(rv_series, min_obs=252)
        logger.info("HAR-RV fit: R²=%.4f, nobs=%d, σ=%.4f",
                     fit_info["r2"], fit_info["nobs"], fit_info["residual_std"])
    except Exception as e:
        logger.error("HAR-RV fit failed: %s", e)
        return

    # --- 3. Generate forecast ---
    _log("Generating 6-month volatility forecast...")
    try:
        fc = model.forecast(rv_series, horizon=FORECAST_HORIZON)
    except Exception as e:
        logger.error("Forecast generation failed: %s", e)
        return

    forecast_date = date.today().isoformat()

    # Build rows for vol_forecasts
    forecast_rows = []
    for _, row in fc.iterrows():
        forecast_rows.append({
            "forecast_date": forecast_date,
            "target_date": row["target_date"],
            "target_name": "volatility",
            "model_name": "har_rv",
            "scenario": "base",
            "value": row["value"],
            "ci_lower_80": row.get("ci_lower_80"),
            "ci_upper_80": row.get("ci_upper_80"),
            "ci_lower_95": row.get("ci_lower_95"),
            "ci_upper_95": row.get("ci_upper_95"),
        })

    if forecast_rows:
        n = upsert_rows(client, "vol_forecasts", forecast_rows)
        logger.info("Upserted %d volatility forecast rows", n)

    # --- 4. Backtest diagnostics ---
    _log("Running walk-forward backtest...")
    try:
        bt = model.backtest(rv_series, test_size=252, step=21)
    except Exception as e:
        logger.warning("Backtest failed: %s", e)
        bt = {"per_horizon": {}, "overall_rmse": None}

    diag_rows = []
    for h, metrics in bt.get("per_horizon", {}).items():
        for metric_name in ("rmse", "mae"):
            if metric_name in metrics:
                diag_rows.append({
                    "forecast_date": forecast_date,
                    "model_name": "har_rv",
                    "target_name": "volatility",
                    "metric_name": metric_name,
                    "metric_value": metrics[metric_name],
                    "eval_window": f"{h}d",
                })

    # Overall RMSE
    if bt.get("overall_rmse") is not None:
        diag_rows.append({
            "forecast_date": forecast_date,
            "model_name": "har_rv",
            "target_name": "volatility",
            "metric_name": "rmse",
            "metric_value": bt["overall_rmse"],
            "eval_window": "overall",
        })

    # Model R²
    diag_rows.append({
        "forecast_date": forecast_date,
        "model_name": "har_rv",
        "target_name": "volatility",
        "metric_name": "r2",
        "metric_value": fit_info["r2"],
        "eval_window": "full",
    })

    # Coefficients as diagnostics
    for coef_name, coef_val in fit_info["coefficients"].items():
        diag_rows.append({
            "forecast_date": forecast_date,
            "model_name": "har_rv",
            "target_name": "volatility",
            "metric_name": f"coef_{coef_name}",
            "metric_value": coef_val,
            "eval_window": "full",
        })

    if diag_rows:
        n = upsert_rows(client, "vol_forecast_diagnostics", diag_rows)
        logger.info("Upserted %d diagnostic rows", n)

    _log("Forecast pipeline complete")
