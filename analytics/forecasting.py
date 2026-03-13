"""HAR-RV volatility forecasting and turnover scenario analysis.

The HAR-RV (Heterogeneous Autoregressive of Realized Volatility) model uses
three time scales — daily, weekly (5-day) and monthly (22-day) — to forecast
realized volatility.  It is estimated via OLS (statsmodels) and requires no
additional packages.

Two-stage approach:
  Stage 1  →  forecast RV with confidence intervals (this module)
  Stage 2  →  forecast turnovers by applying regression coefficients
              to the volatility forecast + user-supplied macro scenario
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR = 252
FORECAST_HORIZON = 126  # ~6 months

# Minimum RV floor (annualised %) — IMOEX cannot be truly zero-vol
RV_FLOOR = 5.0


def _business_days(start: date, n: int) -> list[date]:
    """Generate *n* future business days starting from *start* (exclusive)."""
    days: list[date] = []
    current = start
    while len(days) < n:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Mon-Fri
            days.append(current)
    return days


# ---------------------------------------------------------------------------
# HAR-RV Model
# ---------------------------------------------------------------------------

class HARRVModel:
    """HAR-RV(1,5,22) volatility forecasting model.

    Specification::

        RV_{t+1} = b0  +  b_d * RV_t
                        +  b_w * mean(RV_{t-4:t})
                        +  b_m * mean(RV_{t-21:t})
                        +  epsilon
    """

    def __init__(self) -> None:
        self.params_: np.ndarray | None = None
        self.residual_std_: float = 0.0
        self.r2_: float = 0.0
        self.nobs_: int = 0
        self._is_fitted = False

    # ---- feature construction ----

    @staticmethod
    def build_features(rv: pd.Series) -> pd.DataFrame:
        """Build HAR features: RV_d (lag-1), RV_w (5-day mean), RV_m (22-day mean)."""
        rv = rv.astype(float)
        feat = pd.DataFrame(index=rv.index)
        feat["rv_d"] = rv.shift(1)
        feat["rv_w"] = rv.rolling(5).mean().shift(1)
        feat["rv_m"] = rv.rolling(22).mean().shift(1)
        return feat

    # ---- fit ----

    def fit(self, rv: pd.Series, min_obs: int = 252) -> dict[str, Any]:
        """Fit HAR-RV via OLS on the full history.

        Returns dict with keys: r2, nobs, coefficients, residual_std.
        """
        features = self.build_features(rv)
        y = rv.copy()

        # Align and drop NaN
        combined = pd.concat([y.rename("y"), features], axis=1).dropna()
        if len(combined) < min_obs:
            raise ValueError(
                f"Insufficient observations ({len(combined)}) for HAR-RV model (need {min_obs})"
            )

        Y = combined["y"].values
        X = sm.add_constant(combined[["rv_d", "rv_w", "rv_m"]].values)

        model = sm.OLS(Y, X).fit()

        self.params_ = model.params
        self.residual_std_ = float(np.std(model.resid, ddof=4))
        self.r2_ = float(model.rsquared)
        self.nobs_ = int(model.nobs)
        self._is_fitted = True

        return {
            "r2": round(self.r2_, 4),
            "nobs": self.nobs_,
            "coefficients": {
                "const": round(float(self.params_[0]), 4),
                "rv_d": round(float(self.params_[1]), 4),
                "rv_w": round(float(self.params_[2]), 4),
                "rv_m": round(float(self.params_[3]), 4),
            },
            "residual_std": round(self.residual_std_, 4),
        }

    # ---- forecast ----

    def forecast(
        self,
        rv: pd.Series,
        horizon: int = FORECAST_HORIZON,
        ci_levels: tuple[float, ...] = (0.80, 0.95),
    ) -> pd.DataFrame:
        """Generate iterated multi-step forecast with expanding CI bands.

        Returns DataFrame with columns:
            target_date, value, ci_lower_80, ci_upper_80, ci_lower_95, ci_upper_95
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        rv_vals = rv.astype(float).values
        last_date = rv.index[-1]
        if isinstance(last_date, pd.Timestamp):
            last_date = last_date.date()

        future_dates = _business_days(last_date, horizon)

        # Initialise rolling buffers from the tail of history
        buffer = list(rv_vals[-22:])  # last 22 values

        b0, b_d, b_w, b_m = self.params_

        # Effective persistence for AR-aware CI (mean-reverting uncertainty)
        phi = abs(b_d) + abs(b_w) + abs(b_m)
        phi = min(phi, 0.999)  # ensure convergence
        # Asymptotic variance: sigma_eps^2 / (1 - phi^2)
        sigma_inf = self.residual_std_ / np.sqrt(max(1 - phi**2, 0.01))

        forecasts: list[dict] = []
        for h, target_dt in enumerate(future_dates, start=1):
            rv_d = buffer[-1]
            rv_w = float(np.mean(buffer[-5:]))
            rv_m = float(np.mean(buffer[-22:]))

            point = b0 + b_d * rv_d + b_w * rv_w + b_m * rv_m
            point = max(point, RV_FLOOR)

            # AR-aware CI: grows with persistence, plateaus at sigma_inf
            # sigma_h = sigma_eps * sqrt((1 - phi^(2h)) / (1 - phi^2))
            sigma_h = self.residual_std_ * np.sqrt(
                (1 - phi ** (2 * h)) / max(1 - phi**2, 0.01)
            )
            sigma_h = min(sigma_h, sigma_inf)  # cap at asymptote

            row: dict[str, Any] = {
                "target_date": target_dt.isoformat(),
                "value": round(float(point), 4),
            }
            for level in ci_levels:
                z = sp_stats.norm.ppf(0.5 + level / 2)
                lo = max(point - z * sigma_h, RV_FLOOR)
                hi = point + z * sigma_h
                pct = int(level * 100)
                row[f"ci_lower_{pct}"] = round(float(lo), 4)
                row[f"ci_upper_{pct}"] = round(float(hi), 4)

            forecasts.append(row)
            buffer.append(point)

        return pd.DataFrame(forecasts)

    # ---- backtest ----

    def backtest(
        self,
        rv: pd.Series,
        test_size: int = 252,
        step: int = 21,
        horizons: tuple[int, ...] = (1, 5, 22, 63, 126),
    ) -> dict[str, Any]:
        """Walk-forward backtesting.

        Returns dict:
          - per_horizon: {h: {rmse, mae, n}} for each horizon
          - overall_rmse: RMSE averaged across horizons
        """
        rv_vals = rv.astype(float)
        n = len(rv_vals)

        if n < test_size + 126 + 252:
            # Not enough data for meaningful backtest
            return {"per_horizon": {}, "overall_rmse": None}

        train_end_start = n - test_size
        results: dict[int, list[float]] = {h: [] for h in horizons}

        for t in range(train_end_start, n - max(horizons), step):
            train = rv_vals.iloc[:t]
            try:
                temp_model = HARRVModel()
                temp_model.fit(train, min_obs=200)
                fc = temp_model.forecast(train, horizon=max(horizons))
            except Exception:
                continue

            for h in horizons:
                if t + h > n:
                    continue
                # Compare average forecast over horizon with average actual
                actual_avg = float(rv_vals.iloc[t: t + h].mean())
                fc_avg = float(fc["value"].iloc[:h].mean())
                results[h].append((actual_avg - fc_avg) ** 2)

        per_horizon: dict[int, dict[str, float]] = {}
        for h in horizons:
            if results[h]:
                rmse = float(np.sqrt(np.mean(results[h])))
                mae = float(np.mean(np.sqrt(results[h])))
                per_horizon[h] = {"rmse": round(rmse, 4), "mae": round(mae, 4), "n": len(results[h])}

        overall_rmse = None
        if per_horizon:
            overall_rmse = round(float(np.mean([v["rmse"] for v in per_horizon.values()])), 4)

        return {"per_horizon": per_horizon, "overall_rmse": overall_rmse}


# ---------------------------------------------------------------------------
# Aggregate daily forecast to monthly
# ---------------------------------------------------------------------------

def aggregate_forecast_monthly(fc: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily point forecast + CIs to calendar months.

    Returns DataFrame with columns:
        month_label, value, ci_lower_80, ci_upper_80, ci_lower_95, ci_upper_95
    """
    df = fc.copy()
    df["target_date"] = pd.to_datetime(df["target_date"])
    df["month"] = df["target_date"].dt.to_period("M")

    ci_cols = [c for c in df.columns if c.startswith("ci_")]
    agg_dict = {"value": "mean"}
    for c in ci_cols:
        agg_dict[c] = "mean"

    monthly = df.groupby("month").agg(agg_dict).reset_index()
    monthly["month_label"] = monthly["month"].dt.strftime("%b %Y")

    # Round values
    for col in ["value"] + ci_cols:
        monthly[col] = monthly[col].round(1)

    return monthly


# ---------------------------------------------------------------------------
# Stage 2: Turnover forecasting via regression + macro scenarios
# ---------------------------------------------------------------------------

DEFAULT_SCENARIOS = {
    "base": {"label": "Базовый", "description": "Текущие значения сохраняются"},
    "optimistic": {"label": "Оптимистичный", "description": "Смягчение ДКП, рост нефти"},
    "pessimistic": {"label": "Пессимистичный", "description": "Ужесточение ДКП, падение нефти"},
}


def estimate_turnover_regression(
    daily_vol: pd.DataFrame,
    daily_factors: pd.DataFrame,
    factor_names: list[str] | None = None,
    min_obs: int = 120,
) -> dict[str, dict[str, Any]]:
    """Estimate OLS regression: turnover ~ factors, per instrument class.

    Returns {instrument_class: {coefficients: {factor: beta}, intercept, r2, factor_means, factor_stds}}.
    """
    if daily_vol.empty or daily_factors.empty:
        return {}

    # Default factors relevant for turnover forecasting
    if factor_names is None:
        factor_names = ["volatility", "key_rate", "brent", "usd_rub", "index_level"]

    # Pivot factors to wide format
    fdf = daily_factors.copy()
    fdf["trade_date"] = pd.to_datetime(fdf["trade_date"])
    fdf = fdf[fdf["factor_name"].isin(factor_names)]
    factor_wide = fdf.pivot_table(
        index="trade_date", columns="factor_name", values="value", aggfunc="first"
    )
    factor_wide = factor_wide.dropna()

    # Total turnover per class per day
    vol = daily_vol.copy()
    vol["trade_date"] = pd.to_datetime(vol["trade_date"])
    turnover_by_class = vol.pivot_table(
        index="trade_date", columns="instrument_class",
        values="value_rub", aggfunc="sum", fill_value=0,
    )

    results: dict[str, dict] = {}
    for cls in turnover_by_class.columns:
        y = turnover_by_class[cls]
        common_idx = y.index.intersection(factor_wide.index)
        if len(common_idx) < min_obs:
            continue

        y_c = y.loc[common_idx]
        X_c = factor_wide.loc[common_idx]

        # Drop NaN rows
        mask = np.isfinite(y_c) & X_c.apply(np.isfinite).all(axis=1)
        y_c = y_c[mask]
        X_c = X_c[mask]

        if len(y_c) < min_obs:
            continue

        # Store means/stds for later scoring
        means = X_c.mean()
        stds = X_c.std()
        stds = stds.replace(0, 1)

        X_scaled = (X_c - means) / stds
        X_const = sm.add_constant(X_scaled)

        try:
            model = sm.OLS(y_c.values, X_const).fit()
        except Exception:
            continue

        coefficients = {}
        for i, fname in enumerate(X_c.columns):
            coefficients[fname] = float(model.params[i + 1])

        results[cls] = {
            "intercept": float(model.params[0]),
            "coefficients": coefficients,
            "r2": round(float(model.rsquared), 4),
            "factor_means": means.to_dict(),
            "factor_stds": stds.to_dict(),
            "y_mean": float(y_c.mean()),
        }

    return results


def forecast_turnovers(
    vol_forecast_value: float,
    macro_params: dict[str, float],
    regression_models: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Predict turnover per instrument class given a volatility forecast and macro assumptions.

    Args:
        vol_forecast_value: forecasted average RV (%)
        macro_params: {factor_name: value} e.g. {"key_rate": 21.0, "brent": 65.0, ...}
        regression_models: output of estimate_turnover_regression()

    Returns:
        {instrument_class: predicted_daily_turnover_mln_rub}
    """
    scenario_factors = {"volatility": vol_forecast_value}
    scenario_factors.update(macro_params)

    predictions: dict[str, float] = {}
    for cls, model_info in regression_models.items():
        intercept = model_info["intercept"]
        coefficients = model_info["coefficients"]
        means = model_info["factor_means"]
        stds = model_info["factor_stds"]

        prediction = intercept
        for fname, beta in coefficients.items():
            val = scenario_factors.get(fname)
            if val is None:
                # Use historical mean (neutral assumption)
                val = means.get(fname, 0)
            # Standardise using training stats
            z = (val - means.get(fname, 0)) / stds.get(fname, 1)
            prediction += beta * z

        predictions[cls] = max(0, float(prediction))

    return predictions
