"""Factor validation: significance, stability, out-of-sample robustness."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def single_factor_r2(
    volume_series: pd.Series,
    factor_series: pd.Series,
    use_changes: bool = True,
) -> dict:
    """
    Compute R2 and p-value for a single factor vs volume.

    Returns dict with: r2, pvalue, n_obs, coefficient.
    """
    common = volume_series.index.intersection(factor_series.index)
    if len(common) < 30:
        return {"r2": 0, "pvalue": 1.0, "n_obs": 0, "coefficient": 0}

    y = volume_series.loc[common].copy()
    x = factor_series.loc[common].copy()

    if use_changes:
        y = y.pct_change(fill_method=None).iloc[1:]
        x = x.pct_change(fill_method=None).iloc[1:]
        common = y.index.intersection(x.index)
        y = y.loc[common]
        x = x.loc[common]

    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask]
    x = x[mask]

    if len(y) < 30:
        return {"r2": 0, "pvalue": 1.0, "n_obs": 0, "coefficient": 0}

    # Standardize
    x_std = (x - x.mean()) / x.std() if x.std() > 0 else x
    X = sm.add_constant(x_std)
    model = sm.OLS(y, X).fit()

    return {
        "r2": round(model.rsquared, 4),
        "pvalue": round(model.pvalues.iloc[-1], 4),
        "n_obs": int(model.nobs),
        "coefficient": round(model.params.iloc[-1], 6),
    }


def rolling_r2(
    volume_series: pd.Series,
    factor_series: pd.Series,
    window: int = 120,
    use_changes: bool = True,
) -> pd.DataFrame:
    """
    Compute rolling single-factor R2 over time.

    Returns DataFrame with (date, r2) columns.
    """
    common = volume_series.index.intersection(factor_series.index)
    if len(common) < window + 10:
        return pd.DataFrame()

    y = volume_series.loc[common].copy()
    x = factor_series.loc[common].copy()

    if use_changes:
        y = y.pct_change(fill_method=None).iloc[1:]
        x = x.pct_change(fill_method=None).iloc[1:]
        common = y.index.intersection(x.index)
        y = y.loc[common]
        x = x.loc[common]

    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask]
    x = x[mask]

    r2_values = []
    dates = []
    for i in range(window, len(y)):
        y_win = y.iloc[i - window:i]
        x_win = x.iloc[i - window:i]
        if x_win.std() > 0 and y_win.std() > 0:
            corr = np.corrcoef(x_win, y_win)[0, 1]
            r2_values.append(corr ** 2)
        else:
            r2_values.append(0)
        dates.append(y.index[i])

    return pd.DataFrame({"date": dates, "r2": r2_values})


def compute_factor_stability(
    volume_series: pd.Series,
    factor_df: pd.DataFrame,
    window: int = 120,
) -> pd.DataFrame:
    """
    Compute stability metrics for each factor.

    Returns DataFrame with columns:
    - factor: factor name
    - r2: full-sample R2
    - pvalue: full-sample p-value
    - mean_rolling_r2: average R2 over rolling windows
    - r2_stability: 1 - std of rolling R2 (higher = more stable)
    - oos_r2: out-of-sample R2 (last 30% of data)
    - status: 'green', 'yellow', or 'red'
    """
    results = []

    for col in factor_df.columns:
        factor_s = factor_df[col].dropna()
        # Full-sample
        full = single_factor_r2(volume_series, factor_s)

        # Rolling R2
        rolling = rolling_r2(volume_series, factor_s, window=window)
        if not rolling.empty:
            mean_r2 = rolling["r2"].mean()
            r2_std = rolling["r2"].std()
            r2_stability = max(0, 1 - r2_std * 5)  # penalize high variance
        else:
            mean_r2 = 0
            r2_std = 1
            r2_stability = 0

        # Out-of-sample R2
        n = len(volume_series)
        split = int(n * 0.7)
        if split > 60:
            oos_metrics = single_factor_r2(
                volume_series.iloc[split:],
                factor_s,
            )
            oos_r2 = oos_metrics["r2"]
        else:
            oos_r2 = 0

        # Traffic light status
        status = _traffic_light(full["pvalue"], full["r2"], r2_stability, oos_r2)

        results.append({
            "factor": col,
            "r2": full["r2"],
            "pvalue": full["pvalue"],
            "coefficient": full["coefficient"],
            "n_obs": full["n_obs"],
            "mean_rolling_r2": round(mean_r2, 4),
            "r2_stability": round(r2_stability, 4),
            "oos_r2": round(oos_r2, 4),
            "status": status,
        })

    return pd.DataFrame(results)


def _traffic_light(pvalue: float, r2: float, stability: float, oos_r2: float) -> str:
    """Determine traffic light status based on factor metrics."""
    score = 0
    if pvalue < 0.01:
        score += 3
    elif pvalue < 0.05:
        score += 2
    elif pvalue < 0.10:
        score += 1

    if r2 > 0.05:
        score += 2
    elif r2 > 0.02:
        score += 1

    if stability > 0.6:
        score += 1

    if oos_r2 > 0.02:
        score += 1

    if score >= 5:
        return "green"
    elif score >= 3:
        return "yellow"
    return "red"
