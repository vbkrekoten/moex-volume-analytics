"""OLS regression analysis for factor decomposition."""

import pandas as pd
import numpy as np
import statsmodels.api as sm


def factor_regression(
    volume_series: pd.Series,
    factor_df: pd.DataFrame,
    use_changes: bool = True,
    min_obs: int = 30,
) -> dict:
    """
    OLS regression: Volume ~ Factor1 + Factor2 + ... + FactorN.

    Args:
        volume_series: volume for one instrument class (index = date).
        factor_df: factors DataFrame (index = date, columns = factor names).
        use_changes: if True, regress on pct_change values.
        min_obs: minimum number of observations required.

    Returns dict with keys: r2, adj_r2, coefficients, pvalues, residuals, n_obs.
    """
    empty_result = {
        "r2": 0, "adj_r2": 0, "coefficients": {}, "pvalues": {},
        "residuals": pd.Series(dtype=float), "n_obs": 0,
    }

    # Align
    common = volume_series.index.intersection(factor_df.index)
    if len(common) < min_obs:
        return empty_result

    y = volume_series.loc[common]
    X = factor_df.loc[common]

    if use_changes:
        y = y.pct_change(fill_method=None).dropna()
        X = X.pct_change(fill_method=None).dropna()
        common = y.index.intersection(X.index)
        y = y.loc[common]
        X = X.loc[common]

    # Drop NaN and infinity
    mask = np.isfinite(y) & X.apply(np.isfinite).all(axis=1)
    y = y[mask]
    X = X[mask]

    if len(y) < min_obs or X.shape[1] == 0:
        return empty_result

    # Drop columns with zero variance
    X = X.loc[:, X.std() > 0]
    if X.shape[1] == 0:
        return empty_result

    # Standardize for comparable coefficients (z-score)
    X_scaled = (X - X.mean()) / X.std()

    # OLS
    X_const = sm.add_constant(X_scaled)
    model = sm.OLS(y, X_const).fit()

    return {
        "r2": round(model.rsquared, 4),
        "adj_r2": round(model.rsquared_adj, 4),
        "coefficients": {k: round(v, 4) for k, v in model.params.items() if k != "const"},
        "pvalues": {k: round(v, 4) for k, v in model.pvalues.items() if k != "const"},
        "residuals": model.resid,
        "n_obs": int(model.nobs),
    }


def rolling_regression_r2(
    volume_series: pd.Series,
    factor_df: pd.DataFrame,
    window: int = 120,
    use_changes: bool = True,
) -> pd.DataFrame:
    """
    Rolling OLS R2 over time to detect structural breaks.

    Returns DataFrame with (date, r2) columns.
    """
    common = volume_series.index.intersection(factor_df.index)
    if len(common) < window + 30:
        return pd.DataFrame()

    y = volume_series.loc[common]
    X = factor_df.loc[common]

    if use_changes:
        y = y.pct_change(fill_method=None).dropna()
        X = X.pct_change(fill_method=None).dropna()
        common = y.index.intersection(X.index)
        y = y.loc[common]
        X = X.loc[common]

    mask = np.isfinite(y) & X.apply(np.isfinite).all(axis=1)
    y = y[mask]
    X = X[mask]

    r2_values = []
    dates = []
    for i in range(window, len(y)):
        y_win = y.iloc[i - window:i]
        X_win = X.iloc[i - window:i]

        X_win = X_win.loc[:, X_win.std() > 0]
        if X_win.shape[1] == 0:
            r2_values.append(0)
            dates.append(y.index[i])
            continue

        X_s = (X_win - X_win.mean()) / X_win.std()
        X_c = sm.add_constant(X_s)
        try:
            model = sm.OLS(y_win, X_c).fit()
            r2_values.append(model.rsquared)
        except Exception:
            r2_values.append(0)
        dates.append(y.index[i])

    return pd.DataFrame({"date": dates, "r2": r2_values})
