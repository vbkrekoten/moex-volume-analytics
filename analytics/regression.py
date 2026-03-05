"""OLS regression analysis for factor decomposition."""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


def factor_regression(
    volume_series: pd.Series,
    factor_df: pd.DataFrame,
    use_changes: bool = True,
) -> dict:
    """
    OLS regression: Volume ~ Factor1 + Factor2 + ... + FactorN.

    Args:
        volume_series: weekly volume for one instrument class (index = week_start).
        factor_df: weekly factors DataFrame (index = week_start, columns = factor names).
        use_changes: if True, regress on pct_change values.

    Returns dict with keys: r2, adj_r2, coefficients, pvalues, residuals, n_obs.
    """
    # Align
    common = volume_series.index.intersection(factor_df.index)
    if len(common) < 20:
        return {"r2": 0, "adj_r2": 0, "coefficients": {}, "pvalues": {},
                "residuals": pd.Series(dtype=float), "n_obs": 0}

    y = volume_series.loc[common]
    X = factor_df.loc[common]

    if use_changes:
        y = y.pct_change().dropna()
        X = X.pct_change().dropna()
        common = y.index.intersection(X.index)
        y = y.loc[common]
        X = X.loc[common]

    # Drop NaN and infinity
    mask = np.isfinite(y) & X.apply(np.isfinite).all(axis=1)
    y = y[mask]
    X = X[mask]

    if len(y) < 20 or X.shape[1] == 0:
        return {"r2": 0, "adj_r2": 0, "coefficients": {}, "pvalues": {},
                "residuals": pd.Series(dtype=float), "n_obs": 0}

    # Standardize for comparable coefficients
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), columns=X.columns, index=X.index
    )

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
