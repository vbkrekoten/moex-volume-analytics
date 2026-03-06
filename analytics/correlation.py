"""Correlation analysis: Pearson, Spearman, rolling correlations."""

import pandas as pd
import numpy as np
from scipy import stats


def compute_correlation_matrix(
    volumes_wide: pd.DataFrame,
    factors_wide: pd.DataFrame,
    method: str = "pearson",
    use_changes: bool = True,
) -> pd.DataFrame:
    """
    Compute correlation matrix between volume series and factor series.

    Args:
        volumes_wide: pivot table (date index, instrument_class columns, values).
        factors_wide: pivot table (date index, factor_name columns, values).
        method: 'pearson' or 'spearman'.
        use_changes: if True, compute on day-over-day % changes.

    Returns:
        Correlation matrix DataFrame (volume classes as rows, factors as columns).
    """
    if volumes_wide.empty or factors_wide.empty:
        return pd.DataFrame()

    # Align on common dates
    common = volumes_wide.index.intersection(factors_wide.index)
    if len(common) < 20:
        return pd.DataFrame()

    vol = volumes_wide.loc[common]
    fac = factors_wide.loc[common]

    if use_changes:
        vol = vol.pct_change(fill_method=None).dropna()
        fac = fac.pct_change(fill_method=None).dropna()
        common = vol.index.intersection(fac.index)
        vol = vol.loc[common]
        fac = fac.loc[common]

    result = pd.DataFrame(index=vol.columns, columns=fac.columns, dtype=float)

    for vc in vol.columns:
        for fc in fac.columns:
            v = vol[vc].dropna()
            f = fac[fc].dropna()
            common_idx = v.index.intersection(f.index)
            if len(common_idx) < 20:
                result.loc[vc, fc] = np.nan
                continue
            if method == "spearman":
                corr, _ = stats.spearmanr(v[common_idx], f[common_idx])
            else:
                corr, _ = stats.pearsonr(v[common_idx], f[common_idx])
            result.loc[vc, fc] = round(corr, 4)

    return result.astype(float)


def rolling_correlation(
    volume_series: pd.Series,
    factor_series: pd.Series,
    window: int = 90,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Compute rolling correlation between a volume series and a factor series.

    Args:
        window: rolling window in trading days (default 90 ~ 3 months).

    Returns: DataFrame with (date, correlation) columns.
    """
    common = volume_series.index.intersection(factor_series.index)
    if len(common) < window + 10:
        return pd.DataFrame()

    vol = volume_series.loc[common].pct_change(fill_method=None).dropna()
    fac = factor_series.loc[common].pct_change(fill_method=None).dropna()
    common = vol.index.intersection(fac.index)
    vol = vol.loc[common]
    fac = fac.loc[common]

    if method == "spearman":
        corr = vol.rolling(window).apply(
            lambda x: stats.spearmanr(x, fac.loc[x.index])[0],
            raw=False,
        )
    else:
        corr = vol.rolling(window).corr(fac)

    result = corr.dropna().reset_index()
    result.columns = ["date", "correlation"]
    return result
