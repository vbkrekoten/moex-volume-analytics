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
        volumes_wide: pivot table (week_start index, instrument_class columns, total_value values)
        factors_wide: pivot table (week_start index, factor_name columns, value values)
        method: 'pearson' or 'spearman'
        use_changes: if True, compute on week-over-week % changes to avoid spurious correlation.

    Returns:
        Correlation matrix DataFrame (volume classes as rows, factors as columns).
    """
    if volumes_wide.empty or factors_wide.empty:
        return pd.DataFrame()

    # Align on common dates
    common = volumes_wide.index.intersection(factors_wide.index)
    if len(common) < 10:
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
            if len(common_idx) < 10:
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
    window: int = 26,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Compute rolling correlation between a volume series and a factor series.

    Returns: DataFrame with (week_start, correlation) columns.
    """
    # Align
    common = volume_series.index.intersection(factor_series.index)
    if len(common) < window + 5:
        return pd.DataFrame()

    vol = volume_series.loc[common].pct_change().dropna()
    fac = factor_series.loc[common].pct_change().dropna()
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
    result.columns = ["week_start", "correlation"]
    return result
