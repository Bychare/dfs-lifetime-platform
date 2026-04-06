"""
Statistical test helpers.

Wraps scipy / statsmodels / lifelines into convenient functions
that return results in a format suitable for display in Dash.
"""

import numpy as np
import pandas as pd
from scipy import stats


def normality_test(series: pd.Series, alpha: float = 0.05) -> dict:
    """Shapiro-Wilk test (on a subsample if n > 5000)."""
    s = series.dropna()
    if len(s) > 5000:
        s = s.sample(5000, random_state=42)
    stat, p = stats.shapiro(s)
    return {"test": "Shapiro-Wilk", "statistic": stat, "p_value": p, "normal": p > alpha}


def levene_test(groups: list[pd.Series], alpha: float = 0.05) -> dict:
    """Levene's test for homogeneity of variances."""
    clean = [g.dropna() for g in groups]
    stat, p = stats.levene(*clean)
    return {"test": "Levene", "statistic": stat, "p_value": p, "equal_var": p > alpha}


def kruskal_wallis(groups: list[pd.Series]) -> dict:
    """Kruskal-Wallis H-test."""
    clean = [g.dropna() for g in groups]
    stat, p = stats.kruskal(*clean)
    # Effect size: epsilon-squared
    n = sum(len(g) for g in clean)
    k = len(clean)
    eps_sq = (stat - k + 1) / (n - k)
    return {"test": "Kruskal-Wallis", "statistic": stat, "p_value": p, "epsilon_sq": eps_sq}


def mann_whitney(a: pd.Series, b: pd.Series) -> dict:
    """Mann-Whitney U test."""
    stat, p = stats.mannwhitneyu(a.dropna(), b.dropna(), alternative="two-sided")
    return {"test": "Mann-Whitney U", "statistic": stat, "p_value": p}


def sample_size_proportions(
    p0: float, mde: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """
    Required sample size per group for a two-proportion z-test.
    Lehr's formula via z-quantiles.
    """
    p1 = p0 + mde
    p_bar = (p0 + p1) / 2
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = (
        (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
         + z_beta * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
    ) / mde**2
    return int(np.ceil(n))


def sample_size_continuous(
    sigma: float, mde: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Required sample size per group for a two-sample t-test."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = 2 * ((z_alpha + z_beta) * sigma / mde) ** 2
    return int(np.ceil(n))
