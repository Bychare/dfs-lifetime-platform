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


def sample_size_survival(
    hazard_ratio: float,
    event_rate: float,
    alpha: float = 0.05,
    power: float = 0.80,
    allocation_ratio: float = 0.5,
) -> int:
    """
    Schoenfeld approximation for Cox PH / log-rank designs.

    Returns total sample size across both arms.
    """
    if hazard_ratio <= 0 or np.isclose(hazard_ratio, 1.0):
        raise ValueError("hazard_ratio must be positive and different from 1.")
    if not 0 < event_rate <= 1:
        raise ValueError("event_rate must be in (0, 1].")
    if not 0 < allocation_ratio < 1:
        raise ValueError("allocation_ratio must be in (0, 1).")

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    log_hr = np.log(hazard_ratio)
    required_events = ((z_alpha + z_beta) ** 2) / (
        allocation_ratio * (1 - allocation_ratio) * log_hr**2
    )
    return int(np.ceil(required_events / event_rate))


def proportion_z_test(
    control_success: int,
    control_total: int,
    treatment_success: int,
    treatment_total: int,
    alpha: float = 0.05,
) -> dict:
    """Two-sided z-test for two proportions with Wald CI for the difference."""
    p_control = control_success / control_total
    p_treatment = treatment_success / treatment_total
    diff = p_treatment - p_control

    pooled = (control_success + treatment_success) / (control_total + treatment_total)
    se_pooled = np.sqrt(
        pooled * (1 - pooled) * (1 / control_total + 1 / treatment_total)
    )
    if se_pooled == 0:
        z_stat = 0.0
        p_value = 1.0
    else:
        z_stat = diff / se_pooled
        p_value = 2 * stats.norm.sf(abs(z_stat))

    se_ci = np.sqrt(
        p_control * (1 - p_control) / control_total
        + p_treatment * (1 - p_treatment) / treatment_total
    )
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_low = diff - z_crit * se_ci
    ci_high = diff + z_crit * se_ci

    return {
        "control_rate": p_control,
        "treatment_rate": p_treatment,
        "absolute_diff": diff,
        "relative_uplift": diff / p_control if p_control > 0 else np.nan,
        "z_stat": z_stat,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "significant": p_value < alpha,
    }


def welch_t_test(control: np.ndarray, treatment: np.ndarray, alpha: float = 0.05) -> dict:
    """Welch t-test for a continuous KPI with CI for the mean difference."""
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    stat, p_value = stats.ttest_ind(treatment, control, equal_var=False, nan_policy="omit")

    n0 = len(control)
    n1 = len(treatment)
    mean0 = float(np.nanmean(control))
    mean1 = float(np.nanmean(treatment))
    var0 = float(np.nanvar(control, ddof=1))
    var1 = float(np.nanvar(treatment, ddof=1))
    se = np.sqrt(var0 / n0 + var1 / n1)

    numerator = (var0 / n0 + var1 / n1) ** 2
    denominator = (var0**2) / (n0**2 * (n0 - 1)) + (var1**2) / (n1**2 * (n1 - 1))
    df = numerator / denominator if denominator > 0 else max(min(n0, n1) - 1, 1)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    diff = mean1 - mean0

    pooled_sd = np.sqrt(((n0 - 1) * var0 + (n1 - 1) * var1) / max(n0 + n1 - 2, 1))
    cohen_d = diff / pooled_sd if pooled_sd > 0 else 0.0

    return {
        "control_mean": mean0,
        "treatment_mean": mean1,
        "absolute_diff": diff,
        "relative_uplift": diff / mean0 if mean0 != 0 else np.nan,
        "t_stat": float(stat),
        "p_value": float(p_value),
        "ci_low": diff - t_crit * se,
        "ci_high": diff + t_crit * se,
        "df": float(df),
        "cohen_d": float(cohen_d),
        "significant": p_value < alpha,
    }


def beta_binomial_ab_test(
    control_success: int,
    control_total: int,
    treatment_success: int,
    treatment_total: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    draws: int = 20000,
    seed: int = 42,
) -> dict:
    """Bayesian A/B test with independent Beta priors for binary outcomes."""
    rng = np.random.default_rng(seed)

    control_draws = rng.beta(
        prior_alpha + control_success,
        prior_beta + control_total - control_success,
        size=draws,
    )
    treatment_draws = rng.beta(
        prior_alpha + treatment_success,
        prior_beta + treatment_total - treatment_success,
        size=draws,
    )
    diff_draws = treatment_draws - control_draws

    return {
        "control_posterior_mean": float(control_draws.mean()),
        "treatment_posterior_mean": float(treatment_draws.mean()),
        "prob_treatment_beats_control": float(np.mean(diff_draws > 0)),
        "expected_loss_if_ship": float(np.mean(np.maximum(-diff_draws, 0))),
        "ci_low": float(np.quantile(diff_draws, 0.025)),
        "ci_high": float(np.quantile(diff_draws, 0.975)),
        "diff_draws": diff_draws,
    }


def bootstrap_uplift_ci(
    control: np.ndarray,
    treatment: np.ndarray,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Non-parametric bootstrap CI for the difference in arm means."""
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    if len(control) == 0 or len(treatment) == 0:
        raise ValueError("Both arms must contain at least one observation.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1).")
    if n_boot < 200:
        raise ValueError("n_boot must be at least 200.")

    rng = np.random.default_rng(seed)
    boot_control = rng.choice(control, size=(n_boot, len(control)), replace=True).mean(axis=1)
    boot_treatment = rng.choice(
        treatment, size=(n_boot, len(treatment)), replace=True
    ).mean(axis=1)
    diff_draws = boot_treatment - boot_control
    alpha = 1 - confidence

    return {
        "point_estimate": float(np.mean(treatment) - np.mean(control)),
        "ci_low": float(np.quantile(diff_draws, alpha / 2)),
        "ci_high": float(np.quantile(diff_draws, 1 - alpha / 2)),
        "std_error": float(diff_draws.std(ddof=1)),
        "diff_draws": diff_draws,
    }


def adjust_pvalues(p_values, method: str = "holm") -> np.ndarray:
    """Holm or Benjamini-Hochberg correction for a family of p-values."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1:
        raise ValueError("p_values must be one-dimensional.")
    if np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("p_values must lie in [0, 1].")

    n = len(p_values)
    if n == 0:
        return np.array([], dtype=float)

    order = np.argsort(p_values)
    ranked = p_values[order]

    if method == "holm":
        adjusted_ranked = np.maximum.accumulate((n - np.arange(n)) * ranked)
    elif method == "bh":
        adjusted_ranked = np.minimum.accumulate((ranked * n / (np.arange(n) + 1))[::-1])[::-1]
    else:
        raise ValueError("method must be either 'holm' or 'bh'.")

    adjusted = np.empty(n, dtype=float)
    adjusted[order] = np.clip(adjusted_ranked, 0, 1)
    return adjusted


def obrien_fleming_bounds(n_looks: int, alpha: float = 0.05) -> pd.DataFrame:
    """Two-sided O'Brien-Fleming z-boundaries across equally spaced looks."""
    if n_looks < 2:
        raise ValueError("n_looks must be at least 2.")

    info_frac = np.arange(1, n_looks + 1) / n_looks
    final_z = stats.norm.ppf(1 - alpha / 2)
    bounds = final_z / np.sqrt(info_frac)
    return pd.DataFrame(
        {
            "look": np.arange(1, n_looks + 1),
            "information_fraction": info_frac,
            "z_boundary": bounds,
        }
    )


def sequential_proportion_monitor(
    control: np.ndarray,
    treatment: np.ndarray,
    n_looks: int = 5,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, int | None]:
    """Evaluate cumulative z-statistics against O'Brien-Fleming boundaries."""
    control = np.asarray(control, dtype=int)
    treatment = np.asarray(treatment, dtype=int)
    n = min(len(control), len(treatment))
    if n < n_looks:
        raise ValueError("Need at least as many observations as looks.")

    look_sizes = np.unique(np.ceil(np.linspace(n / n_looks, n, n_looks)).astype(int))
    boundaries = obrien_fleming_bounds(len(look_sizes), alpha=alpha)

    rows = []
    stop_look = None
    for idx, size in enumerate(look_sizes, start=1):
        result = proportion_z_test(
            int(control[:size].sum()),
            size,
            int(treatment[:size].sum()),
            size,
            alpha=alpha,
        )
        boundary = float(boundaries.loc[boundaries["look"] == idx, "z_boundary"].iloc[0])
        crossed = abs(result["z_stat"]) >= boundary
        if crossed and stop_look is None:
            stop_look = idx

        rows.append(
            {
                "look": idx,
                "n_per_arm": size,
                "control_rate": result["control_rate"],
                "treatment_rate": result["treatment_rate"],
                "z_stat": result["z_stat"],
                "p_value": result["p_value"],
                "z_boundary": boundary,
                "crossed": crossed,
            }
        )

    return pd.DataFrame(rows), stop_look
