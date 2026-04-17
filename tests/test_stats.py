"""Basic tests for statistical helpers and data pipeline."""

import numpy as np
import polars as pl

from app.components.stats import (
    adjust_pvalues,
    beta_binomial_ab_test,
    bootstrap_uplift_ci,
    dunn_posthoc,
    normality_test,
    obrien_fleming_bounds,
    proportion_z_test,
    sample_size_continuous,
    sample_size_proportions,
    sample_size_survival,
    two_way_anova,
)


def test_sample_size_proportions_known_case():
    n = sample_size_proportions(p0=0.5, mde=0.05, alpha=0.05, power=0.80)
    assert 1500 < n < 1700


def test_sample_size_continuous_known_case():
    n = sample_size_continuous(sigma=1.0, mde=0.2, alpha=0.05, power=0.80)
    assert 380 < n < 420


def test_normality_test_normal():
    rng = np.random.default_rng(42)
    data = pl.Series(rng.normal(0, 1, 500))
    result = normality_test(data)
    assert result["normal"] is True


def test_normality_test_skewed():
    rng = np.random.default_rng(42)
    data = pl.Series(rng.exponential(1, 500))
    result = normality_test(data)
    assert result["normal"] is False


def test_data_loader_import():
    from app.components.data_loader import CONTEST_TYPE_NAMES
    assert len(CONTEST_TYPE_NAMES) == 6


def test_sample_size_survival_increases_near_null():
    n_strong = sample_size_survival(hazard_ratio=0.75, event_rate=0.25)
    n_weak = sample_size_survival(hazard_ratio=0.9, event_rate=0.25)
    assert n_weak > n_strong > 0


def test_proportion_z_test_detects_difference():
    result = proportion_z_test(120, 1000, 170, 1000)
    assert result["treatment_rate"] > result["control_rate"]
    assert result["p_value"] < 0.01
    assert result["significant"] is True


def test_beta_binomial_ab_test_prefers_better_arm():
    result = beta_binomial_ab_test(120, 1000, 170, 1000, draws=5000, seed=7)
    assert result["prob_treatment_beats_control"] > 0.95
    assert result["ci_high"] > result["ci_low"]


def test_obrien_fleming_bounds_decrease_over_time():
    bounds = obrien_fleming_bounds(n_looks=5, alpha=0.05)
    z_bounds = bounds["z_boundary"].to_list()
    assert z_bounds[0] > z_bounds[-1]


def test_bootstrap_uplift_ci_detects_positive_effect():
    control = np.array([0, 0, 1, 0, 1, 0, 0, 1] * 50)
    treatment = np.array([1, 1, 1, 0, 1, 1, 0, 1] * 50)
    result = bootstrap_uplift_ci(control, treatment, n_boot=1000, seed=11)
    assert result["point_estimate"] > 0
    assert result["ci_high"] > result["ci_low"]
    assert result["ci_low"] > -0.05


def test_adjust_pvalues_holm_and_bh_are_monotone_and_bounded():
    p_values = np.array([0.004, 0.02, 0.03, 0.2])
    holm = adjust_pvalues(p_values, method="holm")
    bh = adjust_pvalues(p_values, method="bh")
    assert np.all((holm >= 0) & (holm <= 1))
    assert np.all((bh >= 0) & (bh <= 1))
    assert holm[0] <= holm[-1]
    assert bh[0] <= bh[-1]


def test_dunn_posthoc_returns_square_pairwise_matrix():
    df = pl.DataFrame(
        {
            "segment": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
            "metric": [1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3, 4.0, 4.1, 4.2, 4.3],
        }
    )
    result = dunn_posthoc(df, "metric", "segment")
    assert result.shape == (3, 4)
    assert result.columns[0] == "segment"


def test_two_way_anova_returns_effect_table_and_cell_means():
    df = pl.DataFrame(
        {
            "response": [1.0, 1.2, 1.1, 1.3, 2.2, 2.3, 2.4, 2.5],
            "factor_a": ["Q1", "Q1", "Q1", "Q1", "Q4", "Q4", "Q4", "Q4"],
            "factor_b": ["NFL", "NFL", "Multi", "Multi", "NFL", "NFL", "Multi", "Multi"],
        }
    )
    anova_df, means_df = two_way_anova(df, "response", "factor_a", "factor_b")
    assert "term" in anova_df.columns
    assert "partial_eta_sq" in anova_df.columns
    assert means_df.height == 4
