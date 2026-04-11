"""Basic tests for statistical helpers and data pipeline."""

import numpy as np
import pandas as pd
import pytest
from app.components.stats import (
    sample_size_proportions,
    sample_size_continuous,
    sample_size_survival,
    normality_test,
    proportion_z_test,
    beta_binomial_ab_test,
    obrien_fleming_bounds,
)


def test_sample_size_proportions_known_case():
    """With p0=0.5, MDE=0.05, alpha=0.05, power=0.80 => ~1565 per group."""
    n = sample_size_proportions(p0=0.5, mde=0.05, alpha=0.05, power=0.80)
    assert 1500 < n < 1700


def test_sample_size_continuous_known_case():
    """sigma=1, MDE=0.2 => ~393 per group (Cohen's d=0.2, small effect)."""
    n = sample_size_continuous(sigma=1.0, mde=0.2, alpha=0.05, power=0.80)
    assert 380 < n < 420


def test_normality_test_normal():
    rng = np.random.default_rng(42)
    data = pd.Series(rng.normal(0, 1, 500))
    result = normality_test(data)
    assert result["normal"] is True


def test_normality_test_skewed():
    rng = np.random.default_rng(42)
    data = pd.Series(rng.exponential(1, 500))
    result = normality_test(data)
    assert result["normal"] is False


def test_data_loader_import():
    """Verify the data loader module is importable."""
    from app.components.data_loader import RAW_DIR, CONTEST_TYPE_NAMES
    assert len(CONTEST_TYPE_NAMES) == 6


def test_sample_size_survival_increases_near_null():
    n_strong = sample_size_survival(hazard_ratio=0.75, event_rate=0.25)
    n_weak = sample_size_survival(hazard_ratio=0.9, event_rate=0.25)
    assert n_weak > n_strong > 0


def test_proportion_z_test_detects_difference():
    result = proportion_z_test(
        control_success=120,
        control_total=1000,
        treatment_success=170,
        treatment_total=1000,
    )
    assert result["treatment_rate"] > result["control_rate"]
    assert result["p_value"] < 0.01
    assert result["significant"] is True


def test_beta_binomial_ab_test_prefers_better_arm():
    result = beta_binomial_ab_test(
        control_success=120,
        control_total=1000,
        treatment_success=170,
        treatment_total=1000,
        draws=5000,
        seed=7,
    )
    assert result["prob_treatment_beats_control"] > 0.95
    assert result["ci_high"] > result["ci_low"]


def test_obrien_fleming_bounds_decrease_over_time():
    bounds = obrien_fleming_bounds(n_looks=5, alpha=0.05)
    assert bounds["z_boundary"].iloc[0] > bounds["z_boundary"].iloc[-1]
