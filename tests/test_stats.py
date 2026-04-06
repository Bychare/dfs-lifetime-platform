"""Basic tests for statistical helpers and data pipeline."""

import numpy as np
import pandas as pd
import pytest
from app.components.stats import (
    sample_size_proportions,
    sample_size_continuous,
    normality_test,
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
