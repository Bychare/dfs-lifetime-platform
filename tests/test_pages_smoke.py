"""Smoke tests for page imports and helper-driven figure generation."""

from pathlib import Path
import importlib
import sys

import dash
import numpy as np
import pandas as pd
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def _import_page(module_name: str, monkeypatch):
    monkeypatch.setattr(dash, "register_page", lambda *args, **kwargs: None)
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def _import_app_entrypoint():
    sys.modules.pop("app.app", None)
    return importlib.import_module("app.app")


def test_page_modules_import_without_server(monkeypatch):
    pytest.importorskip("scipy")
    pytest.importorskip("lifelines")
    pytest.importorskip("statsmodels")
    pytest.importorskip("scikit_posthocs")
    pytest.importorskip("catboost")

    overview = _import_page("pages.overview", monkeypatch)
    survival = _import_page("pages.survival", monkeypatch)
    ab_testing = _import_page("pages.ab_testing", monkeypatch)
    segmentation = _import_page("pages.segmentation", monkeypatch)
    churn_model = _import_page("pages.churn_model", monkeypatch)

    assert overview.layout is not None
    assert survival.layout is not None
    assert ab_testing.layout is not None
    assert segmentation.layout is not None
    assert churn_model.layout is not None


def test_main_app_entrypoint_imports():
    app_module = _import_app_entrypoint()

    assert app_module.app is not None
    assert app_module.server is not None
    assert app_module.app.layout is not None
    assert app_module.app.title == "DFS Analytics Platform"


def test_helper_figures_render_with_minimal_inputs():
    pytest.importorskip("scipy")
    pytest.importorskip("lifelines")
    pytest.importorskip("statsmodels")
    pytest.importorskip("scikit_posthocs")
    pytest.importorskip("catboost")

    from app.components.ab_testing_helpers import posterior_figure, rate_bar_figure, sequential_figure
    from app.components.churn_helpers import (
        calibration_figure,
        decile_lift_figure,
        driver_effect_figure,
        driver_effect_table,
        fit_churn_models,
        importance_figure,
        pr_curve_figure,
        roc_figure,
    )
    from app.components.segmentation_helpers import (
        segment_footprint_figure,
        segment_footprint_table,
        segment_profile_heatmap,
        anova_heatmap_figure,
        dunn_heatmap_figure,
        display_segment_frame,
        risk_lowess_figure,
        segment_box_figure,
        segment_rank_figure,
        segment_summary_table,
    )
    from app.components.survival_helpers import build_milestone_data, display_group_frame, km_figure, km_milestone_figure

    survival_df = pl.DataFrame(
        {
            "UserID": [1, 2, 3, 4],
            "duration_days": [12, 30, 45, 60],
            "is_churned": [1, 0, 1, 0],
            "is_multisport": [0, 1, 0, 1],
            "n_sports": [1, 2, 1, 2],
            "nCont": [3, 12, 8, 20],
            "nDays": [3, 10, 6, 14],
            "Date1st": [None, None, None, None],
            "DateLst": [None, None, None, None],
        }
    )
    plot_df = display_group_frame(survival_df, "is_multisport")
    km_fig = km_figure(plot_df, "is_multisport", "Smoke KM")
    milestone_df = build_milestone_data(survival_df, 5)
    milestone_fig = km_milestone_figure(plot_df, milestone_df, "is_multisport", 5)

    seq_df = pl.DataFrame({"look": [1, 2, 3], "z_stat": [0.4, 1.2, 2.3], "z_boundary": [3.0, 2.4, 2.0]})
    rate_fig = rate_bar_figure(np.array([0, 1, 0, 1]), np.array([1, 1, 1, 0]), alpha=0.05)
    posterior_fig = posterior_figure(np.array([-0.02, 0.01, 0.03, 0.015]))
    seq_fig = sequential_figure(seq_df)

    segmentation_df = pl.DataFrame(
        {
            "TotFees": [10.0, 12.0, 40.0, 42.0, 22.0, 24.0],
            "RiskScore": [2.0, 3.0, 8.0, 9.0, 5.0, 6.0],
            "is_churned": [0, 0, 1, 1, 0, 1],
            "is_multisport": [0, 0, 1, 1, 0, 1],
            "dominant_type": ["A", "A", "B", "B", "C", "C"],
            "risk_quartile": ["Q1", "Q1", "Q4", "Q4", "Q2", "Q2"],
            "buyin_quartile": ["Q1", "Q1", "Q4", "Q4", "Q2", "Q2"],
            "age_group": ["18-24", "18-24", "35-44", "35-44", "25-29", "25-29"],
            "log_total_fees": np.log1p([10.0, 12.0, 40.0, 42.0, 22.0, 24.0]),
            "win_rate": [0.1, 0.2, 0.4, 0.45, 0.25, 0.3],
            "intensity": [1.0, 1.1, 2.5, 2.7, 1.8, 1.9],
            "type_diversity": [0.2, 0.25, 0.8, 0.82, 0.45, 0.5],
        }
    )
    seg_plot_df = display_segment_frame(segmentation_df, "dominant_type")
    seg_summary = segment_summary_table(seg_plot_df, "dominant_type", "TotFees")
    seg_footprint = segment_footprint_table(
        segmentation_df.with_columns(pl.Series("duration_days", [5, 7, 15, 17, 11, 12])),
        "dominant_type",
    )
    seg_box_fig = segment_box_figure(seg_plot_df, "TotFees", "dominant_type")
    seg_rank_fig = segment_rank_figure(seg_summary, "TotFees", "dominant_type")
    seg_footprint_fig = segment_footprint_figure(seg_footprint, "dominant_type")
    seg_profile_fig = segment_profile_heatmap(seg_footprint, "dominant_type")
    dunn_fig = dunn_heatmap_figure(
        pl.DataFrame(
            {
                "dominant_type": ["A", "B", "C"],
                "A": [1.0, 0.03, 0.2],
                "B": [0.03, 1.0, 0.04],
                "C": [0.2, 0.04, 1.0],
            }
        ),
        "dominant_type",
    )
    anova_fig = anova_heatmap_figure(
        pl.DataFrame(
            {
                "risk_quartile": ["Q1", "Q1", "Q4", "Q4"],
                "is_multisport": ["NFL only", "Multi-sport", "NFL only", "Multi-sport"],
                "mean": [2.0, 2.3, 3.2, 3.5],
            }
        ),
        "risk_quartile",
        "is_multisport",
        "log_total_fees",
    )
    lowess_fig = risk_lowess_figure(display_segment_frame(segmentation_df, "is_multisport"), "is_churned", "is_multisport")

    rng = np.random.default_rng(7)
    n = 120
    risk = rng.uniform(1, 35, n)
    age = rng.normal(32, 7, n)
    has_age = rng.binomial(1, 0.55, n)
    age[has_age == 0] = np.nan
    cohort_day = rng.integers(0, 120, n)
    logits = -2.3 + 0.06 * risk + 0.012 * cohort_day - 0.03 * np.nan_to_num(age, nan=31) - 0.45 * has_age
    probs = 1 / (1 + np.exp(-logits))
    target = rng.binomial(1, probs)
    churn_frame = pd.DataFrame(
        {
            "UserID": np.arange(n),
            "Date1st": pd.Timestamp("2014-08-22") + pd.to_timedelta(cohort_day, unit="D"),
            "target": target,
            "RiskScore": risk,
            "age": age,
            "has_age": has_age.astype(float),
            "cohort_day": cohort_day.astype(float),
        }
    )
    churn_bundle = fit_churn_models(churn_frame, random_state=5, catboost_iterations=40)
    churn_roc_fig = roc_figure(churn_bundle)
    churn_pr_fig = pr_curve_figure(churn_bundle)
    churn_cal_fig = calibration_figure(churn_bundle)
    churn_decile_fig = decile_lift_figure(churn_bundle)
    churn_driver_fig = driver_effect_figure(driver_effect_table(churn_bundle))
    churn_importance_fig = importance_figure(churn_bundle)

    assert len(km_fig.data) > 0
    assert len(milestone_fig.data) > 0
    assert len(rate_fig.data) == 1
    assert len(posterior_fig.data) > 0
    assert len(seq_fig.data) == 3
    assert len(seg_box_fig.data) > 0
    assert len(seg_rank_fig.data) == 1
    assert len(seg_footprint_fig.data) >= 2
    assert len(seg_profile_fig.data) == 1
    assert len(dunn_fig.data) == 1
    assert len(anova_fig.data) == 1
    assert len(lowess_fig.data) >= 2
    assert len(churn_roc_fig.data) >= 2
    assert len(churn_pr_fig.data) >= 2
    assert len(churn_cal_fig.data) >= 2
    assert len(churn_decile_fig.data) >= 2
    assert len(churn_driver_fig.data) == 2
    assert len(churn_importance_fig.data) == 1
