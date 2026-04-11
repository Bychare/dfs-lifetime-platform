"""Smoke tests for page imports and helper-driven figure generation."""

from pathlib import Path
import importlib
import sys

import dash
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def _import_page(module_name: str, monkeypatch):
    monkeypatch.setattr(dash, "register_page", lambda *args, **kwargs: None)
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_page_modules_import_without_server(monkeypatch):
    pytest.importorskip("scipy")
    pytest.importorskip("lifelines")

    overview = _import_page("pages.overview", monkeypatch)
    survival = _import_page("pages.survival", monkeypatch)
    ab_testing = _import_page("pages.ab_testing", monkeypatch)

    assert overview.layout is not None
    assert survival.layout is not None
    assert ab_testing.layout is not None


def test_helper_figures_render_with_minimal_inputs():
    pytest.importorskip("scipy")
    pytest.importorskip("lifelines")

    from app.components.ab_testing_helpers import posterior_figure, rate_bar_figure, sequential_figure
    from app.components.survival_helpers import (
        build_milestone_data,
        display_group_frame,
        km_figure,
        km_milestone_figure,
    )

    survival_df = pd.DataFrame(
        {
            "UserID": [1, 2, 3, 4],
            "duration_days": [12, 30, 45, 60],
            "is_churned": [1, 0, 1, 0],
            "is_multisport": [0, 1, 0, 1],
            "n_sports": [1, 2, 1, 2],
            "nCont": [3, 12, 8, 20],
            "nDays": [3, 10, 6, 14],
        }
    )
    plot_df = display_group_frame(survival_df, "is_multisport")
    km_fig = km_figure(plot_df, "is_multisport", "Smoke KM")
    milestone_df = build_milestone_data(survival_df, 5)
    milestone_fig = km_milestone_figure(plot_df, milestone_df, "is_multisport", 5)

    seq_df = pd.DataFrame(
        {
            "look": [1, 2, 3],
            "z_stat": [0.4, 1.2, 2.3],
            "z_boundary": [3.0, 2.4, 2.0],
        }
    )
    rate_fig = rate_bar_figure(np.array([0, 1, 0, 1]), np.array([1, 1, 1, 0]), alpha=0.05)
    posterior_fig = posterior_figure(np.array([-0.02, 0.01, 0.03, 0.015]))
    seq_fig = sequential_figure(seq_df)

    assert len(km_fig.data) > 0
    assert len(milestone_fig.data) > 0
    assert len(rate_fig.data) == 1
    assert len(posterior_fig.data) > 0
    assert len(seq_fig.data) == 3
