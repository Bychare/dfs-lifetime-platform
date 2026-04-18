"""Module 5: Leakage-free churn modeling."""

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, callback, dcc, html

from components.churn_helpers import (
    CHURN_GLOSSARY_MD,
    CHURN_NOTES_MD,
    calibration_figure,
    churn_artifacts,
    decile_lift_figure,
    driver_effect_figure,
    driver_effect_table,
    importance_figure,
    metrics_summary_text,
    metrics_table,
    pr_curve_figure,
    roc_figure,
    score_profile,
    what_if_summary_text,
)
from components.layout_utils import glossary_accordion, kpi_card, methodological_notes, section_header

dash.register_page(__name__, path="/churn-model", name="Churn Model", order=4)


def _metrics_display_table():
    pdf = metrics_table().copy()
    for col in ["ROC-AUC", "PR-AUC"]:
        pdf[col] = pdf[col].map(lambda value: f"{value:.3f}")
    pdf["Brier"] = pdf["Brier"].map(lambda value: f"{value:.3f}")
    pdf["Baseline churn"] = pdf["Baseline churn"].map(lambda value: f"{value:.1%}")
    pdf["Top-decile churn"] = pdf["Top-decile churn"].map(lambda value: f"{value:.1%}")
    pdf["Top-decile lift"] = pdf["Top-decile lift"].map(lambda value: f"{value:.2f}x")
    return dbc.Table.from_dataframe(pdf, striped=True, bordered=True, hover=True, size="sm", className="small")


def _driver_display_table():
    effect_df = driver_effect_table().copy()
    effect_df["LogReg delta pp"] = effect_df["LogReg delta pp"].map(lambda value: f"{value:+.1f} pp")
    effect_df["CatBoost delta pp"] = effect_df["CatBoost delta pp"].map(lambda value: f"{value:+.1f} pp")
    return dbc.Table.from_dataframe(effect_df, striped=True, bordered=True, hover=True, size="sm", className="small")


layout = html.Div([
    dcc.Store(id="churn-init", data="ready"),
    html.H3("Churn Prediction", className="mb-1"),
    html.P(
        "Leakage-free churn scoring built only from information available around the player's first contest: "
        "RiskScore, age availability, age, and signup timing.",
        className="text-muted",
    ),
    glossary_accordion("Glossary: churn modeling concepts used on this page", CHURN_GLOSSARY_MD),
    html.Div(id="churn-kpi-row", className="mb-4"),
    dbc.Alert(
        "This module deliberately uses a weaker but decision-usable setup. It avoids full-season behavioral features that would leak future information into the prediction problem.",
        color="light",
        className="small mb-4",
    ),
    section_header(
        "Model Quality",
        "Compare ranking quality, probability calibration, and risk concentration across the two leakage-free models.",
    ),
    dbc.Row([
        dbc.Col(dcc.Graph(id="churn-roc-fig"), md=6),
        dbc.Col(dcc.Graph(id="churn-pr-fig"), md=6),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="churn-calibration-fig"), md=6),
        dbc.Col(dcc.Graph(id="churn-decile-fig"), md=6),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(html.Div(id="churn-metrics-table"), md=7),
        dbc.Col(dbc.Alert(id="churn-quality-summary", color="info", className="small"), md=5),
    ], className="mb-4"),
    section_header(
        "Driver Analysis",
        "Use scenario deltas and model importance to understand which leakage-free features move churn risk the most.",
    ),
    dbc.Row([
        dbc.Col(dcc.Graph(id="churn-driver-fig"), md=7),
        dbc.Col(dcc.Graph(id="churn-importance-fig"), md=5),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(html.Div(id="churn-driver-table"), md=12),
    ], className="mb-4"),
    section_header(
        "What-If Calculator",
        "Estimate churn risk for a hypothetical player profile using only leakage-free inputs.",
    ),
    dbc.Row([
        dbc.Col([
            dbc.Label("RiskScore"),
            dcc.Slider(
                id="whatif-risk-score",
                min=0,
                max=80,
                step=1,
                value=10,
                marks={0: "0", 5: "5", 10: "10", 20: "20", 40: "40", 60: "60", 80: "80"},
                tooltip={"placement": "bottom"},
            ),
        ], md=4),
        dbc.Col([
            dbc.Label("Age"),
            dbc.Input(id="whatif-age", type="number", min=18, max=80, step=1, value=30),
            dbc.Checklist(
                id="whatif-age-missing",
                options=[{"label": " Age unavailable", "value": "missing"}],
                value=[],
                switch=True,
                className="mt-2",
            ),
        ], md=3),
        dbc.Col([
            dbc.Label("Signup timing"),
            dcc.Slider(
                id="whatif-cohort-day",
                min=0,
                max=150,
                step=1,
                value=30,
                marks={0: "Aug 22", 30: "Sep 21", 60: "Oct 21", 90: "Nov 20", 120: "Dec 20", 150: "Jan"},
                tooltip={"placement": "bottom"},
            ),
        ], md=5),
    ], className="mb-3"),
    html.Div(id="whatif-kpi-row", className="mb-3"),
    dbc.Alert(id="whatif-summary", color="secondary", className="small mb-4"),
    methodological_notes(CHURN_NOTES_MD),
])


@callback(Output("churn-kpi-row", "children"), Input("churn-init", "data"))
def update_churn_kpis(_):
    bundle = churn_artifacts()
    metrics = bundle["metrics"].sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    best = metrics.iloc[0]
    frame = bundle["frame"]
    return dbc.Row([
        dbc.Col(kpi_card("Players", f"{len(frame):,}"), md=True),
        dbc.Col(kpi_card("Observed churn", f"{bundle['baseline_churn']:.1%}", "danger"), md=True),
        dbc.Col(kpi_card("Best ROC-AUC", f"{best['ROC-AUC']:.3f}", "info"), md=True),
        dbc.Col(kpi_card("Best PR-AUC", f"{best['PR-AUC']:.3f}", "warning"), md=True),
        dbc.Col(kpi_card("Top-decile lift", f"{best['Top-decile lift']:.2f}x", "success"), md=True),
    ])


@callback(
    Output("churn-roc-fig", "figure"),
    Output("churn-pr-fig", "figure"),
    Output("churn-calibration-fig", "figure"),
    Output("churn-decile-fig", "figure"),
    Output("churn-metrics-table", "children"),
    Output("churn-quality-summary", "children"),
    Input("churn-init", "data"),
)
def update_churn_quality(_):
    return (
        roc_figure(),
        pr_curve_figure(),
        calibration_figure(),
        decile_lift_figure(),
        _metrics_display_table(),
        metrics_summary_text(),
    )


@callback(
    Output("churn-driver-fig", "figure"),
    Output("churn-importance-fig", "figure"),
    Output("churn-driver-table", "children"),
    Input("churn-init", "data"),
)
def update_churn_drivers(_):
    return driver_effect_figure(), importance_figure(), _driver_display_table()


@callback(
    Output("whatif-kpi-row", "children"),
    Output("whatif-summary", "children"),
    Input("whatif-risk-score", "value"),
    Input("whatif-age", "value"),
    Input("whatif-age-missing", "value"),
    Input("whatif-cohort-day", "value"),
)
def update_what_if(risk_score, age, age_missing, cohort_day):
    has_age = "missing" not in (age_missing or [])
    scores = score_profile(
        risk_score=float(risk_score or 10),
        age=float(age) if age is not None else None,
        has_age=has_age,
        cohort_day=float(cohort_day or 30),
    )

    avg_risk = (scores["Logistic Regression"] + scores["CatBoost"]) / 2
    timing_label = (pd.Timestamp("2014-08-22") + pd.Timedelta(days=float(cohort_day or 30))).strftime("%b %d")

    cards = dbc.Row([
        dbc.Col(kpi_card("Logistic Regression", f"{scores['Logistic Regression']:.1%}", "info"), md=True),
        dbc.Col(kpi_card("CatBoost", f"{scores['CatBoost']:.1%}", "warning"), md=True),
        dbc.Col(kpi_card("Average scenario risk", f"{avg_risk:.1%}", "danger"), md=True),
        dbc.Col(kpi_card("Baseline churn", f"{scores['Baseline']:.1%}", "success"), md=True),
        dbc.Col(kpi_card("Signup date proxy", timing_label), md=True),
    ])
    return cards, what_if_summary_text(scores)
