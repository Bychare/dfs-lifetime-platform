"""Module 3: A/B test design, simulation, and monitoring."""

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, callback, dcc, html

from components.ab_testing_helpers import (
    AB_GLOSSARY_MD,
    AB_NOTES_MD,
    SEGMENTS,
    SIM_METRICS,
    ab_frame,
    bootstrap_summary,
    familywise_metric_table,
    pct,
    posterior_figure,
    rate_bar_figure,
    safe_float,
    safe_int,
    sample_size_curve,
    segment_slice,
    sequential_figure,
    simulate_binary_experiment,
)
from components.layout_utils import glossary_accordion, kpi_card, methodological_notes, section_header
from components.stats import (
    beta_binomial_ab_test,
    proportion_z_test,
    sample_size_continuous,
    sample_size_proportions,
    sample_size_survival,
    sequential_proportion_monitor,
)

dash.register_page(__name__, path="/ab-testing", name="A/B Testing", order=2)


layout = html.Div([
    html.H3("Experimentation Engine", className="mb-1"),
    html.P(
        "Planning, simulation, and monitoring of DFS-style product experiments using sample-size design, "
        "frequentist inference, Bayesian posteriors, and sequential stopping boundaries.",
        className="text-muted",
    ),
    glossary_accordion("Glossary: experimentation terms used on this page", AB_GLOSSARY_MD),
    html.Div(id="ab-kpi-row", className="mb-4"),
    section_header(
        "Sample Size Calculator",
        "Lehr-style approximations for binary and continuous KPIs, plus Schoenfeld planning for survival endpoints.",
    ),
    dbc.Row([
        dbc.Col([
            dbc.Label("Design family"),
            dbc.Select(
                id="ab-family",
                options=[
                    {"label": "Binary KPI: conversion / retention", "value": "binary"},
                    {"label": "Continuous metric: spend / revenue", "value": "continuous"},
                    {"label": "Survival / time-to-event", "value": "survival"},
                ],
                value="binary",
            ),
        ], md=3),
        dbc.Col([dbc.Label("Alpha"), dbc.Input(id="ab-alpha", type="number", min=0.001, max=0.2, step=0.005, value=0.05)], md=2),
        dbc.Col([dbc.Label("Power"), dbc.Input(id="ab-power", type="number", min=0.5, max=0.99, step=0.01, value=0.8)], md=2),
        dbc.Col([dbc.Label("Eligible daily traffic"), dbc.Input(id="ab-traffic", type="number", min=10, step=10, value=400)], md=2),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            dbc.Label("Baseline rate"),
            dbc.Input(id="ab-baseline", type="number", min=0.01, max=0.99, step=0.01, value=0.35),
            html.Small("For binary designs: conversion or retention in control.", className="text-muted"),
        ], md=3),
        dbc.Col([
            dbc.Label("MDE"),
            dbc.Input(id="ab-mde", type="number", min=0.001, step=0.005, value=0.03),
            html.Small("For binary designs: absolute uplift in percentage points.", className="text-muted"),
        ], md=3),
        dbc.Col([
            dbc.Label("Sigma"),
            dbc.Input(id="ab-sigma", type="number", min=0.1, step=0.1, value=1.0),
            html.Small("For continuous designs: standard deviation of the KPI.", className="text-muted"),
        ], md=3),
        dbc.Col([
            dbc.Label("Event rate / hazard ratio"),
            dbc.InputGroup([
                dbc.Input(id="ab-event-rate", type="number", min=0.01, max=1, step=0.01, value=0.23),
                dbc.Input(id="ab-hazard-ratio", type="number", min=0.5, max=1.5, step=0.01, value=0.85),
            ]),
            html.Small("For survival designs: observed event rate and target HR.", className="text-muted"),
        ], md=3),
    ], className="mb-3"),
    html.Div(id="ab-sample-size-cards", className="mb-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="ab-sample-size-fig"), md=8),
        dbc.Col(dbc.Alert(id="ab-sample-size-note", color="light", className="py-2 small"), md=4),
    ], className="mb-4"),
    section_header("A/B Simulator", "Synthetic randomized experiments built from the observed DFS cohort."),
    dbc.Row([
        dbc.Col([
            dbc.Label("KPI"),
            dbc.Select(id="ab-metric", options=[{"label": meta["label"], "value": key} for key, meta in SIM_METRICS.items()], value="retained_30d"),
        ], md=3),
        dbc.Col([
            dbc.Label("Segment"),
            dbc.Select(id="ab-segment", options=[{"label": label, "value": key} for key, label in SEGMENTS.items()], value="all"),
        ], md=3),
        dbc.Col([dbc.Label("Sample per arm"), dbc.Input(id="ab-n-per-arm", type="number", min=100, step=100, value=1500)], md=2),
        dbc.Col([dbc.Label("Relative uplift (%)"), dbc.Input(id="ab-uplift", type="number", min=-50, max=100, step=1, value=8)], md=2),
        dbc.Col([dbc.Label("Seed"), dbc.Input(id="ab-seed", type="number", min=1, step=1, value=42)], md=2),
    ], className="mb-3"),
    html.Div(id="ab-sim-cards", className="mb-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="ab-rate-fig"), md=6),
        dbc.Col(dcc.Graph(id="ab-posterior-fig"), md=6),
    ], className="mb-3"),
    dbc.Alert(id="ab-summary", color="info", className="mb-4"),
    section_header("Inference Robustness", "Bootstrap uplift intervals and familywise KPI checks with Holm/BH corrections."),
    dbc.Row([
        dbc.Col(dbc.Alert(id="ab-bootstrap-summary", color="secondary", className="mb-3"), md=4),
        dbc.Col(html.Div(id="ab-multi-table"), md=8),
    ], className="mb-4"),
    section_header("Sequential Testing", "Interim looks with two-sided O'Brien-Fleming stopping boundaries."),
    dbc.Row([
        dbc.Col([dbc.Label("Number of interim looks"), dbc.Input(id="ab-looks", type="number", min=2, max=10, step=1, value=5)], md=2),
        dbc.Col([dbc.Label("Monitoring alpha"), dbc.Input(id="ab-monitor-alpha", type="number", min=0.001, max=0.2, step=0.005, value=0.05)], md=2),
        dbc.Col([
            dbc.Label("Boundary intuition"),
            dbc.Alert(
                "Early looks require much stronger evidence; the threshold relaxes as information accumulates.",
                color="light",
                className="py-2 mb-0",
            ),
        ], md=8),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="ab-seq-fig"), md=7),
        dbc.Col(html.Div(id="ab-seq-table"), md=5),
    ]),
    methodological_notes(AB_NOTES_MD),
])


@callback(Output("ab-kpi-row", "children"), Input("ab-metric", "value"), Input("ab-segment", "value"))
def update_ab_kpis(metric: str, segment: str):
    df = segment_slice(ab_frame(), segment)
    rate = float(df[metric].mean())
    median_fee = float(df["TotFees"].median())
    churn_rate = float(df["is_churned"].mean())
    mean_risk = float(df["RiskScore"].mean())
    return dbc.Row([
        dbc.Col(kpi_card("Eligible players", f"{df.height:,}"), md=True),
        dbc.Col(kpi_card("Observed baseline", pct(rate), "info"), md=True),
        dbc.Col(kpi_card("Median fee", f"${median_fee:,.0f}"), md=True),
        dbc.Col(kpi_card("Churn Rate", pct(churn_rate), "danger"), md=True),
        dbc.Col(kpi_card("Average Risk Score", f"{mean_risk:.1f}", "warning"), md=True),
    ])


@callback(
    Output("ab-sample-size-cards", "children"),
    Output("ab-sample-size-fig", "figure"),
    Output("ab-sample-size-note", "children"),
    Input("ab-family", "value"),
    Input("ab-alpha", "value"),
    Input("ab-power", "value"),
    Input("ab-traffic", "value"),
    Input("ab-baseline", "value"),
    Input("ab-mde", "value"),
    Input("ab-sigma", "value"),
    Input("ab-event-rate", "value"),
    Input("ab-hazard-ratio", "value"),
)
def update_sample_size(family, alpha, power, traffic, baseline, mde, sigma, event_rate, hazard_ratio):
    alpha = safe_float(alpha, 0.05)
    power = safe_float(power, 0.80)
    traffic = max(safe_int(traffic, 400), 1)
    baseline = float(np.clip(safe_float(baseline, 0.35), 0.01, 0.99))
    mde = max(safe_float(mde, 0.03), 0.001)
    sigma = max(safe_float(sigma, 1.0), 0.001)
    event_rate = float(np.clip(safe_float(event_rate, 0.23), 0.01, 1.0))
    hazard_ratio = max(safe_float(hazard_ratio, 0.85), 0.01)

    if family == "binary":
        per_arm = sample_size_proportions(baseline, mde, alpha=alpha, power=power)
        total = per_arm * 2
        primary_value = f"{per_arm:,} per arm"
        assumptions = f"Baseline = {pct(baseline)}, MDE = {mde * 100:.1f} pp, alpha = {alpha:.3f}, power = {power:.0%}."
    elif family == "continuous":
        per_arm = sample_size_continuous(sigma, mde, alpha=alpha, power=power)
        total = per_arm * 2
        primary_value = f"{per_arm:,} per arm"
        assumptions = f"Sigma = {sigma:.2f}, MDE = {mde:.2f}, alpha = {alpha:.3f}, power = {power:.0%}."
    else:
        total = sample_size_survival(hazard_ratio, event_rate, alpha=alpha, power=power)
        primary_value = f"{total:,} total"
        assumptions = f"Event rate = {pct(event_rate)}, HR = {hazard_ratio:.2f}, alpha = {alpha:.3f}, power = {power:.0%}."

    days = total / traffic
    cards = dbc.Row([
        dbc.Col(kpi_card("Required sample", primary_value), md=True),
        dbc.Col(kpi_card("Total volume", f"{total:,} players", "info"), md=True),
        dbc.Col(kpi_card("Runtime at current traffic", f"{days:.1f} days", "warning"), md=True),
        dbc.Col(kpi_card("Significance level", f"{alpha:.1%}", "danger"), md=True),
    ])
    note = [
        html.Div("Notes", className="fw-semibold mb-2"),
        html.P(assumptions, className="mb-2 small"),
        html.Ul([
            html.Li("Binary design: retention / conversion."),
            html.Li("Continuous design: spend / revenue."),
            html.Li("Survival design: time to churn / milestone."),
            html.Li("Smaller effects require larger samples."),
        ], className="small mb-0 ps-3"),
    ]
    return cards, sample_size_curve(family, baseline, mde, sigma, hazard_ratio, event_rate, alpha, power), note


@callback(
    Output("ab-sim-cards", "children"),
    Output("ab-rate-fig", "figure"),
    Output("ab-posterior-fig", "figure"),
    Output("ab-summary", "children"),
    Output("ab-bootstrap-summary", "children"),
    Output("ab-multi-table", "children"),
    Output("ab-seq-fig", "figure"),
    Output("ab-seq-table", "children"),
    Input("ab-metric", "value"),
    Input("ab-segment", "value"),
    Input("ab-n-per-arm", "value"),
    Input("ab-uplift", "value"),
    Input("ab-seed", "value"),
    Input("ab-looks", "value"),
    Input("ab-monitor-alpha", "value"),
)
def update_simulation(metric, segment, n_per_arm, uplift, seed, looks, monitor_alpha):
    n_per_arm = max(safe_int(n_per_arm, 1500), 50)
    uplift = safe_float(uplift, 8.0)
    seed = safe_int(seed, 42)
    looks = min(max(safe_int(looks, 5), 2), 10)
    monitor_alpha = safe_float(monitor_alpha, 0.05)

    df = segment_slice(ab_frame(), segment)
    values = df[metric]
    control, treatment, baseline, target = simulate_binary_experiment(values, n_per_arm=n_per_arm, uplift_pct=uplift, seed=seed)

    frequentist = proportion_z_test(int(control.sum()), len(control), int(treatment.sum()), len(treatment), alpha=monitor_alpha)
    bayes = beta_binomial_ab_test(int(control.sum()), len(control), int(treatment.sum()), len(treatment), seed=seed)
    boot = bootstrap_summary(control, treatment, seed=seed)
    family_df = familywise_metric_table(df, n_per_arm=n_per_arm, uplift_pct=uplift, alpha=monitor_alpha, seed=seed)
    seq_df, stop_look = sequential_proportion_monitor(control, treatment, n_looks=looks, alpha=monitor_alpha)

    cards = dbc.Row([
        dbc.Col(kpi_card("Target baseline", pct(baseline), "info"), md=True),
        dbc.Col(kpi_card("Injected uplift", f"{uplift:.1f}%", "warning"), md=True),
        dbc.Col(kpi_card("Observed effect", f"{frequentist['absolute_diff'] * 100:.2f} pp"), md=True),
        dbc.Col(kpi_card("Frequentist p-value", f"{frequentist['p_value']:.4f}", "danger"), md=True),
        dbc.Col(kpi_card("P(Treat > Ctrl)", pct(bayes["prob_treatment_beats_control"]), "success"), md=True),
    ])

    summary = (
        f"{SIM_METRICS[metric]['label']} in {SEGMENTS[segment]}: control observes {pct(frequentist['control_rate'])}, "
        f"treatment observes {pct(frequentist['treatment_rate'])}. The injected target rate was {pct(target)}. "
        f"Frequentist 95% CI for uplift: {frequentist['ci_low'] * 100:.2f} to {frequentist['ci_high'] * 100:.2f} pp. "
        f"Bootstrap 95% CI: {boot['ci_low'] * 100:.2f} to {boot['ci_high'] * 100:.2f} pp. "
        f"The Bayesian posterior puts P(Treat > Ctrl) at {pct(bayes['prob_treatment_beats_control'])}. "
        f"{'Sequential monitoring crosses the boundary at look #' + str(stop_look) + '.' if stop_look else 'In this run, no OBF boundary was crossed.'}"
    )

    bootstrap_note = [
        html.Div("Bootstrap 95% CI", className="fw-semibold mb-2"),
        html.P(f"Robust uplift estimate: {boot['ci_low'] * 100:.2f} to {boot['ci_high'] * 100:.2f} pp.", className="mb-2"),
        html.P(
            f"Point estimate = {boot['point_estimate'] * 100:.2f} pp, bootstrap SE = {boot['std_error'] * 100:.2f} pp.",
            className="mb-0 small",
        ),
    ]

    def correction_badge(row):
        if row["holm_significant"]:
            return dbc.Badge("✓ Holm", color="success")
        if row["bh_significant"]:
            return dbc.Badge("✓ BH", color="info")
        return dbc.Badge("n.s.", color="secondary")

    multi_table = dbc.Table(
        [
            html.Thead(html.Tr([html.Th("KPI"), html.Th("Ctrl"), html.Th("Treat"), html.Th("Uplift"), html.Th("p (raw)"), html.Th("p (Holm)"), html.Th("p (BH)"), html.Th("Decision")])),
            html.Tbody([
                html.Tr([
                    html.Td(row["label"]),
                    html.Td(pct(row["control_rate"])),
                    html.Td(pct(row["treatment_rate"])),
                    html.Td(f"{row['absolute_diff'] * 100:.2f} pp"),
                    html.Td(f"{row['p_raw']:.4f}"),
                    html.Td(f"{row['p_holm']:.4f}"),
                    html.Td(f"{row['p_bh']:.4f}"),
                    html.Td(correction_badge(row)),
                ])
                for row in family_df.to_dicts()
            ]),
        ],
        bordered=True,
        striped=True,
        hover=True,
        size="sm",
        className="small",
    )

    seq_table = dbc.Table(
        [
            html.Thead(html.Tr([html.Th("Look"), html.Th("n / arm"), html.Th("Ctrl"), html.Th("Treat"), html.Th("z"), html.Th("Boundary"), html.Th("Stop?")])),
            html.Tbody([
                html.Tr([
                    html.Td(int(row["look"])),
                    html.Td(f"{int(row['n_per_arm']):,}"),
                    html.Td(pct(row["control_rate"])),
                    html.Td(pct(row["treatment_rate"])),
                    html.Td(f"{row['z_stat']:.2f}"),
                    html.Td(f"{row['z_boundary']:.2f}"),
                    html.Td(dbc.Badge("Crossed" if row["crossed"] else "Continue", color="success" if row["crossed"] else "secondary")),
                ])
                for row in seq_df.to_dicts()
            ]),
        ],
        bordered=True,
        striped=True,
        hover=True,
        size="sm",
        className="small",
    )

    return (
        cards,
        rate_bar_figure(control, treatment, alpha=monitor_alpha),
        posterior_figure(bayes["diff_draws"]),
        summary,
        bootstrap_note,
        multi_table,
        sequential_figure(seq_df),
        seq_table,
    )
