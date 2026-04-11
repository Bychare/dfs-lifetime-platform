"""
Module 2: Survival Analysis — Kaplan-Meier, Cox PH, Cohort Retention.

Applies survival analysis methods to DFS player churn:
- Right-censored Kaplan-Meier curves with log-rank tests
- Cox Proportional Hazards with forest plot and Schoenfeld diagnostics
- Weekly cohort retention heatmap
"""

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import html, dcc, callback, Input, Output
from lifelines import KaplanMeierFitter

from components.data_loader import get_players, NFL_SEASON_END
from components.layout_utils import (
    glossary_accordion,
    kpi_card,
    methodological_notes,
    section_header,
)
from components.survival_helpers import (
    MILESTONE_GROUP_OPTIONS,
    SURVIVAL_GLOSSARY_MD,
    SURVIVAL_GROUP_OPTIONS,
    SURVIVAL_NOTES_MD,
    build_milestone_data,
    cohort_retention_heatmap,
    display_group_frame,
    fit_cox,
    forest_plot,
    km_figure,
    km_milestone_figure,
    logrank_summary,
    median_survival_table,
    milestone_summary,
    schoenfeld_figure,
)

dash.register_page(__name__, path="/survival", name="Survival Analysis", order=1)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

layout = html.Div([
    html.H3("Survival Analysis", className="mb-1"),
    html.P(
        "Right-censored analysis of player retention using Kaplan-Meier, "
        "Cox Proportional Hazards, and cohort retention. "
        "77% of players are censored (active at NFL season end).",
        className="text-muted",
    ),

    glossary_accordion(
        "Glossary — abbreviations and symbols used on this page",
        SURVIVAL_GLOSSARY_MD,
    ),

    html.Div(id="surv-kpi-row", className="mb-4"),

    # --- Kaplan-Meier ---
    section_header(
        "Kaplan-Meier Survival Curves",
        "Retention probability over time with 95% Greenwood CI. "
        "Select a grouping variable for stratified comparison + log-rank test.",
    ),

    dbc.Row([
        dbc.Col([
            dbc.Label("Group by"),
            dbc.Select(
                id="km-group-select",
                options=SURVIVAL_GROUP_OPTIONS,
                value="none",
            ),
        ], md=3),
        dbc.Col([
            dbc.Label("Y-axis zoom"),
            dcc.Slider(
                id="km-ymin-slider",
                min=0, max=0.9, step=0.1, value=0,
                marks={0: "0", 0.5: "0.5", 0.7: "0.7", 0.9: "0.9"},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ], md=3),
        dbc.Col([
            dbc.Label("Display"),
            dbc.Checklist(
                id="km-ci-toggle",
                options=[{"label": " Show 95% CI bands", "value": "ci"}],
                value=["ci"],
                switch=True,
                className="mt-1",
            ),
        ], md=2),
        dbc.Col([
            dbc.Label("Show groups"),
            dbc.Checklist(
                id="km-group-filter",
                options=[],
                value=[],
                className="small",
                style={"maxHeight": "120px", "overflowY": "auto"},
            ),
        ], md=4),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="km-plot"), md=8),
        dbc.Col([
            html.Div(id="km-median-table"),
            html.Hr(),
            html.Div(id="km-logrank-text"),
        ], md=4),
    ]),

    # --- Cox PH ---
    section_header(
        "Cox Proportional Hazards Model",
        "Hazard ratios for churn risk factors. HR > 1 = higher churn risk. "
        "Covariates: log(AvgBuyIn), RiskScore, multi-sport, win rate, intensity, type diversity.",
    ),

    dbc.Row([
        dbc.Col(dcc.Graph(id="forest-plot"), md=7),
        dbc.Col(html.Div(id="cox-summary-table"), md=5),
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id="schoenfeld-table"), md=7),
        dbc.Col(dcc.Markdown(id="cox-interpretation", className="small mt-3"), md=5),
    ], className="mt-2"),

    # --- Cohort Retention ---
    section_header(
        "Cohort Retention Heatmap",
        "Weekly retention by cohort (grouped by week of first contest).",
    ),

    dbc.Row([
        dbc.Col(dcc.Graph(id="retention-heatmap"), md=12),
    ]),

    # --- Alternative Endpoint ---
    section_header(
        "Alternative Endpoint: Time to Nth Contest",
        "Instead of 'time to churn' (77% censored, medians not estimable), "
        "we analyze 'time to reaching N contests' — an engagement milestone. "
        "Switching endpoints increases the event rate and makes group "
        "comparisons more powerful.",
    ),

    dbc.Row([
        dbc.Col([
            dbc.Label("Contest milestone (N)"),
            dcc.Slider(
                id="milestone-n-slider",
                min=5, max=50, step=5, value=10,
                marks={n: str(n) for n in range(5, 55, 5)},
                tooltip={"placement": "bottom"},
            ),
        ], md=4),
        dbc.Col([
            dbc.Label("Group by"),
            dbc.Select(
                id="milestone-group-select",
                options=MILESTONE_GROUP_OPTIONS,
                value="none",
            ),
        ], md=4),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="milestone-km-plot"), md=8),
        dbc.Col(dcc.Markdown(id="milestone-summary-text", className="small"), md=4),
    ]),

    methodological_notes(SURVIVAL_NOTES_MD),
])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(Output("surv-kpi-row", "children"), Input("km-group-select", "value"))
def update_surv_kpis(_):
    df = get_players()
    kmf = KaplanMeierFitter()
    kmf.fit(df["duration_days"], event_observed=df["is_churned"])
    median = kmf.median_survival_time_
    median_str = f"{median:.0f} days" if np.isfinite(median) else "Not reached"

    return dbc.Row([
        dbc.Col(kpi_card("Total Players", f"{len(df):,}"), md=True),
        dbc.Col(kpi_card("Events (churned)", f"{df['is_churned'].sum():,}", "danger"), md=True),
        dbc.Col(kpi_card("Censored", f"{(~df['is_churned'].astype(bool)).sum():,}", "success"), md=True),
        dbc.Col(kpi_card("Censoring Rate", f"{(~df['is_churned'].astype(bool)).mean()*100:.1f}%", "info"), md=True),
        dbc.Col(kpi_card("Median Survival", median_str), md=True),
    ])


@callback(
    Output("km-group-filter", "options"),
    Output("km-group-filter", "value"),
    Input("km-group-select", "value"),
)
def update_group_filter(group_col):
    """Populate group filter checklist based on selected grouping."""
    if group_col == "none":
        return [], []

    plot_df = display_group_frame(get_players(), group_col)

    groups = sorted(plot_df[group_col].dropna().unique())
    options = [{"label": f" {g}", "value": g} for g in groups]
    return options, list(groups)  # all selected by default


@callback(
    Output("km-plot", "figure"),
    Output("km-median-table", "children"),
    Output("km-logrank-text", "children"),
    Input("km-group-select", "value"),
    Input("km-ymin-slider", "value"),
    Input("km-ci-toggle", "value"),
    Input("km-group-filter", "value"),
)
def update_km(group_col, y_min, ci_toggle, selected_groups):
    plot_df = display_group_frame(get_players(), group_col)

    title = "Overall Retention" if group_col == "none" else f"Retention by {group_col}"
    gc = None if group_col == "none" else group_col
    show_ci = "ci" in (ci_toggle or [])
    sel = selected_groups if selected_groups else None

    fig = km_figure(plot_df, gc, title, show_ci=show_ci, y_min=y_min, selected_groups=sel)
    median_df = median_survival_table(plot_df, gc)
    table = dbc.Table.from_dataframe(median_df, striped=True, bordered=True, hover=True, size="sm", className="small")

    if group_col == "none":
        logrank_text = html.P(
            "Select a grouping variable to see log-rank test results.",
            className="text-muted fst-italic small",
        )
    else:
        logrank_text = logrank_summary(plot_df, group_col)

    return fig, table, logrank_text


@callback(
    Output("forest-plot", "figure"),
    Output("cox-summary-table", "children"),
    Output("schoenfeld-table", "figure"),
    Output("cox-interpretation", "children"),
    Input("km-group-select", "value"),
)
def update_cox(_):
    df = get_players()
    cph, cox_df = fit_cox(df)

    forest = forest_plot(cph)

    summary = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
    summary.columns = ["HR", "HR Lower 95%", "HR Upper 95%", "p-value"]
    summary = summary.round(4)
    summary.index.name = "Covariate"
    table = dbc.Table.from_dataframe(
        summary.reset_index(), striped=True, bordered=True, hover=True, size="sm", className="small"
    )

    schoenfeld = schoenfeld_figure(cph, cox_df)

    sig_vars = summary[summary["p-value"] < 0.05]
    interp_lines = ["**Significant covariates (p < 0.05):**\n"]
    for var, row in sig_vars.iterrows():
        direction = "increases" if row["HR"] > 1 else "decreases"
        interp_lines.append(
            f"- **{var}**: HR = {row['HR']:.2f} "
            f"({row['HR Lower 95%']:.2f}–{row['HR Upper 95%']:.2f}). "
            f"A one-unit increase {direction} churn hazard by "
            f"{abs(row['HR'] - 1) * 100:.1f}%."
        )
    if not sig_vars.empty:
        interp_lines.append("\n*HR > 1 = higher churn risk (red). HR < 1 = protective (blue).*")
    else:
        interp_lines.append("No covariates reached significance at p < 0.05.")

    interp_lines.append(f"\n**Concordance index:** {cph.concordance_index_:.3f}")

    return forest, table, schoenfeld, "\n".join(interp_lines)


@callback(Output("retention-heatmap", "figure"), Input("km-group-select", "value"))
def update_heatmap(_):
    df = get_players()
    return cohort_retention_heatmap(df)


@callback(
    Output("milestone-km-plot", "figure"),
    Output("milestone-summary-text", "children"),
    Input("milestone-n-slider", "value"),
    Input("milestone-group-select", "value"),
)
def update_milestone(n_contests, group_col):
    df = get_players()
    milestone_df = build_milestone_data(df, n_contests)

    plot_df = display_group_frame(df, group_col)

    gc = None if group_col == "none" else group_col
    fig = km_milestone_figure(plot_df, milestone_df, gc, n_contests)
    summary = milestone_summary(plot_df, milestone_df, n_contests)

    return fig, summary
