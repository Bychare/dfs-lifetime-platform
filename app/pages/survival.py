"""
Module 2: Survival Analysis.
"""

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, callback, dcc, html
from lifelines import KaplanMeierFitter

from components.data_loader import get_players
from components.layout_utils import glossary_accordion, kpi_card, methodological_notes, section_header
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


layout = html.Div([
    html.H3("Survival Analysis", className="mb-1"),
    html.P(
        "Player retention under right censoring using Kaplan-Meier curves, Cox proportional hazards, and cohort retention. "
        "About 77% of players are censored because they remain active through the end of the NFL season.",
        className="text-muted",
    ),
    glossary_accordion("Glossary: abbreviations and concepts used on this page", SURVIVAL_GLOSSARY_MD),
    html.Div(id="surv-kpi-row", className="mb-4"),
    section_header(
        "Kaplan-Meier Survival Curves",
        "Retention over time with 95% Greenwood confidence intervals, stratified comparison, and log-rank testing.",
    ),
    dbc.Row([
        dbc.Col([dbc.Label("Grouping"), dbc.Select(id="km-group-select", options=SURVIVAL_GROUP_OPTIONS, value="none")], md=3),
        dbc.Col([
            dbc.Label("Y-axis zoom"),
            dcc.Slider(id="km-ymin-slider", min=0, max=0.9, step=0.1, value=0, marks={0: "0", 0.5: "0.5", 0.7: "0.7", 0.9: "0.9"}, tooltip={"placement": "bottom", "always_visible": False}),
        ], md=3),
        dbc.Col([
            dbc.Label("Display"),
            dbc.Checklist(id="km-ci-toggle", options=[{"label": " Show 95% CI", "value": "ci"}], value=["ci"], switch=True, className="mt-1"),
        ], md=2),
        dbc.Col([
            dbc.Label("Visible groups"),
            dbc.Checklist(id="km-group-filter", options=[], value=[], className="small", style={"maxHeight": "120px", "overflowY": "auto"}),
        ], md=4),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="km-plot"), md=8),
        dbc.Col([html.Div(id="km-median-table"), html.Hr(), html.Div(id="km-logrank-text")], md=4),
    ]),
    section_header(
        "Cox Proportional Hazards",
        "Hazard ratios for churn drivers. HR > 1 means higher risk, HR < 1 lower risk.",
    ),
    dbc.Row([
        dbc.Col(dcc.Graph(id="forest-plot"), md=7),
        dbc.Col(html.Div(id="cox-summary-table"), md=5),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="schoenfeld-table"), md=7),
        dbc.Col(dcc.Markdown(id="cox-interpretation", className="small mt-3"), md=5),
    ], className="mt-2"),
    section_header("Cohort Retention Heatmap", "Weekly retention by cohort, defined by week of first contest."),
    dbc.Row([dbc.Col(dcc.Graph(id="retention-heatmap"), md=12)]),
    section_header(
        "Alternative Endpoint: Time to Nth Contest",
        "Instead of time to churn, use time to reaching N contests as an engagement milestone endpoint.",
    ),
    dbc.Row([
        dbc.Col([
            dbc.Label("Milestone in contests (N)"),
            dcc.Slider(id="milestone-n-slider", min=5, max=50, step=5, value=10, marks={n: str(n) for n in range(5, 55, 5)}, tooltip={"placement": "bottom"}),
        ], md=4),
        dbc.Col([dbc.Label("Grouping"), dbc.Select(id="milestone-group-select", options=MILESTONE_GROUP_OPTIONS, value="none")], md=4),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="milestone-km-plot"), md=8),
        dbc.Col(dcc.Markdown(id="milestone-summary-text", className="small"), md=4),
    ]),
    methodological_notes(SURVIVAL_NOTES_MD),
])


@callback(Output("surv-kpi-row", "children"), Input("km-group-select", "value"))
def update_surv_kpis(_):
    df = get_players()
    kmf = KaplanMeierFitter()
    kmf.fit(df["duration_days"].to_pandas(), event_observed=df["is_churned"].to_pandas())
    median = kmf.median_survival_time_
    median_str = f"{median:.0f} days" if np.isfinite(median) else "Not reached"

    total = df.height
    events = int(df["is_churned"].sum())
    censored = total - events
    censoring_share = censored / total * 100
    return dbc.Row([
        dbc.Col(kpi_card("Total players", f"{total:,}"), md=True),
        dbc.Col(kpi_card("Events (churn)", f"{events:,}", "danger"), md=True),
        dbc.Col(kpi_card("Censored", f"{censored:,}", "success"), md=True),
        dbc.Col(kpi_card("Censoring share", f"{censoring_share:.1f}%", "info"), md=True),
        dbc.Col(kpi_card("Median survival", median_str), md=True),
    ])


@callback(Output("km-group-filter", "options"), Output("km-group-filter", "value"), Input("km-group-select", "value"))
def update_group_filter(group_col):
    if group_col == "none":
        return [], []
    plot_df = display_group_frame(get_players(), group_col)
    groups = sorted(plot_df.select(group_col).drop_nulls().unique()[group_col].to_list())
    options = [{"label": f" {group}", "value": group} for group in groups]
    return options, groups


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
    fig = km_figure(plot_df, gc, title, show_ci="ci" in (ci_toggle or []), y_min=y_min, selected_groups=selected_groups or None)
    median_df = median_survival_table(plot_df, gc)
    table = dbc.Table.from_dataframe(median_df.to_pandas(), striped=True, bordered=True, hover=True, size="sm", className="small")
    logrank_text = (
        html.P("Choose a grouping variable to display the log-rank results.", className="text-muted fst-italic small")
        if group_col == "none"
        else logrank_summary(plot_df, group_col)
    )
    return fig, table, logrank_text


@callback(
    Output("forest-plot", "figure"),
    Output("cox-summary-table", "children"),
    Output("schoenfeld-table", "figure"),
    Output("cox-interpretation", "children"),
    Input("km-group-select", "value"),
)
def update_cox(_):
    cph, cox_df = fit_cox(get_players())
    forest = forest_plot(cph)

    summary = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
    summary.columns = ["HR", "HR Lower 95%", "HR Upper 95%", "p-value"]
    summary = summary.round(4)
    summary.index.name = "Covariate"
    table = dbc.Table.from_dataframe(summary.reset_index(), striped=True, bordered=True, hover=True, size="sm", className="small")
    schoenfeld = schoenfeld_figure(cph, cox_df)

    sig_vars = summary[summary["p-value"] < 0.05]
    interp_lines = ["**Significant covariates (p < 0.05):**\n"]
    for var, row in sig_vars.iterrows():
        direction = "increases" if row["HR"] > 1 else "decreases"
        interp_lines.append(
            f"- **{var}**: HR = {row['HR']:.2f} ({row['HR Lower 95%']:.2f}-{row['HR Upper 95%']:.2f}). "
            f"A one-unit increase {direction} churn hazard by {abs(row['HR'] - 1) * 100:.1f}%."
        )
    if sig_vars.empty:
        interp_lines.append("No covariate reached p < 0.05.")
    else:
        interp_lines.append("\n*HR > 1 indicates higher churn risk; HR < 1 indicates a protective effect.*")
    interp_lines.append(f"\n**Concordance index:** {cph.concordance_index_:.3f}")
    return forest, table, schoenfeld, "\n".join(interp_lines)


@callback(Output("retention-heatmap", "figure"), Input("km-group-select", "value"))
def update_heatmap(_):
    return cohort_retention_heatmap(get_players())


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
    return km_milestone_figure(plot_df, milestone_df, gc, n_contests), milestone_summary(plot_df, milestone_df, n_contests)
