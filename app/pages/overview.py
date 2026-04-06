"""
Module 1: EDA & Data Overview Dashboard.

Landing page with KPI cards, distributions, correlation matrix,
and US state map.
"""

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output
import plotly.express as px

from components.data_loader import get_players
from components.plots import (
    fig_histogram, fig_box_by_group, fig_correlation_heatmap,
    fig_us_state_map, fig_kpi_cards, PALETTE,
)
from components.layout_utils import kpi_card, section_header

dash.register_page(__name__, path="/", name="Overview & EDA", order=0)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

layout = html.Div([
    html.H3("DFS Player Analytics — Overview", className="mb-1"),
    html.P(
        "DraftKings cohort: 10,385 players, NFL season 2014. "
        "Source: Nelson et al. (2019), The Transparency Project.",
        className="text-muted",
    ),

    # --- KPI row ---
    html.Div(id="kpi-row", className="mb-4"),

    # --- Controls ---
    dbc.Row([
        dbc.Col([
            dbc.Label("Metric", html_for="metric-select"),
            dbc.Select(
                id="metric-select",
                options=[
                    {"label": "Total Fees ($)", "value": "TotFees"},
                    {"label": "Total Winnings ($)", "value": "TotWinnings"},
                    {"label": "Net P&L ($)", "value": "net_pnl"},
                    {"label": "Contests Entered", "value": "nCont"},
                    {"label": "Active Days", "value": "nDays"},
                    {"label": "Avg Buy-In ($)", "value": "AvgBuyIn"},
                    {"label": "Risk Score", "value": "RiskScore"},
                    {"label": "Win Rate", "value": "win_rate"},
                    {"label": "Intensity (contests/day)", "value": "intensity"},
                    {"label": "Type Diversity (entropy)", "value": "type_diversity"},
                ],
                value="TotFees",
            ),
        ], md=3),
        dbc.Col([
            dbc.Label("Log scale", html_for="log-switch"),
            dbc.Checklist(
                id="log-switch",
                options=[{"label": " Log(1 + x)", "value": "log"}],
                value=[],
                switch=True,
                className="mt-1",
            ),
        ], md=2),
        dbc.Col([
            dbc.Label("Group by", html_for="group-select"),
            dbc.Select(
                id="group-select",
                options=[
                    {"label": "None", "value": "none"},
                    {"label": "Churned vs Active", "value": "is_churned"},
                    {"label": "Multi-sport", "value": "is_multisport"},
                    {"label": "Dominant Contest Type", "value": "dominant_type"},
                    {"label": "Risk Quartile", "value": "risk_quartile"},
                    {"label": "Age Group", "value": "age_group"},
                ],
                value="none",
            ),
        ], md=3),
    ], className="mb-4"),

    # --- Charts ---
    dbc.Row([
        dbc.Col(dcc.Graph(id="histogram-chart"), md=6),
        dbc.Col(dcc.Graph(id="box-chart"), md=6),
    ]),

    section_header("Correlation Matrix", "Spearman rank correlations between key metrics"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="corr-heatmap"), md=6),
        dbc.Col(dcc.Graph(id="state-map"), md=6),
    ]),

    section_header("Contest Type Distribution"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="contest-type-chart"), md=6),
        dbc.Col(dcc.Graph(id="churn-by-group-chart"), md=6),
    ]),
])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("kpi-row", "children"),
    Input("metric-select", "value"),  # triggers on page load
)
def update_kpis(_):
    df = get_players()
    kpis = fig_kpi_cards(df)
    return dbc.Row([
        dbc.Col(kpi_card("Players", f"{kpis['total_players']:,}"), md=True),
        dbc.Col(kpi_card("Churn Rate", kpis["churn_rate"], "danger"), md=True),
        dbc.Col(kpi_card("Median Fees", kpis["median_fees"]), md=True),
        dbc.Col(kpi_card("Median Net P&L", kpis["median_net"], "warning"), md=True),
        dbc.Col(kpi_card("% Losers", kpis["pct_losers"], "danger"), md=True),
        dbc.Col(kpi_card("Median Active Days", kpis["median_days"]), md=True),
        dbc.Col(kpi_card("Multi-sport", kpis["multisport_pct"], "info"), md=True),
    ])


@callback(
    Output("histogram-chart", "figure"),
    Input("metric-select", "value"),
    Input("log-switch", "value"),
)
def update_histogram(metric, log_switch):
    df = get_players()
    return fig_histogram(df[metric].dropna(), metric, log_x="log" in log_switch)


@callback(
    Output("box-chart", "figure"),
    Input("metric-select", "value"),
    Input("group-select", "value"),
    Input("log-switch", "value"),
)
def update_box(metric, group, log_switch):
    df = get_players()
    if group == "none":
        group = "is_churned"
    # Map 0/1 to readable labels for binary columns
    plot_df = df.copy()
    if group == "is_churned":
        plot_df[group] = plot_df[group].map({0: "Active", 1: "Churned"})
    elif group == "is_multisport":
        plot_df[group] = plot_df[group].map({0: "NFL only", 1: "Multi-sport"})
    return fig_box_by_group(plot_df, metric, group, log_y="log" in log_switch)


@callback(
    Output("corr-heatmap", "figure"),
    Input("metric-select", "value"),  # just for trigger
)
def update_corr(_):
    df = get_players()
    cols = [
        "TotFees", "TotWinnings", "net_pnl", "nCont", "nDays",
        "AvgBuyIn", "RiskScore", "win_rate", "intensity", "type_diversity",
    ]
    return fig_correlation_heatmap(df, cols)


@callback(
    Output("state-map", "figure"),
    Input("metric-select", "value"),
)
def update_map(_):
    df = get_players()
    # US states only (country == USA)
    us = df[df["country_name"] == "U.S.A."]
    counts = us["state_name"].value_counts()
    # For now, show as bar chart of top-20 states
    top20 = counts.head(20).rename_axis("state").reset_index(name="players")
    fig = px.bar(
        top20,
        x="players",
        y="state",
        orientation="h",
        color_discrete_sequence=[PALETTE[0]],
        labels={"players": "Players", "state": "State"},
    )
    fig.update_layout(
        title="Top 20 US States by Player Count",
        template="plotly_white",
        margin=dict(l=120, r=20, t=50, b=40),
        yaxis=dict(autorange="reversed"),
        height=400,
    )
    return fig


@callback(
    Output("contest-type-chart", "figure"),
    Input("metric-select", "value"),
)
def update_contest_types(_):
    df = get_players()
    counts = df["dominant_type"].value_counts()
    fig = px.pie(
        names=counts.index, values=counts.values,
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(title="Dominant Contest Type", template="plotly_white")
    return fig


@callback(
    Output("churn-by-group-chart", "figure"),
    Input("group-select", "value"),
)
def update_churn_by_group(group):
    df = get_players()
    if group == "none":
        group = "dominant_type"
    plot_df = df.copy()
    if group == "is_multisport":
        plot_df[group] = plot_df[group].map({0: "NFL only", 1: "Multi-sport"})

    churn_rates = (
        plot_df.groupby(group)["is_churned"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "churn_rate", "count": "n"})
    )
    churn_rates["churn_pct"] = churn_rates["churn_rate"] * 100
    fig = px.bar(
        churn_rates, x=group, y="churn_pct",
        text="n", color_discrete_sequence=[PALETTE[1]],
        labels={"churn_pct": "Churn Rate (%)"},
    )
    fig.update_layout(title=f"Churn Rate by {group}", template="plotly_white")
    fig.update_traces(textposition="outside", texttemplate="%{text:,} players")
    return fig
