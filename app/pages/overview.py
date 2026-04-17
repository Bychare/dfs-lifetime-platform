"""
Module 1: EDA & Data Overview Dashboard.
"""

import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import polars as pl
from dash import Input, Output, callback, dcc, html

from components.data_loader import get_players
from components.layout_utils import kpi_card, section_header
from components.plots import PALETTE, fig_box_by_group, fig_correlation_heatmap, fig_histogram, fig_kpi_cards

dash.register_page(__name__, path="/", name="Overview & EDA", order=0)

METRIC_OPTIONS = [
    {"label": "Total fees ($)", "value": "TotFees"},
    {"label": "Total winnings ($)", "value": "TotWinnings"},
    {"label": "Net P&L ($)", "value": "net_pnl"},
    {"label": "Number of contests", "value": "nCont"},
    {"label": "Active days", "value": "nDays"},
    {"label": "Average buy-in ($)", "value": "AvgBuyIn"},
    {"label": "Risk Score", "value": "RiskScore"},
    {"label": "Win Rate", "value": "win_rate"},
    {"label": "Intensity (contests/day)", "value": "intensity"},
    {"label": "Contest-type diversity", "value": "type_diversity"},
]

GROUP_OPTIONS = [
    {"label": "None", "value": "none"},
    {"label": "Churned vs Active", "value": "is_churned"},
    {"label": "Multi-sport", "value": "is_multisport"},
    {"label": "Dominant contest type", "value": "dominant_type"},
    {"label": "Risk quartile", "value": "risk_quartile"},
    {"label": "Age group", "value": "age_group"},
]

METRIC_LABELS = {option["value"]: option["label"] for option in METRIC_OPTIONS}

CORRELATION_COLS = [
    "TotFees",
    "TotWinnings",
    "net_pnl",
    "nCont",
    "nDays",
    "AvgBuyIn",
    "RiskScore",
    "win_rate",
    "intensity",
    "type_diversity",
]

GROUP_LABEL_MAPS = {
    "is_churned": {0: "Active", 1: "Churned"},
    "is_multisport": {0: "NFL only", 1: "Multi-sport"},
}


def _overview_df() -> pl.DataFrame:
    return get_players()


def _apply_group_labels(df: pl.DataFrame, group_col: str) -> pl.DataFrame:
    label_map = GROUP_LABEL_MAPS.get(group_col)
    if not label_map:
        return df
    str_map = {str(key): value for key, value in label_map.items()}
    return df.with_columns(
        pl.col(group_col).cast(pl.Utf8).replace(str_map).alias(group_col)
    )


def _top_states_bar_figure(df: pl.DataFrame):
    top20 = (
        df.filter(pl.col("country_name").is_in(["United States", "U.S.A.", "USA"]))
        .group_by("state_name")
        .len()
        .rename({"len": "players", "state_name": "state"})
        .sort("players", descending=True)
        .head(20)
        .sort("players")
    )
    fig = px.bar(
        top20.to_pandas(),
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
        height=400,
    )
    return fig


def _contest_type_figure(df: pl.DataFrame):
    counts = df.group_by("dominant_type").len().rename({"len": "players"}).sort("players", descending=True)
    fig = px.pie(
        counts.to_pandas(),
        names="dominant_type",
        values="players",
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(title="Dominant Contest Type", template="plotly_white")
    return fig


layout = html.Div([
    html.H3("DFS Player Analytics Overview", className="mb-1"),
    html.P(
        "DraftKings cohort: 10,385 players from the 2014 NFL season. "
        "Source: Nelson et al. (2019), The Transparency Project.",
        className="text-muted",
    ),
    html.Div(id="kpi-row", className="mb-4"),
    dbc.Row([
        dbc.Col([
            dbc.Label("Metric", html_for="metric-select"),
            dbc.Select(id="metric-select", options=METRIC_OPTIONS, value="TotFees"),
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
            dbc.Label("Grouping", html_for="group-select"),
            dbc.Select(id="group-select", options=GROUP_OPTIONS, value="none"),
        ], md=3),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="histogram-chart"), md=6),
        dbc.Col(dcc.Graph(id="box-chart"), md=6),
    ]),
    section_header("Correlation Matrix", "Spearman rank correlations between key player metrics"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="corr-heatmap"), md=6),
        dbc.Col(dcc.Graph(id="state-map"), md=6),
    ]),
    section_header("Contest-Type Composition"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="contest-type-chart"), md=6),
        dbc.Col(dcc.Graph(id="churn-by-group-chart"), md=6),
    ]),
])


@callback(Output("kpi-row", "children"), Input("metric-select", "value"))
def update_kpis(_):
    df = _overview_df()
    kpis = fig_kpi_cards(df)
    return dbc.Row([
        dbc.Col(kpi_card("Players", f"{kpis['total_players']:,}"), md=True),
        dbc.Col(kpi_card("Churn Rate", kpis["churn_rate"], "danger"), md=True),
        dbc.Col(kpi_card("Median fees", kpis["median_fees"]), md=True),
        dbc.Col(kpi_card("Median net P&L", kpis["median_net"], "warning"), md=True),
        dbc.Col(kpi_card("Losing players", kpis["pct_losers"], "danger"), md=True),
        dbc.Col(kpi_card("Median active days", kpis["median_days"]), md=True),
        dbc.Col(kpi_card("Multi-sport share", kpis["multisport_pct"], "info"), md=True),
    ])


@callback(Output("histogram-chart", "figure"), Input("metric-select", "value"), Input("log-switch", "value"))
def update_histogram(metric, log_switch):
    df = _overview_df()
    return fig_histogram(df[metric], METRIC_LABELS[metric], log_x="log" in log_switch)


@callback(
    Output("box-chart", "figure"),
    Input("metric-select", "value"),
    Input("group-select", "value"),
    Input("log-switch", "value"),
)
def update_box(metric, group, log_switch):
    df = _overview_df()
    if group == "none":
        group = "is_churned"
    plot_df = _apply_group_labels(df, group)
    return fig_box_by_group(
        plot_df,
        metric,
        group,
        log_y="log" in log_switch,
        value_label=METRIC_LABELS[metric],
    )


@callback(Output("corr-heatmap", "figure"), Input("metric-select", "value"))
def update_corr(_):
    return fig_correlation_heatmap(_overview_df(), CORRELATION_COLS)


@callback(Output("state-map", "figure"), Input("metric-select", "value"))
def update_map(_):
    return _top_states_bar_figure(_overview_df())


@callback(Output("contest-type-chart", "figure"), Input("metric-select", "value"))
def update_contest_types(_):
    return _contest_type_figure(_overview_df())


@callback(Output("churn-by-group-chart", "figure"), Input("group-select", "value"))
def update_churn_by_group(group):
    df = _overview_df()
    if group == "none":
        group = "dominant_type"
    plot_df = _apply_group_labels(df, group)
    churn_rates = (
        plot_df.group_by(group)
        .agg([
            pl.col("is_churned").mean().alias("churn_rate"),
            pl.len().alias("n"),
        ])
        .with_columns((pl.col("churn_rate") * 100).alias("churn_pct"))
        .sort(group)
    )
    fig = px.bar(
        churn_rates.to_pandas(),
        x=group,
        y="churn_pct",
        text="n",
        color_discrete_sequence=[PALETTE[1]],
        labels={"churn_pct": "Churn Rate (%)"},
    )
    fig.update_layout(title=f"Churn Rate by {group}", template="plotly_white")
    fig.update_traces(textposition="outside", texttemplate="%{text:,} players")
    return fig
