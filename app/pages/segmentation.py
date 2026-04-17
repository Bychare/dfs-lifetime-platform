"""Module 4: Segmentation."""

import dash
import dash_bootstrap_components as dbc
import polars as pl
from dash import Input, Output, callback, dcc, html

from components.data_loader import get_players
from components.layout_utils import glossary_accordion, kpi_card, methodological_notes, section_header
from components.segmentation_helpers import (
    GROUP_OPTIONS,
    SEGMENT_GLOSSARY_MD,
    SEGMENT_METRICS,
    SEGMENT_NOTES_MD,
    display_segment_frame,
    format_metric,
    group_label,
    metric_label,
    pairwise_posthoc_table,
    segment_footprint_figure,
    segment_footprint_table,
    segment_interval_figure,
    segment_profile_heatmap,
    segment_summary_table,
)
from components.stats import dunn_posthoc, kruskal_wallis, mann_whitney

dash.register_page(__name__, path="/segmentation", name="Segmentation", order=3)


def _segment_df():
    return get_players()


def _format_summary_table(df: pl.DataFrame, metric_key: str):
    pdf = df.to_pandas()
    for col in ["Median", "Mean", "Q1", "Q3", "Churn rate"]:
        if col not in pdf.columns:
            continue
        fmt_key = "is_churned" if col == "Churn rate" else metric_key
        pdf[col] = pdf[col].map(lambda value: format_metric(value, fmt_key))
    return dbc.Table.from_dataframe(pdf, striped=True, bordered=True, hover=True, size="sm", className="small")


def _format_pairwise_table(df: pl.DataFrame, metric_key: str):
    if df.height == 0:
        return dbc.Alert("Pairwise comparison table is empty for the current selection.", color="light", className="small")
    pdf = df.to_pandas()
    pdf["Median gap"] = pdf["Median gap"].map(lambda value: format_metric(value, metric_key))
    pdf["Adj p-value"] = pdf["Adj p-value"].map(lambda value: f"{value:.4f}")
    return dbc.Table.from_dataframe(pdf, striped=True, bordered=True, hover=True, size="sm", className="small")


layout = html.Div([
    html.H3("Segment Analysis", className="mb-1"),
    html.P(
        "A trimmed segmentation view that keeps only robust, explainable comparisons: segment medians, interquartile ranges, "
        "and non-parametric significance tests.",
        className="text-muted",
    ),
    glossary_accordion("Glossary: segmentation methods used on this page", SEGMENT_GLOSSARY_MD),
    html.Div(id="seg-kpi-row", className="mb-4"),
    section_header(
        "Segment Comparison",
        "Read this page left to right: segment center and spread first, then the omnibus test, then the pairwise differences worth describing.",
    ),
    dbc.Row([
        dbc.Col([
            dbc.Label("Segment definition"),
            dbc.Select(id="seg-group", options=GROUP_OPTIONS, value="dominant_type"),
        ], md=4),
        dbc.Col([
            dbc.Label("Metric"),
            dbc.Select(
                id="seg-metric",
                options=[{"label": meta["label"], "value": key} for key, meta in SEGMENT_METRICS.items()],
                value="TotFees",
            ),
        ], md=4),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="seg-interval-fig"), md=8),
        dbc.Col(dbc.Alert(id="seg-test-summary", color="light", className="small"), md=4),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            html.H5("Segment Summary", className="mb-2"),
            html.Div(id="seg-summary-table"),
        ], md=7),
        dbc.Col([
            html.H5("Pairwise Differences", className="mb-2"),
            html.Div(id="seg-posthoc-table"),
            dbc.Alert(id="seg-posthoc-summary", color="secondary", className="small mt-3"),
        ], md=5),
    ], className="mb-4"),
    section_header(
        "Segment Footprint",
        "Two extra views: who is large versus economically concentrated, and how each segment behaves relative to the overall portfolio baseline.",
    ),
    dbc.Row([
        dbc.Col(dcc.Graph(id="seg-footprint-fig"), md=6),
        dbc.Col(dcc.Graph(id="seg-profile-fig"), md=6),
    ], className="mb-3"),
    dbc.Alert(id="seg-footprint-summary", color="info", className="small mb-4"),
    methodological_notes(SEGMENT_NOTES_MD),
])


@callback(Output("seg-kpi-row", "children"), Input("seg-group", "value"), Input("seg-metric", "value"))
def update_segmentation_kpis(group_col: str, metric_col: str):
    plot_df = display_segment_frame(_segment_df(), group_col)
    summary_df = segment_summary_table(plot_df, group_col, metric_col)
    top_segment = summary_df.row(0, named=True)
    spread = float(summary_df["Median"].max() - summary_df["Median"].min())
    return dbc.Row([
        dbc.Col(kpi_card("Players in analysis", f"{plot_df.height:,}"), md=True),
        dbc.Col(kpi_card("Segments", f"{summary_df.height}"), md=True),
        dbc.Col(kpi_card("Overall median", format_metric(float(plot_df[metric_col].median()), metric_col), "info"), md=True),
        dbc.Col(kpi_card("Top median segment", str(top_segment[group_col]), "warning"), md=True),
        dbc.Col(kpi_card("Median spread", format_metric(spread, metric_col), "danger"), md=True),
    ])


@callback(
    Output("seg-interval-fig", "figure"),
    Output("seg-test-summary", "children"),
    Output("seg-summary-table", "children"),
    Output("seg-posthoc-table", "children"),
    Output("seg-posthoc-summary", "children"),
    Input("seg-group", "value"),
    Input("seg-metric", "value"),
)
def update_segment_comparison(group_col: str, metric_col: str):
    plot_df = display_segment_frame(_segment_df(), group_col).drop_nulls([metric_col])
    summary_df = segment_summary_table(plot_df, group_col, metric_col)
    groups = sorted(summary_df[group_col].to_list())
    series_list = [plot_df.filter(pl.col(group_col) == group)[metric_col] for group in groups]

    summary_lines = [html.Div(f"Grouping: {group_label(group_col)}", className="fw-semibold mb-2")]
    pairwise_table = dbc.Alert("Pairwise comparisons appear only when there are more than two groups.", color="light", className="small")
    pairwise_summary = "No post-hoc comparison needed."

    if len(groups) == 2:
        result = mann_whitney(series_list[0], series_list[1])
        summary_lines.append(
            html.P(
                f"Mann-Whitney U = {result['statistic']:.1f}, p = {result['p_value']:.4f}. "
                f"This directly compares the two segment distributions for {metric_label(metric_col).lower()}.",
                className="mb-0",
            )
        )
    else:
        result = kruskal_wallis(series_list)
        summary_lines.append(
            html.P(
                f"Kruskal-Wallis H = {result['statistic']:.2f}, p = {result['p_value']:.4f}, "
                f"epsilon^2 = {result['epsilon_sq']:.3f}.",
                className="mb-2",
            )
        )
        dunn_df = dunn_posthoc(plot_df, metric_col, group_col)
        pairwise_df = pairwise_posthoc_table(summary_df, dunn_df, group_col)
        pairwise_table = _format_pairwise_table(pairwise_df, metric_col)
        significant_pairs = pairwise_df.filter(pl.col("Significant") == "Yes").height
        total_pairs = pairwise_df.height
        if significant_pairs:
            top_pair = pairwise_df.filter(pl.col("Significant") == "Yes").row(0, named=True)
            pairwise_summary = (
                f"{significant_pairs} of {total_pairs} pairwise contrasts remain significant after Holm correction. "
                f"Largest retained gap: {top_pair['Higher median']} vs {top_pair['Lower median']} "
                f"({format_metric(float(top_pair['Median gap']), metric_col)})."
            )
        else:
            pairwise_summary = (
                f"No pairwise contrast survived Holm correction across {total_pairs} tested pairs."
            )

    return (
        segment_interval_figure(summary_df, metric_col, group_col),
        summary_lines,
        _format_summary_table(summary_df, metric_col),
        pairwise_table,
        pairwise_summary,
    )


@callback(
    Output("seg-footprint-fig", "figure"),
    Output("seg-profile-fig", "figure"),
    Output("seg-footprint-summary", "children"),
    Input("seg-group", "value"),
)
def update_segment_footprint(group_col: str):
    plot_df = display_segment_frame(_segment_df(), group_col)
    footprint_df = segment_footprint_table(plot_df, group_col)

    top_fee = footprint_df.sort("Fee premium", descending=True).row(0, named=True)
    lowest_churn = footprint_df.sort("Churn rate").row(0, named=True)
    highest_churn = footprint_df.sort("Churn rate", descending=True).row(0, named=True)

    summary = (
        f"{top_fee[group_col]} over-indexes economically: {top_fee['Fee share'] * 100:.1f}% of fees from "
        f"{top_fee['Player share'] * 100:.1f}% of players. "
        f"Lowest churn is in {lowest_churn[group_col]} ({lowest_churn['Churn rate'] * 100:.1f}%), "
        f"while highest churn is in {highest_churn[group_col]} ({highest_churn['Churn rate'] * 100:.1f}%)."
    )

    return (
        segment_footprint_figure(footprint_df, group_col),
        segment_profile_heatmap(footprint_df, group_col),
        summary,
    )
