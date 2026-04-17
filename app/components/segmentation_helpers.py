"""Helper utilities for the segmentation dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.nonparametric.smoothers_lowess import lowess

from components.plots import PALETTE

SEGMENT_GLOSSARY_MD = """
- **Segment**: a player subgroup such as risk quartile, contest-style preference, or multi-sport behavior.
- **Kruskal-Wallis**: non-parametric alternative to one-way ANOVA; tests whether at least one segment differs in distribution.
- **Dunn post-hoc**: pairwise follow-up after Kruskal-Wallis with multiplicity correction.
- **Median and IQR**: a robust center-and-spread summary for skewed player behavior metrics.
"""

SEGMENT_NOTES_MD = """
- Spend, contest counts, and profitability are strongly right-skewed, so medians and non-parametric tests are emphasized.
- Dunn post-hoc p-values are Holm-adjusted to keep familywise error under control.
- Pairwise differences are reported only after a significant omnibus result, so the page stays focused on interpretable contrasts.
- Segment footprint compares share of players with share of total fees, which helps separate large segments from economically concentrated ones.
- Profile indices are normalized to the overall portfolio average (`100 = overall level`) so segments can be compared across mixed units.
"""

SEGMENT_METRICS = {
    "TotFees": {"label": "Total fees ($)", "format": "currency"},
    "AvgBuyIn": {"label": "Average buy-in ($)", "format": "currency"},
    "net_pnl": {"label": "Net P&L ($)", "format": "currency_signed"},
    "nCont": {"label": "Contests entered", "format": "int"},
    "duration_days": {"label": "Active lifetime (days)", "format": "days"},
    "RiskScore": {"label": "Risk score", "format": "float1"},
    "win_rate": {"label": "Win rate", "format": "pct"},
    "intensity": {"label": "Intensity (contests/day)", "format": "float2"},
    "type_diversity": {"label": "Contest-type diversity", "format": "float2"},
    "entries_per_contest": {"label": "Entries per contest", "format": "float2"},
}

LOWESS_OUTCOMES = {
    "is_churned": {"label": "Churn probability", "format": "pct"},
    "TotFees": SEGMENT_METRICS["TotFees"],
    "duration_days": SEGMENT_METRICS["duration_days"],
    "win_rate": SEGMENT_METRICS["win_rate"],
    "intensity": SEGMENT_METRICS["intensity"],
}

ANOVA_RESPONSES = {
    "log_total_fees": {"label": "Log total fees", "format": "float2"},
    "win_rate": SEGMENT_METRICS["win_rate"],
    "intensity": SEGMENT_METRICS["intensity"],
    "type_diversity": SEGMENT_METRICS["type_diversity"],
}

GROUP_OPTIONS = [
    {"label": "Dominant contest type", "value": "dominant_type"},
    {"label": "Risk quartile", "value": "risk_quartile"},
    {"label": "Buy-in quartile", "value": "buyin_quartile"},
    {"label": "Multi-sport", "value": "is_multisport"},
    {"label": "Age group", "value": "age_group"},
    {"label": "Churned vs active", "value": "is_churned"},
]

GROUP_LABELS = {
    "dominant_type": "Dominant contest type",
    "risk_quartile": "Risk quartile",
    "buyin_quartile": "Buy-in quartile",
    "is_multisport": "Sport breadth",
    "age_group": "Age group",
    "is_churned": "Retention state",
}

PROFILE_METRICS = {
    "churn_rate": {"label": "Churn rate", "format": "pct", "direction": "lower"},
    "median_days": {"label": "Median lifetime", "format": "days", "direction": "higher"},
    "median_win_rate": {"label": "Median win rate", "format": "pct", "direction": "higher"},
    "median_intensity": {"label": "Median intensity", "format": "float2", "direction": "higher"},
}

BINARY_GROUP_LABELS = {
    "is_multisport": {"0": "NFL only", "1": "Multi-sport"},
    "is_churned": {"0": "Active", "1": "Churned"},
}


def metric_label(metric_col: str) -> str:
    return SEGMENT_METRICS.get(metric_col, LOWESS_OUTCOMES.get(metric_col, {"label": metric_col}))["label"]


def group_label(group_col: str) -> str:
    return GROUP_LABELS.get(group_col, group_col)


def format_metric(value: float | int | None, metric_key: str) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"

    meta = SEGMENT_METRICS.get(metric_key, LOWESS_OUTCOMES.get(metric_key, ANOVA_RESPONSES.get(metric_key, {})))
    fmt = meta.get("format", "float2")

    if fmt == "currency":
        return f"${value:,.0f}"
    if fmt == "currency_signed":
        return f"${value:,.2f}"
    if fmt == "int":
        return f"{value:,.0f}"
    if fmt == "days":
        return f"{value:,.0f} days"
    if fmt == "pct":
        return f"{value * 100:.1f}%"
    if fmt == "float1":
        return f"{value:.1f}"
    return f"{value:.2f}"


def display_segment_frame(df: pl.DataFrame, group_col: str) -> pl.DataFrame:
    plot_df = df.drop_nulls([group_col])
    label_map = BINARY_GROUP_LABELS.get(group_col)
    if not label_map:
        return plot_df
    return plot_df.with_columns(pl.col(group_col).cast(pl.Utf8).replace(label_map).alias(group_col))


def segment_summary_table(df: pl.DataFrame, group_col: str, metric_col: str) -> pl.DataFrame:
    return (
        df.drop_nulls([group_col, metric_col])
        .group_by(group_col)
        .agg([
            pl.len().alias("Players"),
            pl.col(metric_col).median().alias("Median"),
            pl.col(metric_col).mean().alias("Mean"),
            pl.col(metric_col).quantile(0.25).alias("Q1"),
            pl.col(metric_col).quantile(0.75).alias("Q3"),
            pl.col("is_churned").mean().alias("Churn rate"),
        ])
        .sort("Median", descending=True)
    )


def segment_footprint_table(df: pl.DataFrame, group_col: str) -> pl.DataFrame:
    grouped = (
        df.drop_nulls([group_col])
        .group_by(group_col)
        .agg([
            pl.len().alias("Players"),
            pl.col("TotFees").sum().alias("Total fees"),
            pl.col("is_churned").mean().alias("Churn rate"),
            pl.col("duration_days").median().alias("Median lifetime"),
            pl.col("win_rate").median().alias("Median win rate"),
            pl.col("intensity").median().alias("Median intensity"),
        ])
    )
    total_players = float(grouped["Players"].sum())
    total_fees = float(grouped["Total fees"].sum())
    return (
        grouped.with_columns([
            (pl.col("Players") / total_players).alias("Player share"),
            (pl.col("Total fees") / total_fees).alias("Fee share"),
            ((pl.col("Total fees") / total_fees) - (pl.col("Players") / total_players)).alias("Fee premium"),
        ])
        .sort("Fee share", descending=True)
    )


def segment_box_figure(df: pl.DataFrame, metric_col: str, group_col: str, log_scale: bool = False) -> go.Figure:
    plot_df = df.select([group_col, metric_col]).drop_nulls().to_pandas()
    fig = px.box(
        plot_df,
        x=group_col,
        y=metric_col,
        color=group_col,
        color_discrete_sequence=PALETTE,
        log_y=log_scale,
        points="outliers",
    )
    fig.update_layout(
        title=f"{metric_label(metric_col)} by {group_label(group_col)}",
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=False,
    )
    fig.update_xaxes(title_text=group_label(group_col))
    fig.update_yaxes(title_text=metric_label(metric_col))
    return fig


def segment_interval_figure(summary_df: pl.DataFrame, metric_col: str, group_col: str) -> go.Figure:
    plot_df = summary_df.select([group_col, "Players", "Median", "Q1", "Q3", "Churn rate"]).sort("Median").to_pandas()
    fig = go.Figure()

    for _, row in plot_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["Q1"], row["Q3"]],
                y=[row[group_col], row[group_col]],
                mode="lines",
                line=dict(color="rgba(110,168,254,0.75)", width=8),
                showlegend=False,
                hovertemplate=(
                    f"{group_label(group_col)}={row[group_col]}<br>"
                    f"IQR: {row['Q1']:.3f} to {row['Q3']:.3f}<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=plot_df["Median"],
            y=plot_df[group_col],
            mode="markers+text",
            marker=dict(size=12, color="#0d6efd", line=dict(color="white", width=1)),
            text=[f"{int(n):,} players" for n in plot_df["Players"]],
            textposition="middle right",
            customdata=np.column_stack([plot_df["Q1"], plot_df["Q3"], plot_df["Players"], plot_df["Churn rate"]]),
            hovertemplate=(
                f"{group_label(group_col)}=%{{y}}<br>"
                f"Median {metric_label(metric_col)}=%{{x:.3f}}<br>"
                "IQR=%{customdata[0]:.3f} to %{customdata[1]:.3f}<br>"
                "Players=%{customdata[2]:,.0f}<br>"
                "Churn=%{customdata[3]:.1%}<extra></extra>"
            ),
            name="Median and IQR",
        )
    )

    fig.update_layout(
        title=f"Median and IQR by {group_label(group_col)}",
        template="plotly_white",
        margin=dict(l=80, r=120, t=50, b=40),
        showlegend=False,
    )
    fig.update_xaxes(title_text=metric_label(metric_col))
    fig.update_yaxes(title_text=group_label(group_col))
    return fig


def segment_footprint_figure(footprint_df: pl.DataFrame, group_col: str) -> go.Figure:
    plot_df = footprint_df.select([group_col, "Player share", "Fee share", "Players"]).sort("Fee share").to_pandas()
    fig = go.Figure()

    for _, row in plot_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["Player share"] * 100, row["Fee share"] * 100],
                y=[row[group_col], row[group_col]],
                mode="lines",
                line=dict(color="rgba(108,117,125,0.5)", width=4),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=plot_df["Player share"] * 100,
            y=plot_df[group_col],
            mode="markers",
            marker=dict(size=11, color="#6c757d"),
            name="Player share",
            customdata=plot_df["Players"],
            hovertemplate="Segment=%{y}<br>Player share=%{x:.1f}%<br>Players=%{customdata:,}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["Fee share"] * 100,
            y=plot_df[group_col],
            mode="markers",
            marker=dict(size=13, color="#198754", line=dict(color="white", width=1)),
            name="Fee share",
            customdata=plot_df["Players"],
            hovertemplate="Segment=%{y}<br>Fee share=%{x:.1f}%<br>Players=%{customdata:,}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Segment footprint: player share vs fee share",
        template="plotly_white",
        margin=dict(l=90, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="Share of portfolio (%)")
    fig.update_yaxes(title_text=group_label(group_col))
    return fig


def segment_rank_figure(summary_df: pl.DataFrame, metric_col: str, group_col: str) -> go.Figure:
    plot_df = summary_df.select([group_col, "Median", "Players"]).sort("Median").to_pandas()
    fig = px.bar(
        plot_df,
        x="Median",
        y=group_col,
        orientation="h",
        text="Players",
        color="Median",
        color_continuous_scale="Tealgrn",
    )
    fig.update_layout(
        title=f"Median {metric_label(metric_col)} ranking",
        template="plotly_white",
        margin=dict(l=80, r=20, t=50, b=40),
        coloraxis_showscale=False,
    )
    fig.update_traces(texttemplate="%{text:,} players", textposition="outside")
    fig.update_xaxes(title_text=metric_label(metric_col))
    fig.update_yaxes(title_text=group_label(group_col))
    return fig


def pairwise_posthoc_table(
    summary_df: pl.DataFrame,
    dunn_df: pl.DataFrame,
    group_col: str,
) -> pl.DataFrame:
    median_lookup = dict(zip(summary_df[group_col].to_list(), summary_df["Median"].to_list()))
    matrix = dunn_df.to_pandas().set_index(group_col)
    labels = list(matrix.index)
    rows = []

    for idx, left in enumerate(labels):
        for right in labels[idx + 1:]:
            left_median = float(median_lookup[left])
            right_median = float(median_lookup[right])
            if left_median >= right_median:
                higher, lower = left, right
                gap = left_median - right_median
            else:
                higher, lower = right, left
                gap = right_median - left_median
            p_value = float(matrix.loc[left, right])
            rows.append(
                {
                    "Higher median": higher,
                    "Lower median": lower,
                    "Median gap": gap,
                    "Adj p-value": p_value,
                    "Significant": "Yes" if p_value < 0.05 else "No",
                }
            )

    return (
        pl.DataFrame(rows)
        .sort(["Significant", "Adj p-value", "Median gap"], descending=[True, False, True])
    )


def segment_profile_heatmap(footprint_df: pl.DataFrame, group_col: str) -> go.Figure:
    overall = {
        "churn_rate": float((footprint_df["Churn rate"] * footprint_df["Players"]).sum() / footprint_df["Players"].sum()),
        "median_days": float(footprint_df["Median lifetime"].mean()),
        "median_win_rate": float(footprint_df["Median win rate"].mean()),
        "median_intensity": float(footprint_df["Median intensity"].mean()),
    }

    rows = []
    for row in footprint_df.to_dicts():
        rows.append({
            group_col: row[group_col],
            "churn_rate": float(row["Churn rate"]),
            "median_days": float(row["Median lifetime"]),
            "median_win_rate": float(row["Median win rate"]),
            "median_intensity": float(row["Median intensity"]),
        })
    profile_df = pl.DataFrame(rows)
    heat_rows = []
    for metric_key in PROFILE_METRICS:
        base = overall[metric_key]
        for row in profile_df.to_dicts():
            value = row[metric_key]
            index = 100 * value / base if base else 0.0
            heat_rows.append(
                {
                    group_col: row[group_col],
                    "metric": PROFILE_METRICS[metric_key]["label"],
                    "index_value": index,
                    "raw_value": value,
                    "baseline_value": base,
                }
            )
    heat_df = pl.DataFrame(heat_rows)
    matrix = (
        heat_df.pivot(on="metric", index=group_col, values="index_value")
        .sort(group_col)
        .to_pandas()
        .set_index(group_col)
    )
    raw_matrix = (
        heat_df.pivot(on="metric", index=group_col, values="raw_value")
        .sort(group_col)
        .to_pandas()
        .set_index(group_col)
    )

    customdata = np.dstack([raw_matrix.values, np.broadcast_to(matrix.values, raw_matrix.shape)])
    fig = go.Figure(
        go.Heatmap(
            z=matrix.values,
            x=list(matrix.columns),
            y=list(matrix.index),
            colorscale="RdYlGn",
            zmid=100,
            text=np.round(matrix.values, 0),
            texttemplate="%{text}",
            customdata=customdata,
            hovertemplate="Segment=%{y}<br>Metric=%{x}<br>Index=%{z:.0f}<br>Raw value=%{customdata[0]:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Segment profile index (100 = overall portfolio level)",
        template="plotly_white",
        margin=dict(l=90, r=20, t=50, b=40),
    )
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text=group_label(group_col))
    return fig


def dunn_heatmap_figure(dunn_df: pl.DataFrame, group_col: str) -> go.Figure:
    matrix = dunn_df.to_pandas().set_index(group_col)
    fig = go.Figure(
        go.Heatmap(
            z=matrix.values,
            x=list(matrix.columns),
            y=list(matrix.index),
            colorscale="Blues_r",
            zmin=0,
            zmax=max(float(np.nanmax(matrix.values)), 0.05),
            text=np.round(matrix.values, 3),
            texttemplate="%{text}",
            hovertemplate="Adjusted p = %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Dunn post-hoc adjusted p-values",
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(title_text=group_label(group_col))
    fig.update_yaxes(title_text=group_label(group_col))
    return fig


def anova_heatmap_figure(means_df: pl.DataFrame, factor_a: str, factor_b: str, response_col: str) -> go.Figure:
    matrix = (
        means_df.select([factor_a, factor_b, "mean"])
        .pivot(on=factor_b, index=factor_a, values="mean")
        .sort(factor_a)
        .to_pandas()
        .set_index(factor_a)
    )
    fig = go.Figure(
        go.Heatmap(
            z=matrix.values,
            x=list(matrix.columns),
            y=list(matrix.index),
            colorscale="YlGnBu",
            text=np.round(matrix.values, 2),
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title=f"Mean {metric_label(response_col)} by interaction cell",
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(title_text=group_label(factor_b))
    fig.update_yaxes(title_text=group_label(factor_a))
    return fig


def risk_lowess_figure(df: pl.DataFrame, outcome_col: str, group_col: str, seed: int = 42) -> go.Figure:
    plot_df = display_segment_frame(df, group_col).drop_nulls(["RiskScore", outcome_col])
    pdf = plot_df.select(["RiskScore", outcome_col, group_col]).to_pandas()
    if pdf.empty:
        return go.Figure()

    n_bins = min(10, max(4, pdf["RiskScore"].nunique()))
    bin_labels = [f"D{i}" for i in range(1, n_bins + 1)]
    pdf["risk_bin"] = pd.qcut(pdf["RiskScore"], q=n_bins, labels=bin_labels, duplicates="drop")

    bin_centers = (
        pdf.groupby("risk_bin", observed=False)
        .agg(risk_mid=("RiskScore", "median"), n=("RiskScore", "size"))
        .dropna(subset=["risk_mid"])
        .reset_index()
        .sort_values("risk_mid")
    )
    active_labels = bin_centers["risk_bin"].astype(str).tolist()
    label_to_mid = dict(zip(active_labels, bin_centers["risk_mid"]))

    grouped = (
        pdf.groupby([group_col, "risk_bin"], observed=False)
        .agg(
            risk_mid=("RiskScore", "median"),
            n=("RiskScore", "size"),
            value_mean=(outcome_col, "mean"),
            value_median=(outcome_col, "median"),
            q25=(outcome_col, lambda s: s.quantile(0.25)),
            q75=(outcome_col, lambda s: s.quantile(0.75)),
        )
        .dropna(subset=["risk_mid"])
        .reset_index()
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.78, 0.22],
    )

    for idx, group in enumerate(sorted(pdf[group_col].dropna().unique())):
        color = PALETTE[idx % len(PALETTE)]
        group_pdf = pdf[pdf[group_col] == group].sort_values("RiskScore")
        bin_df = grouped[grouped[group_col] == group].sort_values("risk_mid")

        if outcome_col == "is_churned":
            values = bin_df["value_mean"].to_numpy(dtype=float)
            se = np.sqrt(np.clip(values * (1 - values) / np.maximum(bin_df["n"].to_numpy(dtype=float), 1.0), 0.0, None))
            lower = np.clip(values - 1.96 * se, 0.0, 1.0)
            upper = np.clip(values + 1.96 * se, 0.0, 1.0)
        else:
            values = bin_df["value_median"].to_numpy(dtype=float)
            lower = bin_df["q25"].to_numpy(dtype=float)
            upper = bin_df["q75"].to_numpy(dtype=float)

        marker_sizes = np.clip(np.sqrt(bin_df["n"].to_numpy(dtype=float)) / 2.3, 8, 18)
        customdata = np.column_stack([
            bin_df["risk_bin"].astype(str).to_numpy(),
            bin_df["n"].to_numpy(dtype=int),
            lower,
            upper,
        ])

        fig.add_trace(
            go.Scatter(
                x=bin_df["risk_mid"],
                y=values,
                mode="lines+markers",
                line=dict(color=color, width=3),
                marker=dict(size=marker_sizes, color=color, line=dict(color="white", width=1)),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=np.maximum(upper - values, 0),
                    arrayminus=np.maximum(values - lower, 0),
                    color=color,
                    thickness=1,
                    width=0,
                ),
                name=str(group),
                legendgroup=str(group),
                customdata=customdata,
                hovertemplate=(
                    "Segment=%{fullData.name}<br>"
                    "Risk decile=%{customdata[0]}<br>"
                    "Median RiskScore=%{x:.1f}<br>"
                    "Players=%{customdata[1]:,}<br>"
                    f"{metric_label(outcome_col)}=%{{y:.3f}}<br>"
                    "Band=%{customdata[2]:.3f} to %{customdata[3]:.3f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        smooth = lowess(group_pdf[outcome_col], group_pdf["RiskScore"], frac=0.28, return_sorted=True)
        fig.add_trace(
            go.Scatter(
                x=smooth[:, 0],
                y=smooth[:, 1],
                mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                name=f"{group} LOWESS",
                legendgroup=str(group),
                showlegend=False,
                hovertemplate="RiskScore=%{x:.1f}<br>LOWESS=%{y:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Bar(
            x=bin_centers["risk_mid"],
            y=bin_centers["n"],
            marker=dict(color="rgba(108,117,125,0.35)", line=dict(color="rgba(108,117,125,0.7)", width=1)),
            text=bin_centers["n"],
            texttemplate="%{text:,}",
            textposition="outside",
            hovertemplate="Risk decile=%{customdata}<br>Median RiskScore=%{x:.1f}<br>Players=%{y:,}<extra></extra>",
            customdata=np.array(active_labels),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"RiskScore profile for {metric_label(outcome_col)}",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text=metric_label(outcome_col), row=1, col=1)
    fig.update_yaxes(title_text="Players", row=2, col=1)
    fig.update_xaxes(
        title_text="RiskScore deciles (tick labels) positioned at each decile's median score",
        row=2,
        col=1,
        tickmode="array",
        tickvals=list(label_to_mid.values()),
        ticktext=[f"{label}<br>{label_to_mid[label]:.1f}" for label in label_to_mid],
    )
    if outcome_col == "is_churned":
        fig.update_yaxes(range=[0, 1], tickformat=".0%", row=1, col=1)
    elif outcome_col == "win_rate":
        fig.update_yaxes(tickformat=".0%", row=1, col=1)
    return fig


def lowess_quartile_summary(df: pl.DataFrame, outcome_col: str, group_col: str) -> list[str]:
    summary_df = (
        display_segment_frame(df, group_col)
        .with_columns(
            pl.when(pl.col("RiskScore") <= pl.col("RiskScore").quantile(0.25)).then(pl.lit("Low risk"))
            .when(pl.col("RiskScore") >= pl.col("RiskScore").quantile(0.75)).then(pl.lit("High risk"))
            .otherwise(None)
            .alias("_risk_band")
        )
        .drop_nulls([group_col, outcome_col, "_risk_band"])
        .group_by([group_col, "_risk_band"])
        .agg(pl.col(outcome_col).mean().alias("value"))
        .sort(group_col)
    )

    lines = []
    for group in summary_df[group_col].unique().to_list():
        subset = summary_df.filter(pl.col(group_col) == group)
        low_row = subset.filter(pl.col("_risk_band") == "Low risk")
        high_row = subset.filter(pl.col("_risk_band") == "High risk")
        if low_row.height == 0 or high_row.height == 0:
            continue
        low_value = float(low_row["value"][0])
        high_value = float(high_row["value"][0])
        delta = high_value - low_value
        direction = "higher" if delta >= 0 else "lower"
        lines.append(
            f"{group}: high-risk players are {format_metric(abs(delta), outcome_col)} {direction} than low-risk players on average."
        )
    return lines
