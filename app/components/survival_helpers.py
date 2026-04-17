"""Helpers and constants for the survival analysis page."""

from datetime import timedelta

import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from dash import html
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

try:
    from components.data_loader import NFL_SEASON_END
except ImportError:  # pragma: no cover
    from app.components.data_loader import NFL_SEASON_END

PALETTE = px.colors.qualitative.Set2

SURVIVAL_GROUP_OPTIONS = [
    {"label": "No grouping", "value": "none"},
    {"label": "Multi-sport vs NFL only", "value": "is_multisport"},
    {"label": "Risk Score quartile", "value": "risk_quartile"},
    {"label": "Dominant contest type", "value": "dominant_type"},
    {"label": "Average buy-in quartile", "value": "buyin_quartile"},
    {"label": "Number of sports", "value": "n_sports"},
]

MILESTONE_GROUP_OPTIONS = [
    {"label": "No grouping", "value": "none"},
    {"label": "Multi-sport vs NFL only", "value": "is_multisport"},
    {"label": "Risk Score quartile", "value": "risk_quartile"},
    {"label": "Dominant contest type", "value": "dominant_type"},
    {"label": "Average buy-in quartile", "value": "buyin_quartile"},
]

SURVIVAL_GLOSSARY_MD = """
| Term | Meaning |
|---|---|
| **KM** | Kaplan-Meier: non-parametric estimate of the survival function. |
| **CI** | Confidence interval, shown as a translucent band around the estimate. |
| **Log-rank test** | Test comparing survival curves between groups. Null: the survival functions are equal. |
| **Cox PH** | Cox proportional hazards model, estimating how covariates shift hazard over time. |
| **HR** | Hazard ratio. HR = 1 means no effect, HR > 1 higher churn hazard, HR < 1 lower churn hazard. |
| **FWER** | Family-wise error rate: probability of at least one false positive in a family of tests. |
| **FDR** | False discovery rate: expected share of false positives among rejected hypotheses. |
| **Bonferroni** | Conservative multiplicity correction controlling FWER. |
| **BH** | Benjamini-Hochberg correction controlling FDR. |
| **Censored** | Player still active by the end of observation, so true churn time is unknown. |
| **Median survival** | Time at which 50% of the group has experienced the event. |
| **Concordance index** | Discrimination metric for the Cox model, analogous to AUC in survival settings. |
| **Schoenfeld test** | Diagnostic for the proportional hazards assumption. |
"""

SURVIVAL_NOTES_MD = """
**Censoring and median survival.** In the primary endpoint, time to churn, 77% of
players are right-censored because they remain active through the end of the NFL
season on January 25, 2015. That means many Kaplan-Meier curves never cross 0.50,
so median survival is not estimable within the observation window.

**Why the curves still matter.** Even when the median is not reached, the shape of
the survival curves and the log-rank tests can still reveal strong differences
between groups. The Cox model can also estimate hazard ratios without requiring
the median to be observed.

**Multiple comparisons.** When we compare several groups, pairwise tests quickly
accumulate. Bonferroni offers strict FWER control, while Benjamini-Hochberg is
less conservative and targets FDR instead.

**Single-season limitation.** This dataset covers roughly one NFL season. A player
censored on January 25 could churn on January 26, so we cannot cleanly separate
long-term loyalty from season-boundary effects.

**Why the alternative endpoint helps.** Time to Nth contest increases the event
rate relative to time to churn. With more observed events, group comparisons are
better powered and median times often become estimable.
"""


def display_group_frame(df: pl.DataFrame, group_col: str | None) -> pl.DataFrame:
    if group_col == "is_multisport":
        return df.with_columns(
            pl.col("is_multisport")
            .cast(pl.Utf8)
            .replace({"0": "NFL only", "1": "Multi-sport"})
            .alias("is_multisport")
        )
    if group_col == "n_sports":
        return df.with_columns((pl.col("n_sports").cast(pl.Utf8) + pl.lit(" sports")).alias("n_sports"))
    return df


def _hex_to_rgb(color: str) -> str:
    if color.startswith("rgb"):
        return color.replace("rgb(", "").replace(")", "")
    h = color.lstrip("#")
    return ",".join(str(int(h[i:i + 2], 16)) for i in (0, 2, 4))


def km_figure(
    df: pl.DataFrame,
    group_col: str | None,
    title: str,
    show_ci: bool = True,
    y_min: float = 0.0,
    selected_groups: list | None = None,
) -> go.Figure:
    frame = df.to_pandas()
    fig = go.Figure()
    kmf = KaplanMeierFitter()

    if group_col is None or group_col == "none":
        kmf.fit(frame["duration_days"], event_observed=frame["is_churned"], label="All players")
        if show_ci:
            ci = kmf.confidence_interval_survival_function_
            fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=ci.iloc[:, 0], mode="lines", line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=ci.iloc[:, 1], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(102,194,165,0.2)", showlegend=False))
        fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=kmf.survival_function_.iloc[:, 0], mode="lines", name="All players", line=dict(color=PALETTE[0], width=2)))
    else:
        groups = sorted(frame[group_col].dropna().unique())
        if selected_groups:
            groups = [group for group in groups if group in selected_groups]
        for i, group in enumerate(groups):
            subset = frame[frame[group_col] == group]
            color = PALETTE[i % len(PALETTE)]
            kmf.fit(subset["duration_days"], event_observed=subset["is_churned"], label=str(group))
            if show_ci:
                ci = kmf.confidence_interval_survival_function_
                fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=ci.iloc[:, 0], mode="lines", line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=ci.iloc[:, 1], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=f"rgba({_hex_to_rgb(color)},0.15)", showlegend=False))
            fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=kmf.survival_function_.iloc[:, 0], mode="lines", name=str(group), line=dict(color=color, width=2)))

    fig.update_layout(
        title=title,
        xaxis_title="Days since first contest",
        yaxis_title="Survival probability (retention)",
        yaxis=dict(range=[y_min, 1.05]),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    ranked = np.argsort(pvals)
    adjusted = np.empty(n)
    for i, idx in enumerate(ranked):
        adjusted[idx] = pvals[idx] * n / (i + 1)
    adjusted_sorted = adjusted[np.argsort(pvals)]
    for i in range(n - 2, -1, -1):
        adjusted_sorted[i + 1] = min(adjusted_sorted[i + 1], 1.0)
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])
    adjusted[np.argsort(pvals)] = adjusted_sorted
    return np.minimum(adjusted, 1.0)


def logrank_summary(df: pl.DataFrame, group_col: str):
    clean = df.select([group_col, "duration_days", "is_churned"]).drop_nulls().to_pandas()
    if clean[group_col].nunique() < 2:
        return html.P("Not enough groups for comparison.", className="text-muted")

    result = multivariate_logrank_test(clean["duration_days"], clean[group_col], clean["is_churned"])
    header = html.Div([
        html.Strong("Multivariate log-rank test"),
        html.Ul([
            html.Li(f"Test statistic: {result.test_statistic:.2f}"),
            html.Li(f"p-value: {result.p_value:.4g}"),
            html.Li(f"Number of groups: {clean[group_col].nunique()}"),
        ], className="small mb-2"),
    ])

    groups = sorted(clean[group_col].unique())
    if not (2 <= len(groups) <= 6):
        return header

    pairs = []
    raw_pvals = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1 = clean[clean[group_col] == groups[i]]
            g2 = clean[clean[group_col] == groups[j]]
            result_pair = logrank_test(
                g1["duration_days"],
                g2["duration_days"],
                event_observed_A=g1["is_churned"],
                event_observed_B=g2["is_churned"],
            )
            pairs.append((groups[i], groups[j]))
            raw_pvals.append(result_pair.p_value)

    n_tests = len(raw_pvals)
    raw_pvals = np.array(raw_pvals)
    bonf_pvals = np.minimum(raw_pvals * n_tests, 1.0)
    bh_pvals = _benjamini_hochberg(raw_pvals)

    rows = []
    for idx, (g1, g2) in enumerate(pairs):
        if bonf_pvals[idx] < 0.05:
            verdict, badge_color = "✓ Bonf.", "success"
        elif bh_pvals[idx] < 0.05:
            verdict, badge_color = "✓ BH only", "info"
        elif raw_pvals[idx] < 0.05:
            verdict, badge_color = "~ raw only", "warning"
        else:
            verdict, badge_color = "✗ n.s.", "secondary"
        rows.append(html.Tr([
            html.Td(f"{g1} vs {g2}", className="small"),
            html.Td(f"{raw_pvals[idx]:.4g}", className="small text-end"),
            html.Td(f"{bonf_pvals[idx]:.4g}", className="small text-end"),
            html.Td(f"{bh_pvals[idx]:.4g}", className="small text-end"),
            html.Td(dbc.Badge(verdict, color=badge_color, className="small")),
        ]))

    return html.Div([
        header,
        html.Strong(f"Pairwise log-rank tests ({n_tests} comparisons):", className="small"),
        dbc.Table(
            [
                html.Thead(html.Tr([html.Th("Comparison"), html.Th("p (raw)"), html.Th("p (Bonf.)"), html.Th("p (BH)"), html.Th("Significant?")])),
                html.Tbody(rows),
            ],
            bordered=True,
            striped=True,
            hover=True,
            size="sm",
            className="small",
        ),
    ])


def median_survival_table(df: pl.DataFrame, group_col: str | None) -> pl.DataFrame:
    frame = df.to_pandas()
    kmf = KaplanMeierFitter()
    rows = []
    groups_iter = [("All", frame)] if group_col is None or group_col == "none" else list(frame.groupby(group_col))
    for name, group in groups_iter:
        kmf.fit(group["duration_days"], event_observed=group["is_churned"])
        median = kmf.median_survival_time_
        rows.append({
            "Group": str(name),
            "N": len(group),
            "Events": int(group["is_churned"].sum()),
            "Censored": int((~group["is_churned"].astype(bool)).sum()),
            "Median (days)": f"{median:.0f}" if np.isfinite(median) else "Not reached",
        })
    return pl.DataFrame(rows)


def fit_cox(df: pl.DataFrame):
    cox_df = (
        df.select([
            "duration_days",
            "is_churned",
            pl.col("AvgBuyIn").log1p().alias("log_avg_buyin"),
            pl.col("RiskScore").alias("risk_score"),
            "is_multisport",
            "win_rate",
            "intensity",
            "type_diversity",
        ])
        .drop_nulls()
        .to_pandas()
    )
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="duration_days", event_col="is_churned")
    return cph, cox_df


def forest_plot(cph: CoxPHFitter) -> go.Figure:
    summary = cph.summary.copy()
    summary["HR"] = summary["exp(coef)"]
    summary["HR_lower"] = summary["exp(coef) lower 95%"]
    summary["HR_upper"] = summary["exp(coef) upper 95%"]
    summary = summary.sort_values("HR")

    fig = go.Figure()
    for variable, row in summary.iterrows():
        color = "crimson" if row["HR"] > 1 else "steelblue"
        fig.add_trace(go.Scatter(x=[row["HR_lower"], row["HR_upper"]], y=[variable, variable], mode="lines", line=dict(color=color, width=2), showlegend=False))
    fig.add_trace(go.Scatter(
        x=summary["HR"],
        y=summary.index,
        mode="markers",
        marker=dict(size=10, color=["crimson" if hr > 1 else "steelblue" for hr in summary["HR"]]),
        text=[f"HR={row['HR']:.2f} ({row['HR_lower']:.2f}-{row['HR_upper']:.2f}), p={row['p']:.3g}" for _, row in summary.iterrows()],
        hoverinfo="text",
        showlegend=False,
    ))
    fig.add_vline(x=1, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title="Cox PH Hazard Ratios",
        xaxis_title="Hazard Ratio (log scale)",
        xaxis=dict(type="log"),
        template="plotly_white",
        height=350,
        margin=dict(l=150, r=30, t=60, b=50),
    )
    fig.add_annotation(x=0.5, y=1.08, xref="paper", yref="paper", text="← Lower churn risk | Higher churn risk →", showarrow=False, font=dict(size=10, color="gray"))
    return fig


def schoenfeld_figure(cph: CoxPHFitter, cox_df) -> go.Figure:
    from lifelines.statistics import proportional_hazard_test

    results = proportional_hazard_test(cph, cox_df, time_transform="rank")
    summary = results.summary
    fig = go.Figure(data=[go.Table(
        header=dict(values=["Covariate", "Test statistic", "p-value", "PH holds?"], fill_color="lightsteelblue", align="left", font=dict(size=12)),
        cells=dict(
            values=[
                summary.index.tolist(),
                [f"{value:.3f}" for value in summary["test_statistic"]],
                [f"{value:.4g}" for value in summary["p"]],
                ["Yes" if p >= 0.05 else "No" for p in summary["p"]],
            ],
            align="left",
            font=dict(size=11),
            height=28,
        ),
    )])
    fig.update_layout(title="Schoenfeld Test: Proportional Hazards Check", height=250, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def cohort_retention_heatmap(df: pl.DataFrame) -> go.Figure:
    frame = df.with_columns(pl.col("Date1st").dt.week().alias("cohort_week")).to_pandas()
    cohort_sizes = frame.groupby("cohort_week").size()
    valid_cohorts = cohort_sizes[cohort_sizes >= 20].index.tolist()

    retention_matrix = []
    cohort_labels = []
    for cohort_week in valid_cohorts:
        cohort_df = frame[frame["cohort_week"] == cohort_week]
        cohort_start = cohort_df["Date1st"].min()
        n_total = len(cohort_df)
        row = []
        for week in range(20):
            week_cutoff = cohort_start + timedelta(weeks=week)
            if week_cutoff > NFL_SEASON_END:
                row.append(np.nan)
            else:
                row.append((cohort_df["DateLst"] >= week_cutoff).sum() / n_total * 100)
        retention_matrix.append(row)
        cohort_labels.append(f"Week {cohort_week} (n={n_total})")

    retention_matrix = np.array(retention_matrix)
    week_labels = [f"W+{idx}" for idx in range(retention_matrix.shape[1])]
    fig = go.Figure(data=go.Heatmap(
        z=retention_matrix,
        x=week_labels,
        y=cohort_labels,
        colorscale="Blues",
        text=np.where(np.isnan(retention_matrix), "", np.char.add(np.round(retention_matrix, 1).astype(str), "%")),
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(title="Retention %"),
    ))
    fig.update_layout(
        title="Weekly Cohort Retention Heatmap",
        xaxis_title="Weeks since first contest",
        yaxis_title="Cohort (week of first contest)",
        template="plotly_white",
        height=max(350, len(valid_cohorts) * 25 + 100),
        margin=dict(l=120, r=20, t=60, b=50),
    )
    return fig


def build_milestone_data(df: pl.DataFrame, n_contests: int) -> pl.DataFrame:
    return (
        df.select(["UserID", "duration_days", "nCont", "nDays"])
        .with_columns([
            (pl.col("nCont") >= n_contests).cast(pl.Int8).alias("milestone_event"),
            pl.when(pl.col("nCont") >= n_contests)
            .then(pl.col("duration_days") * n_contests / pl.col("nCont"))
            .otherwise(pl.col("duration_days"))
            .clip(lower_bound=1)
            .round(0)
            .cast(pl.Int64)
            .alias("milestone_time"),
        ])
    )


def km_milestone_figure(df: pl.DataFrame, milestone_df: pl.DataFrame, group_col: str | None, n: int) -> go.Figure:
    combined = df.join(milestone_df.select(["UserID", "milestone_time", "milestone_event"]), on="UserID", how="left").to_pandas()
    fig = go.Figure()
    kmf = KaplanMeierFitter()

    if group_col is None or group_col == "none":
        kmf.fit(combined["milestone_time"], event_observed=combined["milestone_event"], label=f"Time to {n} contests")
        ci = kmf.confidence_interval_survival_function_
        fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=ci.iloc[:, 0], mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=ci.iloc[:, 1], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(102,194,165,0.2)", showlegend=False))
        fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=kmf.survival_function_.iloc[:, 0], mode="lines", name="All players", line=dict(color=PALETTE[0], width=2)))
    else:
        groups = sorted(combined[group_col].dropna().unique())
        for i, group in enumerate(groups):
            subset = combined[combined[group_col] == group]
            color = PALETTE[i % len(PALETTE)]
            kmf.fit(subset["milestone_time"], event_observed=subset["milestone_event"], label=str(group))
            ci = kmf.confidence_interval_survival_function_
            fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=ci.iloc[:, 0], mode="lines", line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=ci.iloc[:, 1], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=f"rgba({_hex_to_rgb(color)},0.15)", showlegend=False))
            fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=kmf.survival_function_.iloc[:, 0], mode="lines", name=str(group), line=dict(color=color, width=2)))

    fig.update_layout(
        title=f"Time to the {n}th Contest",
        xaxis_title="Days since first contest",
        yaxis_title=f"P(not yet reached {n} contests)",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig


def milestone_summary(df: pl.DataFrame, milestone_df: pl.DataFrame, n: int) -> str:
    total = milestone_df.height
    reached = int(milestone_df["milestone_event"].sum())
    censored = total - reached
    pct_reached = reached / total * 100

    milestone_pd = milestone_df.to_pandas()
    kmf = KaplanMeierFitter()
    kmf.fit(milestone_pd["milestone_time"], event_observed=milestone_pd["milestone_event"])
    median = kmf.median_survival_time_
    median_str = f"{median:.0f} days" if np.isfinite(median) else "Not reached"

    return "\n".join([
        f"**Endpoint: reaching {n} contests**",
        f"- Reached the milestone: {reached:,} ({pct_reached:.1f}%)",
        f"- Censored / not reached: {censored:,} ({100 - pct_reached:.1f}%)",
        f"- Median time to milestone: {median_str}",
        "",
        "This endpoint emphasizes engagement depth rather than time to churn. Because the event rate is higher than in the primary churn endpoint, medians are often estimable and group comparisons become more informative.",
    ])
