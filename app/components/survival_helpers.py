"""Helpers and constants for the survival analysis page."""

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import html
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

try:
    from components.data_loader import NFL_SEASON_END
except ImportError:  # pragma: no cover - fallback for package-style imports in tests
    from app.components.data_loader import NFL_SEASON_END

PALETTE = px.colors.qualitative.Set2

SURVIVAL_GROUP_OPTIONS = [
    {"label": "No grouping (overall)", "value": "none"},
    {"label": "Multi-sport vs NFL-only", "value": "is_multisport"},
    {"label": "Risk Score quartile", "value": "risk_quartile"},
    {"label": "Dominant contest type", "value": "dominant_type"},
    {"label": "Avg Buy-In quartile", "value": "buyin_quartile"},
    {"label": "Number of sports", "value": "n_sports"},
]

MILESTONE_GROUP_OPTIONS = [
    {"label": "No grouping", "value": "none"},
    {"label": "Multi-sport vs NFL-only", "value": "is_multisport"},
    {"label": "Risk Score quartile", "value": "risk_quartile"},
    {"label": "Dominant contest type", "value": "dominant_type"},
    {"label": "Avg Buy-In quartile", "value": "buyin_quartile"},
]

SURVIVAL_GLOSSARY_MD = """
| Abbreviation | Meaning |
|---|---|
| **KM** | Kaplan–Meier — non-parametric estimator of the survival function. The step-wise curves on the plots below. |
| **CI** | Confidence Interval — range that contains the true value with 95% probability. Shown as shaded bands around KM curves. |
| **Log-rank test** | Statistical test comparing survival curves between groups. Null hypothesis: all groups have the same survival function. |
| **Cox PH** | Cox Proportional Hazards — regression model that estimates how covariates affect the rate of event occurrence (churn). |
| **HR** | Hazard Ratio — output of the Cox model. HR = 1: no effect. HR > 1: higher churn risk. HR < 1: lower churn risk (protective). |
| **p-value** | Probability of observing the data (or more extreme) if the null hypothesis were true. Smaller = stronger evidence against H₀. |
| **FWER** | Family-Wise Error Rate — probability of making at least one false positive across all tests. Controlled by Bonferroni correction. |
| **FDR** | False Discovery Rate — expected proportion of false positives among all rejected hypotheses. Controlled by Benjamini–Hochberg. |
| **Bonferroni** | Conservative correction: multiply each p-value by the number of tests. Controls FWER. |
| **BH** | Benjamini–Hochberg — less conservative correction. Controls FDR. More powerful than Bonferroni. |
| **✓ Bonf.** | Significant after Bonferroni correction (strongest evidence). |
| **✓ BH only** | Significant after BH correction but not Bonferroni (credible but less certain). |
| **~ raw only** | Nominally significant (p < 0.05) but does not survive any correction for multiple testing. |
| **✗ n.s.** | Not significant — no evidence of difference between the two groups. |
| **Censored** | Player still active at season end — we know they survived *at least* this long, but not when (or if) they will churn. |
| **Event** | The observed outcome of interest — here, player churn (last activity before season end). |
| **Median survival** | Time at which 50% of the group has experienced the event. "Not reached" = fewer than 50% churned by end of observation. |
| **Concordance index** | Measure of Cox model's discrimination (0.5 = random, 1.0 = perfect). Analogous to AUC for binary outcomes. |
| **Schoenfeld test** | Diagnostic test checking whether the proportional hazards assumption holds. If violated (p < 0.05), the HR may change over time. |
"""

SURVIVAL_NOTES_MD = """
**Censoring and median survival.** In the primary endpoint (time to churn),
77% of players are right-censored — they remain active at the end of the
NFL season (Jan 25, 2015). This means the Kaplan–Meier curve does not cross
0.50 for most subgroups, and the median survival time is not estimable.
This is a normal and expected result when the observation period is short
relative to the event rate. "Not reached" means the median lies somewhere
beyond the end of the data — it is not an error.

**What the curves still show.** Even without estimable medians, the
*shape* of the survival curves and log-rank tests reveal significant
differences between groups. The Cox PH model estimates hazard ratios
regardless of whether medians are reached, since it relies on the
ordering of events, not absolute survival times.

**Multiple comparison corrections.** When comparing K groups pairwise,
we run K×(K−1)/2 tests. Without correction, the probability of at least
one false positive grows rapidly (family-wise error rate). Two corrections
are applied:

- **Bonferroni** — multiplies each p-value by the number of tests.
  Conservative: controls the probability that *any* false positive occurs
  (FWER). Simple but may miss real differences.
- **Benjamini–Hochberg (BH)** — controls the *expected proportion* of
  false positives among rejected hypotheses (FDR). Less conservative,
  more powerful. Preferred when many comparisons are made and some
  false positives are tolerable.

A result marked "✓ Bonf." survives the strictest correction.
"✓ BH only" is significant after FDR control but not Bonferroni —
still a credible finding but with slightly less certainty.
"~ raw only" is nominally significant but does not survive any correction.

**Limitations of a single-season follow-up.** The observation window covers
approximately 5 months (one NFL season). Players censored at Jan 25 may have
churned on Jan 26 — we cannot distinguish long-term loyalists from
season-boundary artifacts. Multi-season data would be needed for
true lifetime value estimation.

**Alternative endpoint rationale.** The "time to Nth contest" endpoint
demonstrates how endpoint selection affects analytical power. When the
primary endpoint has too few events (23% churn), switching to an engagement
milestone like "reaching 10 contests" raises the event rate to ~75%, making
medians estimable and group comparisons more powerful. The time to milestone
is estimated via linear interpolation:
`t_N = duration × (N / total_contests)` for players who reached the milestone.
"""


def display_group_frame(df: pd.DataFrame, group_col: str | None) -> pd.DataFrame:
    """Map technical group values into display-friendly labels."""
    plot_df = df.copy()
    if group_col == "is_multisport":
        plot_df["is_multisport"] = plot_df["is_multisport"].map({0: "NFL only", 1: "Multi-sport"})
    elif group_col == "n_sports":
        plot_df["n_sports"] = plot_df["n_sports"].astype(str) + " sports"
    return plot_df


def _hex_to_rgb(color: str) -> str:
    if color.startswith("rgb"):
        return color.replace("rgb(", "").replace(")", "")
    h = color.lstrip("#")
    return ",".join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))


def km_figure(
    df: pd.DataFrame,
    group_col: str | None,
    title: str,
    show_ci: bool = True,
    y_min: float = 0.0,
    selected_groups: list | None = None,
) -> go.Figure:
    """Build a KM plot with optional confidence intervals."""
    fig = go.Figure()
    kmf = KaplanMeierFitter()

    if group_col is None or group_col == "none":
        kmf.fit(df["duration_days"], event_observed=df["is_churned"], label="All players")
        if show_ci:
            ci = kmf.confidence_interval_survival_function_
            fig.add_trace(go.Scatter(
                x=kmf.survival_function_.index, y=ci.iloc[:, 0],
                mode="lines", line=dict(width=0), showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=kmf.survival_function_.index, y=ci.iloc[:, 1],
                mode="lines", line=dict(width=0), fill="tonexty",
                fillcolor="rgba(102,194,165,0.2)", showlegend=False,
            ))
        fig.add_trace(go.Scatter(
            x=kmf.survival_function_.index, y=kmf.survival_function_.iloc[:, 0],
            mode="lines", name="All players", line=dict(color=PALETTE[0], width=2),
        ))
    else:
        groups = sorted(df[group_col].dropna().unique())
        if selected_groups:
            groups = [g for g in groups if g in selected_groups]
        for i, grp in enumerate(groups):
            mask = df[group_col] == grp
            color = PALETTE[i % len(PALETTE)]
            kmf.fit(
                df.loc[mask, "duration_days"],
                event_observed=df.loc[mask, "is_churned"],
                label=str(grp),
            )
            if show_ci:
                ci = kmf.confidence_interval_survival_function_
                fig.add_trace(go.Scatter(
                    x=kmf.survival_function_.index, y=ci.iloc[:, 0],
                    mode="lines", line=dict(width=0), showlegend=False,
                ))
                fig.add_trace(go.Scatter(
                    x=kmf.survival_function_.index, y=ci.iloc[:, 1],
                    mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor=f"rgba({_hex_to_rgb(color)},0.15)", showlegend=False,
                ))
            fig.add_trace(go.Scatter(
                x=kmf.survival_function_.index, y=kmf.survival_function_.iloc[:, 0],
                mode="lines", name=str(grp), line=dict(color=color, width=2),
            ))

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


def logrank_summary(df: pd.DataFrame, group_col: str):
    """Dash component with multivariate and pairwise log-rank results."""
    clean = df[[group_col, "duration_days", "is_churned"]].dropna()
    if clean[group_col].nunique() < 2:
        return html.P("Not enough groups for comparison.", className="text-muted")

    result = multivariate_logrank_test(
        clean["duration_days"], clean[group_col], clean["is_churned"]
    )

    header = html.Div([
        html.Strong("Multivariate log-rank test"),
        html.Ul([
            html.Li(f"Test statistic: {result.test_statistic:.2f}"),
            html.Li(f"p-value: {result.p_value:.4g}"),
            html.Li(f"Groups: {clean[group_col].nunique()}"),
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
            r = logrank_test(
                g1["duration_days"], g2["duration_days"],
                event_observed_A=g1["is_churned"],
                event_observed_B=g2["is_churned"],
            )
            pairs.append((groups[i], groups[j]))
            raw_pvals.append(r.p_value)

    n_tests = len(raw_pvals)
    raw_pvals = np.array(raw_pvals)
    bonf_pvals = np.minimum(raw_pvals * n_tests, 1.0)
    bh_pvals = _benjamini_hochberg(raw_pvals)

    table_rows = []
    for k, (g1, g2) in enumerate(pairs):
        sig_bonf = bonf_pvals[k] < 0.05
        sig_bh = bh_pvals[k] < 0.05
        sig_raw = raw_pvals[k] < 0.05
        if sig_bonf:
            verdict = "✓ Bonf."
            badge_color = "success"
        elif sig_bh:
            verdict = "✓ BH only"
            badge_color = "info"
        elif sig_raw:
            verdict = "~ raw only"
            badge_color = "warning"
        else:
            verdict = "✗ n.s."
            badge_color = "secondary"

        table_rows.append(html.Tr([
            html.Td(f"{g1} vs {g2}", className="small"),
            html.Td(f"{raw_pvals[k]:.4g}", className="small text-end"),
            html.Td(f"{bonf_pvals[k]:.4g}", className="small text-end"),
            html.Td(f"{bh_pvals[k]:.4g}", className="small text-end"),
            html.Td(dbc.Badge(verdict, color=badge_color, className="small")),
        ]))

    table = dbc.Table(
        [
            html.Thead(html.Tr([
                html.Th("Comparison"),
                html.Th("p (raw)"),
                html.Th("p (Bonf.)"),
                html.Th("p (BH)"),
                html.Th("Sig.?"),
            ])),
            html.Tbody(table_rows),
        ],
        bordered=True, striped=True, hover=True, size="sm",
        className="small",
    )

    footer = html.Div([
        html.P([
            html.Em(
                f"Bonferroni: α/n = 0.05/{n_tests} = {0.05/n_tests:.4f} "
                f"(conservative, controls FWER)."
            ),
        ], className="small mb-1"),
        html.P([
            html.Em(
                "Benjamini–Hochberg: controls FDR at 5% "
                "(less conservative, more powerful)."
            ),
        ], className="small mb-0"),
    ], className="mt-2")

    return html.Div([
        header,
        html.Strong(f"Pairwise log-rank tests ({n_tests} comparisons):", className="small"),
        table,
        footer,
    ])


def median_survival_table(df: pd.DataFrame, group_col: str | None) -> pd.DataFrame:
    """Median survival summary by group."""
    kmf = KaplanMeierFitter()
    rows = []

    if group_col is None or group_col == "none":
        groups_iter = [("All", df)]
    else:
        groups_iter = [(name, grp) for name, grp in df.groupby(group_col)]

    for name, grp in groups_iter:
        kmf.fit(grp["duration_days"], event_observed=grp["is_churned"])
        median = kmf.median_survival_time_
        rows.append({
            "Group": str(name),
            "N": len(grp),
            "Events": int(grp["is_churned"].sum()),
            "Censored": int((~grp["is_churned"].astype(bool)).sum()),
            "Median (days)": f"{median:.0f}" if np.isfinite(median) else "Not reached",
        })

    return pd.DataFrame(rows)


def fit_cox(df: pd.DataFrame) -> tuple[CoxPHFitter, pd.DataFrame]:
    """Fit the page's Cox proportional hazards specification."""
    cox_df = df[["duration_days", "is_churned"]].copy()
    cox_df["log_avg_buyin"] = np.log1p(df["AvgBuyIn"])
    cox_df["risk_score"] = df["RiskScore"]
    cox_df["is_multisport"] = df["is_multisport"]
    cox_df["win_rate"] = df["win_rate"]
    cox_df["intensity"] = df["intensity"]
    cox_df["type_diversity"] = df["type_diversity"]
    cox_df = cox_df.dropna()

    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="duration_days", event_col="is_churned")
    return cph, cox_df


def forest_plot(cph: CoxPHFitter) -> go.Figure:
    """Forest plot of Cox hazard ratios."""
    summary = cph.summary.copy()
    summary["HR"] = summary["exp(coef)"]
    summary["HR_lower"] = summary["exp(coef) lower 95%"]
    summary["HR_upper"] = summary["exp(coef) upper 95%"]
    summary = summary.sort_values("HR")

    fig = go.Figure()

    for var, row in summary.iterrows():
        color = "crimson" if row["HR"] > 1 else "steelblue"
        fig.add_trace(go.Scatter(
            x=[row["HR_lower"], row["HR_upper"]], y=[var, var],
            mode="lines", line=dict(color=color, width=2), showlegend=False,
        ))

    fig.add_trace(go.Scatter(
        x=summary["HR"], y=summary.index, mode="markers",
        marker=dict(size=10, color=["crimson" if hr > 1 else "steelblue" for hr in summary["HR"]]),
        text=[
            f"HR={row['HR']:.2f} ({row['HR_lower']:.2f}-{row['HR_upper']:.2f}), p={row['p']:.3g}"
            for _, row in summary.iterrows()
        ],
        hoverinfo="text", showlegend=False,
    ))

    fig.add_vline(x=1, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title="Cox PH — Hazard Ratios (Forest Plot)",
        xaxis_title="Hazard Ratio (log scale)",
        xaxis=dict(type="log"),
        template="plotly_white",
        height=350,
        margin=dict(l=150, r=30, t=60, b=50),
    )
    fig.add_annotation(
        x=0.5, y=1.08, xref="paper", yref="paper",
        text="← Lower churn risk | Higher churn risk →",
        showarrow=False, font=dict(size=10, color="gray"),
    )
    return fig


def schoenfeld_figure(cph: CoxPHFitter, cox_df: pd.DataFrame) -> go.Figure:
    """Diagnostic table for the proportional hazards assumption."""
    from lifelines.statistics import proportional_hazard_test

    test_results = proportional_hazard_test(cph, cox_df, time_transform="rank")
    summary = test_results.summary

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Covariate", "Test Statistic", "p-value", "Proportional?"],
            fill_color="lightsteelblue", align="left", font=dict(size=12),
        ),
        cells=dict(
            values=[
                summary.index.tolist(),
                [f"{v:.3f}" for v in summary["test_statistic"]],
                [f"{v:.4g}" for v in summary["p"]],
                ["Yes" if p >= 0.05 else "No (violated)" for p in summary["p"]],
            ],
            align="left", font=dict(size=11), height=28,
        ),
    )])

    fig.update_layout(
        title="Schoenfeld Test — Proportional Hazards Assumption",
        height=250,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def cohort_retention_heatmap(df: pd.DataFrame) -> go.Figure:
    """Weekly cohort retention heatmap."""
    df = df.copy()
    df["cohort_week"] = df["Date1st"].dt.isocalendar().week.astype(int)
    cohort_sizes = df.groupby("cohort_week").size()
    valid_cohorts = cohort_sizes[cohort_sizes >= 20].index.tolist()

    retention_matrix = []
    cohort_labels = []
    for cw in valid_cohorts:
        cohort_df = df[df["cohort_week"] == cw]
        cohort_start = cohort_df["Date1st"].min()
        n_total = len(cohort_df)
        row = []

        for w in range(20):
            week_cutoff = cohort_start + pd.Timedelta(weeks=w)
            if week_cutoff > NFL_SEASON_END:
                row.append(np.nan)
            else:
                still_active = (cohort_df["DateLst"] >= week_cutoff).sum()
                row.append(still_active / n_total * 100)

        retention_matrix.append(row)
        cohort_labels.append(f"Wk {cw} (n={n_total})")

    retention_matrix = np.array(retention_matrix)
    week_labels = [f"W+{i}" for i in range(retention_matrix.shape[1])]

    fig = go.Figure(data=go.Heatmap(
        z=retention_matrix, x=week_labels, y=cohort_labels,
        colorscale="Blues",
        text=np.where(
            np.isnan(retention_matrix), "",
            np.char.add(np.round(retention_matrix, 1).astype(str), "%")
        ),
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


def build_milestone_data(df: pd.DataFrame, n_contests: int) -> pd.DataFrame:
    """Build time-to-milestone data for time to Nth contest."""
    out = df[["UserID", "duration_days", "nCont", "nDays"]].copy()
    reached = out["nCont"] >= n_contests
    out["milestone_time"] = np.where(
        reached,
        out["duration_days"] * n_contests / out["nCont"],
        out["duration_days"],
    )
    out["milestone_time"] = out["milestone_time"].clip(lower=1).round().astype(int)
    out["milestone_event"] = reached.astype(int)
    return out


def km_milestone_figure(
    df: pd.DataFrame,
    milestone_df: pd.DataFrame,
    group_col: str | None,
    n: int,
) -> go.Figure:
    """KM curves for the time-to-milestone endpoint."""
    fig = go.Figure()
    kmf = KaplanMeierFitter()
    combined = df.join(milestone_df[["milestone_time", "milestone_event"]])

    if group_col is None or group_col == "none":
        kmf.fit(
            combined["milestone_time"],
            event_observed=combined["milestone_event"],
            label=f"Time to {n} contests",
        )
        ci = kmf.confidence_interval_survival_function_
        fig.add_trace(go.Scatter(
            x=kmf.survival_function_.index, y=ci.iloc[:, 0],
            mode="lines", line=dict(width=0), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=kmf.survival_function_.index, y=ci.iloc[:, 1],
            mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(102,194,165,0.2)", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=kmf.survival_function_.index, y=kmf.survival_function_.iloc[:, 0],
            mode="lines", name="All players", line=dict(color=PALETTE[0], width=2),
        ))
    else:
        groups = sorted(combined[group_col].dropna().unique())
        for i, grp in enumerate(groups):
            mask = combined[group_col] == grp
            color = PALETTE[i % len(PALETTE)]
            kmf.fit(
                combined.loc[mask, "milestone_time"],
                event_observed=combined.loc[mask, "milestone_event"],
                label=str(grp),
            )
            ci = kmf.confidence_interval_survival_function_
            fig.add_trace(go.Scatter(
                x=kmf.survival_function_.index, y=ci.iloc[:, 0],
                mode="lines", line=dict(width=0), showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=kmf.survival_function_.index, y=ci.iloc[:, 1],
                mode="lines", line=dict(width=0), fill="tonexty",
                fillcolor=f"rgba({_hex_to_rgb(color)},0.15)", showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=kmf.survival_function_.index, y=kmf.survival_function_.iloc[:, 0],
                mode="lines", name=str(grp), line=dict(color=color, width=2),
            ))

    fig.update_layout(
        title=f"Time to {n}th Contest (Engagement Milestone)",
        xaxis_title="Days since first contest",
        yaxis_title=f"P(not yet reached {n} contests)",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig


def milestone_summary(df: pd.DataFrame, milestone_df: pd.DataFrame, n: int) -> str:
    """Summary text for the milestone endpoint."""
    total = len(milestone_df)
    reached = milestone_df["milestone_event"].sum()
    censored = total - reached
    pct_reached = reached / total * 100

    kmf = KaplanMeierFitter()
    kmf.fit(milestone_df["milestone_time"], event_observed=milestone_df["milestone_event"])
    median = kmf.median_survival_time_
    median_str = f"{median:.0f} days" if np.isfinite(median) else "Not reached"

    lines = [
        f"**Endpoint: reaching {n} contests**\n",
        f"- Reached milestone: {reached:,} ({pct_reached:.1f}%)",
        f"- Censored (didn't reach): {censored:,} ({100-pct_reached:.1f}%)",
        f"- Median time to milestone: {median_str}",
        "",
        f"*Interpretation: this curve shows how quickly players accumulate "
        f"engagement. A steeper drop = faster contest adoption. "
        f"Unlike churn (where 77% are censored), this endpoint has "
        f"{pct_reached:.0f}% events — making medians estimable and "
        f"group comparisons more powerful.*",
    ]
    return "\n".join(lines)
