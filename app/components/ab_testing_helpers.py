"""Helpers and constants for the A/B testing page."""

from functools import lru_cache

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from scipy import stats

try:
    from components.data_loader import get_players
    from components.stats import (
        adjust_pvalues,
        bootstrap_uplift_ci,
        proportion_z_test,
        sample_size_continuous,
        sample_size_proportions,
        sample_size_survival,
    )
except ImportError:  # pragma: no cover
    from app.components.data_loader import get_players
    from app.components.stats import (
        adjust_pvalues,
        bootstrap_uplift_ci,
        proportion_z_test,
        sample_size_continuous,
        sample_size_proportions,
        sample_size_survival,
    )

PALETTE = px.colors.qualitative.Set2

SIM_METRICS = {
    "retained_30d": {
        "label": "Retained 30 days",
        "description": "The player remained active for at least 30 days after the first contest.",
    },
    "retained_60d": {
        "label": "Retained 60 days",
        "description": "A stricter retention endpoint for more durable cohorts.",
    },
    "reached_10_contests": {
        "label": "Reached 10 contests",
        "description": "A milestone KPI that is useful when churn is heavily censored.",
    },
    "profitable_player": {
        "label": "Positive net P&L",
        "description": "Share of players who finished the season profitable.",
    },
}

SEGMENTS = {
    "all": "All players",
    "multisport": "Multi-sport players",
    "nfl_only": "NFL-only players",
    "high_risk": "High risk, Q4",
    "low_risk": "Low risk, Q1",
    "high_buyin": "High buy-in, Q4",
}

AB_GLOSSARY_MD = """
| Term | Meaning |
|---|---|
| **A/B test** | A randomized controlled experiment comparing a control and treatment experience. |
| **Control** | The current product experience used as the comparison baseline. |
| **Treatment** | The new product version whose effect we want to estimate. |
| **Baseline rate** | Expected KPI value in the control group before launch. |
| **MDE** | Minimum Detectable Effect: the smallest effect worth powering the experiment for. |
| **Alpha** | Type I error rate. Probability of a false positive when there is no true effect. |
| **Power** | Probability of detecting a real effect of the target size. Equal to 1 − β. |
| **Hazard ratio (HR)** | Relative event intensity for survival endpoints. HR < 1 means slower churn, HR > 1 faster churn. |
| **z-test** | Frequentist test for comparing two proportions under a normal approximation. |
| **95% CI** | Plausible range for the estimated uplift. If it crosses 0, the direction is not yet clear. |
| **Posterior** | Bayesian distribution of possible effects after combining prior beliefs and observed data. |
| **P(Treat > Ctrl)** | Posterior probability that the treatment outperforms control. |
| **Expected loss** | Expected downside from shipping treatment if it is actually worse. |
| **Bootstrap CI** | Non-parametric interval based on resampling observations with replacement. |
| **Multiple testing** | Evaluating several KPIs or segments at once. Without correction, false positives accumulate. |
| **Holm correction** | Stepwise adjustment controlling FWER, the chance of at least one false positive. |
| **BH correction** | Benjamini-Hochberg adjustment controlling FDR, the expected share of false discoveries. |
| **Sequential testing** | Monitoring an experiment at interim looks rather than only once at the end. |
| **O'Brien-Fleming boundary** | Conservative early-stopping rule: very high threshold early, softer later on. |
| **Information fraction** | Share of planned information already collected at a given interim look. |
"""

AB_NOTES_MD = """
**Planning assumptions.** The sample-size calculator uses normal approximations:
two-proportion planning for binary KPIs, two-sample planning for continuous
metrics, and Schoenfeld's formula for survival endpoints. This is excellent for
early design work, but final plans should still be checked against actual
traffic, variance, and operational constraints.

**What the simulator does and does not do.** The simulator does not recreate a
historical randomized experiment. It takes the observed DFS baseline and injects
an artificial treatment effect. That makes it strong for explanation, demos, and
portfolio storytelling, but not for validating a real product decision.

**Frequentist and Bayesian blocks answer different questions.** The z-test and
its confidence interval quantify how surprising the observed difference would be
under a null of no effect. The Bayesian block instead describes the posterior
distribution of uplift and the probability that treatment beats control.

**Bootstrap and multiplicity corrections add robustness.** The bootstrap CI is a
useful stress test when the normal approximation feels too optimistic. Holm and
BH corrections become important when we inspect a family of KPIs rather than a
single metric.

**Sequential monitoring protects against naive peeking.** Repeatedly checking
the same experiment with a fixed 0.05 threshold inflates false positives.
O'Brien-Fleming designs allow early stopping, but demand much stronger evidence
at the first looks.
"""


@lru_cache(maxsize=1)
def ab_frame() -> pl.DataFrame:
    df = get_players()
    return df.with_columns([
        (pl.col("duration_days") >= 30).cast(pl.Int8).alias("retained_30d"),
        (pl.col("duration_days") >= 60).cast(pl.Int8).alias("retained_60d"),
        (pl.col("nCont") >= 10).cast(pl.Int8).alias("reached_10_contests"),
        (pl.col("net_pnl") > 0).cast(pl.Int8).alias("profitable_player"),
    ])


def segment_slice(df: pl.DataFrame, segment: str) -> pl.DataFrame:
    if segment == "multisport":
        return df.filter(pl.col("is_multisport") == 1)
    if segment == "nfl_only":
        return df.filter(pl.col("is_multisport") == 0)
    if segment == "high_risk":
        return df.filter(pl.col("risk_quartile") == "Q4 (High)")
    if segment == "low_risk":
        return df.filter(pl.col("risk_quartile") == "Q1 (Low)")
    if segment == "high_buyin":
        return df.filter(pl.col("buyin_quartile") == "Q4 (High)")
    return df


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def simulate_binary_experiment(
    values: pl.Series,
    n_per_arm: int,
    uplift_pct: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    rng = np.random.default_rng(seed)
    clean_values = values.drop_nulls().to_numpy().astype(int)
    baseline = float(clean_values.mean())
    target = float(np.clip(baseline * (1 + uplift_pct / 100), 0.001, 0.999))

    replace = n_per_arm > len(clean_values)
    control = rng.choice(clean_values, size=n_per_arm, replace=replace)
    treatment = rng.choice(clean_values, size=n_per_arm, replace=replace)
    treatment = _shift_binary_rate(treatment, target, rng)
    return control, treatment, baseline, target


def _shift_binary_rate(values: np.ndarray, target: float, rng: np.random.Generator) -> np.ndarray:
    values = values.astype(int, copy=True)
    current = float(values.mean())
    if current < target:
        zero_idx = np.where(values == 0)[0]
        flip_prob = min(1.0, (target - current) / max(1 - current, 1e-9))
        values[zero_idx[rng.random(len(zero_idx)) < flip_prob]] = 1
    elif current > target:
        one_idx = np.where(values == 1)[0]
        flip_prob = min(1.0, (current - target) / max(current, 1e-9))
        values[one_idx[rng.random(len(one_idx)) < flip_prob]] = 0
    return values


def _arm_ci(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    rate = successes / total
    z = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(rate * (1 - rate) / total)
    return max(rate - z * se, 0.0), min(rate + z * se, 1.0)


def rate_bar_figure(control: np.ndarray, treatment: np.ndarray, alpha: float) -> go.Figure:
    control_rate = float(control.mean())
    treatment_rate = float(treatment.mean())
    ci0 = _arm_ci(int(control.sum()), len(control), alpha=alpha)
    ci1 = _arm_ci(int(treatment.sum()), len(treatment), alpha=alpha)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Control", "Treatment"],
        y=[control_rate * 100, treatment_rate * 100],
        marker_color=[PALETTE[1], PALETTE[0]],
        error_y=dict(
            type="data",
            array=[
                max((ci0[1] - control_rate) * 100, 0),
                max((ci1[1] - treatment_rate) * 100, 0),
            ],
        ),
        text=[f"{control_rate * 100:.1f}%", f"{treatment_rate * 100:.1f}%"],
        textposition="outside",
    ))
    fig.update_layout(
        title="Observed KPI by Arm",
        yaxis_title="Rate (%)",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def posterior_figure(diff_draws: np.ndarray) -> go.Figure:
    fig = px.histogram(
        x=diff_draws * 100,
        nbins=50,
        color_discrete_sequence=[PALETTE[2]],
        labels={"x": "Treatment - Control (pp)"},
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#495057")
    fig.update_layout(
        title="Posterior Uplift Distribution",
        template="plotly_white",
        bargap=0.05,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def sequential_figure(seq_df: pl.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=seq_df["look"].to_list(),
        y=seq_df["z_stat"].to_list(),
        mode="lines+markers",
        name="Observed z-statistic",
        line=dict(color=PALETTE[0], width=3),
    ))
    fig.add_trace(go.Scatter(
        x=seq_df["look"].to_list(),
        y=seq_df["z_boundary"].to_list(),
        mode="lines+markers",
        name="Upper OBF boundary",
        line=dict(color=PALETTE[1], dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=seq_df["look"].to_list(),
        y=(-seq_df["z_boundary"]).to_list(),
        mode="lines+markers",
        name="Lower OBF boundary",
        line=dict(color=PALETTE[1], dash="dot"),
    ))
    fig.update_layout(
        title="Sequential Monitoring",
        xaxis_title="Interim look",
        yaxis_title="z-statistic",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def sample_size_curve(
    family: str,
    baseline: float,
    mde: float,
    sigma: float,
    hazard_ratio: float,
    event_rate: float,
    alpha: float,
    power: float,
) -> go.Figure:
    if family == "binary":
        grid = np.linspace(max(0.01, mde / 2), max(0.12, mde * 2), 12)
        sizes = [sample_size_proportions(baseline, x, alpha=alpha, power=power) for x in grid]
        x = grid * 100
        x_title = "Absolute MDE (pp)"
    elif family == "continuous":
        grid = np.linspace(max(0.1, mde / 2), max(2.0, mde * 2), 12)
        sizes = [sample_size_continuous(sigma, x, alpha=alpha, power=power) for x in grid]
        x = grid
        x_title = "Absolute MDE"
    else:
        low = max(0.65, hazard_ratio - 0.15) if hazard_ratio < 1 else max(1.02, hazard_ratio - 0.08)
        high = min(0.98, hazard_ratio + 0.08) if hazard_ratio < 1 else min(1.35, hazard_ratio + 0.15)
        grid = np.linspace(low, high, 12)
        sizes = [sample_size_survival(x, event_rate, alpha=alpha, power=power) for x in grid]
        x = grid
        x_title = "Hazard ratio"

    fig = px.line(
        x=x,
        y=sizes,
        markers=True,
        color_discrete_sequence=[PALETTE[3]],
        labels={"x": x_title, "y": "Required sample size"},
    )
    fig.update_layout(
        title="Effect-Size Sensitivity",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def familywise_metric_table(
    df: pl.DataFrame,
    n_per_arm: int,
    uplift_pct: float,
    alpha: float,
    seed: int,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    indices = np.arange(df.height)
    replace = n_per_arm > df.height
    control_idx = rng.choice(indices, size=n_per_arm, replace=replace)
    treatment_idx = rng.choice(indices, size=n_per_arm, replace=replace)
    control_df = df[control_idx]
    treatment_df = df[treatment_idx]

    rows = []
    raw_p = []
    for idx, (metric, meta) in enumerate(SIM_METRICS.items()):
        control = control_df[metric].to_numpy().astype(int)
        treatment = treatment_df[metric].to_numpy().astype(int)
        target = float(np.clip(control.mean() * (1 + uplift_pct / 100), 0.001, 0.999))
        shifted = _shift_binary_rate(treatment, target, np.random.default_rng(seed + idx + 1))
        result = proportion_z_test(
            int(control.sum()),
            len(control),
            int(shifted.sum()),
            len(shifted),
            alpha=alpha,
        )
        raw_p.append(result["p_value"])
        rows.append(
            {
                "metric": metric,
                "label": meta["label"],
                "control_rate": result["control_rate"],
                "treatment_rate": result["treatment_rate"],
                "absolute_diff": result["absolute_diff"],
                "p_raw": result["p_value"],
            }
        )

    holm = adjust_pvalues(raw_p, method="holm")
    bh = adjust_pvalues(raw_p, method="bh")
    for row, p_holm, p_bh in zip(rows, holm, bh):
        row["p_holm"] = p_holm
        row["p_bh"] = p_bh
        row["holm_significant"] = bool(p_holm < alpha)
        row["bh_significant"] = bool(p_bh < alpha)

    return pl.DataFrame(rows)


def bootstrap_summary(control: np.ndarray, treatment: np.ndarray, seed: int) -> dict:
    return bootstrap_uplift_ci(control, treatment, confidence=0.95, n_boot=2000, seed=seed)
