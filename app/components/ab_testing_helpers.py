"""Helpers and constants for the A/B testing page."""

from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

try:
    from components.data_loader import get_players
    from components.stats import (
        sample_size_continuous,
        sample_size_proportions,
        sample_size_survival,
    )
except ImportError:  # pragma: no cover - fallback for package-style imports in tests
    from app.components.data_loader import get_players
    from app.components.stats import (
        sample_size_continuous,
        sample_size_proportions,
        sample_size_survival,
    )

PALETTE = px.colors.qualitative.Set2

SIM_METRICS = {
    "retained_30d": {
        "label": "30-day retention",
        "description": "Player stayed active for at least 30 days after first contest.",
    },
    "retained_60d": {
        "label": "60-day retention",
        "description": "Stricter retention endpoint for stickier cohorts.",
    },
    "reached_10_contests": {
        "label": "Reached 10 contests",
        "description": "Useful milestone KPI when churn is heavily censored.",
    },
    "profitable_player": {
        "label": "Positive net P&L",
        "description": "Share of players finishing the season with positive net result.",
    },
}

SEGMENTS = {
    "all": "All players",
    "multisport": "Multi-sport players",
    "nfl_only": "NFL-only players",
    "high_risk": "High-risk quartile",
    "low_risk": "Low-risk quartile",
    "high_buyin": "High buy-in quartile",
}

AB_GLOSSARY_MD = """
| Term | Meaning |
|---|---|
| **A/B test** | Randomized controlled experiment comparing a control variant with a treatment variant. |
| **Control** | Existing product experience used as the benchmark. |
| **Treatment** | New experience whose effect we want to estimate. |
| **Baseline rate** | Expected KPI value in control before launching the test. |
| **MDE** | Minimum Detectable Effect — the smallest effect size worth powering the experiment for. |
| **Alpha** | Type I error rate. Probability of a false positive if there is truly no effect. |
| **Power** | Probability of detecting a real effect of at least the chosen size. Equal to 1 − Type II error. |
| **Hazard ratio (HR)** | Relative event rate in a survival-style endpoint. HR < 1 means slower churn; HR > 1 means faster churn. |
| **z-test** | Frequentist test comparing two proportions under a normal approximation. |
| **95% CI** | Interval of plausible values for the estimated uplift. If it crosses 0, the result is not clearly directional. |
| **Posterior** | Bayesian distribution of possible treatment effects after combining prior beliefs with observed data. |
| **P(Treat > Ctrl)** | Posterior probability that the treatment arm truly outperforms control. |
| **Expected loss** | Expected downside from shipping treatment if it is actually worse than control. |
| **Sequential testing** | Monitoring the test at interim looks rather than only once at the end. |
| **O'Brien-Fleming boundary** | Conservative early stopping rule. Very high threshold at early looks, approaching the usual cutoff near the end. |
| **Information fraction** | Share of the planned sample already observed at a given interim look. |
"""

AB_NOTES_MD = """
**Planning assumptions.** The sample-size calculator uses normal approximations:
two-proportion design for binary KPIs, two-sample design for continuous KPIs,
and Schoenfeld's event-based approximation for survival endpoints. These are
appropriate for early planning, but final design choices should still be
checked against real traffic allocation, variance, and business constraints.

**What the simulator is and is not.** The simulator is not replaying a true
historical randomized experiment. Instead, it takes an empirical baseline from
the DFS cohort and injects a configurable treatment effect. That makes it good
for intuition-building, stakeholder demos, and interview storytelling, but not
for validating a real product decision.

**Frequentist and Bayesian outputs answer different questions.** The z-test and
its confidence interval quantify how surprising the observed difference would be
under a null hypothesis of no effect. The Bayesian block instead estimates a
posterior distribution of uplift and reports the probability that treatment
beats control. Showing both is useful in portfolio work because many product
teams mix classical experimentation with decision-theoretic framing.

**Sequential monitoring prevents naive peeking.** If we repeatedly test the
same experiment with a fixed 0.05 threshold, false positives inflate. The
O'Brien-Fleming design keeps early stopping possible, but demands much stronger
evidence at early looks. This mirrors real experimentation platforms where
product teams want to monitor progress without invalidating inference.

**Binary KPI simplification.** The current simulator focuses on binary outcomes
such as retention, milestone completion, or profitability flags. This keeps the
demo interpretable and aligned with the project narrative. A natural next step
would be adding continuous metrics like revenue per player or contest fees.
"""


@lru_cache(maxsize=1)
def ab_frame() -> pd.DataFrame:
    """Cached analytical frame used by the A/B testing page."""
    df = get_players().copy()
    df["retained_30d"] = (df["duration_days"] >= 30).astype(int)
    df["retained_60d"] = (df["duration_days"] >= 60).astype(int)
    df["reached_10_contests"] = (df["nCont"] >= 10).astype(int)
    df["profitable_player"] = (df["net_pnl"] > 0).astype(int)
    return df


def segment_slice(df: pd.DataFrame, segment: str) -> pd.DataFrame:
    """Return the requested player segment."""
    if segment == "multisport":
        return df[df["is_multisport"] == 1]
    if segment == "nfl_only":
        return df[df["is_multisport"] == 0]
    if segment == "high_risk":
        return df[df["risk_quartile"] == "Q4 (High)"]
    if segment == "low_risk":
        return df[df["risk_quartile"] == "Q1 (Low)"]
    if segment == "high_buyin":
        return df[df["buyin_quartile"] == "Q4 (High)"]
    return df


def pct(value: float, digits: int = 1) -> str:
    """Format a proportion as a percent string."""
    return f"{value * 100:.{digits}f}%"


def safe_float(value, default: float) -> float:
    """Best-effort float conversion with fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default: int) -> int:
    """Best-effort int conversion with fallback."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def simulate_binary_experiment(
    values: pd.Series,
    n_per_arm: int,
    uplift_pct: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Bootstrap two binary arms and inject a target uplift into treatment."""
    rng = np.random.default_rng(seed)
    baseline = float(values.mean())
    target = float(np.clip(baseline * (1 + uplift_pct / 100), 0.001, 0.999))

    replace = n_per_arm > len(values)
    control = rng.choice(values.to_numpy(dtype=int), size=n_per_arm, replace=replace)
    treatment = rng.choice(values.to_numpy(dtype=int), size=n_per_arm, replace=replace)

    current = float(treatment.mean())
    if current < target:
        zero_idx = np.where(treatment == 0)[0]
        flip_prob = min(1.0, (target - current) / max(1 - current, 1e-9))
        flips = zero_idx[rng.random(len(zero_idx)) < flip_prob]
        treatment[flips] = 1
    elif current > target:
        one_idx = np.where(treatment == 1)[0]
        flip_prob = min(1.0, (current - target) / max(current, 1e-9))
        flips = one_idx[rng.random(len(one_idx)) < flip_prob]
        treatment[flips] = 0

    return control, treatment, baseline, target


def _arm_ci(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    rate = successes / total
    z = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(rate * (1 - rate) / total)
    return max(rate - z * se, 0.0), min(rate + z * se, 1.0)


def rate_bar_figure(control: np.ndarray, treatment: np.ndarray, alpha: float) -> go.Figure:
    """Observed control/treatment rates with Wald confidence intervals."""
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
    """Posterior histogram of treatment uplift."""
    fig = px.histogram(
        x=diff_draws * 100,
        nbins=50,
        color_discrete_sequence=[PALETTE[2]],
        labels={"x": "Treatment - Control (pp)"},
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#495057")
    fig.update_layout(
        title="Posterior Distribution of Uplift",
        template="plotly_white",
        bargap=0.05,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def sequential_figure(seq_df: pd.DataFrame) -> go.Figure:
    """Observed sequential z-statistics against stopping boundaries."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=seq_df["look"],
        y=seq_df["z_stat"],
        mode="lines+markers",
        name="Observed z",
        line=dict(color=PALETTE[0], width=3),
    ))
    fig.add_trace(go.Scatter(
        x=seq_df["look"],
        y=seq_df["z_boundary"],
        mode="lines+markers",
        name="Upper OBF bound",
        line=dict(color=PALETTE[1], dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=seq_df["look"],
        y=-seq_df["z_boundary"],
        mode="lines+markers",
        name="Lower OBF bound",
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
    """Sensitivity curve for the sample-size calculator."""
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
        if hazard_ratio < 1:
            low = max(0.65, hazard_ratio - 0.15)
            high = min(0.98, hazard_ratio + 0.08)
        else:
            low = max(1.02, hazard_ratio - 0.08)
            high = min(1.35, hazard_ratio + 0.15)
        grid = np.linspace(low, high, 12)
        sizes = [sample_size_survival(x, event_rate, alpha=alpha, power=power) for x in grid]
        x = grid
        x_title = "Hazard ratio"

    fig = px.line(
        x=x,
        y=sizes,
        markers=True,
        color_discrete_sequence=[PALETTE[3]],
        labels={"x": x_title, "y": "Required sample"},
    )
    fig.update_layout(
        title="Sensitivity to Effect Size",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig
