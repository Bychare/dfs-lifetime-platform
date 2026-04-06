"""
Reusable Plotly figure factories for the DFS analytics platform.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
PALETTE = px.colors.qualitative.Set2
BG_TRANSPARENT = "rgba(0,0,0,0)"


def _base_layout(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply consistent styling to a figure."""
    fig.update_layout(
        title=title,
        template="plotly_white",
        paper_bgcolor=BG_TRANSPARENT,
        plot_bgcolor=BG_TRANSPARENT,
        font=dict(family="Inter, system-ui, sans-serif", size=12),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------

def fig_histogram(series: pd.Series, name: str, log_x: bool = False, nbins: int = 50) -> go.Figure:
    """Histogram with optional log-x scale."""
    fig = go.Figure()
    values = np.log1p(series) if log_x else series
    xlabel = f"log(1 + {name})" if log_x else name
    fig.add_trace(go.Histogram(x=values, nbinsx=nbins, marker_color=PALETTE[0], opacity=0.85))
    _base_layout(fig, f"Distribution of {name}")
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text="Count")
    return fig


def fig_box_by_group(df: pd.DataFrame, value_col: str, group_col: str, log_y: bool = False) -> go.Figure:
    """Box plot of value_col split by group_col."""
    fig = px.box(
        df, x=group_col, y=value_col, color=group_col,
        color_discrete_sequence=PALETTE, log_y=log_y,
    )
    _base_layout(fig, f"{value_col} by {group_col}")
    fig.update_layout(showlegend=False)
    return fig


def fig_correlation_heatmap(df: pd.DataFrame, cols: list[str]) -> go.Figure:
    """Correlation heatmap for selected numeric columns."""
    corr = df[cols].corr(method="spearman")
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
        )
    )
    _base_layout(fig, "Spearman Correlation Matrix")
    fig.update_layout(height=500, width=600)
    return fig


def fig_us_state_map(state_counts: pd.Series) -> go.Figure:
    """Choropleth map of player counts by US state."""
    # State name -> abbreviation lookup (common ones)
    fig = px.choropleth(
        locations=state_counts.index,
        locationmode="USA-states",
        color=state_counts.values,
        scope="usa",
        color_continuous_scale="Blues",
        labels={"color": "Players"},
    )
    _base_layout(fig, "Players by US State")
    fig.update_layout(geo=dict(bgcolor=BG_TRANSPARENT))
    return fig


def fig_kpi_cards(df: pd.DataFrame) -> dict:
    """Compute key metrics for KPI display."""
    return {
        "total_players": len(df),
        "churn_rate": f"{df['is_churned'].mean() * 100:.1f}%",
        "median_fees": f"${df['TotFees'].median():,.0f}",
        "median_net": f"${df['net_pnl'].median():,.2f}",
        "pct_losers": f"{(df['net_pnl'] < 0).mean() * 100:.1f}%",
        "median_days": f"{df['nDays'].median():.0f}",
        "multisport_pct": f"{df['is_multisport'].mean() * 100:.1f}%",
    }
