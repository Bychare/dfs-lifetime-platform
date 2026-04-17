"""
Reusable Plotly figure factories for the DFS analytics platform.
"""

import numpy as np
import polars as pl
import plotly.express as px
import plotly.graph_objects as go

PALETTE = px.colors.qualitative.Set2
BG_TRANSPARENT = "rgba(0,0,0,0)"


def _base_layout(fig: go.Figure, title: str = "") -> go.Figure:
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


def fig_histogram(series: pl.Series, name: str, log_x: bool = False, nbins: int = 50) -> go.Figure:
    values = series.drop_nulls().to_numpy()
    values = np.log1p(values) if log_x else values
    xlabel = f"log(1 + {name})" if log_x else name

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values, nbinsx=nbins, marker_color=PALETTE[0], opacity=0.85))
    _base_layout(fig, f"Distribution: {name}")
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text="Players")
    return fig


def fig_box_by_group(
    df: pl.DataFrame,
    value_col: str,
    group_col: str,
    log_y: bool = False,
    value_label: str | None = None,
    group_label: str | None = None,
) -> go.Figure:
    plot_df = df.select([group_col, value_col]).drop_nulls().to_pandas()
    fig = px.box(
        plot_df,
        x=group_col,
        y=value_col,
        color=group_col,
        color_discrete_sequence=PALETTE,
        log_y=log_y,
    )
    display_value = value_label or value_col
    display_group = group_label or group_col
    _base_layout(fig, f"{display_value} by {display_group}")
    fig.update_yaxes(title_text=display_value)
    fig.update_xaxes(title_text=display_group)
    fig.update_layout(showlegend=False)
    return fig


def fig_correlation_heatmap(df: pl.DataFrame, cols: list[str]) -> go.Figure:
    corr = df.select(cols).to_pandas().corr(method="spearman")
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
        )
    )
    _base_layout(fig, "Spearman Correlation Matrix")
    fig.update_layout(height=500, width=600)
    return fig


def fig_us_state_map(state_counts: pl.DataFrame) -> go.Figure:
    fig = px.choropleth(
        state_counts.to_pandas(),
        locations="state",
        locationmode="USA-states",
        color="players",
        scope="usa",
        color_continuous_scale="Blues",
        labels={"players": "Players"},
    )
    _base_layout(fig, "Players by US State")
    fig.update_layout(geo=dict(bgcolor=BG_TRANSPARENT))
    return fig


def fig_kpi_cards(df: pl.DataFrame) -> dict:
    total_players = df.height
    churn_rate = df["is_churned"].mean()
    median_fees = df["TotFees"].median()
    median_net = df["net_pnl"].median()
    pct_losers = df.select((pl.col("net_pnl") < 0).mean()).item()
    median_days = df["nDays"].median()
    multisport_pct = df["is_multisport"].mean()

    return {
        "total_players": total_players,
        "churn_rate": f"{churn_rate * 100:.1f}%",
        "median_fees": f"${median_fees:,.0f}",
        "median_net": f"${median_net:,.2f}",
        "pct_losers": f"{pct_losers * 100:.1f}%",
        "median_days": f"{median_days:.0f}",
        "multisport_pct": f"{multisport_pct * 100:.1f}%",
    }
