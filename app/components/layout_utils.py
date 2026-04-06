"""
Reusable Dash layout components: KPI cards, section headers, etc.
"""

import dash_bootstrap_components as dbc
from dash import html


def kpi_card(title: str, value: str, color: str = "primary") -> dbc.Card:
    """Small KPI card with a title and big number."""
    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="text-muted mb-1", style={"fontSize": "0.8rem"}),
            html.H4(value, className=f"text-{color} mb-0"),
        ]),
        className="shadow-sm",
    )


def section_header(title: str, subtitle: str = "") -> html.Div:
    """Section header with optional subtitle."""
    children = [html.H4(title, className="mb-1")]
    if subtitle:
        children.append(html.P(subtitle, className="text-muted"))
    return html.Div(children, className="mb-3 mt-4")


def placeholder_page(title: str, description: str) -> html.Div:
    """Placeholder for modules not yet implemented."""
    return html.Div([
        html.H3(title, className="mt-4"),
        dbc.Alert(
            [
                html.I(className="bi bi-cone-striped me-2"),
                description,
            ],
            color="info",
            className="mt-3",
        ),
    ])
