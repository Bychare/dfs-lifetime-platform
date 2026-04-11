"""
Reusable Dash layout components: KPI cards, section headers, etc.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html


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


def glossary_accordion(title: str, markdown_text: str) -> dbc.Accordion:
    """Collapsed glossary block with Markdown content."""
    return dbc.Accordion(
        [dbc.AccordionItem(dcc.Markdown(markdown_text, className="small"), title=title)],
        start_collapsed=True,
        className="mb-3",
    )


def methodological_notes(markdown_text: str, border_color: str = "info") -> html.Div:
    """Styled methodological notes card."""
    return html.Div(
        [
            section_header("Methodological Notes"),
            dbc.Card(
                dbc.CardBody(dcc.Markdown(markdown_text, className="small")),
                className=f"mb-4 border-{border_color}",
            ),
        ]
    )


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
