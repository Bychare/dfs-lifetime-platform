"""
DFS Lifetime & Experimentation Platform — main Dash app.

Run locally:
    python app/app.py

Production:
    gunicorn app.app:server -b 0.0.0.0:8050
"""

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, page_container

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder="pages",
    external_stylesheets=[
        dbc.themes.FLATLY,
        dbc.icons.BOOTSTRAP,
    ],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "DFS Analytics Platform"
server = app.server  # for gunicorn

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
sidebar = dbc.Nav(
    [
        dbc.NavLink(
            [html.I(className=f"bi {icon} me-2"), label],
            href=href,
            active="exact",
        )
        for label, href, icon in [
            ("Overview & EDA",   "/",              "bi-bar-chart-line"),
            ("Survival Analysis", "/survival",     "bi-heart-pulse"),
            ("A/B Testing",      "/ab-testing",    "bi-toggles"),
            ("Segmentation",     "/segmentation",  "bi-diagram-3"),
            ("Churn Model",      "/churn-model",   "bi-cpu"),
        ]
    ],
    vertical=True,
    pills=True,
    className="flex-column",
)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                # Sidebar
                dbc.Col(
                    [
                        html.H5("DFS Analytics", className="text-primary mb-3 mt-3"),
                        html.Hr(),
                        sidebar,
                        html.Hr(),
                        html.Small(
                            [
                                "Data: ",
                                html.A(
                                    "Transparency Project",
                                    href="http://thetransparencyproject.org/",
                                    target="_blank",
                                ),
                                html.Br(),
                                "Nelson et al. (2019)",
                            ],
                            className="text-muted",
                        ),
                    ],
                    width=2,
                    className="bg-light vh-100 position-fixed",
                    style={"overflowY": "auto", "paddingTop": "0.5rem"},
                ),
                # Main content
                dbc.Col(
                    page_container,
                    width=10,
                    className="ms-auto",
                    style={"paddingTop": "1rem", "paddingBottom": "2rem"},
                ),
            ]
        ),
    ],
    fluid=True,
)


if __name__ == "__main__":
    app.run(debug=True, port=8050)
