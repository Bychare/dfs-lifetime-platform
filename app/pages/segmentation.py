"""Module 4: Segmentation."""

import dash
from components.layout_utils import placeholder_page

dash.register_page(__name__, path="/segmentation", name="Segmentation", order=3)

layout = placeholder_page(
    "Segment Analysis",
    "Kruskal-Wallis plus Dunn's test by contest type, two-way ANOVA, and "
    "RiskScore analysis with LOWESS. Planned for the next build cycle.",
)
