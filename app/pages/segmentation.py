"""Module 4: Segmentation — ANOVA / Kruskal-Wallis pipeline."""

import dash
from components.layout_utils import placeholder_page

dash.register_page(__name__, path="/segmentation", name="Segmentation", order=3)

layout = placeholder_page(
    "Segment Analysis",
    "Kruskal–Wallis + Dunn's test by contest type, two-way ANOVA, "
    "RiskScore analysis with LOWESS. Week 3 priority.",
)
