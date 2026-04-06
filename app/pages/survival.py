"""Module 2: Survival Analysis — Kaplan-Meier, Cox PH, cohort retention."""

import dash
from components.layout_utils import placeholder_page

dash.register_page(__name__, path="/survival", name="Survival Analysis", order=1)

layout = placeholder_page(
    "Survival Analysis",
    "Kaplan–Meier curves, Cox Proportional Hazards model, and cohort retention heatmap. "
    "Coming next — this is the Week 1 priority after EDA.",
)
