"""Module 3: A/B Test Design & Analysis Engine."""

import dash
from components.layout_utils import placeholder_page

dash.register_page(__name__, path="/ab-testing", name="A/B Testing", order=2)

layout = placeholder_page(
    "A/B Test Engine",
    "Sample size calculator (Lehr, Schoenfeld), Bayesian A/B analysis, "
    "sequential testing with O'Brien–Fleming boundaries. Week 2 priority.",
)
