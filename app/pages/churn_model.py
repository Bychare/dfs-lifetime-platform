"""Module 5: Churn Model."""

import dash
from components.layout_utils import placeholder_page

dash.register_page(__name__, path="/churn-model", name="Churn Model", order=4)

layout = placeholder_page(
    "Churn Prediction Model",
    "Logistic regression plus CatBoost, ROC/PR-AUC, calibration, SHAP values, "
    "and an interactive what-if calculator. Planned for the next build cycle.",
)
