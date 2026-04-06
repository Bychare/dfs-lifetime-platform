"""Module 5: Churn Prediction Model — LogReg, GBM, SHAP, What-If."""

import dash
from components.layout_utils import placeholder_page

dash.register_page(__name__, path="/churn-model", name="Churn Model", order=4)

layout = placeholder_page(
    "Churn Prediction Model",
    "Logistic regression + CatBoost, ROC/PR-AUC, calibration plot, "
    "SHAP values, interactive what-if calculator. Week 3 priority.",
)
