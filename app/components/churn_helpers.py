"""Helpers for the leakage-free churn modeling dashboard."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from catboost import CatBoostClassifier
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from components.data_loader import get_players
from components.plots import PALETTE

SEASON_START = pd.Timestamp("2014-08-22")
MODEL_FEATURES = ["RiskScore", "age", "has_age", "cohort_day"]
FEATURE_LABELS = {
    "RiskScore": "RiskScore",
    "age": "Age",
    "has_age": "Age available",
    "cohort_day": "Signup timing",
}

CHURN_GLOSSARY_MD = """
- **Leakage-free model**: uses only information available at or near the player's first contest, not full-season aggregates.
- **ROC-AUC**: ranking quality across all classification thresholds.
- **PR-AUC**: precision-recall tradeoff; especially useful when churn is less common than retention.
- **Brier score**: probability accuracy; lower is better.
- **Calibration**: whether predicted churn probabilities match observed churn rates.
- **Lift by decile**: how concentrated churn becomes in the highest-risk score buckets.
- **What-if analysis**: scenario-based estimate of how risk changes for a hypothetical player profile.
"""

CHURN_NOTES_MD = """
- This module uses a deliberately conservative leakage-free setup built from `RiskScore`, age availability, age, and signup timing.
- Full-season behavior models are stronger on paper, but they are not suitable for real prediction because they use information that would only be known after much of the season has already happened.
- In this dataset the clean leakage-free setup reaches roughly `0.73-0.74 ROC-AUC`, while a full-season benchmark is around `0.82-0.83 ROC-AUC`.
- Driver comparisons are shown as scenario deltas, not as causal effects.
"""

DEFAULT_SPLIT_SEED = 42
DEFAULT_TEST_SIZE = 0.20
DEFAULT_CATBOOST_ITERS = 250


def churn_model_frame(df=None) -> pd.DataFrame:
    source = get_players() if df is None else df
    pdf = source.select(["UserID", "Date1st", "is_churned", "RiskScore", "age", "has_age"]).to_pandas()
    pdf["Date1st"] = pd.to_datetime(pdf["Date1st"])
    pdf["cohort_day"] = (pdf["Date1st"] - SEASON_START).dt.days.clip(lower=0)
    pdf["target"] = pdf["is_churned"].astype(int)
    return pdf[["UserID", "Date1st", "target", *MODEL_FEATURES]].copy()


def _logreg_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                MODEL_FEATURES,
            )
        ]
    )
    return Pipeline(
        [
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=3000, solver="lbfgs")),
        ]
    )


def _catboost_model(random_state: int, iterations: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=iterations,
        depth=5,
        learning_rate=0.05,
        l2_leaf_reg=5,
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
    )


def _metric_row(model: str, y_true: np.ndarray, prob: np.ndarray) -> dict:
    top_cut = float(np.quantile(prob, 0.9))
    top_mask = prob >= top_cut
    baseline = float(np.mean(y_true))
    top_rate = float(np.mean(y_true[top_mask])) if top_mask.any() else baseline
    lift = top_rate / baseline if baseline > 0 else np.nan
    return {
        "Model": model,
        "ROC-AUC": float(roc_auc_score(y_true, prob)),
        "PR-AUC": float(average_precision_score(y_true, prob)),
        "Brier": float(brier_score_loss(y_true, prob)),
        "Baseline churn": baseline,
        "Top-decile churn": top_rate,
        "Top-decile lift": float(lift),
    }


def _curve_frame(y_true: np.ndarray, prob: np.ndarray, model: str) -> dict[str, pd.DataFrame]:
    fpr, tpr, _ = roc_curve(y_true, prob)
    precision, recall, _ = precision_recall_curve(y_true, prob)
    frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=8, strategy="quantile")
    return {
        "roc": pd.DataFrame({"x": fpr, "y": tpr, "Model": model}),
        "pr": pd.DataFrame({"x": recall, "y": precision, "Model": model}),
        "calibration": pd.DataFrame({"x": mean_pred, "y": frac_pos, "Model": model}),
    }


def _decile_frame(y_true: np.ndarray, prob: np.ndarray, model: str) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y_true, "prob": prob})
    df["decile"] = pd.qcut(df["prob"].rank(method="first"), q=10, labels=list(range(1, 11)))
    grouped = (
        df.groupby("decile", observed=False)
        .agg(actual_churn=("y_true", "mean"), predicted_churn=("prob", "mean"), players=("y_true", "size"))
        .reset_index()
    )
    grouped["Model"] = model
    return grouped


def _fit_models(
    frame: pd.DataFrame,
    random_state: int = DEFAULT_SPLIT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
    catboost_iterations: int = DEFAULT_CATBOOST_ITERS,
) -> dict:
    X = frame[MODEL_FEATURES].copy()
    y = frame["target"].to_numpy()

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y))
    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y[train_idx]
    y_test = y[test_idx]

    logreg = _logreg_pipeline()
    logreg.fit(X_train, y_train)
    prob_logreg = logreg.predict_proba(X_test)[:, 1]

    catboost = _catboost_model(random_state=random_state, iterations=catboost_iterations)
    catboost.fit(X_train, y_train)
    prob_catboost = catboost.predict_proba(X_test)[:, 1]

    metrics = pd.DataFrame(
        [
            _metric_row("Logistic Regression", y_test, prob_logreg),
            _metric_row("CatBoost", y_test, prob_catboost),
        ]
    )

    curves = pd.concat(
        [
            _curve_frame(y_test, prob_logreg, "Logistic Regression")["roc"],
            _curve_frame(y_test, prob_catboost, "CatBoost")["roc"],
        ],
        ignore_index=True,
    )
    pr_curves = pd.concat(
        [
            _curve_frame(y_test, prob_logreg, "Logistic Regression")["pr"],
            _curve_frame(y_test, prob_catboost, "CatBoost")["pr"],
        ],
        ignore_index=True,
    )
    calibration = pd.concat(
        [
            _curve_frame(y_test, prob_logreg, "Logistic Regression")["calibration"],
            _curve_frame(y_test, prob_catboost, "CatBoost")["calibration"],
        ],
        ignore_index=True,
    )
    deciles = pd.concat(
        [
            _decile_frame(y_test, prob_logreg, "Logistic Regression"),
            _decile_frame(y_test, prob_catboost, "CatBoost"),
        ],
        ignore_index=True,
    )

    importance = pd.DataFrame(
        {
            "Feature": [FEATURE_LABELS[col] for col in MODEL_FEATURES],
            "CatBoost importance": catboost.get_feature_importance(),
        }
    ).sort_values("CatBoost importance", ascending=False)

    return {
        "frame": frame,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "logreg": logreg,
        "catboost": catboost,
        "metrics": metrics,
        "roc_curve": curves,
        "pr_curve": pr_curves,
        "calibration": calibration,
        "deciles": deciles,
        "importance": importance,
        "baseline_churn": float(frame["target"].mean()),
    }


def fit_churn_models(
    frame: pd.DataFrame,
    random_state: int = DEFAULT_SPLIT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
    catboost_iterations: int = DEFAULT_CATBOOST_ITERS,
) -> dict:
    return _fit_models(
        frame=frame,
        random_state=random_state,
        test_size=test_size,
        catboost_iterations=catboost_iterations,
    )


@lru_cache(maxsize=1)
def churn_artifacts() -> dict:
    frame = churn_model_frame()
    return _fit_models(frame)


def _profile_defaults(frame: pd.DataFrame) -> dict:
    age_known = frame["age"].dropna()
    return {
        "RiskScore": float(frame["RiskScore"].median()),
        "age": float(age_known.median()),
        "has_age": 1.0,
        "cohort_day": float(frame["cohort_day"].median()),
    }


def driver_effect_table(artifacts: dict | None = None) -> pd.DataFrame:
    bundle = churn_artifacts() if artifacts is None else artifacts
    frame = bundle["frame"]
    reference = _profile_defaults(frame)

    age_known = frame["age"].dropna()
    scenarios = [
        {
            "Feature": "RiskScore",
            "Low": float(frame["RiskScore"].quantile(0.25)),
            "High": float(frame["RiskScore"].quantile(0.75)),
            "Low label": f"{frame['RiskScore'].quantile(0.25):.1f}",
            "High label": f"{frame['RiskScore'].quantile(0.75):.1f}",
        },
        {
            "Feature": "Age",
            "Low": float(age_known.quantile(0.25)),
            "High": float(age_known.quantile(0.75)),
            "Low label": f"{age_known.quantile(0.25):.0f}",
            "High label": f"{age_known.quantile(0.75):.0f}",
        },
        {
            "Feature": "Age available",
            "Low": 0.0,
            "High": 1.0,
            "Low label": "Missing",
            "High label": "Available",
        },
        {
            "Feature": "Signup timing",
            "Low": float(frame["cohort_day"].quantile(0.25)),
            "High": float(frame["cohort_day"].quantile(0.75)),
            "Low label": f"Day {frame['cohort_day'].quantile(0.25):.0f}",
            "High label": f"Day {frame['cohort_day'].quantile(0.75):.0f}",
        },
    ]

    rows = []
    for scenario in scenarios:
        low_profile = reference.copy()
        high_profile = reference.copy()
        if scenario["Feature"] == "Age available":
            low_profile["has_age"] = 0.0
            low_profile["age"] = np.nan
            high_profile["has_age"] = 1.0
            high_profile["age"] = reference["age"]
        elif scenario["Feature"] == "Age":
            low_profile["age"] = scenario["Low"]
            high_profile["age"] = scenario["High"]
            low_profile["has_age"] = 1.0
            high_profile["has_age"] = 1.0
        elif scenario["Feature"] == "RiskScore":
            low_profile["RiskScore"] = scenario["Low"]
            high_profile["RiskScore"] = scenario["High"]
        else:
            low_profile["cohort_day"] = scenario["Low"]
            high_profile["cohort_day"] = scenario["High"]

        low_df = pd.DataFrame([low_profile])[MODEL_FEATURES]
        high_df = pd.DataFrame([high_profile])[MODEL_FEATURES]
        low_lr = float(bundle["logreg"].predict_proba(low_df)[:, 1][0])
        high_lr = float(bundle["logreg"].predict_proba(high_df)[:, 1][0])
        low_cb = float(bundle["catboost"].predict_proba(low_df)[:, 1][0])
        high_cb = float(bundle["catboost"].predict_proba(high_df)[:, 1][0])
        rows.append(
            {
                "Feature": scenario["Feature"],
                "Low case": scenario["Low label"],
                "High case": scenario["High label"],
                "LogReg delta pp": (high_lr - low_lr) * 100,
                "CatBoost delta pp": (high_cb - low_cb) * 100,
            }
        )

    effect_df = pd.DataFrame(rows)
    effect_df["max_abs"] = effect_df[["LogReg delta pp", "CatBoost delta pp"]].abs().max(axis=1)
    return effect_df.sort_values("max_abs", ascending=False).drop(columns="max_abs")


def score_profile(
    risk_score: float,
    age: float | None,
    has_age: bool,
    cohort_day: float,
    artifacts: dict | None = None,
) -> dict:
    bundle = churn_artifacts() if artifacts is None else artifacts
    profile = pd.DataFrame(
        [
            {
                "RiskScore": float(risk_score),
                "age": float(age) if has_age and age is not None else np.nan,
                "has_age": 1.0 if has_age else 0.0,
                "cohort_day": float(cohort_day),
            }
        ]
    )[MODEL_FEATURES]
    prob_lr = float(bundle["logreg"].predict_proba(profile)[:, 1][0])
    prob_cb = float(bundle["catboost"].predict_proba(profile)[:, 1][0])
    baseline = bundle["baseline_churn"]
    return {
        "Logistic Regression": prob_lr,
        "CatBoost": prob_cb,
        "Baseline": baseline,
    }


def metrics_table(artifacts: dict | None = None) -> pd.DataFrame:
    bundle = churn_artifacts() if artifacts is None else artifacts
    return bundle["metrics"].copy()


def roc_figure(artifacts: dict | None = None) -> go.Figure:
    bundle = churn_artifacts() if artifacts is None else artifacts
    curve_df = bundle["roc_curve"]
    fig = go.Figure()
    for idx, model in enumerate(curve_df["Model"].unique()):
        subset = curve_df[curve_df["Model"] == model]
        auc = float(bundle["metrics"].loc[bundle["metrics"]["Model"] == model, "ROC-AUC"].iloc[0])
        fig.add_trace(
            go.Scatter(
                x=subset["x"],
                y=subset["y"],
                mode="lines",
                line=dict(color=PALETTE[idx % len(PALETTE)], width=3),
                name=f"{model} (AUC {auc:.3f})",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            name="Random",
        )
    )
    fig.update_layout(
        title="ROC Curve",
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(title_text="False positive rate")
    fig.update_yaxes(title_text="True positive rate")
    return fig


def calibration_figure(artifacts: dict | None = None) -> go.Figure:
    bundle = churn_artifacts() if artifacts is None else artifacts
    cal_df = bundle["calibration"]
    fig = go.Figure()
    for idx, model in enumerate(cal_df["Model"].unique()):
        subset = cal_df[cal_df["Model"] == model]
        fig.add_trace(
            go.Scatter(
                x=subset["x"],
                y=subset["y"],
                mode="lines+markers",
                line=dict(color=PALETTE[idx % len(PALETTE)], width=3),
                marker=dict(size=8),
                name=model,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            name="Perfect calibration",
        )
    )
    fig.update_layout(
        title="Calibration",
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(title_text="Predicted churn probability")
    fig.update_yaxes(title_text="Observed churn rate")
    return fig


def decile_lift_figure(artifacts: dict | None = None) -> go.Figure:
    bundle = churn_artifacts() if artifacts is None else artifacts
    decile_df = bundle["deciles"]
    baseline = bundle["baseline_churn"]
    fig = go.Figure()
    for idx, model in enumerate(decile_df["Model"].unique()):
        subset = decile_df[decile_df["Model"] == model].sort_values("decile")
        fig.add_trace(
            go.Scatter(
                x=subset["decile"].astype(int),
                y=subset["actual_churn"],
                mode="lines+markers",
                line=dict(color=PALETTE[idx % len(PALETTE)], width=3),
                marker=dict(size=8),
                name=model,
                customdata=subset[["predicted_churn", "players"]],
                hovertemplate=(
                    "Risk decile=%{x}<br>"
                    "Observed churn=%{y:.1%}<br>"
                    "Predicted churn=%{customdata[0]:.1%}<br>"
                    "Players=%{customdata[1]:,.0f}<extra></extra>"
                ),
            )
        )
    fig.add_hline(y=baseline, line_dash="dash", line_color="gray", annotation_text=f"Baseline {baseline:.1%}")
    fig.update_layout(
        title="Observed churn by predicted risk decile",
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(title_text="Predicted risk decile (1 = lowest, 10 = highest)")
    fig.update_yaxes(title_text="Observed churn rate", tickformat=".0%")
    return fig


def driver_effect_figure(effect_df: pd.DataFrame | None = None) -> go.Figure:
    df = driver_effect_table() if effect_df is None else effect_df
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=df["Feature"],
            x=df["LogReg delta pp"],
            orientation="h",
            name="Logistic Regression",
            marker_color=PALETTE[0],
            text=[f"{value:.1f} pp" for value in df["LogReg delta pp"]],
            textposition="outside",
            customdata=np.column_stack([df["Low case"], df["High case"]]),
            hovertemplate="Feature=%{y}<br>Low=%{customdata[0]}<br>High=%{customdata[1]}<br>Delta=%{x:.1f} pp<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=df["Feature"],
            x=df["CatBoost delta pp"],
            orientation="h",
            name="CatBoost",
            marker_color=PALETTE[1],
            text=[f"{value:.1f} pp" for value in df["CatBoost delta pp"]],
            textposition="outside",
            customdata=np.column_stack([df["Low case"], df["High case"]]),
            hovertemplate="Feature=%{y}<br>Low=%{customdata[0]}<br>High=%{customdata[1]}<br>Delta=%{x:.1f} pp<extra></extra>",
        )
    )
    fig.update_layout(
        title="Scenario-based driver sensitivity",
        template="plotly_white",
        margin=dict(l=100, r=20, t=50, b=40),
        barmode="group",
    )
    fig.update_xaxes(title_text="Change in predicted churn risk (percentage points)")
    fig.update_yaxes(title_text="")
    return fig


def importance_figure(artifacts: dict | None = None) -> go.Figure:
    bundle = churn_artifacts() if artifacts is None else artifacts
    imp_df = bundle["importance"].sort_values("CatBoost importance")
    fig = go.Figure(
        go.Bar(
            x=imp_df["CatBoost importance"],
            y=imp_df["Feature"],
            orientation="h",
            marker_color=PALETTE[2],
            text=[f"{value:.1f}" for value in imp_df["CatBoost importance"]],
            textposition="outside",
            name="CatBoost importance",
        )
    )
    fig.update_layout(
        title="CatBoost feature importance",
        template="plotly_white",
        margin=dict(l=100, r=20, t=50, b=40),
    )
    fig.update_xaxes(title_text="Importance")
    fig.update_yaxes(title_text="")
    return fig


def pr_curve_figure(artifacts: dict | None = None) -> go.Figure:
    bundle = churn_artifacts() if artifacts is None else artifacts
    curve_df = bundle["pr_curve"]
    baseline = bundle["baseline_churn"]
    fig = go.Figure()
    for idx, model in enumerate(curve_df["Model"].unique()):
        subset = curve_df[curve_df["Model"] == model]
        auc = float(bundle["metrics"].loc[bundle["metrics"]["Model"] == model, "PR-AUC"].iloc[0])
        fig.add_trace(
            go.Scatter(
                x=subset["x"],
                y=subset["y"],
                mode="lines",
                line=dict(color=PALETTE[idx % len(PALETTE)], width=3),
                name=f"{model} (PR-AUC {auc:.3f})",
            )
        )
    fig.add_hline(y=baseline, line_dash="dash", line_color="gray", annotation_text=f"Baseline {baseline:.1%}")
    fig.update_layout(
        title="Precision-Recall Curve",
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(title_text="Recall")
    fig.update_yaxes(title_text="Precision")
    return fig


def metrics_summary_text(artifacts: dict | None = None) -> str:
    bundle = churn_artifacts() if artifacts is None else artifacts
    metrics = bundle["metrics"].sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    best = metrics.iloc[0]
    other = metrics.iloc[1]
    return (
        f"Best holdout ranking comes from {best['Model']} with ROC-AUC {best['ROC-AUC']:.3f}, "
        f"PR-AUC {best['PR-AUC']:.3f}, and top-decile lift {best['Top-decile lift']:.2f}x. "
        f"For context, a non-leakage benchmark built on full-season behavior reached about 0.83 ROC-AUC, "
        f"so the clean setup gives up roughly 0.08 ROC-AUC to stay decision-usable. "
        f"{other['Model']} remains close enough to serve as an interpretable baseline."
    )


def what_if_summary_text(scores: dict) -> str:
    avg_risk = (scores["Logistic Regression"] + scores["CatBoost"]) / 2
    baseline = scores["Baseline"]
    if avg_risk >= baseline + 0.10:
        band = "high-risk"
    elif avg_risk >= baseline + 0.03:
        band = "elevated-risk"
    elif avg_risk <= baseline - 0.03:
        band = "lower-risk"
    else:
        band = "near-baseline"
    return (
        f"This profile looks {band}: average predicted churn is {avg_risk:.1%} versus a baseline churn rate of "
        f"{baseline:.1%}. Logistic Regression estimates {scores['Logistic Regression']:.1%}, "
        f"while CatBoost estimates {scores['CatBoost']:.1%}."
    )


def risk_marks(frame: pd.DataFrame | None = None) -> dict[int, str]:
    df = churn_model_frame() if frame is None else frame
    marks = {}
    for quantile in [0.05, 0.25, 0.50, 0.75, 0.95]:
        value = int(round(float(df["RiskScore"].quantile(quantile))))
        marks[value] = str(value)
    return marks
