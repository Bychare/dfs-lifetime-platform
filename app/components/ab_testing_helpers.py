"""Helpers and constants for the A/B testing page."""

from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

try:
    from components.data_loader import get_players
    from components.stats import (
        adjust_pvalues,
        bootstrap_uplift_ci,
        proportion_z_test,
        sample_size_continuous,
        sample_size_proportions,
        sample_size_survival,
    )
except ImportError:  # pragma: no cover - fallback for package-style imports in tests
    from app.components.data_loader import get_players
    from app.components.stats import (
        adjust_pvalues,
        bootstrap_uplift_ci,
        proportion_z_test,
        sample_size_continuous,
        sample_size_proportions,
        sample_size_survival,
    )

PALETTE = px.colors.qualitative.Set2

SIM_METRICS = {
    "retained_30d": {
        "label": "Удержание 30 дней",
        "description": "Игрок оставался активным минимум 30 дней после первого контеста.",
    },
    "retained_60d": {
        "label": "Удержание 60 дней",
        "description": "Более строгая retention-конечная точка для устойчивых когорт.",
    },
    "reached_10_contests": {
        "label": "Достиг 10 контестов",
        "description": "Полезный milestone KPI, когда churn сильно цензурирован.",
    },
    "profitable_player": {
        "label": "Положительный net P&L",
        "description": "Доля игроков, закончивших сезон с положительным финансовым результатом.",
    },
}

SEGMENTS = {
    "all": "Все игроки",
    "multisport": "Мультиспортивные игроки",
    "nfl_only": "Только NFL",
    "high_risk": "Высокий риск, Q4",
    "low_risk": "Низкий риск, Q1",
    "high_buyin": "Высокий бай-ин, Q4",
}

AB_GLOSSARY_MD = """
| Термин | Значение |
|---|---|
| **A/B-тест** | Рандомизированный контролируемый эксперимент, сравнивающий контрольный и тестовый вариант. |
| **Контроль** | Текущий продуктовый опыт, используемый как база сравнения. |
| **Тест / treatment** | Новая версия продукта, эффект которой мы хотим оценить. |
| **Базовая конверсия** | Ожидаемое значение KPI в контрольной группе до запуска эксперимента. |
| **MDE** | Minimum Detectable Effect — минимальный эффект, под который имеет смысл рассчитывать мощность теста. |
| **Alpha** | Ошибка первого рода. Вероятность ложноположительного вывода при отсутствии эффекта. |
| **Power** | Вероятность обнаружить реальный эффект заданного размера. Равна 1 − β. |
| **Hazard ratio (HR)** | Относительная интенсивность событий в survival-endpoint. HR < 1 означает более медленный churn, HR > 1 — более быстрый. |
| **z-тест** | Частотный тест сравнения двух долей в нормальном приближении. |
| **95% CI** | Интервал правдоподобных значений оценённого uplift. Если он пересекает 0, результат не имеет явного направления. |
| **Posterior** | Байесовское распределение возможных эффектов после объединения prior и наблюдаемых данных. |
| **P(Treat > Ctrl)** | Апостериорная вероятность того, что тестовая группа лучше контрольной. |
| **Expected loss** | Ожидаемый ущерб от выката treatment, если на самом деле он хуже контроля. |
| **Bootstrap CI** | Непараметрический интервал, полученный ресэмплингом наблюдений с возвращением. Полезен как робастная проверка uplift. |
| **Множественные сравнения** | Ситуация, когда одновременно оценивается несколько KPI или сегментов. Без поправки число ложноположительных выводов растёт. |
| **Holm correction** | Пошаговая поправка, контролирующая FWER: вероятность хотя бы одного ложного срабатывания в семействе тестов. |
| **BH correction** | Поправка Benjamini-Hochberg, контролирующая FDR: ожидаемую долю ложных находок среди всех значимых. |
| **Sequential testing** | Мониторинг эксперимента на промежуточных просмотрах, а не только в финале. |
| **Граница O'Brien-Fleming** | Консервативное правило ранней остановки: очень высокий порог в начале и более мягкий к концу теста. |
| **Information fraction** | Доля уже собранной информации относительно полного плана эксперимента. |
"""

AB_NOTES_MD = """
**Планировочные допущения.** Калькулятор размера выборки использует нормальные
аппроксимации: дизайн для двух долей для бинарных KPI, двухвыборочный дизайн
для непрерывных метрик и формулу Шёнфельда для survival-endpoints. Это удобно
для раннего планирования, но финальный дизайн всё равно нужно сверять с реальным
трафиком, дисперсией и бизнес-ограничениями.

**Что симулятор делает и чего не делает.** Симулятор не воспроизводит реальный
исторический рандомизированный эксперимент. Он берёт эмпирический baseline из
DFS-когорты и искусственно добавляет treatment-эффект. Поэтому он хорош для
объяснения логики, демо и портфолио, но не для валидации реального продуктового
решения.

**Частотный и байесовский блоки отвечают на разные вопросы.** z-тест и его
доверительный интервал показывают, насколько неожиданна наблюдаемая разница при
нулевой гипотезе отсутствия эффекта. Байесовский блок, напротив, оценивает
апостериорное распределение uplift и вероятность того, что treatment лучше
control. Для портфолио это сильный плюс, потому что продуктовые команды часто
смешивают классическую экспериментальную статистику и decision-oriented подход.

**Bootstrap и поправки на множественные сравнения добавляют надёжность.**
Bootstrap CI полезен как робастная проверка на случай, когда нормальное
приближение выглядит слишком уверенно. Поправки Holm и BH нужны тогда, когда
мы одновременно смотрим не на один KPI, а на семейство метрик: например,
retention, milestone и profitability.

**Sequential monitoring защищает от наивного peeking.** Если многократно
проверять один и тот же эксперимент с фиксированным порогом 0.05, число ложных
срабатываний растёт. Дизайн O'Brien-Fleming позволяет раннюю остановку, но
требует гораздо более сильных доказательств на ранних просмотрах. Это хорошо
соответствует реальным experimentation platforms.

**Упрощение до бинарных KPI.** Сейчас симулятор сфокусирован на бинарных исходах:
retention, достижение milestone и прибыльность игрока. Это делает модуль более
читаемым и лучше вписывает его в общую историю проекта. Естественное расширение
дальше — добавить непрерывные метрики вроде выручки на игрока или contest fees.
"""


@lru_cache(maxsize=1)
def ab_frame() -> pd.DataFrame:
    """Cached analytical frame used by the A/B testing page."""
    df = get_players().copy()
    df["retained_30d"] = (df["duration_days"] >= 30).astype(int)
    df["retained_60d"] = (df["duration_days"] >= 60).astype(int)
    df["reached_10_contests"] = (df["nCont"] >= 10).astype(int)
    df["profitable_player"] = (df["net_pnl"] > 0).astype(int)
    return df


def segment_slice(df: pd.DataFrame, segment: str) -> pd.DataFrame:
    """Return the requested player segment."""
    if segment == "multisport":
        return df[df["is_multisport"] == 1]
    if segment == "nfl_only":
        return df[df["is_multisport"] == 0]
    if segment == "high_risk":
        return df[df["risk_quartile"] == "Q4 (High)"]
    if segment == "low_risk":
        return df[df["risk_quartile"] == "Q1 (Low)"]
    if segment == "high_buyin":
        return df[df["buyin_quartile"] == "Q4 (High)"]
    return df


def pct(value: float, digits: int = 1) -> str:
    """Format a proportion as a percent string."""
    return f"{value * 100:.{digits}f}%"


def safe_float(value, default: float) -> float:
    """Best-effort float conversion with fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default: int) -> int:
    """Best-effort int conversion with fallback."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def simulate_binary_experiment(
    values: pd.Series,
    n_per_arm: int,
    uplift_pct: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Bootstrap two binary arms and inject a target uplift into treatment."""
    rng = np.random.default_rng(seed)
    baseline = float(values.mean())
    target = float(np.clip(baseline * (1 + uplift_pct / 100), 0.001, 0.999))

    replace = n_per_arm > len(values)
    control = rng.choice(values.to_numpy(dtype=int), size=n_per_arm, replace=replace)
    treatment = rng.choice(values.to_numpy(dtype=int), size=n_per_arm, replace=replace)
    treatment = _shift_binary_rate(treatment, target, rng)

    return control, treatment, baseline, target


def _shift_binary_rate(values: np.ndarray, target: float, rng: np.random.Generator) -> np.ndarray:
    """Adjust a sampled binary arm toward a target rate by flipping outcomes."""
    values = values.astype(int, copy=True)
    current = float(values.mean())
    if current < target:
        zero_idx = np.where(values == 0)[0]
        flip_prob = min(1.0, (target - current) / max(1 - current, 1e-9))
        flips = zero_idx[rng.random(len(zero_idx)) < flip_prob]
        values[flips] = 1
    elif current > target:
        one_idx = np.where(values == 1)[0]
        flip_prob = min(1.0, (current - target) / max(current, 1e-9))
        flips = one_idx[rng.random(len(one_idx)) < flip_prob]
        values[flips] = 0
    return values


def _arm_ci(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    rate = successes / total
    z = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(rate * (1 - rate) / total)
    return max(rate - z * se, 0.0), min(rate + z * se, 1.0)


def rate_bar_figure(control: np.ndarray, treatment: np.ndarray, alpha: float) -> go.Figure:
    """Observed control/treatment rates with Wald confidence intervals."""
    control_rate = float(control.mean())
    treatment_rate = float(treatment.mean())
    ci0 = _arm_ci(int(control.sum()), len(control), alpha=alpha)
    ci1 = _arm_ci(int(treatment.sum()), len(treatment), alpha=alpha)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Control", "Treatment"],
        y=[control_rate * 100, treatment_rate * 100],
        marker_color=[PALETTE[1], PALETTE[0]],
        error_y=dict(
            type="data",
            array=[
                max((ci0[1] - control_rate) * 100, 0),
                max((ci1[1] - treatment_rate) * 100, 0),
            ],
        ),
        text=[f"{control_rate * 100:.1f}%", f"{treatment_rate * 100:.1f}%"],
        textposition="outside",
    ))
    fig.update_layout(
        title="Наблюдаемый KPI по группам",
        yaxis_title="Доля (%)",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def posterior_figure(diff_draws: np.ndarray) -> go.Figure:
    """Posterior histogram of treatment uplift."""
    fig = px.histogram(
        x=diff_draws * 100,
        nbins=50,
        color_discrete_sequence=[PALETTE[2]],
        labels={"x": "Treatment - Control (п.п.)"},
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#495057")
    fig.update_layout(
        title="Апостериорное распределение uplift",
        template="plotly_white",
        bargap=0.05,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def sequential_figure(seq_df: pd.DataFrame) -> go.Figure:
    """Observed sequential z-statistics against stopping boundaries."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=seq_df["look"],
        y=seq_df["z_stat"],
        mode="lines+markers",
        name="Наблюдаемая z-статистика",
        line=dict(color=PALETTE[0], width=3),
    ))
    fig.add_trace(go.Scatter(
        x=seq_df["look"],
        y=seq_df["z_boundary"],
        mode="lines+markers",
        name="Верхняя граница OBF",
        line=dict(color=PALETTE[1], dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=seq_df["look"],
        y=-seq_df["z_boundary"],
        mode="lines+markers",
        name="Нижняя граница OBF",
        line=dict(color=PALETTE[1], dash="dot"),
    ))
    fig.update_layout(
        title="Последовательный мониторинг",
        xaxis_title="Промежуточный просмотр",
        yaxis_title="z-статистика",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def sample_size_curve(
    family: str,
    baseline: float,
    mde: float,
    sigma: float,
    hazard_ratio: float,
    event_rate: float,
    alpha: float,
    power: float,
) -> go.Figure:
    """Sensitivity curve for the sample-size calculator."""
    if family == "binary":
        grid = np.linspace(max(0.01, mde / 2), max(0.12, mde * 2), 12)
        sizes = [sample_size_proportions(baseline, x, alpha=alpha, power=power) for x in grid]
        x = grid * 100
        x_title = "Абсолютный MDE (п.п.)"
    elif family == "continuous":
        grid = np.linspace(max(0.1, mde / 2), max(2.0, mde * 2), 12)
        sizes = [sample_size_continuous(sigma, x, alpha=alpha, power=power) for x in grid]
        x = grid
        x_title = "Абсолютный MDE"
    else:
        if hazard_ratio < 1:
            low = max(0.65, hazard_ratio - 0.15)
            high = min(0.98, hazard_ratio + 0.08)
        else:
            low = max(1.02, hazard_ratio - 0.08)
            high = min(1.35, hazard_ratio + 0.15)
        grid = np.linspace(low, high, 12)
        sizes = [sample_size_survival(x, event_rate, alpha=alpha, power=power) for x in grid]
        x = grid
        x_title = "Hazard ratio"

    fig = px.line(
        x=x,
        y=sizes,
        markers=True,
        color_discrete_sequence=[PALETTE[3]],
        labels={"x": x_title, "y": "Требуемый размер выборки"},
    )
    fig.update_layout(
        title="Чувствительность к размеру эффекта",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def familywise_metric_table(
    df: pd.DataFrame,
    n_per_arm: int,
    uplift_pct: float,
    alpha: float,
    seed: int,
) -> pd.DataFrame:
    """Evaluate a family of binary KPIs with Holm and BH corrections."""
    rng = np.random.default_rng(seed)
    replace = n_per_arm > len(df)
    control_idx = rng.choice(df.index.to_numpy(), size=n_per_arm, replace=replace)
    treatment_idx = rng.choice(df.index.to_numpy(), size=n_per_arm, replace=replace)
    control_df = df.loc[control_idx]
    treatment_df = df.loc[treatment_idx]

    rows = []
    raw_p = []
    for idx, (metric, meta) in enumerate(SIM_METRICS.items()):
        control = control_df[metric].to_numpy(dtype=int)
        treatment = treatment_df[metric].to_numpy(dtype=int)
        target = float(np.clip(control.mean() * (1 + uplift_pct / 100), 0.001, 0.999))
        shifted = _shift_binary_rate(
            treatment,
            target,
            np.random.default_rng(seed + idx + 1),
        )
        result = proportion_z_test(
            int(control.sum()),
            len(control),
            int(shifted.sum()),
            len(shifted),
            alpha=alpha,
        )
        raw_p.append(result["p_value"])
        rows.append(
            {
                "metric": metric,
                "label": meta["label"],
                "control_rate": result["control_rate"],
                "treatment_rate": result["treatment_rate"],
                "absolute_diff": result["absolute_diff"],
                "p_raw": result["p_value"],
            }
        )

    holm = adjust_pvalues(raw_p, method="holm")
    bh = adjust_pvalues(raw_p, method="bh")
    for row, p_holm, p_bh in zip(rows, holm, bh):
        row["p_holm"] = p_holm
        row["p_bh"] = p_bh
        row["holm_significant"] = p_holm < alpha
        row["bh_significant"] = p_bh < alpha

    return pd.DataFrame(rows)


def bootstrap_summary(control: np.ndarray, treatment: np.ndarray, seed: int) -> dict:
    """Convenience wrapper for a bootstrap uplift summary."""
    return bootstrap_uplift_ci(control, treatment, confidence=0.95, n_boot=2000, seed=seed)
