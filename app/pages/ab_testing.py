"""Module 3: A/B Test design, simulation, and monitoring."""

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, callback, dcc, html

from components.ab_testing_helpers import (
    AB_GLOSSARY_MD,
    AB_NOTES_MD,
    SEGMENTS,
    SIM_METRICS,
    ab_frame,
    bootstrap_summary,
    familywise_metric_table,
    pct,
    posterior_figure,
    rate_bar_figure,
    safe_float,
    safe_int,
    sample_size_curve,
    segment_slice,
    sequential_figure,
    simulate_binary_experiment,
)
from components.layout_utils import (
    glossary_accordion,
    kpi_card,
    methodological_notes,
    section_header,
)
from components.stats import (
    beta_binomial_ab_test,
    proportion_z_test,
    sample_size_continuous,
    sample_size_proportions,
    sample_size_survival,
    sequential_proportion_monitor,
)

dash.register_page(__name__, path="/ab-testing", name="A/B-тестирование", order=2)


layout = html.Div([
    html.H3("Движок экспериментов", className="mb-1"),
    html.P(
        "Планирование, симуляция и мониторинг продуктовых экспериментов в стиле DFS "
        "с использованием расчёта размера выборки, частотного вывода, байесовских "
        "апостериорных распределений и sequential-границ остановки.",
        className="text-muted",
    ),

    glossary_accordion(
        "Глоссарий — термины экспериментального дизайна на этой странице",
        AB_GLOSSARY_MD,
    ),

    html.Div(id="ab-kpi-row", className="mb-4"),

    section_header(
        "Калькулятор размера выборки",
        "Аппроксимации в стиле Lehr для бинарных и непрерывных KPI плюс "
        "планирование по Шёнфельду для survival-endpoints.",
    ),
    dbc.Row([
        dbc.Col([
            dbc.Label("Тип дизайна"),
            dbc.Select(
                id="ab-family",
                options=[
                    {"label": "Бинарный KPI: конверсия / retention", "value": "binary"},
                    {"label": "Непрерывная метрика дохода", "value": "continuous"},
                    {"label": "Survival / time-to-event", "value": "survival"},
                ],
                value="binary",
            ),
        ], md=3),
        dbc.Col([
            dbc.Label("Alpha"),
            dbc.Input(id="ab-alpha", type="number", min=0.001, max=0.2, step=0.005, value=0.05),
        ], md=2),
        dbc.Col([
            dbc.Label("Power"),
            dbc.Input(id="ab-power", type="number", min=0.5, max=0.99, step=0.01, value=0.8),
        ], md=2),
        dbc.Col([
            dbc.Label("Дневной поток подходящих пользователей"),
            dbc.Input(id="ab-traffic", type="number", min=10, step=10, value=400),
        ], md=2),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            dbc.Label("Базовый уровень"),
            dbc.Input(id="ab-baseline", type="number", min=0.01, max=0.99, step=0.01, value=0.35),
            html.Small("Для бинарного дизайна: конверсия или retention в control.", className="text-muted"),
        ], md=3),
        dbc.Col([
            dbc.Label("MDE"),
            dbc.Input(id="ab-mde", type="number", min=0.001, step=0.005, value=0.03),
            html.Small("Для бинарного дизайна: абсолютный uplift в процентных пунктах.", className="text-muted"),
        ], md=3),
        dbc.Col([
            dbc.Label("Sigma"),
            dbc.Input(id="ab-sigma", type="number", min=0.1, step=0.1, value=1.0),
            html.Small("Для непрерывного дизайна: стандартное отклонение KPI.", className="text-muted"),
        ], md=3),
        dbc.Col([
            dbc.Label("Event rate / hazard ratio"),
            dbc.InputGroup([
                dbc.Input(id="ab-event-rate", type="number", min=0.01, max=1, step=0.01, value=0.23),
                dbc.Input(id="ab-hazard-ratio", type="number", min=0.5, max=1.5, step=0.01, value=0.85),
            ]),
            html.Small("Для survival-дизайна: наблюдаемая частота событий и целевой HR.", className="text-muted"),
        ], md=3),
    ], className="mb-3"),
    html.Div(id="ab-sample-size-cards", className="mb-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="ab-sample-size-fig"), md=8),
        dbc.Col(dbc.Alert(id="ab-sample-size-note", color="light", className="py-2 small"), md=4),
    ], className="mb-4"),

    section_header(
        "A/B-симулятор",
        "Синтетические рандомизированные эксперименты на основе наблюдаемой DFS-когорты.",
    ),
    dbc.Row([
        dbc.Col([
            dbc.Label("KPI"),
            dbc.Select(
                id="ab-metric",
                options=[{"label": meta["label"], "value": key} for key, meta in SIM_METRICS.items()],
                value="retained_30d",
            ),
        ], md=3),
        dbc.Col([
            dbc.Label("Сегмент"),
            dbc.Select(
                id="ab-segment",
                options=[{"label": label, "value": key} for key, label in SEGMENTS.items()],
                value="all",
            ),
        ], md=3),
        dbc.Col([
            dbc.Label("Размер на группу"),
            dbc.Input(id="ab-n-per-arm", type="number", min=100, step=100, value=1500),
        ], md=2),
        dbc.Col([
            dbc.Label("Относительный uplift (%)"),
            dbc.Input(id="ab-uplift", type="number", min=-50, max=100, step=1, value=8),
        ], md=2),
        dbc.Col([
            dbc.Label("Seed"),
            dbc.Input(id="ab-seed", type="number", min=1, step=1, value=42),
        ], md=2),
    ], className="mb-3"),
    html.Div(id="ab-sim-cards", className="mb-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="ab-rate-fig"), md=6),
        dbc.Col(dcc.Graph(id="ab-posterior-fig"), md=6),
    ], className="mb-3"),
    dbc.Alert(id="ab-summary", color="info", className="mb-4"),

    section_header(
        "Устойчивость вывода",
        "Робастный bootstrap-интервал для uplift и семейство KPI с поправками Holm/BH.",
    ),
    dbc.Row([
        dbc.Col(dbc.Alert(id="ab-bootstrap-summary", color="secondary", className="mb-3"), md=4),
        dbc.Col(html.Div(id="ab-multi-table"), md=8),
    ], className="mb-4"),

    section_header(
        "Последовательное тестирование",
        "Промежуточные просмотры с двусторонними границами O'Brien-Fleming.",
    ),
    dbc.Row([
        dbc.Col([
            dbc.Label("Число промежуточных просмотров"),
            dbc.Input(id="ab-looks", type="number", min=2, max=10, step=1, value=5),
        ], md=2),
        dbc.Col([
            dbc.Label("Alpha для мониторинга"),
            dbc.Input(id="ab-monitor-alpha", type="number", min=0.001, max=0.2, step=0.005, value=0.05),
        ], md=2),
        dbc.Col([
            dbc.Label("Интуиция по границам"),
            dbc.Alert(
                "На ранних просмотрах требуется значительно более высокая z-статистика; по мере накопления информации порог снижается.",
                color="light",
                className="py-2 mb-0",
            ),
        ], md=8),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="ab-seq-fig"), md=7),
        dbc.Col(html.Div(id="ab-seq-table"), md=5),
    ]),

    methodological_notes(AB_NOTES_MD),
])


@callback(
    Output("ab-kpi-row", "children"),
    Input("ab-metric", "value"),
    Input("ab-segment", "value"),
)
def update_ab_kpis(metric: str, segment: str):
    df = segment_slice(ab_frame(), segment)
    rate = float(df[metric].mean())
    return dbc.Row([
        dbc.Col(kpi_card("Подходящие игроки", f"{len(df):,}"), md=True),
        dbc.Col(kpi_card("Наблюдаемый baseline", pct(rate), "info"), md=True),
        dbc.Col(kpi_card("Медианный fee", f"${df['TotFees'].median():,.0f}"), md=True),
        dbc.Col(kpi_card("Churn Rate", pct(df['is_churned'].mean()), "danger"), md=True),
        dbc.Col(kpi_card("Средний Risk Score", f"{df['RiskScore'].mean():.1f}", "warning"), md=True),
    ])


@callback(
    Output("ab-sample-size-cards", "children"),
    Output("ab-sample-size-fig", "figure"),
    Output("ab-sample-size-note", "children"),
    Input("ab-family", "value"),
    Input("ab-alpha", "value"),
    Input("ab-power", "value"),
    Input("ab-traffic", "value"),
    Input("ab-baseline", "value"),
    Input("ab-mde", "value"),
    Input("ab-sigma", "value"),
    Input("ab-event-rate", "value"),
    Input("ab-hazard-ratio", "value"),
)
def update_sample_size(
    family: str,
    alpha,
    power,
    traffic,
    baseline,
    mde,
    sigma,
    event_rate,
    hazard_ratio,
):
    alpha = safe_float(alpha, 0.05)
    power = safe_float(power, 0.80)
    traffic = max(safe_int(traffic, 400), 1)
    baseline = float(np.clip(safe_float(baseline, 0.35), 0.01, 0.99))
    mde = max(safe_float(mde, 0.03), 0.001)
    sigma = max(safe_float(sigma, 1.0), 0.001)
    event_rate = float(np.clip(safe_float(event_rate, 0.23), 0.01, 1.0))
    hazard_ratio = max(safe_float(hazard_ratio, 0.85), 0.01)

    if family == "binary":
        per_arm = sample_size_proportions(baseline, mde, alpha=alpha, power=power)
        total = per_arm * 2
        primary_value = f"{per_arm:,} на группу"
        assumptions = (
            f"Baseline = {pct(baseline)}, MDE = {mde * 100:.1f} п.п., "
            f"alpha = {alpha:.3f}, power = {power:.0%}."
        )
    elif family == "continuous":
        per_arm = sample_size_continuous(sigma, mde, alpha=alpha, power=power)
        total = per_arm * 2
        primary_value = f"{per_arm:,} на группу"
        assumptions = (
            f"Sigma = {sigma:.2f}, MDE = {mde:.2f}, "
            f"alpha = {alpha:.3f}, power = {power:.0%}."
        )
    else:
        total = sample_size_survival(hazard_ratio, event_rate, alpha=alpha, power=power)
        per_arm = int(np.ceil(total / 2))
        primary_value = f"{total:,} всего"
        assumptions = (
            f"Event rate = {pct(event_rate)}, HR = {hazard_ratio:.2f}, "
            f"alpha = {alpha:.3f}, power = {power:.0%}."
        )

    days = total / traffic
    cards = dbc.Row([
        dbc.Col(kpi_card("Требуемая выборка", primary_value), md=True),
        dbc.Col(kpi_card("Общий объём", f"{total:,} игроков", "info"), md=True),
        dbc.Col(kpi_card("Длительность при этом трафике", f"{days:.1f} дней", "warning"), md=True),
        dbc.Col(kpi_card("Уровень значимости", f"{alpha:.1%}", "danger"), md=True),
    ])
    fig = sample_size_curve(
        family, baseline, mde, sigma, hazard_ratio, event_rate, alpha, power
    )
    note = [
        html.Div("Пояснения", className="fw-semibold mb-2"),
        html.P(assumptions, className="mb-2 small"),
        html.Ul([
            html.Li("Бинарный дизайн: retention / conversion."),
            html.Li("Непрерывный дизайн: spend / revenue."),
            html.Li("Survival-дизайн: time to churn / milestone."),
            html.Li("Чем меньше эффект, тем больше требуемая выборка."),
        ], className="small mb-0 ps-3"),
    ]
    return cards, fig, note


@callback(
    Output("ab-sim-cards", "children"),
    Output("ab-rate-fig", "figure"),
    Output("ab-posterior-fig", "figure"),
    Output("ab-summary", "children"),
    Output("ab-bootstrap-summary", "children"),
    Output("ab-multi-table", "children"),
    Output("ab-seq-fig", "figure"),
    Output("ab-seq-table", "children"),
    Input("ab-metric", "value"),
    Input("ab-segment", "value"),
    Input("ab-n-per-arm", "value"),
    Input("ab-uplift", "value"),
    Input("ab-seed", "value"),
    Input("ab-looks", "value"),
    Input("ab-monitor-alpha", "value"),
)
def update_simulation(
    metric: str,
    segment: str,
    n_per_arm,
    uplift,
    seed,
    looks,
    monitor_alpha,
):
    n_per_arm = max(safe_int(n_per_arm, 1500), 50)
    uplift = safe_float(uplift, 8.0)
    seed = safe_int(seed, 42)
    looks = min(max(safe_int(looks, 5), 2), 10)
    monitor_alpha = safe_float(monitor_alpha, 0.05)

    df = segment_slice(ab_frame(), segment)
    values = df[metric].dropna().astype(int)
    control, treatment, baseline, target = simulate_binary_experiment(
        values, n_per_arm=n_per_arm, uplift_pct=uplift, seed=seed
    )

    frequentist = proportion_z_test(
        int(control.sum()),
        len(control),
        int(treatment.sum()),
        len(treatment),
        alpha=monitor_alpha,
    )
    bayes = beta_binomial_ab_test(
        int(control.sum()),
        len(control),
        int(treatment.sum()),
        len(treatment),
        seed=seed,
    )
    boot = bootstrap_summary(control, treatment, seed=seed)
    family_df = familywise_metric_table(
        df, n_per_arm=n_per_arm, uplift_pct=uplift, alpha=monitor_alpha, seed=seed
    )
    seq_df, stop_look = sequential_proportion_monitor(
        control, treatment, n_looks=looks, alpha=monitor_alpha
    )

    cards = dbc.Row([
        dbc.Col(kpi_card("Целевой baseline", pct(baseline), "info"), md=True),
        dbc.Col(kpi_card("Внесённый uplift", f"{uplift:.1f}%", "warning"), md=True),
        dbc.Col(kpi_card("Наблюдаемый эффект", f"{frequentist['absolute_diff'] * 100:.2f} п.п."), md=True),
        dbc.Col(kpi_card("Частотный p-value", f"{frequentist['p_value']:.4f}", "danger"), md=True),
        dbc.Col(kpi_card("P(Treat > Ctrl)", pct(bayes["prob_treatment_beats_control"]), "success"), md=True),
    ])

    summary = (
        f"{SIM_METRICS[metric]['label']} в сегменте {SEGMENTS[segment]}: в control наблюдается "
        f"{pct(frequentist['control_rate'])}, в treatment — {pct(frequentist['treatment_rate'])}. "
        f"Целевой уровень после внесения эффекта составлял {pct(target)}. "
        f"Частотный 95% CI для uplift: от {frequentist['ci_low'] * 100:.2f} до "
        f"{frequentist['ci_high'] * 100:.2f} п.п. "
        f"Bootstrap 95% CI: от {boot['ci_low'] * 100:.2f} до {boot['ci_high'] * 100:.2f} п.п. "
        f"Байесовский posterior даёт вероятность превосходства treatment над control "
        f"на уровне {pct(bayes['prob_treatment_beats_control'])}. "
        f"{'Последовательный мониторинг пересекает границу на просмотре №' + str(stop_look) + '.' if stop_look else 'В этом прогоне границы OBF не были пересечены.'}"
    )

    bootstrap_note = [
        html.Div("Bootstrap 95% CI", className="fw-semibold mb-2"),
        html.P(
            f"Робастная оценка uplift: от {boot['ci_low'] * 100:.2f} до "
            f"{boot['ci_high'] * 100:.2f} п.п.",
            className="mb-2",
        ),
        html.P(
            f"Точечная оценка = {boot['point_estimate'] * 100:.2f} п.п., "
            f"bootstrap SE = {boot['std_error'] * 100:.2f} п.п.",
            className="mb-0 small",
        ),
    ]

    def correction_badge(row):
        if row["holm_significant"]:
            return dbc.Badge("✓ Holm", color="success")
        if row["bh_significant"]:
            return dbc.Badge("✓ BH", color="info")
        return dbc.Badge("n.s.", color="secondary")

    multi_table = dbc.Table(
        [
            html.Thead(html.Tr([
                html.Th("KPI"),
                html.Th("Ctrl"),
                html.Th("Treat"),
                html.Th("Uplift"),
                html.Th("p (raw)"),
                html.Th("p (Holm)"),
                html.Th("p (BH)"),
                html.Th("Решение"),
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(row["label"]),
                    html.Td(pct(row["control_rate"])),
                    html.Td(pct(row["treatment_rate"])),
                    html.Td(f"{row['absolute_diff'] * 100:.2f} п.п."),
                    html.Td(f"{row['p_raw']:.4f}"),
                    html.Td(f"{row['p_holm']:.4f}"),
                    html.Td(f"{row['p_bh']:.4f}"),
                    html.Td(correction_badge(row)),
                ])
                for _, row in family_df.iterrows()
            ]),
        ],
        bordered=True,
        striped=True,
        hover=True,
        size="sm",
        className="small",
    )

    seq_table = dbc.Table(
        [
            html.Thead(html.Tr([
                html.Th("Просмотр"),
                html.Th("n / группа"),
                html.Th("Ctrl"),
                html.Th("Treat"),
                html.Th("z"),
                html.Th("Граница"),
                html.Th("Остановить?"),
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(int(row["look"])),
                    html.Td(f"{int(row['n_per_arm']):,}"),
                    html.Td(pct(row["control_rate"])),
                    html.Td(pct(row["treatment_rate"])),
                    html.Td(f"{row['z_stat']:.2f}"),
                    html.Td(f"{row['z_boundary']:.2f}"),
                    html.Td(
                        dbc.Badge(
                            "Пересечена" if row["crossed"] else "Продолжать",
                            color="success" if row["crossed"] else "secondary",
                        )
                    ),
                ])
                for _, row in seq_df.iterrows()
            ]),
        ],
        bordered=True,
        striped=True,
        hover=True,
        size="sm",
        className="small",
    )

    return (
        cards,
        rate_bar_figure(control, treatment, alpha=monitor_alpha),
        posterior_figure(bayes["diff_draws"]),
        summary,
        bootstrap_note,
        multi_table,
        sequential_figure(seq_df),
        seq_table,
    )
