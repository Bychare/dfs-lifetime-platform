# DFS Lifetime Platform — журнал преобразований

Дата: 2026-04-06

## 1) Исправление ошибки импорта при запуске приложения

### Симптом

При запуске `python3 app/app.py` возникала ошибка:

- `ModuleNotFoundError: No module named 'app.components'; 'app' is not a package`

### Причина

Страницы Dash (`app/pages/*.py`) использовали абсолютные импорты вида
`from app.components...`.
При запуске через `app/app.py` модуль `app.py` конфликтует с именем пакета `app`,
из-за чего `app.components` не резолвится как пакет.

### Что изменено

Во всех файлах страниц импорты заменены с `from app.components...` на `from components...`:

- `app/pages/survival.py`
- `app/pages/ab_testing.py`
- `app/pages/segmentation.py`
- `app/pages/churn_model.py`
- `app/pages/overview.py`

Также внутри `update_map` в `app/pages/overview.py`:

- `from app.components.data_loader import load_codes`
  заменен на
- `from components.data_loader import load_codes`

### Результат

Приложение успешно стартует без ошибки импорта (`Dash is running on http://127.0.0.1:8050/`).

---

## 2) Исправление ошибки Plotly в `update_map`

### Симптом

Возникала ошибка:

- `ValueError: Cannot accept list of column references or list of columns for both x and y`

Источник: `app/pages/overview.py`, функция `update_map`, вызов `px.bar(...)`.

### Причина

В `px.bar` передавались массивы `x=top20.values` и `y=top20.index` без явного `data_frame`,
и Plotly Express трактовал это как некорректные ссылки на колонки.

### Что изменено

В `app/pages/overview.py`:

1. Данные для графика переведены в явный DataFrame:

- `top20 = counts.head(20).rename_axis("state").reset_index(name="players")`

2. Вызов `px.bar` переведен на колонночный стиль:

- `px.bar(top20, x="players", y="state", orientation="h", ...)`

3. Удален неиспользуемый код в `update_map`, связанный с `load_codes()/states`,
который не участвовал в построении текущего графика.

### Результат

Ошибка `ValueError` устранена, построение bar chart проходит корректно.

---

## Проверки

- Линтер по измененным файлам: без ошибок.
- Smoke-test формирования графика `px.bar` для `update_map`: успешный.

