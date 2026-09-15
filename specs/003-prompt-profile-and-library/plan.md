# Implementation Plan: Prompt profile and scene library

- Specification: specs/003-prompt-profile-and-library/spec.md
- Status: Active

## Architecture

- `prompts.py` — чистый модуль сборки промтов: guardrail-фраза + валидатор +
  mode-каркасы (константы) + канонический negative + профили/библиотека сцен.
- `models/<id>/prompt_profile.json` — tracked профиль модели
  (`default_profile`).
- `models/library.json` — общая библиотека сцен (`scene_presets`).
- `tests/test_prompts.py` — import-safe тесты (pytest-совместимые, работают
  без pytest через `__main__`-раннер).

## Decisions

### D-001 — Guardrail в коде, а не только в текстах workflow
Фраза «{max(23,age)}-year-old fictional» инъектируется всегда + финальная
самопроверка результата; канонический негатив идёт первым и не вычитается.
Отклонено: полагаться только на ручные промты — человеческий фактор.

### D-002 — Word-boundary валидатор
`\b`-границы исключают ложные срабатывания (kidney, canteen); список термов —
константа модуля, case-insensitive.

### D-003 — Каркасы mode как константы модуля
candidates/expand/post каркасы зафиксированы рядом с семантикой workflow JSON;
смена текстов — правка констант + тестов.

### D-004 — Профиль и библиотека — tracked JSON
Конфигурация в Git, генерат — нет; библиотека общая, профиль — per-model.
`models/library.json` генерируется из `scene_presets()` — единый источник.

## Affected contours

- Repository: `prompts.py`, `tests/`, `models/valery23/prompt_profile.json`,
  `models/library.json`.
- GPU/VPS: без изменений; интеграция в `queue_workflow`/manager — отдельные
  задачи.

## Risks

- Риск: дублирование текстов `prompts.py` и `queue_workflow.py` до интеграции.
  Митигируется: `VARIATION_PROMPTS` остаются источником expand-текстов,
  `library.json` сгенерирован из `scene_presets()`; при интеграции
  `queue_workflow.py` перейдёт на `prompts.py`.
- Риск: ложные блокировки легитимных слов. Митигируется word-boundary + тесты.

## Test design

- unit: `tests/test_prompts.py` — guardrail во всех mode, floor возраста,
  валидатор (позитивы и false-positive-контроль), канон negative, порядок
  сборки, shape профиля, библиотека сцен (P0).
- compile: `python -m py_compile prompts.py tests/test_prompts.py` (P0).
- json: `python -m json.tool models/valery23/prompt_profile.json`,
  `python -m json.tool models/library.json` (P0).
