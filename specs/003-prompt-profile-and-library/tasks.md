# Tasks: Prompt profile and scene library

- Specification: specs/003-prompt-profile-and-library/spec.md
- Plan: specs/003-prompt-profile-and-library/plan.md

## Delivery tasks

- [x] T001 Создать `prompts.py` (MODES, AGE_FLOOR, GuardrailError,
      guardrail_phrase, validate_positive, build_positive, build_negative,
      default_profile; scene_presets удалён 16.09.2026). DoD: py_compile; чистый модуль без I/O.
- [x] T002 Создать `tests/test_prompts.py` (import-safe, pytest-совместимые).
      DoD: py_compile; `__main__`-раннер зелёный.
- [x] T003 Создать `models/valery23/prompt_profile.json` из
      `default_profile(character)`. DoD: json.tool; совпадает с функцией.
- [x] T004 Создать `models/library.json` (первичный источник — пресеты;
      с 16.09.2026 единственный источник сцен — библиотека через UI,
      `scene_presets()` удалён).
      DoD: json.tool; ≥8 expand + ≥2 candidates + ≥2 post.
- [x] T005 Подключить `prompts.py` к `queue_workflow.py`/manager.
      DoD: отдельная задача (queue_workflow — другой воркер).

## Acceptance traceability

- AC-1 (guardrail во всех mode, floor) ← T001, T002.
- AC-2 (валидатор + word-boundary) ← T001, T002.
- AC-3 (negative-канон не вычитается) ← T001, T002.
- AC-4 (профиль/библиотека консистентны) ← T003, T004.

## Definition of Done

- [x] T001–T004 выполнены, проверки зелёные (py_compile / json.tool /
      import-smoke).
- [ ] Коммит не содержит секретов и сгенерированного вывода.
- [x] Интеграция в пайплайн — отдельная задача (T005).
