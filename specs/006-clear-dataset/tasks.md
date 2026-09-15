# Tasks: Clear dataset (bulk soft-delete reset)

- Specification: specs/006-clear-dataset/spec.md
- Plan: specs/006-clear-dataset/plan.md

## Delivery tasks

- [x] T001 Endpoint `DELETE /api/models/{id}/dataset?confirm=<id>`:
      массовый soft-delete в `DATASET_TRASH` (каталог `images/` целиком
      одним `os.replace`, копия manifest.json в trash, сброс манифеста в
      `{"version": 1, "images": []}`), confirm=id, сериализация через
      `_manifest_lock`, 409 при активной джобе (`_model_has_active_job`),
      no-op для пустого датасета. Реализовано в `wipe_model_dataset`
      (`management/main.py`); DoD: py_compile; 6 unit-тестов в
      `tests/test_clear_dataset.py`.
- [x] T002 UI: кнопка «Очистить датасет» + подтверждение вводом id +
      обновление галереи/статистики. Реализовано: диалог `wipeDialog` в
      `management/static/index.html` (превью, статистика, ввод id);
      серверная проверка UI — в рамках T003.
- [ ] T003 Серверный прогон: рестарт manager, очистка датасета + повторный
      expand/caption (SYSADMIN-HANDOFF). DoD: trash заполнен, `images/`
      пуст, manifest создан заново, `/api/dataset` пуст.

## Acceptance traceability

- AC-1 (только dataset/, всё в trash) ← T001, T003.
- AC-2 (confirm обязателен) ← T001, T002.
- AC-3 (пустой список, manifest создается заново) ← T001, T003.
- AC-4 (инвариант пар при частичном сбое) ← T001.

## Definition of Done

- [x] T001–T002 выполнены и верифицированы (py_compile; tests/
      test_clear_dataset.py — 6 тестов).
- [ ] T003 (серверный smoke) выполнен и верифицирован.
- [ ] Коммит не содержит секретов, датасетов, весов, runtime-вывода.
- [ ] Серверный прогон (T003) отражён в artifacts при следующем значимом
      изменении.
