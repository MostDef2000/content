# Tasks: Explicit dataset and manifest-aware captions

- Specification: specs/004-explicit-dataset/spec.md
- Plan: specs/004-explicit-dataset/plan.md

## Delivery tasks

- [x] T001 Добавить в `lora/caption.py` `import prompts` (sys.path как в
      queue_workflow.py), age-floor и trigger-word фикс базовой строки.
      DoD: py_compile; формат
      `<model_id>, fictional adult woman age <age>, …`.
- [x] T002 Реализовать manifest-слой: `manifest_path`, `load_manifest`
      (чтение + reconcile), `save_manifest` (атомарно), `caption_for`.
      DoD: py_compile; `version > 1` → ошибка; новые/удалённые файлы
      реконсилятся.
- [x] T003 Переписать `main()`: caption_for по тегу, validate_positive до
      записи, одна запись манифеста после цикла, сводка
      `Created N captions (M tasteful, K explicit) for <id>.`
      DoD: py_compile; fixture-прогон с explicit-записью.
- [ ] T004 Прогнать caption.py на реальном датасете сервера и заполнить
      теги манифеста (SYSADMIN-HANDOFF). DoD: сводка без ошибок guardrail.

## Acceptance traceability

- AC-1 (сигнатура и py_compile) ← T001.
- AC-2 (manifest-слой) ← T002.
- AC-3 (fixture explicit) ← T003.
- AC-4 (guardrail-отказ) ← T003.

## Definition of Done

- [x] T001–T003 выполнены и верифицированы (py_compile, --help, fixture).
- [ ] Коммит не содержит секретов, датасетов, весов, runtime-вывода.
- [ ] Серверный прогон (T004) отражён в artifacts при следующем значимом
      изменении.
