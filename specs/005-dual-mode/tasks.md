# Tasks: Dual-mode posts (public/private)

- Specification: specs/005-dual-mode/spec.md
- Plan: specs/005-dual-mode/plan.md

## Delivery tasks

- [x] T001 Добавить `--mode` в парсер `post` (choices public/private,
      default public, help с путями). DoD: `post --help` показывает флаг.
- [x] T002 Перевести `generate_post` на `subdir` из `args.mode`
      (posts / posts-private); транзитный `filename_prefix` не менять.
      DoD: py_compile; логика destination покрывает оба режима.
- [ ] T003 Серверный прогон поста с `--mode private` (SYSADMIN-HANDOFF).
      DoD: папка в `models/<id>/posts-private/`, состав файлов полный.

## Acceptance traceability

- AC-1 (--mode в help) ← T001.
- AC-2 (py_compile) ← T001, T002.
- AC-3 (destination по режиму) ← T002, T003.
- AC-4 (транзит неизменен) ← T002.

## Definition of Done

- [x] T001–T002 выполнены и верифицированы (py_compile, --help).
- [ ] Коммит не содержит секретов, датасетов, весов, runtime-вывода.
- [ ] Серверный прогон (T003) отражён в artifacts при следующем значимом
      изменении.
