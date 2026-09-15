# Tasks: Multi-model registry

- Specification: specs/002-multi-model-registry/spec.md
- Plan: specs/002-multi-model-registry/plan.md

## Delivery tasks

- [x] T001 Создать `models/registry.json` с моделью `valery23` (active).
      DoD: `python -m json.tool models/registry.json` проходит.
- [x] T002 Создать `models/valery23/character.json` (корневой character.json +
      top-level `body`). DoD: json.tool; `body` на одном уровне с `appearance`.
- [x] T003 Расширить `.gitignore` per-model путями (reference, dataset,
      lora/output, posts, posts-private). DoD: генерат-папки игнорируются,
      tracked-файлы — нет.
- [x] T004 Создать `scripts/migrate_to_registry.py` (идемпотентная миграция:
      registry, per-model папки, перенос reference/dataset/posts, character,
      seed профиля; бэкап + rollback-заметка, `--dry-run`).
      DoD: `py_compile` проходит; повторный запуск — no-op.
- [x] T005 Обновить README/AGENTS: per-model «Папки», LoRA-копия
      `runtime/models/loras/<id>.safetensors`, trash, упоминание миграции.
      DoD: разделы консистентны с layout.
- [ ] T006 Запустить миграцию на сервере (SYSADMIN-HANDOFF).
      DoD: `models/valery23/` заполнен, повторный запуск — no-op.
- [x] T007 Подключить manager/UI к реестру (переключение активной, удаление в
      trash). DoD: отдельная задача/спека.

## Acceptance traceability

- AC-1 (registry/character валидны, valery23 active) ← T001, T002.
- AC-2 (.gitignore per-model) ← T003.
- AC-3 (миграция идемпотентна, с бэкапом) ← T004, T006.
- AC-4 (документация layout/trash) ← T005.

## Definition of Done

- [x] T001–T005 выполнены и верифицированы (py_compile / json.tool).
- [ ] Коммит не содержит секретов, датасетов, весов, runtime-вывода.
- [ ] Серверный прогон миграции (T006) отражён в artifacts при следующем
      значимом изменении.
