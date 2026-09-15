# Implementation Plan: Multi-model registry

- Specification: specs/002-multi-model-registry/spec.md
- Status: Active

## Architecture

- Реестр: `models/registry.json` — плоский JSON (`version` + `models[]`);
  источник истины о моделях; активная модель = ровно одна запись с
  `active=true`.
- Данные модели: `models/<id>/…` (reference/candidates, dataset/images,
  lora/output, posts, posts-private) + tracked `character.json` /
  `prompt_profile.json`.
- Runtime: тяжёлые и сгенерированные данные остаются в `runtime/`; LoRA-копия —
  `runtime/models/loras/<id>.safetensors`; удалённые модели — `runtime/trash/`.
- Миграция: `scripts/migrate_to_registry.py` (pathlib, `--dry-run`, бэкап в
  `runtime/backup/migrate_to_registry-<ts>/`).

## Decisions

### D-001 — Реестр как tracked JSON, не БД
Один файл, читается агентами и человеком, версионируется Git'ом. Отклонено:
SQLite/сервис — избыточно для <10 моделей.

### D-002 — Данные per-model в `models/<id>`, runtime — для тяжёлого
Tracked-конфигурация в Git, генерат в gitignored подпапках; `runtime/` — только
монтируемые в ComfyUI копии, бэкапы и trash. Отклонено: хранить датасет в
`runtime/` — теряется связь модель↔данные.

### D-003 — Удаление = перенос в `runtime/trash/`
Мягкое удаление с timestamp; восстановимо вручную. Отклонено: немедленный
`rm -rf` — необратимая потеря датасетов и LoRA.

### D-004 — Миграция отдельным идемпотентным скриптом
Один прогон на сервере (SYSADMIN-HANDOFF), повторный запуск — no-op,
существующие целевые файлы никогда не перезаписываются.

## Affected contours

- Repository: `models/`, `scripts/`, `.gitignore`, документация.
- GPU/VPS: однократный запуск миграции через handoff; код менеджера —
  отдельная задача.

## Risks

- Риск: расхождение старых корневых путей (`reference/`, `dataset/`, `posts/`)
  и новых per-model. Митигируется миграцией и обновлением README;
  `queue_workflow.py` переводится на per-model пути отдельной задачей.
- Риск: случайное удаление модели. Митигируется trash-политикой.

## Test design

- compile: `python -m py_compile scripts/migrate_to_registry.py` (P0).
- json: `python -m json.tool models/registry.json` (P0).
- unit/idempotency: повторный запуск миграции на копии дерева — no-op (P1).
- runtime-manual: запуск миграции на сервере по handoff, проверка папок (P2).
