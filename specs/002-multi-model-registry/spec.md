# Feature Specification: Multi-model registry

- Feature: 002-multi-model-registry
- Status: Active

## Product outcome

Проект `content` поддерживает несколько взрослых вымышленных AI-моделей:
`models/registry.json` — единый реестр (id, имя, возраст, рост, теги,
активная модель, lora-файл, группы), а данные каждой модели живут изолированно
в `models/<id>/`. Valery (`valery23`) — первая модель реестра.

## User scenarios

1. Оператор открывает пульт и видит список моделей реестра; активная модель
   помечена; генерация (candidates/expand/post) идёт от активной модели.
2. Оператор переключает активную модель — реестр обновляется, новые артефакты
   (референсы, датасет, LoRA-копия, посты) пишутся в папки этой модели.
3. Оператор удаляет модель — её папка переносится в `runtime/trash/` (мягкое,
   восстановимое удаление), а не стирается безвозвратно.

## Requirements

- `models/registry.json` (tracked): `version`, `models[]` с полями `id`,
  `name`, `age`, `height_cm`, `tags`, `active`, `created`, `lora`, `groups`;
  активна ровно одна модель.
- Per-model папки: `models/<id>/{reference/candidates,dataset/images,lora/output,posts,posts-private}`
  + tracked-файлы `models/<id>/character.json`, `models/<id>/prompt_profile.json`.
- Сгенерированные per-model данные не попадают в Git (`.gitignore`).
- LoRA-копия для генерации: `runtime/models/loras/<id>.safetensors`.
- Удаление модели = перенос её папки в `runtime/trash/`.
- Миграция старого single-model layout: `scripts/migrate_to_registry.py`
  (идемпотентная, с бэкапом в `runtime/backup/`).
- Все модели — взрослые вымышленные персонажи (guardrail — см. 003).

## Acceptance criteria

- [ ] `models/registry.json` валиден (json.tool), `valery23` активна.
- [ ] `models/valery23/character.json` = корневой character.json + top-level `body`.
- [ ] `.gitignore` покрывает `models/<id>/{reference,dataset,lora/output,posts,posts-private}`.
- [ ] `scripts/migrate_to_registry.py` идемпотентен (повторный запуск — no-op),
      поддерживает `--dry-run`, делает бэкап перед переносами.
- [ ] README/AGENTS описывают per-model layout, LoRA-копию и trash-удаление.

## NFR

- NFR-1: реестр — один JSON-файл без внешних зависимостей — evidence:
  `models/registry.json`.
- NFR-2: миграция безопасна для повторного запуска и не требует остановки
  ComfyUI — evidence: docstring скрипта + `--dry-run`.

## Compatibility and boundaries

Не меняет существующие workflow JSON, `queue_workflow.py`, `caption.py`,
`train_lora.sh` (совместимость путей до отдельных задач); manager/UI
подключаются к реестру позже. Не трогает `dataset/`, `posts/`, `runtime/`
вне read-only миграции и бэкапа.
