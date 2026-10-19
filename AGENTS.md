# Content Agent Entry Point

Status: Active

Проект: локальная генерация консистентной взрослой AI-модели (Valery) —
ComfyUI + FLUX.1-dev FP8 + LoRA + веб-пульт (`manager`).
GitHub: `MostDef2000/content` (private). Инфраструктура: GPU-сервер (ZeroTier
`10.123.239.102`, за NAT) + РУ-хост (`82.146.37.153`, ZeroTier 10.123.239.101,
nginx-фронт + домен `3d.mostdef.ru`; Caddy установлен, но отключён). NL-хост не
используется. Серверные действия: рутинные деплои кода — через
dispatch-воркфлоу (deploy lane A: подтверждение владельца в чате →
`gh workflow run`, см. `.github/workflows/deploy.yml`); всё остальное (bootstrap,
миграции, секреты, nginx, фаервол) — только через сисадмин-агента
(`SYSADMIN-HANDOFF v1`, модель полос в AGENTS.md песочницы).

## Canonical entry points

- SDD: `.specify/memory/constitution.md`, `specs/README.md`, активные
  `specs/<feature>/{spec,plan,tasks}.md`.
- Деплой: `.github/workflows/deploy.yml` + `deploy/deploy-wrapper` (dispatch,
  lane A); серверная история — issue #1 (canonical checklist), `AUDIT.md`
  (отработанные SYSADMIN-HANDOFF-экземпляры из репо удалены, аудит в
  `AUDIT.md`).
- Запуск и поток генерации: `README.md`.

## Source authorization

Общие правила допуска — в глобальном уставе `~/.config/opencode/AGENTS.md`
(первоисточник: profiles/global/** в репо-мастерской ai-playground); фраза
допуска — `commit approved`; она никогда не авторизует серверные мутации и
деплои.

Рутинные деплои кода — через GitHub deploy-workflow (dispatch по протоколу
DEPLOY-TRIGGER после подтверждения владельца — lane A, см.
`.github/workflows/deploy.yml`); всё остальное (bootstrap, миграции, секреты,
nginx, фаервол, ручной SSH: compose up, скачивание моделей, Caddy) — только
через SYSADMIN-HANDOFF.

## Hard rules

1. `main` — единственный источник истины репозитория.
2. Одна задача = один canonical Issue; значимые изменения ссылаются на один feature spec.
3. Тяжёлые GPU-задачи — строго по одной (busy-лок в `manager`); 12 ГБ VRAM общие с YOLO-воркером.
4. Никогда не коммитить: `.env`, секреты, токены, `runtime/`, `dataset/`, `posts/`,
   `reference/`, `ai-toolkit/`, `lora/output/`, веса моделей, логи, сгенерированный вывод —
   и те же данные в per-model путях: `models/*/reference/`, `models/*/dataset/`,
   `models/*/lora/output/`, `models/*/posts/`, `models/*/posts-private/`.
5. Персонаж — взрослый вымышленный; без реальных людей и несовершеннолетних.
6. Секреты только в `.env` на сервере (`chmod 600`); в чат и Git не попадают.

## Данные и папки (фаза 1)

- Реестр моделей: `models/registry.json`; данные модели — `models/<id>/`
  (tracked: `character.json`, `prompt_profile.json`; gitignored генерат:
  `reference/candidates/`, `dataset/images/`, `lora/output/`, `posts/`,
  `posts-private/`). Библиотека сцен: `models/library.json`.
- LoRA-копия для генерации — `runtime/models/loras/<id>.safetensors`.
- FLUX-слот LoRA: runtime/models/loras/<id>-flux.safetensors (registry lora_flux) — для FLUX/uncensor постов; SDXL-слот <id>.safetensors остаётся дефолтом.
- Удаление модели — перенос её папки в `runtime/trash/` (мягкое удаление).
- Разовая миграция старого single-model layout —
  `scripts/migrate_to_registry.py` (идемпотентна, бэкап в `runtime/backup/`).
