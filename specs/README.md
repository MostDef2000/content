# Content Specifications

Status: Active

Этот каталог — долговременный SDD-слой проекта `content`.

## Source-of-truth hierarchy

- GitHub Issue: канонический backlog, авторизация и audit-история (issue #1 — деплой).
- `specs/<feature>/spec.md`: ЧТО/ЗАЧЕМ + измеримые NFR и evidence.
- `plan.md`: КАК, решения, отклонённые альтернативы, риски, тест-дизайн.
- `tasks.md`: ограниченное исполнение, acceptance-трассируемость, DoD.
- Исходный код: реализация.
- `AUDIT.md` / `SYSADMIN-HANDOFF.md`: инфраструктурный контур и передача сисадмин-агенту.
- Runtime-свидетельства: operational-истина, записывается обратно в активные артефакты.

## Feature identifiers

`NNN-feature-slug`; полное имя каталога каноническое.

## Normal flow

```text
Issue + видимая область работ
-> commit approved
-> spec / plan / tasks
-> реализация + проверки (py_compile / json / bash -n / compose config)
-> push в main (в рамках принятой области)
-> деплой через SYSADMIN-HANDOFF v1 (отдельный цикл с агентом)
-> runtime-фидбек обратно в artifacts
```

## Active features

- `001-sdd-adoption` — подключение SDD-процесса оркестратора к проекту.
- `002-multi-model-registry` — реестр моделей (`models/registry.json`),
  per-model папки `models/<id>/`, мягкое удаление в `runtime/trash/`,
  идемпотентная миграция `scripts/migrate_to_registry.py`.
- `003-prompt-profile-and-library` — `prompts.py` (сборка positive/negative с
  guardrail), per-model `prompt_profile.json`, библиотека сцен
  `models/library.json`.
- `004-explicit-dataset` — manifest-aware `lora/caption.py`: per-model
  `models/<id>/dataset/manifest.json` (теги tasteful/explicit), age-floor и
  trigger-word фикс базовой каптион-строки, guardrail-проверка до записи.
- `005-dual-mode` — флаг `--mode public|private` в `queue_workflow.py post`:
  папки назначения `models/<id>/posts/` и `models/<id>/posts-private/`.
- `006-clear-dataset` — массовая очистка обучающего датасета модели
  (`models/<id>/dataset/`) с мягким удалением в `runtime/trash/dataset/`
  (manager endpoint `DELETE /api/models/{id}/dataset?confirm=<id>` +
  UI-диалог, подтверждение вводом id); реализовано в Фазе 3 — unit-тесты
  `tests/test_clear_dataset.py`, серверный smoke — SYSADMIN-HANDOFF.
- `007-group-photos` — подкоманда `group` в `queue_workflow.py`: группа из
  1..5 моделей (последовательная генерация, guardrail по возрасту члена),
  PIL-композит (`row`/`grid2`/`grid3`), папка поста первичной модели
  (`posts`/`posts-private`) с `group.json`; зависимость `Pillow>=10,<13` в
  `management/requirements.txt` (ленивый импорт).
