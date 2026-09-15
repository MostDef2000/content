# Feature Specification: Clear dataset (bulk soft-delete reset)

- Feature: 006-clear-dataset
- Status: Active

> Примечание: отдельный контракт фичи в dispatch Фазы 3 отсутствовал; spec
> собран по существующим паттернам репозитория (per-image soft-delete
> датасета в manager, layout 002, manifest 004), подтверждён оркестратором
> и реализован в Фазе 3.

## Product outcome

Оператор может одной операцией очистить весь обучающий датасет модели —
пары `models/<id>/dataset/images/*` + `.txt` и манифест
`models/<id>/dataset/manifest.json` — с мягким удалением в
`runtime/trash/dataset/<model_id>-wipe-<timestamp>/` (тот же контур, что
существующее по-файловое удаление в manager). Это подготовка к чистому
пере-расширению датасета (новый эталон, изменённый промт персонажа) без
ручного удаления файлов по одному и без потери данных (trash никогда не
автоочищается).

## User scenarios

1. Оператор выбрал новый эталон и хочет собрать датасет заново — нажимает
   «Очистить датасет» в UI (или вызывает endpoint) и подтверждает вводом id
    модели; все пары изображение+каптион и manifest переносятся в trash
    (копия `manifest.json` — вместе с ними), `images/` остаётся пустой,
    манифест сразу сбрасывается в пустой (`{"version": 1, "images": []}`).
2. Попытка без подтверждения (confirm ≠ id) отклоняется — ничего не
   переносится.
3. Данные восстановимы вручную из `runtime/trash/dataset/` (мягкое
   удаление, как у моделей и одиночных кадров).

## Requirements

- Операция затрагивает только `models/<id>/dataset/` (`images/`,
  `manifest.json`); `reference/`, `posts/`, `posts-private/`,
  `lora/output/` и данные других моделей не трогаются.
- Мягкое удаление атомарно: весь каталог `images/` переносится одним
  `os.replace` (тот же том) в `runtime/trash/dataset/<model_id>-wipe-<ts>/`
  (коллизия имени → суффиксы `-2`, `-3`, …); копия `manifest.json` кладётся
  в trash-каталог, манифест сбрасывается в `{"version": 1, "images": []}`,
  пустой `images/` пересоздаётся. Сбой переноса оставляет датасет нетронутым
  либо перенесённым целиком (никаких полу-удалённых состояний внутри
  images/).
- Подтверждение обязательно: `confirm` = `model_id` в
  `DELETE /api/models/{model_id}/dataset?confirm=<id>` (как
  `DELETE /api/models/{id}` у удаления модели).
- Очистка при активной джобе модели отклоняется с 409: хелпер
  `_model_has_active_job` (queued И running; solo-джоба ИЛИ групповое
  членство по списку `models`).
- После очистки `lora/caption.py`/manager работают с пустым датасетом без
  ошибок: манифест создаётся заново при следующей записи (reconcile 004).
- Очистка пустого датасета — безопасный no-op (без мусорных папок в trash).
- Guardrail-слои (003/004) и схема манифеста (version 1) не меняются.

## Acceptance criteria

- [ ] Операция переносит только содержимое `models/<id>/dataset/`; всё
      перенесённое лежит в `runtime/trash/dataset/<id>-wipe-<ts>/`.
- [ ] Без подтверждения (confirm ≠ id) — отказ (4xx/отмена), ничего не
      перенесено.
- [ ] После очистки `/api/dataset` отдаёт пустой список; последующий
      expand/caption.py создают манифест заново.
- [ ] Перенос атомарен (один `os.replace` всего каталога): сбой оставляет
      images/ нетронутым либо переносит его целиком (инвариант пар jpg+txt
      сохранён).

## NFR

- NFR-1: отсутствие потери данных — evidence: move в
  `runtime/trash/dataset/`, не unlink.
- NFR-2: изоляция модели — evidence: пути операции ограничены
  `models/<id>/dataset/` и `runtime/trash/dataset/`.

## Compatibility and boundaries

Развивает 002 (per-model layout, trash) и 004 (manifest-reconcile); не
меняет `scripts/queue_workflow.py`, `lora/caption.py`, `train_lora.sh`.
Затрагивает manager (`management/main.py` endpoint + `static/index.html`
UI). Реализовано в Фазе 3 (endpoint, UI-диалог, unit-тесты
`tests/test_clear_dataset.py`); серверный smoke — SYSADMIN-HANDOFF.
