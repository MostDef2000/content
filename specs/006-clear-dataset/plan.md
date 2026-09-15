# Implementation Plan: Clear dataset (bulk soft-delete reset)

- Specification: specs/006-clear-dataset/spec.md
- Status: Active

## Architecture

- Manager: `DELETE /api/models/{model_id}/dataset?confirm=<id>` — атомарный
  перенос всего каталога `images/` одним `os.replace` в
  `runtime/trash/dataset/<model_id>-wipe-<int(time.time())>/` (коллизия
  имени → суффиксы `-2`, `-3`, …); копия `manifest.json` в trash-каталог,
  сброс манифеста в `{"version": 1, "images": []}`, пустой `images/`
  пересоздаётся; переиспользует `DATASET_TRASH`. Роут зарегистрирован до
  single-image delete (`/dataset/{filename}`), чтобы пути не пересекались.
- Guard: отказ 409 при активной джобе модели — `_model_has_active_job`
  (queued И running; solo-джоба ИЛИ групповое членство).
- UI: кнопка «Очистить датасет» в карточке датасета с подтверждением вводом
  id модели (как `deleteModel`).
- Манифест: копия переносится в trash вместе с каталогом; сам манифест
  сбрасывается сразу в пустой reconciled-вид (004) — «создание заново при
  следующей записи» остаётся в силе для последующих записей.
- Операции сериализуются существующим `_manifest_lock` (общий с caption.py
  подпроцессом и per-image delete).

## Decisions

### D-001 — Мягкое удаление в существующий DATASET_TRASH
Соответствует удалению модели и per-image delete; данные восстановимы
вручную, инвариант «trash не автоочищается» сохраняется. Отклонено: жёсткий
`rm -rf` и отдельная корзина под датасеты.

### D-002 — Подтверждение вводом model_id
Массовая деструктивная операция; тот же UX-барьер, что при удалении модели.

### D-003 — Ручной запуск, без GPU-лока
Операция дисковая и мгновенная; не претендует на GPU. Достаточно
`_manifest_lock` против гонок с caption/expand-эндпоинтами manager.

## Affected contours

- Repository: `management/main.py`, `management/static/index.html`,
  `specs/`.
- Server: рестарт `valery-manager-1` — через SYSADMIN-HANDOFF (без
  пересборки образа: зависимости не меняются).

## Risks

- Риск: очистка во время активного expand/caption-процесса — гонка файлов.
  Митигация: `_manifest_lock` + отказ при занятом GPU/джобе (как у других
  мутирующих endpoints manager).
- Риск: сиротские `.txt` без картинок при частичном сбое. Митигация: парный
  перенос (image+caption), как в per-image delete; инвариант пар сохранён.

## Test design

- compile: `python -m py_compile management/main.py` (P0).
- unit: `tests/test_clear_dataset.py` — 6 тестов (confirm, перенос в trash +
  сброс манифеста, 409 при solo- и групповой джобе, no-op пустого датасета,
  суффикс коллизии `-2`) (P0).
- cli/api-manual: отказ без confirm; перенос в trash при confirm (P1).
- runtime-manual: очистка датасета тестовой модели, проверка trash и
  последующего expand + caption (P2, SYSADMIN-HANDOFF).
