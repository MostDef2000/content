# Implementation Plan: Group photos (multi-model composite)

- Specification: specs/007-group-photos/spec.md
- Status: Active

## Architecture

- Подкоманда `group` в `scripts/queue_workflow.py`: парсер + `generate_group`
  + helpers `_cover_crop`/`_compose_grid` и константа `GROUP_LAYOUT_COLUMNS`.
- Пайплайн: pre-flight (валидация members, ленивый PIL, реестр/character/
  profile/lora на каждого члена) → staging `runtime/output/group/<batch>` →
  последовательная генерация членов (flux_lora, транзит
  `group/<batch>/<id>`, `shutil.move` в staging `NN_<id><ext>`) →
  PIL-композит → папка поста первичной модели (photo.jpg, caption.txt,
  prompt.txt, group.json) → rollback при любом сбое.
- `main()`: `resolve_model_id` вызывается только для подкоманд с флагом
  `--model` (`hasattr`) — у `group` единого model_id нет, члены резолвятся
  в собственном pre-flight.

## Decisions

### D-001 — 0-based индекс члена i
Seeds в `group.json` определены как `[--seed+i for i in range(N)]`; тот же
0-based i используется для seed члена, имени staging-файла (`{i:02d}_<id>`)
и прогресс-принта. Единый i исключает расхождение записанных seeds с
фактическими. Отклонено: 1-based нумерация файлов/принта — потребовала бы
другой формулы seeds.

### D-002 — Rollback удаляет destination только созданный этим запуском
`destination.mkdir(exist_ok=False)` при уже существующей папке бросает
`FileExistsError`; безусловный `rmtree` в except-блоке удалил бы прежнюю
(чужую) папку. Переменная `destination` присваивается только после
успешного `mkdir`, поэтому rollback затрагивает лишь созданное этим
запуском. Отклонено: безусловный `rmtree(destination)`.

### D-003 — Ленивый импорт Pillow
`from PIL import Image` внутри `generate_group` (fail-fast до очереди) и
`_compose_grid`; `py_compile` и `--help` работают без Pillow. Зависимость
объявлена в `management/requirements.txt` (`Pillow>=10,<13`), образ manager
пересобирается деплоем. Отклонено: модульный импорт (ломал окружения без
Pillow, включая песочницы и pre-flight на сервере).

### D-004 — cell = первый член, scale-to-cover + center-crop
Первый член задаёт геометрию (детерминированно, без эвристик «самого
большого»); кадры других членов приводятся к cell cover-масштабированием с
центральным кропом, равные — вставляются как есть. Отклонено: distort
(искажение пропорций) и letterbox (полосы).

### D-005 — prompt.txt: primary positive первой строкой + секции по членам
Первая строка повторяет формат одиночного поста (`positive + "\n"`) —
совместимость с существующими потребителями prompt.txt; далее для каждого
члена `"--- <id> ---\n<positive_i>\n"` — возраста членов могут отличаться,
guardrail-фразы (и positives) поэтому различаются; полный аудит в одном
файле. Positive primary дублируется в своей секции — осознанно, для
однообразного парсинга секций.

### D-006 — Staging внутри runtime/output/group
Тот же контур транзита, что posts (002/005): перенос transit→staging
атомарен в рамках одной ФС; при откате staging удаляется только пока пуст,
недописанные transit-остатки допустимы (как у posts).

## Affected contours

- Repository: `scripts/queue_workflow.py`, `management/requirements.txt`,
  `specs/`.
- Server: пересборка образа manager (pip установит Pillow) и рестарт —
  через SYSADMIN-HANDOFF; GPU-контур ComfyUI без изменений.

## Risks

- Риск: 12 ГБ VRAM общие с YOLO-воркером. Митигируется последовательной
  генерацией (по одному члену за раз — структура цикла), как в expand.
- Риск: разные пропорции членов → центральный кроп теряет края кадра.
  Митигируется эталоном cell от первого члена и единым разрешением графа.
- Риск: коллизия имени (два запуска в одну секунду или занятый `--name`) →
  `exist_ok=False` падает; прежняя папка не затрагивается (D-002).
- Риск: несовместимость Pillow вне диапазона 10..13 — фикс диапазоном в
  requirements (локально проверен резолв 12.3.0).

## Test design

- compile: `python -m py_compile scripts/queue_workflow.py
  lora/caption.py management/main.py` (P0).
- cli: корневой `--help` (group в списке) и `group --help` (все флаги,
  choices, defaults) (P0).
- fixture: PIL Image.new → реальные `_cover_crop`/`_compose_grid` — размеры
  row/grid2/grid3 и заливка пустого тайла (P0).
- unit-manual: ValueError на дубликатах/>5, RuntimeError на неизвестном id,
  FileNotFoundError на отсутствии LoRA — все до создания staging (P1).
- runtime-manual: group из 2 моделей на сервере после пересборки образа
  manager (SYSADMIN-HANDOFF) (P2).
