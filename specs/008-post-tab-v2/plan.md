# Implementation Plan: Post tab v2 (per-post model binding, scene meta, caption cleanup)

- Specification: specs/008-post-tab-v2/spec.md
- Status: Active

## Architecture

- Сервер: `job_post` (management/main.py) — одна точка привязки модели.
  `payload.model_id` (опц.) → `_model_or_404` (реестр, 404/422) → вся
  существующая цепочка (LoRA-слоты `_lora_name_for_engine`, character/
  profile, guardrail-возраст, папка поста) работает с выбранной моделью,
  потому что переменная `model`/`model_id` просто переопределяется до
  этих шагов. Дефолт (пусто/нет ключа) — `_active_model_or_404()`, т.е.
  прежний код-путь.
- CLI: `queue_workflow.py post` уже принимает `--model` (T002 spec 007:
  `resolve_model_id` только при наличии `--model`) — сервер передаёт
  выбранный id; изменений резолва не нужно. Добавляются только
  `--scene-id`/`--scene-name` (pass-through) и запись `post.json` в финале
  `generate_post` (паттерн — `group.json` из spec 007, `_now_iso`-стиль
  timestamp; seed записывается фактический из `args.seed`).
- API: `list_models` дополняет каждую запись полем `engine`; логика
  вынесена в helper `_model_engine(model)` (тот же расчёт слотов, что в
  `_active_model_summary`, без дублирования).
- UI: селектор «Модель поста» (`postModel`) над сценой; `state.postModel`
  (`""` = активная); тело джобы поста несёт `model_id` только когда
  выбран не-дефолт. Дефолт движка: при смене селектора — `setPostEngine`
  по `engine` выбранной модели (из `/api/models`), `postEngineFor` = id
  выбранной модели; поллинг `/api/status` перезаписывает движок только
  пока селектор в дефолте (активная) — прежняя семантика без конфликта.
  Карточка поста: строка сцены под фото (`scene_name` или «—»), клик —
  переключение на вкладку «Библиотека сцен» + inline-редактор
  (`openLibraryEdit`) найденной по `scene_id` сцены; кнопка
  «сгенерировать заново» — `POST /api/jobs/post {scene_id, model_id,
  engine, mode}` из meta (seed не передаётся → авто).

## Decisions

### D-001 — model_id резолвится в job_post, CLI-флаг уже есть
Единственная правка сервера — валидация и переопределение `model_id` в
`job_post`; cmd-конвейер (`--model`), слоты и guardrail переиспользуются
как есть. Отклонено: отдельный эндпоинт `/api/posts` с собственным
резолвом — дубль валидации и расхождение контрактов.

### D-002 — post.json пишет queue_workflow, а не менеджер
Папку поста создаёт и финализирует `generate_post` (timestamp-имя по
умолчанию неизвестно серверу до конца джобы); запись meta в том же
try-блоке, что photo/prompt.txt, откатывается вместе с папкой при сбое
(rollback `rmtree` уже удаляет недописанный meta). Отклонено: серверный
писатель по опросу папок — гонки с 5-секундным поллингом UI.

### D-003 — scene_name дублируется в meta (не только scene_id)
Клик-редактор требует живую библиотеку, но имя на карточке должно
переживать удаление сцены из библиотеки (требование владельца: «если
сцена удалена — имя из meta всё равно показывается, клик не открывает
редактор»). Отклонено: резолв имени на лету из library.json — имя
исчезает вместе со сценой.

### D-004 — engine в /api/models, а не отдельный эндпоинт
Минимальный diff: одно вычисляемое поле в существующем ответе, которое
UI и так загружает (`loadModels`); отдельный «лёгкий параметр» — лишний
эндпоинт и лишний запрос. Расчёт слотов выделен в `_model_engine`.

### D-005 — дефолт движка сервера остаётся flux
Серверная семантика `_validate_engine_fields` (default flux) не меняется:
UI после этой волны всегда шлёт движок явно (пер-модельный дефолт
применяется на клиенте). Смена серверного дефолта сломала бы back-compat
клиентов без `engine` (например, CLI-трек), а п.15 (серверный дефолт по
arch) — отдельный пункт, не эта подволна.

### D-006 — caption-контракт не трогается (п.10 — только UI)
`payload.caption`, `--caption ""` в cmd, caption.txt в generate_post и
тесты контракта (test_post_empty_caption_accepted_no_400 и др.) остаются:
Instagram-трек постит каптионы через API/CLI. Удаляется только
текстовое поле формы вкладки «Пост» и его значение в теле джобы.

## Affected contours

- Repository: `management/main.py`, `management/static/index.html`,
  `scripts/queue_workflow.py`, `tests/`, `specs/008-post-tab-v2/`,
  комментарий в шапке `.github/workflows/deploy.yml` (док-минор:
  ci.yml теперь validate+test).
- Server: деплой lane A (рестарт manager-контейнера); миграций нет —
  старые посты читаются без post.json (`scene_name: null`).

## Risks

- Риск: UI перезапишет ручной выбор движка при смене модели в селекторе.
  Митигируется прежней семантикой `postEngineFor` (переустановка только
  при смене модели).
- Риск: перезапуск поста после удаления модели/сцены из реестра/библиотеки
  → 404 от API (модель) или «scene not found» (сцена резолвится на сервере)
  — alert в UI; имя сцены на карточке остаётся из meta.
- Риск: расхождение meta с фактическим запуском (meta пишется из args,
  а не из результата ComfyUI). Seed/модель/движок/режим — аргументы
  запуска, фактический seed = args.seed (один пост = один seed), поэтому
  расхождение возможно только при ручной правке папки.
- Риск: битый/чужой post.json в папке поста → list-эндпоинты деградируют
  к `scene_name: null` (try/except при чтении), карточка не падает.

## Test design

- `tests/test_post_model_override.py` — стиль `test_engine_validation.py`:
  temp-WORKSPACE до импорта main, `_LaunchSpy`/`_with_spy`, `_with_registry`
  (реестр пер-тест, восстановление seed-JSON), try/finally, двойной раннер
  (pytest + `__main__`). Кейсы: AC-1..AC-6 (+ cmd `--scene-id/--scene-name`,
  post.json-файл generate_post, list_posts meta/старый пост).
- Регресс: полный `pytest tests/ -q` (113 прежних, включая caption-контракт),
  per-file `__main__` нового файла, py_compile обоих python-контуров,
  grep-проверки формы поста.
