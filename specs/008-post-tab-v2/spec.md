# Feature Specification: Post tab v2 (per-post model binding, scene meta, caption cleanup)

- Feature: 008-post-tab-v2
- Status: Active
- Source: issue #2 (MostDef2000/content), комментарий владельца 5707498338, пункты 10, 11, 14

## Product outcome

Вкладка «Пост» manager-а становится самостоятельным рабочим местом:

1. **п.14 — привязка поста к модели.** Селектор модели на вкладке «Пост»;
   выбор привязывает только этот пост (промт-профиль, LoRA-слоты, возраст/
   guardrail выбранной модели), активная модель НЕ меняется (датасет,
   кандидаты, трейн — по-прежнему от активной). По умолчанию — активная
   модель. `POST /api/jobs/post` принимает опциональный `model_id` (пусто/
   отсутствует → активная модель, прежнее поведение).
2. **п.11 — метаданные сцены на карточке поста.** Каждый новый пост пишет
   лёгкий `post.json` (scene_id, scene_name, model_id, engine, mode, seed,
   created_at). В галерее постов под фото — название использованной сцены;
   клик открывает редактор этой сцены из библиотеки сцен (тот же inline-
   редактор вкладки «Библиотека»); шорткат «сгенерировать заново»
   перезапускает пост по сохранённым scene_id/model_id/engine/mode
   (seed пустой = авто).
3. **п.10 — чистая форма поста.** Поле «Caption (для Instagram,
   необязательно)» из вкладки «Пост» удалено полностью. Остаются seed, имя
   папки и стандартные поля поста. Бэкенд-контракт caption не меняется:
   `POST /api/jobs/post` по-прежнему принимает опциональный `caption`
   (Instagram-трек постит каптионы через API/CLI), CLI `--caption` и
   тесты контракта остаются.

## User scenarios

1. Оператор выбирает в селекторе «Модель поста» модель B при активной A и
   генерирует пост: cmd джобы несёт `--model B`, LoRA-слот/профиль/guardrail
   — от B, пост пишется в `models/B/posts|posts-private/`; бейдж активной
   модели и `/api/status.active_model` остаются A.
2. Оператор оставляет селектор «по умолчанию (активная)» — поведение
   побайтово прежнее (model_id = активная, back-compat для существующих
   клиентов без `model_id`).
3. Неизвестный `model_id` → HTTP 4xx «модель не найдена», джоба не
   запускается, активная модель не меняется.
4. Оператор открывает карточку готового поста: под фото — имя сцены;
   клик открывает редактор этой сцены в библиотеке (правка + «Сохранить»
   пишут `models/library.json`); кнопка «сгенерировать заново» запускает
   `POST /api/jobs/post {scene_id, model_id, engine, mode}` из `post.json`
   (новый seed — авто).
5. Старый пост без `post.json` (и пост с удалённой из библиотеки сценой):
   имя сцены показывается из meta (или «—», если meta нет), клик по имени
   не открывает редактор, кнопка перезапуска по-прежнему доступна.

## Requirements

- `job_post`: `payload.model_id` — опциональная строка; непустая →
  `_model_or_404` (неизвестный id → 404 «model not found», формат → 422);
  валиден → джоба/cmd/запись используют выбранную модель (LoRA-резолв по
  слотам выбранной модели, character.json/prompt_profile.json и
  guardrail-возраст — выбранной модели). Отсутствие/пустота → активная
  модель (прежнее поведение). Group-эндпоинт не затронут;
  `/api/status` не меняется.
- `GET /api/models`: каждая модель несёт вычисленный `engine`
  (`"sdxl" | "flux" | null` — по развёрнутым LoRA-слотам, та же семантика,
  что у `_active_model_summary.engine`) — источник дефолта движка UI.
- UI вкладки «Пост»: селектор «Модель поста» (вариант «по умолчанию —
  активная» + все модели реестра); дефолт движка = arch LoRA выбранной
  модели (перезастановка при смене селектора; ручной выбор движка не
  перезаписывается поллингом, пока модель та же — прежняя семантика
  `postEngineFor`, адаптированная на селектор).
- `queue_workflow.py post`: флаги `--scene-id`/`--scene-name` (опциональные,
  default `""`); финал поста всегда пишет `post.json`
  (`{scene_id, scene_name, model_id, engine, mode, seed, created_at}`).
- `list_posts` / `list_posts_private`: читают `post.json` (битый → как
  отсутствующий) и добавляют в карточку `scene_id`, `scene_name`,
  `model_id`, `engine`, `mode`, `seed`; без `post.json` → `scene_name: null`.
- Форма поста без Caption-поля; остальные подписи caption (групповая форма,
  «Caption all» датасета, бэкенд `--caption`) не тронуты.

## Measurable acceptance

- AC-1: `job_post` с валидным `model_id` → cmd содержит `--model <id>`,
  слоты/профиль выбранной модели, запись джобы отражает `model_id`,
  активная модель в реестре не изменилась. (tests/test_post_model_override.py)
- AC-2: неизвестный `model_id` → 4xx «модель не найдена», `_launch` не
  вызывается. (tests/test_post_model_override.py)
- AC-3: payload без `model_id` → прежнее поведение (активная модель,
  back-compat). (tests/test_post_model_override.py)
- AC-4: guardrail применяется по character.json выбранной модели
  (возраст/фраза другой модели). (tests/test_post_model_override.py)
- AC-5: `generate_post` пишет `post.json` с полным набором полей;
  `list_posts`/`list_posts_private` отдают `scene_name` из meta и
  `scene_name: null` для старых постов. (tests/test_post_model_override.py)
- AC-6: `GET /api/models` отдаёт `engine` на каждую модель.
  (tests/test_post_model_override.py)
- AC-7: в форме вкладки «Пост» нет поля Caption; групповая форма и
  датасет-«Caption all» не тронуты. (grep по index.html)
- AC-8: `python3 -m py_compile management/main.py scripts/queue_workflow.py`;
  `python3 -m pytest tests/ -q` — 113 прежних + новые зелёные; per-file
  раннер нового тест-файла зелёный.
