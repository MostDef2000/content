# Tasks: Post tab v2 (per-post model binding, scene meta, caption cleanup)

- Specification: specs/008-post-tab-v2/spec.md
- Plan: specs/008-post-tab-v2/plan.md
- Issue: #2 (комментарий 5707498338, пункты 10, 11, 14) — канонический issue.

## Delivery tasks

- [x] T001 Сервер (п.14): `job_post` принимает опциональный `payload.model_id`
      — пусто/нет → активная модель (back-compat); указан → `_model_or_404`
      (4xx «модель не найдена»), LoRA-слоты/профиль/guardrail и cmd
      `--model <id>` — от выбранной; запись джобы отражает `model_id`;
      активная модель не меняется; group-эндпоинт не тронут. DoD:
      py_compile; tests/test_post_model_override.py — AC-1..AC-4.
- [x] T002 Сервер: `GET /api/models` добавляет вычисляемый `engine`
      (`_model_engine` — та же семантика слотов, что у
      `_active_model_summary`). DoD: py_compile; тест AC-6.
- [x] T003 CLI (п.11): `queue_workflow.py post` — флаги `--scene-id`/
      `--scene-name` (default ""); `generate_post` в финале пишет
      `post.json` {scene_id, scene_name, model_id, engine, mode, seed,
      created_at}; сбой записи откатывается с папкой. DoD: py_compile;
      fixture-тест post.json (AC-5).
- [x] T004 Сервер: `list_posts`/`list_posts_private` читают `post.json`
      (битый = отсутствующий) → `scene_id`, `scene_name`, `model_id`,
      `engine`, `mode`, `seed`; старые посты — `scene_name: null`. DoD:
      py_compile; тест (AC-5).
- [x] T005 UI (п.14): селектор «Модель поста» (дефолт — активная), тело
      джобы несёт `model_id` только для не-дефолта; дефолт движка = arch
      LoRA выбранной модели (`/api/models` engine, адаптация
      setPostEngine/postEngineFor на селектор). DoD: код в index.html
      (секция data-tab="post"); серверная проверка — тестами T001.
- [x] T006 UI (п.11): карточка поста — имя сцены под фото (у старых
      постов без post.json строка сцены не рендерится вовсе — без имени
      и без клика; у удалённой сцены имя из meta, тоже без клика),
      клик → вкладка
      «Библиотека сцен» + inline-редактор этой сцены; кнопка
      «сгенерировать заново» → POST /api/jobs/post {scene_id, model_id,
      engine, mode} (seed авто). DoD: код в index.html (postCard).
- [x] T007 UI (п.10): удалить поле «Caption (для Instagram,
      необязательно)» из формы вкладки «Пост» (label + textarea + значение
      в теле джобы). Не трогать: caption-логику queue_workflow/CLI,
      caption групповой формы, «Caption all» датасета, caption-тесты
      бэкенд-контракта. DoD: grep — поле удалено только из пост-формы.
- [x] T008 Тесты: tests/test_post_model_override.py (стиль
      test_engine_validation.py: _with_registry/_with_spy, try/finally,
      pytest + per-file __main__). DoD: AC-1..AC-6 зелёные обоими
      раннерами.
- [x] T009 Док-минор: шапка .github/workflows/deploy.yml — устаревший
      комментарий «ci.yml пока валидирует только compose/скрипты/YAML»
      приведён к факту (validate + test/pytest); больше ничего в файле
      не меняется. DoD: diff deploy.yml — только блок комментария.
- [x] T010 Верификация: `python3 -m py_compile management/main.py
      scripts/queue_workflow.py`; `python3 -m pytest tests/ -q` — 113
      прежних + новые; per-file раннер; grep caption; `git status
      --short` — только allowed_paths. DoD: все команды зелёные,
      evidence в отчёте.

## Acceptance traceability

- AC-1..AC-4 ← T001, T008.
- AC-5 ← T003, T004, T008.
- AC-6 ← T002, T008.
- AC-7 ← T007 (grep).
- AC-8 ← T010.

## Definition of Done

- [x] T001–T010 выполнены; py_compile/pytest/per-file/grep/git-status
      зелёные; вне allowed_paths изменений нет (включая models/**,
      lora/**, comfy/**, requirements.txt, ci.yml/security.yml).
