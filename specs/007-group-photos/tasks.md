# Tasks: Group photos (multi-model composite)

- Specification: specs/007-group-photos/spec.md
- Plan: specs/007-group-photos/plan.md

## Delivery tasks

- [x] T001 Парсер `group` (--models CSV 1..5 без дубликатов, --prompt,
      --layout, --mode, --caption, --name, --seed, --lora-strength,
      --negative). DoD: `group --help` показывает все флаги, choices
      layout/mode и defaults.
- [x] T002 `main()`: `resolve_model_id` только при наличии `--model`
      (hasattr). DoD: py_compile; candidates/expand/post без поведенческих
      изменений.
- [x] T003 `generate_group` pre-flight: валидация members (ValueError),
      ленивый PIL (RuntimeError с текстом requirements), реестр/
      character/profile/lora на каждого члена. DoD: все ошибки — до очереди
      и до создания staging.
- [x] T004 Генерация членов: staging `runtime/output/group/<batch>`,
      транзит `group/<batch>/<id>`, seed `--seed+i` (0-based), move в
      staging `NN_<id><ext>`. DoD: py_compile + code review.
- [x] T005 Композит: `_cover_crop` + `_compose_grid` (row/grid2/grid3,
      cell = первый член, заливка 230,230,230). DoD: fixture-прогон с
      проверкой размеров и пустого тайла.
- [x] T006 Финал + rollback: photo.jpg (JPEG q92), caption.txt, prompt.txt,
      group.json; destination удаляется только созданный этим запуском;
      staging — только пока пуст. DoD: code review except-блока.
- [x] T007 `management/requirements.txt`: `Pillow>=10,<13`. DoD: строка
      добавлена; pip-резолв проверен локально (12.3.0).
- [ ] T008 Серверный прогон group (2 модели) после пересборки образа
      manager (SYSADMIN-HANDOFF). DoD: папка в `models/<primary>/posts/`,
      состав файлов полный, композит корректен.
- [x] T009 Manager backend: `POST /api/jobs/group` — валидация payload
      (`_validate_group_payload`: 1..5 уникальных членов, реестр, LoRA,
      caption, prompt/scene_id XOR, layout/mode, коллизия имени) и запуск
      `scripts/queue_workflow.py group`; члены перечисляются в записи джобы
      (видны `_model_has_active_job`). DoD: py_compile;
      `tests/test_group_validation.py` — 14 тестов.
- [x] T010 Manager UI: вкладка «Групповой кадр» (выбор членов, layout,
      mode, caption, seed, запуск джобы через `/api/jobs/group`). DoD:
      код в `management/static/index.html` (секция `data-tab="group"`);
      серверная проверка — в рамках T008.

## Acceptance traceability

- AC-1 (py_compile + корневой help) ← T001, T002, T007.
- AC-2 (group --help: флаги/choices/defaults) ← T001.
- AC-3 (валидация и fail-fast без записи) ← T003.
- AC-4 (fixture композита) ← T005.
- AC-5 (Pillow-ошибка) ← T003, T007.
- Серверный прогон ← T008 (внеполосно, SYSADMIN-HANDOFF).
- Manager-контур (backend + UI, вне AC spec 007) ← T009, T010; evidence:
  `tests/test_group_validation.py` — 14 тестов, py_compile
  `management/main.py`.

## Definition of Done

- [x] T001–T007 выполнены и верифицированы (py_compile, --help, fixture,
      pre-flight прогоны без записи в runtime/).
- [x] T009–T010 (manager backend/UI) выполнены и верифицированы
      (py_compile; tests/test_group_validation.py — 14 тестов).
- [ ] Коммит не содержит секретов, датасетов, весов, runtime-вывода.
- [ ] Серверный прогон (T008) отражён в artifacts при следующем значимом
      изменении.
