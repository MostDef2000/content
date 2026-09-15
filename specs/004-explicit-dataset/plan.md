# Implementation Plan: Explicit dataset and manifest-aware captions

- Specification: specs/004-explicit-dataset/spec.md
- Status: Active

## Architecture

- `lora/caption.py` остаётся единственным входом каптионинга; добавляется
  слой манифеста: `manifest_path`, `load_manifest` (чтение + reconcile с
  диском), `save_manifest` (атомарно), `caption_for(tag, base)`.
- Манифест: `models/<id>/dataset/manifest.json` рядом с `images/`; тег
  записи определяет вариант каптиона (`tasteful`/`explicit`).
- Guardrail: `prompts.validate_positive` прогоняется на итоговом каптионе
  (база + возможный explicit-контекст) перед записью каждого .txt.
- `sys.path.insert(0, str(ROOT))` + `import prompts` — как в
  `scripts/queue_workflow.py`.

## Decisions

### D-001 — Манифест в gitignored `dataset/`, не в tracked-конфиге
Датасет и его метаданные — генерат; теги меняются часто и локально.
Отклонено: tracked `models/<id>/dataset_manifest.json` — смешение конфига и
контента, риск коммита датасет-метаданных.

### D-002 — Reconcile с диском при каждом запуске
Файлы без записи → новая запись (tag=tasteful, caption=null, created=mtime
ISO); записи без файла → исключаются; `version > 1` → RuntimeError.
Отклонено: строгий режим без reconcile — рассинхрон после ручного
добавления/удаления кадров.

### D-003 — Validate-до-записи и один save_manifest
Предупреждение + `return 1` без записи .txt при hits; манифест сохраняется
один раз после цикла (при отказе — не сохраняется этой попыткой).
Отклонено: писать каптион и сообщать о блокировке постфактум — сломанный
датасет для ai-toolkit.

### D-004 — EXPLICIT_BODY_CONTEXT константой без blocked-термов
Строка описывает взрослое вымышленное тело и не содержит термов
`prompts.GUARDRAIL_PATTERNS`, поэтому проходит `validate_positive`.

## Affected contours

- Repository: `lora/caption.py`, `specs/`.
- Server: пересборка не требуется (чистый Python, без новых зависимостей).

## Risks

- Риск: ручная правка .txt расходится с `caption` в манифесте. Митигируется
  тем, что манифест — производная: `caption` обновляется только caption.py.
- Риск: опечатка в `tag` даёт tasteful-каптион вместо explicit. Митигируется
  тем, что любое значение кроме `explicit` трактуется как tasteful (без
  падения), а сверка тегов — на операторе при ревью датасета.

## Test design

- compile: `python -m py_compile lora/caption.py` (P0).
- cli: `python lora/caption.py --help` — сигнатура без изменений (P0).
- fixture: временный `models/demo` с манифестом (explicit-запись) и пустым
  a.jpg; `--force` → .txt содержит model_id, age-floor и
  EXPLICIT_BODY_CONTEXT (P0).
- runtime-manual: прогон на реальном датасете valery23 (P2).
