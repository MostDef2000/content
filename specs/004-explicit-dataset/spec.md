# Feature Specification: Explicit dataset and manifest-aware captions

- Feature: 004-explicit-dataset
- Status: Active

## Product outcome

Датасет LoRA каждой модели поддерживает два класса изображений — `tasteful`
(базовая каптион-строка) и `explicit` (базовая строка + явный контекст
взрослого вымышленного тела) — через per-model манифест
`models/<id>/dataset/manifest.json`. `lora/caption.py` читает манифест,
сверяет его с файлами на диске и пишет .txt-каптионы соответственно;
guardrail `prompts.validate_positive` применяется к каждому каптиону до
записи файла.

## User scenarios

1. Оператор помечает часть датасета как `explicit` в манифесте и запускает
   `python lora/caption.py --model <id> [--force]` — для `explicit`-кадров
   .txt содержит базовую каптион-строку плюс `EXPLICIT_BODY_CONTEXT`, для
   остальных — только базовую.
2. Оператор добавляет/удаляет изображения в
   `models/<id>/dataset/images/` — при следующем запуске caption.py
   дополняет манифест записями для новых файлов (tag=tasteful, caption=null,
   created=mtime ISO) и исключает записи без файла.
3. Попытка записать каптион с блокируемым термом прерывается предупреждением
   (exit code 1); файл в этой попытке не пишется.

## Requirements

- Манифест: `models/<id>/dataset/manifest.json` (рядом с `images/`, не
  внутри; gitignored вместе с датасетом), формат
  `{"version": 1, "images": [{"filename", "tag", "caption", "created"}]}`;
  `version > 1` — ошибка.
- CLI-сигнатура сохраняется: `python lora/caption.py --model <id> [--force]`.
- Age-floor: возраст в базовой каптион-строке не ниже `prompts.AGE_FLOOR`
  (23): `max(prompts.AGE_FLOOR, int(character.get("age", prompts.AGE_FLOOR)))`.
- Trigger-word фикс: базовая строка начинается с `<model_id>` вместо
  литерала `[trigger]`; формат
  `<model_id>, fictional adult woman age <age>, hair, eyes, skin, face, build`.
- `EXPLICIT_BODY_CONTEXT` — константа caption.py, не содержащая термов из
  `prompts.GUARDRAIL_PATTERNS`.
- `caption_for(tag, base)`: `tasteful → base`;
  `explicit → base + ", " + EXPLICIT_BODY_CONTEXT`.
- Запись .txt только после успешного `prompts.validate_positive(caption)`;
  при hits — предупреждение и `return 1`, файл не пишется.
- Манифест сохраняется атомарно (tempfile в той же папке + `os.replace`)
  один раз после цикла.
- Все модели — взрослые вымышленные персонажи; explicit-контекст описывает
  только взрослое вымышленное тело (guardrail — см. 003).

## Acceptance criteria

- [ ] `python -m py_compile lora/caption.py` проходит.
- [ ] `python lora/caption.py --help` показывает прежнюю сигнатуру
      (`--model`, `--force`).
- [ ] Fixture-прогон: `explicit`-запись манифеста даёт .txt с `<model_id>`,
      `fictional adult woman age <max(AGE_FLOOR, age)>` и
      EXPLICIT_BODY_CONTEXT.
- [ ] Манифест реконсилится с диском: новые файлы получают записи
      (tag=tasteful), записи без файла исключаются; `version > 1` — ошибка.
- [ ] Каптион с блокируемым термом не пишется, скрипт завершается кодом 1.

## NFR

- NFR-1: атомарная запись манифеста исключает повреждение при сбое —
  evidence: tempfile + `os.replace` в `save_manifest`.
- NFR-2: guardrail-проверка выполняется до любой записи — evidence: вызов
  `prompts.validate_positive` в `main()` перед `write_text`.

## Compatibility and boundaries

Не меняет `scripts/queue_workflow.py`, workflows ComfyUI, `train_lora.sh`.
Манифест лежит рядом с `images/` и остаётся gitignored
(`models/*/dataset/`). Не трогает `dataset/`, `runtime/`, `posts/`.
