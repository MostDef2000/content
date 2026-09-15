# Feature Specification: Prompt profile and scene library

- Feature: 003-prompt-profile-and-library
- Status: Active

## Product outcome

Промты собираются из переиспользуемых частей: per-model
`prompt_profile.json` (лицо/тело/стиль/негатив/сцены) и общей библиотеки сцен
`models/library.json`. Модуль `prompts.py` — единственная точка сборки
positive/negative с жёстким guardrail: только взрослая вымышленная модель,
никаких реальных людей и несовершеннолетних.

## User scenarios

1. Оператор генерирует candidates/expand/post — positive собирается из профиля
   активной модели + каркаса mode + сцены; guardrail-фраза
   «{max(23,age)}-year-old fictional» присутствует всегда.
2. Оператор выбирает сцену из библиотеки (expand-вариации датасета, casting,
   post) вместо ручного набора текста.
3. Оператор пишет свой negative — канонические защитные термы (child, teen,
   underage, minor, real person, …) не могут быть удалены его текстом.

## Requirements

- `prompts.py` (корень, без I/O): `MODES=("candidates","expand","post")`,
  `AGE_FLOOR=23`, `GuardrailError`, `guardrail_phrase(age)`,
  `validate_positive` (word-boundary, case-insensitive: child/teen/teenage/
  underage/minor/loli/shota/schoolgirl/schoolboy/real person|people|woman|man|
  human/celebrity/politician), `build_positive` (identity+guardrail → face →
  body → mode-каркас → style → scene → tail → финальная самопроверка),
  `build_negative` (канонический негатив + mode-добавки + user, dedupe по
  словам, порядок сохранён), `default_profile`, `scene_presets`.
- `models/<id>/prompt_profile.json` — результат `default_profile(character)`.
- `models/library.json` — сцены `{id,name,mode,text,tags}`: 8 expand-текстов
  из `VARIATION_PROMPTS` + ≥2 candidates + ≥2 post.
- Тесты `tests/test_prompts.py` (import-safe, pytest-совместимые).

## Acceptance criteria

- [ ] `build_positive` содержит guardrail-фразу во всех mode;
      `guardrail_phrase(17)` → «23-year-old fictional» (floor).
- [ ] `validate_positive("teen girl")` находит нарушения; «kidney bean»,
      «canteen», «a healthy woman» — чисто (word-boundary).
- [ ] `build_negative()` всегда содержит «real person» и «child»; user-текст
      не вычитает канонические термы.
- [ ] `scene_presets()` непуст и содержит mode="expand"; `models/library.json`
      консистентен с `scene_presets()`.
- [ ] `py_compile prompts.py tests/test_prompts.py`; import-smoke проходит.

## NFR

- NFR-1: `prompts.py` без I/O (чистые функции) — тестируется без сети и
  ComfyUI — evidence: модуль + тесты.
- NFR-2: guardrail — дублирующий слой (код + workflow-тексты), без ложных
  срабатываний на словах-«ложных друзьях» (kidney, canteen) — evidence:
  word-boundary тесты.

## Compatibility and boundaries

Не меняет `queue_workflow.py`, `caption.py`, `train_lora.sh`, workflow JSON
(подключение модуля — отдельные задачи). Guardrail в коде не отменяет
человеческую модерацию постов перед публикацией.
