# Feature Specification: Group photos (multi-model composite)

- Feature: 007-group-photos
- Status: Active

## Product outcome

`scripts/queue_workflow.py group` генерирует одно составное фото из 1..5
зарегистрированных моделей: каждый член генерируется отдельно через
`flux_lora` (LoRA и guardrail-фраза возраста — свои на члена), результаты
собираются PIL-композитом (layouts `row`/`grid2`/`grid3`), и готовая папка
поста пишется в per-model layout первичной (первой в списке) модели —
`models/<primary>/posts/` или `models/<primary>/posts-private/` (режимы
005): `photo.jpg`, `caption.txt`, `prompt.txt` и `group.json` (метаданные
группы).

## User scenarios

1. Оператор запускает
   `python scripts/queue_workflow.py group --models valery23,anna --prompt "…"
   --caption "…"` — члены генерируются строго последовательно (seed члена =
   `--seed + i`, где i — 0-based индекс члена), композит сохраняется как
   `photo.jpg` в папке первичной модели.
2. Оператор выбирает layout: `--layout row` (1 колонка), `grid2` (2
   колонки), `grid3` (3 колонки). Кадры не-первого члена с иным размером
   нормализуются scale-to-cover + center-crop до размера первого члена;
   незаполненные тайлы остаются заливкой (230, 230, 230).
3. Сбой на любом шаге (генерация члена, композит, запись файлов) откатывает
   запуск: папка назначения удаляется только если создана этим запуском,
   transit-staging — только пока он пуст; исключение пробрасывается дальше.
4. Pillow отсутствует в окружении — команда падает до очереди с ошибкой
   «Pillow required for group composite; rebuild manager image (see
   management/requirements.txt)» (ленивый импорт).

## Requirements

- Парсер `group`: `--models` (CSV, required; 1..5 членов без дубликатов,
  иначе `ValueError`), `--prompt` (required), `--layout
  {row,grid2,grid3}` (default `row`), `--mode {public,private}` (default
  `public`), `--caption` (required), `--name` (опц.), `--seed` int
  (default 27191), `--lora-strength` float (default 0.8), `--negative`
  (default `""`); `--url` — общий корневой флаг (как у других подкоманд).
- Pre-flight до очереди и до создания staging: валидация members; ленивый
  `from PIL import Image` (ImportError → `RuntimeError` с текстом выше);
  для каждого члена: запись реестра (иначе `RuntimeError`),
  `character.json` (иначе `FileNotFoundError`), `prompt_profile.json`
  (`{}` если нет), lora из реестра либо `<id>.safetensors`, проверка файла
  `runtime/models/loras/<lora>` (иначе `FileNotFoundError` — как
  generate_post).
- Staging: `runtime/output/group/<batch>` (batch = `%Y%m%d-%H%M%S`);
  транзитный `filename_prefix` члена в ComfyUI — `group/<batch>/<member_id>`;
  результат переносится в staging как `{i:02d}_{member_id}{ext}`.
- Генерация члена i: `positive = guardrail_positive(--prompt, возраст
  члена)`; `negative = build_negative(<profile.negative> + (--negative),
  mode="post")`; workflow `flux_lora`: node 6 = positive, node 31 seed =
  `--seed + i`, node 40 = lora/strength, node 9 = транзитный префикс;
  `set_negative`.
- Композит (PIL): cell = размер первого члена; columns = {row:1, grid2:2,
  grid3:3}; rows = ceil(N/columns); canvas RGB с заливкой (230, 230, 230).
- Финал: primary = members[0]; name = `--name` или timestamp
  (`%Y-%m-%d-%H%M%S`); путь в `--name` запрещён (`ValueError`);
  destination = `models/<primary>/(posts|posts-private)/<name>`
  (`exist_ok=False`); файлы: `photo.jpg` (JPEG quality=92), `caption.txt`
  (`--caption.strip() + "\n"`), `prompt.txt` (positive primary + `"\n"` +
  для каждого члена `"--- <id> ---\n<positive_i>\n"`), `group.json`
  (`{"members", "layout", "mode", "seeds": [--seed+i for i in range(N)],
  "loras": {id: lora}, "caption"}`); при успехе `print(destination)`.
- Rollback вокруг генерации/композита/финала: при любом исключении удалить
  destination (только созданный этим запуском) и staging (пока пуст) и
  re-raise.
- `Pillow>=10,<13` в `management/requirements.txt` (пересборка образа
  manager); импорт PIL — только внутри group-кода: `py_compile` и `--help`
  работают без Pillow.
- Подкоманды candidates/expand/post и `main()`-резолв `--model` для них —
  без изменений.

## Acceptance criteria

- [ ] `python -m py_compile scripts/queue_workflow.py` проходит; также
      проходят `py_compile lora/caption.py management/main.py`; корневой
      `--help` показывает `group` рядом с candidates/expand/post.
- [ ] `group --help` показывает все флаги; `--layout`/`--mode` имеют choices
      row|grid2|grid3 и public|private; defaults row/public.
- [ ] >5 членов или дубликаты в `--models` → `ValueError` до очереди и до
      создания staging; неизвестный id → `RuntimeError`; отсутствующий
      LoRA-файл → `FileNotFoundError` (все — fail-fast, без записи в
      `runtime/`).
- [ ] Fixture-прогон композита (импорт реальных `_cover_crop`/
      `_compose_grid`): row из 2 членов разных размеров → размер
      (cell_w, 2·cell_h); grid2 из 3 членов → (2·cell_w, 2·cell_h) с пустым
      тайлом (230,230,230); grid3 из 2 членов → (3·cell_w, cell_h).
- [ ] Pillow отсутствует → `RuntimeError` с текстом из
      management/requirements.txt до любых сетевых вызовов.

## NFR

- NFR-1: единственная новая зависимость — Pillow, импорт ленивый —
  evidence: `from PIL import Image` только внутри `generate_group`/
  `_compose_grid`; строка в `management/requirements.txt`.
- NFR-2: fail-fast pre-flight не создаёт staging/destination и не делает
  сетевых вызовов — evidence: ошибки валидации воспроизводятся без
  `runtime/output/group/` и без ComfyUI.
- NFR-3: rollback не затрагивает чужие данные — evidence: destination
  удаляется только после успешного `mkdir` этого запуска; staging — только
  если пуст.

## Compatibility and boundaries

Не меняет подкоманды candidates/expand/post, workflows JSON,
`lora/caption.py`. Транзит — только
`runtime/output/group/`; destination — существующие gitignored папки
`models/<id>/posts|posts-private/` (002, 005). Не трогает `dataset/`,
`reference/`, `runtime/` вне транзита group. Все члены — взрослые
вымышленные модели; guardrail 003 применяется к каждому члену (фраза по его
возрасту).

Manager-контур (реализовано в Фазе 3, вне изначального scope этой spec —
см. tasks.md T009/T010): `POST /api/jobs/group` с чистой валидацией
`_validate_group_payload` и вкладка «Групповой кадр» в UI.
