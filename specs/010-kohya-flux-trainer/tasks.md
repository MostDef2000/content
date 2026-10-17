# Tasks: kohya sd-scripts FLUX LoRA trainer (W3)

- Specification: specs/010-kohya-flux-trainer/spec.md
- Plan: specs/010-kohya-flux-trainer/plan.md
- Source: SYSADMIN-HANDOFF №35R + решение владельца (ai-toolkit OOM → kohya).

## Delivery tasks

- [x] T001 compose: одна строка в manager volumes после ai-toolkit-маунта —
      `/opt/sea-speed-worker/valery/kohya:/workspace/kohya`. DoD:
      `yaml.safe_load(compose.yaml)`; diff compose.yaml = 1 строка.
- [x] T002 Шаблоны: `lora/train_config_flux_kohya.template.toml` (аргументы
      скрипта: fp8_base, blocks_to_swap=20, cache_text_encoder_outputs_to_disk,
      cache_latents_to_disk, network_module networks.lora_flux, dim 32/alpha 16,
      timestep_sampling shift, discrete_flow_shift 3.1582, guidance 1.0,
      AdamW8bit, lr 1e-4 constant, bf16, max_train_steps {{STEPS}}) и
      `train_config_flux_kohya_dataset.template.toml`
      ([general]/[[datasets]]/[[datasets.subsets]], {{DATASET_DIR}},
      {{RESOLUTION}}, {{BATCH}}). Числовые плейсхолдеры закавычены — шаблоны
      валидны для tomllib. DoD: tomllib на оба; в шаблоне присутствуют все
      экономные флаги из контракта.
- [x] T003 `scripts/train_lora.sh` — ветка `arch=flux` на kohya: guard
      `flux_train_network.py` → exit 2; preflight (числовые env `^[0-9]+$`,
      резолв 4 локальных моделей по раскладке download_models.sh → exit 2 с
      подсказкой); рендер двух TOML (sed, числа без кавычек);
      SKIP_TRAIN-шов на месте (после рендера, до тренировки); бутстрап venv
      (venv python → torch-cu130 (D-005) → requirements, порядок в комментарии);
      `export HF_HOME`; `accelerate launch --num_processes 1
      --num_machines 1 --mixed_precision bf16 flux_train_network.py
      --config_file <f1> --dataset_config <f2>`; деплой tmp+mv →
      `runtime/models/loras/<id>-flux.safetensors` (дефис, chmod 644, лог
      `LoRA ready (flux)`), пустой выход → rc 3. SDXL-ветка и
      `copy_lora_output` не тронуты. DoD: `bash -n`; fixture-прогоны
      (SKIP_TRAIN-рендер без плейсхолдерных остатков; 3 негативных exit 2;
      sdxl SKIP_TRAIN без изменений).
- [x] T004 Спека 010 (этот комплект): решения оркестратора #1-#3
      зафиксированы, лимиты и DoD дыма 20 шагов записаны.
- [x] T005 (test-воркер): tests/test_train_lora_flux.py — паттерн
      изоляции WORKSPACE tmpdir (как test_clear_dataset.py): рендер обоих
      TOML, guard'ы exit 2, deploy tmp+mv в слот с дефисом; мёртвый
      train_config.flux.template.yaml удалён.
- [x] T007 Compat-шим sd-scripts (прод-крэш, spec 010 follow-up): в
      Comfy-Org flux1-dev-fp8 безпрефиксный маркер `scaled_fp8` стоит рано
      в порядке ключей (НЕ первым — keys[0] префиксован, прод-доказательство:
      крэш meta-тензора на unet.to("cpu"), а не ранний ValueError от max()
      в analyze_checkpoint_state) — strip-цикл ComfyUI-префикса в
      `library/flux_utils.py load_flow_model` (upstream @690ea7f9) ломается
      на нём посреди dict, и ОСТАВШИЕСЯ префиксованные ключи остаются
      непереименованными → мета-тензоры → крэш на unet.to("cpu"); scaled-fp8
      T5 сверх того несёт per-tensor `.scale_weight`, которые kohya
      игнорирует. Решение: `scripts/kohya_compat_shim.py` (strip на каждый
      ключ без break; деquant пар `.scale_weight` → bf16 через fp32, дроп
      маркера, каст «голого» fp8; `patch_flux_utils` оборачивает ОБА
      лоадера — `load_flow_model` И `load_t5xxl` (T5 грузится после возврата
      load_flow_model) — подменяя `load_safetensors` на время вызова
      настоящего лоадера с restore в finally; idempotent
      `sys.meta_path`-finder через `install()`; torch импортируется
      лениво) — бутстрап копирует шим в site-packages venv и пишет
      `sitecustomize.py` (import + install(), `grep -q`-гвард по строке
      активации против дублей и против легаси import-only файла) после
      rebuild-гварда, на каждом flux-запуске. DoD: структурные
      AST-пины + fixture-прогоны установки в tests/test_kohya_compat_shim.py
      и tests/test_train_lora_flux.py (178 passed, 15 skipped — torch-gated
      fp8-математика/behavioral-пины без torch в CI; приёмка fp8-пути —
      прод-дым T006).
- [ ] T006 Дымовой прогон 20 шагов на проде (старт — владелец; см. spec DoD).

## Acceptance criteria

- AC-1 `train_lora.sh <id>` без arch = sdxl, поведение байт-в-байт прежнее
  (fixture SKIP_TRAIN-прогон; в git diff нет строк SDXL-ветки).
- AC-2 `train_lora.sh <id> flux` рендерит оба TOML без остатков
  плейсхолдеров; числа — голые int; 4 модели — абсолютные пути общего
  runtime-маунта; экономные флаги присутствуют.
- AC-3 Отсутствие kohya/файла модели/нечисловой override → clean exit 2 с
  actionable-сообщением, без записей в runtime.
- AC-4 SKIP_TRAIN=1: только рендер, exit 0, бутстрап/запуск не выполняются.
- AC-5 Деплой: newest *.safetensors из `<id>_flux_v1` → атомарно (tmp+mv) в
  `runtime/models/loras/<id>-flux.safetensors`.
