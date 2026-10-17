# Feature Specification: kohya sd-scripts FLUX LoRA trainer (W3)

- Feature: 010-kohya-flux-trainer
- Status: Active
- Source: SYSADMIN-HANDOFF №35R (подтверждённые факты сервера) + решение
  владельца: ai-toolkit детерминированно OOM-ится на FLUX-трейне 12 ГБ —
  ветка `arch=flux` переводится на kohya sd-scripts с экономными режимами

## Product outcome

FLUX-LoRA-тренинг (`train_lora.sh <id> flux`) реально работает на карте
12 ГБ:

1. **Тренер — kohya sd-scripts** (`flux_train_network.py`, clone @ `690ea7f9`,
   bind-маунт `/opt/sea-speed-worker/valery/kohya:/workspace/kohya` в manager).
   ai-toolkit для FLUX больше не запускается; SDXL-ветка (ai-toolkit yaml)
   не меняется вообще.
2. **Экономные режимы 12 ГБ**: `fp8_base`, `blocks_to_swap=20`,
   `cache_text_encoder_outputs_to_disk`, `cache_latents_to_disk` (+ парные
   `cache_text_encoder_outputs`/`cache_latents`, которые sd-scripts включает
   автоматически). Профиль подтверждён в хэндоффе №35R.
3. **Контракт результата не меняется**: слот FLUX-LoRA —
    `runtime/models/loras/<id>-flux.safetensors` (ДЕФИС — контракт
   `management/main.py _lora_name_for_engine`, реестровый слот `lora_flux`;
   решение оркестратора от 2026-10-16: дефис, отступление от буквы ордера).
   Выход тренера: `models/<id>/lora/output/<id>_flux_v1/`.
4. **Первый запуск самодостаточен**: venv бутстрапится внутри контейнера
   (хостовый uv-venv не работает — symlink на хостовый интерпретатор);
   venv живёт на персистентном rw-маунте `runtime/kohya-venv` (вне
   хост-владельческого чекаута kohya) и самолечится: `rm -rf` +
   `python3 -m venv --clear` при проваленной probe-пробе
   (`import torch, torchvision`; прод-кейс train-1792131743 — Errno 2 на
   битом venv);
   4 файла базовых моделей резолвятся из общего ComfyUI-runtime
   (раскладка `scripts/download_models.sh`); HF-кэш — общий
   `/workspace/runtime/hf-cache`, токен — из `.env` существующим механизмом.

## User scenarios

1. Владелец жмёт «Train FLUX» в UI: manager запускает
   `bash scripts/train_lora.sh <id> flux` (команда с двумя аргументами —
   контракт тестов `test_engine_validation` не менялся); скрипт рендерит
   `runtime/train/<id>_flux_kohya.toml` (аргументы скрипта) и
   `runtime/train/<id>_flux_kohya_dataset.toml` (датасет), запускает
   `accelerate launch flux_train_network.py --config_file <f1>
   --dataset_config <f2>`, в конце деплоит свежий чекпойнт в FLUX-слот
   (tmp+mv, атомарно) и логирует `LoRA ready (flux): …`.
2. `train_lora.sh <id>` без arch — SDXL как раньше (ветка не тронута).
3. Окружение сломано (нет маунта kohya / нет файла модели / нечисловой
   `STEPS`) — clean exit 2 с actionable-сообщением, ничего не рендерится и
   не бутстрапится.
4. `SKIP_TRAIN=1` (тестовый шов, контракт прежний) — оба TOML рендерятся,
   бутстрап venv и запуск тренера не выполняются.

## Requirements

- guard: `${ROOT}/kohya/flux_train_network.py` отсутствует → exit 2
  (fail-fast, паттерн sdxl-гуарда).
- bootstrap venv (только flux, только если НЕ выполняется ВЕСЬ набор:
  `venv/bin/python -c "import torch, torchvision"` проходит И есть
  исполняемый `venv/bin/accelerate` И `.torch-flavor` в корне venv содержит
  текущий слаг индекса): `python3 -m venv` → cu130 torch/torchvision
  (D-005) → запись `.torch-flavor` → `requirements.txt` клона (порядок
  обязателен — requirements не должен переустанавливать CUDA-сборку torch).
- `export HF_HOME=${ROOT}/runtime/hf-cache` перед запуском; HF_TOKEN —
  из `.env` (существующий `set -a`-блок), в скрипте литералов нет.
- Рендер: sed, как в sdxl-ветке; строковые плейсхолдеры с кавычками,
  числовые (`{{STEPS}}/{{SEED}}/{{RESOLUTION}}/{{BATCH}}`) — снимаются
  кавычки (шаблоны остаются валидным TOML, проверяется tomllib);
  переопределения env валидируются `^[0-9]+$` (дефолты 1500/1/768/1).
- Запуск: `venv/bin/accelerate launch --num_processes 1 --num_machines 1
  --mixed_precision bf16 flux_train_network.py --config_file <f1>
  --dataset_config <f2>` — ДВА файла: sd-scripts flattens `--config_file`
  в argparse-namespace, а `--dataset_config` валидируется строгим
  voluptuous-схемой (extra keys запрещены), один файл не может обслужить
  оба опции (решение оркестратора #2).
- Модели — локальные single-file чекпоинты (автозагрузки с HF нет):
  `pretrained_model_name_or_path` (DiT, BFL-формат), `ae`, `clip_l`,
  `t5xxl` — резолв по раскладке `download_models.sh`
  (`runtime/models/checkpoints|vae|text_encoders`), отсутствующий файл →
  exit 2 с подсказкой команды. Решение оркестратора #3.
- Деплой: новый чекпойнт из `<output_dir>` → tmp+mv в
  `runtime/models/loras/<id>-flux.safetensors` (дефис, решение
  оркестратора #1), chmod 644, лог в stdout как у sdxl.
- compose.yaml: ровно одна добавленная строка маунта в manager volumes.

## Limits (лимиты и операционные запреты)

- Первый flux-запуск: бутстрап venv ~10–20 мин (torch cu130 ~2.5 ГБ).
- (D-005) torch-индекс — **cu130**, не cu124: GPU-бокс Blackwell (sm_120),
  cu124-колёса не содержат ядер sm_120 → первый conv2d умирает с
  "CUDA error: no kernel image is available" (прод-кейс train-1792158920).
  Guard бутстрапа требует маркер-файл `.torch-flavor` со слагом индекса
  (закрывает F1-остаток): venv без маркера или со старым слагом пересобирается
  сам — включая переход с уже установленного cu124 (прецедент: ai-toolkit
  venv на torch 2.13.0+cu130).
- `blocks_to_swap=20` замедляет шаги (своп блоков GPU↔CPU) — фоновая джоба.
- Полный трейн стартует владелец из UI; на время трейна ComfyUI останавливать
  (12 ГБ общие, hard rule 3 — busy-лок менеджера).
- Тесты доставлены test-воркером в этом же диффе: контракт cmd/SKIP_TRAIN
  `test_engine_validation` сохранён, flux-часть расширена (`tests/test_train_lora_flux.py`,
  15 тестов + 3 новых контракта шаблонов); старый `lora/train_config.flux.template.yaml`
  УДАЛЁН вместе с его тестом `test_train_config_flux_template_is_flux`
  (скрипт его больше не читает; удаление предусмотрено plan D-004).

## DoD

- Валидации: `bash -n`; tomllib на ОБА шаблона; `yaml.safe_load(compose.yaml)`;
  `git status --short` — только заявленные файлы; в diff нет строк
  SDXL-ветки; в скрипте нет секретов.
- Дымовой прогон на проде: 20 шагов (`STEPS=20 bash scripts/train_lora.sh
  <id> flux`) — чекпойнт появляется в FLUX-слоте, `LoRA ready (flux)` в
  логе джобы. (Старт — владелец.)
