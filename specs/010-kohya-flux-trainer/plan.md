# Implementation Plan: kohya sd-scripts FLUX LoRA trainer (W3)

- Specification: specs/010-kohya-flux-trainer/spec.md
- Status: Active

## Architecture

- **Скрипт**: `scripts/train_lora.sh`, только ветка `arch=flux`. SDXL-ветка
  (`cd ai-toolkit` + `run.py` + `copy_lora_output`) сохранена байт-в-байт;
  для flux добавлены свои хелперы (`resolve_existing`, `require_flux_int`,
  `render_kohya_toml`, `deploy_flux_lora`), чтобы не трогать общий
  `copy_lora_output`, которым пользуется SDXL.
- **Два TOML-рендера** (решение оркестратора #2, сверено с исходником
  sd-scripts @ `690ea7f9`):
  - `library/args.py::read_config_from_file` **уплощает все секции**
    `--config_file`-TOML в argparse-namespace → имена ключей = имена CLI
    опций; числовые значения обязательны как int (строки ломают арифметику),
    поэтому в шаблонах они закавычены, а рендер снимает кавычки.
  - `train_network.py:908` берёт датасет только из `--dataset_config`;
    `ConfigSanitizer` валидирует его voluptuous-схемой с PREVENT_EXTRA
    («extra keys not allowed» — официальный пункт Error Guide), т.е.
    скриптовые ключи в датасет-файле запрещены. Один merged-файл не работает
    на обеих опциях → два шаблона + два рендера.
- **Датасет**: плоская папка картинок + .txt каптионы не совпадает с
  DreamBooth-конвенцией «parent dir с N_<class> подпапками», поэтому
  `[[datasets.subsets]] image_dir = <dataset/images>` через
  `--dataset_config` (DreamBooth-метод с изображениями прямо в image_dir —
  как `folder_path` в ai-toolkit-шаблоне). `enable_bucket` + `bucket_no_upscale`.
- **Модели**: `flux_utils.load_*` делает только локальный `safe_open` —
  автозагрузки весов из HF нет. Четыре обязательных файла
  (`pretrained_model_name_or_path`, `ae`, `clip_l`, `t5xxl`) резолвятся из
  раскладки `scripts/download_models.sh` на общем runtime-маунте:
  `checkpoints/flux1-dev-fp8.safetensors` (BFL, fp8 грузится as-is при
  `fp8_base`), `vae/ae.safetensors`,
  `text_encoders/{clip_l.safetensors, t5xxl_fp8_e4m3fn_scaled.safetensors}`
  (fp8-e4m3fn T5 принят скриптом, `flux_train_network.py:157-160`).
  Отсутствие → clean exit 2 c подсказкой `scripts/download_models.sh`.
- **venv**: контейнерный python3.12; health-check
  `venv/bin/python -c "import torch, torchvision"` (покрывает и отсутствие
  venv, и полудохлый бутстрап); порядок инсталлов torch-cu130 (D-005) → requirements
  зафиксирован комментарием в коде (requirements не должен затирать
  CUDA-сборку torch).
- **HF**: `HF_HOME=/workspace/runtime/hf-cache` — общий кэш (хэндофф №35R);
  HF_TOKEN попадает в env существующим `set -a; source .env` блоком — новых
  секретных механизмов нет.
- **Именование слота**: дефис `<id>-flux.safetensors` — контракт
  `main.py:_lora_name_for_engine` (fallback реестрового `lora_flux`) и
  тестов `test_engine_validation.py:479` (решение оркестратора #1).
- **Тестируемость**: чистые функции с параметром ROOT
  (`deploy_flux_lora MODEL_ID [ROOT]`), FAIL-FAST-блоки до любых записей,
  SKIP_TRAIN-шов не смещён (рендер → шов → bootstrap/запуск); изоляция
  WORKSPACE через tmpdir — паттерн `test_clear_dataset.py`.

## Decisions

### D-001 — два TOML-рендера вместо одного конфига
Единственный файл `--config_file` не может нести датасет: `[[datasets]]`
уплощается в невостребованный атрибут namespace, а отдельный датасет-механизм
требует `--dataset_config` со строгой схемой. Разводим на два шаблона
(решение оркестратора #2). Рендер числовых плейсхолдеров через
`s/"{{STEPS}}"/${STEPS}/` — шаблон остаётся валидным TOML (tomllib), значения
после рендера — голые int.

### D-002 — локальные модели из раскладки download_models.sh
Работа ордера «T5/CLIP докачаются» неверна для sd-scripts: он не умеет
качать веса. Резолвим уже скачанные ComfyUI-копии (convention win) с
fail-fast и командой починки в сообщении; fp8-варианты совместимы с
`fp8_base` (строки 107-108 `flux_train_network.py`).

### D-003 — слот с дефисом
Ордер называл `<id>_flux.safetensors`, но оракул (`main.py:541`) и тесты
требуют дефис; прод-контракт не ломаем (решение оркестратора #1).

### D-004 — старый flux-yaml-шаблон удалён test-воркером
`lora/train_config.flux.template.yaml` больше не используется скриптом;
test-воркер (зона правки тестов) удалил файл вместе с
`test_train_config_flux_template_is_flux` и заменил контракт на новые TOML
(тесты доставлены в том же диффе, см. spec §Limits).

## Risks / mitigations

- Первый запуск медленный (venv 10–20 мин) → бутстрап идёт ПОСЛЕ SKIP_TRAIN-шва
  и логируется; повторные запуски — быстрые (health-check).
- fp8-чексуммы Comfy-Org «scaled»-вариантов — приняты sd-scripts (fp8 as-is).
- 12 ГБ с blockswap — шаги медленнее, но OOM-профиль снят (fp8 + swap 20 +
  оба дисковых кэша).
