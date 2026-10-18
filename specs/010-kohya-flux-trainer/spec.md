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
- установка compat-шима (D-006) — на каждом flux-запуске, ПОСЛЕ
  rebuild-гварда (в т.ч. при здоровом venv) и до запуска тренера:
  безусловный `cp` в site-packages + `sitecustomize.py` (import + install(),
  grep-гвард по строке активации).
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
  оба опции (решение оркестратора #2). Перед запуском — TE-cache-pass
  (D-007), сам запуск несёт `KOHYA_SKIP_TEXT_ENCODERS=1` (env-prefix).
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
- (D-006) ComfyUI scaled-fp8 чекпойнты грузятся через compat-шим
  (`scripts/kohya_compat_shim.py`, копируется на каждом flux-запуске (после
  rebuild-гварда),
  автозагрузка через `sitecustomize.py`: import + install()): upstream
  strip-цикл префикса `model.diffusion_model.` ломается на БЕЗпрефиксном
  ключе посреди dict — маркер `scaled_fp8` стоит рано в порядке ключей
  Comfy-Org flux1-dev-fp8, но НЕ первым (keys[0] префиксован — прод-доказательство:
  крэш «Cannot copy out of meta tensor» на unet.to("cpu") (прод-кейс
  train-1792241951), а не ранний ValueError от max() в
  analyze_checkpoint_state, который случился бы при непереименованном
  ПЕРВОМ ключе): цикл обрывает переименование, и ВСЕ ОСТАВШИЕСЯ
  префиксованные ключи остаются непереименованными → unet остаётся meta.
  T5 `t5xxl_fp8_e4m3fn_scaled` несёт per-тензоровые `.scale_weight`,
  которые kohya игнорирует (мусорные TE-выходы). Шим чинит оба: снимает
  префикс со всех ключей и обрабатывает fp8-скейлы до assign; оборачивает
  ОБА лоадера с РАЗНЫМИ целями — `load_flow_model`: ре-квантование в
  СОБСТВЕННЫЙ fp8-dtype чекпоинта — корректно-округлённая ОДНОРАЗОВАЯ
  квантизация истинных значений, т.е. тот же end-state, что у
  PLAIN-fp8 fp8_base-флоу kohya для сырого fp8-чекпоинта (НЕ бит-в-бит со
  старым путём bf16-материализация → kohya-каст: тот был двойным
  округлением e4m3(bf16(x)) против одноразового e4m3(x) здесь — расхождение
  ≤1 fp8 ulp только в double-rounding ничьих, ~1/32 элементов нормальной
  бинады — оценка из анализа ничьих округления, не измеренное свойство
  этих весов, а тот старый путь ни одного прогона не завершил); kohya-каст в
  flux_train_network.py:129, если вообще выполняется, касается только
  не-fp8 остатка (same-dtype Tensor.to() аллокаций не делает); `load_t5xxl`:
  явный bf16
  (качество TE-кэша — fp8-T5 запёк бы шум fp8-квантования в эмбеддинги
  каптионов; T5 грузится после возврата load_flow_model, тот же
  swap-паттерн с restore в finally). Бюджет host-RAM (ref:
  train-1792246674 и train-1792249442 — оба тихие смерти по сигнатуре
  OOM/SIGKILL, диагноз выводной и не измерен; фикс — снижение давления
  памяти, подтверждение только по T006): kill-line по факту SIGKILL
  < ~26 ГБ (старая джоба жила при ~18 ГБ); после шима пики — оценки:
  фаза flow ≈ размер DiT-sd (1:1 fp8→fp8 с прогрессивным освобождением),
  фаза T5 = DiT-sd + 1.7 (CLIP) + 9.4 (T5-bf16) — цифры остаются
  оценками до замера: для 17-ГБ-DiT сценария ≈ ~28.1 ГБ, для
  12-ГБ-DiT ≈ ~23.1 ГБ (kill-line < ~26 ГБ выведена из двух SIGKILL;
  какой сценарий реален, решает preflight-замер, а не фаза смерти).
  Preflight в train_lora.sh меряет размер чекпойнта (из заголовка
  safetensors) и доступную память ДО любой аллокации и срывает запуск
  ненулевым кодом до старта — чистый abort префлайта заменяет SIGKILL;
  гейт энфорсит ЭФФЕКТИВНЫЙ бюджет min(MemAvailable, cgroup_limit −
  cgroup_usage) с fail-closed при finite-лимите с нечитаемым usage
  (все значения в логе; шов корня — MEM_PREFLIGHT_CGROUP_ROOT);
  если замер покажет нехватку — раунд 4: файловый repack в чистый
  fp8, чанковый, пик ~12.5 ГБ (оценка). Приёмка T006: в логе допустима ЛЮБАЯ из строк
  «Loaded fp8 FLUX model» или «Cast FLUX model to fp8» — критерий
  прохождения отсутствие SIGKILL, не конкретная строка; в логе ДОЛЖНЫ
  присутствовать значения PREFLIGHT (префлайт до аллокаций; шов
  MEM_PREFLIGHT_SLACK_BYTES — для тестовых прогонов), и abort префлайта —
   УСПЕШНЫЙ исход гейта, а не провал трейнера. Если фаза T5
   тоже не влезет — следующий шаг T5-в-fp8 с трейдоффом качества TE
   (решение владельца); этот рычаг также требует снизить
   MEM_PREFLIGHT_SLACK_BYTES (дефолт 13 ГБ кодировал T5-bf16-фазу —
   co-residency, устранённую D-007; теперь дефолт 3 ГБ, см. D-007).
- (D-007) Двухпроходная TE-cache-архитектура памяти (Valery W3): проход 1 —
  standalone-писатель кэша `tools/cache_text_encoder_outputs.py` (venv
  python, НЕ accelerate launch; flux-ветка грузит ТОЛЬКО clip_l + t5xxl,
  без DiT/VAE) запускается train_lora.sh ПЕРЕД тренером с явными флагами,
  зеркалящими отрендеренный TOML (`--flux`,
  `--pretrained_model_name_or_path <DiT>` — state-analysis выбирает dev →
  длина T5-токенов 512, тот же дефолт, что у тренера; `--clip_l`,
  `--t5xxl`, `--dataset_config <rendered dataset TOML>`,
  `--cache_text_encoder_outputs_to_disk`, `--mixed_precision bf16`;
  t5xxl_max_token_length/apply_t5_attn_mask/text_encoder_batch_size НЕ
  передаются, т.к. TOML их не ставит — отсутствующие значения идентичны в
  обоих проходах; `--config_file` не используется — конкретный блокер:
  отрендеренный TOML несёт network-* ключи (network_module, network_dim,
  network_alpha, network_train_unet_only), а их регистрирует train_network.py,
  НЕ семейство add_training_arguments у tools/-писателя кэша — его парсер эти
  ключи не знает. Идентичность кэша между проходами — конструктивная:
  npz-путь в апстриме выводится из АБСОЛЮТНОГО пути картинки
  (strategy_flux.get_outputs_npz_path = splitext(image_abs_path)+суффикс),
  а не из output_dir/cwd; оба прохода читают один и тот же отрендеренный
  dataset-TOML → места кэша совпадают по конструкции; is_partial=False
  захардкожен в standalone (:170) и фактически False у тренера
  (train_clip_l/train_t5xxl False при unet-only) — именование и содержимое
  кэша согласованы). Проход 2 — тренер без TE:
  upstream flux_train_network.py грузит CLIP-L (:141) и T5XXL (:153)
  БЕЗУСЛОВНО даже при полном дисковом кэше, а с network_train_unet_only
  LoRA get_models_for_text_encoding (:206-213) возвращает None — TE никогда
  не кодируют через себя и являются мёртвым грузом ~11 ГБ RAM. Запуск
  тренера несёт `KOHYA_SKIP_TEXT_ENCODERS=1` (env-prefix, только на
  инвокацию тренера, никогда на cache-pass): шим по НЕпустому значению
  переменной возвращает вместо загрузки заглушки из `load_clip_l`/
  `load_t5xxl` — контракт стаба: настоящий `.dtype` (torch.bfloat16 — его
  читает fp8-проверка на :157-160), no-op `eval()/to()/train()` с
  возвратом self (поверхность :144/:154, :246-247/:284-286), а любой
  `encode`/`forward` поднимает RuntimeError с литералом «text encoder
  cache» — громкий срыв при неполном кэше (заглушка, «молча кодирующая»
  мусор, запрещена контрактом). Путь DiT (`load_flow_model`) под гейтом
  остаётся живым (шим-фиксапы применяются как раньше). Порядок во flux-ветке:
  рендер TOML → preflight (позиция неизменна) → venv bootstrap → установка
  шима → CACHE PASS (после bootstrap+шима, до тренера) → запуск тренера;
  ненулевой rc cache-pass обрывает ВЕСЬ запуск
  с маркером «TE-CACHE-PASS ... failed» — тренер никогда не стартует на
  неполном кэше (fail-fast на уровне скрипта поверх стаб-триппера шима).
  WHY слэк 13→3 ГБ: это КАЛИБРОВКА ДИЗАЙНА, а не ослабление гейта — старые
  13 ГБ (T5 9.4 + CLIP 1.7 + ~1.9 хедрум) кодировали co-residency T5-bf16-фазы
  с DiT в тренере, которой при двухпроходной схеме больше НЕ СУЩЕСТВУЕТ
  (TE никогда не в одном процессе с DiT); формула префлайта НЕ меняется
  (ckpt + slack; slack 3 ГиБ), пик cache-pass по HOST-RAM (~12.3 ГБ:
  T5-bf16 + CLIP, без DiT) доминируется той же формулой и далеко ниже
  kill-line (~26 ГБ) — риск SIGKILL на проходе 1 пренебрежим; отдельный
  режим отказа прохода 1 — VRAM OOM (~10.9 из 12 ГБ, впритык) — падает
  чистым rc cache-pass, а не SIGKILL; т.е. один гейт покрывает оба прохода.
  Пиковые цифры — оценки из анализа (Box: 23.37 ГБ avail; DiT 17.05 ГБ;
  тренер без TE ~19.4 ГБ CPU), приёмка — по T006.
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
