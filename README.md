# Instagram Model — локальная генерация на RTX 5070 12GB

Генерация постоянной взрослой AI-модели (23 года, вымышленный персонаж) через **ComfyUI + FLUX.1-dev FP8 + LoRA**, управляемая через веб-UI. Видео не генерируется на этом этапе. Все модели работают **строго по очереди** — на 12 ГБ нельзя держать две тяжёлые модели одновременно.

Вывод — папки `posts/YYYY-MM-DD/photo.jpg + caption.txt` для ручной публикации. Никаких подключений к Instagram/Facebook.

> **Статус на сервере:** это стек — **дополнительная опция** поверх базового YOLO-воркера. Он использует тот же GPU, поэтому тяжёлые задачи (генерация FLUX / обучение LoRA) сериализуются через busy-лок в `manager`. Во время обучения или тяжёлой генерации YOLO-воркер желательно приостановить, чтобы не конкурировать за VRAM.

## Архитектура

```
[Браузер] ──HTTPS+basicauth──> [РУ 82.146.37.153: nginx (3d.mostdef.ru)] ──ZeroTier──> [GPU 10.123.239.102: manager :8000]
                                                                                  │ docker.sock / compose
                                                                                  ▼
                                                                          ComfyUI :8188 (flux1-dev-fp8, kontext)
                                                                          ai-toolkit venv (обучение LoRA)
```

- **GPU-сервер** (RTX 5070 12 ГБ, драйвер 595.91, CUDA 13.2, ZeroTier 10.123.239.102): ComfyUI + `manager` (FastAPI) + ai-toolkit.
- **РУ-хост** (82.146.37.153, ZeroTier 10.123.239.101): nginx vhost с TLS (Let's Encrypt) + `basicauth`, reverse-proxy на `10.123.239.102:8000`. Caddy установлен, но отключён (конфликт 80/443 с nginx) — `deploy/Caddyfile` хранится как референс.
- Публично торчит только nginx. `8188` и `8000` закрыты фаерволом (8000 разрешён только с РУ-хоста по ZeroTier).

## Требования

- GPU-сервер: Docker + Compose v2, `nvidia-container-toolkit`, ZeroTier-доступен VPS.
- РУ-хост: nginx (TLS + basicauth), домен `3d.mostdef.ru` с A-записью на 82.146.37.153.
- ~40 ГБ диска под веса+venv (data-том `/opt/sea-speed-worker`, 623 ГБ свободно).
- Принять лицензию https://huggingface.co/black-forest-labs/FLUX.1-dev и токен https://huggingface.co/settings/tokens.

## Быстрый старт (GPU-сервер)

> ⚠️ Проект разворачивать на **data-диске** `/opt/sea-speed-worker` (623 ГБ свободно), а не на системном (68 ГБ). Иначе веса ~35 ГБ + датасет + чекпоинты LoRA быстро заполнят системный диск. `sda1` (NTFS) не используется.

```bash
# если ещё не клонирован:
git clone git@github.com:MostDef2000/content.git /opt/sea-speed-worker/valery
cd /opt/sea-speed-worker/valery
cp .env.example .env
# впиши HF_TOKEN=hf_xxx в .env (нужен только для скачивания gated весов)

bash scripts/prepare_server.sh
bash scripts/download_models.sh      # flux1-dev-fp8, kontext fp8, энкодеры, VAE
docker compose up -d                 # поднимает comfyui + manager
```

Открой UI: `https://3d.mostdef.ru` (логин/пароль из `basicauth`).

### Поток в UI
1. **Кандидаты** — сгенерировать 8 эталонных лиц (`reference/candidates/`).
2. Выбрать один файл в выпадающем списке **Датасет** → **Расширить датасет** (Kontext-вариации).
3. Дописать подписи: `python lora/caption.py` (затем вручную отредактировать `dataset/images/*.txt`).
4. **Обучить LoRA** (ai-toolkit, 12 ГБ, долго) → скопировать `models/<id>/lora/output/.../*.safetensors` в `runtime/models/loras/<id>.safetensors` (для `valery23` — `valery23.safetensors`).
5. **Сгенерировать пост** → папка `posts/<name>/photo.jpg + caption.txt`.

### Или из консоли (без UI)
```bash
python scripts/queue_workflow.py candidates --count 8
python scripts/queue_workflow.py expand --reference reference/candidates/valery-03.jpg --count 8
python lora/caption.py
bash scripts/install_trainer.sh
bash scripts/train_lora.sh
python scripts/queue_workflow.py post --prompt "..." --caption "..." --name 2026-09-12-gym
```

## Папки

Per-model layout (фаза 1 редизайна — specs/002, specs/003):

- `management/` — управляющее приложение (FastAPI + HTML)
- `comfy/` — workflow JSON (bootstrap, kontext_variation, flux_lora)
- `models/registry.json` — реестр моделей (id, имя, возраст, теги, активная модель, lora-файл)
- `models/library.json` — библиотека сцен промтов (expand / candidates / post)
- `models/<id>/` — данные конкретной модели:
  - `character.json`, `prompt_profile.json` — tracked-конфигурация
  - `reference/candidates/` — кандидаты лица
  - `dataset/images/` — обучающий датасет (jpg + txt)
  - `lora/output/` — чекпоинты LoRA
  - `posts/`, `posts-private/` — готовые папки для ручной выкладки
- `runtime/models/loras/<id>.safetensors` — LoRA-копия модели для ComfyUI
- `runtime/input`, `runtime/output` — монтируются в ComfyUI
- `runtime/trash/` — удалённые модели (мягкое удаление, восстановимо)
- `runtime/backup/` — бэкапы миграции
- `deploy/Caddyfile` — референс конфига (живой фронт на РУ — nginx, см. AUDIT.md §10)

Разовая миграция со старого single-model layout (корневые `reference/`,
`dataset/`, `posts/`): `python scripts/migrate_to_registry.py` — идемпотентна,
делает бэкап в `runtime/backup/`, повторный запуск безопасен.

## Безопасность

- Только Caddy с `basicauth`+HTTPS снаружи; приложение не публикуется напрямую.
- `manager` монтирует `/var/run/docker.sock` — запускать от выделенного пользователя, не давать сокет никому кроме `manager`.
- Никаких токенов/паролей в репозитории; `.env` в `.gitignore`.
- Персонаж только взрослый вымышленный, без реальных людей и несовершеннолетних.
- busy-лок защищает лимит 12 ГБ (одна тяжёлая задача за раз).

## Проверка

```bash
curl localhost:8188/system_stats     # ComfyUI
curl localhost:8000/api/status        # manager
# с РУ-хоста: curl -u user:pass https://3d.mostdef.ru/api/status
# прямой http://GPU_IP:8000 и :8188 из интернета — должны быть закрыты
```

## Остановка

```bash
docker compose down
```
