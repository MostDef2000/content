# Instagram Model — локальная генерация на RTX 5070 12GB

Генерация постоянной взрослой AI-модели (23 года, вымышленный персонаж) через **ComfyUI + FLUX.1-dev FP8 + LoRA**, управляемая через веб-UI. Видео не генерируется на этом этапе. Все модели работают **строго по очереди** — на 12 ГБ нельзя держать две тяжёлые модели одновременно.

Вывод — папки `posts/YYYY-MM-DD/photo.jpg + caption.txt` для ручной публикации. Никаких подключений к Instagram/Facebook.

> **Статус на сервере:** это стек — **дополнительная опция** поверх базового YOLO-воркера. Он использует тот же GPU, поэтому тяжёлые задачи (генерация FLUX / обучение LoRA) сериализуются через busy-лок в `manager`. Во время обучения или тяжёлой генерации YOLO-воркер желательно приостановить, чтобы не конкурировать за VRAM.

## Архитектура

```
[Браузер] ──HTTPS+basicauth──> [LIFESIM: Caddy (3d.mostdef.ru, 135.106.195.18)] ──ZeroTier──> [GPU-ХОСТ: manager :8000]
                                                                                  │ docker.sock / compose
                                                                                  ▼
                                                                          ComfyUI :8188 (flux1-dev-fp8, kontext)
                                                                          ai-toolkit venv (обучение LoRA)
```

- **GPU-сервер** (RTX 5070 12 ГБ, драйвер 595.91, CUDA 13.2, ZeroTier `10.123.239.102`, за NAT): ComfyUI + `manager` (FastAPI) + ai-toolkit.
- **lifesim-хост** (публичный, `135.106.195.18`): Caddy vhost с TLS (Let's Encrypt, сертификат выпущен 2026-10-22, действует до 2027-01-20; HTTP/2 + HSTS) + `basicauth`, reverse-proxy на `GPU-ХОСТ:8000` по ZeroTier.
- Публично торчит только Caddy на lifesim. `8188` и `8000` наружу закрыты (GPU за NAT; `8000` доступен только по ZeroTier — с lifesim).

## Требования

- GPU-сервер: Docker + Compose v2, `nvidia-container-toolkit`, ZeroTier-доступен lifesim.
- lifesim-хост: Caddy (TLS + basicauth), домен `3d.mostdef.ru` с A-записью на публичный IP lifesim (`135.106.195.18`).
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
4. **Обучить LoRA** (ai-toolkit, 12 ГБ, долго) → скопировать `models/<id>/lora/output/.../*.safetensors` в `runtime/models/loras/<id>.safetensors` (FLUX-слот — `<id>-flux.safetensors`).
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
- `deploy/Caddyfile` — референс конфига (живой фронт — Caddy на lifesim, см. AUDIT.md, аудит 2026-10-22)

Разовая миграция со старого single-model layout (корневые `reference/`,
`dataset/`, `posts/`): `python scripts/migrate_to_registry.py` — идемпотентна,
делает бэкап в `runtime/backup/`, повторный запуск безопасен.

## Безопасность

- Только Caddy на lifesim с `basicauth`+HTTPS снаружи; приложение не публикуется напрямую.
- `manager` монтирует `/var/run/docker.sock` — запускать от выделенного пользователя, не давать сокет никому кроме `manager`.
- Никаких токенов/паролей в репозитории; `.env` в `.gitignore`.
- Персонаж только взрослый вымышленный, без реальных людей и несовершеннолетних.
- busy-лок защищает лимит 12 ГБ (одна тяжёлая задача за раз).

## Проверка

```bash
curl localhost:8188/system_stats     # ComfyUI
curl localhost:8000/api/status        # manager
curl https://3d.mostdef.ru/api/status # публично: без auth → 401, с auth → 200 (comfy_up: true)
# прямой http://GPU_IP:8000 и :8188 из интернета — должны быть закрыты
```

## Остановка

```bash
docker compose down
```

## Deploy (GitHub Actions → GPU-бокс)

> ⚠️ **DEPLOY ROUTE STALE — requires re-wiring to lifesim (separate task, pending).**
> Транспорт текущего `.github/workflows/deploy.yml` — SSH с runner на РУ-хост
> (82.146.37.153) и дальше ProxyJump на GPU-бокс (ZeroTier 10.123.239.102) —
> мёртв вместе с РУ-хостом: инцидент 2026-10-17, хост уничтожен провайдером
> 2026-10-22. До перепайки dispatch-деплой не работает; `deploy.yml` намеренно
> не переписывался. Описание воркфлоу ниже сохранено как исторический референс
> для задачи перепайки.

Историческое описание (транспорт через РУ): обновление работающего стека на
GPU-сервере — dispatch-воркфлоу `.github/workflows/deploy.yml` по проверенному
паттерну sea-speed (deploy-ubuntu-worker.yml). Транспорт: GitHub-hosted runner
не видит ZeroTier, поэтому SSH с runner на РУ-хост и дальше ProxyJump на
GPU-бокс. Хост-ключи закреплены (pinned) в repo variables, `ssh-keyscan` не
используется.

Поток:

1. Изменения смёржены в `main`; после подтверждения в чате (человеческий гейт
   — владелец, 2026-10-10) владелец запускает Actions → `deploy` с входом
   `deploy_sha` — точный 40-hex SHA из `main`.
2. Job `gate` fail-closed проверяет, что SHA — предок `origin/main`
   (`git fetch` + `git merge-base --is-ancestor`). Доверенного CI-сигнала пока
   нет (`.github/workflows/ci.yml` только валидирует compose и скрипты); когда
   ci.yml обрастёт покрытием, в gate добавляется проверка trusted-прогона —
   как в stroy.
3. Job `deploy` (environment `production`, concurrency `content-production`)
   настраивает SSH и выполняет на боксе единственную принудительную команду:
   `content-deploy-v1 <sha>` (forced command у deploy-пользователя).
4. Бокс сам подтягивает исходники (`git fetch` + checkout точного SHA) и
   собирает из них образы (`docker compose build comfyui manager` → `up -d`)
   — без артфакт-конверта, та же pull-of-source модель, что у sea-speed.
   Затем health-проверка ~120 с: ComfyUI `http://127.0.0.1:8188/` и manager
   `http://127.0.0.1:8000/` (порт из `.env`, tolerant: 200/302/404 = жив).
   Успех = маркер `DEPLOYMENT_ACCEPTED sha=... previous=...` в транскрипте;
   транскрипт сохраняется артефактом `content-deploy-transcript-<sha>`.

Rollback: при любой ошибке после записи предыдущего SHA воркфлоу-скрипт
автоматически возвращается на него (checkout + `docker compose build` +
`up -d`) и печатает `DEPLOY_ROLLED_BACK`. Откат пересобирает образы и поэтому
медленнее двухслотового swap у sea-speed; кэш сборки обычно делает пересборку
быстрой.

Bootstrap (однократно, Lane B) — клон репо на бокс, `.env` с `HF_TOKEN`
(значение живёт только на боксе), deploy-пользователь с forced command,
список GitHub secrets/variables — в шапке `deploy/deploy-wrapper`.

GitHub Secrets/Vars (имена; значения не хранятся в репозитории):

- secrets: `CONTENT_VPS_SSH_PRIVATE_KEY`, `CONTENT_BOX_SSH_PRIVATE_KEY`,
  `CONTENT_VPS_HOST`, `CONTENT_VPS_USER`, `CONTENT_BOX_HOST`, `CONTENT_BOX_USER`
- vars: `CONTENT_VPS_SSH_PORT` (опц., default 22), `CONTENT_VPS_KNOWN_HOSTS`,
  `CONTENT_BOX_KNOWN_HOSTS` (pinned host keys)

Риски (Lane C — ревью отдельно, сейчас не меняются): manager-порт публикуется
на всех интерфейсах (`${MANAGER_PORT:-8000}:8000`), manager монтирует
`/var/run/docker.sock` (root-эквивалент), ufw на боксе неактивен — решения
владельца, зафиксированы в шапке `deploy/deploy-wrapper`.
