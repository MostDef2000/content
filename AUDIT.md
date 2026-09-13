# Audit: GPU-сервер для локальной генерации AI-модели (Valery)

**Назначение:** сервер крутит генерацию фото (FLUX.1-dev FP8 + LoRA персонажа) и обучение LoRA. Управление — через веб-UI, опубликованный на VPS. GPU-сервер сам по себе **не публикуется**.

> Этот стек — **дополнительная опция** поверх базового YOLO-воркера. Он делит GPU; тяжёлые задачи сериализуются busy-локом в `manager`. Во время обучения/тяжёлой генерации YOLO желательно приостановить.

## 1. Железо и система (проверено сисадмин-агентом)
- GPU: **RTX 5070 (GB205), 12227 MiB, compute_cap 12.0 (sm_120)**.
- Драйвер: **595.91.07**, CUDA **13.2**.
- ОС: Ubuntu 26.04 (отмечено как «совместимая»).
- `nvidia-container-toolkit` **1.20.0** + `/etc/docker/daemon.json` с nvidia runtime (виден в `docker info`).
- Docker **29.8.0** + Compose **v5.5.1**.
- Диски: системный `/` **68.3 ГБ** свободно (97.9 ГБ всего); **data-том** `/opt/sea-speed-worker` **623.9 ГБ** свободно (737 ГБ, LVM `ubuntu--vg-sea--speed--worker`, ext4). `sda1` NTFS «Новый том» — не смонтирован, не используется (Windows-раздел).
- Сеть: **ZeroTier 10.123.239.102**, доступен VPS, публичного входа нет.
- GPU уже используется YOLO-воркером (~1.6 ГБ/12 ГБ) — подтверждает необходимость busy-лока.

## 2. Что запущено (на GPU-сервере)
| Сервис | Порт | Публикация | Назначение |
|---|---|---|---|
| `comfyui` (Compose) | 8188 | только localhost/compose-net | генерация FLUX (dev + kontext) |
| `manager` (FastAPI, наш код) | 8000 | только приватный IP → VPS Caddy | пульт управления |
| `ai-toolkit` (venv, вне compose) | — | нет | обучение LoRA, дёргается `scripts/train_lora.sh` |

Лимит **12 ГБ**: тяжёлые модели грузятся строго по очереди. `manager` ставит busy-лок — нельзя запустить генерацию и обучение одновременно.

## 3. Сетевая модель
```
Браузер ──HTTPS+basicauth──> VPS 77.105.142.206:Caddy (ui.example.com)
                                 │ reverse_proxy
                                 ▼ (ZeroTier)
                            GPU 10.123.239.102:manager :8000 ──docker.sock/compose──> comfyui :8188
```
- Публично торчит только Caddy на VPS.
- GPU-сервер: `8188` и `8000` закрыты извне фаерволом; `8000` разрешён только с IP VPS (ZeroTier).
- Никаких токенов/паролей Instagram в сети не ходит (публикация ручная).

## 4. Секреты
- `HF_TOKEN` (gated FLUX.1-dev) — только на этап скачивания, в `.env` на сервере (`chmod 600`), не коммитится. После `download_models.sh` может быть удалён.
- Репозиторий `MostDef2000/content` — **private**; `.gitignore` исключает `.env`, `runtime/`, `dataset/`, `posts/`, `reference/`, `ai-toolkit/`, `lora/output/`.
- В коде нет хардкода секретов.

## 5. Чек-лист для агента
**GPU-сервер:**
- [ ] `nvidia-smi` видит 5070 12 ГБ; `docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi` работает.
- [ ] UFW: разрешить `22` (SSH, key-only), разрешить `8000` только с IP VPS; запретить `8188` и `8000` для остальных.
- [ ] Клонировать `MostDef2000/content`; `cp .env.example .env`, вписать `HF_TOKEN`.
- [ ] `bash scripts/prepare_server.sh` → `bash scripts/download_models.sh` → `docker compose up -d`.
- [ ] Проверить: `curl localhost:8188/system_stats` и `curl localhost:8000/api/status`.
- [ ] SSH: отключить пароль-логин, fail2ban.

**VPS (77.105.142.206):**
- [ ] Caddy с `basicauth` (bcrypt) + авто-HTTPS на `ui.example.com`; `reverse_proxy http://10.123.239.102:8000`.
- [ ] Прямой доступ на `GPU:8000`/`:8188` из интернета запрещён.
- [ ] (Опц.) rate-limit / fail2ban на Caddy.

## 6. Харденинг
- Только Caddy с `basicauth`+HTTPS снаружи.
- `manager` монтирует `/var/run/docker.sock` — выделенный пользователь, ограничить права.
- Фаервол блокирует всё, кроме SSH и прокси-порта с VPS.
- Логи `manager` в `runtime/jobs/*.log`.
- Бэкап `dataset/` и `lora/output/` (самое ценное).
- Мониторинг VRAM (`nvidia-smi`) и диска.
- Контент: только взрослый вымышленный персонаж, без реальных людей и несовершеннолетних.

## 7. Verification
1. `nvidia-smi` → 5070, 12 ГБ, driver ≥580.
2. `docker compose config` валиден; `docker compose up -d` поднимает `comfyui`+`manager`.
3. `curl localhost:8188/system_stats` → JSON.
4. `curl localhost:8000/api/status` → `{comfy_up, gpu_busy, disk_free_gb, ...}`.
5. С VPS: `curl -u user:pass https://ui.example.com/api/status` работает; прямой `http://GPU_IP:8000` и `:8188` — отказ.

## 8. Фактические замеры (ответ сисадмин-агента, 12.09)
- GPU: 12227 MiB total / 1607 MiB used (YOLO ~1.6 ГБ). nvidia runtime активен (`io.containerd.runc.v2, nvidia, runc`). CUDA 13.2, драйвер 595.91.07.
- `/var/lib/docker` — 4.1 ГБ на системном диске (только образы/кэш контейнера).
- Data-том `/opt/sea-speed-worker`: 737.2 ГБ, свободно 623.9 ГБ; занято `releases` 49 ГБ, `runtimes` 5.3 ГБ, `shared` ~0.
- **Решение по диску:** проект и `runtime/` (веса ~35 ГБ, dataset, posts, lora/output) разворачиваются на data-томе (`/opt/sea-speed-worker/valery/`). Системный диск не трогать.
- VPS `77.105.142.206`: **не замерен** — хост не в `SYSADMIN_REMOTE_HOSTS` (разрешены 82.146.37.153, 10.123.239.102). Обход через NL: порт 22 открыт, ключа нет (`Permission denied`), 8443/2222 закрыты. Нужно добавить хост в белый список MCP или прислать вывод 4 команд (см. unresolved).

## 9. Откат (записано агентом в JOURNAL.md)
- daemon.json — `rm /etc/docker/daemon.json`.
- repo — `rm /etc/apt/sources.list.d/nvidia-container-toolkit.list /usr/share/keyrings/nvidia-container-toolkit-keyring.asc`.
- toolkit — `apt-get remove nvidia-container-toolkit`.
