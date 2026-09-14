# Audit: GPU-сервер для локальной генерации AI-модели (Valery)

**Назначение:** сервер крутит генерацию фото (FLUX.1-dev FP8 + LoRA персонажа) и обучение LoRA. Управление — через веб-UI, опубликованный на РУ-хосте (Caddy). GPU-сервер сам по себе **не публикуется**.

> Этот стек — **дополнительная опция** поверх базового YOLO-воркера. Он делит GPU; тяжёлые задачи сериализуются busy-локом в `manager`. Во время обучения/тяжёлой генерации YOLO желательно приостановить.

## 1. Железо и система (проверено сисадмин-агентом)
- GPU: **RTX 5070 (GB205), 12227 MiB, compute_cap 12.0 (sm_120)**.
- Драйвер: **595.91.07**, CUDA **13.2**.
- ОС: Ubuntu 26.04 (отмечено как «совместимая»).
- `nvidia-container-toolkit` **1.20.0** + `/etc/docker/daemon.json` с nvidia runtime (виден в `docker info`).
- Docker **29.8.0** + Compose **v5.5.1**.
- Диски: системный `/` **68.3 ГБ** свободно (97.9 ГБ всего); **data-том** `/opt/sea-speed-worker` **623.9 ГБ** свободно (737 ГБ, LVM `ubuntu--vg-sea--speed--worker`, ext4). `sda1` NTFS «Новый том» — не смонтирован, не используется (Windows-раздел).
- Сеть: **ZeroTier 10.123.239.102**, доступен РУ-хост (10.123.239.101), публичного входа нет.
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
Браузер ──HTTPS+basicauth──> РУ 82.146.37.153:Caddy (3d.mostdef.ru)
                                 │ reverse_proxy
                                 ▼ (ZeroTier)
                            GPU 10.123.239.102:manager :8000 ──docker.sock/compose──> comfyui :8188
```
- Публично торчит только Caddy на РУ-хосте.
- GPU-сервер: `8188` и `8000` закрыты извне фаерволом; `8000` разрешён только с РУ-хоста (ZeroTier).
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

**РУ-хост (82.146.37.153):**
- [x] nginx vhost с TLS (Let's Encrypt) + `basicauth` на `3d.mostdef.ru`; `proxy_pass http://10.123.239.102:8000` (Caddy отключён из-за конфликта 80/443 — см. §10).
- [ ] Прямой доступ на `GPU:8000`/`:8188` из интернета запрещён.
- [ ] (Опц.) rate-limit / fail2ban на Caddy.

## 6. Харденинг
- Только nginx с `basicauth`+HTTPS снаружи (на РУ).
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
5. С РУ-хоста: `curl -u user:pass https://3d.mostdef.ru/api/status` работает; прямой `http://GPU_IP:8000` и `:8188` — отказ.

## 8. Фактические замеры (ответ сисадмин-агента, 12.09)
- GPU: 12227 MiB total / 1607 MiB used (YOLO ~1.6 ГБ). nvidia runtime активен (`io.containerd.runc.v2, nvidia, runc`). CUDA 13.2, драйвер 595.91.07.
- `/var/lib/docker` — 4.1 ГБ на системном диске (только образы/кэш контейнера).
- Data-том `/opt/sea-speed-worker`: 737.2 ГБ, свободно 623.9 ГБ; занято `releases` 49 ГБ, `runtimes` 5.3 ГБ, `shared` ~0.
- **Решение по диску:** проект и `runtime/` (веса ~35 ГБ, dataset, posts, lora/output) разворачиваются на data-томе (`/opt/sea-speed-worker/valery/`). Системный диск не трогать.
- Фронт (Caddy) — **РУ-хост `82.146.37.153`**, а не NL `77.105.142.206`: РУ состоит в ZeroTier (`10.123.239.101/24`) и имеет маршрут до GPU-сервера; NL в ZeroTier не состоит, до `10.123.239.102:8000` недоступен (ping 100% loss, tcp closed), caddy там не установлен. Новая инфраструктура не создаётся (см. issue #1, комментарий 13.09).
- Публикация портов на GPU-сервере не нужна вовсе: он за NAT без публичного IP; `8188` — только localhost/compose-net, `8000` — только ZeroTier.
- Домен `3d.mostdef.ru` — A-запись должна указывать на `82.146.37.153` (проверить `getent hosts 3d.mostdef.ru` на сервере; из песочницы DNS не резолвится).

## 9. Откат (записано агентом в JOURNAL.md)
- daemon.json — `rm /etc/docker/daemon.json`.
- repo — `rm /etc/apt/sources.list.d/nvidia-container-toolkit.list /usr/share/keyrings/nvidia-container-toolkit-keyring.asc`.
- toolkit — `apt-get remove nvidia-container-toolkit`.

## 10. Деплой выполнен (13–14.09, хэндоф №2, сисадмин-агент)
- **Шаги 1–2:** 19 файлов + `.env` (600) на `/opt/sea-speed-worker/valery` (sha256 tar сверен); образ `valery-comfyui` (CUDA 13.0.1 + PyTorch cu130 + ComfyUI master, ~13 ГБ, системный диск); 5 весов ~34.6 ГБ на 1 ТБ-томе, побайтово сверены с HF API. 5 файлов, восстанавливавшихся после усечения MCP, дополнительно сверены побайтово с эталоном `17b55a8` — совпадение.
- **Шаг 3:** `comfyui` 127.0.0.1:8188 + `manager` :8000; локальные smoke 200 (`comfy_up:true`, GPU свободна, диск 587.5 ГБ).
- **Шаг 4 — отклонение от плана:** живой фронт — **nginx** vhost `3d.mostdef.ru` (TLS Let's Encrypt + basicauth), `nginx -t` ok; Caddy установлен, но отключён (конфликт 80/443 с существующим nginx на РУ — там есть другие сайты). `deploy/Caddyfile` в репо — референс. DNS проверен; LE-сертификат на 89 дней с автообновлением; authentik-интерфейс перенесён на `auth.mostdef.ru` (200, владелец подтвердил); `/etc/nginx/.htpasswd_3d` (640, www-data); бэкап прежнего конфига `/etc/caddy/Caddyfile.bak.*`.
- **Шаг 5:** smoke с NL: без пароля 401, с паролем 200 + JSON; прямой `10.123.239.102:8000` с NL недоступен (NAT) — как задумано.
- **Грабли весов (устранены агентом):** 404 на kontext в репо HF — файл лежит в `split_files/`; недосозданный каталог `vae/`.
- **Доступ к РУ:** через релей Воркер→РУ (ZeroTier :2222, ключ `id_ed25519_valery_relay` на Воркере, pubkey добавлен в authorized_keys root РУ владельцем); SSH с NL фильтруется на пути (предположительно провайдер — тикет VDSina опционален).
- Мониторинг не задет: netdata, gpu-statsd, auth-стек — healthy.
- **Решения владельца:** ротации HF_TOKEN и SYSADMIN_TOKEN отложены (токены остаются); засвет пароля basicauth и HF_TOKEN в чате агента принят владельцем.
- **Рекомендации оркестратора (исполнение — через сисадмин-агента):** пакет caddy на РУ удалить при следующем заходе (отключён, пользы нет, референс в git); перенос docker data-root на 1 ТБ-том — согласован и готов, требуется окно простоя ~10–15 мин (вкл. YOLO) по выбору владельца.

## 11. Хэндоф №3 выполнен (14.09, сисадмин-агент; коммит 20f1fda)
- **UI:** `management/main.py` + `static/index.html` обновлены на Воркере (gzip+base64 чанками, бэкапы `*.bak-h3`, хэши = эталонам), `valery-manager-1` перезапущен. Кнопка «Выйти» в шапке; `/logout` → 401 (FastAPI + nginx `location = /logout`, всегда 401). Basic-auth logout честный: браузер заново показывает форму входа, «Отмена» = разлогинен.
- **HTTPS (микро-задача):** редирект :80 уже был; добавлен HSTS (`max-age=31536000`). Сертификат **перевыпущен** (LE, CN `3d.mostdef.ru`, до 13.12.2026, вып. 14.09 16:22) — старый был битым, источник «Not secure» у владельца. Подтверждено владельцем в инкогнито; в обычном профиле — чистка site-data. Дополнение: сертификат верифицирован с двух сторон (NL снаружи `verify ok` / РУ изнутри, ECDSA, сер. 6c1dfa81…); в `location /` vhost добавлены `HSTS` + `Cache-Control: no-cache` (с дублем HSTS — наследование `add_header` отменяется своим `add_header`).
- **Перенос data-root:** `/var/lib/docker` → `/opt/sea-speed-worker/docker-data` (daemon.json + `"data-root"`, бэкап `daemon.json.bak.dataroot`, nvidia runtime сохранён); простой **~6 мин**; все контейнеры вернулись, потери данных нет. `/var/lib/docker.old` — откат, не удалять до ≥24 ч аптайма (решение отдельно).
- **Находка агента:** docker с containerd-snapshotter — образы (28 ГБ, вкл. `valery-comfyui` 16.9 ГБ) лежат в `/var/lib/containerd` на системном диске (50 занято / 44 свободно). Перенос data-root освободил только метаданные. **Хэндоф №4 (не срочно):** перенос `/var/lib/containerd` той же схемой, отдельное окно.
- **Caddy на РУ удалён** (`apt purge`), бэкапы конфига оставлены; референс в git: `deploy/Caddyfile`.
- Внешняя проверка: UI открывается у владельца (инкогнито ✓), `/api/status` через домен 200.

## 12. Продуктовая итерация UI (14.09, хэндофы №5–№8; коммиты 9e13fd6, cb09321, 8810723, 74abbcb)
- **№5 (`9e13fd6`):** превью в UI — галереи кандидатов (клик = выбор эталона), датасета и постов; `/api/dataset`, статик-монты `/files/{reference,dataset,posts}`. Хранение на 1 ТБ-томе подтверждено (737 ГБ, ~587 свободно).
- **№6 (`cb09321`):** lifecycle ComfyUI без compose-плагина (в менеджере `docker.io` без plugin — `docker start/stop/restart <container>` по имени через `_docker()`); автостарт ComfyUI при постановке задачи (`_ensure_comfy`, ожидание API до 30 с); честные алерты в UI (`callLifecycle`).
- **№7 (`8810723`):** `queue_workflow.py` — дефолт `--url` из `os.environ["COMFY_URL"]` (`http://comfyui:8188`); источник «Connection refused» при «Сгенерировать» — захардкоженный `127.0.0.1:8188` внутри контейнера менеджера. Без рестарта.
- **№8 (`74abbcb`, выполнен 14.09):** редактируемый промт персонажа в UI (textarea в карточке «Кандидаты»; сохранение в `runtime/character_prompt.txt`; `GET /api/character`; `--prompt` в queue_workflow; очистка поля = возврат к дефолту шаблона); лайтбокс для всех превью (клик — крупное фото, Esc/клик по фону — закрыть; у кандидатов клик также выбирает эталон). Рестарт `valery-manager-1` после `gpu_busy:false`; бэкапы `*.bak-h8`; при переносе base64-чанков 2 опечатки пойманы поэтапной сверкой md5 до установки. `.tmp-handoff8/` очищена по подтверждению оркестратора.
- **Незакрытое №8→№9 (коммит `8b00b9b`):** `/api/status` + `cpu_percent` (дельта `/proc/stat`, в потоке) и `ram_percent` (`/proc/meminfo`) — нагрузка хоста Воркера; бейджи `cpu`/`ram` в шапке (пороги 60/85%). Деплой — хэндоф №9 (ожидает).
- **Конвейер в работе:** первые кандидаты сгенерированы; далее — выбор эталона → «Расширить датасет» → подписи → «Обучить LoRA».
