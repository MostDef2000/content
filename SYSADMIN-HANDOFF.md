# SYSADMIN-HANDOFF v1

Хэндоф №3 — **сервисный техперерыв + обновление UI**. Хэндоф №2 (деплой)
выполнен, результаты в `AUDIT.md` §10 и issue #1. Владелец разрешил окно
простоя «сейчас». Режим `prepare-changes`: исполнение по шагам с
подтверждением владельца; тяжёлый шаг (перенос data-root) — владелец говорит
«начинай», агент объявляет старт простоя.

## objective
1. Обновить UI на Воркере (кнопка «Выйти» + серверный `/logout`).
2. Добавить `location = /logout` в nginx на РУ.
3. Перенести docker data-root на 1 ТБ-том (`/var/lib/docker` → `/opt/sea-speed-worker/docker-data`).
4. Удалить отключённый пакет caddy на РУ (решение оркестратора, issue #1).

## hosts
- Воркер: `10.123.239.102` (ZeroTier). Стек: контейнеры `valery-comfyui-1`,
  `valery-manager-1`; образ `valery-comfyui` на системном диске.
- РУ: `82.146.37.153` через релей Воркер→РУ:2222 (ключ `id_ed25519_valery_relay`).
- NL: не используется.

## mode
prepare-changes (исполнение — после подтверждения владельца в сессии агента)

## forbidden
- Не удалять `/var/lib/docker.old` (или как назван старый каталог) до верификации — это откат.
- Не трогать контейнеры `netdata`/`gpu-statsd` и YOLO-воркер (кроме вынужденной
  остановки на время переноса data-root — вернуть сразу после).
- Не выводить пароли/HF_TOKEN; `.htpasswd_3d` не трогать.
- В daemon.json только ДОБАВИТЬ `"data-root"` — не затирать nvidia runtime конфиг.

## порядок (по шагам, подтверждение между шагами)
1. **UI-файлы на Воркер:** скачать из репо (HEAD main) `management/main.py` и
   `management/static/index.html` через MCP; сверить sha256 с эталонами
   оркестратора (передаст владелец); положить в
   `/opt/sea-speed-worker/valery/management/…`; `docker restart valery-manager-1`.
   Smoke: `curl -s localhost:8000/ | grep -o Выйти` → есть;
   `curl -s -o /dev/null -w '%{http_code}' localhost:8000/logout` → 401.
2. **nginx /logout на РУ:** в vhost `3d.mostdef.ru` добавить
   `location = /logout { return 401; }` (return в rewrite-фазе — всегда 401);
   `nginx -t` → reload. Проверка с РУ или NL:
   `curl -u admin:'<пароль>' -o /dev/null -w '%{http_code}' https://3d.mostdef.ru/logout` → 401.
3. **Перенос data-root** (простой ~10–15 мин, YOLO ляжет):
   - подготовка: `df -h /` и `du -sh /var/lib/docker`; убедиться, что на
     `/opt/sea-speed-worker` ≥ 2× размер /var/lib/docker;
   - `docker compose down` в `/opt/sea-speed-worker/valery`; остановить прочие
     контейнеры (вкл. YOLO);
   - `systemctl stop docker docker.socket`;
   - `rsync -aHAX /var/lib/docker/ /opt/sea-speed-worker/docker-data/`
     (сверить `du -sh` обеих сторон);
   - `/etc/docker/daemon.json`: добавить `"data-root": "/opt/sea-speed-worker/docker-data"`
     (merge, nvidia runtime сохранить; backup daemon.json.bak.dataroot);
   - `mv /var/lib/docker /var/lib/docker.old`;
   - `systemctl start docker`; `docker ps` — все контейнеры на месте
     (`restart: unless-stopped` поднимет valery-стек; YOLO — вернуть владельцу);
   - smoke: 8188/system_stats 200, 8000/api/status 200; YOLO-воркер — работоспособен;
   - rollback при проблемах: stop docker → вернуть daemon.json → `rm -rf
     /opt/sea-speed-worker/docker-data` → `mv /var/lib/docker.old /var/lib/docker`
     → start docker.
4. **Caddy на РУ:** `apt-get remove --purge caddy`; конфиг-бэкапы
   `/etc/caddy/Caddyfile.bak.*` можно оставить (референс в git: `deploy/Caddyfile`).
5. **Финальный smoke + отчёт:** внешний smoke `https://3d.mostdef.ru/api/status`
   (401 без пароля / 200 с паролем); вернуть `status`, `commands_run`, `evidence`,
   `risks`, `unresolved`, `kb_updated`. `/var/lib/docker.old` НЕ удалять — решение
   об удалении после суток аптайма, отдельно.
