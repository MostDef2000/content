# SYSADMIN-HANDOFF v1

Хэндоф №2 — **развёртывание** стека. Предыдущий (инвентаризация дисков,
read-only) выполнен, результаты в `AUDIT.md` §8. Это handoff режима
`prepare-changes`: агент готовит и **исполняет план только после явного
подтверждения пользователя в своей сессии**.

## objective
Развёртывание стека генерации: ComfyUI + `manager` на GPU-сервере,
Caddy + HTTPS на РУ-хосте, домен `3d.mostdef.ru`. Без публикации портов
GPU-сервера наружу.

## hosts
- GPU: `10.123.239.102` (ZeroTier; RTX 5070 12 ГБ, драйвер 595.91, CUDA 13.2,
  Docker 29.8 + Compose v5.5.1, nvidia runtime активен). За NAT, публичного IP нет.
- РУ: `82.146.37.153` (ZeroTier 10.123.239.101; Caddy + домен `3d.mostdef.ru`).
  NL `77.105.142.206` **не используется** (не в ZeroTier, caddy не установлен).
- Алиасы из `~/.ssh/config` у оркестратора недоступны — подставьте свои.

## mode
prepare-changes (исполнение — после подтверждения пользователя в сессии агента)

## forbidden
- Не публиковать порты `8188`/`8000` в интернет; не менять фаервол наружу GPU.
- Не выводить и не пересылать `HF_TOKEN`, пароли, приватные ключи, bcrypt-хэш basicauth.
- Не трогать базовый YOLO-воркер без согласования (делит GPU); тяжёлые задачи — после его остановки.
- Не коммитить `.env`, `runtime/`, датасет, веса в репозиторий.

## context
- Репозиторий: `MostDef2000/content` (private); чек-лист — issue #1; аудит — `AUDIT.md`.
- Целевой каталог проекта: `/opt/sea-speed-worker/valery` (data-том 737 ГБ,
  свободно ~624 ГБ; системный диск 97.9 ГБ — не трогать).
- Секреты: `HF_TOKEN` в `.env` (chmod 600) — только на этап скачивания весов.
  Токен пользователь передаёт агенту напрямую, не через хэндоф.

## план (исполнять по шагам, подтверждение пользователя между шагами)
1. **GPU — подготовка:** клонировать репо в `/opt/sea-speed-worker/valery`;
   `cp .env.example .env`, вписать `HF_TOKEN` (от пользователя), `chmod 600 .env`;
   `bash scripts/prepare_server.sh`.
2. **GPU — веса:** `bash scripts/download_models.sh` (~35 ГБ → data-том).
3. **GPU — старт:** `docker compose up -d`; проверить
   `curl localhost:8188/system_stats` и `curl localhost:8000/api/status`.
4. **РУ — Caddy:** установить/проверить caddy; `caddy hash-password '<пароль от
   пользователя>'`; взять `deploy/Caddyfile` из репо, вписать bcrypt-хэш;
   проверить DNS `getent hosts 3d.mostdef.ru` → `82.146.37.153`
   (если A-записи нет — сообщить, ждём пользователя); `caddy validate`; reload.
5. **Проверка сквозняком:** с РУ `curl -u user:pass https://3d.mostdef.ru/api/status`
   → JSON; прямой `http://10.123.239.102:8000` извне — отказ.
6. **Фидбек:** вернуть `status`, `commands_run`, `evidence`, `risks`, `unresolved`,
   `kb_updated`; оркестратор запишет результаты в `AUDIT.md` и issue #1.

## откат
- GPU: `docker compose down`; `rm -rf /opt/sea-speed-worker/valery` (осторожно —
  только каталог valery, не трогать releases/runtimes).
- РУ: удалить сайт-блок Caddy, reload.
