---
name: Deployment task
about: Шаг развёртывания стека на GPU-сервере или lifesim (публичный край Caddy)
title: "[deploy] "
labels: deployment
assignees: ""
---

## Сервер
- [ ] GPU (ZeroTier 10.123.239.102)
- [ ] lifesim (135.106.195.18) — публичный край, Caddy на `3d.mostdef.ru`

## Шаг из чек-листа (issue #1)
<!-- например: download_models.sh / docker compose up / Caddy reverse_proxy -->

## Команды
```bash

```

## Проверка
- [ ] `curl localhost:8188/system_stats` (GPU)
- [ ] `curl localhost:8000/api/status` (GPU)
- [ ] `curl https://3d.mostdef.ru/api/status` (публично: без auth → 401, с auth → 200)
- [ ] Доступ к GPU — только SSH-обёртка с lifesim (порт 22024, пользователь
      `seaspeedadmin`); прямой SSH на GPU запрещён

## Результат / логи
<!-- вставь вывод проверок; без секретов/токенов -->

## Блокеры
<!-- что остановило шаг, если не выполнен; заметка: деплой-маршрут manager в перепайке (транспорт через РУ уничтожен) — см. README "DEPLOY ROUTE STALE" -->
