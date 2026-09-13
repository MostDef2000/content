---
name: Deployment task
about: Шаг развёртывания стека на GPU- или VPS-сервере
title: "[deploy] "
labels: deployment
assignees: ""
---

## Сервер
- [ ] GPU (ZeroTier 10.123.239.102)
- [ ] РУ-хост (82.146.37.153)

## Шаг из чек-листа (issue #1)
<!-- например: download_models.sh / docker compose up / Caddy reverse_proxy -->

## Команды
```bash

```

## Проверка
- [ ] `curl localhost:8188/system_stats` (GPU)
- [ ] `curl localhost:8000/api/status` (GPU)
- [ ] `curl -u user:pass https://3d.mostdef.ru/api/status` (РУ-хост)

## Результат / логи
<!-- вставь вывод проверок; без секретов/токенов -->

## Блокеры
<!-- что остановило шаг, если не выполнен -->
