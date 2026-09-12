---
name: Deployment task
about: Шаг развёртывания стека на GPU- или VPS-сервере
title: "[deploy] "
labels: deployment
assignees: ""
---

## Сервер
- [ ] GPU (ZeroTier 10.123.239.102)
- [ ] VPS (77.105.142.206)

## Шаг из чек-листа (issue #1)
<!-- например: download_models.sh / docker compose up / Caddy reverse_proxy -->

## Команды
```bash

```

## Проверка
- [ ] `curl localhost:8188/system_stats` (GPU)
- [ ] `curl localhost:8000/api/status` (GPU)
- [ ] `curl -u user:pass https://ui.example.com/api/status` (VPS)

## Результат / логи
<!-- вставь вывод проверок; без секретов/токенов -->

## Блокеры
<!-- что остановило шаг, если не выполнен -->
