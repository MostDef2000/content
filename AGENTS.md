# Content Agent Entry Point

Status: Active

Проект: локальная генерация консистентной взрослой AI-модели (Valery) —
ComfyUI + FLUX.1-dev FP8 + LoRA + веб-пульт (`manager`).
GitHub: `MostDef2000/content` (private). Инфраструктура: GPU-сервер (ZeroTier
`10.123.239.102`, за NAT) + РУ-хост (`82.146.37.153`, ZeroTier 10.123.239.101,
Caddy + домен `3d.mostdef.ru`). NL-хост не используется. Серверные действия — только
через сисадмин-агента (`SYSADMIN-HANDOFF v1`).

## Canonical entry points

- SDD: `.specify/memory/constitution.md`, `specs/README.md`, активные
  `specs/<feature>/{spec,plan,tasks}.md`.
- Деплой: issue #1 (canonical checklist), `AUDIT.md`, `SYSADMIN-HANDOFF.md`.
- Запуск и поток генерации: `README.md`.

## Source authorization

Полная видимая область работ (scope) предъявляется в сообщении ассистента;
авторизация — точная фраза `commit approved` в **следующем** сообщении
пользователя. Приём возобновляет только ту же самую принятую область и не
создаёт новой. Серверные мутации (compose up, скачивание моделей, Caddy,
фаервол) выполняются через handoff-цикл, не через source authorization.

## Hard rules

1. `main` — единственный источник истины репозитория.
2. Одна задача = один canonical Issue; значимые изменения ссылаются на один feature spec.
3. Тяжёлые GPU-задачи — строго по одной (busy-лок в `manager`); 12 ГБ VRAM общие с YOLO-воркером.
4. Никогда не коммитить: `.env`, секреты, токены, `runtime/`, `dataset/`, `posts/`,
   `reference/`, `ai-toolkit/`, `lora/output/`, веса моделей, логи, сгенерированный вывод.
5. Персонаж — взрослый вымышленный; без реальных людей и несовершеннолетних.
6. Секреты только в `.env` на сервере (`chmod 600`); в чат и Git не попадают.
