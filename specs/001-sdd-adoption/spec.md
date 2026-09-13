# Feature Specification: SDD Adoption

- Feature: 001-sdd-adoption
- Status: Active

## Product outcome

Оркестратор и агенты работают над проектом `content` по единому
spec-driven процессу воркспейса: долговременные артефакты (constitution,
spec/plan/tasks), канонический Issue как backlog, авторизация `commit approved`
после явной области работ, серверные действия через SYSADMIN-HANDOFF.

## User scenarios

1. Новая значимая работа начинается с Issue и видимой области работ; после
   `commit approved` создаётся/обновляется тройка spec/plan/tasks.
2. Серверные изменения (деплой, модели, фаервол) всегда идут через
   `SYSADMIN-HANDOFF v1`, а не через source authorization.
3. Runtime-результаты (например, реальные замеры дисков от сисадмин-агента)
   записываются обратно в активные артефакты.

## Requirements

- `.specify/memory/constitution.md` — канонический SDD-слой проекта.
- `specs/README.md` — иерархия источников истины и normal flow.
- `AGENTS.md` — точка входа агентов: правила, авторизация, hard limits.
- Авторизация source-работ: видимая область работ + `commit approved` в следующем сообщении.
- Секреты, веса, датасет, посты и сгенерированный вывод никогда не попадают в Git.

## Acceptance criteria

- [ ] Три SDD-артефакта (`constitution`, `specs/README.md`, `AGENTS.md`) в репозитории.
- [ ] Активная фича `001-sdd-adoption` имеет spec/plan/tasks с трассируемостью.
- [ ] Текущее незавершённое состояние (деплой, данные агента) отражено в artifacts.

## NFR

- NFR-1: документация читается агентом без доступа к чату — evidence: файлы в `main`.
- NFR-2: процесс не тяжелее проекта (упрощён против sea-speed) — evidence: сравнение размеров constitution.

## Compatibility and boundaries

Не заменяет governance воркспейса; не создаёт production-авторитет; не трогает
YOLO-воркер и sea-speed инфраструктуру, кроме использования data-тома
`/opt/sea-speed-worker/valery/` по согласованию.
