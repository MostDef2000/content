# Implementation Plan: SDD Adoption

- Specification: specs/001-sdd-adoption/spec.md
- Status: Active

## Architecture

SDD-слой — статические markdown-артефакты в репозитории: `.specify/memory/constitution.md`
(правила процесса), `specs/README.md` (иерархия и flow), `AGENTS.md` (точка входа
агентов), `specs/001-sdd-adoption/{spec,plan,tasks}.md` (первая фича — само подключение).
Инфраструктурный контур (серверы) остаётся вне SDD-авторизации и идёт через handoff.

## Decisions

### D-001 — Упрощённый constitution
Полный governance sea-speed (standing delegation, production contours, checkpoints)
избыточен для проекта без автономного production. Оставлены: outcome-first,
traceability, feature identity, авторизация `commit approved`, handoff для серверов.
Отклонено: копирование sea-speed constitution целиком — неоправданная сложность.

### D-002 — Issue #1 остаётся каноническим деплой-бэклогом
Деплой уже трассируется issue #1 + AUDIT.md; дублировать его отдельной фичей не нужно.
Runtime-замеры агента (data-том 737 ГБ) записаны в AUDIT.md.

### D-003 — Русскоязычные артефакты
README/AUDIT/issue уже на русском; SDD-артефакты — тоже, для консистентности.
Отклонено: английский (расхождение с остальной документацией проекта).

### D-004 — Data-том для тяжёлых данных
Проект разворачивается в `/opt/sea-speed-worker/valery/` (737 ГБ LV); системный диск
(68 ГБ) только под Docker-образы. Отражено в README и SYSADMIN-HANDOFF.

## Affected contours

- Repository: documentation/SDD layer only.
- GPU/VPS: изменений нет; деплой пойдёт отдельным handoff-циклом.

## Risks

- Риск: процесс забросить. Митигируется тем, что issue-шаблоны уже ведут к spec.
- Риск: расхождение artifacts и реальности. Митигируется правилом V (runtime-фидбек).

## Test design

- unit: `Risk profile: NOT REQUIRED` (документация).
- runtime-manual: агент входит в репозиторий и находит канонические точки входа (P2).
