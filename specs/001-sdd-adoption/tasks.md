# Tasks: SDD Adoption

- Specification: specs/001-sdd-adoption/spec.md
- Plan: specs/001-sdd-adoption/plan.md

## Delivery tasks

- [x] T001 Создать `.specify/memory/constitution.md` (упрощённый, v1.0.0).
- [x] T002 Создать `specs/README.md` (иерархия, flow, активные фичи).
- [x] T003 Создать `AGENTS.md` (точка входа, авторизация, hard limits).
- [x] T004 Создать `specs/001-sdd-adoption/{spec,plan,tasks}.md`.
- [x] T005 Отразить фактические замеры агента (data-том) в README/AUDIT.
- [ ] T006 Запушить артефакты в `main` (после `commit approved`).
- [ ] T007 Проверить, что деплой-флоу агента ссылается на SDD-точки входа (issue #1 комментарий).

## Acceptance traceability

- AC-1 (артефакты в main) ← T001–T004, T006; evidence: файлы в `main`, commit SHA.
- AC-2 (трассируемость активной фичи) ← T004; evidence: этот файл.
- AC-3 (текущее состояние отражено) ← T005; evidence: AUDIT.md §8, README warning.

## Definition of Done

- [ ] Все задачи выполнены, AC подтверждены evidence.
- [ ] Коммит в `main` не содержит секретов/артефактов генерации.
- [ ] Runtime-фидбек деплоя будет записан обратно при следующем значимом изменении.
