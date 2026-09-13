# Content Specifications

Status: Active

Этот каталог — долговременный SDD-слой проекта `content`.

## Source-of-truth hierarchy

- GitHub Issue: канонический backlog, авторизация и audit-история (issue #1 — деплой).
- `specs/<feature>/spec.md`: ЧТО/ЗАЧЕМ + измеримые NFR и evidence.
- `plan.md`: КАК, решения, отклонённые альтернативы, риски, тест-дизайн.
- `tasks.md`: ограниченное исполнение, acceptance-трассируемость, DoD.
- Исходный код: реализация.
- `AUDIT.md` / `SYSADMIN-HANDOFF.md`: инфраструктурный контур и передача сисадмин-агенту.
- Runtime-свидетельства: operational-истина, записывается обратно в активные артефакты.

## Feature identifiers

`NNN-feature-slug`; полное имя каталога каноническое.

## Normal flow

```text
Issue + видимая область работ
-> commit approved
-> spec / plan / tasks
-> реализация + проверки (py_compile / json / bash -n / compose config)
-> push в main (в рамках принятой области)
-> деплой через SYSADMIN-HANDOFF v1 (отдельный цикл с агентом)
-> runtime-фидбек обратно в artifacts
```

## Active features

- `001-sdd-adoption` — подключение SDD-процесса оркестратора к проекту.
