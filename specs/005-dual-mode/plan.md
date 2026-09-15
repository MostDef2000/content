# Implementation Plan: Dual-mode posts (public/private)

- Specification: specs/005-dual-mode/spec.md
- Status: Active

## Architecture

- Флаг `--mode` в подкоманде `post` (`choices=("public", "private")`,
  default `public`).
- `generate_post` вычисляет `subdir` из `args.mode` и строит
  `destination = ROOT / "models" / model_id / subdir / post_name`.
- Транзит ComfyUI не меняется: рендер всегда идёт в `runtime/output/posts/`
  (`filename_prefix = f"posts/{model_id}"`), затем `shutil.move` в
  per-model папку назначения.

## Decisions

### D-001 — Две фиксированные per-model папки вместо произвольного --dest
`posts/` и `posts-private/` уже зарезервированы layout'ом (002) и
gitignored; произвольный путь усложнил бы аудит и рисковал бы выходом из
`models/<id>/`. Отклонено: `--dest <path>`.

### D-002 — Транзитная папка ComfyUI общая
`filename_prefix` остаётся `posts/<model_id>`: отдельный транзит потребовал
бы правки output-безопасности и не даёт выигрыша — итоговый режим
определяется только папкой назначения. Отклонено: транзит
`posts-private/` в runtime.

### D-003 — default=public
Существующие вызовы и автопостинг продолжают работать без изменений;
публичный режим — безопасное значение по умолчанию.

## Affected contours

- Repository: `scripts/queue_workflow.py`, `specs/`.
- Server: изменений деплоя не требуется.

## Risks

- Риск: приватный пост по ошибке генерируется в `posts/` (забытый флаг).
  Митигируется явным help и ревью перед выкладкой; обратное (public вместо
  private) исключено default'ом только при полном отсутствии флага.
- Риск: путаница при ручной выкладке из двух папок. Митигируется
  раздельными папками layout'а (002) и неизменным составом файлов поста.

## Test design

- compile: `python -m py_compile scripts/queue_workflow.py` (P0).
- cli: `python scripts/queue_workflow.py post --help` — `--mode` виден,
  choices и default корректны (P0).
- code-review: subdir/destination-логика и неизменность filename_prefix
  (P1).
- runtime-manual: пост с `--mode private` на сервере (P2).
