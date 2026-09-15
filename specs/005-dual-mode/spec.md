# Feature Specification: Dual-mode posts (public/private)

- Feature: 005-dual-mode
- Status: Active

## Product outcome

`scripts/queue_workflow.py post` пишет результат в одну из двух per-model
папок: `models/<id>/posts/` (публичные посты) или
`models/<id>/posts-private/` (приватные) — по флагу
`--mode {public,private}` (по умолчанию `public`). Состав папки поста
(photo, caption.txt, prompt.txt) и rollback не меняются.

## User scenarios

1. Оператор генерирует публичный пост — по умолчанию
   `python scripts/queue_workflow.py post …` кладёт папку в
   `models/<id>/posts/<name>/`.
2. Оператор генерирует приватный пост — тот же вызов с `--mode private`
   кладёт папку в `models/<id>/posts-private/<name>/`.
3. Сбой генерации удаляет недописанную папку поста (rollback
   `shutil.rmtree`) в обоих режимах.

## Requirements

- Парсер: `post.add_argument("--mode", choices=("public", "private"),
  default="public", help="public → models/<id>/posts/, private →
  models/<id>/posts-private/")`.
- `generate_post`: `subdir = "posts" if args.mode == "public" else
  "posts-private"`; `destination = ROOT / "models" / model_id / subdir /
  post_name`.
- Транзитная папка ComfyUI не меняется: `filename_prefix` остаётся
  `f"posts/{model_id}"` (runtime/output/posts/<id>/…).
- Состав файлов поста (photo<ext>, caption.txt, prompt.txt), запрет пути в
  `--name` и rollback `shutil.rmtree` — без изменений.
- Обе папки назначения остаются gitignored (см. 002): `models/*/posts/`,
  `models/*/posts-private/`.
- Guardrail-сборка positive/negative (003) — без изменений.

## Acceptance criteria

- [ ] `python scripts/queue_workflow.py post --help` показывает `--mode` с
      choices public/private и default public.
- [ ] `python -m py_compile scripts/queue_workflow.py` проходит.
- [ ] `--mode private` вычисляет destination в
      `models/<id>/posts-private/`; `--mode public` и отсутствие флага — в
      `models/<id>/posts/`.
- [ ] Транзитный `filename_prefix` ComfyUI остаётся `posts/<model_id>`.

## NFR

- NFR-1: изменение локально в парсере `post`/`generate_post`, без новых
  зависимостей — evidence: diff scripts/queue_workflow.py.
- NFR-2: rollback-поведение идентично в обоих режимах — evidence: единый
  except-блок с `shutil.rmtree` вокруг записи файлов поста.

## Compatibility and boundaries

Не меняет подкоманды candidates/expand, workflows JSON, `lora/caption.py`,
manager. Не трогает `posts/`, `posts-private/`, `runtime/` вне
существующего транзита `runtime/output/posts/`.
