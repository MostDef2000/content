#!/usr/bin/env python3
"""One-shot idempotent migration: single-model layout -> multi-model registry.

Run on the server from the repository root (Phase 1, specs/002):

    python scripts/migrate_to_registry.py [--dry-run]

What it does:
  1. Creates models/registry.json if missing (valery23, active).
  2. Creates models/valery23/{reference/candidates,dataset/images,lora/output,
     posts,posts-private}.
  3. Moves root reference/candidates/*, dataset/images/* and posts/* into
     models/valery23/ (targets that already exist are kept; sources of skipped
     files stay in place — nothing is overwritten or deleted).
  4. Creates models/valery23/character.json from the root character.json,
     adding the top-level "body" field (from appearance.build) if absent.
  5. Seeds models/valery23/prompt_profile.json from runtime/character_prompt.txt
     if no profile exists yet (the runtime prompt becomes the "casting" scene).

Backup / rollback:
  Before any move, non-empty sources are copied into
  runtime/backup/migrate_to_registry-<timestamp>/ (mirrored repo layout).
  To roll back, copy the backup contents back over the repository root:

      cp -a runtime/backup/migrate_to_registry-<timestamp>/ .

  The script is idempotent: re-running it is a no-op for everything already
  migrated, and it never deletes data (empty source directories are left in
  place).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "valery23"
DEFAULT_BODY = "tall athletic fitness physique, defined waist, proportional full bust"

REGISTRY = {
    "version": 1,
    "models": [
        {
            "id": MODEL_ID,
            "name": "Valery",
            "age": 23,
            "height_cm": 178,
            "tags": ["fitness", "editorial"],
            "active": True,
            "created": "2026-09-15T00:00:00Z",
            "lora": f"{MODEL_ID}.safetensors",
            "groups": [],
        }
    ],
}

PER_MODEL_DIRS = (
    "reference/candidates",
    "dataset/images",
    "lora/output",
    "posts",
    "posts-private",
)

MOVE_SOURCES = (
    ("reference/candidates", "reference/candidates"),
    ("dataset/images", "dataset/images"),
    ("posts", "posts"),
)


def log(dry_run: bool, message: str) -> None:
    prefix = "[dry-run] " if dry_run else ""
    print(f"{prefix}{message}")


def backup_sources(dry_run: bool) -> Path | None:
    existing = [
        source
        for source in (ROOT / relative for relative, _ in MOVE_SOURCES)
        if source.is_dir() and any(source.iterdir())
    ]
    if not existing:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = ROOT / "runtime" / "backup" / f"migrate_to_registry-{stamp}"
    log(dry_run, f"backup {len(existing)} source path(s) -> {backup_root}")
    if dry_run:
        return backup_root
    for source in existing:
        target = backup_root / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
    return backup_root


def move_tree(source: Path, target: Path, dry_run: bool) -> int:
    """Move children of source into target; never overwrite existing targets."""
    if not source.is_dir():
        return 0
    moved = 0
    for child in sorted(source.iterdir()):
        destination = target / child.name
        if destination.exists():
            log(dry_run, f"keep existing {destination} (source {child} left in place)")
            continue
        log(dry_run, f"move {child} -> {destination}")
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(child), str(destination))
        moved += 1
    return moved


def migrate_registry(models_dir: Path, dry_run: bool) -> None:
    registry_path = models_dir / "registry.json"
    if registry_path.exists():
        log(dry_run, f"registry exists: {registry_path}")
        return
    log(dry_run, f"create {registry_path}")
    if not dry_run:
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(
            json.dumps(REGISTRY, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def migrate_character(root_character: Path, model_dir: Path, dry_run: bool) -> None:
    target = model_dir / "character.json"
    if not root_character.exists():
        log(dry_run, f"no root {root_character.name}, skip")
        return
    if target.exists():
        log(dry_run, f"character exists: {target}")
        return
    data = json.loads(root_character.read_text(encoding="utf-8"))
    if not data.get("body"):
        data["body"] = data.get("appearance", {}).get("build", DEFAULT_BODY)
    log(dry_run, f"create {target} (with top-level body)")
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def migrate_prompt_seed(runtime_prompt: Path, model_dir: Path, dry_run: bool) -> None:
    profile_path = model_dir / "prompt_profile.json"
    if profile_path.exists():
        log(dry_run, f"profile exists: {profile_path}")
        return
    if not runtime_prompt.exists():
        log(dry_run, f"no {runtime_prompt}, skip profile seed")
        return
    profile = {
        "version": 1,
        "face": "",
        "body": "",
        "style": "",
        "negative": "",
        "scenes": {"casting": runtime_prompt.read_text(encoding="utf-8").strip()},
    }
    log(dry_run, f"seed {profile_path} from {runtime_prompt}")
    if not dry_run:
        profile_path.write_text(
            json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate single-model layout to the model registry")
    parser.add_argument("--dry-run", action="store_true", help="print actions without changing anything")
    args = parser.parse_args(argv)
    dry = args.dry_run

    model_dir = ROOT / "models" / MODEL_ID
    for relative in PER_MODEL_DIRS:
        target = model_dir / relative
        log(dry, f"mkdir {target}")
        if not dry:
            target.mkdir(parents=True, exist_ok=True)

    migrate_registry(ROOT / "models", dry)
    migrate_character(ROOT / "character.json", model_dir, dry)

    backup_sources(dry)
    for source_relative, target_relative in MOVE_SOURCES:
        move_tree(ROOT / source_relative, model_dir / target_relative, dry)

    migrate_prompt_seed(ROOT / "runtime" / "character_prompt.txt", model_dir, dry)
    print("Migration complete." if not dry else "Dry run complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
