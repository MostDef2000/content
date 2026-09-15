from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# prompts.py (guardrail validation) lives at the repo root, next to lora/.
sys.path.insert(0, str(ROOT))
import prompts  # noqa: E402

# Appended to the base caption for manifest entries tagged "explicit"
# (feature 004). Deliberately contains no terms matched by
# prompts.GUARDRAIL_PATTERNS, so it passes validate_positive.
EXPLICIT_BODY_CONTEXT = (
    "nude, natural adult female body, bare skin, anatomically correct, "
    "consistent body proportions, consistent bust waist and hip shape, "
    "neutral full-body reference framing"
)


def resolve_model_id(requested: str | None) -> str:
    """--model wins; otherwise the active model from models/registry.json."""
    registry_path = ROOT / "models" / "registry.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"Model registry not found: {registry_path}")
    models = json.loads(registry_path.read_text(encoding="utf-8")).get("models") or []
    if not models:
        raise RuntimeError("models/registry.json contains no models — register a model first")
    if requested:
        for entry in models:
            if str(entry.get("id")) == requested:
                return requested
        known = ", ".join(str(entry.get("id")) for entry in models)
        raise RuntimeError(f"Unknown model id {requested!r}; registered models: {known}")
    for entry in models:
        if entry.get("active"):
            return str(entry["id"])
    return str(models[0]["id"])


def base_caption(model_id: str) -> str:
    character_path = ROOT / "models" / model_id / "character.json"
    if not character_path.exists():
        raise FileNotFoundError(f"character.json not found for model {model_id!r}: {character_path}")
    character = json.loads(character_path.read_text(encoding="utf-8"))
    appearance = character["appearance"]
    age = max(prompts.AGE_FLOOR, int(character.get("age", prompts.AGE_FLOOR)))
    return (
        f"{model_id}, fictional adult woman age {age}, "
        f"{appearance['hair']}, {appearance['eyes']}, {appearance['skin']}, "
        f"{appearance['face']}, {appearance['build']}"
    )


def manifest_path(model_id: str) -> Path:
    """Per-model dataset manifest, next to images/ (not inside)."""
    return ROOT / "models" / model_id / "dataset" / "manifest.json"


def load_manifest(model_id: str, image_dir: Path) -> dict:
    """Read the dataset manifest and reconcile it with the files on disk.

    Files without an entry get ``{"tag": "tasteful", "caption": null,
    "created": <mtime ISO UTC, Z-suffix>}``; entries without a file are
    dropped; corrupt/structurally invalid manifests fall back to empty;
    ``version > 1`` is an error.
    """
    path = manifest_path(model_id)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {"version": 1, "images": []}
    else:
        data = {"version": 1, "images": []}
    if not isinstance(data, dict) or not isinstance(data.get("images"), list):
        data = {"version": 1, "images": []}
    version = int(data.get("version", 1))
    if version > 1:
        raise RuntimeError(
            f"Unsupported dataset manifest version {version} in {path} (expected 1)"
        )
    by_name = {
        str(entry.get("filename")): entry
        for entry in data["images"]
        if isinstance(entry, dict) and entry.get("filename")
    }
    reconciled: list[dict] = []
    for image in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        entry = by_name.pop(image.name, None)
        if entry is None:
            entry = {
                "filename": image.name,
                "tag": "tasteful",
                "caption": None,
                "created": datetime.fromtimestamp(
                    image.stat().st_mtime, tz=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        reconciled.append(entry)
    data["version"] = 1
    data["images"] = reconciled
    return data


def save_manifest(model_id: str, data: dict) -> None:
    """Atomically write the manifest: tempfile in the same folder + os.replace."""
    path = manifest_path(model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".manifest-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def caption_for(tag: str, base: str) -> str:
    """tasteful → base caption; explicit → base + explicit body context."""
    if tag == "explicit":
        return f"{base}, {EXPLICIT_BODY_CONTEXT}"
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description="Create initial ai-toolkit captions")
    parser.add_argument("--model", default=None, help="Model id from models/registry.json (default: active)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing captions")
    args = parser.parse_args()

    model_id = resolve_model_id(args.model)
    image_dir = ROOT / "models" / model_id / "dataset" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        parser.error(f"No JPG or PNG images found in {image_dir}")

    base = base_caption(model_id)
    manifest = load_manifest(model_id, image_dir)
    entries = {str(entry.get("filename")): entry for entry in manifest.get("images", [])}

    written = 0
    tasteful = 0
    explicit = 0
    for image in images:
        entry = entries.get(image.name)
        if entry is None:  # defensive: load_manifest reconciles, so this is a no-op in practice
            entry = {"filename": image.name, "tag": "tasteful", "caption": None}
            manifest["images"].append(entry)
            entries[image.name] = entry
        target = image.with_suffix(".txt")
        if target.exists() and not args.force:
            continue
        tag = str(entry.get("tag") or "tasteful")
        caption = caption_for(tag, base)
        hits = prompts.validate_positive(caption)
        if hits:
            print(
                f"Blocked terms in caption for {image.name}: {', '.join(hits)} — "
                "file not written; fix the character/profile fields and rerun."
            )
            return 1
        target.write_text(caption + "\n", encoding="utf-8")
        entry["caption"] = caption
        written += 1
        if tag == "explicit":
            explicit += 1
        else:
            tasteful += 1

    save_manifest(model_id, manifest)
    print(f"Created {written} captions ({tasteful} tasteful, {explicit} explicit) for {model_id}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
