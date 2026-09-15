from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


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
    age = int(character.get("age", 23))
    return (
        f"[trigger], fictional adult woman age {age}, "
        f"{appearance['hair']}, {appearance['eyes']}, {appearance['skin']}, "
        f"{appearance['face']}, {appearance['build']}"
    )


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

    caption = base_caption(model_id)
    written = 0
    for image in images:
        target = image.with_suffix(".txt")
        if target.exists() and not args.force:
            continue
        target.write_text(caption + "\n", encoding="utf-8")
        written += 1

    print(f"Created {written} captions for {model_id}. Review each .txt and append its pose, clothes, and setting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
