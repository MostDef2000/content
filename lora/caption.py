from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def base_caption() -> str:
    character = json.loads((ROOT / "character.json").read_text(encoding="utf-8"))
    appearance = character["appearance"]
    return (
        "[trigger], fictional adult woman age 23, "
        f"{appearance['hair']}, {appearance['eyes']}, {appearance['skin']}, "
        f"{appearance['face']}, {appearance['build']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Create initial ai-toolkit captions")
    parser.add_argument("--force", action="store_true", help="Overwrite existing captions")
    args = parser.parse_args()

    image_dir = ROOT / "dataset" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        parser.error(f"No JPG or PNG images found in {image_dir}")

    caption = base_caption()
    written = 0
    for image in images:
        target = image.with_suffix(".txt")
        if target.exists() and not args.force:
            continue
        target.write_text(caption + "\n", encoding="utf-8")
        written += 1

    print(f"Created {written} captions. Review each .txt and append its pose, clothes, and setting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
