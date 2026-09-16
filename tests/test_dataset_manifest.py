"""Import-safe tests for lora/caption.py dataset-manifest helpers (Phase 2,
feature 004): reconcile-with-disk, caption_for, base_caption trigger fix,
save round-trip and the manifest version guard; plus
queue_workflow.next_dataset_index (append-instead-of-overwrite on repeated
expand runs). pytest-compatible; pytest is not required.

Run either way:
    python -m pytest tests/test_dataset_manifest.py -q
    python tests/test_dataset_manifest.py

The tests never touch the real repo models/ tree or the server WORKSPACE:
caption.py resolves manifest/character paths through its module-level ROOT,
which is temporarily pointed at a tempfile directory per test.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # content/
sys.path.insert(0, str(REPO_ROOT))  # for `import prompts`
sys.path.insert(0, str(REPO_ROOT / "lora"))  # for `import caption`
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # for `import queue_workflow`

import caption  # noqa: E402  (lora/caption.py)
import prompts  # noqa: E402
import queue_workflow  # noqa: E402  (scripts/queue_workflow.py)

MODEL_ID = "unittestmodel"
FIXED_TS = 1767225600  # 2026-01-01T00:00:00Z — deterministic image mtimes


@contextmanager
def temp_model_root():
    """Point caption.ROOT at a tempfile dir for the duration of the block."""
    original = caption.ROOT
    with tempfile.TemporaryDirectory(prefix="content-manifest-tests-") as tmp:
        caption.ROOT = Path(tmp)
        try:
            yield Path(tmp)
        finally:
            caption.ROOT = original


def _make_model(root: Path, *, manifest: dict | None) -> Path:
    """Create models/<id>/dataset/images with a.jpg+b.jpg (fixed mtimes) and an
    optional manifest.json; return the image dir (passed explicitly to the loader)."""
    image_dir = root / "models" / MODEL_ID / "dataset" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for name in ("a.jpg", "b.jpg"):
        target = image_dir / name
        target.write_bytes(b"\xff\xd8\xff stub jpg")
        os.utime(target, (FIXED_TS, FIXED_TS))
    if manifest is not None:
        (root / "models" / MODEL_ID / "dataset" / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    return image_dir


def _manifest_fixture() -> dict:
    # a.jpg carries a caption (as it would after a caption run) so the stats
    # check can assert captioned=1; c.jpg is an orphaned entry with no file.
    return {
        "version": 1,
        "images": [
            {"filename": "a.jpg", "tag": "explicit", "caption": "x",
             "created": "2026-01-01T00:00:00Z"},
            {"filename": "c.jpg", "tag": "tasteful", "caption": "x",
             "created": "2026-01-01T00:00:00Z"},
        ],
    }


def _stats(images: list[dict]) -> dict:
    """Same math as the dataset stats in management.main."""
    return {
        "total": len(images),
        "tasteful": sum(1 for e in images if e.get("tag") == "tasteful"),
        "explicit": sum(1 for e in images if e.get("tag") == "explicit"),
        "captioned": sum(1 for e in images if e.get("caption")),
    }


def test_load_manifest_reconciles_with_disk():
    with temp_model_root() as root:
        image_dir = _make_model(root, manifest=_manifest_fixture())
        data = caption.load_manifest(MODEL_ID, image_dir)
        assert data["version"] == 1
        # files on disk win: sorted, orphaned c.jpg (no file) excluded,
        # new b.jpg gets the tasteful default entry
        assert [e["filename"] for e in data["images"]] == ["a.jpg", "b.jpg"]
        by_name = {e["filename"]: e for e in data["images"]}
        assert by_name["a.jpg"]["tag"] == "explicit"  # existing entry preserved
        assert by_name["a.jpg"]["caption"] == "x"
        assert by_name["b.jpg"]["tag"] == "tasteful"
        assert by_name["b.jpg"]["caption"] is None
        expected_created = datetime.fromtimestamp(
            (image_dir / "b.jpg").stat().st_mtime, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert by_name["b.jpg"]["created"] == expected_created
        assert _stats(data["images"]) == {
            "total": 2,
            "tasteful": 1,
            "explicit": 1,
            "captioned": 1,
        }


def test_caption_for_tasteful_returns_base():
    base = f"{MODEL_ID}, fictional adult woman age 23, brown hair"
    assert caption.caption_for("tasteful", base) == base


def test_caption_for_explicit_appends_clean_body_context():
    base = f"{MODEL_ID}, fictional adult woman age 23, brown hair"
    explicit = caption.caption_for("explicit", base)
    assert explicit.endswith(caption.EXPLICIT_BODY_CONTEXT)
    assert base in explicit
    # the explicit body context must never introduce guardrail-blocked terms
    assert prompts.validate_positive(explicit) == []


def test_base_caption_uses_model_id_not_placeholder():
    with temp_model_root() as root:
        character = {
            "name": "Unit",
            "age": 23,
            "appearance": {
                "hair": "brown hair",
                "eyes": "green eyes",
                "skin": "fair skin",
                "face": "oval face",
                "build": "athletic build",
            },
        }
        char_path = root / "models" / MODEL_ID / "character.json"
        char_path.parent.mkdir(parents=True, exist_ok=True)
        char_path.write_text(json.dumps(character), encoding="utf-8")

        base = caption.base_caption(MODEL_ID)
        assert base.startswith(f"{MODEL_ID}, "), base  # real id, not a placeholder
        assert "[trigger]" not in base
        assert "fictional adult woman age 23" in base
        assert prompts.validate_positive(base) == []

        # ages below the guardrail floor are clamped to AGE_FLOOR
        character["age"] = 17
        char_path.write_text(json.dumps(character), encoding="utf-8")
        assert "fictional adult woman age 23" in caption.base_caption(MODEL_ID)


def test_save_manifest_round_trip():
    with temp_model_root() as root:
        image_dir = _make_model(root, manifest=None)
        data = caption.load_manifest(MODEL_ID, image_dir)
        data["images"][0]["tag"] = "explicit"

        caption.save_manifest(MODEL_ID, data)
        # saved next to images/, exactly where the loader reads it back from
        assert caption.manifest_path(MODEL_ID) == (
            root / "models" / MODEL_ID / "dataset" / "manifest.json"
        )
        assert caption.manifest_path(MODEL_ID).is_file()

        reloaded = caption.load_manifest(MODEL_ID, image_dir)
        assert reloaded == data


def test_load_manifest_rejects_version_gt_1():
    with temp_model_root() as root:
        image_dir = _make_model(root, manifest={"version": 2, "images": []})
        try:
            caption.load_manifest(MODEL_ID, image_dir)
        except RuntimeError:
            pass
        else:
            raise AssertionError("RuntimeError expected for manifest version 2")


def test_next_dataset_index_empty_dir_is_1():
    with tempfile.TemporaryDirectory(prefix="content-nextindex-") as tmp:
        dataset_dir = Path(tmp)
        assert queue_workflow.next_dataset_index(dataset_dir, MODEL_ID) == 1


def test_next_dataset_index_after_full_run():
    with tempfile.TemporaryDirectory(prefix="content-nextindex-") as tmp:
        dataset_dir = Path(tmp)
        # a first expand run: anchor _00 plus variations _01.._08
        for idx in range(9):
            (dataset_dir / f"{MODEL_ID}_{idx:02d}.jpg").write_bytes(b"\xff\xd8\xff stub")
        assert queue_workflow.next_dataset_index(dataset_dir, MODEL_ID) == 9


def test_next_dataset_index_hole_uses_max_plus_one():
    with tempfile.TemporaryDirectory(prefix="content-nextindex-") as tmp:
        dataset_dir = Path(tmp)
        # sparse: only _00 and _05 exist (e.g. deleted variants in between)
        for idx in (0, 5):
            (dataset_dir / f"{MODEL_ID}_{idx:02d}.jpg").write_bytes(b"\xff\xd8\xff stub")
        assert queue_workflow.next_dataset_index(dataset_dir, MODEL_ID) == 6


def test_next_dataset_index_ignores_non_matching_stems():
    with tempfile.TemporaryDirectory(prefix="content-nextindex-") as tmp:
        dataset_dir = Path(tmp)
        # other model's image and a stray notes file must not count
        (dataset_dir / "other_01.jpg").write_bytes(b"\xff\xd8\xff stub")
        (dataset_dir / "valery23_notes.txt").write_bytes(b"notes")
        assert queue_workflow.next_dataset_index(dataset_dir, MODEL_ID) == 1


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if not (name.startswith("test_") and callable(function)):
            continue
        try:
            function()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}: {exc!r}")
        else:
            print(f"PASS {name}")
    raise SystemExit(1 if failures else 0)
