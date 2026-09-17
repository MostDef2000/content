"""Import-safe tests for management.main dataset wipe (feature 006,
clear-dataset): confirm guard, active-job 409 (solo + group jobs), soft-move
of the whole images/ dir into runtime/trash/dataset/ with a manifest copy,
manifest reset, empty-dir no-op and trash-name collision suffixing.
pytest-compatible; pytest is not required.

Run either way:
    python -m pytest tests/test_clear_dataset.py -q
    python tests/test_clear_dataset.py

WORKSPACE is pointed at a tempfile directory BEFORE management.main is
imported (the module resolves WORKSPACE and mounts /static at import time),
so the tests never depend on the real /workspace layout. The per-model
registry/model dirs are written into the temp WORKSPACE after import —
main reads registry.json from disk on every call.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # content/
sys.path.insert(0, str(REPO_ROOT))

_TMP_WORKSPACE = tempfile.TemporaryDirectory(prefix="wipe-ws-")
# StaticFiles(directory=...) is mounted at import time and requires the dir.
Path(_TMP_WORKSPACE.name, "management", "static").mkdir(parents=True, exist_ok=True)

_PREV_WORKSPACE = os.environ.get("WORKSPACE")
os.environ["WORKSPACE"] = _TMP_WORKSPACE.name
try:
    import management.main as main  # noqa: E402
finally:
    if _PREV_WORKSPACE is None:
        os.environ.pop("WORKSPACE", None)
    else:
        os.environ["WORKSPACE"] = _PREV_WORKSPACE


# In a shared pytest process management.main's module-level paths freeze at
# first import (first file's temp WORKSPACE). Re-point them at THIS file's
# temp WORKSPACE before the seeds below and re-bind per module at run time.
def _bind_paths() -> None:
    ws = Path(_TMP_WORKSPACE.name)
    main.WORKSPACE = ws
    main.JOBS_DIR = ws / "runtime" / "jobs"
    main.MODELS_DIR = ws / "models"
    main.REGISTRY_PATH = ws / "models" / "registry.json"
    main.LIBRARY_PATH = ws / "models" / "library.json"
    main.LORAS_DIR = ws / "runtime" / "models" / "loras"
    main.TRASH_DIR = ws / "runtime" / "trash"
    main.DATASET_TRASH = main.TRASH_DIR / "dataset"


_bind_paths()  # direct runner: single file per process

try:
    import pytest

    @pytest.fixture(scope="module", autouse=True)
    def _bind_file_paths():
        _bind_paths()
        yield
except ModuleNotFoundError:  # direct runner: pytest is optional
    pass

MODEL_ID = "tst01"  # satisfies ^[a-z0-9][a-z0-9-]{1,62}$
FIXED_TS = 1767225600  # 2026-01-01T00:00:00Z — frozen time.time for the collision test

OLD_MANIFEST = {
    "version": 1,
    "images": [
        {"filename": "a.png", "tag": "tasteful", "caption": "caption a",
         "created": "2026-01-01T00:00:00Z"},
        {"filename": "b.png", "tag": "explicit", "caption": "caption b",
         "created": "2026-01-01T00:00:00Z"},
    ],
}


def _run(coro):
    return asyncio.run(coro)


def _seed_dataset() -> None:
    """(Re)create models/tst01/dataset with a.png, b.png, a.txt, b.txt,
    extra.json and the valid 2-entry manifest. Every test re-seeds first, so
    tests are independent of execution order."""
    image_dir = main._dataset_image_dir(MODEL_ID)
    image_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ("a.png", b"fake-png-a"),
        ("b.png", b"fake-png-b"),
        ("a.txt", b"caption a"),
        ("b.txt", b"caption b"),
        ("extra.json", b"{}"),
    ):
        (image_dir / name).write_bytes(data)
    main._save_manifest(MODEL_ID, OLD_MANIFEST)


def _expect_http_status(status: int, coro) -> main.HTTPException:
    try:
        _run(coro)
    except main.HTTPException as exc:
        assert exc.status_code == status, f"expected {status}, got {exc.status_code}"
        return exc
    raise AssertionError(f"HTTPException({status}) expected, wipe returned normally")


# Registry fixture: a single model "tst01" (import-time bootstrap seeded a
# default valery23 registry into the temp WORKSPACE — overwrite it; main
# re-reads registry.json from disk on every call).
main.REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
main.REGISTRY_PATH.write_text(
    json.dumps(
        {
            "version": 1,
            "models": [
                {
                    "id": MODEL_ID,
                    "name": "Tst",
                    "age": 23,
                    "height_cm": 178,
                    "tags": [],
                    "active": True,
                    "created": "2026-01-01T00:00:00Z",
                    "lora": "tst01.safetensors",
                    "groups": [],
                }
            ],
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
_seed_dataset()


def test_wipe_requires_confirm_equal_to_model_id():
    _seed_dataset()
    exc = _expect_http_status(400, main.wipe_model_dataset(MODEL_ID, confirm="nope"))
    assert exc.detail == "confirm must equal the model id", exc.detail
    # nothing was moved
    assert (main._dataset_image_dir(MODEL_ID) / "a.png").is_file()


def test_wipe_moves_dataset_to_trash_and_resets_manifest():
    _seed_dataset()
    result = _run(main.wipe_model_dataset(MODEL_ID, confirm=MODEL_ID))
    assert result["ok"] is True
    assert result["model_id"] == MODEL_ID
    assert result["trash_path"] is not None

    trash_dir = Path(result["trash_path"])
    assert trash_dir.is_dir()
    assert trash_dir.parent == main.DATASET_TRASH
    assert trash_dir.name.startswith(f"{MODEL_ID}-wipe-"), trash_dir.name
    # all five files plus a copy of the pre-wipe manifest
    assert {p.name for p in trash_dir.iterdir()} == {
        "a.png", "b.png", "a.txt", "b.txt", "extra.json", "manifest.json",
    }
    assert json.loads((trash_dir / "manifest.json").read_text(encoding="utf-8")) == OLD_MANIFEST

    # images/ is recreated and empty, manifest reset
    image_dir = main._dataset_image_dir(MODEL_ID)
    assert image_dir.is_dir()
    assert list(image_dir.iterdir()) == []
    assert main._load_manifest(MODEL_ID) == {"version": 1, "images": []}

    assert result["deleted"] == {"images": 2, "captions": 2, "other": 1}
    assert result["manifest"] == {"version": 1, "images": []}


def test_wipe_blocked_by_active_solo_job_then_allowed_after_done():
    _seed_dataset()
    main.jobs["jx"] = {"status": "running", "model_id": MODEL_ID}
    try:
        exc = _expect_http_status(409, main.wipe_model_dataset(MODEL_ID, confirm=MODEL_ID))
        assert exc.detail == "model has an active job", exc.detail
        assert (main._dataset_image_dir(MODEL_ID) / "a.png").is_file()  # untouched
        # once the job finishes, the same wipe goes through
        main.jobs["jx"]["status"] = "done"
        result = _run(main.wipe_model_dataset(MODEL_ID, confirm=MODEL_ID))
        assert result["ok"] is True
        assert result["trash_path"] is not None
    finally:
        main.jobs.pop("jx", None)


def test_wipe_blocked_by_group_job_listing_model():
    _seed_dataset()
    main.jobs["jx"] = {"status": "queued", "models": [MODEL_ID]}
    try:
        exc = _expect_http_status(409, main.wipe_model_dataset(MODEL_ID, confirm=MODEL_ID))
        assert exc.detail == "model has an active job", exc.detail
        assert (main._dataset_image_dir(MODEL_ID) / "a.png").is_file()  # untouched
    finally:
        main.jobs.pop("jx", None)


def test_wipe_empty_dataset_is_noop():
    _seed_dataset()
    _run(main.wipe_model_dataset(MODEL_ID, confirm=MODEL_ID))  # consume the seed
    result = _run(main.wipe_model_dataset(MODEL_ID, confirm=MODEL_ID))
    assert result["ok"] is True
    assert result["trash_path"] is None  # no trash folder for an empty dir
    assert result["deleted"] == {"images": 0, "captions": 0, "other": 0}
    assert main._load_manifest(MODEL_ID) == {"version": 1, "images": []}


def test_wipe_trash_name_collision_gets_suffix():
    _seed_dataset()
    real_time = main.time.time
    main.time.time = lambda: FIXED_TS  # freeze so two wipes share one base name
    try:
        first = _run(main.wipe_model_dataset(MODEL_ID, confirm=MODEL_ID))
        _seed_dataset()  # second wipe needs files again
        second = _run(main.wipe_model_dataset(MODEL_ID, confirm=MODEL_ID))
    finally:
        main.time.time = real_time
    assert Path(first["trash_path"]).name == f"{MODEL_ID}-wipe-{FIXED_TS}"
    assert Path(second["trash_path"]).name == f"{MODEL_ID}-wipe-{FIXED_TS}-2"


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
