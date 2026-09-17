"""Import-safe tests for management.main._validate_group_payload (feature 007,
group-photos): the pure validation helper behind POST /api/jobs/group.
Covers the happy path passthrough, model-count/uniqueness 422s, unknown-model
404, missing-LoRA 400, caption rules, the prompt/scene_id XOR, layout/mode
allowlists, scene resolution from models/library.json and the post-name
collision 409. pytest-compatible; pytest is not required.

Run either way:
    python -m pytest tests/test_group_validation.py -q
    python tests/test_group_validation.py

WORKSPACE is pointed at a tempfile directory BEFORE management.main is
imported (the module resolves WORKSPACE and mounts /static at import time),
so the tests never depend on the real /workspace layout. The registry, LoRA
stubs and scene library are written into the temp WORKSPACE after import —
main reads registry.json/library.json from disk on every call.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # content/
sys.path.insert(0, str(REPO_ROOT))

_TMP_WORKSPACE = tempfile.TemporaryDirectory(prefix="group-ws-")
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


def _model_entry(model_id: str, lora: str, *, active: bool) -> dict:
    return {
        "id": model_id,
        "name": model_id,
        "age": 23,
        "height_cm": 178,
        "tags": [],
        "active": active,
        "created": "2026-01-01T00:00:00Z",
        "lora": lora,
        "groups": [],
    }


# Fixture (written after import; main re-reads the registry/library from disk
# on every call). m3 exists in the registry but its LoRA file is absent from
# LORAS_DIR — that is the missing-LoRA case.
main.REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
main.REGISTRY_PATH.write_text(
    json.dumps(
        {
            "version": 1,
            "models": [
                _model_entry("m1", "m1.safetensors", active=True),
                _model_entry("m2", "m2.safetensors", active=False),
                _model_entry("m3", "m3.safetensors", active=False),
            ],
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
main.LORAS_DIR.mkdir(parents=True, exist_ok=True)
(main.LORAS_DIR / "m1.safetensors").write_bytes(b"")  # empty stubs: existence is what matters
(main.LORAS_DIR / "m2.safetensors").write_bytes(b"")
SCENE_TEXT = "a quiet studio shot"
main.LIBRARY_PATH.write_text(
    json.dumps({"version": 1, "scenes": [{"id": "sc1", "name": "Studio",
                                          "mode": "post", "text": SCENE_TEXT,
                                          "tags": []}]}),
    encoding="utf-8",
)


def _valid_payload(**overrides) -> dict:
    payload = {
        "models": ["m1", "m2"],
        "prompt": "p",
        "layout": "grid2",
        "mode": "private",
        "caption": "c",
        "seed": 5,
        "lora_strength": 0.9,
        "name": "grp1",
    }
    payload.update(overrides)
    return payload


def _expect_http_status(payload: dict, status: int) -> main.HTTPException:
    try:
        main._validate_group_payload(payload)
    except main.HTTPException as exc:
        assert exc.status_code == status, f"expected {status}, got {exc.status_code}"
        return exc
    raise AssertionError(
        f"HTTPException({status}) expected, _validate_group_payload returned normally"
    )


def test_valid_group_payload_passthrough():
    plan = main._validate_group_payload(_valid_payload())
    assert plan["ids"] == ["m1", "m2"]
    assert [entry["id"] for entry in plan["entries"]] == ["m1", "m2"]
    assert plan["prompt_text"] == "p"
    assert plan["layout"] == "grid2"
    assert plan["mode"] == "private"
    assert plan["caption"] == "c"
    assert plan["seed"] == 5
    assert plan["strength"] == 0.9
    assert plan["name"] == "grp1"
    assert plan["negative"] == ""


def test_more_than_five_models_422():
    exc = _expect_http_status(_valid_payload(models=["m1", "m2", "m3", "m4", "m5", "m6"]), 422)
    assert "1..5" in exc.detail, exc.detail


def test_duplicate_model_ids_422():
    exc = _expect_http_status(_valid_payload(models=["m1", "m1"]), 422)
    assert "unique" in exc.detail, exc.detail


def test_empty_models_list_422():
    _expect_http_status(_valid_payload(models=[]), 422)


def test_unknown_model_404():
    exc = _expect_http_status(_valid_payload(models=["zzz"]), 404)
    assert exc.detail == "model not found", exc.detail


def test_member_without_lora_file_400():
    # m3 is in the registry but m3.safetensors is absent from LORAS_DIR
    exc = _expect_http_status(_valid_payload(models=["m1", "m3"]), 400)
    assert "LoRA" in exc.detail, exc.detail


def test_empty_caption_400():
    _expect_http_status(_valid_payload(caption=""), 400)


def test_missing_caption_key_400():
    payload = _valid_payload()
    payload.pop("caption")
    _expect_http_status(payload, 400)


def test_no_prompt_no_scene_id_400():
    payload = _valid_payload()
    payload.pop("prompt")  # no scene_id either
    _expect_http_status(payload, 400)


def test_prompt_and_scene_id_together_400():
    _expect_http_status(_valid_payload(scene_id="sc1"), 400)  # prompt "p" still set


def test_invalid_layout_422():
    _expect_http_status(_valid_payload(layout="weird"), 422)


def test_invalid_mode_422():
    _expect_http_status(_valid_payload(mode="secret"), 422)


def test_scene_id_resolves_text_from_library():
    payload = _valid_payload()
    payload.pop("prompt")
    payload["scene_id"] = "sc1"
    plan = main._validate_group_payload(payload)
    assert plan["prompt_text"] == SCENE_TEXT


def test_name_collision_409():
    (main.MODELS_DIR / "m1" / "posts" / "grp2").mkdir(parents=True)
    exc = _expect_http_status(_valid_payload(name="grp2", mode="public"), 409)
    assert exc.detail == "post name already exists", exc.detail


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
