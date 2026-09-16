"""Reference-persistence unit tests (defect: the UI reference selection lived
only in client state and was lost on F5). Hard persistence = the "reference"
key of models/<id>/character.json, written by main.set_model_reference.

Checks:
- the helper writes the "reference" key and preserves every other field;
- a second write (switch) rewrites only the key;
- 404 for a file that does not exist on disk (character.json untouched);
- 400 for an empty/whitespace filename;
- the endpoint 404s for an unknown model and returns
  {"ok": True, "reference": ...} on the happy path (a dataset/images
  filename is accepted too).

pytest-compatible; pytest is not required.

Run either way:
    python -m pytest tests/test_reference_persistence.py -q
    python tests/test_reference_persistence.py

WORKSPACE is pointed at a tempfile directory BEFORE management.main is
imported (same protocol as tests/test_dual_mode.py); all paths below are
derived from main.WORKSPACE so the tests never depend on the real layout.
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

_TMP_WORKSPACE = tempfile.TemporaryDirectory(prefix="ref-persist-ws-")
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

MODEL_ID = "refmodel"  # satisfies ^[a-z0-9][a-z0-9-]{1,62}$
CHARACTER_PATH = main.WORKSPACE / "models" / MODEL_ID / "character.json"


def _run(coro):
    return asyncio.run(coro)


def _make_model() -> dict:
    """models/<id>/{character.json, reference/candidates/cand_a.jpg,
    dataset/images/ds_0001.jpg} in the temp WORKSPACE. Returns the original
    character dict (for the field-preservation assertions)."""
    model_dir = main.WORKSPACE / "models" / MODEL_ID
    (model_dir / "reference" / "candidates").mkdir(parents=True, exist_ok=True)
    (model_dir / "dataset" / "images").mkdir(parents=True, exist_ok=True)
    (model_dir / "reference" / "candidates" / "cand_a.jpg").write_bytes(b"\xff\xd8\xff stub jpg")
    (model_dir / "dataset" / "images" / "ds_0001.jpg").write_bytes(b"\xff\xd8\xff stub jpg")
    character = {
        "name": "Valery",
        "age": 23,
        "height_cm": 178,
        "identity": "fictional adult AI model",
        "appearance": {"hair": "long natural hair", "eyes": "light eyes"},
        "body": "athletic adult build",
        "style": "editorial",
        "reference_look": "calm gaze",
    }
    main._atomic_write_json(CHARACTER_PATH, character)
    return character


def _read_character() -> dict:
    return json.loads(CHARACTER_PATH.read_text(encoding="utf-8"))


def _write_registry(model_id: str) -> None:
    reg = main.WORKSPACE / "models" / "registry.json"
    reg.write_text(
        json.dumps({"version": 1, "models": [{"id": model_id, "name": "Valery", "age": 23, "active": True}]}),
        encoding="utf-8",
    )


def test_set_model_reference_writes_key_and_preserves_fields():
    original = _make_model()
    saved = main.set_model_reference(main.WORKSPACE, MODEL_ID, "cand_a.jpg")
    assert saved == "cand_a.jpg"
    character = _read_character()
    assert character["reference"] == "cand_a.jpg"
    for key, value in original.items():  # nothing else may be touched
        assert character[key] == value, f"field {key!r} was modified"


def test_set_model_reference_switch_rewrites_only_reference():
    _make_model()
    main.set_model_reference(main.WORKSPACE, MODEL_ID, "cand_a.jpg")
    main.set_model_reference(main.WORKSPACE, MODEL_ID, "ds_0001.jpg")  # dataset/images accepted
    character = _read_character()
    assert character["reference"] == "ds_0001.jpg"
    assert character["name"] == "Valery"
    assert character["appearance"] == {"hair": "long natural hair", "eyes": "light eyes"}


def test_set_model_reference_missing_file_404():
    _make_model()
    try:
        main.set_model_reference(main.WORKSPACE, MODEL_ID, "does-not-exist.jpg")
    except main.HTTPException as exc:
        assert exc.status_code == 404, f"expected 404, got {exc.status_code}"
    else:
        raise AssertionError("HTTPException(404) expected for missing file")
    assert "reference" not in _read_character()  # character.json untouched on failure


def test_set_model_reference_invalid_file_400():
    _make_model()
    for bad in ("", "   "):
        try:
            main.set_model_reference(main.WORKSPACE, MODEL_ID, bad)
        except main.HTTPException as exc:
            assert exc.status_code == 400, f"{bad!r}: expected 400, got {exc.status_code}"
        else:
            raise AssertionError(f"HTTPException(400) expected for {bad!r}")


def test_endpoint_unknown_model_404():
    _make_model()
    try:
        _run(main.update_model_reference("ghostmodel", {"file": "cand_a.jpg"}))
    except main.HTTPException as exc:
        assert exc.status_code == 404, f"expected 404, got {exc.status_code}"
    else:
        raise AssertionError("HTTPException(404) expected for unknown model")


def test_endpoint_happy_path_returns_ok_and_reference():
    _make_model()
    _write_registry(MODEL_ID)
    result = _run(main.update_model_reference(MODEL_ID, {"file": "ds_0001.jpg"}))
    assert result == {"ok": True, "reference": "ds_0001.jpg"}
    assert _read_character()["reference"] == "ds_0001.jpg"


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
