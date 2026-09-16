"""Post soft-delete unit tests (DELETE /api/models/<id>/posts/<name>).

The post folder models/<id>/<posts|posts-private>/<name>/ is moved (atomic
os.replace) into runtime/trash/posts/<kind>-<name>-<unixts>/ — recoverable
manually, like the model/dataset soft deletes. The pure helper
main.delete_model_post does kind/name validation + the move; the endpoint
adds _model_or_404 and the confirm-must-equal-name check (wipe pattern).

Checks:
- happy path: the folder lands in runtime/trash/posts/ with its contents
  intact, the original path is gone, and the helper returns the name;
- kind="private" moves models/<id>/posts-private/<name> the same way;
- 404 for a post folder that does not exist on disk (nothing is moved);
- 400 when the endpoint confirm does not equal the post name (folder
  untouched), then {"ok": True, "deleted": ..., "kind": ...} on the happy path;
- 400/422 for an invalid kind (folder untouched);
- 400 for a traversal name ("../x") — the post it would resolve to survives.

pytest-compatible; pytest is not required.

Run either way:
    python -m pytest tests/test_post_delete.py -q
    python tests/test_post_delete.py

WORKSPACE is pointed at a tempfile directory BEFORE management.main is
imported (same protocol as tests/test_reference_persistence.py); all paths
below are derived from main.WORKSPACE so the tests never depend on the real
layout.
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

_TMP_WORKSPACE = tempfile.TemporaryDirectory(prefix="post-delete-ws-")
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

MODEL_ID = "postmodel"  # satisfies ^[a-z0-9][a-z0-9-]{1,62}$
MODEL_DIR = main.WORKSPACE / "models" / MODEL_ID
TRASH_POSTS = main.WORKSPACE / "runtime" / "trash" / "posts"
PHOTO_BYTES = b"\xff\xd8\xff stub jpg"


def _run(coro):
    return asyncio.run(coro)


def _make_post(kind: str, name: str) -> Path:
    """models/<id>/<posts|posts-private>/<name>/ with photo.jpg + caption.txt."""
    kind_dir = "posts" if kind == "public" else "posts-private"
    post_dir = MODEL_DIR / kind_dir / name
    post_dir.mkdir(parents=True, exist_ok=True)
    (post_dir / "photo.jpg").write_bytes(PHOTO_BYTES)
    (post_dir / "caption.txt").write_text("caption of the post\n", encoding="utf-8")
    return post_dir


def _write_registry(model_id: str) -> None:
    reg = main.WORKSPACE / "models" / "registry.json"
    reg.write_text(
        json.dumps({"version": 1, "models": [{"id": model_id, "name": "Valery", "age": 23, "active": True}]}),
        encoding="utf-8",
    )


def _trash_matches(prefix: str) -> list[Path]:
    return sorted(TRASH_POSTS.glob(f"{prefix}-*")) if TRASH_POSTS.is_dir() else []


def test_delete_model_post_happy_path_public():
    post_dir = _make_post("public", "post_pub")
    saved = main.delete_model_post(main.WORKSPACE, MODEL_ID, "post_pub", "public")
    assert saved == "post_pub"
    assert not post_dir.exists()  # the original folder is gone
    moved = _trash_matches("public-post_pub")
    assert len(moved) == 1, f"expected exactly one trash folder, got {moved}"
    assert (moved[0] / "photo.jpg").read_bytes() == PHOTO_BYTES  # contents intact
    assert (moved[0] / "caption.txt").read_text(encoding="utf-8") == "caption of the post\n"


def test_delete_model_post_private_kind():
    post_dir = _make_post("private", "pp_1")
    saved = main.delete_model_post(main.WORKSPACE, MODEL_ID, "pp_1", "private")
    assert saved == "pp_1"
    assert not post_dir.exists()
    moved = _trash_matches("private-pp_1")
    assert len(moved) == 1, f"expected exactly one trash folder, got {moved}"
    assert (moved[0] / "photo.jpg").read_bytes() == PHOTO_BYTES


def test_delete_model_post_missing_post_404():
    (MODEL_DIR / "posts").mkdir(parents=True, exist_ok=True)  # empty folder, no posts
    try:
        main.delete_model_post(main.WORKSPACE, MODEL_ID, "ghost_post", "public")
    except main.HTTPException as exc:
        assert exc.status_code == 404, f"expected 404, got {exc.status_code}"
    else:
        raise AssertionError("HTTPException(404) expected for missing post")
    assert _trash_matches("public-ghost_post") == []  # nothing was moved


def test_endpoint_confirm_mismatch_400_then_happy_path():
    _write_registry(MODEL_ID)
    post_dir = _make_post("public", "post_ep")
    try:
        _run(main.delete_post(MODEL_ID, "post_ep", kind="public", confirm="wrong"))
    except main.HTTPException as exc:
        assert exc.status_code == 400, f"expected 400, got {exc.status_code}"
    else:
        raise AssertionError("HTTPException(400) expected for confirm mismatch")
    assert post_dir.is_dir()  # the folder is untouched on failure
    result = _run(main.delete_post(MODEL_ID, "post_ep", kind="public", confirm="post_ep"))
    assert result == {"ok": True, "deleted": "post_ep", "kind": "public"}
    assert not post_dir.exists()
    assert len(_trash_matches("public-post_ep")) == 1


def test_delete_model_post_invalid_kind():
    post_dir = _make_post("public", "post_kind")
    for bad in ("tasteful", "", "PRIVATE"):
        try:
            main.delete_model_post(main.WORKSPACE, MODEL_ID, "post_kind", bad)
        except main.HTTPException as exc:
            assert exc.status_code in (400, 422), f"{bad!r}: expected 400/422, got {exc.status_code}"
        else:
            raise AssertionError(f"HTTPException(400/422) expected for kind={bad!r}")
    assert post_dir.is_dir()  # nothing was moved


def test_delete_model_post_traversal_name_400():
    sibling = _make_post("public", "x")  # the post a naive basename of "../x" would hit
    try:
        main.delete_model_post(main.WORKSPACE, MODEL_ID, "../x", "public")
    except main.HTTPException as exc:
        assert exc.status_code == 400, f"expected 400, got {exc.status_code}"
    else:
        raise AssertionError("HTTPException(400) expected for traversal name")
    assert sibling.is_dir()  # the sibling post survived
    assert _trash_matches("public-x") == []


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
