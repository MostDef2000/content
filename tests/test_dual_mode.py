"""Import-safe tests for management.main dual-mode privacy invariants
(feature 005): "posts-private" is never in the /files whitelist, and
serve_model_file blocks path traversal out of the whitelisted kind dir.
pytest-compatible; pytest is not required.

Run either way:
    python -m pytest tests/test_dual_mode.py -q
    python tests/test_dual_mode.py

WORKSPACE is pointed at a tempfile directory BEFORE management.main is
imported (the module resolves WORKSPACE and mounts /static at import time),
so the tests never depend on the real /workspace layout.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # content/
sys.path.insert(0, str(REPO_ROOT))

_TMP_WORKSPACE = tempfile.TemporaryDirectory(prefix="dual-mode-ws-")
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

MODEL_ID = "testmodel"  # satisfies ^[a-z0-9][a-z0-9-]{1,62}$


def _run(coro):
    return asyncio.run(coro)


def _make_model_files() -> None:
    """models/<id>/posts/p.jpg and models/<id>/posts-private/secret.jpg in the
    temp WORKSPACE (import of main must already have happened)."""
    posts = main.MODELS_DIR / MODEL_ID / "posts"
    private = main.MODELS_DIR / MODEL_ID / "posts-private"
    posts.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    (posts / "p.jpg").write_bytes(b"\xff\xd8\xff stub jpg")
    (private / "secret.jpg").write_bytes(b"\xff\xd8\xff secret jpg")


def test_posts_private_never_in_file_whitelist():
    assert "posts-private" not in main._FILE_KIND_DIRS
    # sanity: the public kinds are whitelisted, so the invariant is meaningful
    for kind in ("reference", "dataset", "posts"):
        assert kind in main._FILE_KIND_DIRS, kind


def test_serve_model_file_serves_whitelisted_kind():
    _make_model_files()
    response = _run(main.serve_model_file(MODEL_ID, "posts", "p.jpg"))
    assert isinstance(response, main.FileResponse)
    assert response.status_code == 200


def test_serve_model_file_blocks_traversal():
    _make_model_files()
    try:
        _run(main.serve_model_file(MODEL_ID, "posts", "../posts-private/secret.jpg"))
    except main.HTTPException as exc:
        assert exc.status_code == 400, f"expected 400, got {exc.status_code}"
    else:
        raise AssertionError("HTTPException(400) expected for ../ traversal into posts-private")


def test_serve_model_file_never_serves_posts_private_kind():
    _make_model_files()
    try:
        _run(main.serve_model_file(MODEL_ID, "posts-private", "secret.jpg"))
    except main.HTTPException as exc:
        assert exc.status_code == 404, f"expected 404, got {exc.status_code}"
    else:
        raise AssertionError("HTTPException(404) expected: posts-private must never be served")


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
