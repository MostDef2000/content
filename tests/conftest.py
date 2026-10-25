"""Shared fixtures for the content test suite (issue #17 two-root model).

management.main freezes its path constants at import time (and mounts
StaticFiles on WORKSPACE/management/static), so the two-root layout is
exercised by a fixture that sets BOTH env vars to distinct directories and
reloads the module; the previous module state is restored afterwards so the
per-file _bind_paths() rebinding in the other test files keeps working.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent  # content/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Module-level path constants of management.main that the reload re-resolves
# (and that must be restored afterwards for the sibling test files).
_PATH_CONSTANTS = (
    "WORKSPACE",
    "DATA_ROOT",
    "COMFY_URL",
    "JOBS_DIR",
    "REPO_MODELS_DIR",
    "REGISTRY_PATH",
    "LIBRARY_PATH",
    "LORAS_DIR",
    "TRASH_DIR",
    "DATASET_TRASH",
)


@pytest.fixture()
def two_roots(tmp_path, monkeypatch):
    """Point WORKSPACE (tracked checkout) and VALERY_DATA_ROOT (data volume)
    at two DISTINCT tmp dirs and reload management.main so its import-time
    constants re-resolve. Yields (workspace, data_root). Per-model DATA lands
    under data_root/models/<id>/ (prod-mirroring "models" segment) while the
    tracked configs sit under workspace/models/<id>/ — both roots carry a
    "models" segment but they are DIFFERENT roots (#17). Restores the previous
    env + module constants + jobs dict in teardown."""
    import importlib

    import management.main as main  # already imported by the test modules; lazy here

    workspace = tmp_path / "workspace"
    data_root = tmp_path / "data"
    (workspace / "management" / "static").mkdir(parents=True)  # StaticFiles mount on reload
    data_root.mkdir()

    prev_env = {
        key: os.environ.get(key)
        for key in ("WORKSPACE", "VALERY_DATA_ROOT", "VALERY_REGISTRY_PATH")
    }
    prev_paths = {name: getattr(main, name) for name in _PATH_CONSTANTS}
    prev_jobs = dict(main.jobs)

    monkeypatch.setenv("WORKSPACE", str(workspace))
    monkeypatch.setenv("VALERY_DATA_ROOT", str(data_root))
    importlib.reload(main)
    try:
        yield workspace, data_root
    finally:
        for key, value in prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for name, value in prev_paths.items():
            setattr(main, name, value)
        main.jobs.clear()
        main.jobs.update(prev_jobs)
