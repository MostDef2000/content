"""Issue #17 acceptance: the per-model DATA/config two-root split.

POST /api/models (create_model) must lay a fresh model out across BOTH roots:

  * per-model DATA (dataset/images, posts, posts-private,
    reference/candidates, lora/output) under VALERY_DATA_ROOT/models/<id>/ —
    WITH the "models" segment, 1:1 with the prod host tree
    /opt/sea-speed-worker/valery/models/<id>;
  * tracked configs (character.json, prompt_profile.json) under
    WORKSPACE/models/<id>/, registry at WORKSPACE/models/registry.json;
  * the checkout config dir holds ONLY the two json files (zero generated
    data lands in the checkout).

I.e. a freshly created model is fully functional (configs readable, data
dirs addressable, registry persisted) with zero compose/mount changes.

The two_roots fixture (tests/conftest.py) sets WORKSPACE and
VALERY_DATA_ROOT to distinct tmp dirs and reloads management.main, so the
import-time constants match the env — the issue's "env must win" semantics.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # content/
sys.path.insert(0, str(REPO_ROOT))

MODEL_ID = "split01"  # satisfies ^[a-z0-9][a-z0-9-]{1,62}$


def _run(coro):
    return asyncio.run(coro)


def test_create_model_splits_data_and_config_roots(two_roots):
    import management.main as main

    workspace, data_root = two_roots

    entry = _run(main.create_model({"id": MODEL_ID, "name": "Split", "age": 23}))
    assert entry["id"] == MODEL_ID, entry

    # --- DATA root: the per-model dirs, WITH the "models" segment ----------
    model_data = data_root / "models" / MODEL_ID
    for sub in (
        "dataset/images",
        "posts",
        "posts-private",
        "reference/candidates",
        "lora/output",
    ):
        assert (model_data / sub).is_dir(), f"missing data dir under VALERY_DATA_ROOT: {sub}"
    assert main._model_dir(MODEL_ID) == model_data

    # --- Checkout: tracked configs only ------------------------------------
    model_config = workspace / "models" / MODEL_ID
    assert (model_config / "character.json").is_file()
    assert (model_config / "prompt_profile.json").is_file()
    assert main._model_config_dir(MODEL_ID) == model_config
    # zero generated data may land in the checkout config dir
    assert sorted(p.name for p in model_config.iterdir()) == [
        "character.json",
        "prompt_profile.json",
    ], sorted(p.name for p in model_config.iterdir())
    # no stray data tree under the checkout's models/ for this model
    assert not (workspace / "models" / MODEL_ID / "dataset").exists()

    # --- Tracked registry/library stay in the checkout ---------------------
    assert main.REGISTRY_PATH == workspace / "models" / "registry.json"
    assert main.REGISTRY_PATH.is_file()
    ids = {str(m.get("id")) for m in main._load_registry().get("models", [])}
    assert MODEL_ID in ids, ids

    # --- The fresh model is fully functional across both roots -------------
    character = main._load_character(MODEL_ID)  # reads the checkout config
    assert character["name"] == "Split", character
    stats = main._model_stats(MODEL_ID, None)  # reads the DATA root
    assert stats == {"candidates": 0, "dataset": 0, "posts": 0, "lora_file": False}, stats


if __name__ == "__main__":
    # Direct runner convenience: the two_roots fixture needs pytest, so this
    # file is pytest-only.
    raise SystemExit("run with: python -m pytest tests/test_data_root_split.py -q")
