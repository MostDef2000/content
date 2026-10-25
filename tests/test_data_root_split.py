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
import json
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


def test_registry_path_env_moves_state_out_of_checkout(two_roots, monkeypatch):
    """Issue #1: VALERY_REGISTRY_PATH moves the runtime registry onto the data
    volume — out of the deploy checkout (deploys re-checkout WORKSPACE and
    must not clobber or be blocked by the registry). Without the env the
    tracked checkout path stays the fallback (local single-root runs)."""
    import importlib

    import management.main as main

    workspace, data_root = two_roots
    # Fallback (no env): tracked checkout path — covered in detail above.
    assert main.REGISTRY_PATH == workspace / "models" / "registry.json"

    reg_path = data_root / "runtime" / "registry.json"
    monkeypatch.setenv("VALERY_REGISTRY_PATH", str(reg_path))
    importlib.reload(main)
    assert main.REGISTRY_PATH == reg_path, main.REGISTRY_PATH
    assert reg_path.parent.is_dir(), "env-путь: parent каталог создаётся при старте"

    reg = {"version": 1, "models": [{"id": MODEL_ID, "name": "Split", "active": True}]}
    checkout_before = (workspace / "models" / "registry.json").read_text(encoding="utf-8")
    _run(main._save_registry(reg))
    assert reg_path.is_file(), reg_path
    ids = {str(m.get("id")) for m in main._load_registry().get("models", [])}
    assert MODEL_ID in ids, ids
    # the checkout copy is untouched by writes to the env path
    assert (workspace / "models" / "registry.json").read_text(encoding="utf-8") == checkout_before


def test_registry_path_env_migrates_checkout_copy(two_roots, monkeypatch, capsys):
    """Issue #1 migration guard: first start with VALERY_REGISTRY_PATH set and
    an empty new path copies the checkout-resident registry to the data volume
    instead of letting _ensure_registry re-seed from the legacy layout."""
    import importlib

    import management.main as main

    workspace, data_root = two_roots
    checkout_reg = workspace / "models" / "registry.json"
    assert checkout_reg.is_file(), "fixture seeding created the checkout registry"
    seed = checkout_reg.read_text(encoding="utf-8")

    reg_path = data_root / "runtime" / "registry.json"
    assert not reg_path.exists()
    monkeypatch.setenv("VALERY_REGISTRY_PATH", str(reg_path))
    importlib.reload(main)
    assert main.REGISTRY_PATH == reg_path
    assert reg_path.is_file(), "migration guard copied the checkout registry"
    assert reg_path.read_text(encoding="utf-8") == seed
    # _ensure_registry must NOT have fired its legacy branch (no valery23 seed)
    assert "valery23" not in json.loads(reg_path.read_text(encoding="utf-8"))


def test_model_stats_counts_dataset_images_not_cache_pairs(two_roots):
    """Issue #1: stats.dataset counts dataset images, not every file in
    dataset/images — kohya trainer caches (.npz latent/TE pairs) and .txt
    captions live there too, and the bare file count showed 368 for a
    87-image prod dataset (ui stats must match what train_lora.sh trains on)."""
    import management.main as main

    two_roots
    images_dir = main._dataset_image_dir(MODEL_ID)
    images_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "a.jpg",
        "b.png",
        "a.txt",
        "b.txt",
        f"{MODEL_ID}_00_1024x1280_flux.npz",
        f"{MODEL_ID}_00_flux_te.npz",
    ):
        (images_dir / name).write_bytes(b"x")
    stats = main._model_stats(MODEL_ID, None)
    assert stats["dataset"] == 2, stats


if __name__ == "__main__":
    # Direct runner convenience: the two_roots fixture needs pytest, so this
    # file is pytest-only.
    raise SystemExit("run with: python -m pytest tests/test_data_root_split.py -q")
