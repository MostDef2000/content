"""Import-safe tests for the post-tab v2 server seams (specs/008-post-tab-v2,
issue #2 items 10/11/14; AC-1..AC-6 of the spec's Measurable acceptance):

AC-1/AC-2/AC-3 (п.14, per-post model binding): job_post with a valid
payload.model_id binds ONLY that post — the cmd carries --model <id>, the job
record reflects it, the LoRA is resolved from the selected model's registry
slot (a missing slot file 400s naming the SELECTED model's file), the prompt
profile comes from the selected model, and the registry active flag stays
untouched; an unknown model_id 404s with the «модель не найдена» text and
launches nothing; a payload without model_id keeps the back-compat behaviour
(the active model — the cmd has carried --model <active id> since before this
feature, so the flag itself is not the compat surface, the selection is).

AC-4 (guardrail of the selected model): the injected guardrail phrase follows
the SELECTED model's character.json age (m2 is 30 → "30-year-old fictional"
in the cmd prompt, while the active 23-year m1 still gets "23-year-old").

AC-5 (п.11, scene meta): job_post resolves the scene display name once
(_scene_name) and passes --scene-id/--scene-name through the CLI; the real
queue_workflow parser accepts them and generate_post writes post.json with
{scene_id, scene_name, model_id, engine, mode, seed, created_at}; a Namespace
without the scene keys (older direct callers) still produces post.json with
the scene fields null; list_posts/list_posts_private expose the meta and
report scene_name null for legacy posts (no post.json / corrupt file).

AC-6 (п.14, engine per model): GET /api/models (list_models) carries an
"engine" on every entry — "sdxl" when the SDXL slot file is deployed, "flux"
when only the FLUX slot is, None when neither file exists.

pytest-compatible; pytest is not required.

Run either way:
    python -m pytest tests/test_post_model_override.py -q
    python tests/test_post_model_override.py

WORKSPACE is pointed at a tempfile directory BEFORE management.main is
imported (the module resolves WORKSPACE and mounts /static at import time),
so the tests never depend on the real /workspace layout. The registry, LoRA
stubs, characters, profiles and scene library are written into the temp
WORKSPACE after import — main reads them from disk on every call. The
endpoints are tested with main._launch monkeypatched (a spy), so no
docker/ComfyUI is touched; the generate_post leg of the chain runs with
queue_workflow.ROOT pointed at the SAME temp workspace and queue_and_wait
stubbed, which keeps the whole write/read cycle (cmd → parser → post.json →
list_posts) inside the sandbox.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # content/
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import queue_workflow  # noqa: E402

_TMP_WORKSPACE = tempfile.TemporaryDirectory(prefix="post-model-ws-")
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
# Issue #17 two-root model: per-model DATA (posts/loras/runtime) lives under
# DATA_ROOT; registry/library and the per-model configs stay in the checkout
# (WORKSPACE/models/).
def _bind_paths() -> None:
    ws = Path(_TMP_WORKSPACE.name)
    data = ws / "data"  # distinct VALERY_DATA_ROOT inside the same temp tree
    main.WORKSPACE = ws
    main.DATA_ROOT = data
    main.REPO_MODELS_DIR = ws / "models"
    main.JOBS_DIR = data / "runtime" / "jobs"
    main.REGISTRY_PATH = ws / "models" / "registry.json"
    main.LIBRARY_PATH = ws / "models" / "library.json"
    main.LORAS_DIR = data / "runtime" / "models" / "loras"
    main.TRASH_DIR = data / "runtime" / "trash"
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

MODEL_ID = "m1"  # active, SDXL slot deployed, age 23 (back-compat baseline)
OTHER_ID = "m2"  # NOT active, FLUX slot deployed, age 30 (the override target)
NO_SLOT_ID = "m3"  # no slot files at all (engine None in /api/models)
SCENE_ID = "s1"
SCENE_NAME = "Sunset pier"
_M2_FLUX_LORA = f"{OTHER_ID}-flux.safetensors"
_M1_SDXL_LORA = f"{MODEL_ID}.safetensors"


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# Fixtures (written after import; main re-reads registry/character/profile/
# library from disk on every call).
# --------------------------------------------------------------------------- #
main.REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)


def _registry_json(models: list[dict]) -> str:
    return json.dumps({"version": 1, "models": models}, ensure_ascii=False, indent=2) + "\n"


def _m1_entry(**overrides) -> dict:
    entry = {
        "id": MODEL_ID,
        "name": "Valery",
        "age": 23,
        "height_cm": 178,
        "tags": [],
        "active": True,
        "created": "2026-01-01T00:00:00Z",
        "lora": _M1_SDXL_LORA,
        "groups": [],
    }
    entry.update(overrides)
    return entry


def _m2_entry(**overrides) -> dict:
    entry = {
        "id": OTHER_ID,
        "name": "Mira",
        "age": 30,
        "height_cm": 170,
        "tags": [],
        "active": False,
        "created": "2026-01-02T00:00:00Z",
        "lora": f"{OTHER_ID}.safetensors",  # SDXL slot name, file NOT deployed
        "lora_flux": _M2_FLUX_LORA,  # FLUX slot name, file deployed below
        "groups": [],
    }
    entry.update(overrides)
    return entry


def _m3_entry(**overrides) -> dict:
    entry = {
        "id": NO_SLOT_ID,
        "name": "Nova",
        "age": 23,
        "height_cm": 175,
        "tags": [],
        "active": False,
        "created": "2026-01-03T00:00:00Z",
        "groups": [],
    }
    entry.update(overrides)
    return entry


_SEED_REGISTRY = [_m1_entry(), _m2_entry(), _m3_entry()]
main.REGISTRY_PATH.write_text(_registry_json(_SEED_REGISTRY), encoding="utf-8")

main.LORAS_DIR.mkdir(parents=True, exist_ok=True)
(main.LORAS_DIR / _M1_SDXL_LORA).write_bytes(b"")  # empty stubs: existence decides
(main.LORAS_DIR / _M2_FLUX_LORA).write_bytes(b"")

# character.json/prompt_profile.json are tracked configs → checkout models/<id>;
# the LoRA stubs live in the DATA root's runtime loras dir.
_m1_dir = main._model_config_dir(MODEL_ID)
_m2_dir = main._model_config_dir(OTHER_ID)
for directory in (_m1_dir, _m2_dir):
    directory.mkdir(parents=True, exist_ok=True)
(_m1_dir / "character.json").write_text(
    json.dumps({"name": "Valery", "age": 23, "appearance": {"face": "oval face", "build": "athletic"}}),
    encoding="utf-8",
)
(_m1_dir / "prompt_profile.json").write_text(
    json.dumps({"version": 1, "face": "oval face", "body": "athletic", "style": "", "negative": "", "scenes": {}}),
    encoding="utf-8",
)
(_m2_dir / "character.json").write_text(
    json.dumps({"name": "Mira", "age": 30, "appearance": {"face": "round face", "build": "slim"}}),
    encoding="utf-8",
)
(_m2_dir / "prompt_profile.json").write_text(
    json.dumps(
        {"version": 1, "face": "round face", "body": "slim", "style": "", "negative": "m2 profile negative", "scenes": {}}
    ),
    encoding="utf-8",
)
main.LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
main.LIBRARY_PATH.write_text(
    json.dumps(
        {
            "version": 1,
            "scenes": [
                {"id": SCENE_ID, "name": SCENE_NAME, "mode": "post", "text": "golden hour at the pier", "tags": []},
            ],
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

_SEED_REGISTRY_JSON = main.REGISTRY_PATH.read_text(encoding="utf-8")


def _with_registry(models: list[dict]):
    """Temporarily replace the registry with the given models list; the seed
    JSON is restored after the block (self-sufficient tests)."""
    import contextlib

    @contextlib.contextmanager
    def _patched():
        main.REGISTRY_PATH.write_text(_registry_json(models), encoding="utf-8")
        try:
            yield
        finally:
            main.REGISTRY_PATH.write_text(_SEED_REGISTRY_JSON, encoding="utf-8")

    return _patched()


class _LaunchSpy:
    """Stands in for main._launch: records (kind, cmd, model_id, job_id) and
    registers a queued job record, without touching docker/ComfyUI."""

    def __init__(self):
        self.calls: list[tuple[str, list[str], str | None, str]] = []

    async def __call__(self, kind: str, cmd: list[str], *, model_id: str | None = None) -> dict:
        job_id = f"{kind}-test-{len(self.calls) + 1}"
        self.calls.append((kind, cmd, model_id, job_id))
        main.jobs[job_id] = {
            "id": job_id,
            "kind": kind,
            "model_id": model_id,
            "status": "queued",
            "cmd": cmd,
        }
        return {"job_id": job_id, "status": "queued"}


def _with_spy(spy: _LaunchSpy):
    import contextlib

    @contextlib.contextmanager
    def _patched():
        original = main._launch
        main._launch = spy
        try:
            yield
        finally:
            main._launch = original

    return _patched()


def _cmd_flag(cmd: list[str], flag: str) -> str:
    """Value of a two-token flag in a captured cmd (KeyError when absent)."""
    return cmd[cmd.index(flag) + 1]


def _without_m2_flux_stub():
    """Temporarily remove m2's FLUX slot file (restored after the block) so a
    test can exercise the missing-LoRA path of the SELECTED model."""
    import contextlib

    @contextlib.contextmanager
    def _patched():
        stub = main.LORAS_DIR / _M2_FLUX_LORA
        stub.unlink(missing_ok=True)
        try:
            yield
        finally:
            stub.write_bytes(b"")

    return _patched()


# --------------------------------------------------------------------------- #
# AC-1 — job_post with a valid payload.model_id binds the selected model.
# --------------------------------------------------------------------------- #
def test_post_model_override_cmd_and_job_record():
    spy = _LaunchSpy()
    with _with_spy(spy):
        result = _run(
            main.job_post(
                {
                    "model_id": OTHER_ID,
                    "prompt": "a quiet studio shot",
                    "seed": 42,
                }
            )
        )
    assert len(spy.calls) == 1, spy.calls
    kind, cmd, launch_model_id, job_id = spy.calls[-1]
    assert kind == "post"
    # The cmd binds the SELECTED model, not the active one.
    assert _cmd_flag(cmd, "--model") == OTHER_ID, cmd
    # The job record reflects the selected model too.
    assert launch_model_id == OTHER_ID
    assert main.jobs[job_id]["model_id"] == OTHER_ID
    assert result == {"job_id": job_id, "status": "queued"}
    # Prompt profile comes from m2 (its profile negative is distinct).
    assert _cmd_flag(cmd, "--negative") == "m2 profile negative", cmd
    # Seed/engine as before (back-compat of the remaining fields).
    assert _cmd_flag(cmd, "--seed") == "42", cmd
    assert _cmd_flag(cmd, "--engine") == "flux", cmd  # server default engine
    # The active model in the registry is NOT touched: m1 stays active, m2
    # gains no active flag, and the resolver still returns m1.
    registry = json.loads(main.REGISTRY_PATH.read_text(encoding="utf-8"))
    entries = {str(entry["id"]): entry for entry in registry["models"]}
    assert entries[MODEL_ID].get("active") is True, entries[MODEL_ID]
    assert not entries[OTHER_ID].get("active"), entries[OTHER_ID]
    active = main._active_model()
    assert active is not None and active["id"] == MODEL_ID, active


def test_post_model_override_lora_resolved_from_selected_model():
    # LoRA resolution follows the SELECTED model's registry slot: with m2's
    # FLUX slot file removed (m1's SDXL stub still deployed, so an
    # active-model-based resolver could not produce this filename) a flux post
    # for m2 must 400 naming m2-flux.safetensors and launch nothing.
    spy = _LaunchSpy()
    with _without_m2_flux_stub(), _with_spy(spy):
        try:
            _run(main.job_post({"model_id": OTHER_ID, "prompt": "a quiet studio shot"}))
        except main.HTTPException as exc:
            assert exc.status_code == 400, exc.status_code
            assert _M2_FLUX_LORA in exc.detail, exc.detail
        else:
            raise AssertionError("HTTPException(400) expected for m2 flux post without its FLUX LoRA")
    assert spy.calls == [], "no job may be launched when the selected model's LoRA is missing"
    # Pure resolver: m2's flux engine consumes m2's registry slot.
    assert main._lora_name_for_engine(_m2_entry(), "flux") == _M2_FLUX_LORA
    assert main._lora_name_for_engine(_m2_entry(), "sdxl") == f"{OTHER_ID}.safetensors"


# --------------------------------------------------------------------------- #
# AC-2 — unknown model_id → 404 «модель не найдена», no launch.
# --------------------------------------------------------------------------- #
def test_post_unknown_model_id_404_no_launch():
    spy = _LaunchSpy()
    with _with_spy(spy):
        try:
            _run(main.job_post({"model_id": "ghost", "prompt": "a quiet studio shot"}))
        except main.HTTPException as exc:
            assert exc.status_code == 404, exc.status_code
            assert "модель не найдена" in exc.detail, exc.detail
            assert "ghost" in exc.detail, exc.detail
        else:
            raise AssertionError("HTTPException(404) expected for an unknown model_id")
    assert spy.calls == [], "no job may be launched for an unknown model_id"
    # The registry is untouched after the rejection.
    assert json.loads(main.REGISTRY_PATH.read_text(encoding="utf-8"))["models"] == _SEED_REGISTRY


# --------------------------------------------------------------------------- #
# AC-3 — payload without model_id keeps the active-model behaviour.
# --------------------------------------------------------------------------- #
def test_post_without_model_id_keeps_active_model():
    # Back-compat: no model_id (the old UI body) → the active model m1 is
    # bound (cmd --model m1, job record m1) and the guardrail/profile follow
    # m1. sdxl is chosen so the launch passes with m1's deployed SDXL stub.
    spy = _LaunchSpy()
    with _with_spy(spy):
        _run(
            main.job_post(
                {
                    "prompt": "a quiet studio shot",
                    "engine": "sdxl",
                    "seed": 42,
                }
            )
        )
    kind, cmd, launch_model_id, job_id = spy.calls[-1]
    assert kind == "post"
    assert _cmd_flag(cmd, "--model") == MODEL_ID, cmd
    assert launch_model_id == MODEL_ID
    assert main.jobs[job_id]["model_id"] == MODEL_ID
    assert "23-year-old fictional" in _cmd_flag(cmd, "--prompt"), cmd  # m1's character age
    # An empty/whitespace model_id behaves like an absent one (UI "active"
    # selector value), not like an unknown id.
    spy2 = _LaunchSpy()
    with _with_spy(spy2):
        _run(main.job_post({"model_id": "   ", "prompt": "a quiet studio shot", "engine": "sdxl"}))
    assert _cmd_flag(spy2.calls[-1][1], "--model") == MODEL_ID, spy2.calls[-1][1]


# --------------------------------------------------------------------------- #
# AC-4 — the guardrail follows the SELECTED model's character.json.
# --------------------------------------------------------------------------- #
def test_post_guardrail_uses_selected_model_age():
    # m2's character.json carries age 30 while the active m1 is 23: a raw
    # prompt for m2 must get the 30-year guardrail phrase injected (the
    # active model's 23 must not leak into the selected-model post).
    spy = _LaunchSpy()
    with _with_spy(spy):
        _run(main.job_post({"model_id": OTHER_ID, "prompt": "a quiet studio shot"}))
    prompt = _cmd_flag(spy.calls[-1][1], "--prompt")
    assert "30-year-old fictional" in prompt, prompt
    assert "23-year-old" not in prompt, prompt


def test_post_guardrail_age_below_floor_clamped_for_selected_model():
    # The floor applies to the selected model too: m2's character age 20 must
    # be clamped to prompts.AGE_FLOOR (23), never injected as "20-year-old".
    (main._model_config_dir(OTHER_ID) / "character.json").write_text(
        json.dumps({"name": "Mira", "age": 20}), encoding="utf-8"
    )
    try:
        spy = _LaunchSpy()
        with _with_spy(spy):
            _run(main.job_post({"model_id": OTHER_ID, "prompt": "a quiet studio shot"}))
        prompt = _cmd_flag(spy.calls[-1][1], "--prompt")
        assert f"{main.prompts.AGE_FLOOR}-year-old fictional" in prompt, prompt
        assert "20-year-old" not in prompt, prompt
    finally:
        (main._model_config_dir(OTHER_ID) / "character.json").write_text(
            json.dumps({"name": "Mira", "age": 30, "appearance": {"face": "round face", "build": "slim"}}),
            encoding="utf-8",
        )


# --------------------------------------------------------------------------- #
# AC-5 — scene meta: job_post cmd → real parser → generate_post → post.json
#        → list_posts. The whole chain runs against the temp workspace.
# --------------------------------------------------------------------------- #
def _generate_post_from_cmd(cmd: list[str]) -> tuple[dict, Path]:
    """Replay a spy-captured job_post cmd through the REAL queue_workflow
    parser and run generate_post with ROOT pointed at the shared temp
    workspace (queue_and_wait stubbed, the shipped flux graph copied in).
    Verifies the server→CLI seam: --scene-id/--scene-name are accepted by the
    parser and land verbatim in post.json. Returns (post.json meta,
    destination) — the caller cleans the destination up."""
    assert cmd[:3] == ["python", "scripts/queue_workflow.py", "post"], cmd
    args = queue_workflow.parser().parse_args(cmd[2:])
    ws = Path(_TMP_WORKSPACE.name)
    out_root = Path(tempfile.mkdtemp(prefix="post-model-out-", dir=str(ws)))
    (out_root / "photo.png").write_bytes(b"stub")
    comfy_dir = ws / "comfy"
    comfy_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "comfy" / "workflow_flux_lora.json", comfy_dir / "workflow_flux_lora.json")

    def _fake_queue(*_args, **_kwargs):
        return [{"subfolder": "", "filename": "photo.png"}]

    original_root = queue_workflow.ROOT
    original_data_root = queue_workflow.DATA_ROOT
    original_out_root = queue_workflow.OUTPUT_ROOT
    original_queue = queue_workflow.queue_and_wait
    queue_workflow.ROOT = ws
    queue_workflow.DATA_ROOT = main.DATA_ROOT  # (#17) destinations + lora preflight under DATA_ROOT
    queue_workflow.OUTPUT_ROOT = out_root
    queue_workflow.queue_and_wait = _fake_queue
    try:
        # Mirror queue_workflow.main(): --model → resolve_model_id → model_id.
        args.model_id = queue_workflow.resolve_model_id(args.model)
        queue_workflow.generate_post(args)
        destination = main.DATA_ROOT / "models" / args.model_id / ("posts" if args.mode == "public" else "posts-private") / args.name
        meta = json.loads((destination / "post.json").read_text(encoding="utf-8"))
        return meta, destination
    finally:
        queue_workflow.ROOT = original_root
        queue_workflow.DATA_ROOT = original_data_root
        queue_workflow.OUTPUT_ROOT = original_out_root
        queue_workflow.queue_and_wait = original_queue
        shutil.rmtree(comfy_dir, ignore_errors=True)


def test_post_scene_meta_written_by_generate_post():
    # Full chain for the SELECTED model m2 with a library scene: the cmd
    # carries the resolved --scene-id/--scene-name, the real parser accepts
    # them, and post.json records the complete meta set.
    spy = _LaunchSpy()
    with _with_spy(spy):
        _run(
            main.job_post(
                {
                    "model_id": OTHER_ID,
                    "scene_id": SCENE_ID,
                    "prompt": "a quiet studio shot",
                    "seed": 42,
                    "name": "chain-post",
                }
            )
        )
    cmd = spy.calls[-1][1]
    assert _cmd_flag(cmd, "--scene-id") == SCENE_ID, cmd
    assert _cmd_flag(cmd, "--scene-name") == SCENE_NAME, cmd
    destination = main.DATA_ROOT / "models" / OTHER_ID / "posts" / "chain-post"
    try:
        meta, destination = _generate_post_from_cmd(cmd)
        assert destination.is_dir(), destination
        assert (destination / "photo.png").is_file(), "the stub photo must be moved into the post folder"
        assert set(meta) == {
            "scene_id",
            "scene_name",
            "model_id",
            "engine",
            "mode",
            "seed",
            "created_at",
        }, meta
        assert meta["scene_id"] == SCENE_ID, meta
        assert meta["scene_name"] == SCENE_NAME, meta
        assert meta["model_id"] == OTHER_ID, meta
        assert meta["engine"] == "flux", meta
        assert meta["mode"] == "public", meta
        assert meta["seed"] == 42, meta
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", meta["created_at"]), meta
    finally:
        shutil.rmtree(destination, ignore_errors=True)


def test_post_cmd_without_scene_carries_empty_scene_flags():
    # No scene_id in the payload → the cmd still carries the flags with empty
    # values and post.json records both scene fields as null (the card then
    # shows «—»).
    spy = _LaunchSpy()
    with _with_spy(spy):
        _run(main.job_post({"model_id": OTHER_ID, "prompt": "a quiet studio shot", "name": "no-scene-post"}))
    cmd = spy.calls[-1][1]
    assert _cmd_flag(cmd, "--scene-id") == "", cmd
    assert _cmd_flag(cmd, "--scene-name") == "", cmd
    destination = main.DATA_ROOT / "models" / OTHER_ID / "posts" / "no-scene-post"
    try:
        meta, destination = _generate_post_from_cmd(cmd)
        assert meta["scene_id"] is None, meta
        assert meta["scene_name"] is None, meta
        assert meta["model_id"] == OTHER_ID, meta
    finally:
        shutil.rmtree(destination, ignore_errors=True)


def test_generate_post_older_namespace_without_scene_keys():
    # getattr tolerance: a direct caller (older scripts/tests) passes a
    # Namespace without scene_id/scene_name — generate_post must still write
    # post.json with the scene fields null.
    import argparse

    with tempfile.TemporaryDirectory(prefix="post-model-legacy-") as tmp:
        root = Path(tmp) / "repo"
        data = Path(tmp) / "data"  # VALERY_DATA_ROOT (#17)
        data.mkdir()
        out_root = Path(tmp) / "comfy-output"
        (root / "models" / OTHER_ID).mkdir(parents=True)
        (root / "models" / "registry.json").write_text(_registry_json([_m2_entry()]), encoding="utf-8")
        (root / "models" / OTHER_ID / "character.json").write_text(
            json.dumps({"name": "Mira", "age": 30}), encoding="utf-8"
        )
        loras = data / "runtime" / "models" / "loras"
        loras.mkdir(parents=True)
        (loras / _M2_FLUX_LORA).write_bytes(b"")
        (root / "comfy").mkdir(parents=True)
        # Minimal flux graph: nodes 6/31/40/9 touched by generate_post; no
        # "33" so set_negative stays a no-op.
        (root / "comfy" / "workflow_flux_lora.json").write_text(
            json.dumps(
                {
                    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
                    "31": {"class_type": "KSampler", "inputs": {"seed": 0}},
                    "40": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "", "strength_model": 0.8}},
                    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": ""}},
                }
            ),
            encoding="utf-8",
        )
        out_root.mkdir(parents=True)
        (out_root / "photo.png").write_bytes(b"stub")

        def _fake_queue(*_args, **_kwargs):
            return [{"subfolder": "", "filename": "photo.png"}]

        original_root = queue_workflow.ROOT
        original_data_root = queue_workflow.DATA_ROOT
        original_out_root = queue_workflow.OUTPUT_ROOT
        original_queue = queue_workflow.queue_and_wait
        queue_workflow.ROOT = root
        queue_workflow.DATA_ROOT = data
        queue_workflow.OUTPUT_ROOT = out_root
        queue_workflow.queue_and_wait = _fake_queue
        try:
            queue_workflow.generate_post(
                argparse.Namespace(
                    model_id=OTHER_ID,
                    engine="flux",
                    uncensor=False,
                    prompt="a quiet studio shot, 30-year-old fictional",
                    negative="",
                    caption="",
                    name="legacy-ns",
                    seed=7,
                    lora_strength=0.8,
                    mode="private",
                    url="http://127.0.0.1:9",
                )
            )
            destination = data / "models" / OTHER_ID / "posts-private" / "legacy-ns"
            meta = json.loads((destination / "post.json").read_text(encoding="utf-8"))
            assert meta["scene_id"] is None, meta
            assert meta["scene_name"] is None, meta
            assert meta["model_id"] == OTHER_ID, meta
            assert meta["engine"] == "flux", meta
            assert meta["mode"] == "private", meta
            assert meta["seed"] == 7, meta
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", meta["created_at"]), meta
        finally:
            queue_workflow.ROOT = original_root
            queue_workflow.DATA_ROOT = original_data_root
            queue_workflow.OUTPUT_ROOT = original_out_root
            queue_workflow.queue_and_wait = original_queue


def test_post_parser_scene_flags():
    # Parser seam on its own: --scene-id/--scene-name are optional post flags
    # (default "") so older cmds without them keep parsing.
    args = queue_workflow.parser().parse_args(["post", "--prompt", "x"])
    assert args.scene_id == "" and args.scene_name == "", vars(args)
    args = queue_workflow.parser().parse_args(
        ["post", "--prompt", "x", "--scene-id", SCENE_ID, "--scene-name", SCENE_NAME]
    )
    assert args.scene_id == SCENE_ID, args
    assert args.scene_name == SCENE_NAME, args


def _write_post_folder(kind_dir: str, name: str, meta: dict | None, *, corrupt: bool = False) -> Path:
    post = main.DATA_ROOT / "models" / MODEL_ID / kind_dir / name
    post.mkdir(parents=True, exist_ok=True)
    (post / "photo.png").write_bytes(b"stub")
    if corrupt:
        (post / "post.json").write_text("{not json", encoding="utf-8")
    elif meta is not None:
        (post / "post.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return post


def test_list_posts_exposes_scene_meta():
    # Reader side (active model m1): a post with post.json exposes the meta
    # fields; a legacy post (no file) and a corrupt file both degrade to
    # scene_name null — the card shows «—» instead of failing.
    with_meta = _write_post_folder(
        "posts",
        "with-meta",
        {
            "scene_id": SCENE_ID,
            "scene_name": SCENE_NAME,
            "model_id": MODEL_ID,
            "engine": "sdxl",
            "mode": "public",
            "seed": 7,
            "created_at": "2026-10-10T10:00:00Z",
        },
    )
    legacy = _write_post_folder("posts", "legacy", None)
    corrupt = _write_post_folder("posts", "corrupt", None, corrupt=True)
    private = _write_post_folder(
        "posts-private",
        "priv-meta",
        {
            "scene_id": "s9",
            "scene_name": "Night city",
            "model_id": MODEL_ID,
            "engine": "flux",
            "mode": "private",
            "seed": 9,
            "created_at": "2026-10-10T11:00:00Z",
        },
    )
    try:
        posts = {entry["name"]: entry for entry in _run(main.list_posts())["posts"]}
        assert set(posts) >= {"with-meta", "legacy", "corrupt"}, sorted(posts)
        entry = posts["with-meta"]
        assert entry["scene_name"] == SCENE_NAME, entry
        assert entry["scene_id"] == SCENE_ID, entry
        assert entry["model_id"] == MODEL_ID, entry
        assert entry["engine"] == "sdxl", entry
        assert entry["mode"] == "public", entry
        assert entry["seed"] == 7, entry
        legacy_entry = posts["legacy"]
        assert legacy_entry["scene_name"] is None, legacy_entry
        assert legacy_entry["scene_id"] is None, legacy_entry
        assert legacy_entry["seed"] is None, legacy_entry
        corrupt_entry = posts["corrupt"]
        assert corrupt_entry["scene_name"] is None, corrupt_entry
        private_posts = {entry["name"]: entry for entry in _run(main.list_posts_private())["posts"]}
        assert private_posts["priv-meta"]["scene_name"] == "Night city", private_posts
    finally:
        for folder in (with_meta, legacy, corrupt, private):
            shutil.rmtree(folder, ignore_errors=True)


# --------------------------------------------------------------------------- #
# AC-6 — /api/models carries a per-model engine.
# --------------------------------------------------------------------------- #
def test_api_models_engine_per_model():
    # engine per deployed slot: m1 has the SDXL slot file → "sdxl"; m2 has
    # only the FLUX slot file (its SDXL slot name is registered but not
    # deployed) → "flux"; m3 has no slot files → None.
    payload = _run(main.list_models())
    entries = {entry["id"]: entry for entry in payload["models"]}
    assert set(entries) == {MODEL_ID, OTHER_ID, NO_SLOT_ID}, sorted(entries)
    assert all("engine" in entry for entry in payload["models"]), payload["models"]
    assert entries[MODEL_ID]["engine"] == "sdxl", entries[MODEL_ID]
    assert entries[OTHER_ID]["engine"] == "flux", entries[OTHER_ID]
    assert entries[NO_SLOT_ID]["engine"] is None, entries[NO_SLOT_ID]
    assert payload["active_id"] == MODEL_ID, payload


def test_model_engine_helper_slot_rules():
    # Pure helper behind AC-6: SDXL slot wins when both files exist, flux
    # only when the SDXL file is absent, None when neither file is deployed.
    both = _m2_entry(lora=_M1_SDXL_LORA)  # m1.safetensors is deployed
    assert main._model_engine(both) == "sdxl", both
    assert main._model_engine(_m2_entry()) == "flux"
    assert main._model_engine(_m3_entry()) is None
    assert main._model_engine({"id": "ghost"}) is None


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
