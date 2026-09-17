"""Import-safe tests for management.main generation-engine validation
(feature Phase 4: SDXL/Pony engine for post, optional uncensor LoRA).
Covers the pure _validate_engine_fields validator (default flux, valid sdxl,
invalid engine 400, non-bool uncensor 400, engine field dropped when
allow_engine=False) and the cmd lines built by the job endpoints: post carries
--engine sdxl (plus --uncensor when true) and the job record gains "engine",
while group carries neither flag. Also covers the flux-LoRA gating: a missing
LoRA file must 400 a flux post but not an sdxl post (the sdxl graph does carry
 a LoraLoaderModelOnly "40", but the engine is graceful: without the file the
 node is dropped from the graph), and candidates/expand carry --uncensor when
 requested. Also covers the sdxl side of the character-LoRA story at the
 queue_workflow level: workflow_sdxl.json wires a model-only LoraLoaderModelOnly
 (node "40") into the KSampler, an sdxl post with a deployed LoRA file sends
 node "40" with the registry lora name and the requested strength, and an
 sdxl post WITHOUT a LoRA file stays graceful (graph rewired straight from the
 checkpoint, post still produced). Plus the SDXL/Pony training template
 (lora/train_config.template.yaml: arch sdxl + ddpm, no flux-only knobs).
Also covers the optional post caption: an empty/absent caption no longer 400s,
the cmd still carries --caption "" (group keeps the required caption), the job
record stores null for an empty caption, and queue_workflow.generate_post skips
caption.txt for an empty caption but writes it for a filled one.
pytest-compatible; pytest is not required.

Run either way:
    python -m pytest tests/test_engine_validation.py -q
    python tests/test_engine_validation.py

WORKSPACE is pointed at a tempfile directory BEFORE management.main is
imported (the module resolves WORKSPACE and mounts /static at import time),
so the tests never depend on the real /workspace layout. The registry, LoRA
stub, character and profile are written into the temp WORKSPACE after import
— main reads them from disk on every call. The endpoints are tested with
main._launch monkeypatched (a spy), so no docker/ComfyUI is touched.
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import queue_workflow  # noqa: E402  (pure library_scene_texts helper)

_TMP_WORKSPACE = tempfile.TemporaryDirectory(prefix="engine-ws-")
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

MODEL_ID = "m1"  # satisfies ^[a-z0-9][a-z0-9-]{1,62}$


def _run(coro):
    return asyncio.run(coro)


# Fixtures (written after import; main re-reads registry/character/profile
# from disk on every call). One active model with a LoRA stub in LORAS_DIR.
main.REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
main.REGISTRY_PATH.write_text(
    json.dumps(
        {
            "version": 1,
            "models": [
                {
                    "id": MODEL_ID,
                    "name": "Valery",
                    "age": 23,
                    "height_cm": 178,
                    "tags": [],
                    "active": True,
                    "created": "2026-01-01T00:00:00Z",
                    "lora": f"{MODEL_ID}.safetensors",
                    "groups": [],
                },
            ],
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
main.LORAS_DIR.mkdir(parents=True, exist_ok=True)
(main.LORAS_DIR / f"{MODEL_ID}.safetensors").write_bytes(b"")  # empty stub: existence is what matters
_MODEL_DIR = main.MODELS_DIR / MODEL_ID
_MODEL_DIR.mkdir(parents=True, exist_ok=True)
(_MODEL_DIR / "character.json").write_text(
    json.dumps({"name": "Valery", "age": 23, "appearance": {"face": "oval face", "build": "athletic"}}),
    encoding="utf-8",
)
(_MODEL_DIR / "prompt_profile.json").write_text(
    json.dumps({"version": 1, "face": "oval face", "body": "athletic", "style": "", "negative": "", "scenes": {}}),
    encoding="utf-8",
)


def _expect_http_status(payload: dict, status: int, *, allow_engine: bool = True) -> main.HTTPException:
    try:
        main._validate_engine_fields(payload, allow_engine=allow_engine)
    except main.HTTPException as exc:
        assert exc.status_code == status, f"expected {status}, got {exc.status_code}"
        return exc
    raise AssertionError(
        f"HTTPException({status}) expected, _validate_engine_fields returned normally"
    )


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


def _without_lora_stub():
    """Temporarily remove the character LoRA stub from LORAS_DIR (restored
    after the block) so a test can exercise the missing-LoRA paths."""
    import contextlib

    @contextlib.contextmanager
    def _patched():
        stub = main.LORAS_DIR / f"{MODEL_ID}.safetensors"
        existed = stub.is_file()
        if existed:
            stub.unlink()
        try:
            yield
        finally:
            if existed:
                stub.write_bytes(b"")

    return _patched()


def test_default_engine_is_flux():
    fields = main._validate_engine_fields({}, allow_engine=True)
    assert fields == {"engine": "flux", "uncensor": False}
    fields = main._validate_engine_fields({"engine": "flux"}, allow_engine=True)
    assert fields == {"engine": "flux", "uncensor": False}


def test_valid_engine_sdxl():
    fields = main._validate_engine_fields({"engine": "sdxl", "uncensor": True}, allow_engine=True)
    assert fields == {"engine": "sdxl", "uncensor": True}


def test_invalid_engine_400():
    exc = _expect_http_status({"engine": "pony"}, 400)
    assert "engine" in exc.detail, exc.detail


def test_uncensor_non_bool_400():
    _expect_http_status({"uncensor": "yes"}, 400)
    _expect_http_status({"uncensor": 1}, 400)  # int is not a JSON bool


def test_engine_field_dropped_when_not_allowed():
    # candidates/expand take only --uncensor: engine is ignored (never errors)
    # and stays flux.
    fields = main._validate_engine_fields({"engine": "sdxl", "uncensor": True}, allow_engine=False)
    assert fields == {"engine": "flux", "uncensor": True}


def test_post_cmd_contains_engine_sdxl_and_uncensor():
    spy = _LaunchSpy()
    with _with_spy(spy):
        _run(
            main.job_post(
                {
                    "caption": "a caption",
                    "prompt": "a quiet studio shot",
                    "engine": "sdxl",
                    "uncensor": True,
                }
            )
        )
    kind, cmd, model_id, job_id = spy.calls[-1]
    assert kind == "post" and model_id == MODEL_ID
    assert "--engine" in cmd, cmd
    assert cmd[cmd.index("--engine") + 1] == "sdxl", cmd
    assert "--uncensor" in cmd, cmd
    assert main.jobs[job_id]["engine"] == "sdxl"


def test_group_cmd_has_no_engine_or_uncensor():
    spy = _LaunchSpy()
    with _with_spy(spy):
        _run(main.job_group({"models": [MODEL_ID], "prompt": "a shared scene", "caption": "a caption"}))
    kind, cmd, model_id, _job_id = spy.calls[-1]
    assert kind == "group" and model_id == MODEL_ID
    assert "--engine" not in cmd, cmd
    assert "--uncensor" not in cmd, cmd


def test_post_sdxl_no_lora_file_ok():
    # The SDXL (Pony) engine degrades gracefully: a missing character LoRA
    # must not 400 an sdxl post (defect: sdxl posts were blocked until a FLUX
    # LoRA was trained).
    spy = _LaunchSpy()
    with _without_lora_stub(), _with_spy(spy):
        result = _run(
            main.job_post(
                {
                    "caption": "a caption",
                    "prompt": "a quiet studio shot",
                    "engine": "sdxl",
                }
            )
        )
    kind, cmd, model_id, job_id = spy.calls[-1]
    assert kind == "post" and model_id == MODEL_ID
    assert result == {"job_id": job_id, "status": "queued"}
    assert "--engine" in cmd, cmd
    assert cmd[cmd.index("--engine") + 1] == "sdxl", cmd


def test_post_flux_no_lora_file_400():
    # The flux graph loads the character LoRA: a missing file must still 400
    # and launch nothing.
    spy = _LaunchSpy()
    with _without_lora_stub(), _with_spy(spy):
        try:
            _run(
                main.job_post(
                    {
                        "caption": "a caption",
                        "prompt": "a quiet studio shot",
                        "engine": "flux",
                    }
                )
            )
        except main.HTTPException as exc:
            assert exc.status_code == 400, exc.status_code
            assert "LoRA" in exc.detail, exc.detail
        else:
            raise AssertionError("HTTPException(400) expected for flux post without LoRA")
    assert spy.calls == [], "no job may be launched when the flux LoRA is missing"


def test_candidates_uncensor_flag_in_cmd():
    spy = _LaunchSpy()
    with _with_spy(spy):
        _run(main.job_candidates({"prompt": "a quiet studio shot", "uncensor": True}))
    kind, cmd, model_id, _job_id = spy.calls[-1]
    assert kind == "candidates" and model_id == MODEL_ID
    assert "--uncensor" in cmd, cmd


def test_expand_uncensor_flag_in_cmd():
    candidates_dir = _MODEL_DIR / "reference" / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    (candidates_dir / f"{MODEL_ID}-01.jpg").write_bytes(b"")
    spy = _LaunchSpy()
    with _with_spy(spy):
        _run(main.job_expand({"reference": f"{MODEL_ID}-01.jpg", "uncensor": True}))
    kind, cmd, model_id, _job_id = spy.calls[-1]
    assert kind == "expand" and model_id == MODEL_ID
    assert "--uncensor" in cmd, cmd


def test_expand_no_seed_gets_random_seed_in_cmd_and_job():
    # Payload without seed → random seed in [0, 2**31) is passed to the cmd
    # AND stored on the job record (field "seed").
    candidates_dir = _MODEL_DIR / "reference" / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    (candidates_dir / f"{MODEL_ID}-02.jpg").write_bytes(b"")
    spy = _LaunchSpy()
    with _with_spy(spy):
        _run(main.job_expand({"reference": f"{MODEL_ID}-02.jpg"}))
    kind, cmd, model_id, job_id = spy.calls[-1]
    assert kind == "expand" and model_id == MODEL_ID
    assert "--seed" in cmd, cmd
    seed = int(cmd[cmd.index("--seed") + 1])
    assert 0 <= seed < 2**31, seed
    assert main.jobs[job_id]["seed"] == seed


def test_expand_explicit_seed_in_cmd():
    candidates_dir = _MODEL_DIR / "reference" / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    (candidates_dir / f"{MODEL_ID}-03.jpg").write_bytes(b"")
    spy = _LaunchSpy()
    with _with_spy(spy):
        _run(main.job_expand({"reference": f"{MODEL_ID}-03.jpg", "seed": 777}))
    kind, cmd, model_id, job_id = spy.calls[-1]
    assert kind == "expand" and model_id == MODEL_ID
    assert cmd[cmd.index("--seed") + 1] == "777", cmd
    assert main.jobs[job_id]["seed"] == 777


def test_candidates_no_seed_gets_random_seed_in_cmd_and_job():
    # Payload without seed (UI leaves the field empty) → random seed in
    # [0, 2**31) is passed to the cmd AND stored on the job record.
    spy = _LaunchSpy()
    with _with_spy(spy):
        _run(main.job_candidates({"prompt": "a quiet studio shot"}))
    kind, cmd, model_id, job_id = spy.calls[-1]
    assert kind == "candidates" and model_id == MODEL_ID
    assert "--seed" in cmd, cmd
    seed = int(cmd[cmd.index("--seed") + 1])
    assert 0 <= seed < 2**31, seed
    assert main.jobs[job_id]["seed"] == seed


def test_post_no_seed_gets_random_seed_in_cmd_and_job():
    # Payload without seed (UI leaves the field empty) → random seed in
    # [0, 2**31) is passed to the cmd AND stored on the job record.
    spy = _LaunchSpy()
    with _with_spy(spy):
        _run(
            main.job_post(
                {
                    "caption": "a caption",
                    "prompt": "a quiet studio shot",
                }
            )
        )
    kind, cmd, model_id, job_id = spy.calls[-1]
    assert kind == "post" and model_id == MODEL_ID
    assert "--seed" in cmd, cmd
    seed = int(cmd[cmd.index("--seed") + 1])
    assert 0 <= seed < 2**31, seed
    assert main.jobs[job_id]["seed"] == seed


def test_candidates_explicit_seed_in_cmd():
    # An explicit seed is used as-is (int validation), never randomized.
    spy = _LaunchSpy()
    with _with_spy(spy):
        _run(main.job_candidates({"prompt": "a quiet studio shot", "seed": 555}))
    kind, cmd, model_id, job_id = spy.calls[-1]
    assert kind == "candidates" and model_id == MODEL_ID
    assert cmd[cmd.index("--seed") + 1] == "555", cmd
    assert main.jobs[job_id]["seed"] == 555


def test_group_no_seed_gets_random_seed_in_cmd_and_job():
    # Payload without seed (UI leaves the field empty) → random seed in
    # [0, 2**31) is passed to the cmd AND stored on the job record
    # (one seed per group run; per-member seed+i is queue_workflow's job).
    spy = _LaunchSpy()
    with _with_spy(spy):
        _run(main.job_group({"models": [MODEL_ID], "prompt": "a shared scene", "caption": "a caption"}))
    kind, cmd, model_id, job_id = spy.calls[-1]
    assert kind == "group" and model_id == MODEL_ID
    assert "--seed" in cmd, cmd
    seed = int(cmd[cmd.index("--seed") + 1])
    assert 0 <= seed < 2**31, seed
    assert main.jobs[job_id]["seed"] == seed


def test_post_empty_caption_accepted_no_400():
    # Optional caption (Instagram post text): an empty/blank caption must not
    # 400; the cmd still carries --caption "" and the job record stores null.
    spy = _LaunchSpy()
    with _with_spy(spy):
        result = _run(main.job_post({"caption": "   ", "prompt": "a quiet studio shot"}))
    kind, cmd, model_id, job_id = spy.calls[-1]
    assert kind == "post" and model_id == MODEL_ID
    assert result == {"job_id": job_id, "status": "queued"}
    assert "--caption" in cmd, cmd
    assert cmd[cmd.index("--caption") + 1] == "", cmd
    assert main.jobs[job_id]["caption"] is None


def test_post_missing_caption_key_accepted():
    # No "caption" key in the payload at all → same contract as an empty one.
    spy = _LaunchSpy()
    with _with_spy(spy):
        result = _run(main.job_post({"prompt": "a quiet studio shot"}))
    kind, cmd, model_id, job_id = spy.calls[-1]
    assert kind == "post" and model_id == MODEL_ID
    assert result == {"job_id": job_id, "status": "queued"}
    assert cmd[cmd.index("--caption") + 1] == "", cmd
    assert main.jobs[job_id]["caption"] is None


def test_post_filled_caption_unchanged():
    # A filled caption behaves exactly as before the change.
    spy = _LaunchSpy()
    with _with_spy(spy):
        _run(main.job_post({"caption": "hello #fitness", "prompt": "a quiet studio shot"}))
    kind, cmd, model_id, job_id = spy.calls[-1]
    assert kind == "post" and model_id == MODEL_ID
    assert cmd[cmd.index("--caption") + 1] == "hello #fitness", cmd
    assert main.jobs[job_id]["caption"] == "hello #fitness"


def test_post_caption_arg_optional_in_parser():
    # post: --caption is optional (default ""); group: --caption stays required.
    args = queue_workflow.parser().parse_args(["post", "--prompt", "x"])
    assert args.caption == ""
    import contextlib
    import io

    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        try:
            queue_workflow.parser().parse_args(["group", "--models", MODEL_ID, "--prompt", "x"])
        except SystemExit:
            pass  # argparse exits on the still-required --caption
        else:
            raise AssertionError("group --caption must remain required")


def _run_generate_post(caption: str, name: str) -> dict:
    """Run queue_workflow.generate_post in a temp ROOT with a stubbed
    queue_and_wait; return a snapshot of the created post folder (taken
    before the temp tree is cleaned up)."""
    import argparse

    with tempfile.TemporaryDirectory(prefix="gw-post-") as tmp:
        root = Path(tmp) / "repo"
        out_root = Path(tmp) / "comfy-output"
        model_id = "pm1"
        (root / "models" / model_id).mkdir(parents=True)
        (root / "models" / "registry.json").write_text(
            json.dumps({"version": 1, "models": [{"id": model_id, "active": True, "lora": f"{model_id}.safetensors"}]}),
            encoding="utf-8",
        )
        (root / "models" / model_id / "character.json").write_text(
            json.dumps({"name": "Valery", "age": 23}), encoding="utf-8"
        )
        loras = root / "runtime" / "models" / "loras"
        loras.mkdir(parents=True)
        (loras / f"{model_id}.safetensors").write_bytes(b"")  # flux preflight: existence matters
        (root / "comfy").mkdir(parents=True)
        # Minimal flux_lora graph: nodes 6/31/40/9 touched by generate_post; no "33"
        # so set_negative stays a no-op.
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
        original_out_root = queue_workflow.OUTPUT_ROOT
        original_queue = queue_workflow.queue_and_wait
        queue_workflow.ROOT = root
        queue_workflow.OUTPUT_ROOT = out_root
        queue_workflow.queue_and_wait = _fake_queue
        try:
            queue_workflow.generate_post(
                argparse.Namespace(
                    model_id=model_id,
                    engine="flux",
                    uncensor=False,
                    prompt="",
                    negative="",
                    caption=caption,
                    name=name,
                    seed=1,
                    lora_strength=0.8,
                    mode="public",
                    url="http://127.0.0.1:9",
                )
            )
            destination = root / "models" / model_id / "posts" / name
            caption_path = destination / "caption.txt"
            return {
                "destination": destination,
                # Snapshot the facts inside the with-block: the temp tree is
                # torn down when the helper returns, so Paths must not be
                # re-checked by the test itself.
                "destination_exists": destination.is_dir(),
                "has_photo": (destination / "photo.png").is_file(),
                "caption_text": caption_path.read_text(encoding="utf-8") if caption_path.is_file() else None,
                "has_prompt": (destination / "prompt.txt").is_file(),
            }
        finally:
            queue_workflow.ROOT = original_root
            queue_workflow.OUTPUT_ROOT = original_out_root
            queue_workflow.queue_and_wait = original_queue


def test_generate_post_empty_caption_no_file():
    # Empty --caption: the post folder is still created (photo + prompt), but
    # caption.txt must NOT exist (list_posts then reports caption=None).
    result = _run_generate_post("", "empty-cap")
    assert result["destination_exists"], result["destination"]
    assert result["has_photo"], "photo.png must still be moved into the post folder"
    assert result["caption_text"] is None, "caption.txt must not be created for an empty caption"
    assert result["has_prompt"]


def test_generate_post_filled_caption_writes_file():
    # A filled caption is written to caption.txt (stripped + trailing newline)
    # exactly as before the change.
    result = _run_generate_post("  my caption  ", "filled-cap")
    assert result["destination_exists"], result["destination"]
    assert result["has_photo"]
    assert result["caption_text"] == "my caption\n", result["caption_text"]
    assert result["has_prompt"]


SDXL_MODEL_ID = "pm-sdxl"  # satisfies the registry id pattern, isolated from the flux stubs


def _run_generate_post_sdxl(*, with_lora: bool, lora_strength: float = 0.65) -> dict:
    """Run queue_workflow.generate_post (engine=sdxl) in a temp ROOT with a
    stubbed queue_and_wait; return the workflow facts captured at queue time
    plus the created post folder (snapshot taken before the temp tree is
    cleaned up). The SDXL checkpoint preflight reads the module-level
    SDXL_CKPT_PATH (bound to the real ROOT at import time), so it is patched
    to a stub in the temp tree as well."""
    import argparse

    with tempfile.TemporaryDirectory(prefix="gw-post-sdxl-") as tmp:
        root = Path(tmp) / "repo"
        out_root = Path(tmp) / "comfy-output"
        (root / "models" / SDXL_MODEL_ID).mkdir(parents=True)
        (root / "models" / "registry.json").write_text(
            json.dumps(
                {"version": 1, "models": [{"id": SDXL_MODEL_ID, "active": True, "lora": f"{SDXL_MODEL_ID}.safetensors"}]}
            ),
            encoding="utf-8",
        )
        (root / "models" / SDXL_MODEL_ID / "character.json").write_text(
            json.dumps({"name": "Valery", "age": 23}), encoding="utf-8"
        )
        loras = root / "runtime" / "models" / "loras"
        loras.mkdir(parents=True)
        if with_lora:
            (loras / f"{SDXL_MODEL_ID}.safetensors").write_bytes(b"")  # existence is what matters
        ckpt = root / "runtime" / "models" / "checkpoints" / "ponyDiffusionV6XL.safetensors"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b"")  # sdxl preflight: the checkpoint must exist
        (root / "comfy").mkdir(parents=True)
        # Minimal sdxl graph: CheckpointLoaderSimple "30" → LoraLoaderModelOnly
        # "40" → KSampler "31"; CLIPTextEncode "6"/"33" on the checkpoint;
        # SaveImage "9".
        (root / "comfy" / "workflow_sdxl.json").write_text(
            json.dumps(
                {
                    "30": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "ponyDiffusionV6XL.safetensors"}},
                    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["30", 1]}},
                    "31": {"class_type": "KSampler", "inputs": {"seed": 0, "model": ["40", 0]}},
                    "33": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["30", 1]}},
                    "40": {
                        "class_type": "LoraLoaderModelOnly",
                        "inputs": {"lora_name": "placeholder.safetensors", "strength_model": 0.0, "model": ["30", 0]},
                    },
                    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": ""}},
                }
            ),
            encoding="utf-8",
        )
        out_root.mkdir(parents=True)
        (out_root / "photo.png").write_bytes(b"stub")

        captured: dict = {}

        def _fake_queue(url: str, workflow: dict) -> list:
            captured["workflow"] = workflow
            return [{"subfolder": "", "filename": "photo.png"}]

        original_root = queue_workflow.ROOT
        original_out_root = queue_workflow.OUTPUT_ROOT
        original_queue = queue_workflow.queue_and_wait
        original_ckpt = queue_workflow.SDXL_CKPT_PATH
        queue_workflow.ROOT = root
        queue_workflow.OUTPUT_ROOT = out_root
        queue_workflow.queue_and_wait = _fake_queue
        queue_workflow.SDXL_CKPT_PATH = ckpt
        try:
            queue_workflow.generate_post(
                argparse.Namespace(
                    model_id=SDXL_MODEL_ID,
                    engine="sdxl",
                    uncensor=False,
                    prompt="",
                    negative="",
                    caption="",
                    name="sdxl-post",
                    seed=1,
                    lora_strength=lora_strength,
                    mode="public",
                    url="http://127.0.0.1:9",
                )
            )
            destination = root / "models" / SDXL_MODEL_ID / "posts" / "sdxl-post"
            return {
                "destination_exists": destination.is_dir(),
                "has_photo": (destination / "photo.png").is_file(),
                # The workflow dict itself outlives the temp tree (it is a
                # plain in-memory object), so the test can inspect it directly.
                "workflow": captured.get("workflow"),
            }
        finally:
            queue_workflow.ROOT = original_root
            queue_workflow.OUTPUT_ROOT = original_out_root
            queue_workflow.queue_and_wait = original_queue
            queue_workflow.SDXL_CKPT_PATH = original_ckpt


def test_generate_post_sdxl_with_lora_wires_lora_loader():
    # The SDXL (Pony) graph carries the character LoRA too: with the file
    # deployed, node "40" must send the registry lora name with the requested
    # (non-default) strength and the KSampler stays wired through it.
    result = _run_generate_post_sdxl(with_lora=True, lora_strength=0.65)
    assert result["destination_exists"] and result["has_photo"], result
    workflow = result["workflow"]
    assert workflow is not None, "queue_and_wait must have been called"
    assert workflow["40"]["inputs"]["lora_name"] == f"{SDXL_MODEL_ID}.safetensors"
    assert workflow["40"]["inputs"]["strength_model"] == 0.65
    assert workflow["31"]["inputs"]["model"] == ["40", 0]


def test_generate_post_sdxl_without_lora_rewires_to_checkpoint():
    # No trained LoRA: the sdxl post must NOT fail (unlike flux, which is
    # fail-fast). The graph is rewired straight from the checkpoint
    # (KSampler ← "30"), node "40" is dropped from the graph (ComfyUI
    # validates every node of the submitted prompt), and the post is
    # produced without identity.
    result = _run_generate_post_sdxl(with_lora=False)
    assert result["destination_exists"] and result["has_photo"], "sdxl post must succeed without a LoRA"
    workflow = result["workflow"]
    assert workflow is not None, "queue_and_wait must have been called"
    assert workflow["31"]["inputs"]["model"] == ["30", 0]
    # The LoraLoader node must not reach ComfyUI at all: its lora_name would
    # fail validation while the file is missing.
    assert "40" not in workflow


def test_sdxl_workflow_lora_loader_wired():
    # comfy/workflow_sdxl.json ships the character LoRA in the model chain:
    # CheckpointLoaderSimple "30" → LoraLoaderModelOnly "40" → KSampler "31".
    # Model-only LoRA: the CLIP nodes ("6"/"33") stay on the checkpoint.
    workflow = json.loads((REPO_ROOT / "comfy" / "workflow_sdxl.json").read_text(encoding="utf-8"))
    assert workflow["40"]["class_type"] == "LoraLoaderModelOnly"
    assert workflow["40"]["inputs"]["model"] == ["30", 0]
    assert workflow["31"]["inputs"]["model"] == ["40", 0]
    assert workflow["6"]["inputs"]["clip"] == ["30", 1]
    assert workflow["33"]["inputs"]["clip"] == ["30", 1]


def test_train_config_template_is_sdxl_pony():
    # The training template targets the SDXL (Pony) engine (FLUX training
    # track closed — the model does not fit the card): arch sdxl + explicit
    # ddpm noise scheduler, and no flux-only knobs.
    template = (REPO_ROOT / "lora" / "train_config.template.yaml").read_text(encoding="utf-8")
    assert "arch: sdxl" in template
    assert "noise_scheduler: ddpm" in template
    for forbidden in ("is_flux", "quantize", "flowmatch", "low_vram"):
        assert forbidden not in template, f"{forbidden!r} must not appear in the SDXL training template"


def test_library_scene_texts_mode_filter_and_bad_inputs():
    import tempfile

    with tempfile.TemporaryDirectory(prefix="lib-scenes-") as tmp:
        tmp_path = Path(tmp)
        scenes = [
            {"id": "a", "name": "A", "mode": "expand", "text": "expand one", "tags": []},
            {"id": "b", "name": "B", "mode": "post", "text": "post one", "tags": []},
            {"id": "c", "name": "C", "mode": "expand", "text": "expand two", "tags": []},
        ]
        # Mode filter on a plain top-level list.
        lib = tmp_path / "library.json"
        lib.write_text(json.dumps(scenes), encoding="utf-8")
        assert queue_workflow.library_scene_texts(str(lib), "expand") == ["expand one", "expand two"]
        assert queue_workflow.library_scene_texts(str(lib), "post") == ["post one"]
        assert queue_workflow.library_scene_texts(str(lib), "candidates") == []
        # The tracked wrapper format {"version": 1, "scenes": [...]} as well.
        wrapper = tmp_path / "library_wrapped.json"
        wrapper.write_text(json.dumps({"version": 1, "scenes": scenes}), encoding="utf-8")
        assert queue_workflow.library_scene_texts(str(wrapper), "expand") == ["expand one", "expand two"]
        # Missing file → [].
        assert queue_workflow.library_scene_texts(str(tmp_path / "absent.json"), "expand") == []
        # Broken JSON → [].
        bad = tmp_path / "broken.json"
        bad.write_text("{not json", encoding="utf-8")
        assert queue_workflow.library_scene_texts(str(bad), "expand") == []
        # A ready list (no file) works too.
        assert queue_workflow.library_scene_texts(scenes, "post") == ["post one"]


def test_expand_empty_library_exits_before_network():
    # workorder без --prompt + пустая/отсутствующая library.json:
    # library_scene_texts возвращает [], и expand_dataset обязан SystemExit
    # «В библиотеке нет сцен…» ДО каких-либо сетевых вызовов (queue_and_wait
    # патчем атрибута подменён на fail-loud заглушку — до неё дойти нельзя).
    import argparse

    with tempfile.TemporaryDirectory(prefix="empty-lib-") as tmp:
        tmp_root = Path(tmp)
        model_id = "m-empty"
        model_dir = tmp_root / "models" / model_id
        model_dir.mkdir(parents=True)
        (model_dir / "character.json").write_text(
            json.dumps({"name": "Empty", "age": 23}), encoding="utf-8"
        )
        # load_workflow() читает шаблон до проверки библиотеки — заглушка {}.
        (tmp_root / "comfy").mkdir(parents=True)
        (tmp_root / "comfy" / "workflow_kontext_variation.json").write_text("{}", encoding="utf-8")
        reference = tmp_root / "reference.jpg"
        reference.write_bytes(b"")

        # Пустая библиотека: и отсутствующий файл, и обёртка с пустым scenes → [].
        lib_path = tmp_root / "models" / "library.json"
        assert queue_workflow.library_scene_texts(str(lib_path), "expand") == []
        lib_path.write_text(json.dumps({"version": 1, "scenes": []}), encoding="utf-8")
        assert queue_workflow.library_scene_texts(str(lib_path), "expand") == []

        args = argparse.Namespace(
            model_id=model_id,
            reference=str(reference),
            prompt=None,  # без --prompt → ротация сцен из библиотеки
            negative="",
            uncensor=False,
            count=2,
            seed=1,
            url="http://127.0.0.1:9",
        )

        def _no_network(*_args, **_kwargs):
            raise AssertionError("queue_and_wait must not be called with an empty library")

        original_root = queue_workflow.ROOT
        original_queue = queue_workflow.queue_and_wait
        queue_workflow.ROOT = tmp_root  # не писать в реальный репозиторий
        queue_workflow.queue_and_wait = _no_network
        try:
            try:
                queue_workflow.expand_dataset(args)
            except SystemExit as exc:
                assert exc.code == "В библиотеке нет сцен режима expand — добавьте через UI", exc.code
            else:
                raise AssertionError("SystemExit expected for expand with an empty scene library")
        finally:
            queue_workflow.ROOT = original_root
            queue_workflow.queue_and_wait = original_queue


def test_stream_job_strips_skip_train_from_child_env():
    # SKIP_TRAIN is the fail-fast test seam of scripts/train_lora.sh: if it
    # ever reaches a job's child env (e.g. set on the server via compose/
    # systemd), real training is silently skipped in prod (job "done", no
    # LoRA). _stream_job must strip it from the env it passes along.
    captured: dict = {}

    class _FakeStdout:
        def __aiter__(self):
            async def _gen():
                return
                yield  # pragma: no cover  (never reached)

            return _gen()

    class _FakeProcess:
        returncode = 0

        def __init__(self) -> None:
            self.stdout = _FakeStdout()

        async def wait(self) -> int:
            return self.returncode

    async def _fake_exec(*args, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeProcess()

    original_exec = asyncio.create_subprocess_exec
    original_skip = os.environ.get("SKIP_TRAIN")
    os.environ["SKIP_TRAIN"] = "1"  # hostile server env
    job_id = "skiptrain-test"
    main.jobs[job_id] = {"id": job_id, "kind": "train", "status": "queued"}
    asyncio.create_subprocess_exec = _fake_exec
    try:
        _run(main._stream_job(job_id, ["bash", "scripts/train_lora.sh"]))
    finally:
        asyncio.create_subprocess_exec = original_exec
        if original_skip is None:
            os.environ.pop("SKIP_TRAIN", None)
        else:
            os.environ["SKIP_TRAIN"] = original_skip
        main.jobs.pop(job_id, None)

    env = captured["kwargs"].get("env")
    assert env is not None, "env must be passed to create_subprocess_exec"
    assert "SKIP_TRAIN" not in env, "SKIP_TRAIN must not reach the child process"
    assert env.get("PYTHONUNBUFFERED") == "1"


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
