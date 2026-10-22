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
Plus the LoRA-architecture trap (issue #2 п.15): _lora_arch sniffs the
safetensors header (unet keys → sdxl, transformer keys → flux, else None),
a flux post with an SDXL-arch LoRA 400s (and an sdxl post with a FLUX-arch
LoRA), matching-arch posts launch, undetectable arch stays fail-open, and
_active_model_summary reports the deployed slot as the recommended "engine".
Plus the dual-arch LoRA slots: the flux engine consumes the "lora_flux" slot
(<id>-flux.safetensors) and the sdxl engine the "lora" slot (<id>.safetensors)
— _lora_name_for_engine resolution, flux-post gating per slot, the SDXL-arch
backstop inside the FLUX slot, the summary engine per deployed slot, and the
train-start arch seam (default sdxl, flux cmd + registry slot, bogus 400).
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
import re
import struct
import sys
import tempfile
import tomllib
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


# In a shared pytest process management.main's module-level paths freeze at
# first import (first file's temp WORKSPACE). Re-point them at THIS file's
# temp WORKSPACE before the seeds below and re-bind per module at run time.
# Issue #17 two-root model: per-model DATA (and the runtime tree) live under
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
# Tracked configs live in the checkout (WORKSPACE/models/<id>); the data
# side of this model (posts/loras/etc.) is seeded per-test under main.DATA_ROOT.
_MODEL_CONFIG_DIR = main._model_config_dir(MODEL_ID)
_MODEL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
(_MODEL_CONFIG_DIR / "character.json").write_text(
    json.dumps({"name": "Valery", "age": 23, "appearance": {"face": "oval face", "build": "athletic"}}),
    encoding="utf-8",
)
(_MODEL_CONFIG_DIR / "prompt_profile.json").write_text(
    json.dumps({"version": 1, "face": "oval face", "body": "athletic", "style": "", "negative": "", "scenes": {}}),
    encoding="utf-8",
)
_MODEL_DIR = main.DATA_ROOT / "models" / MODEL_ID  # per-model DATA (reference/posts/...)


def _write_lora_stub(keys: list[str]) -> None:
    """Overwrite the default empty LoRA stub with a minimal SAFETENSORS file
    whose header carries the given tensor keys (restore via _reset_lora_stub)."""
    header = json.dumps(
        {k: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for k in keys}
    ).encode("utf-8")
    (main.LORAS_DIR / f"{MODEL_ID}.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\x00" * 4
    )


def _reset_lora_stub() -> None:
    (main.LORAS_DIR / f"{MODEL_ID}.safetensors").write_bytes(b"")


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


# --------------------------------------------------------------------------- #
# Dual-arch (FLUX slot) helpers. The module seed stays single-slot (m1 with
# "lora": <id>.safetensors, no "lora_flux") — flux-slot tests rewrite the
# registry per test and restore the seed JSON afterwards.
# --------------------------------------------------------------------------- #
_SEED_REGISTRY_JSON = main.REGISTRY_PATH.read_text(encoding="utf-8")
_FLUX_LORA = f"{MODEL_ID}-flux.safetensors"


def _seed_entry(**overrides) -> dict:
    """The seed registry entry for MODEL_ID with per-test overrides."""
    entry = {
        "id": MODEL_ID,
        "name": "Valery",
        "age": 23,
        "height_cm": 178,
        "tags": [],
        "active": True,
        "created": "2026-01-01T00:00:00Z",
        "lora": f"{MODEL_ID}.safetensors",
        "groups": [],
    }
    entry.update(overrides)
    return entry


def _with_registry(models: list[dict]):
    """Temporarily replace the registry with the given models list; the seed
    JSON is restored after the block (self-sufficient tests)."""
    import contextlib

    @contextlib.contextmanager
    def _patched():
        main.REGISTRY_PATH.write_text(
            json.dumps({"version": 1, "models": models}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        try:
            yield
        finally:
            main.REGISTRY_PATH.write_text(_SEED_REGISTRY_JSON, encoding="utf-8")

    return _patched()


def _write_flux_stub(keys: list[str]) -> None:
    """Write a minimal SAFETENSORS file with the given tensor keys into the
    FLUX slot (m1-flux.safetensors)."""
    header = json.dumps(
        {k: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for k in keys}
    ).encode("utf-8")
    (main.LORAS_DIR / _FLUX_LORA).write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 4)


def _remove_flux_stub() -> None:
    (main.LORAS_DIR / _FLUX_LORA).unlink(missing_ok=True)


def test_default_engine_is_flux():
    fields = main._validate_engine_fields({}, allow_engine=True)
    assert fields == {"engine": "flux", "uncensor": False}
    fields = main._validate_engine_fields({"engine": "flux"}, allow_engine=True)
    assert fields == {"engine": "flux", "uncensor": False}


def test_valid_engine_sdxl():
    # sdxl + uncensor: True → 400 with FLUX detail
    exc = _expect_http_status({"engine": "sdxl", "uncensor": True}, 400)
    assert "движка FLUX" in exc.detail, exc.detail
    # sdxl + uncensor: False/absent → success
    fields = main._validate_engine_fields({"engine": "sdxl", "uncensor": False}, allow_engine=True)
    assert fields == {"engine": "sdxl", "uncensor": False}


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


def test_post_cmd_sdxl_without_uncensor():
    spy = _LaunchSpy()
    with _with_spy(spy):
        _run(
            main.job_post(
                {
                    "caption": "a caption",
                    "prompt": "a quiet studio shot",
                    "engine": "sdxl",
                    "uncensor": False,
                }
            )
        )
    kind, cmd, model_id, job_id = spy.calls[-1]
    assert kind == "post" and model_id == MODEL_ID
    assert "--engine" in cmd, cmd
    assert cmd[cmd.index("--engine") + 1] == "sdxl", cmd
    assert "--uncensor" not in cmd, cmd
    assert main.jobs[job_id]["engine"] == "sdxl"

def test_post_sdxl_with_uncensor_400():
    spy = _LaunchSpy()
    with _with_spy(spy):
        try:
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
        except main.HTTPException as exc:
            assert exc.status_code == 400, exc.status_code
            assert "движка FLUX" in exc.detail, exc.detail
        else:
            raise AssertionError("HTTPException(400) expected for sdxl post with uncensor")
    assert spy.calls == [], "no job may be launched when sdxl post is 400ed by uncensor"


def test_post_flux_sdxl_arch_in_flux_slot_400():
    # Engine trap (issue #2 п.15), dual-arch backstop: an SDXL-arch LoRA
    # (unet keys) deployed into the FLUX slot must 400 a flux post — ComfyUI
    # would silently ignore the SDXL deltas on the FLUX transformer. A flux
    # post with it must 400 and launch nothing.
    spy = _LaunchSpy()
    try:
        _write_flux_stub(["lora_unet_down_blocks_0_attentions_0_to_k.lora_down.weight"])
        with _with_registry([_seed_entry(lora_flux=_FLUX_LORA)]), _with_spy(spy):
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
                assert "SDXL" in exc.detail, exc.detail
            else:
                raise AssertionError("HTTPException(400) expected for flux post with an SDXL-arch LoRA in the FLUX slot")
        assert spy.calls == [], "no job may be launched when the flux post is 400ed by the LoRA arch"
    finally:
        _remove_flux_stub()


def test_post_sdxl_with_flux_lora_400():
    # The mirror trap: a FLUX-arch LoRA (transformer keys) cannot load on the
    # SDXL (Pony) checkpoint — an sdxl post with it must 400 and launch
    # nothing.
    spy = _LaunchSpy()
    try:
        _write_lora_stub(["lora_transformer_blocks_0_attn.to_k.lora_down.weight"])
        with _with_spy(spy):
            try:
                _run(
                    main.job_post(
                        {
                            "caption": "a caption",
                            "prompt": "a quiet studio shot",
                            "engine": "sdxl",
                        }
                    )
                )
            except main.HTTPException as exc:
                assert exc.status_code == 400, exc.status_code
                assert "FLUX" in exc.detail, exc.detail
            else:
                raise AssertionError("HTTPException(400) expected for sdxl post with a FLUX-arch LoRA")
        assert spy.calls == [], "no job may be launched when the sdxl post is 400ed by the LoRA arch"
    finally:
        _reset_lora_stub()


def test_post_sdxl_with_sdxl_lora_ok():
    # Matching arch: an SDXL-arch LoRA passes the sdxl post (the graph carries
    # the LoRA loader) — the job launches with --engine sdxl, no --uncensor.
    spy = _LaunchSpy()
    try:
        _write_lora_stub(["lora_unet_down_blocks_0_attentions_0_to_k.lora_down.weight"])
        with _with_spy(spy):
            _run(
                main.job_post(
                    {
                        "caption": "a caption",
                        "prompt": "a quiet studio shot",
                        "engine": "sdxl",
                    }
                )
            )
        kind, cmd, model_id, _job_id = spy.calls[-1]
        assert kind == "post" and model_id == MODEL_ID
        assert "--engine" in cmd, cmd
        assert cmd[cmd.index("--engine") + 1] == "sdxl", cmd
        assert "--uncensor" not in cmd, cmd
    finally:
        _reset_lora_stub()


def test_post_flux_with_flux_lora_ok():
    # Matching arch: a FLUX-arch LoRA in the FLUX slot passes the flux post —
    # the job launches with --engine flux.
    spy = _LaunchSpy()
    try:
        _write_flux_stub(["lora_transformer_blocks_0_attn.to_k.lora_down.weight"])
        with _with_registry([_seed_entry(lora_flux=_FLUX_LORA)]), _with_spy(spy):
            _run(
                main.job_post(
                    {
                        "caption": "a caption",
                        "prompt": "a quiet studio shot",
                        "engine": "flux",
                    }
                )
            )
        kind, cmd, model_id, _job_id = spy.calls[-1]
        assert kind == "post" and model_id == MODEL_ID
        assert "--engine" in cmd, cmd
        assert cmd[cmd.index("--engine") + 1] == "flux", cmd
    finally:
        _remove_flux_stub()


def test_lora_name_for_engine():
    # Pure slot resolver: sdxl → registry "lora" (fallback <id>.safetensors),
    # flux → registry "lora_flux" (fallback <id>-flux.safetensors).
    assert main._lora_name_for_engine({"id": MODEL_ID, "lora": "custom.safetensors"}, "sdxl") == "custom.safetensors"
    assert main._lora_name_for_engine({"id": MODEL_ID}, "sdxl") == f"{MODEL_ID}.safetensors"
    assert (
        main._lora_name_for_engine({"id": MODEL_ID, "lora_flux": "custom-flux.safetensors"}, "flux")
        == "custom-flux.safetensors"
    )
    assert main._lora_name_for_engine({"id": MODEL_ID}, "flux") == f"{MODEL_ID}-flux.safetensors"


def test_post_flux_uses_flux_slot():
    # Dual-arch: a flux post resolves the FLUX slot (registry "lora_flux"),
    # not the SDXL slot — with ONLY the flux stub deployed (the SDXL seed
    # stub is removed for the duration, so the old single-slot resolver
    # would 400 on the missing m1.safetensors) the post launches.
    spy = _LaunchSpy()
    try:
        (main.LORAS_DIR / _FLUX_LORA).write_bytes(b"")  # empty stub: arch None → fail-open
        (main.LORAS_DIR / f"{MODEL_ID}.safetensors").unlink()  # SDXL slot absent
        with _with_registry([_seed_entry(lora_flux=_FLUX_LORA)]), _with_spy(spy):
            result = _run(
                main.job_post(
                    {
                        "caption": "a caption",
                        "prompt": "a quiet studio shot",
                        "engine": "flux",
                    }
                )
            )
        kind, cmd, model_id, job_id = spy.calls[-1]
        assert kind == "post" and model_id == MODEL_ID
        assert cmd[cmd.index("--engine") + 1] == "flux", cmd
        assert main.jobs[job_id]["engine"] == "flux"
        assert result == {"job_id": job_id, "status": "queued"}
    finally:
        _reset_lora_stub()  # restore the SDXL seed stub removed above
        _remove_flux_stub()


def test_post_flux_missing_flux_lora_400():
    # The registry points at the FLUX slot but the file was never copied: a
    # flux post must 400 with the expected filename in the detail and launch
    # nothing.
    spy = _LaunchSpy()
    with _with_registry([_seed_entry(lora_flux=_FLUX_LORA)]), _with_spy(spy):
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
            assert "ожидался m1-flux.safetensors" in exc.detail, exc.detail
        else:
            raise AssertionError("HTTPException(400) expected for a flux post without the FLUX LoRA")
    assert spy.calls == [], "no job may be launched when the FLUX LoRA is missing"


def test_lora_arch_detection():
    # Unit coverage of the header-sniffing helper: fail-open on missing/
    # empty/corrupt files, correct arch on valid headers.
    # Missing file → None.
    assert main._lora_arch(main.LORAS_DIR / "absent.safetensors") is None
    # Empty stub (default) → None.
    assert main._lora_arch(main.LORAS_DIR / f"{MODEL_ID}.safetensors") is None
    try:
        # JSON garbage / random bytes → None (struct unpack guard or decode).
        (main.LORAS_DIR / f"{MODEL_ID}.safetensors").write_bytes(b"\x99" * 32)
        assert main._lora_arch(main.LORAS_DIR / f"{MODEL_ID}.safetensors") is None
        # Valid headers → "sdxl" / "flux".
        _write_lora_stub(["lora_unet_down_blocks_0_attentions_0_to_k.lora_down.weight"])
        assert main._lora_arch(main.LORAS_DIR / f"{MODEL_ID}.safetensors") == "sdxl"
        _write_lora_stub(["lora_transformer_blocks_0_attn.to_k.lora_down.weight"])
        assert main._lora_arch(main.LORAS_DIR / f"{MODEL_ID}.safetensors") == "flux"
    finally:
        _reset_lora_stub()


def test_lora_arch_kohya_flux_naming_is_flux():
    # prod candidates-1792384501 (W3 acceptance): the FIRST LoRA trained
    # by kohya sd-scripts (spec 010 W3 trainer) produced a valid
    # FLUX+LoRA image (rc=0 — a REAL FLUX LoRA), yet manager _lora_arch
    # (main.py:503) reported "sdxl" for it, and /api/jobs/group +
    # /api/jobs/post (engine=flux) now reject the freshly trained LoRA
    # with «LoRA — SDXL (Pony) архитектура» (/api/jobs/candidates is
    # unaffected — no sniff). Mechanism: the sniffer was written for
    # ai-toolkit naming (SDXL "lora_unet_*", FLUX "lora_transformer_*")
    # and checks any("unet" in key) FIRST — while kohya (sd-scripts)
    # names FLUX tensors lora_unet_double_blocks_*/lora_unet_single_
    # blocks_* (the word "unet" is present in FLUX keys too). Fix
    # direction (code worker): FLUX-specific discriminators FIRST —
    # kohya FLUX keys contain "double_blocks"/"single_blocks" (also
    # img_mod_lin/txt_mod_lin variants for other trainers); only if none
    # match, fall back to "unet"→sdxl / "transformer"→flux; else None
    # (fail-open unchanged). Landed RED-first against the pre-fix sniffer
    # (base ce26d5e): the kohya keys hit any("unet" in key) → "sdxl" —
    # the exact prod misclassification. PERMANENT guard: red signature
    # (assert 'sdxl' == 'flux') if the FLUX-token-first order is reverted.
    try:
        _write_flux_stub(["lora_unet_double_blocks_0_img_mod_lin.lora_down.weight"])
        assert main._lora_arch(main.LORAS_DIR / _FLUX_LORA) == "flux", (
            "kohya (sd-scripts) FLUX LoRA keys carry BOTH 'unet' and "
            "'double_blocks' — the FLUX discriminator must win over the "
            "ai-toolkit 'unet' heuristic (prod candidates-1792384501 "
            "misclassified the freshly trained LoRA as sdxl)"
        )
        _write_flux_stub(["lora_unet_single_blocks_19_modulation_lin.lora_down.weight"])
        assert main._lora_arch(main.LORAS_DIR / _FLUX_LORA) == "flux", (
            "the kohya single_blocks key must classify as flux too"
        )
        _write_flux_stub([
            "lora_unet_double_blocks_0_img_mod_lin.lora_down.weight",
            "lora_unet_single_blocks_19_modulation_lin.lora_down.weight",
            "lora_unet_txt_norm.weight",
        ])
        assert main._lora_arch(main.LORAS_DIR / _FLUX_LORA) == "flux", (
            "a full kohya FLUX header must classify as flux"
        )
    finally:
        _remove_flux_stub()


def test_lora_arch_ai_toolkit_mapping_and_failopen_unchanged():
    # Sibling guard for the fix reorder — GREEN today (it mirrors the
    # landed ai-toolkit behavior), its bite is against the fix: putting
    # the kohya FLUX discriminators first must NOT break the existing
    # mappings. ai-toolkit SDXL keys (no double_blocks/single_blocks)
    # stay "sdxl"; lora_transformer_* stays "flux"; an absent FLUX-slot
    # file stays None (fail-open: callers keep current behavior instead
    # of a false 400).
    try:
        _write_flux_stub(["lora_unet_down_blocks_0_attentions_0_to_k.lora_down.weight"])
        assert main._lora_arch(main.LORAS_DIR / _FLUX_LORA) == "sdxl", (
            "ai-toolkit SDXL naming must keep classifying sdxl after the "
            "kohya-FLUX discriminator lands"
        )
        _write_flux_stub(["lora_transformer_blocks_0_attn.to_k.lora_down.weight"])
        assert main._lora_arch(main.LORAS_DIR / _FLUX_LORA) == "flux", (
            "ai-toolkit FLUX naming must keep classifying flux"
        )
    finally:
        _remove_flux_stub()
    assert main._lora_arch(main.LORAS_DIR / _FLUX_LORA) is None, (
        "absent FLUX-slot file -> None (fail-open unchanged)"
    )


def test_active_model_summary_engine():
    # Dual-arch: "engine" reports which LoRA slot is deployed on disk — sdxl
    # whenever the sdxl-slot file exists (even if its header sniffs another
    # arch or is an empty stub: the arch trap is the post/group 400 guard's
    # business), flux only when the sdxl slot is absent but the flux slot is
    # deployed, None when no slot file exists.
    try:
        _write_lora_stub(["lora_unet_down_blocks_0_attentions_0_to_k.lora_down.weight"])
        summary = main._active_model_summary({"id": "m1", "name": "Valery", "lora": "m1.safetensors"})
        assert summary["engine"] == "sdxl", summary
        assert summary["lora_file"] is True, summary
        _write_lora_stub(["lora_transformer_blocks_0_attn.to_k.lora_down.weight"])
        summary = main._active_model_summary({"id": "m1", "name": "Valery", "lora": "m1.safetensors"})
        assert summary["engine"] == "sdxl", summary
        _reset_lora_stub()
        summary = main._active_model_summary({"id": "m1", "name": "Valery", "lora": "m1.safetensors"})
        assert summary["engine"] == "sdxl", summary  # empty stub: the slot file exists
        summary = main._active_model_summary({"id": "m1", "name": "Valery", "lora": None})
        assert summary["engine"] is None, summary
        assert summary["lora_file"] is False, summary
    finally:
        _reset_lora_stub()


def test_active_model_summary_engine_dualarch():
    # Both slots deployed → sdxl wins; only the FLUX slot deployed → flux;
    # neither file → None.
    try:
        (main.LORAS_DIR / f"{MODEL_ID}.safetensors").write_bytes(b"")
        (main.LORAS_DIR / _FLUX_LORA).write_bytes(b"")
        summary = main._active_model_summary(
            {"id": MODEL_ID, "name": "Valery", "lora": f"{MODEL_ID}.safetensors", "lora_flux": _FLUX_LORA}
        )
        assert summary["engine"] == "sdxl", summary
        (main.LORAS_DIR / f"{MODEL_ID}.safetensors").unlink()
        summary = main._active_model_summary(
            {"id": MODEL_ID, "name": "Valery", "lora": f"{MODEL_ID}.safetensors", "lora_flux": _FLUX_LORA}
        )
        assert summary["engine"] == "flux", summary
        _remove_flux_stub()
        summary = main._active_model_summary(
            {"id": MODEL_ID, "name": "Valery", "lora": f"{MODEL_ID}.safetensors", "lora_flux": _FLUX_LORA}
        )
        assert summary["engine"] is None, summary
    finally:
        _reset_lora_stub()
        _remove_flux_stub()


def test_group_cmd_has_no_engine_or_uncensor():
    # The group is FLUX-only, so its preflight resolves the members' FLUX
    # slots: the fixture gives m1 a deployed flux stub for the duration.
    spy = _LaunchSpy()
    try:
        (main.LORAS_DIR / _FLUX_LORA).write_bytes(b"")
        with _with_registry([_seed_entry(lora_flux=_FLUX_LORA)]), _with_spy(spy):
            _run(main.job_group({"models": [MODEL_ID], "prompt": "a shared scene", "caption": "a caption"}))
        kind, cmd, model_id, _job_id = spy.calls[-1]
        assert kind == "group" and model_id == MODEL_ID
        assert "--engine" not in cmd, cmd
        assert "--uncensor" not in cmd, cmd
    finally:
        _remove_flux_stub()


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
    try:
        (main.LORAS_DIR / _FLUX_LORA).write_bytes(b"")
        with _with_registry([_seed_entry(lora_flux=_FLUX_LORA)]), _with_spy(spy):
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
    finally:
        _remove_flux_stub()


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
    try:
        (main.LORAS_DIR / _FLUX_LORA).write_bytes(b"")
        with _with_registry([_seed_entry(lora_flux=_FLUX_LORA)]), _with_spy(spy):
            _run(main.job_group({"models": [MODEL_ID], "prompt": "a shared scene", "caption": "a caption"}))
        kind, cmd, model_id, job_id = spy.calls[-1]
        assert kind == "group" and model_id == MODEL_ID
        assert "--seed" in cmd, cmd
        seed = int(cmd[cmd.index("--seed") + 1])
        assert 0 <= seed < 2**31, seed
        assert main.jobs[job_id]["seed"] == seed
    finally:
        _remove_flux_stub()


def test_post_empty_caption_accepted_no_400():
    # Optional caption (Instagram post text): an empty/blank caption must not
    # 400; the cmd still carries --caption "" and the job record stores null.
    spy = _LaunchSpy()
    try:
        (main.LORAS_DIR / _FLUX_LORA).write_bytes(b"")
        with _with_registry([_seed_entry(lora_flux=_FLUX_LORA)]), _with_spy(spy):
            result = _run(main.job_post({"caption": "   ", "prompt": "a quiet studio shot"}))
        kind, cmd, model_id, job_id = spy.calls[-1]
        assert kind == "post" and model_id == MODEL_ID
        assert result == {"job_id": job_id, "status": "queued"}
        assert "--caption" in cmd, cmd
        assert cmd[cmd.index("--caption") + 1] == "", cmd
        assert main.jobs[job_id]["caption"] is None
    finally:
        _remove_flux_stub()


def test_post_missing_caption_key_accepted():
    # No "caption" key in the payload at all → same contract as an empty one.
    spy = _LaunchSpy()
    try:
        (main.LORAS_DIR / _FLUX_LORA).write_bytes(b"")
        with _with_registry([_seed_entry(lora_flux=_FLUX_LORA)]), _with_spy(spy):
            result = _run(main.job_post({"prompt": "a quiet studio shot"}))
        kind, cmd, model_id, job_id = spy.calls[-1]
        assert kind == "post" and model_id == MODEL_ID
        assert result == {"job_id": job_id, "status": "queued"}
        assert cmd[cmd.index("--caption") + 1] == "", cmd
        assert main.jobs[job_id]["caption"] is None
    finally:
        _remove_flux_stub()


def test_post_filled_caption_unchanged():
    # A filled caption behaves exactly as before the change.
    spy = _LaunchSpy()
    try:
        (main.LORAS_DIR / _FLUX_LORA).write_bytes(b"")
        with _with_registry([_seed_entry(lora_flux=_FLUX_LORA)]), _with_spy(spy):
            _run(main.job_post({"caption": "hello #fitness", "prompt": "a quiet studio shot"}))
        kind, cmd, model_id, job_id = spy.calls[-1]
        assert kind == "post" and model_id == MODEL_ID
        assert cmd[cmd.index("--caption") + 1] == "hello #fitness", cmd
        assert main.jobs[job_id]["caption"] == "hello #fitness"
    finally:
        _remove_flux_stub()


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
        data = Path(tmp) / "data"  # VALERY_DATA_ROOT (#17): per-model posts + runtime loras
        data.mkdir()
        out_root = Path(tmp) / "comfy-output"
        model_id = "pm1"
        (root / "models" / model_id).mkdir(parents=True)
        (root / "models" / "registry.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "models": [
                        {
                            "id": model_id,
                            "active": True,
                            "lora": f"{model_id}.safetensors",
                            "lora_flux": f"{model_id}-flux.safetensors",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (root / "models" / model_id / "character.json").write_text(
            json.dumps({"name": "Valery", "age": 23}), encoding="utf-8"
        )
        loras = data / "runtime" / "models" / "loras"
        loras.mkdir(parents=True)
        (loras / f"{model_id}-flux.safetensors").write_bytes(b"")  # flux preflight: the FLUX slot must exist
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
            destination = data / "models" / model_id / "posts" / name
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
            queue_workflow.DATA_ROOT = original_data_root
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
    SDXL_CKPT_PATH (bound to the real DATA_ROOT at import time), so it is
    patched to a stub in the temp tree as well."""
    import argparse

    with tempfile.TemporaryDirectory(prefix="gw-post-sdxl-") as tmp:
        root = Path(tmp) / "repo"
        data = Path(tmp) / "data"  # VALERY_DATA_ROOT (#17): per-model posts + runtime loras/ckpt
        data.mkdir()
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
        loras = data / "runtime" / "models" / "loras"
        loras.mkdir(parents=True)
        if with_lora:
            (loras / f"{SDXL_MODEL_ID}.safetensors").write_bytes(b"")  # existence is what matters
        ckpt = data / "runtime" / "models" / "checkpoints" / "ponyDiffusionV6XL.safetensors"
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
        original_data_root = queue_workflow.DATA_ROOT
        original_out_root = queue_workflow.OUTPUT_ROOT
        original_queue = queue_workflow.queue_and_wait
        original_ckpt = queue_workflow.SDXL_CKPT_PATH
        queue_workflow.ROOT = root
        queue_workflow.DATA_ROOT = data
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
            destination = data / "models" / SDXL_MODEL_ID / "posts" / "sdxl-post"
            return {
                "destination_exists": destination.is_dir(),
                "has_photo": (destination / "photo.png").is_file(),
                # The workflow dict itself outlives the temp tree (it is a
                # plain in-memory object), so the test can inspect it directly.
                "workflow": captured.get("workflow"),
            }
        finally:
            queue_workflow.ROOT = original_root
            queue_workflow.DATA_ROOT = original_data_root
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


def test_train_config_flux_kohya_templates_are_valid_toml_as_is():
    # The dual-arch FLUX templates (spec 010, kohya sd-scripts; rendered by
    # train_lora.sh <id> flux) must parse with tomllib WITHOUT rendering:
    # numeric placeholders are kept QUOTED ("{{STEPS}}" etc.) precisely so the
    # templates stay valid TOML as committed — that quoting is the invariant
    # this test pins. The render step swaps the quoted numeric placeholders
    # for bare integers (sd-scripts requires bare ints there).
    main_tpl = tomllib.loads(
        (REPO_ROOT / "lora" / "train_config_flux_kohya.template.toml").read_text(encoding="utf-8")
    )
    ds_tpl = tomllib.loads(
        (REPO_ROOT / "lora" / "train_config_flux_kohya_dataset.template.toml").read_text(encoding="utf-8")
    )
    # Quoted placeholders parse as strings — the templates are unrendered.
    assert main_tpl["max_train_steps"] == "{{STEPS}}"
    assert main_tpl["seed"] == "{{SEED}}"
    assert main_tpl["save_every_n_steps"] == "{{STEPS}}"
    assert main_tpl["output_name"] == "{{MODEL_ID}}_flux_v1"
    assert main_tpl["output_dir"] == "{{OUTPUT_DIR}}"
    for key in ("pretrained_model_name_or_path", "ae", "clip_l", "t5xxl"):
        assert main_tpl[key].startswith("{{") and main_tpl[key].endswith("}}"), key
    assert ds_tpl["datasets"][0]["resolution"] == "{{RESOLUTION}}"
    assert ds_tpl["datasets"][0]["batch_size"] == "{{BATCH}}"
    assert ds_tpl["datasets"][0]["subsets"][0]["image_dir"] == "{{DATASET_DIR}}"


def test_train_config_flux_kohya_template_low_vram_flags():
    # The 12 GB economical profile the spec mandates (the whole reason the flux
    # branch moved from ai-toolkit to kohya sd-scripts): fp8 base + CPU block
    # swap + disk caches, the FLUX network module, the FLUX sampler/guidance
    # knobs and the "<id>_flux_v1" output name that deploy_flux_lora searches.
    main_tpl = tomllib.loads(
        (REPO_ROOT / "lora" / "train_config_flux_kohya.template.toml").read_text(encoding="utf-8")
    )
    assert main_tpl["fp8_base"] is True
    assert main_tpl["blocks_to_swap"] == 20
    assert main_tpl["cache_text_encoder_outputs"] is True
    assert main_tpl["cache_text_encoder_outputs_to_disk"] is True
    assert main_tpl["cache_latents_to_disk"] is True
    assert main_tpl["network_module"] == "networks.lora_flux"
    assert main_tpl["network_dim"] == 32
    assert main_tpl["network_alpha"] == 16
    assert main_tpl["timestep_sampling"] == "shift"
    assert main_tpl["guidance_scale"] == 1.0
    assert main_tpl["output_name"] == "{{MODEL_ID}}_flux_v1"


def test_train_config_flux_kohya_dataset_template_sections_exact():
    # sd-scripts validates --dataset_config with a strict voluptuous schema
    # that REJECTS any non-dataset key/section, so the dataset template may
    # contain exactly [general] / [[datasets]] / [[datasets.subsets]] and
    # nothing else (a merged script-args + dataset file cannot serve both
    # options — hence the second template).
    raw = (REPO_ROOT / "lora" / "train_config_flux_kohya_dataset.template.toml").read_text(encoding="utf-8")
    headers = re.findall(r"^\s*\[+[A-Za-z_.]+\]+\s*$", raw, re.MULTILINE)
    assert sorted(h.strip() for h in headers) == sorted(
        ["[general]", "[[datasets]]", "[[datasets.subsets]]"]
    ), f"unexpected section headers: {headers}"
    ds_tpl = tomllib.loads(raw)
    assert set(ds_tpl) == {"general", "datasets"}, "only [general] + [[datasets]] at top level"
    assert len(ds_tpl["datasets"]) == 1
    assert set(ds_tpl["datasets"][0]) == {"resolution", "batch_size", "subsets"}
    assert len(ds_tpl["datasets"][0]["subsets"]) == 1
    assert set(ds_tpl["datasets"][0]["subsets"][0]) == {"image_dir", "num_repeats"}
    # No script-args keys leaked into the dataset file (would abort the run).
    for forbidden in ("pretrained_model_name_or_path", "max_train_steps", "network_module", "fp8_base"):
        assert forbidden not in ds_tpl, f"{forbidden!r} must not appear in the dataset template"


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
        original_data_root = queue_workflow.DATA_ROOT
        original_queue = queue_workflow.queue_and_wait
        queue_workflow.ROOT = tmp_root  # не писать в реальный репозиторий
        queue_workflow.DATA_ROOT = tmp_root  # (#17) runtime/input и dataset тоже под DATA_ROOT
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
            queue_workflow.DATA_ROOT = original_data_root
            queue_workflow.queue_and_wait = original_queue


def test_train_start_default_arch_is_sdxl():
    # POST /api/train/start without arch: back-compat default "sdxl" — the cmd
    # carries "sdxl" as the second positional argument and the registry entry
    # gains the SDXL slot name ("lora").
    spy = _LaunchSpy()
    with _with_registry([_seed_entry()]), _with_spy(spy):
        result = _run(main.train_start(None))
    kind, cmd, model_id, job_id = spy.calls[-1]
    assert kind == "train" and model_id == MODEL_ID
    assert cmd[:2] == ["bash", "scripts/train_lora.sh"], cmd
    assert cmd[2] == MODEL_ID and cmd[3] == "sdxl", cmd
    assert result == {"job_id": job_id, "status": "queued"}


def test_train_start_arch_flux_cmd_and_registry():
    # arch="flux": the cmd carries "flux" as the second positional argument
    # and the registry entry gains the FLUX slot ("lora_flux"), not "lora".
    spy = _LaunchSpy()
    with _with_registry([_seed_entry()]), _with_spy(spy):
        _run(main.train_start({"arch": "flux"}))
        entry = next(
            m for m in json.loads(main.REGISTRY_PATH.read_text(encoding="utf-8"))["models"]
            if m["id"] == MODEL_ID
        )
    kind, cmd, model_id, _job_id = spy.calls[-1]
    assert kind == "train" and model_id == MODEL_ID
    assert cmd[2] == MODEL_ID and cmd[3] == "flux", cmd
    assert entry.get("lora_flux") == _FLUX_LORA, entry
    assert entry.get("lora") == f"{MODEL_ID}.safetensors", entry


def test_train_start_arch_bogus_400():
    # arch outside {sdxl, flux} → 400 and nothing launched.
    spy = _LaunchSpy()
    with _with_registry([_seed_entry()]), _with_spy(spy):
        try:
            _run(main.train_start({"arch": "bogus"}))
        except main.HTTPException as exc:
            assert exc.status_code == 400, exc.status_code
            assert "arch" in exc.detail, exc.detail
        else:
            raise AssertionError("HTTPException(400) expected for a bogus train arch")
        entry = next(
            m for m in json.loads(main.REGISTRY_PATH.read_text(encoding="utf-8"))["models"]
            if m["id"] == MODEL_ID
        )
        assert "lora_flux" not in entry, "a rejected arch must not touch the registry"
    assert spy.calls == [], "no job may be launched when the train arch is invalid"


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
