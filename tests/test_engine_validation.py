"""Import-safe tests for management.main generation-engine validation
(feature Phase 4: SDXL/Pony engine for post, optional uncensor LoRA).
Covers the pure _validate_engine_fields validator (default flux, valid sdxl,
invalid engine 400, non-bool uncensor 400, engine field dropped when
allow_engine=False) and the cmd lines built by the job endpoints: post carries
--engine sdxl (plus --uncensor when true) and the job record gains "engine",
while group carries neither flag. Also covers the flux-LoRA gating: a missing
LoRA file must 400 a flux post but not an sdxl post (the SDXL graph has no
LoraLoader), and candidates/expand carry --uncensor when requested.
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
    # The SDXL (Pony) graph has no LoraLoader: a missing character LoRA must
    # not 400 an sdxl post (defect: sdxl posts were blocked until a FLUX LoRA
    # was trained).
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
