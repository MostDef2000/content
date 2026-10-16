"""ComfyUI wake-up scope of management.main._launch (prod bug found on smoke
train-1792131743): _launch calls _ensure_comfy() for EVERY job kind, so
"Train FLUX" starts ComfyUI right next to the trainer — while spec 010 (hard
rule 3, 12 GB shared VRAM) requires ComfyUI to stay STOPPED during training.
Contract: kind="train" must NOT wake ComfyUI; generation kinds (post,
candidates) must keep waking it.

Isolation follows test_clear_dataset.py / test_engine_validation.py:
WORKSPACE points at a tempfile dir BEFORE management.main is imported, and
main._ensure_comfy / main._serial_job are monkeypatched at call time (a
recorder and a no-op), so no docker, no ComfyUI and no subprocess is touched.
pytest-compatible; pytest is not required.

Run either way:
    python -m pytest tests/test_train_no_comfy.py -q
    python tests/test_train_no_comfy.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # content/
sys.path.insert(0, str(REPO_ROOT))

_TMP_WORKSPACE = tempfile.TemporaryDirectory(prefix="launch-comfy-ws-")
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

MODEL_ID = "m1"


def _launch_probe(kind: str, cmd: list[str]) -> tuple[list[str], dict]:
    """Run main._launch(kind, cmd) with _ensure_comfy replaced by a recorder
    and _serial_job by a no-op (the created task must never reach
    subprocess/docker). Returns (ensure_comfy call log, launch result)."""
    calls: list[str] = []

    async def _fake_ensure_comfy() -> str | None:
        calls.append(kind)
        return None

    async def _fake_serial_job(job_id: str, _cmd: list[str]) -> None:
        pass

    original_ensure = main._ensure_comfy
    original_serial = main._serial_job
    jobs_snapshot = dict(main.jobs)
    main._ensure_comfy = _fake_ensure_comfy
    main._serial_job = _fake_serial_job
    main.jobs.clear()  # no "running" job may 409 the probe
    try:
        async def _scenario():
            result = await main._launch(kind, cmd, model_id=MODEL_ID)
            await asyncio.sleep(0)  # let the (patched) job task start and finish
            return result

        result = asyncio.run(_scenario())
    finally:
        main._ensure_comfy = original_ensure
        main._serial_job = original_serial
        main.jobs.clear()
        main.jobs.update(jobs_snapshot)
    return calls, result


def test_launch_train_does_not_wake_comfy():
    # TRAIN must not touch ComfyUI: the trainer needs the whole 12 GB card
    # (spec 010 limits: ComfyUI is stopped for the duration of a train run).
    # RED on current main.py: _launch calls _ensure_comfy() unconditionally.
    calls, result = _launch_probe("train", ["bash", "scripts/train_lora.sh", MODEL_ID, "flux"])
    assert result["status"] == "queued", result
    assert calls == [], (
        "_launch('train') must not call _ensure_comfy: training and ComfyUI "
        f"cannot share the 12 GB card (spec 010); recorder saw {calls}"
    )


def test_launch_post_still_wakes_comfy():
    # Generation keeps the wake-up contract: a post job must call
    # _ensure_comfy exactly once (regression guard for the fix).
    calls, result = _launch_probe("post", ["python", "scripts/queue_workflow.py", "post"])
    assert result["status"] == "queued", result
    assert calls == ["post"], calls


def test_launch_candidates_still_wakes_comfy():
    # Same for the candidates job kind — every non-train kind stays gated.
    calls, result = _launch_probe("candidates", ["python", "scripts/queue_workflow.py", "expand"])
    assert result["status"] == "queued", result
    assert calls == ["candidates"], calls


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
