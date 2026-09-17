"""VRAM monitor in /api/status (the "vram" field, feature: vram badge).

GET /api/status gains "vram": {"used", "total", "percent"} — MiB, percent to
1 decimal — from `nvidia-smi --query-gpu=memory.used,memory.total
--format=csv,noheader,nounits` (run in a worker thread, 5 s cap, same timeout
class as _comfy_up). The helper degrades to vram: null when nvidia-smi is
missing, times out, or emits unexpected output, so the rest of the payload
is never broken.

Checks:
- valid nvidia-smi output (mocked subprocess.run): vram carries
  used/total/percent and the payload keeps exactly its known keys;
- nvidia-smi unavailable (mocked subprocess.run raising FileNotFoundError):
  vram is null, the response is still valid, and the other fields keep
  their keys and types.

pytest-compatible; pytest is not required.

Run either way:
    python -m pytest tests/test_status_vram.py -q
    python tests/test_status_vram.py

WORKSPACE (and COMFY_URL, for a fast comfy_up failure) are pointed at a
tempfile directory / unreachable port BEFORE management.main is imported
(same protocol as tests/test_reference_persistence.py). In a full-suite run
the module may already be imported with another file's workspace — the
assertions below only rely on the /api/status payload shape, so that is
safe either way.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent  # content/
sys.path.insert(0, str(REPO_ROOT))

_TMP_WORKSPACE = tempfile.TemporaryDirectory(prefix="status-vram-ws-")
# StaticFiles(directory=...) is mounted at import time and requires the dir.
Path(_TMP_WORKSPACE.name, "management", "static").mkdir(parents=True, exist_ok=True)

_PREV_WORKSPACE = os.environ.get("WORKSPACE")
_PREV_COMFY_URL = os.environ.get("COMFY_URL")
os.environ["WORKSPACE"] = _TMP_WORKSPACE.name
os.environ["COMFY_URL"] = "http://127.0.0.1:1"  # fast refusal: comfy_up -> False
try:
    import management.main as main  # noqa: E402
finally:
    if _PREV_WORKSPACE is None:
        os.environ.pop("WORKSPACE", None)
    else:
        os.environ["WORKSPACE"] = _PREV_WORKSPACE
    if _PREV_COMFY_URL is None:
        os.environ.pop("COMFY_URL", None)
    else:
        os.environ["COMFY_URL"] = _PREV_COMFY_URL


def _run(coro):
    return asyncio.run(coro)


# The exact /api/status payload shape after the vram field landed.
EXPECTED_KEYS = {
    "comfy_up",
    "comfy_url",
    "gpu_busy",
    "disk_free_gb",
    "cpu_percent",
    "ram_percent",
    "vram",
    "active_jobs",
    "active_model",
}


def _fake_smi(used: str, total: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["nvidia-smi"], returncode=0, stdout=f"{used}, {total}\n", stderr=""
    )


def test_status_vram_present_with_valid_nvidia_smi():
    with mock.patch.object(main.subprocess, "run", return_value=_fake_smi("8423", "12288")) as fake_run:
        payload = _run(main.status())
    assert fake_run.called  # the handler really went through nvidia-smi
    assert payload["vram"] == {"used": 8423, "total": 12288, "percent": 68.5}
    assert set(payload) == EXPECTED_KEYS  # shape unchanged apart from "vram"


def test_status_vram_null_when_nvidia_smi_unavailable():
    with mock.patch.object(main.subprocess, "run", side_effect=FileNotFoundError("nvidia-smi")):
        payload = _run(main.status())
    assert payload["vram"] is None  # degraded, not broken
    assert set(payload) == EXPECTED_KEYS
    # the rest of the payload keeps its types
    assert isinstance(payload["comfy_up"], bool)
    assert isinstance(payload["gpu_busy"], bool)
    assert isinstance(payload["disk_free_gb"], float)
    assert isinstance(payload["cpu_percent"], float)
    assert isinstance(payload["ram_percent"], float)
    assert isinstance(payload["active_jobs"], list)


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
