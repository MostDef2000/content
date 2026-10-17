"""Import-safe tests for management.main._validate_group_payload (feature 007,
group-photos): the pure validation helper behind POST /api/jobs/group.
Covers the happy path passthrough, model-count/uniqueness 422s, unknown-model
404, missing-FLUX-LoRA 400 (dual-arch: the group resolves each member's FLUX
slot "lora_flux", fallback <id>-flux.safetensors), the SDXL-arch-in-FLUX-slot
backstop 400, caption rules, the prompt/scene_id XOR, layout/mode
allowlists, scene resolution from models/library.json and the post-name
collision 409. pytest-compatible; pytest is not required.

Run either way:
    python -m pytest tests/test_group_validation.py -q
    python tests/test_group_validation.py

WORKSPACE is pointed at a tempfile directory BEFORE management.main is
imported (the module resolves WORKSPACE and mounts /static at import time),
so the tests never depend on the real /workspace layout. The registry, LoRA
stubs and scene library are written into the temp WORKSPACE after import —
main reads registry.json/library.json from disk on every call.

Also covers the GET /api/jobs log_lines query parameter (spec 010 follow-up:
the hard-coded lines[-40:] tail hides the trainer's root-cause lines): the
40-line default stays back-compatible, an explicit size returns exactly that
many LAST lines, the upper bound is 2000 (clamp or 422 — both accepted), and
garbage/negative values 422 FastAPI-natively. These run through a minimal
stdlib ASGI GET harness against the real app (no httpx/TestClient in the CI
deps, and Query() validation only exists at the request layer).
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # content/
sys.path.insert(0, str(REPO_ROOT))

_TMP_WORKSPACE = tempfile.TemporaryDirectory(prefix="group-ws-")
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
def _bind_paths() -> None:
    ws = Path(_TMP_WORKSPACE.name)
    main.WORKSPACE = ws
    main.JOBS_DIR = ws / "runtime" / "jobs"
    main.MODELS_DIR = ws / "models"
    main.REGISTRY_PATH = ws / "models" / "registry.json"
    main.LIBRARY_PATH = ws / "models" / "library.json"
    main.LORAS_DIR = ws / "runtime" / "models" / "loras"
    main.TRASH_DIR = ws / "runtime" / "trash"
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


def _model_entry(model_id: str, lora: str, *, active: bool, lora_flux: str | None = None) -> dict:
    return {
        "id": model_id,
        "name": model_id,
        "age": 23,
        "height_cm": 178,
        "tags": [],
        "active": active,
        "created": "2026-01-01T00:00:00Z",
        "lora": lora,
        "lora_flux": lora_flux,
        "groups": [],
    }


# Fixture (written after import; main re-reads the registry/library from disk
# on every call). m1/m2 carry both LoRA slots (empty flux stubs: arch None →
# fail-open). m3 exists in the registry with only the SDXL slot ("lora") and
# NO "lora_flux" key at all — its FLUX-slot file (fallback name
# m3-flux.safetensors) is absent from LORAS_DIR: the missing-FLUX-LoRA case.
main.REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
main.REGISTRY_PATH.write_text(
    json.dumps(
        {
            "version": 1,
            "models": [
                _model_entry("m1", "m1.safetensors", active=True, lora_flux="m1-flux.safetensors"),
                _model_entry("m2", "m2.safetensors", active=False, lora_flux="m2-flux.safetensors"),
                _model_entry("m3", "m3.safetensors", active=False),
            ],
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
main.LORAS_DIR.mkdir(parents=True, exist_ok=True)
(main.LORAS_DIR / "m1.safetensors").write_bytes(b"")  # empty stubs: existence is what matters
(main.LORAS_DIR / "m2.safetensors").write_bytes(b"")
(main.LORAS_DIR / "m1-flux.safetensors").write_bytes(b"")
(main.LORAS_DIR / "m2-flux.safetensors").write_bytes(b"")
SCENE_TEXT = "a quiet studio shot"
main.LIBRARY_PATH.write_text(
    json.dumps({"version": 1, "scenes": [{"id": "sc1", "name": "Studio",
                                          "mode": "post", "text": SCENE_TEXT,
                                          "tags": []}]}),
    encoding="utf-8",
)


def _valid_payload(**overrides) -> dict:
    payload = {
        "models": ["m1", "m2"],
        "prompt": "p",
        "layout": "grid2",
        "mode": "private",
        "caption": "c",
        "seed": 5,
        "lora_strength": 0.9,
        "name": "grp1",
    }
    payload.update(overrides)
    return payload


def _expect_http_status(payload: dict, status: int) -> main.HTTPException:
    try:
        main._validate_group_payload(payload)
    except main.HTTPException as exc:
        assert exc.status_code == status, f"expected {status}, got {exc.status_code}"
        return exc
    raise AssertionError(
        f"HTTPException({status}) expected, _validate_group_payload returned normally"
    )


def test_valid_group_payload_passthrough():
    plan = main._validate_group_payload(_valid_payload())
    assert plan["ids"] == ["m1", "m2"]
    assert [entry["id"] for entry in plan["entries"]] == ["m1", "m2"]
    assert plan["prompt_text"] == "p"
    assert plan["layout"] == "grid2"
    assert plan["mode"] == "private"
    assert plan["caption"] == "c"
    assert plan["seed"] == 5
    assert plan["strength"] == 0.9
    assert plan["name"] == "grp1"
    assert plan["negative"] == ""


def test_more_than_five_models_422():
    exc = _expect_http_status(_valid_payload(models=["m1", "m2", "m3", "m4", "m5", "m6"]), 422)
    assert "1..5" in exc.detail, exc.detail


def test_duplicate_model_ids_422():
    exc = _expect_http_status(_valid_payload(models=["m1", "m1"]), 422)
    assert "unique" in exc.detail, exc.detail


def test_empty_models_list_422():
    _expect_http_status(_valid_payload(models=[]), 422)


def test_unknown_model_404():
    exc = _expect_http_status(_valid_payload(models=["zzz"]), 404)
    assert exc.detail == "model not found", exc.detail


def test_member_without_lora_file_400():
    # m3 has only the SDXL slot ("lora") and no "lora_flux" at all: the group
    # (FLUX-only) resolves the FLUX slot to the fallback m3-flux.safetensors,
    # which is absent from LORAS_DIR → 400.
    exc = _expect_http_status(_valid_payload(models=["m1", "m3"]), 400)
    assert "LoRA" in exc.detail, exc.detail
    assert "FLUX" in exc.detail, exc.detail


def test_group_member_without_flux_lora_400():
    # A member fine for SDXL posts (SDXL slot deployed) but without a FLUX
    # slot: the group (FLUX-only) must 400 with the model id and "FLUX" in the
    # detail (the fallback <id>-flux.safetensors file does not exist either).
    seed = json.loads(main.REGISTRY_PATH.read_text(encoding="utf-8"))
    models = [
        {**entry, "lora_flux": None} if entry.get("id") == "m1" else entry
        for entry in seed["models"]
    ]
    main.REGISTRY_PATH.write_text(json.dumps({"version": 1, "models": models}), encoding="utf-8")
    stub = main.LORAS_DIR / "m1-flux.safetensors"
    stub_existed = stub.is_file()
    if stub_existed:
        stub.unlink()
    try:
        exc = _expect_http_status(_valid_payload(models=["m1", "m2"]), 400)
        assert "m1" in exc.detail, exc.detail
        assert "FLUX" in exc.detail, exc.detail
    finally:
        main.REGISTRY_PATH.write_text(json.dumps(seed), encoding="utf-8")
        if stub_existed:
            stub.write_bytes(b"")


def test_group_member_with_sdxl_lora_400():
    # Group shots are FLUX-only: a member whose FLUX-slot LoRA is SDXL-arch
    # (unet keys in the safetensors header of m1-flux.safetensors) must 400
    # with the model id and "SDXL" in the detail (empty/corrupt stubs stay
    # fail-open → arch None → pass).
    stub = main.LORAS_DIR / "m1-flux.safetensors"
    header = json.dumps(
        {"lora_unet_down_blocks_0_attentions_0_to_k.lora_down.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ).encode("utf-8")
    stub.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 4)
    try:
        exc = _expect_http_status(_valid_payload(models=["m1", "m2"]), 400)
        assert "m1" in exc.detail, exc.detail
        assert "SDXL" in exc.detail, exc.detail
    finally:
        stub.write_bytes(b"")  # restore the empty seed stub (self-sufficient test)


def test_empty_caption_400():
    _expect_http_status(_valid_payload(caption=""), 400)


def test_missing_caption_key_400():
    payload = _valid_payload()
    payload.pop("caption")
    _expect_http_status(payload, 400)


def test_no_prompt_no_scene_id_400():
    payload = _valid_payload()
    payload.pop("prompt")  # no scene_id either
    _expect_http_status(payload, 400)


def test_prompt_and_scene_id_together_400():
    _expect_http_status(_valid_payload(scene_id="sc1"), 400)  # prompt "p" still set


def test_invalid_layout_422():
    _expect_http_status(_valid_payload(layout="weird"), 422)


def test_invalid_mode_422():
    _expect_http_status(_valid_payload(mode="secret"), 422)


def test_scene_id_resolves_text_from_library():
    payload = _valid_payload()
    payload.pop("prompt")
    payload["scene_id"] = "sc1"
    plan = main._validate_group_payload(payload)
    assert plan["prompt_text"] == SCENE_TEXT


def test_name_collision_409():
    (main.MODELS_DIR / "m1" / "posts" / "grp2").mkdir(parents=True)
    exc = _expect_http_status(_valid_payload(name="grp2", mode="public"), 409)
    assert exc.detail == "post name already exists", exc.detail


# ---------------------------------------------------------------------------
# GET /api/jobs log_lines (tail-window query parameter)
# ---------------------------------------------------------------------------
# The endpoint accepts log_lines: int = Query(default=40, ge=1, le=2000)
# (spec 010 follow-up: the trainer's root error sits ABOVE the fixed 40-line
# tail): the default stays 40 for back-compat, an explicit size returns
# exactly that many LAST lines, and anything outside 1..2000 — garbage,
# zero/negative, or above the upper bound — is a FastAPI-native 422 raised
# before the handler runs. No TestClient/httpx in the CI deps, so a minimal
# stdlib ASGI GET harness drives the REAL app: that validation only exists
# at the request layer, calling the coroutine directly cannot pin it.

LOG_JOB_ID = "loglines-job"


def _seed_log_job(line_count: int) -> None:
    main.JOBS_DIR.mkdir(parents=True, exist_ok=True)
    log = main.JOBS_DIR / f"{LOG_JOB_ID}.log"
    log.write_text("".join(f"line {i:05d}\n" for i in range(1, line_count + 1)), encoding="utf-8")
    main.jobs[LOG_JOB_ID] = {
        "id": LOG_JOB_ID,
        "kind": "train",
        "model_id": "m1",
        "status": "failed",
        "log_path": str(log),
    }


def _drop_log_job() -> None:
    main.jobs.pop(LOG_JOB_ID, None)
    (main.JOBS_DIR / f"{LOG_JOB_ID}.log").unlink(missing_ok=True)


async def _api_get(path: str, query: str = "") -> tuple[int, str]:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": query.encode("utf-8"),
        "root_path": "",
        "headers": [(b"host", b"testserver")],
        "client": ("test", 1234),
        "server": ("testserver", 80),
    }

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    status: list[int] = []
    body: list[bytes] = []

    async def send(message: dict) -> None:
        if message["type"] == "http.response.start":
            status.append(message["status"])
        elif message["type"] == "http.response.body":
            body.append(message.get("body", b""))

    await main.app(scope, receive, send)
    assert status, f"ASGI harness got no response for GET {path}?{query}"
    return status[0], b"".join(body).decode("utf-8", errors="replace")


def _jobs_tail(query: str = "") -> tuple[int, list[str] | None, str]:
    """GET /api/jobs?<query>; returns (status, seeded job's log_tail or None,
    raw body)."""
    status, raw = asyncio.run(_api_get("/api/jobs", query))
    tail = None
    if status == 200:
        entry = next((j for j in json.loads(raw)["jobs"] if j["id"] == LOG_JOB_ID), None)
        tail = entry.get("log_tail") if entry is not None else None
    return status, tail, raw


def test_jobs_log_tail_default_is_at_most_40_last_lines():
    # Back-compat pin (green today, must stay green after the fix): with no
    # query parameter the response keeps the existing hard-coded tail window
    # — at most 40 lines, the LAST ones in file order.
    _seed_log_job(500)
    try:
        status, tail, raw = _jobs_tail()
        assert status == 200, raw
        assert tail is not None, raw
        assert len(tail) <= 40, f"default tail grew beyond the compat window: {len(tail)} lines"
        assert tail == [f"line {i:05d}" for i in range(500 - len(tail) + 1, 501)]
    finally:
        _drop_log_job()


def test_jobs_log_lines_120_returns_exactly_the_last_120():
    # An accepted in-range size returns EXACTLY that many last lines (was:
    # the parameter ignored → hard-coded 40-line tail).
    _seed_log_job(500)
    try:
        status, tail, raw = _jobs_tail("log_lines=120")
        assert status == 200, raw
        assert tail is not None, raw
        assert len(tail) == 120, f"log_lines=120 must return exactly 120 lines, got {len(tail)}"
        assert tail == [f"line {i:05d}" for i in range(381, 501)], "must be the LAST 120 lines"
    finally:
        _drop_log_job()


def test_jobs_log_lines_huge_clamps_to_2000_or_rejects():
    # Upper bound 2000 for log_lines=99999, accepting either shape: manual
    # clamp (200 with exactly 2000 lines on a 2500-line log) or FastAPI
    # le=2000 (422 naming log_lines — the committed implementation; the
    # 2000/2001 edge is pinned exactly by the boundary tests below).
    # Reading MORE than the bound, or silently ignoring it, must fail here.
    _seed_log_job(2500)
    try:
        status, tail, raw = _jobs_tail("log_lines=99999")
        if status == 422:
            assert "log_lines" in raw, raw
            return
        assert status == 200, raw
        assert tail is not None, raw
        assert len(tail) == 2000, f"log_lines=99999 must clamp to the 2000 upper bound, got {len(tail)}"
        assert tail[-1] == "line 02500", tail[-1]
    finally:
        _drop_log_job()


def test_jobs_log_lines_garbage_or_negative_422():
    # FastAPI-native semantics pinned per the work order: Query(int, ge=1,
    # le=2000) rejects a non-integer and a below-1 value with 422 naming
    # log_lines (was: unknown query params silently ignored → 200).
    _seed_log_job(50)
    try:
        for query in ("log_lines=abc", "log_lines=-5"):
            status, _tail, raw = _jobs_tail(query)
            assert status == 422, (query, status, raw[:400])
            assert "log_lines" in raw, (query, raw[:400])
    finally:
        _drop_log_job()


def test_jobs_log_lines_upper_bound_2000_ok_2001_rejected():
    # N1 pin of the le=2000 edge from both sides: 2000 is the LAST accepted
    # value (200 with exactly 2000 lines on a 2500-line log — the tail must
    # be the last 2000, not the whole file), 2001 is the FIRST rejected one
    # (422 naming log_lines, before the handler runs).
    _seed_log_job(2500)
    try:
        status, tail, raw = _jobs_tail("log_lines=2000")
        assert status == 200, raw
        assert tail is not None, raw
        assert len(tail) == 2000, f"log_lines=2000 must return exactly 2000 lines, got {len(tail)}"
        assert tail[0] == "line 00501" and tail[-1] == "line 02500", (tail[0], tail[-1])

        status, _tail, raw = _jobs_tail("log_lines=2001")
        assert status == 422, (status, raw[:400])
        assert "log_lines" in raw, raw[:400]
    finally:
        _drop_log_job()


def test_jobs_log_lines_zero_rejected_by_ge_1():
    # N1 pin of the lower edge: ge=1 accepts 1 (covered by the default-40
    # and 120-line tests) but rejects 0 with a 422 naming log_lines — a
    # zero window would silently return an empty tail instead of erroring.
    _seed_log_job(50)
    try:
        status, _tail, raw = _jobs_tail("log_lines=0")
        assert status == 422, (status, raw[:400])
        assert "log_lines" in raw, raw[:400]
    finally:
        _drop_log_job()


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
