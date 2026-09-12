from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace")).resolve()
COMFY_URL = os.environ.get("COMFY_URL", "http://comfyui:8188").rstrip("/")
COMPOSE_FILE = WORKSPACE / "compose.yaml"
JOBS_DIR = WORKSPACE / "runtime" / "jobs"

app = FastAPI(title="Valery Model Manager")

# Serialize heavy GPU work: only one generation/training job at a time (12 GB VRAM).
job_lock = asyncio.Lock()
jobs: dict[str, dict[str, Any]] = {}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _safe_name(value: str, *, allow_slash: bool = False) -> str:
    if allow_slash:
        cleaned = os.path.normpath(value)
    else:
        cleaned = os.path.basename(value)
    if not cleaned or cleaned.startswith(".") or ".." in cleaned.split("/"):
        raise HTTPException(status_code=400, detail="Unsafe name")
    return cleaned


async def _run_compose(*args: str) -> dict[str, Any]:
    if not COMPOSE_FILE.exists():
        raise HTTPException(status_code=500, detail="compose.yaml not found")
    return await _await_subprocess(
        ["docker", "compose", "-f", str(COMPOSE_FILE), *args],
        cwd=WORKSPACE,
    )


async def _await_subprocess(cmd: list[str], cwd: Path) -> dict[str, Any]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    return {
        "returncode": proc.returncode,
        "output": stdout.decode(errors="replace") if stdout else "",
    }


async def _stream_job(job_id: str, cmd: list[str]) -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = JOBS_DIR / f"{job_id}.log"
    jobs[job_id].update(status="running", log_path=str(log_path))
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(WORKSPACE),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        with log_path.open("wb") as log_file:
            assert proc.stdout is not None
            async for line in proc.stdout:
                log_file.write(line)
                log_file.flush()
        await proc.wait()
        jobs[job_id].update(
            status="done" if proc.returncode == 0 else "error",
            returncode=proc.returncode,
            finished_at=time.time(),
        )
    except Exception as exc:  # noqa: BLE001
        jobs[job_id].update(status="error", error=str(exc), finished_at=time.time())
    finally:
        jobs[job_id]["gpu_busy"] = False


def _comfy_up() -> bool:
    try:
        with urllib.request.urlopen(f"{COMFY_URL}/system_stats", timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _disk_free_gb() -> float:
    total, used, free = shutil.disk_usage(WORKSPACE)
    return round(free / (1024**3), 1)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WORKSPACE / "management" / "static" / "index.html")


app.mount(
    "/static",
    StaticFiles(directory=str(WORKSPACE / "management" / "static")),
    name="static",
)


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #
@app.get("/api/status")
async def status() -> dict[str, Any]:
    busy = any(j.get("status") == "running" for j in jobs.values())
    return {
        "comfy_up": _comfy_up(),
        "comfy_url": COMFY_URL,
        "gpu_busy": busy,
        "disk_free_gb": _disk_free_gb(),
        "active_jobs": [j for j in jobs.values() if j.get("status") == "running"],
    }


# --------------------------------------------------------------------------- #
# ComfyUI lifecycle
# --------------------------------------------------------------------------- #
@app.post("/api/comfy/start")
async def comfy_start() -> dict[str, Any]:
    result = await _run_compose("start", "comfyui")
    return {"ok": result["returncode"] == 0, "output": result["output"]}


@app.post("/api/comfy/stop")
async def comfy_stop() -> dict[str, Any]:
    result = await _run_compose("stop", "comfyui")
    return {"ok": result["returncode"] == 0, "output": result["output"]}


@app.post("/api/comfy/restart")
async def comfy_restart() -> dict[str, Any]:
    result = await _run_compose("restart", "comfyui")
    return {"ok": result["returncode"] == 0, "output": result["output"]}


# --------------------------------------------------------------------------- #
# Generation / training jobs (serialized via job_lock)
# --------------------------------------------------------------------------- #
async def _launch(kind: str, cmd: list[str]) -> dict[str, Any]:
    if any(j.get("status") == "running" for j in jobs.values()):
        raise HTTPException(status_code=409, detail="GPU is busy with another job")
    job_id = f"{kind}-{int(time.time())}"
    jobs[job_id] = {"id": job_id, "kind": kind, "status": "queued", "cmd": cmd}
    asyncio.create_task(_serial_job(job_id, cmd))
    return {"job_id": job_id, "status": "queued"}


async def _serial_job(job_id: str, cmd: list[str]) -> None:
    async with job_lock:
        jobs[job_id]["gpu_busy"] = True
        await _stream_job(job_id, cmd)


@app.post("/api/jobs/candidates")
async def job_candidates(payload: dict[str, Any]) -> dict[str, Any]:
    count = int(payload.get("count", 8))
    seed = int(payload.get("seed", 17023))
    return await _launch(
        "candidates",
        ["python", "scripts/queue_workflow.py", "candidates", "--count", str(count), "--seed", str(seed)],
    )


@app.post("/api/jobs/expand")
async def job_expand(payload: dict[str, Any]) -> dict[str, Any]:
    reference = _safe_name(str(payload.get("reference", "")), allow_slash=True)
    src = (WORKSPACE / "reference" / "candidates" / reference).resolve()
    if not src.exists() or WORKSPACE / "reference" / "candidates" not in src.parents:
        raise HTTPException(status_code=400, detail="reference not found")
    count = int(payload.get("count", 8))
    seed = int(payload.get("seed", 38447))
    return await _launch(
        "expand",
        [
            "python", "scripts/queue_workflow.py", "expand",
            "--reference", str(src), "--count", str(count), "--seed", str(seed),
        ],
    )


@app.post("/api/jobs/post")
async def job_post(payload: dict[str, Any]) -> dict[str, Any]:
    prompt = str(payload.get("prompt", "")).strip()
    caption = str(payload.get("caption", "")).strip()
    if not prompt or not caption:
        raise HTTPException(status_code=400, detail="prompt and caption required")
    name = _safe_name(str(payload.get("name", ""))) if payload.get("name") else None
    seed = int(payload.get("seed", 27191))
    strength = float(payload.get("lora_strength", 0.8))
    cmd = [
        "python", "scripts/queue_workflow.py", "post",
        "--prompt", prompt, "--caption", caption,
        "--seed", str(seed), "--lora-strength", str(strength),
    ]
    if name:
        cmd += ["--name", name]
    return await _launch("post", cmd)


@app.post("/api/train/start")
async def train_start() -> dict[str, Any]:
    return await _launch("train", ["bash", "scripts/train_lora.sh"])


@app.get("/api/jobs")
async def list_jobs() -> dict[str, Any]:
    out = []
    for job in sorted(jobs.values(), key=lambda j: j.get("id", ""), reverse=True):
        entry = {k: v for k, v in job.items() if k != "log_path"}
        log_path = job.get("log_path")
        if log_path and Path(log_path).exists():
            lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
            entry["log_tail"] = lines[-40:]
        out.append(entry)
    return {"jobs": out}


# --------------------------------------------------------------------------- #
# Browse outputs
# --------------------------------------------------------------------------- #
@app.get("/api/references")
async def list_references() -> dict[str, Any]:
    folder = WORKSPACE / "reference" / "candidates"
    if not folder.exists():
        return {"references": []}
    files = sorted(p.name for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return {"references": files}


@app.get("/api/posts")
async def list_posts() -> dict[str, Any]:
    folder = WORKSPACE / "posts"
    if not folder.exists():
        return {"posts": []}
    posts = []
    for post in sorted(folder.iterdir()):
        if not post.is_dir():
            continue
        photo = next(post.glob("photo.*"), None)
        caption = post / "caption.txt"
        posts.append(
            {
                "name": post.name,
                "photo": photo.name if photo else None,
                "caption": caption.read_text(encoding="utf-8").strip() if caption.exists() else None,
            }
        )
    return {"posts": posts}
