from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import prompts  # prompt assembly + adult-fictional guardrails (WORKSPACE root)

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace")).resolve()
COMFY_URL = os.environ.get("COMFY_URL", "http://comfyui:8188").rstrip("/")
JOBS_DIR = WORKSPACE / "runtime" / "jobs"
MODELS_DIR = WORKSPACE / "models"
REGISTRY_PATH = WORKSPACE / "models" / "registry.json"
LIBRARY_PATH = WORKSPACE / "models" / "library.json"
LORAS_DIR = WORKSPACE / "runtime" / "models" / "loras"
TRASH_DIR = WORKSPACE / "runtime" / "trash"  # soft-delete bin, never auto-cleaned

MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# /files serving whitelist — "posts-private" is never served.
_FILE_KIND_DIRS = {
    "reference": "reference/candidates",
    "dataset": "dataset/images",
    "posts": "posts",
}

app = FastAPI(title="Model Manager")

# Serialize heavy GPU work: only one generation/training job at a time (12 GB VRAM).
job_lock = asyncio.Lock()
# Serialize registry writes (single-user app: last-write-wins is acceptable).
_registry_lock = asyncio.Lock()
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


async def _docker(*args: str) -> dict[str, Any]:
    return await _await_subprocess(["docker", *args], cwd=WORKSPACE)


async def _comfy_container() -> str | None:
    result = await _docker("ps", "-a", "--filter", "name=comfyui", "--format", "{{.Names}}")
    names = [n.strip() for n in result["output"].splitlines() if n.strip()]
    return names[0] if names else None


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
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}  # stream prints instead of buffering
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(WORKSPACE),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
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


def _cpu_percent() -> float:
    def snapshot() -> tuple[int, int]:
        with open("/proc/stat") as f:
            vals = list(map(int, f.readline().split()[1:]))
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        return idle, sum(vals)

    idle0, total0 = snapshot()
    time.sleep(0.2)
    idle1, total1 = snapshot()
    dt = total1 - total0
    return round((1 - (idle1 - idle0) / dt) * 100, 1) if dt > 0 else 0.0


def _ram_percent() -> float:
    info: dict[str, int] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, val = line.split(":", 1)
            info[key] = int(val.strip().split()[0])
    total = info.get("MemTotal", 0)
    avail = info.get("MemAvailable", 0)
    return round((total - avail) / total * 100, 1) if total else 0.0


# --------------------------------------------------------------------------- #
# Model registry (Phase 1)
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON atomically: temp file in the same directory, then os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)


def _load_registry() -> dict[str, Any]:
    try:
        reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"version": 1, "models": []}
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "models": []}
    if not isinstance(reg, dict) or not isinstance(reg.get("models"), list):
        return {"version": 1, "models": []}
    return reg


async def _save_registry(reg: dict[str, Any]) -> None:
    """Persist the registry atomically, serialized by _registry_lock."""
    async with _registry_lock:
        await asyncio.to_thread(_atomic_write_json, REGISTRY_PATH, reg)


def _active_model() -> dict[str, Any] | None:
    """First model with active=True, else the first model, else None."""
    models = [m for m in _load_registry().get("models", []) if isinstance(m, dict)]
    for model in models:
        if model.get("active"):
            return model
    return models[0] if models else None


def _active_model_or_404() -> dict[str, Any]:
    model = _active_model()
    if model is None:
        raise HTTPException(status_code=404, detail="no active model")
    return model


def _model_dir(model_id: str) -> Path:
    """Per-model data dir (never created here)."""
    return MODELS_DIR / model_id


def _model_subdirs(model_id: str) -> list[Path]:
    base = _model_dir(model_id)
    return [
        base / "reference" / "candidates",
        base / "dataset" / "images",
        base / "lora" / "output",
        base / "posts",
        base / "posts-private",
    ]


def _active_model_dir(*parts: str) -> Path | None:
    model = _active_model()
    if model is None:
        return None
    return Path(MODELS_DIR, str(model["id"]), *parts)


def _seed_character(name: str, age: int) -> dict[str, Any]:
    """Reasonable tracked-character defaults for a newly created model."""
    appearance = {
        "hair": "long natural hair, soft waves",
        "eyes": "light eyes",
        "skin": "natural skin texture, healthy tone",
        "face": "symmetrical oval face, soft natural features",
        "build": "athletic adult build",
    }
    return {
        "name": name,
        "age": age,
        "height_cm": 178,
        "identity": "fictional adult AI model",
        "appearance": appearance,
        "body": str(appearance.get("build", "")),
        "style": "",
        "reference_look": "",
    }


def _ensure_registry() -> None:
    """Idempotent startup bootstrap: registry + default model layout.

    Fresh server without models/registry.json: migrate the legacy single-model
    layout if the root character.json exists, otherwise seed an empty default
    model. Never overwrites existing per-model files.
    """
    if REGISTRY_PATH.exists():
        return

    root_character: dict[str, Any] | None = None
    root_path = WORKSPACE / "character.json"
    if root_path.exists():
        try:
            loaded = json.loads(root_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and str(loaded.get("name", "")).strip():
                root_character = loaded
        except (OSError, json.JSONDecodeError):
            root_character = None

    model_id = "valery23"
    if root_character is None:
        character = _seed_character("Valery", prompts.AGE_FLOOR)
        lora: str | None = None
    else:
        character = dict(root_character)
        appearance = character.get("appearance")
        if isinstance(appearance, dict):
            character.setdefault("body", str(appearance.get("build", "")))
        else:
            character["appearance"] = {}
            character.setdefault("body", "")
        lora = "valery23.safetensors"

    for sub in _model_subdirs(model_id):
        sub.mkdir(parents=True, exist_ok=True)

    character_path = _model_dir(model_id) / "character.json"
    if not character_path.exists():
        _atomic_write_json(character_path, character)

    profile_path = _model_dir(model_id) / "prompt_profile.json"
    if not profile_path.exists():
        try:
            profile = prompts.default_profile(character)
        except (KeyError, TypeError):
            profile = prompts.default_profile(_seed_character("Valery", prompts.AGE_FLOOR))
        _atomic_write_json(profile_path, profile)

    entry = {
        "id": model_id,
        "name": str(character.get("name", "Valery")),
        "age": _model_age(character),
        "height_cm": int(character.get("height_cm", 178) or 178),
        "tags": ["fitness", "editorial"],
        "active": True,
        "created": _now_iso(),
        "lora": lora,
        "groups": [],
    }
    _atomic_write_json(REGISTRY_PATH, {"version": 1, "models": [entry]})


def _validate_model_id(model_id: str) -> str:
    if not MODEL_ID_RE.fullmatch(model_id):
        raise HTTPException(status_code=422, detail="model id must match ^[a-z0-9][a-z0-9-]{1,62}$")
    return model_id


def _model_age(character: dict[str, Any]) -> int:
    """Character age with the guardrail floor applied (bad data -> floor)."""
    try:
        return max(prompts.AGE_FLOOR, int(character.get("age", prompts.AGE_FLOOR)))
    except (TypeError, ValueError):
        return prompts.AGE_FLOOR


def _load_character(model_id: str) -> dict[str, Any]:
    try:
        character = json.loads((_model_dir(model_id) / "character.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise HTTPException(status_code=404, detail="character.json not found for model")
    if not isinstance(character, dict):
        raise HTTPException(status_code=404, detail="character.json not found for model")
    return character


def _load_profile(model_id: str) -> dict[str, Any]:
    try:
        profile = json.loads((_model_dir(model_id) / "prompt_profile.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise HTTPException(status_code=404, detail="prompt_profile.json not found for model")
    if not isinstance(profile, dict):
        raise HTTPException(status_code=404, detail="prompt_profile.json not found for model")
    return profile


def _load_library() -> dict[str, Any]:
    try:
        lib = json.loads(LIBRARY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "scenes": []}
    if not isinstance(lib, dict) or not isinstance(lib.get("scenes"), list):
        return {"version": 1, "scenes": []}
    return lib


def _model_or_404(model_id: str) -> dict[str, Any]:
    _validate_model_id(model_id)
    model = next((m for m in _load_registry().get("models", []) if str(m.get("id")) == model_id), None)
    if model is None:
        raise HTTPException(status_code=404, detail="model not found")
    return model


def _check_clean(text: str, field: str, violations: list[str]) -> None:
    hits = prompts.validate_positive(text)
    if hits:
        violations.append(f"{field}: {', '.join(hits)}")


def _count_files(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file())


def _count_dirs(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_dir())


def _model_stats(model_id: str, lora: str | None) -> dict[str, Any]:
    base = _model_dir(model_id)
    return {
        "candidates": _count_files(base / "reference" / "candidates"),
        "dataset": _count_files(base / "dataset" / "images"),
        "posts": _count_dirs(base / "posts"),
        "lora_file": (LORAS_DIR / str(lora)).is_file() if lora else False,
    }


def _active_model_summary(model: dict[str, Any] | None) -> dict[str, Any] | None:
    if not model:
        return None
    lora = model.get("lora")
    return {
        "id": model.get("id"),
        "name": model.get("name"),
        "lora": lora,
        "lora_file": bool(lora) and (LORAS_DIR / str(lora)).is_file(),
    }


def _resolve_scene_text(model_id: str, scene_id: str) -> str:
    """Scene text by id: model library first, then the per-model profile scenes."""
    for scene in _load_library().get("scenes", []):
        if isinstance(scene, dict) and str(scene.get("id", "")) == scene_id:
            return str(scene.get("text", "")).strip()
    profile_scenes = _load_profile(model_id).get("scenes")
    if isinstance(profile_scenes, dict) and profile_scenes.get(scene_id):
        return str(profile_scenes[scene_id]).strip()
    raise HTTPException(status_code=404, detail=f"scene not found: {scene_id}")


def _build_model_prompt(
    character: dict[str, Any], profile: dict[str, Any], mode: str, scene: str = ""
) -> str:
    """Assemble the positive prompt via prompts.build_positive (hard guardrail)."""
    try:
        return prompts.build_positive(
            name=str(character.get("name", "")),
            age=_model_age(character),
            face=str(profile.get("face", "")),
            body=str(profile.get("body", "")),
            style=str(profile.get("style", "")),
            mode=mode,
            scene=scene,
        )
    except prompts.GuardrailError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"invalid character/profile data: {exc}")


def _guardrail_raw_prompt(prompt: str, age: int) -> str:
    """Client-supplied raw prompt: validate, then enforce the guardrail phrase."""
    hits = prompts.validate_positive(prompt)
    if hits:
        raise HTTPException(status_code=422, detail="blocked terms in prompt: " + ", ".join(hits))
    phrase = prompts.guardrail_phrase(age)
    if phrase not in prompt:
        prompt = f"{prompt}, {phrase}"
    return prompt


def _assert_clean_prompt(prompt: str) -> None:
    """Final self-check for composed raw prompts (mirrors prompts.build_positive)."""
    hits = prompts.validate_positive(prompt)
    if hits:
        raise HTTPException(status_code=422, detail="blocked terms in prompt: " + ", ".join(hits))


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WORKSPACE / "management" / "static" / "index.html")


@app.get("/logout")
async def logout() -> Response:
    """Force HTTP Basic auth re-prompt (works behind nginx too: /logout returns 401 there)."""
    return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Model Manager"'})


app.mount(
    "/static",
    StaticFiles(directory=str(WORKSPACE / "management" / "static")),
    name="static",
)

# Startup bootstrap (single app process): job dir + idempotent registry setup.
JOBS_DIR.mkdir(parents=True, exist_ok=True)
_ensure_registry()


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #
@app.get("/api/status")
async def status() -> dict[str, Any]:
    busy = any(j.get("status") == "running" for j in jobs.values())
    cpu = await asyncio.to_thread(_cpu_percent)
    return {
        "comfy_up": _comfy_up(),
        "comfy_url": COMFY_URL,
        "gpu_busy": busy,
        "disk_free_gb": _disk_free_gb(),
        "cpu_percent": cpu,
        "ram_percent": _ram_percent(),
        "active_jobs": [j for j in jobs.values() if j.get("status") == "running"],
        "active_model": _active_model_summary(_active_model()),
    }


# --------------------------------------------------------------------------- #
# Models registry API
# --------------------------------------------------------------------------- #
@app.get("/api/models")
async def list_models() -> dict[str, Any]:
    reg = _load_registry()
    active = _active_model()
    models = []
    for model in reg.get("models", []):
        if not isinstance(model, dict):
            continue
        model_id = str(model.get("id", "")).strip()
        if not model_id:
            continue
        models.append({**model, "stats": _model_stats(model_id, model.get("lora"))})
    return {"active_id": active.get("id") if active else None, "models": models}


@app.post("/api/models", status_code=201)
async def create_model(payload: dict[str, Any]) -> dict[str, Any]:
    model_id = _validate_model_id(str(payload.get("id", "")))
    name = str(payload.get("name", "")).strip()
    if not name:
        raise HTTPException(status_code=422, detail="name required")
    try:
        age = int(str(payload.get("age", "")))
    except ValueError:
        raise HTTPException(status_code=422, detail="age must be an integer")
    if age < prompts.AGE_FLOOR:
        raise HTTPException(status_code=422, detail=f"age must be >= {prompts.AGE_FLOOR}")
    tags = [str(t) for t in (payload.get("tags") or [])]
    groups = [str(g) for g in (payload.get("groups") or [])]

    violations: list[str] = []
    _check_clean(name, "name", violations)
    for index, tag in enumerate(tags):
        _check_clean(tag, f"tags[{index}]", violations)
    if violations:
        raise HTTPException(status_code=422, detail="blocked terms — " + "; ".join(violations))

    reg = _load_registry()
    if any(str(m.get("id")) == model_id for m in reg.get("models", []) if isinstance(m, dict)):
        raise HTTPException(status_code=409, detail="model id already exists")
    if _model_dir(model_id).exists():
        raise HTTPException(status_code=409, detail="model folder already exists")

    character = _seed_character(name, age)
    for sub in _model_subdirs(model_id):
        sub.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(_model_dir(model_id) / "character.json", character)
    _atomic_write_json(_model_dir(model_id) / "prompt_profile.json", prompts.default_profile(character))

    entry = {
        "id": model_id,
        "name": name,
        "age": age,
        "height_cm": 178,
        "tags": tags,
        "active": not reg.get("models"),
        "created": _now_iso(),
        "lora": None,
        "groups": groups,
    }
    reg.setdefault("models", []).append(entry)
    await _save_registry(reg)
    return entry


@app.post("/api/models/{model_id}/activate")
async def activate_model(model_id: str) -> dict[str, Any]:
    _validate_model_id(model_id)
    reg = _load_registry()
    target = next(
        (m for m in reg.get("models", []) if isinstance(m, dict) and str(m.get("id")) == model_id),
        None,
    )
    if target is None:
        raise HTTPException(status_code=404, detail="model not found")
    for model in reg["models"]:
        if isinstance(model, dict):
            model["active"] = model is target
    await _save_registry(reg)
    return {"ok": True, "active_id": model_id}


@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str, confirm: str = "") -> dict[str, Any]:
    _validate_model_id(model_id)
    if confirm != model_id:
        raise HTTPException(status_code=400, detail="confirm must equal the model id")
    reg = _load_registry()
    target = next(
        (m for m in reg.get("models", []) if isinstance(m, dict) and str(m.get("id")) == model_id),
        None,
    )
    if target is None:
        raise HTTPException(status_code=404, detail="model not found")
    if target.get("active"):
        raise HTTPException(status_code=400, detail="activate another model first")
    if any(j.get("model_id") == model_id and j.get("status") == "running" for j in jobs.values()):
        raise HTTPException(status_code=409, detail="model has a running job")

    model_dir = _model_dir(model_id)
    if model_dir.exists():
        TRASH_DIR.mkdir(parents=True, exist_ok=True)
        trash_path = TRASH_DIR / f"{model_id}-{int(time.time())}"
        try:
            os.replace(model_dir, trash_path)  # atomic same-filesystem move (soft delete)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"could not move model to trash: {exc}")
    reg["models"] = [m for m in reg.get("models", []) if str(m.get("id")) != model_id]
    await _save_registry(reg)
    return {"ok": True}


# --------------------------------------------------------------------------- #
# Prompt profile API
# --------------------------------------------------------------------------- #
@app.get("/api/prompt-profile")
async def get_prompt_profile(model_id: str | None = None) -> dict[str, Any]:
    model = _model_or_404(model_id) if model_id else _active_model_or_404()
    model_id = str(model["id"])
    character = _load_character(model_id)
    profile = _load_profile(model_id)
    return {
        "model_id": model_id,
        "character": {"name": character.get("name"), "age": character.get("age")},
        "profile": {
            "face": str(profile.get("face", "")),
            "body": str(profile.get("body", "")),
            "style": str(profile.get("style", "")),
            "negative": str(profile.get("negative", "")),
            "scenes": profile.get("scenes") or {},
        },
        "guardrail": {
            "phrase": prompts.guardrail_phrase(_model_age(character)),
            "canonical_negative": prompts.CANONICAL_NEGATIVE,
        },
        "presets": prompts.scene_presets(),
    }


@app.post("/api/prompt-profile")
async def update_prompt_profile(payload: dict[str, Any]) -> dict[str, Any]:
    requested_id = str(payload.get("model_id") or "").strip()
    model = _model_or_404(requested_id) if requested_id else _active_model_or_404()
    model_id = str(model["id"])

    face = str(payload.get("face", "")).strip()
    body = str(payload.get("body", "")).strip()
    style = str(payload.get("style", "")).strip()
    negative = str(payload.get("negative", "")).strip()
    scenes_raw = payload.get("scenes")
    if not isinstance(scenes_raw, dict):
        raise HTTPException(status_code=422, detail="scenes must be an object")
    scenes = {str(k): str(v) for k, v in scenes_raw.items()}

    violations: list[str] = []
    for field, value in (("face", face), ("body", body), ("style", style)):
        _check_clean(value, field, violations)
    for scene_id, text in scenes.items():
        _check_clean(text, f"scenes.{scene_id}", violations)
    if violations:
        raise HTTPException(status_code=422, detail="blocked terms — " + "; ".join(violations))

    profile = {
        "version": 1,
        "face": face,
        "body": body,
        "style": style,
        "negative": negative,
        "scenes": scenes,
    }
    _atomic_write_json(_model_dir(model_id) / "prompt_profile.json", profile)
    return {"ok": True, "profile": profile}


# --------------------------------------------------------------------------- #
# Scene library API
# --------------------------------------------------------------------------- #
@app.get("/api/library")
async def get_library() -> dict[str, Any]:
    return {"scenes": _load_library().get("scenes", [])}


@app.post("/api/library")
async def update_library(payload: dict[str, Any]) -> dict[str, Any]:
    scenes_raw = payload.get("scenes")
    if not isinstance(scenes_raw, list):
        raise HTTPException(status_code=422, detail="scenes must be a list")

    scenes: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    violations: list[str] = []
    for index, item in enumerate(scenes_raw):
        if not isinstance(item, dict):
            raise HTTPException(status_code=422, detail=f"scenes[{index}] must be an object")
        scene_id = str(item.get("id", "")).strip()
        name = str(item.get("name", "")).strip()
        mode = str(item.get("mode", "")).strip()
        text = str(item.get("text", "")).strip()
        tags = [str(t) for t in (item.get("tags") or [])]
        if not scene_id:
            raise HTTPException(status_code=422, detail=f"scenes[{index}].id required")
        if scene_id in seen_ids:
            raise HTTPException(status_code=422, detail=f"duplicate scene id: {scene_id}")
        seen_ids.add(scene_id)
        if mode not in prompts.MODES:
            raise HTTPException(
                status_code=422, detail=f"scenes[{index}].mode must be one of {list(prompts.MODES)}"
            )
        _check_clean(name, f"scenes[{index}].name", violations)
        _check_clean(text, f"scenes[{index}].text", violations)
        scenes.append({"id": scene_id, "name": name, "mode": mode, "text": text, "tags": tags})
    if violations:
        raise HTTPException(status_code=422, detail="blocked terms — " + "; ".join(violations))

    _atomic_write_json(LIBRARY_PATH, {"version": 1, "scenes": scenes})
    return {"ok": True, "scenes": scenes}


# --------------------------------------------------------------------------- #
# Generated-artifact previews (candidates / dataset / posts) — one whitelisted
# endpoint; "posts-private" is never served.
# --------------------------------------------------------------------------- #
@app.get("/files/{model_id}/{kind}/{filename:path}")
async def serve_model_file(model_id: str, kind: str, filename: str) -> FileResponse:
    _validate_model_id(model_id)
    if kind not in _FILE_KIND_DIRS:
        raise HTTPException(status_code=404, detail="unknown file kind")
    base = (_model_dir(model_id) / _FILE_KIND_DIRS[kind]).resolve()
    candidate = (base / filename).resolve()
    if base not in candidate.parents:
        raise HTTPException(status_code=400, detail="Unsafe path")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(candidate)


# --------------------------------------------------------------------------- #
# ComfyUI lifecycle
# --------------------------------------------------------------------------- #
async def _comfy_lifecycle(action: str) -> dict[str, Any]:
    name = await _comfy_container()
    if not name:
        raise HTTPException(status_code=500, detail="ComfyUI container not found (run compose up on the worker)")
    result = await _docker(action, name)
    return {"ok": result["returncode"] == 0, "output": result["output"]}


@app.post("/api/comfy/start")
async def comfy_start() -> dict[str, Any]:
    return await _comfy_lifecycle("start")


@app.post("/api/comfy/stop")
async def comfy_stop() -> dict[str, Any]:
    return await _comfy_lifecycle("stop")


@app.post("/api/comfy/restart")
async def comfy_restart() -> dict[str, Any]:
    return await _comfy_lifecycle("restart")


# --------------------------------------------------------------------------- #
# Generation / training jobs (serialized via job_lock)
# --------------------------------------------------------------------------- #
async def _ensure_comfy() -> str | None:
    """Start ComfyUI if down. Returns error message or None."""
    if _comfy_up():
        return None
    name = await _comfy_container()
    if not name:
        return "ComfyUI container not found (run compose up on the worker)"
    result = await _docker("start", name)
    if result["returncode"] != 0:
        return f"docker start failed: {result['output']}"
    for _ in range(15):  # up to ~30s for the API to come up
        await asyncio.sleep(2)
        if _comfy_up():
            return None
    return "ComfyUI started but its API is not responding after 30s"


async def _launch(kind: str, cmd: list[str], *, model_id: str | None = None) -> dict[str, Any]:
    if any(j.get("status") == "running" for j in jobs.values()):
        raise HTTPException(status_code=409, detail="GPU is busy with another job")
    err = await _ensure_comfy()
    if err:
        raise HTTPException(status_code=503, detail=err)
    job_id = f"{kind}-{int(time.time())}"
    jobs[job_id] = {
        "id": job_id,
        "kind": kind,
        "model_id": model_id,
        "status": "queued",
        "cmd": cmd,
    }
    asyncio.create_task(_serial_job(job_id, cmd))
    return {"job_id": job_id, "status": "queued"}


async def _serial_job(job_id: str, cmd: list[str]) -> None:
    async with job_lock:
        jobs[job_id]["gpu_busy"] = True
        await _stream_job(job_id, cmd)


@app.post("/api/jobs/candidates")
async def job_candidates(payload: dict[str, Any]) -> dict[str, Any]:
    model = _active_model_or_404()
    model_id = str(model["id"])
    character = _load_character(model_id)
    profile = _load_profile(model_id)
    count = int(payload.get("count", 8))
    seed = int(payload.get("seed", 17023))
    scene_id = str(payload.get("scene_id") or payload.get("scene") or "").strip()
    scene_text = _resolve_scene_text(model_id, scene_id) if scene_id else ""

    raw = str(payload.get("prompt", "")).strip()
    if raw:
        # Raw prompt: validated, guardrail phrase enforced, used as-is.
        prompt = _guardrail_raw_prompt(raw, _model_age(character))
        if scene_text:
            prompt = f"{prompt}, {scene_text}"
        _assert_clean_prompt(prompt)
    else:
        prompt = _build_model_prompt(character, profile, "candidates", scene_text)

    negative = prompts.build_negative(profile.get("negative", ""), mode="candidates")
    cmd = [
        "python", "scripts/queue_workflow.py", "candidates",
        "--count", str(count), "--seed", str(seed),
        "--prompt", prompt,
        "--model", model_id, "--negative", negative,
    ]
    return await _launch("candidates", cmd, model_id=model_id)


@app.post("/api/jobs/expand")
async def job_expand(payload: dict[str, Any]) -> dict[str, Any]:
    model = _active_model_or_404()
    model_id = str(model["id"])
    reference = _safe_name(str(payload.get("reference", "")), allow_slash=True)
    candidates_dir = (_model_dir(model_id) / "reference" / "candidates").resolve()
    src = (candidates_dir / reference).resolve()
    if not src.exists() or candidates_dir not in src.parents:
        raise HTTPException(status_code=400, detail="reference not found")
    count = int(payload.get("count", 8))
    seed = int(payload.get("seed", 38447))
    profile = _load_profile(model_id)
    scene_id = str(payload.get("scene_id") or payload.get("scene") or "").strip()
    cmd = [
        "python", "scripts/queue_workflow.py", "expand",
        "--reference", str(src), "--count", str(count), "--seed", str(seed),
        "--model", model_id,
        "--negative", str(profile.get("negative", "")),
    ]
    if scene_id:
        # Resolved scene text overrides the expand scene rotation (guardrail
        # phrase is enforced downstream by queue_workflow.expand_dataset).
        cmd += ["--prompt", _resolve_scene_text(model_id, scene_id)]
    return await _launch("expand", cmd, model_id=model_id)


@app.post("/api/jobs/post")
async def job_post(payload: dict[str, Any]) -> dict[str, Any]:
    model = _active_model_or_404()
    model_id = str(model["id"])
    # Registry lora wins; older entries may lack it — fall back to the canonical
    # <id>.safetensors name produced by scripts/train_lora.sh deployment.
    lora = str(model.get("lora") or f"{model_id}.safetensors")
    if not (LORAS_DIR / lora).is_file():
        raise HTTPException(
            status_code=400,
            detail=f"LoRA не обучена или не скопирована в runtime/models/loras/ (ожидался {lora})",
        )
    character = _load_character(model_id)
    profile = _load_profile(model_id)
    caption = str(payload.get("caption", "")).strip()
    if not caption:
        raise HTTPException(status_code=400, detail="caption required")
    scene_id = str(payload.get("scene_id") or payload.get("scene") or "").strip()
    scene_text = _resolve_scene_text(model_id, scene_id) if scene_id else ""

    raw = str(payload.get("prompt", "")).strip()
    if raw:
        # Raw prompt: validated, guardrail phrase enforced, scene text appended.
        prompt = _guardrail_raw_prompt(raw, _model_age(character))
        if scene_text:
            prompt = f"{prompt}, {scene_text}"
        _assert_clean_prompt(prompt)
    elif scene_text:
        prompt = _build_model_prompt(character, profile, "post", scene_text)
    else:
        raise HTTPException(status_code=400, detail="prompt or scene_id required")

    name = _safe_name(str(payload.get("name", ""))) if payload.get("name") else None
    seed = int(payload.get("seed", 27191))
    strength = float(payload.get("lora_strength", 0.8))
    cmd = [
        "python", "scripts/queue_workflow.py", "post",
        "--prompt", prompt, "--caption", caption,
        "--seed", str(seed), "--lora-strength", str(strength),
        "--model", model_id,
        "--negative", str(profile.get("negative", "")),
    ]
    if name:
        cmd += ["--name", name]
    return await _launch("post", cmd, model_id=model_id)


@app.post("/api/train/start")
async def train_start() -> dict[str, Any]:
    model = _active_model_or_404()
    model_id = str(model["id"])
    # Register the LoRA filename the training produces BEFORE launching, so the
    # UI badge and post-generation can resolve <id>.safetensors immediately.
    reg = _load_registry()
    entry = next(
        (m for m in reg.get("models", []) if isinstance(m, dict) and str(m.get("id")) == model_id),
        None,
    )
    if entry is not None:
        entry["lora"] = f"{model_id}.safetensors"
        await _save_registry(reg)
    return await _launch("train", ["bash", "scripts/train_lora.sh", model_id], model_id=model_id)


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
# Browse outputs (active model)
# --------------------------------------------------------------------------- #
@app.get("/api/references")
async def list_references() -> dict[str, Any]:
    folder = _active_model_dir("reference", "candidates")
    if folder is None or not folder.exists():
        return {"references": []}
    files = sorted(p.name for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return {"references": files}


@app.get("/api/dataset")
async def list_dataset() -> dict[str, Any]:
    folder = _active_model_dir("dataset", "images")
    if folder is None or not folder.exists():
        return {"images": []}
    files = sorted(p.name for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return {"images": files}


@app.get("/api/posts")
async def list_posts() -> dict[str, Any]:
    folder = _active_model_dir("posts")
    if folder is None or not folder.exists():
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
