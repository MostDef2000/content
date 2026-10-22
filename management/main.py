from __future__ import annotations

import asyncio
import errno
import json
import os
import random
import re
import shutil
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

from fastapi import FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import prompts  # prompt assembly + adult-fictional guardrails (WORKSPACE root)

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace")).resolve()
# Issue #17 two-root layout: WORKSPACE stays the tracked checkout (static files,
# subprocess CWD, docker ctx, registry/library + per-model CONFIG), while every
# per-model DATA path (reference/dataset/posts/posts-private/lora-output) and the
# runtime tree (jobs/loras/trash/...) resolve under VALERY_DATA_ROOT. Per-model
# DATA keeps the "models" segment — DATA_ROOT/models/<id>/ — 1:1 with the real
# prod host tree /opt/sea-speed-worker/valery/models/<id> (the compose mount maps
# the data volume onto /opt/sea-speed-worker/valery). Default WORKSPACE keeps
# local single-root runs working; prod compose mounts the data volume at /data
# (VALERY_DATA_ROOT=/data, 1:1 with /opt/sea-speed-worker/valery).
DATA_ROOT = Path(os.environ.get("VALERY_DATA_ROOT", str(WORKSPACE))).resolve()
COMFY_URL = os.environ.get("COMFY_URL", "http://comfyui:8188").rstrip("/")
JOBS_DIR = DATA_ROOT / "runtime" / "jobs"
# Both "models" segments are intentional but DISTINCT roots (issue #17):
#   * DATA_ROOT/models/<id> — per-model DATA on the data volume (1:1 with the
#     prod host tree /opt/sea-speed-worker/valery/models/<id>);
#   * REPO_MODELS_DIR = WORKSPACE/models/<id> — tracked per-model CONFIGS in
#     the checkout (character.json, prompt_profile.json).
REPO_MODELS_DIR = WORKSPACE / "models"  # tracked: registry/library + per-model configs
REGISTRY_PATH = REPO_MODELS_DIR / "registry.json"
LIBRARY_PATH = REPO_MODELS_DIR / "library.json"
LORAS_DIR = DATA_ROOT / "runtime" / "models" / "loras"
TRASH_DIR = DATA_ROOT / "runtime" / "trash"  # soft-delete bin, never auto-cleaned

MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_UPLOAD_MB = 20  # per-file cap for dataset uploads
POST_ENGINES = ("flux", "sdxl")  # generation engines accepted by /api/jobs/post

# /files serving whitelist — "posts-private" is served under the same
# basicauth as every other kind (owner decision 16.09: preview re-enabled).
_FILE_KIND_DIRS = {
    "reference": "reference/candidates",
    "dataset": "dataset/images",
    "posts": "posts",
    "posts-private": "posts-private",
}
DATASET_TRASH = TRASH_DIR / "dataset"  # soft-delete bin for dataset image pairs

app = FastAPI(title="Model Manager")

# Serialize heavy GPU work: only one generation/training job at a time (12 GB VRAM).
job_lock = asyncio.Lock()
# Serialize registry writes (single-user app: last-write-wins is acceptable).
_registry_lock = asyncio.Lock()
# Serialize dataset manifest reads/writes and the caption.py subprocess.
_manifest_lock = asyncio.Lock()
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


async def _await_subprocess(
    cmd: list[str], cwd: Path, env: dict[str, str] | None = None
) -> dict[str, Any]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
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
    env.pop("SKIP_TRAIN", None)  # train_lora.sh test seam — must not reach a prod child process
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
    total, used, free = shutil.disk_usage(DATA_ROOT)
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


def _vram_usage() -> dict[str, Any] | None:
    """VRAM used/total in MiB from nvidia-smi; None when the tool is missing,
    times out, or emits unexpected output, so /api/status degrades gracefully."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    lines = (proc.stdout or "").strip().splitlines()
    if not lines:
        return None
    try:
        used_s, total_s = (part.strip() for part in lines[0].split(",")[:2])
        used, total = int(used_s), int(total_s)
    except ValueError:
        return None
    if total <= 0:
        return None
    return {"used": used, "total": total, "percent": round(used / total * 100, 1)}


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
    """Per-model DATA dir under the data root — DATA_ROOT/models/<id> (the
    "models" segment mirrors the prod host tree /opt/sea-speed-worker/valery/
    models/<id>; distinct from the checkout's REPO_MODELS_DIR). Never created
    here."""
    return DATA_ROOT / "models" / model_id


def _model_config_dir(model_id: str) -> Path:
    """Per-model tracked-config dir in the checkout (character.json,
    prompt_profile.json) — NOT data; data lives under _model_dir()."""
    return REPO_MODELS_DIR / model_id


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
    # Same shape as _model_dir: DATA_ROOT/models/<id> (+ optional subpath).
    return Path(DATA_ROOT, "models", str(model["id"]), *parts)


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

    character_path = _model_config_dir(model_id) / "character.json"
    if not character_path.exists():
        _atomic_write_json(character_path, character)

    profile_path = _model_config_dir(model_id) / "prompt_profile.json"
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
        character = json.loads((_model_config_dir(model_id) / "character.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise HTTPException(status_code=404, detail="character.json not found for model")
    if not isinstance(character, dict):
        raise HTTPException(status_code=404, detail="character.json not found for model")
    return character


def _load_profile(model_id: str) -> dict[str, Any]:
    try:
        profile = json.loads((_model_config_dir(model_id) / "prompt_profile.json").read_text(encoding="utf-8"))
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


def set_model_reference(workspace: Path, data_root: Path, model_id: str, filename: str) -> str:
    """Hard-persist the selected reference: set ONLY the "reference" key of
    the model's tracked character.json in the checkout (workspace/models/<id>/,
    every other field is preserved) and rewrite the file atomically (tempfile +
    os.replace, same as the other writers). The file must exist on disk in the
    model's DATA reference/candidates/ (the source of the UI reference picker,
    data_root/models/<id>/) or in its dataset/images/. Two roots because of the
    #17 split: lookups read DATA, the write lands in the tracked config dir.
    Raises HTTPException(400) for an empty name and HTTPException(404) when
    the file or character.json is missing."""
    filename = str(filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="file must be a non-empty filename")
    filename = _safe_name(filename)  # basename only — path traversal is impossible
    folders = (
        data_root / "models" / model_id / "reference" / "candidates",
        data_root / "models" / model_id / "dataset" / "images",
    )
    if not any(folder.is_dir() and (folder / filename).is_file() for folder in folders):
        raise HTTPException(status_code=404, detail=f"reference file not found: {filename}")
    character_path = workspace / "models" / model_id / "character.json"
    try:
        character = json.loads(character_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise HTTPException(status_code=404, detail="character.json not found for model")
    if not isinstance(character, dict):
        raise HTTPException(status_code=404, detail="character.json not found for model")
    character["reference"] = filename
    _atomic_write_json(character_path, character)
    return filename


def _safe_post_name(value: str) -> str:
    """A post folder name: a plain basename only. A raw value that carries a
    path (traversal like "../x") is rejected with 400 before _safe_name could
    neutralise it — a decorated name must never resolve to a real post."""
    raw = str(value or "").strip()
    if not raw or raw != os.path.basename(raw):
        raise HTTPException(status_code=400, detail="Unsafe name")
    return _safe_name(raw)  # dot/hidden names → 400 as well


def _move_to_trash(src: Path, dst: Path) -> None:
    """Move src to its trash destination dst (shared by delete_model,
    wipe_model_dataset, delete_model_post and delete_model_dataset_image).
    Prod topology
    (train-1792396861): /workspace/models and /workspace/runtime live on
    DIFFERENT mounts, and os.replace is atomic only on the SAME filesystem —
    cross-device moves die with EXDEV (Errno 18), reproduced live. Two
    validator-required guards:

      * EXDEV-ONLY fallback (validator condition): the shutil.move
        copy+delete fallback runs ONLY when exc.errno == errno.EXDEV —
        any other OSError (permissions, I/O, ...) is re-raised as-is to
        the per-site 500 wrappers, so an operator-visible failure can
        never be masked by a silently-succeeding fallback;
      * existing-dst rejection (validator requirement): a pre-existing
        destination raises FileExistsError(EEXIST) — shutil.move would
        otherwise silently NEST the source into an existing directory.
        delete_model's timestamped dst has no "-2"/"-3" collision
        suffixing; the suffix loops in delete_model_post /
        wipe_model_dataset prevent collisions upstream, and this explicit
        check closes the hole at helper level for every caller (the move
        becomes loud — a 500 — instead of nested).

    The per-site `except OSError -> HTTPException(500)` wrappers then fire
    when the fast path fails with a non-EXDEV error OR when both
    mechanisms fail. No API/behavior change otherwise."""
    if os.path.exists(dst):
        raise FileExistsError(errno.EEXIST, f"trash destination already exists: {dst}")
    try:
        os.replace(src, dst)  # atomic same-fs fast path
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise  # not a cross-device condition — surface it, do not fall back
        shutil.move(str(src), str(dst))  # EXDEV fallback (copy+delete)


def delete_model_post(data_root: Path, model_id: str, name: str, kind: str) -> str:
    """Soft-delete one post folder: <data_root>/models/<id>/<kind_dir>/<name>/ →
    <data_root>/runtime/trash/posts/<kind>-<name>-<unixts>/ (_move_to_trash:
    os.replace fast path, shutil.move cross-device fallback; parents created,
    "-2"/"-3" suffix on collision). kind "public" → "posts",
    "private" → "posts-private". Raises HTTPException(422) for an invalid
    kind, HTTPException(400) for an unsafe/traversal name and
    HTTPException(404) when the post folder is missing. Returns the name."""
    kind_dir = {"public": "posts", "private": "posts-private"}.get(kind)
    if kind_dir is None:
        raise HTTPException(status_code=422, detail="kind must be public or private")
    name = _safe_post_name(name)
    post_dir = data_root / "models" / model_id / kind_dir / name
    if not post_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"post not found: {name}")
    trash_dir = data_root / "runtime" / "trash" / "posts"
    trash_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{kind}-{name}-{int(time.time())}"
    trash_path = trash_dir / base_name
    attempt = 2
    while trash_path.exists():  # collision → "-2", "-3", ...
        trash_path = trash_dir / f"{base_name}-{attempt}"
        attempt += 1
    try:
        _move_to_trash(post_dir, trash_path)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"could not move post to trash: {exc}")
    return name


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


def _lora_arch(path: Path) -> str | None:
    """Detect LoRA architecture from safetensors header tensor keys.

    ai-toolkit names: SDXL deltas carry "unet" (lora_unet_*), FLUX carry
    "transformer" (lora_transformer_*). kohya (sd-scripts) FLUX LoRAs —
    the W3 trainer, prod candidates-1792384501 — name tensors
    lora_unet_double_blocks_*/single_blocks_*/img_mod_lin (also
    txt_mod_lin variants), so the word "unet" alone is NOT an sdxl
    discriminator: the FLUX-specific tokens are checked FIRST, and only
    if none match does "unet" mean ai-toolkit SDXL. Returns "sdxl" |
    "flux", or None when the file is missing/truncated/corrupt/ambiguous
    (fail-open: the caller keeps current behaviour instead of a false 400).
    """
    try:
        with open(path, "rb") as fh:
            raw_len = fh.read(8)
            if len(raw_len) < 8:
                return None
            header_len = struct.unpack("<Q", raw_len)[0]
            if header_len > 10_000_000:
                return None
            # Read ONLY the header (8 + header_len bytes), never the whole
            # file: trained LoRAs are hundreds of MB.
            header = fh.read(header_len)
        if len(header) < header_len:
            return None  # truncated file
        parsed = json.loads(header)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    # FLUX-specific discriminators FIRST (prod candidates-1792384501): the
    # kohya (sd-scripts) W3 trainer names FLUX tensors
    # lora_unet_double_blocks_*/lora_unet_single_blocks_* (also
    # img_mod_lin/txt_mod_lin variants) — the word "unet" is present in
    # kohya FLUX keys too, so it must not win the classification.
    if any(
        "double_blocks" in key
        or "single_blocks" in key
        or "img_mod_lin" in key
        or "txt_mod_lin" in key
        for key in parsed
    ):
        return "flux"
    # ai-toolkit naming (unchanged): SDXL deltas carry "unet", FLUX carry
    # "transformer" — reached only when NO kohya FLUX token matched.
    if any("unet" in key for key in parsed):
        return "sdxl"
    if any("transformer" in key for key in parsed):
        return "flux"
    return None


def _lora_name_for_engine(model: dict[str, Any], engine: str) -> str:
    """Registry lora slot per engine: sdxl → "lora" (fallback <id>.safetensors),
    flux → "lora_flux" (fallback <id>-flux.safetensors)."""
    model_id = str(model.get("id"))
    if engine == "flux":
        return str(model.get("lora_flux") or f"{model_id}-flux.safetensors")
    return str(model.get("lora") or f"{model_id}.safetensors")


def _model_engine(model: dict[str, Any]) -> str | None:
    """Recommended post engine by deployed LoRA slots (dual-arch): sdxl when
    the sdxl-slot file is deployed, else flux when the flux-slot file is,
    else None. Slot existence decides — a wrong-arch file in a slot is the
    arch guard's (400) business at post/group time, not the engine default's.
    Shared by _active_model_summary (/api/status) and list_models (/api/models,
    per-model engine default for the post-tab selector, issue #2 п.14)."""
    lora = model.get("lora")
    if lora and (LORAS_DIR / str(lora)).is_file():
        return "sdxl"
    flux_lora = model.get("lora_flux")
    if flux_lora and (LORAS_DIR / str(flux_lora)).is_file():
        return "flux"
    return None


def _active_model_summary(model: dict[str, Any] | None) -> dict[str, Any] | None:
    if not model:
        return None
    lora = model.get("lora")
    lora_file = bool(lora) and (LORAS_DIR / str(lora)).is_file()
    return {
        "id": model.get("id"),
        "name": model.get("name"),
        "lora": lora,
        "lora_file": lora_file,
        "engine": _model_engine(model),  # recommended post engine: "sdxl" | "flux" | None
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


def _scene_name(model_id: str, scene_id: str) -> str | None:
    """Scene display name by id from the shared library (empty name → the id;
    None when the id is not a library scene — e.g. a per-model profile scene).
    Recorded into post.json so the card keeps the name even after the scene
    is deleted from the library (issue #2 п.11)."""
    for scene in _load_library().get("scenes", []):
        if isinstance(scene, dict) and str(scene.get("id", "")) == scene_id:
            return str(scene.get("name") or "").strip() or scene_id
    return None


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
    vram = await asyncio.to_thread(_vram_usage)
    return {
        "comfy_up": await asyncio.to_thread(_comfy_up),
        "comfy_url": COMFY_URL,
        "gpu_busy": busy,
        "disk_free_gb": _disk_free_gb(),
        "cpu_percent": cpu,
        "ram_percent": await asyncio.to_thread(_ram_percent),
        "vram": vram,
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
        models.append({**model, "stats": _model_stats(model_id, model.get("lora")), "engine": _model_engine(model)})
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
    if _model_dir(model_id).exists() or _model_config_dir(model_id).exists():
        raise HTTPException(status_code=409, detail="model folder already exists")

    character = _seed_character(name, age)
    # Two-root layout (issue #17): DATA dirs under the data root, tracked
    # configs (character.json/prompt_profile.json) in the checkout — a fresh
    # model is fully functional with zero extra mounts.
    for sub in _model_subdirs(model_id):
        sub.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(_model_config_dir(model_id) / "character.json", character)
    _atomic_write_json(_model_config_dir(model_id) / "prompt_profile.json", prompts.default_profile(character))

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


def _model_has_active_job(model_id: str) -> bool:
    """True while a queued/running job references this model — solo jobs via
    model_id, group jobs via any member of their `models` list."""
    return any(
        j.get("status") in ("queued", "running")
        and (j.get("model_id") == model_id or model_id in (j.get("models") or []))
        for j in jobs.values()
    )


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
    if _model_has_active_job(model_id):
        raise HTTPException(status_code=409, detail="model has an active job")

    model_dir = _model_dir(model_id)
    config_dir = _model_config_dir(model_id)
    if model_dir.exists() or config_dir.exists():
        TRASH_DIR.mkdir(parents=True, exist_ok=True)
        trash_path = TRASH_DIR / f"{model_id}-{int(time.time())}"
        try:
            if model_dir.exists():
                _move_to_trash(model_dir, trash_path)
            else:
                # Config-only delete (no data dir on disk): the configs still
                # need a landing dir under the same timestamped trash path.
                trash_path.mkdir(parents=True)
            if config_dir.exists():
                # Same trash destination, nested "config/": the tracked
                # character.json/prompt_profile.json leave the checkout while
                # everything stays recoverable under one trash folder. Both
                # moves route through _move_to_trash, so the EXDEV-only
                # cross-device fallback covers both sources.
                _move_to_trash(config_dir, trash_path / "config")
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
        "character": {
            "name": character.get("name"),
            "age": character.get("age"),
            "reference": character.get("reference") or None,
        },
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
    _atomic_write_json(_model_config_dir(model_id) / "prompt_profile.json", profile)
    return {"ok": True, "profile": profile}


@app.post("/api/models/{model_id}/reference")
async def update_model_reference(model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Persist the selected reference portrait (pipeline step 2 picker).
    Hard persistence: only the "reference" key of character.json is updated,
    so the selection survives F5 and the 5s refresh (defect: client state only)."""
    _model_or_404(model_id)
    reference = set_model_reference(WORKSPACE, DATA_ROOT, model_id, str(payload.get("file", "")))
    return {"ok": True, "reference": reference}


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
# Dataset manifest API (Phase 2, feature 004 explicit-dataset)
# --------------------------------------------------------------------------- #
def _dataset_manifest_path(model_id: str) -> Path:
    """Per-model dataset manifest — same file and schema as lora/caption.py."""
    return _model_dir(model_id) / "dataset" / "manifest.json"


def _dataset_image_dir(model_id: str) -> Path:
    return _model_dir(model_id) / "dataset" / "images"


def _load_manifest(model_id: str) -> dict[str, Any]:
    """Read the dataset manifest and reconcile it with the files on disk.

    Mirrors lora/caption.py load_manifest: files without an entry get
    {"tag": "tasteful", "caption": None, "created": <mtime ISO UTC>};
    entries without a file are dropped; version > 1 is an error.
    """
    path = _dataset_manifest_path(model_id)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {"version": 1, "images": []}
    else:
        data = {"version": 1, "images": []}
    if not isinstance(data, dict) or not isinstance(data.get("images"), list):
        data = {"version": 1, "images": []}
    version = int(data.get("version", 1))
    if version > 1:
        raise RuntimeError(f"Unsupported dataset manifest version {version} in {path} (expected 1)")
    by_name = {
        str(entry.get("filename")): entry
        for entry in data["images"]
        if isinstance(entry, dict) and entry.get("filename")
    }
    image_dir = _dataset_image_dir(model_id)
    reconciled: list[dict[str, Any]] = []
    if image_dir.is_dir():
        for image in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            entry = by_name.pop(image.name, None)
            if entry is None:
                entry = {
                    "filename": image.name,
                    "tag": "tasteful",
                    "caption": None,
                    "created": datetime.fromtimestamp(
                        image.stat().st_mtime, tz=timezone.utc
                    ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            reconciled.append(entry)
    return {"version": 1, "images": reconciled}


def _save_manifest(model_id: str, data: dict[str, Any]) -> None:
    """Atomically persist the dataset manifest (tempfile in the same dir + os.replace)."""
    _atomic_write_json(_dataset_manifest_path(model_id), data)


@app.get("/api/models/{model_id}/dataset")
async def get_model_dataset(model_id: str) -> dict[str, Any]:
    _model_or_404(model_id)
    manifest = _load_manifest(model_id)
    images = manifest["images"]
    return {
        "model_id": model_id,
        "version": manifest["version"],
        "images": images,
        "stats": {
            "total": len(images),
            "tasteful": sum(1 for e in images if e.get("tag") == "tasteful"),
            "explicit": sum(1 for e in images if e.get("tag") == "explicit"),
            "captioned": sum(1 for e in images if e.get("caption")),
        },
    }


@app.put("/api/models/{model_id}/dataset")
async def retag_model_dataset(model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Partial merge-retag by filename: only `tag` is updated (caption/created kept)."""
    _model_or_404(model_id)
    updates_raw = payload.get("images")
    if not isinstance(updates_raw, list) or not updates_raw:
        raise HTTPException(status_code=422, detail="images must be a non-empty list")

    image_dir = _dataset_image_dir(model_id)
    updates: dict[str, str] = {}
    for index, item in enumerate(updates_raw):
        if not isinstance(item, dict):
            raise HTTPException(status_code=422, detail=f"images[{index}] must be an object")
        filename = _safe_name(str(item.get("filename", "")))
        if Path(filename).suffix.lower() not in IMAGE_SUFFIXES:
            raise HTTPException(status_code=422, detail=f"images[{index}].filename must be jpg/jpeg/png")
        tag = str(item.get("tag", "")).strip()
        if tag not in ("tasteful", "explicit"):
            raise HTTPException(status_code=422, detail=f"images[{index}].tag must be tasteful or explicit")
        if filename in updates:
            raise HTTPException(status_code=422, detail=f"duplicate filename in payload: {filename}")
        if not (image_dir / filename).is_file():
            raise HTTPException(status_code=404, detail=f"dataset image not found: {filename}")
        updates[filename] = tag

    async with _manifest_lock:
        manifest = _load_manifest(model_id)
        for entry in manifest["images"]:
            if entry["filename"] in updates:
                entry["tag"] = updates[entry["filename"]]
        _save_manifest(model_id, manifest)
    return {"ok": True, "manifest": {"version": manifest["version"], "images": manifest["images"]}}


@app.post("/api/models/{model_id}/dataset/upload")
async def upload_model_dataset(
    model_id: str,
    files: List[UploadFile] = File(...),
    tag: str = Form("tasteful"),
) -> dict[str, Any]:
    """Multipart dataset upload; every file is validated before the first byte
    is written (all-or-nothing)."""
    _model_or_404(model_id)
    if tag not in ("tasteful", "explicit"):
        raise HTTPException(status_code=422, detail="tag must be tasteful or explicit")
    if not files:
        raise HTTPException(status_code=422, detail="at least one file required")

    image_dir = _dataset_image_dir(model_id)
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    validated: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    for index, upload in enumerate(files):
        filename = _safe_name(str(upload.filename or ""))
        suffix = Path(filename).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise HTTPException(status_code=415, detail=f"files[{index}]: only jpg/jpeg/png are allowed")
        data = await upload.read()
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413, detail=f"files[{index}] ({filename}): exceeds {MAX_UPLOAD_MB} MB limit"
            )
        magic_ok = data.startswith(b"\x89PNG") if suffix == ".png" else data.startswith(b"\xff\xd8\xff")
        if not magic_ok:
            raise HTTPException(
                status_code=415, detail=f"files[{index}] ({filename}): content does not match the image type"
            )
        if filename in seen or (image_dir / filename).exists():
            raise HTTPException(status_code=409, detail=f"files[{index}]: name collision: {filename}")
        seen.add(filename)
        validated.append((filename, data))

    async with _manifest_lock:
        manifest = _load_manifest(model_id)
        image_dir.mkdir(parents=True, exist_ok=True)
        added: list[dict[str, Any]] = []
        for filename, data in validated:
            target = image_dir / filename
            tmp = target.with_name(f".{filename}.upload.tmp")
            try:
                tmp.write_bytes(data)
                os.replace(tmp, target)  # atomic same-filesystem move
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
            entry = {"filename": filename, "tag": tag, "caption": None, "created": _now_iso()}
            manifest["images"].append(entry)
            added.append(entry)
        _save_manifest(model_id, manifest)
    return {"ok": True, "added": added, "manifest": {"version": manifest["version"], "images": manifest["images"]}}


@app.delete("/api/models/{model_id}/dataset")
async def wipe_model_dataset(model_id: str, confirm: str = "") -> dict[str, Any]:
    """Bulk soft-delete (feature 006): move the whole dataset images/ dir into
    runtime/trash/dataset/<model_id>-wipe-<ts>/ (_move_to_trash: os.replace
    fast path, shutil.move cross-device fallback; manifest copied along) and
    reset to an empty manifest. Registered before the
    single-image delete route so both paths stay unambiguous."""
    async with _manifest_lock:
        _model_or_404(model_id)
        if confirm != model_id:
            raise HTTPException(status_code=400, detail="confirm must equal the model id")
        if _model_has_active_job(model_id):
            raise HTTPException(status_code=409, detail="model has an active job")

        image_dir = _dataset_image_dir(model_id)
        files = [p for p in image_dir.iterdir() if p.is_file()] if image_dir.is_dir() else []
        image_files = [p for p in files if p.suffix.lower() in IMAGE_SUFFIXES]
        image_stems = {p.stem for p in image_files}
        captions = sum(1 for p in files if p.suffix.lower() == ".txt" and p.stem in image_stems)
        counts = {
            "images": len(image_files),
            "captions": captions,
            "other": len(files) - len(image_files) - captions,
        }

        trash_path: str | None = None
        if files:  # empty (or missing) images/ → safe no-op, no trash folder
            DATASET_TRASH.mkdir(parents=True, exist_ok=True)
            base_name = f"{model_id}-wipe-{int(time.time())}"
            trash_dir = DATASET_TRASH / base_name
            attempt = 2
            while trash_dir.exists():  # collision → "-2", "-3", ...
                trash_dir = DATASET_TRASH / f"{base_name}-{attempt}"
                attempt += 1
            try:
                _move_to_trash(image_dir, trash_dir)
            except OSError as exc:
                raise HTTPException(status_code=500, detail=f"could not move dataset to trash: {exc}")
            manifest_path = _dataset_manifest_path(model_id)
            if manifest_path.exists():
                shutil.copy2(manifest_path, trash_dir / "manifest.json")
            trash_path = str(trash_dir)

        image_dir.mkdir(parents=True, exist_ok=True)
        _save_manifest(model_id, {"version": 1, "images": []})
    return {
        "ok": True,
        "model_id": model_id,
        "trash_path": trash_path,
        "deleted": counts,
        "manifest": {"version": 1, "images": []},
    }


@app.delete("/api/models/{model_id}/dataset/{filename}")
async def delete_model_dataset_image(model_id: str, filename: str) -> dict[str, Any]:
    """Soft-delete one dataset image and its <filename>.txt pair into
    runtime/trash/dataset/ (keeps image_count == caption_count in images/)."""
    _model_or_404(model_id)
    filename = _safe_name(filename)
    if Path(filename).suffix.lower() not in IMAGE_SUFFIXES:
        raise HTTPException(status_code=422, detail="filename must be jpg/jpeg/png")
    image_path = _dataset_image_dir(model_id) / filename
    caption_path = image_path.with_suffix(".txt")
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="dataset image not found")

    async with _manifest_lock:
        trash_dir = DATASET_TRASH / f"{model_id}-{int(time.time())}"
        trash_dir.mkdir(parents=True, exist_ok=True)
        try:
            _move_to_trash(image_path, trash_dir / filename)
            if caption_path.is_file():
                _move_to_trash(caption_path, trash_dir / caption_path.name)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"could not move image to trash: {exc}")
        manifest = _load_manifest(model_id)
        manifest["images"] = [e for e in manifest["images"] if e.get("filename") != filename]
        _save_manifest(model_id, manifest)
    return {"ok": True, "manifest": {"version": manifest["version"], "images": manifest["images"]}}


@app.post("/api/models/{model_id}/dataset/caption")
async def caption_model_dataset(model_id: str) -> dict[str, Any]:
    """Run lora/caption.py --force for this model (CPU task, not a GPU job)."""
    _model_or_404(model_id)
    if any(
        j.get("kind") == "train"
        and j.get("model_id") == model_id
        and j.get("status") in ("queued", "running")
        for j in jobs.values()
    ):
        raise HTTPException(status_code=409, detail="training is active for this model")
    async with _manifest_lock:
        # env merged explicitly (not relied on inherit-on-None) so the caption
        # child always sees THIS manager's effective data root, even when
        # VALERY_DATA_ROOT was set only in the manager's environment (#17).
        result = await _await_subprocess(
            ["python", "lora/caption.py", "--model", model_id, "--force"],
            cwd=WORKSPACE,
            env={**os.environ, "VALERY_DATA_ROOT": str(DATA_ROOT)},
        )
    output_tail = [line for line in result["output"].splitlines() if line.strip()][-20:]
    return {"ok": result["returncode"] == 0, "returncode": result["returncode"], "output_tail": output_tail}


# --------------------------------------------------------------------------- #
# Generated-artifact previews (candidates / dataset / posts / posts-private)
# — one whitelisted endpoint under the same basicauth as the rest of the app.
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
    # train must not wake ComfyUI: training does not use it, and per spec 010
    # ComfyUI is stopped for the duration of a train run (shared 12 GB card).
    err = None if kind == "train" else await _ensure_comfy()
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


def _validate_engine_fields(payload: dict[str, Any], *, allow_engine: bool) -> dict[str, Any]:
    """Validate the generation-engine payload fields (feature: SDXL/Pony engine).

    Pure (no job is launched), so unit tests can call it directly.
    ``allow_engine`` is True only for the post job: ``engine``
    ("flux"|"sdxl", default "flux") is a post-only field, while ``uncensor``
    (bool, default false) is accepted by candidates/expand/post (flux paths
    only — group takes neither). Invalid values raise HTTPException(400)."""
    if allow_engine:
        engine = str(payload.get("engine") or "flux").strip().lower()
        if engine not in POST_ENGINES:
            raise HTTPException(status_code=400, detail=f"engine must be one of {list(POST_ENGINES)}")
        if engine == "sdxl" and payload.get("uncensor"):
            raise HTTPException(status_code=400, detail="Параметр uncensor доступен только для движка FLUX")
    else:
        engine = "flux"
    uncensor = payload.get("uncensor", False)
    if not isinstance(uncensor, bool):
        raise HTTPException(status_code=400, detail="uncensor must be a boolean (true/false)")
    return {"engine": engine, "uncensor": uncensor}


@app.post("/api/jobs/candidates")
async def job_candidates(payload: dict[str, Any]) -> dict[str, Any]:
    model = _active_model_or_404()
    model_id = str(model["id"])
    character = _load_character(model_id)
    profile = _load_profile(model_id)
    count = int(payload.get("count", 8))
    # No seed in the payload (UI leaves the field empty) → random seed in
    # [0, 2**31); an explicit seed is used as-is (int validation as before).
    seed_raw = payload.get("seed")
    if seed_raw is None or not str(seed_raw).strip():
        seed = random.randint(0, 2**31 - 1)
    else:
        seed = int(seed_raw)
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
    engine_fields = _validate_engine_fields(payload, allow_engine=False)
    cmd = [
        "python", "scripts/queue_workflow.py", "candidates",
        "--count", str(count), "--seed", str(seed),
        "--prompt", prompt,
        "--model", model_id, "--negative", negative,
    ]
    if engine_fields["uncensor"]:
        cmd += ["--uncensor"]
    result = await _launch("candidates", cmd, model_id=model_id)
    # The actual seed (random or explicit) on the job record for the UI/audit.
    jobs[result["job_id"]]["seed"] = seed
    return result


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
    # No seed in the payload (UI leaves the field empty) → random seed in
    # [0, 2**31); an explicit seed is used as-is (int validation as before).
    seed_raw = payload.get("seed")
    if seed_raw is None:
        seed = random.randint(0, 2**31 - 1)
    else:
        seed = int(seed_raw)
    profile = _load_profile(model_id)
    scene_id = str(payload.get("scene_id") or payload.get("scene") or "").strip()
    engine_fields = _validate_engine_fields(payload, allow_engine=False)
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
    if engine_fields["uncensor"]:
        cmd += ["--uncensor"]
    result = await _launch("expand", cmd, model_id=model_id)
    # The actual seed (random or explicit) on the job record for the UI/audit.
    jobs[result["job_id"]]["seed"] = seed
    return result


@app.post("/api/jobs/post")
async def job_post(payload: dict[str, Any]) -> dict[str, Any]:
    # Per-post model binding (issue #2 п.14): an explicit payload.model_id
    # binds ONLY this post — prompt profile, LoRA slots, guardrail age and
    # the destination folder all resolve from that model. Empty/absent keeps
    # the back-compat behaviour (the active model). The registry active flag
    # is never touched: dataset/candidates/train keep following the active
    # model (/api/status reports it unchanged).
    requested_id = str(payload.get("model_id") or "").strip()
    if requested_id:
        try:
            model = _model_or_404(requested_id)
        except HTTPException as exc:
            if exc.status_code == 404:
                raise HTTPException(
                    status_code=404, detail=f"модель не найдена: {requested_id}"
                ) from exc
            raise
    else:
        model = _active_model_or_404()
    model_id = str(model["id"])
    engine_fields = _validate_engine_fields(payload, allow_engine=True)
    engine = engine_fields["engine"]
    # The character LoRA file is required only for the flux engine: the sdxl
    # (Pony) graph does carry a LoraLoaderModelOnly "40", wired in only when
    # the LoRA file exists (graceful: the node is dropped otherwise), so sdxl
    # posts must not 400 on a missing LoRA (mirrored by the preflight in
    # scripts/queue_workflow.py). Additionally, the safetensors header is
    # sniffed for the architecture: a trained LoRA of the wrong arch must
    # fail fast — ComfyUI silently ignores an SDXL LoRA on the FLUX
    # transformer (and vice versa), producing posts without the identity.
    # Undetectable arch (empty/corrupt stub) stays fail-open: the check is
    # skipped and the current behaviour is kept.
    # Dual-arch slots: the FLUX engine consumes the FLUX slot ("lora_flux",
    # fallback <id>-flux.safetensors) and the sdxl engine the SDXL slot
    # ("lora", fallback <id>.safetensors) — both as produced by
    # scripts/train_lora.sh <id> <arch> deployment. Older registry entries
    # may lack the slot key — the fallback name applies.
    lora = _lora_name_for_engine(model, engine)
    lora_path = LORAS_DIR / lora
    if engine == "flux":
        if not lora_path.is_file():
            raise HTTPException(
                status_code=400,
                detail=f"LoRA не обучена или не скопирована в runtime/models/loras/ (ожидался {lora})",
            )
        if _lora_arch(lora_path) == "sdxl":
            raise HTTPException(
                status_code=400,
                detail=f"LoRA модели {model_id} — SDXL (Pony) архитектура: для FLUX-поста неприменима, выберите движок Pony SDXL",
            )
    elif engine == "sdxl":
        if _lora_arch(lora_path) == "flux":
            raise HTTPException(
                status_code=400,
                detail=f"LoRA модели {model_id} — FLUX архитектура: для Pony SDXL поста неприменима, выберите движок FLUX",
            )
    character = _load_character(model_id)
    profile = _load_profile(model_id)
    # Optional caption (Instagram post text): an empty value is accepted; the
    # cmd still carries --caption "" and queue_workflow then skips caption.txt.
    caption = str(payload.get("caption", "")).strip()
    # Dual-mode (feature 005): public → posts/, private → posts-private/.
    # Guardrail validation applies to both modes.
    mode = str(payload.get("mode") or "public").strip()
    if mode not in ("public", "private"):
        raise HTTPException(status_code=422, detail="mode must be public or private")
    scene_id = str(payload.get("scene_id") or payload.get("scene") or "").strip()
    scene_text = _resolve_scene_text(model_id, scene_id) if scene_id else ""
    # Scene meta for post.json (issue #2 п.11): the display name is resolved
    # once here (library lookup, fallback the id) and passed through the CLI,
    # so the post folder keeps the name even if the scene is deleted later.
    scene_name = _scene_name(model_id, scene_id) if scene_id else None

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
    # No seed in the payload (UI leaves the field empty) → random seed in
    # [0, 2**31); an explicit seed is used as-is (int validation as before).
    seed_raw = payload.get("seed")
    if seed_raw is None or not str(seed_raw).strip():
        seed = random.randint(0, 2**31 - 1)
    else:
        seed = int(seed_raw)
    strength = float(payload.get("lora_strength", 0.8))
    cmd = [
        "python", "scripts/queue_workflow.py", "post",
        "--prompt", prompt, "--caption", caption,
        "--seed", str(seed), "--lora-strength", str(strength),
        "--model", model_id,
        "--negative", str(profile.get("negative", "")),
        "--mode", mode,
    ]
    if name:
        cmd += ["--name", name]
    # --engine is a post-only flag (candidates/expand are flux-only);
    # --uncensor is added only when requested (flux paths only, no-op for sdxl).
    cmd += ["--engine", engine]
    # Scene meta for post.json (issue #2 п.11): the id/name resolved above are
    # recorded verbatim by generate_post (no second library lookup in the CLI).
    cmd += ["--scene-id", scene_id, "--scene-name", scene_name or ""]
    if engine_fields["uncensor"]:
        cmd += ["--uncensor"]
    result = await _launch("post", cmd, model_id=model_id)
    # Engine on the job record so /api/jobs shows which engine generated the post,
    # and the actual seed (random or explicit) for the UI/audit.
    jobs[result["job_id"]]["engine"] = engine
    jobs[result["job_id"]]["seed"] = seed
    # Actual caption on the job record: empty → null (consistent with
    # list_posts, which reports caption=None when caption.txt is absent).
    jobs[result["job_id"]]["caption"] = caption or None
    return result


def _validate_group_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the group-job payload (feature 007). Pure — no job is launched,
    so unit tests can call it directly. Returns ids/entries/prompt_text/
    layout/mode/caption/name/seed/strength/negative. The guardrail phrase is
    NOT injected here: scripts/queue_workflow.py injects it per member (ages
    may differ), so only blocked terms are checked."""
    models_raw = payload.get("models")
    if not isinstance(models_raw, list) or not 1 <= len(models_raw) <= 5:
        raise HTTPException(status_code=422, detail="models must be a list of 1..5 unique model ids")
    ids = [str(item).strip() for item in models_raw]
    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=422, detail="models must be a list of 1..5 unique model ids")

    entries: list[dict[str, Any]] = []
    for member_id in ids:
        entry = _model_or_404(member_id)
        entries.append(entry)
        # Group shots are FLUX-only: every member consumes its FLUX slot
        # ("lora_flux", fallback <id>-flux.safetensors — same rule as the
        # flux branch of job_post / queue_workflow.generate_group).
        lora = _lora_name_for_engine(entry, "flux")
        if not (LORAS_DIR / lora).is_file():
            raise HTTPException(
                status_code=400,
                detail=f"FLUX-LoRA для модели {member_id} не обучена или не скопирована в runtime/models/loras/ (ожидался {lora})",
            )
        # Backstop: a file deployed into the FLUX slot but carrying SDXL-arch
        # deltas would be silently ignored by the FLUX transformer (no
        # identity). Undetectable arch (empty/corrupt stub) stays fail-open.
        if _lora_arch(LORAS_DIR / lora) == "sdxl":
            raise HTTPException(
                status_code=400,
                detail=f"Модель {member_id}: LoRA — SDXL (Pony) архитектура; групповые кадры пока FLUX-only — уберите её из группы",
            )

    caption = str(payload.get("caption", "")).strip()
    if not caption:
        raise HTTPException(status_code=400, detail="caption required")

    # Exactly one of prompt / scene_id (XOR).
    raw = str(payload.get("prompt", "")).strip()
    scene_id = str(payload.get("scene_id", "")).strip()
    if bool(raw) == bool(scene_id):
        raise HTTPException(status_code=400, detail="prompt or scene_id required")
    if raw:
        prompt_text = raw
    else:
        prompt_text = _resolve_scene_text(ids[0], scene_id)
    _assert_clean_prompt(prompt_text)

    layout = str(payload.get("layout") or "row").strip()
    if layout not in ("row", "grid2", "grid3"):
        raise HTTPException(status_code=422, detail="layout must be row, grid2 or grid3")
    mode = str(payload.get("mode") or "public").strip()
    if mode not in ("public", "private"):
        raise HTTPException(status_code=422, detail="mode must be public or private")

    name = None
    if payload.get("name"):
        name = _safe_name(str(payload["name"]))
        subdir = "posts" if mode == "public" else "posts-private"
        if (_model_dir(ids[0]) / subdir / name).exists():
            raise HTTPException(status_code=409, detail="post name already exists")

    # No seed in the payload (UI leaves the field empty) → random seed in
    # [0, 2**31); an explicit seed is used as-is (int validation as before).
    # One seed per group run; per-member seed+i is queue_workflow's job.
    seed_raw = payload.get("seed")
    if seed_raw is None or not str(seed_raw).strip():
        seed = random.randint(0, 2**31 - 1)
    else:
        seed = int(seed_raw)

    return {
        "ids": ids,
        "entries": entries,
        "prompt_text": prompt_text,
        "layout": layout,
        "mode": mode,
        "caption": caption,
        "name": name,
        "seed": seed,
        "strength": float(payload.get("lora_strength", 0.8)),
        "negative": str(payload.get("negative", "")),
    }


@app.post("/api/jobs/group")
async def job_group(payload: dict[str, Any]) -> dict[str, Any]:
    plan = _validate_group_payload(payload)
    ids = plan["ids"]
    cmd = [
        "python", "scripts/queue_workflow.py", "group",
        "--models", ",".join(ids),
        "--prompt", plan["prompt_text"],
        "--layout", plan["layout"],
        "--mode", plan["mode"],
        "--caption", plan["caption"],
        "--seed", str(plan["seed"]),
        "--lora-strength", str(plan["strength"]),
    ]
    if plan["name"]:
        cmd += ["--name", plan["name"]]
    if plan["negative"]:
        cmd += ["--negative", plan["negative"]]
    result = await _launch("group", cmd, model_id=ids[0])
    # Group members on the job record so _model_has_active_job sees every one,
    # and the actual seed (random or explicit) for the UI/audit.
    jobs[result["job_id"]]["models"] = ids
    jobs[result["job_id"]]["seed"] = plan["seed"]
    return {**result, "models": ids, "primary": ids[0], "layout": plan["layout"]}


@app.post("/api/train/start")
async def train_start(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    model = _active_model_or_404()
    model_id = str(model["id"])
    # arch selects the LoRA architecture (and thus the training template and
    # the runtime slot the result is deployed to): "sdxl" (default) or "flux".
    arch = str((payload or {}).get("arch") or "sdxl").strip().lower()
    if arch not in ("sdxl", "flux"):
        raise HTTPException(status_code=400, detail="arch must be sdxl or flux")
    # Register the LoRA filename the training produces BEFORE launching, so the
    # UI badge and post-generation can resolve the trained slot immediately:
    # flux → "lora_flux" (<id>-flux.safetensors), sdxl → "lora" (<id>.safetensors).
    reg = _load_registry()
    entry = next(
        (m for m in reg.get("models", []) if isinstance(m, dict) and str(m.get("id")) == model_id),
        None,
    )
    if entry is not None:
        if arch == "flux":
            entry["lora_flux"] = f"{model_id}-flux.safetensors"
        else:
            entry["lora"] = f"{model_id}.safetensors"
        await _save_registry(reg)
    return await _launch("train", ["bash", "scripts/train_lora.sh", model_id, arch], model_id=model_id)


# log_lines tail-window query parameter (spec 010 follow-up): the trainer's
# root error usually sits ABOVE the fixed 40-line tail, so a prod failure was
# undiagnosable without SSH-ing to the server to read the full log file.
# Default stays 40 for back-compat; ge=1/le=2000 bound the window (FastAPI
# answers 422 for garbage/negative/huge values before the handler runs).
@app.get("/api/jobs")
async def list_jobs(log_lines: int = Query(default=40, ge=1, le=2000)) -> dict[str, Any]:
    out = []
    for job in sorted(jobs.values(), key=lambda j: j.get("id", ""), reverse=True):
        entry = {k: v for k, v in job.items() if k != "log_path"}
        log_path = job.get("log_path")
        if log_path and Path(log_path).exists():
            lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
            entry["log_tail"] = lines[-log_lines:]
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


def _read_post_meta(post: Path) -> dict[str, Any]:
    """Best-effort post.json meta (scene binding, issue #2 п.11): a missing,
    unreadable or corrupt file — and a non-object payload — yield {}, so
    legacy posts simply report scene_name None (the card shows «—»)."""
    try:
        meta = json.loads((post / "post.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _post_meta_fields(meta: dict[str, Any]) -> dict[str, Any]:
    """The card-facing subset of post.json meta (None when absent)."""
    return {
        "scene_id": str(meta.get("scene_id") or "") or None,
        "scene_name": str(meta.get("scene_name") or "") or None,
        "model_id": str(meta.get("model_id") or "") or None,
        "engine": str(meta.get("engine") or "") or None,
        "mode": str(meta.get("mode") or "") or None,
        "seed": meta.get("seed") if isinstance(meta.get("seed"), int) else None,
    }


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
                **_post_meta_fields(_read_post_meta(post)),
            }
        )
    return {"posts": posts}


@app.get("/api/posts-private")
async def list_posts_private() -> dict[str, Any]:
    """Private (explicit-mode) posts of the active model. Preview is included:
    photo name per post and "posts-private" is served by /files under the same
    basicauth (owner decision 16.09)."""
    folder = _active_model_dir("posts-private")
    if folder is None or not folder.exists():
        return {"posts": []}
    posts = []
    for post in sorted(folder.iterdir()):
        if not post.is_dir():
            continue
        photo = next(post.glob("photo.*"), None)
        caption = post / "caption.txt"
        prompt = post / "prompt.txt"
        posts.append(
            {
                "name": post.name,
                "photo": photo.name if photo else None,
                "caption": caption.read_text(encoding="utf-8").strip() if caption.exists() else None,
                "prompt": prompt.read_text(encoding="utf-8").strip() if prompt.exists() else None,
                "created": datetime.fromtimestamp(post.stat().st_mtime, tz=timezone.utc).isoformat(),
                **_post_meta_fields(_read_post_meta(post)),
            }
        )
    return {"posts": posts}


@app.delete("/api/models/{model_id}/posts/{post_name}")
async def delete_post(
    model_id: str,
    post_name: str,
    kind: str = "public",
    confirm: str = "",
) -> dict[str, Any]:
    """Soft-delete one post (kind: public → posts/, private →
    posts-private/): the folder moves to runtime/trash/posts/ via
    delete_model_post. confirm must equal the post name (wipe pattern)."""
    _model_or_404(model_id)
    if kind not in ("public", "private"):
        raise HTTPException(status_code=422, detail="kind must be public or private")
    name = _safe_post_name(post_name)  # traversal/empty/dot names → 400
    if confirm != name:
        raise HTTPException(status_code=400, detail="confirm must equal the post name")
    delete_model_post(DATA_ROOT, model_id, name, kind)
    return {"ok": True, "deleted": name, "kind": kind}
