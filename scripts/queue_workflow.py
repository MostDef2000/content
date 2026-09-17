from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "runtime" / "output"

# Generation engines (Phase 4): flux = FLUX.1-dev FP8 + character LoRA;
# sdxl = Pony Diffusion V6 XL txt2img. ComfyUI's models dir is runtime/models
# (compose.yaml mounts it to /opt/ComfyUI/models) — the checkpoints live in
# runtime/models/checkpoints/ (next to flux1-dev-fp8.safetensors) and the
# LoRAs in runtime/models/loras/ (same dir as the character LoRAs).
SDXL_CKPT = "ponyDiffusionV6XL.safetensors"
SDXL_CKPT_PATH = ROOT / "runtime" / "models" / "checkpoints" / SDXL_CKPT
UNCENSOR_LORA = "flux1-uncensored.safetensors"
UNCENSOR_LORA_PATH = ROOT / "runtime" / "models" / "loras" / UNCENSOR_LORA
# Pony (SDXL) prompt scaffolding: quality-score prefix on the positive (the
# guarded character/scene composition is kept verbatim), fixed low-quality
# terms on the negative.
SDXL_POSITIVE_PREFIX = "score_9, score_8_up, score_7_up, source_photo"
SDXL_NEGATIVE = "score_6, score_5, score_4, worst quality, low quality, blurry, deformed, watermark"
UNCENSOR_NODE_ID = "43"

# prompts.py (guardrail prompt assembly) lives at the repo root, next to scripts/.
sys.path.insert(0, str(ROOT))
import prompts  # noqa: E402


def request_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode())
    except urllib.error.URLError as exc:
        raise RuntimeError(f"ComfyUI request failed: {exc}") from exc


def load_workflow(name: str) -> dict[str, Any]:
    path = ROOT / "comfy" / f"workflow_{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def queue_and_wait(base_url: str, workflow: dict[str, Any]) -> list[dict[str, Any]]:
    response = request_json(
        f"{base_url}/prompt",
        {"prompt": workflow, "client_id": str(uuid.uuid4())},
    )
    prompt_id = response.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI did not return prompt_id: {response}")

    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        history = request_json(f"{base_url}/history/{prompt_id}")
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI generation failed: {status}")
            images: list[dict[str, Any]] = []
            for output in entry.get("outputs", {}).values():
                images.extend(output.get("images", []))
            if images:
                return images
        time.sleep(2)
    raise TimeoutError("ComfyUI did not finish within 30 minutes")


def output_path(image: dict[str, Any]) -> Path:
    path = (OUTPUT_ROOT / image.get("subfolder", "") / image["filename"]).resolve()
    if OUTPUT_ROOT.resolve() not in path.parents:
        raise RuntimeError("ComfyUI returned an unsafe output path")
    return path


def load_registry() -> dict[str, Any]:
    path = ROOT / "models" / "registry.json"
    if not path.exists():
        raise FileNotFoundError(f"Model registry not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _active_id() -> str:
    """Active model id from models/registry.json (falls back to the first entry)."""
    models = load_registry().get("models") or []
    if not models:
        raise RuntimeError("models/registry.json contains no models — register a model first")
    for entry in models:
        if entry.get("active"):
            return str(entry["id"])
    return str(models[0]["id"])


def resolve_model_id(requested: str | None) -> str:
    """--model wins; otherwise the active registry model."""
    if requested:
        models = load_registry().get("models") or []
        if not any(str(entry.get("id")) == requested for entry in models):
            known = ", ".join(str(entry.get("id")) for entry in models)
            raise RuntimeError(f"Unknown model id {requested!r}; registered models: {known}")
        return requested
    return _active_id()


def load_character(model_id: str) -> dict[str, Any]:
    path = ROOT / "models" / model_id / "character.json"
    if not path.exists():
        raise FileNotFoundError(f"character.json not found for model {model_id!r}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def library_scene_texts(path_or_list, mode: str) -> list[str]:
    """Scene texts for ``mode`` from the user's scene library (models/library.json).

    ``path_or_list`` may be a path to the library file — either a top-level
    ``[{"id","name","mode","text","tags"}]`` list or the tracked
    ``{"version": 1, "scenes": [...]}`` wrapper — or a ready list of scene
    dicts. Missing file or broken JSON never raises: the result is simply
    empty, so a fresh/absent library degrades to "no scenes" instead of a
    crash.
    """
    data = path_or_list
    if not isinstance(data, list):
        try:
            data = json.loads(Path(data).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if isinstance(data, dict):
            data = data.get("scenes")
    if not isinstance(data, list):
        return []
    texts: list[str] = []
    for scene in data:
        if not isinstance(scene, dict) or scene.get("mode") != mode:
            continue
        text = str(scene.get("text") or "").strip()
        if text:
            texts.append(text)
    return texts


def guardrail_positive(raw: str, age: int) -> str:
    """User-supplied positive: blocked-term check + guardrail phrase injection."""
    positive = raw.strip()
    hits = prompts.validate_positive(positive)
    if hits:
        raise ValueError(f"--prompt contains blocked terms: {', '.join(hits)}")
    phrase = prompts.guardrail_phrase(age)
    if phrase not in positive:
        positive = f"{positive}, {phrase}"
    return positive


def set_negative(workflow: dict[str, Any], text: str) -> None:
    """Fill the negative CLIPTextEncode (node "33") when the workflow has one.

    The Kontext variation graph (BasicGuider) has no negative input, so it has
    no node "33" — there the canonical negative is enforced by the positive
    scene texts and ComfyUI-side defaults instead.
    """
    node = workflow.get("33")
    if node is not None and node.get("class_type") == "CLIPTextEncode":
        node["inputs"]["text"] = text


def inject_uncensor_lora(workflow: dict[str, Any], loader_node: str, consumer_node: str) -> bool:
    """Best-effort uncensor LoRA for the flux path (--uncensor).

    Inserts a LoraLoaderModelOnly (node ``UNCENSOR_NODE_ID``) with
    ``flux1-uncensored.safetensors`` (strength_model 1.0) between the model
    loader (``loader_node``) and its first consumer (``consumer_node``),
    mirroring the existing flux_lora character-LoRA inclusion (node "40").
    A missing LoRA file never fails the run: it logs a warning and the graph
    is left unwrapped. Returns True when the LoRA was injected.
    """
    if not UNCENSOR_LORA_PATH.is_file():
        print(f"uncensor lora not found, продолжаю без неё: {UNCENSOR_LORA_PATH}")
        return False
    workflow[UNCENSOR_NODE_ID] = {
        "inputs": {
            "lora_name": UNCENSOR_LORA,
            "strength_model": 1.0,
            "model": [loader_node, 0],
        },
        "class_type": "LoraLoaderModelOnly",
    }
    workflow[consumer_node]["inputs"]["model"] = [UNCENSOR_NODE_ID, 0]
    return True


def generate_candidates(args: argparse.Namespace) -> None:
    model_id = args.model_id
    character = load_character(model_id)
    age = int(character.get("age", prompts.AGE_FLOOR))

    destination = ROOT / "models" / model_id / "reference" / "candidates"
    destination.mkdir(parents=True, exist_ok=True)
    template = load_workflow("bootstrap")
    if args.uncensor:
        # bootstrap graph: CheckpointLoaderSimple "30" → KSampler "31"
        inject_uncensor_lora(template, "30", "31")
    if args.prompt:
        # Override the positive character prompt (the non-empty CLIPTextEncode).
        positive = guardrail_positive(args.prompt, age)
        for node in template.values():
            if node.get("class_type") == "CLIPTextEncode" and node.get("inputs", {}).get("text"):
                node["inputs"]["text"] = positive
                break
    # Negative node "33": canonical guardrail terms, plus optional --negative terms.
    set_negative(template, prompts.build_negative(args.negative, mode="candidates"))

    for index in range(1, args.count + 1):
        workflow = copy.deepcopy(template)
        workflow["31"]["inputs"]["seed"] = args.seed + index - 1
        workflow["9"]["inputs"]["filename_prefix"] = f"reference/{model_id}-{index:02d}"
        print(f"Generating candidate {index}/{args.count}...")
        images = queue_and_wait(args.url, workflow)
        source = output_path(images[0])
        target = destination / f"{model_id}-{index:02d}{source.suffix}"
        shutil.move(source, target)
        print(target)


def next_dataset_index(dataset_dir: Path, model_id: str) -> int:
    """Next free _NN suffix so repeated expand runs append instead of overwrite."""
    pattern = re.compile(rf"^{re.escape(model_id)}_(\d+)$")
    indices = [int(m.group(1)) for p in dataset_dir.glob(f"{model_id}_*") if (m := pattern.match(p.stem))]
    return (max(indices) + 1) if indices else 1


def expand_dataset(args: argparse.Namespace) -> None:
    model_id = args.model_id
    source = Path(args.reference).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Reference image not found: {source}")
    # ComfyUI LoadImage reads from runtime/input — copy there sequentially (one image at a time)
    input_dir = ROOT / "runtime" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = ROOT / "models" / model_id / "dataset" / "images"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Keep original as first dataset entry (reference anchor is written once —
    # re-copying on each run would overwrite it if the reference changed).
    if not (dataset_dir / f"{model_id}_00.jpg").exists():
        shutil.copy2(source, dataset_dir / f"{model_id}_00.jpg")

    age = int(load_character(model_id).get("age", prompts.AGE_FLOOR))
    template = load_workflow("kontext_variation")
    if args.uncensor:
        # kontext graph: UNETLoader "12" → ModelSamplingFlux "30"
        inject_uncensor_lora(template, "12", "30")
    # Place reference where LoadImage can find it (sequential use — never two models at once on 12GB)
    staged = input_dir / f"{model_id}_reference.jpg"
    shutil.copy2(source, staged)
    # Kontext graph has no negative node ("33") — set_negative stays a no-op
    # here; the guardrail is enforced on the positive texts below instead.
    set_negative(template, prompts.build_negative(args.negative, mode="expand"))

    if args.prompt:
        # One override prompt replaces the scene rotation for every variation.
        positives = [guardrail_positive(args.prompt, age)]
    else:
        # Scene rotation comes only from the user's library (models/library.json,
        # maintained in the manager UI). No built-in presets anymore.
        mode = "expand"
        texts = library_scene_texts(ROOT / "models" / "library.json", mode)
        if not texts:
            sys.exit(f"В библиотеке нет сцен режима {mode} — добавьте через UI")
        # Guardrail phrase injected per scene: Kontext has no negative node.
        positives = [guardrail_positive(text, age) for text in texts]

    start = next_dataset_index(dataset_dir, model_id)
    for offset in range(args.count):
        idx = start + offset
        prompt = positives[(idx - 1) % len(positives)]
        workflow = copy.deepcopy(template)
        workflow["6"]["inputs"]["text"] = prompt
        workflow["41"]["inputs"]["image"] = staged.name
        workflow["25"]["inputs"]["noise_seed"] = args.seed + idx
        workflow["9"]["inputs"]["filename_prefix"] = f"dataset/{model_id}-{idx:02d}"
        print(f"Generating dataset variation {idx} ({offset + 1}/{args.count}): {prompt[:60]}...")
        images = queue_and_wait(args.url, workflow)
        img = output_path(images[0])
        target = dataset_dir / f"{model_id}_{idx:02d}{img.suffix}"
        shutil.move(img, target)
        print(target)
    print(
        f"Dataset ready in {dataset_dir} ({len(list(dataset_dir.glob('*.jpg')))} images). "
        f"Run lora/caption.py --model {model_id} next."
    )


def generate_post(args: argparse.Namespace) -> None:
    model_id = args.model_id
    # Engine preflight (fail fast before any readiness checks / queueing):
    # the sdxl path needs the Pony checkpoint in the shared ComfyUI models dir.
    if args.engine == "sdxl" and not SDXL_CKPT_PATH.is_file():
        sys.exit(f"SDXL checkpoint not found: {SDXL_CKPT_PATH} — нужен handoff на скачивание")
    entry = next(
        (item for item in load_registry().get("models") or [] if str(item.get("id")) == model_id),
        {},
    )
    character = load_character(model_id)
    profile_path = ROOT / "models" / model_id / "prompt_profile.json"
    profile: dict[str, Any] = {}
    if profile_path.exists():
        profile = json.loads(profile_path.read_text(encoding="utf-8"))

    # Character LoRA preflight. Registry lora wins; older entries may lack
    # it — fall back to the canonical <id>.safetensors name produced by
    # scripts/train_lora.sh deployment. Both engines use the character LoRA
    # (LoraLoaderModelOnly node "40"): the flux engine fails fast on a
    # missing file, the sdxl (Pony) engine degrades gracefully (warn +
    # straight checkpoint wire, handled in the sdxl branch below).
    registry_lora = str(entry.get("lora") or f"{model_id}.safetensors")
    if args.engine == "flux":
        lora_path = ROOT / "runtime" / "models" / "loras" / registry_lora
        if not lora_path.is_file():
            raise FileNotFoundError(
                f"LoRA for model {model_id!r} is not available: expected {lora_path} "
                f"(registry lora: {entry.get('lora')!r}). Train it with "
                f"'scripts/train_lora.sh {model_id}' and copy the .safetensors into "
                "runtime/models/loras/."
            )

    age = int(character.get("age", prompts.AGE_FLOOR))
    if args.prompt:
        positive = guardrail_positive(args.prompt, age)
    else:
        appearance = character.get("appearance", {})
        positive = prompts.build_positive(
            name=str(character.get("name", model_id)),
            age=age,
            face=str(profile.get("face") or appearance.get("face", "")),
            body=str(profile.get("body") or character.get("body") or appearance.get("build", "")),
            style=str(profile.get("style") or character.get("style", "")),
            mode="post",
            scene=str((profile.get("scenes") or {}).get("post", "")),
        )

    if args.engine == "sdxl":
        # Pony (SDXL) txt2img: the same prompt-composition principle (character
        # + scene, guardrail phrase enforced upstream), wrapped in the pony
        # quality-score prefix; fixed pony negative.
        positive = f"{SDXL_POSITIVE_PREFIX}, {positive}"
        workflow = load_workflow("sdxl")
        workflow["6"]["inputs"]["text"] = positive
        workflow["31"]["inputs"]["seed"] = args.seed
        workflow["9"]["inputs"]["filename_prefix"] = f"posts/{model_id}"
        set_negative(workflow, SDXL_NEGATIVE)
        # Character LoRA (node "40") for the Pony engine too. Unlike flux this
        # is graceful: without a trained LoRA the post runs without identity
        # (graph rewired straight from the checkpoint), not an error.
        sdxl_lora_path = ROOT / "runtime" / "models" / "loras" / registry_lora
        if sdxl_lora_path.is_file():
            workflow["40"]["inputs"]["lora_name"] = registry_lora
            workflow["40"]["inputs"]["strength_model"] = args.lora_strength
        else:
            print(f"LoRA {model_id} не найдена — пост будет без идентичности (обучи через „Обучить LoRA“)")
            workflow["31"]["inputs"]["model"] = ["30", 0]
            # "40" is dropped from the graph: ComfyUI validates every node of the submitted prompt.
            workflow.pop("40", None)
    else:
        workflow = load_workflow("flux_lora")
        if args.uncensor:
            # flux_lora graph: CheckpointLoaderSimple "30" → LoraLoaderModelOnly "40"
            inject_uncensor_lora(workflow, "30", "40")
        workflow["6"]["inputs"]["text"] = positive
        workflow["31"]["inputs"]["seed"] = args.seed
        workflow["40"]["inputs"]["strength_model"] = args.lora_strength
        workflow["40"]["inputs"]["lora_name"] = registry_lora
        workflow["9"]["inputs"]["filename_prefix"] = f"posts/{model_id}"
        set_negative(workflow, prompts.build_negative(args.negative, mode="post"))

    post_name = args.name or datetime.now().strftime("%Y-%m-%d-%H%M%S")
    if Path(post_name).name != post_name:
        raise ValueError("Post name must not contain a path")
    # Dual-mode destination (feature 005): transit folder in ComfyUI stays
    # "posts/<model_id>" — only the final per-model folder switches.
    subdir = "posts" if args.mode == "public" else "posts-private"
    destination = ROOT / "models" / model_id / subdir / post_name
    destination.mkdir(parents=True, exist_ok=False)

    try:
        images = queue_and_wait(args.url, workflow)
        source = output_path(images[0])
        shutil.move(source, destination / f"photo{source.suffix}")
        # Optional caption: caption.txt is written only for a non-empty value
        # (list_posts reports caption=None when the file is absent).
        if args.caption.strip():
            (destination / "caption.txt").write_text(args.caption.strip() + "\n", encoding="utf-8")
        (destination / "prompt.txt").write_text(positive.strip() + "\n", encoding="utf-8")
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    print(destination)


GROUP_LAYOUT_COLUMNS = {"row": 1, "grid2": 2, "grid3": 3}


def _cover_crop(image: Any, width: int, height: int) -> Any:
    """Scale ``image`` to cover width×height, then center-crop to exactly that."""
    scale = max(width / image.width, height / image.height)
    scaled = image.resize((round(image.width * scale), round(image.height * scale)))
    left = (scaled.width - width) // 2
    top = (scaled.height - height) // 2
    return scaled.crop((left, top, left + width, top + height))


def _compose_grid(images: list[Any], columns: int) -> Any:
    """Paste member images into a columns-wide grid canvas.

    The first image defines the cell size; images of a different size are
    scaled to cover the cell and center-cropped, equal-size ones are pasted
    as-is. Unfilled tiles keep the neutral canvas fill (230, 230, 230).
    """
    from PIL import Image  # lazy: only the group composite needs Pillow

    cell_w, cell_h = images[0].size
    rows = math.ceil(len(images) / columns)
    canvas = Image.new("RGB", (columns * cell_w, rows * cell_h), (230, 230, 230))
    for index, image in enumerate(images):
        cell = image if image.size == (cell_w, cell_h) else _cover_crop(image, cell_w, cell_h)
        if cell.mode != "RGB":
            cell = cell.convert("RGB")
        canvas.paste(cell, ((index % columns) * cell_w, (index // columns) * cell_h))
    return canvas


def generate_group(args: argparse.Namespace) -> None:
    members = [item.strip() for item in args.models.split(",")]
    if not 1 <= len(members) <= 5:
        raise ValueError(f"--models needs 1..5 members, got {len(members)}")
    if len(set(members)) != len(members):
        raise ValueError(f"--models contains duplicate members: {members}")

    # Pillow is only needed for the composite; import lazily so py_compile and
    # --help work in environments without it (fail fast before queuing).
    try:
        from PIL import Image  # noqa: F401 — used below to open member images
    except ImportError as exc:
        raise RuntimeError(
            "Pillow required for group composite; rebuild manager image "
            "(see management/requirements.txt)"
        ) from exc

    # Pre-flight before queuing anything: registry entry, character, profile
    # and deployed LoRA for every member (mirrors generate_post checks).
    registry_models = load_registry().get("models") or []
    plans: list[dict[str, Any]] = []
    for member_id in members:
        entry = next(
            (item for item in registry_models if str(item.get("id")) == member_id),
            None,
        )
        if entry is None:
            known = ", ".join(str(item.get("id")) for item in registry_models)
            raise RuntimeError(f"Unknown model id {member_id!r}; registered models: {known}")
        character = load_character(member_id)
        profile_path = ROOT / "models" / member_id / "prompt_profile.json"
        profile: dict[str, Any] = {}
        if profile_path.exists():
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
        # Registry lora wins; older entries may lack it — fall back to the
        # canonical <id>.safetensors name (same rule as generate_post).
        registry_lora = str(entry.get("lora") or f"{member_id}.safetensors")
        lora_path = ROOT / "runtime" / "models" / "loras" / registry_lora
        if not lora_path.is_file():
            raise FileNotFoundError(
                f"LoRA for model {member_id!r} is not available: expected {lora_path} "
                f"(registry lora: {entry.get('lora')!r}). Train it with "
                f"'scripts/train_lora.sh {member_id}' and copy the .safetensors into "
                "runtime/models/loras/."
            )
        plans.append(
            {
                "id": member_id,
                "lora": registry_lora,
                "profile": profile,
                "age": int(character.get("age", prompts.AGE_FLOOR)),
            }
        )

    batch = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Transit stays per-member in ComfyUI (group/<batch>/<member_id>); the
    # staging folder collects the moved results for the composite.
    staging_root = OUTPUT_ROOT / "group" / batch
    staging_root.mkdir(parents=True)

    destination: Path | None = None
    try:
        member_paths: list[Path] = []
        for index, plan in enumerate(plans):
            member_id = str(plan["id"])
            positive = guardrail_positive(args.prompt, int(plan["age"]))
            negative = prompts.build_negative(
                str(plan["profile"].get("negative", ""))
                + (f", {args.negative}" if args.negative else ""),
                mode="post",
            )
            workflow = load_workflow("flux_lora")
            workflow["6"]["inputs"]["text"] = positive
            workflow["31"]["inputs"]["seed"] = args.seed + index
            workflow["40"]["inputs"]["lora_name"] = str(plan["lora"])
            workflow["40"]["inputs"]["strength_model"] = args.lora_strength
            workflow["9"]["inputs"]["filename_prefix"] = f"group/{batch}/{member_id}"
            set_negative(workflow, negative)
            print(f"Generating group member {index}/{len(plans)} ({member_id})...")
            images = queue_and_wait(args.url, workflow)
            source = output_path(images[0])
            target = staging_root / f"{index:02d}_{member_id}{source.suffix}"
            shutil.move(source, target)
            member_paths.append(target)
            plan["positive"] = positive

        member_images = [Image.open(path) for path in member_paths]
        canvas = _compose_grid(member_images, GROUP_LAYOUT_COLUMNS[args.layout])

        primary = members[0]
        name = args.name or datetime.now().strftime("%Y-%m-%d-%H%M%S")
        if Path(name).name != name:
            raise ValueError("Group name must not contain a path")
        # Destination follows the primary member's per-model posts layout (005).
        subdir = "posts" if args.mode == "public" else "posts-private"
        new_destination = ROOT / "models" / primary / subdir / name
        new_destination.mkdir(parents=True, exist_ok=False)
        destination = new_destination

        canvas.save(new_destination / "photo.jpg", format="JPEG", quality=92)
        (new_destination / "caption.txt").write_text(args.caption.strip() + "\n", encoding="utf-8")
        # First line: the primary member's positive (single-post format);
        # then a labeled section per member (ages may differ → phrases differ).
        prompt_sections = [f"{plans[0]['positive']}\n"]
        for plan in plans:
            prompt_sections.append(f"--- {plan['id']} ---\n{plan['positive']}\n")
        (new_destination / "prompt.txt").write_text("".join(prompt_sections), encoding="utf-8")
        group_meta = {
            "members": members,
            "layout": args.layout,
            "mode": args.mode,
            "seeds": [args.seed + index for index in range(len(plans))],
            "loras": {str(plan["id"]): str(plan["lora"]) for plan in plans},
            "caption": args.caption.strip(),
        }
        (new_destination / "group.json").write_text(
            json.dumps(group_meta, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        # Rollback: remove the destination only if this run created it (never
        # a pre-existing folder), and the staging folder only while empty.
        if destination is not None:
            shutil.rmtree(destination, ignore_errors=True)
        if staging_root.is_dir() and not any(staging_root.iterdir()):
            staging_root.rmdir()
        raise
    # Success: staging is transit, not an artifact — drop the moved member
    # files and remove the staging folder while empty (runtime/output/group/
    # itself stays in place).
    for path in member_paths:
        path.unlink(missing_ok=True)
    if staging_root.is_dir() and not any(staging_root.iterdir()):
        staging_root.rmdir()
    print(destination)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Queue sequential generation jobs in ComfyUI (FLUX / SDXL-Pony)")
    # Manager container reaches ComfyUI by compose service name; respect COMFY_URL.
    root.add_argument("--url", default=os.environ.get("COMFY_URL", "http://comfyui:8188"))
    commands = root.add_subparsers(dest="command", required=True)

    candidates = commands.add_parser("candidates", help="Generate reference faces")
    candidates.add_argument("--count", type=int, choices=range(1, 21), default=8)
    candidates.add_argument("--seed", type=int, default=17023)
    candidates.add_argument("--prompt", default="", help="Override the character prompt (guardrail phrase enforced)")
    candidates.add_argument("--negative", default="", help="Extra negative terms; canonical guardrail terms are always kept")
    candidates.add_argument("--model", default=None, help="Model id from models/registry.json (default: active)")
    candidates.add_argument(
        "--uncensor",
        action="store_true",
        help="Wrap the flux model in the flux1-uncensored LoRA (missing file: warn and continue without it)",
    )
    candidates.set_defaults(handler=generate_candidates)

    expand = commands.add_parser("expand", help="Expand single reference into dataset via Kontext (sequential)")
    expand.add_argument("--reference", required=True, help="Path to chosen candidate, e.g. models/<id>/reference/candidates/<id>-03.jpg")
    expand.add_argument("--count", type=int, choices=range(1, 21), default=8)
    expand.add_argument("--seed", type=int, default=38447)
    expand.add_argument("--prompt", default="", help="Override the scene rotation with one prompt (guardrail phrase enforced)")
    expand.add_argument("--negative", default="", help="Extra negative terms; canonical guardrail terms are always kept")
    expand.add_argument("--model", default=None, help="Model id from models/registry.json (default: active)")
    expand.add_argument(
        "--uncensor",
        action="store_true",
        help="Wrap the kontext model in the flux1-uncensored LoRA (missing file: warn and continue without it)",
    )
    expand.set_defaults(handler=expand_dataset)

    post = commands.add_parser("post", help="Generate one folder for manual posting")
    post.add_argument("--prompt", default="", help="Positive prompt; omit to build it from character.json + prompt_profile.json")
    post.add_argument("--caption", default="", help="Optional Instagram caption; empty value → caption.txt is not created")
    post.add_argument("--name")
    post.add_argument("--seed", type=int, default=27191)
    post.add_argument("--lora-strength", type=float, default=0.8)
    post.add_argument("--negative", default="", help="Extra negative terms; canonical guardrail terms are always kept")
    post.add_argument(
        "--mode",
        choices=("public", "private"),
        default="public",
        help="public → models/<id>/posts/, private → models/<id>/posts-private/",
    )
    post.add_argument("--model", default=None, help="Model id from models/registry.json (default: active)")
    post.add_argument(
        "--engine",
        choices=("flux", "sdxl"),
        default="flux",
        help="Generation engine: flux (FLUX.1-dev FP8 + character LoRA) or sdxl (Pony Diffusion V6 XL txt2img)",
    )
    post.add_argument(
        "--uncensor",
        action="store_true",
        help="Wrap the flux model in the flux1-uncensored LoRA (flux engine only; missing file: warn and continue)",
    )
    post.set_defaults(handler=generate_post)

    group = commands.add_parser("group", help="Generate a multi-model group composite (1..5 members)")
    group.add_argument(
        "--models",
        required=True,
        help="Comma-separated member ids from models/registry.json (1..5, no duplicates)",
    )
    group.add_argument("--prompt", required=True, help="Shared scene prompt; guardrail phrase injected per member age")
    group.add_argument(
        "--layout",
        choices=("row", "grid2", "grid3"),
        default="row",
        help="row → 1 column, grid2 → 2 columns, grid3 → 3 columns",
    )
    group.add_argument(
        "--mode",
        choices=("public", "private"),
        default="public",
        help="Destination follows the primary member: public → models/<primary>/posts/, private → posts-private/",
    )
    group.add_argument("--caption", required=True)
    group.add_argument("--name", help="Destination folder name under models/<primary>/<subdir>/ (default: timestamp)")
    group.add_argument("--seed", type=int, default=27191)
    group.add_argument("--lora-strength", type=float, default=0.8)
    group.add_argument("--negative", default="", help="Shared extra negative terms; canonical guardrail terms are always kept")
    group.set_defaults(handler=generate_group)
    return root


def main() -> int:
    args = parser().parse_args()
    if hasattr(args, "model"):
        args.model_id = resolve_model_id(args.model)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
