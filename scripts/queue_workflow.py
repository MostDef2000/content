from __future__ import annotations

import argparse
import copy
import json
import os
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


def generate_candidates(args: argparse.Namespace) -> None:
    model_id = args.model_id
    character = load_character(model_id)
    age = int(character.get("age", prompts.AGE_FLOOR))

    destination = ROOT / "models" / model_id / "reference" / "candidates"
    destination.mkdir(parents=True, exist_ok=True)
    template = load_workflow("bootstrap")
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

    # Keep original as first dataset entry
    shutil.copy2(source, dataset_dir / f"{model_id}_00.jpg")

    age = int(load_character(model_id).get("age", prompts.AGE_FLOOR))
    template = load_workflow("kontext_variation")
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
        scenes = prompts.scene_presets("expand")
        if not scenes:
            raise RuntimeError("No expand scenes available (prompts.scene_presets('expand') is empty)")
        # Guardrail phrase injected per scene: Kontext has no negative node.
        positives = [guardrail_positive(str(scene["text"]), age) for scene in scenes]

    for idx in range(1, args.count + 1):
        prompt = positives[(idx - 1) % len(positives)]
        workflow = copy.deepcopy(template)
        workflow["6"]["inputs"]["text"] = prompt
        workflow["41"]["inputs"]["image"] = staged.name
        workflow["25"]["inputs"]["noise_seed"] = args.seed + idx
        workflow["9"]["inputs"]["filename_prefix"] = f"dataset/{model_id}-{idx:02d}"
        print(f"Generating dataset variation {idx}/{args.count}: {prompt[:60]}...")
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
    entry = next(
        (item for item in load_registry().get("models") or [] if str(item.get("id")) == model_id),
        {},
    )
    character = load_character(model_id)
    profile_path = ROOT / "models" / model_id / "prompt_profile.json"
    profile: dict[str, Any] = {}
    if profile_path.exists():
        profile = json.loads(profile_path.read_text(encoding="utf-8"))

    # Fail fast before queuing anything if the trained LoRA is not deployed.
    # Registry lora wins; older entries may lack it — fall back to the canonical
    # <id>.safetensors name produced by scripts/train_lora.sh deployment.
    registry_lora = str(entry.get("lora") or f"{model_id}.safetensors")
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

    workflow = load_workflow("flux_lora")
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
        (destination / "caption.txt").write_text(args.caption.strip() + "\n", encoding="utf-8")
        (destination / "prompt.txt").write_text(positive.strip() + "\n", encoding="utf-8")
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    print(destination)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Queue sequential FLUX jobs in ComfyUI")
    # Manager container reaches ComfyUI by compose service name; respect COMFY_URL.
    root.add_argument("--url", default=os.environ.get("COMFY_URL", "http://comfyui:8188"))
    commands = root.add_subparsers(dest="command", required=True)

    candidates = commands.add_parser("candidates", help="Generate reference faces")
    candidates.add_argument("--count", type=int, choices=range(1, 21), default=8)
    candidates.add_argument("--seed", type=int, default=17023)
    candidates.add_argument("--prompt", default="", help="Override the character prompt (guardrail phrase enforced)")
    candidates.add_argument("--negative", default="", help="Extra negative terms; canonical guardrail terms are always kept")
    candidates.add_argument("--model", default=None, help="Model id from models/registry.json (default: active)")
    candidates.set_defaults(handler=generate_candidates)

    expand = commands.add_parser("expand", help="Expand single reference into dataset via Kontext (sequential)")
    expand.add_argument("--reference", required=True, help="Path to chosen candidate, e.g. models/<id>/reference/candidates/<id>-03.jpg")
    expand.add_argument("--count", type=int, choices=range(1, 21), default=8)
    expand.add_argument("--seed", type=int, default=38447)
    expand.add_argument("--prompt", default="", help="Override the scene rotation with one prompt (guardrail phrase enforced)")
    expand.add_argument("--negative", default="", help="Extra negative terms; canonical guardrail terms are always kept")
    expand.add_argument("--model", default=None, help="Model id from models/registry.json (default: active)")
    expand.set_defaults(handler=expand_dataset)

    post = commands.add_parser("post", help="Generate one folder for manual posting")
    post.add_argument("--prompt", default="", help="Positive prompt; omit to build it from character.json + prompt_profile.json")
    post.add_argument("--caption", required=True)
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
    post.set_defaults(handler=generate_post)
    return root


def main() -> int:
    args = parser().parse_args()
    args.model_id = resolve_model_id(args.model)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
