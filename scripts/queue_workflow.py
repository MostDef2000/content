from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "runtime" / "output"


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


def generate_candidates(args: argparse.Namespace) -> None:
    destination = ROOT / "reference" / "candidates"
    destination.mkdir(parents=True, exist_ok=True)
    template = load_workflow("bootstrap")

    for index in range(1, args.count + 1):
        workflow = copy.deepcopy(template)
        workflow["31"]["inputs"]["seed"] = args.seed + index - 1
        workflow["9"]["inputs"]["filename_prefix"] = f"reference/valery-{index:02d}"
        print(f"Generating candidate {index}/{args.count}...")
        images = queue_and_wait(args.url, workflow)
        source = output_path(images[0])
        target = destination / f"valery-{index:02d}{source.suffix}"
        shutil.move(source, target)
        print(target)


VARIATION_PROMPTS = [
    "same fictional adult woman, identical face, slight head turn, soft smile, fitted olive athletic top, neutral studio, photorealistic portrait",
    "same woman, identical face, hair slightly pulled back, white fitted tee, bright daylight, casual confident pose",
    "same woman, identical face, gym setting, black sports bra and leggings, tasteful athletic framing, strong posture",
    "same woman, identical face, evening rooftop, fitted dark fashion top, city lights bokeh, confident gaze",
    "same woman, identical face, close-up beauty shot, dewy skin, minimal makeup, shallow depth of field",
    "same woman, identical face, leaning against wall, beige crop top, relaxed fit, urban street style",
    "same woman, identical face, fitness studio mirror, athletic wear, dynamic stance, professional lighting",
    "same woman, identical face, outdoor cafe, light summer dress, natural smile, lifestyle editorial",
]


def expand_dataset(args: argparse.Namespace) -> None:
    source = Path(args.reference).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Reference image not found: {source}")
    # ComfyUI LoadImage reads from runtime/input — copy there sequentially (one image at a time)
    input_dir = ROOT / "runtime" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = ROOT / "dataset" / "images"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Keep original as first dataset entry
    shutil.copy2(source, dataset_dir / "valery_00.jpg")

    template = load_workflow("kontext_variation")
    # Place reference where LoadImage can find it (sequential use — never two models at once on 12GB)
    staged = input_dir / "reference_selected.jpg"
    shutil.copy2(source, staged)

    for idx, prompt in enumerate(VARIATION_PROMPTS[: args.count], start=1):
        workflow = copy.deepcopy(template)
        workflow["6"]["inputs"]["text"] = prompt
        workflow["41"]["inputs"]["image"] = staged.name
        workflow["25"]["inputs"]["noise_seed"] = args.seed + idx
        workflow["9"]["inputs"]["filename_prefix"] = f"dataset/valery-{idx:02d}"
        print(f"Generating dataset variation {idx}/{args.count}: {prompt[:60]}...")
        images = queue_and_wait(args.url, workflow)
        img = output_path(images[0])
        target = dataset_dir / f"valery_{idx:02d}{img.suffix}"
        shutil.move(img, target)
        print(target)
    print(f"Dataset ready in {dataset_dir} ({len(list(dataset_dir.glob('*.jpg')))} images). Run lora/caption.py next.")


def generate_post(args: argparse.Namespace) -> None:
    workflow = load_workflow("flux_lora")
    workflow["6"]["inputs"]["text"] = args.prompt
    workflow["31"]["inputs"]["seed"] = args.seed
    workflow["40"]["inputs"]["strength_model"] = args.lora_strength
    workflow["9"]["inputs"]["filename_prefix"] = "staging/valery"

    post_name = args.name or datetime.now().strftime("%Y-%m-%d-%H%M%S")
    if Path(post_name).name != post_name:
        raise ValueError("Post name must not contain a path")
    destination = ROOT / "posts" / post_name
    destination.mkdir(parents=True, exist_ok=False)

    try:
        images = queue_and_wait(args.url, workflow)
        source = output_path(images[0])
        shutil.move(source, destination / f"photo{source.suffix}")
        (destination / "caption.txt").write_text(args.caption.strip() + "\n", encoding="utf-8")
        (destination / "prompt.txt").write_text(args.prompt.strip() + "\n", encoding="utf-8")
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
    candidates.set_defaults(handler=generate_candidates)

    expand = commands.add_parser("expand", help="Expand single reference into dataset via Kontext (sequential)")
    expand.add_argument("--reference", required=True, help="Path to chosen candidate, e.g. reference/candidates/valery-03.jpg")
    expand.add_argument("--count", type=int, choices=range(1, 21), default=8)
    expand.add_argument("--seed", type=int, default=38447)
    expand.set_defaults(handler=expand_dataset)

    post = commands.add_parser("post", help="Generate one folder for manual posting")
    post.add_argument("--prompt", required=True)
    post.add_argument("--caption", required=True)
    post.add_argument("--name")
    post.add_argument("--seed", type=int, default=27191)
    post.add_argument("--lora-strength", type=float, default=0.8)
    post.set_defaults(handler=generate_post)
    return root


def main() -> int:
    args = parser().parse_args()
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
