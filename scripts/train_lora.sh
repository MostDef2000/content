#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLKIT="${ROOT}/ai-toolkit"

# Copy the newest *.safetensors checkpoint from models/<model id>/lora/output/
# into runtime/models/loras/<model id>.safetensors (where ComfyUI/manager load
# the character LoRA). Second arg overrides ROOT for fixture testing:
#   copy_lora_output MODEL_ID ROOT
copy_lora_output() {
  local model_id="$1"
  local root="${2:-${ROOT}}"
  local out_dir="${root}/models/${model_id}/lora/output"
  local dest="${root}/runtime/models/loras/${model_id}.safetensors"
  local newest
  newest="$(find "${out_dir}" -type f -name '*.safetensors' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -n 1 | cut -d' ' -f2- || true)"
  if [[ -z "${newest}" ]]; then
    echo "Training finished but no *.safetensors checkpoint found under ${out_dir}; nothing to copy to ${dest}." >&2
    return 3
  fi
  mkdir -p "$(dirname "${dest}")"
  cp -f "${newest}" "${dest}"
  echo "LoRA ready: ${newest} -> ${dest}"
}

if [[ ! -x "${TOOLKIT}/venv/bin/python" ]]; then
  echo "Run scripts/install_trainer.sh first." >&2
  exit 2
fi

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

# MODEL_ID: first positional argument; default = active model from models/registry.json.
if [[ $# -ge 1 ]]; then
  MODEL_ID="$1"
else
  MODEL_ID="$("${TOOLKIT}/venv/bin/python" -c 'import json,sys
models = (json.load(open(sys.argv[1], encoding="utf-8")) or {}).get("models") or []
if not models:
    sys.exit("models/registry.json contains no models")
print(next((str(m["id"]) for m in models if m.get("active")), str(models[0]["id"])))' "${ROOT}/models/registry.json")"
fi

if [[ ! "${MODEL_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
  echo "Invalid model id: ${MODEL_ID}" >&2
  exit 2
fi

DATASET_DIR="${ROOT}/models/${MODEL_ID}/dataset/images"
TEMPLATE="${ROOT}/lora/train_config.template.yaml"
RENDERED="${ROOT}/runtime/train/${MODEL_ID}.yaml"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Dataset directory not found for model '${MODEL_ID}': ${DATASET_DIR}" >&2
  exit 2
fi

image_count="$(find "${DATASET_DIR}" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)"
caption_count="$(find "${DATASET_DIR}" -maxdepth 1 -type f -iname '*.txt' | wc -l)"
if [[ "${image_count}" -lt 15 ]] || [[ "${image_count}" -ne "${caption_count}" ]]; then
  echo "Training '${MODEL_ID}' requires at least 15 images and one matching .txt caption per image in ${DATASET_DIR}." >&2
  exit 2
fi

if [[ -f "${TEMPLATE}" ]]; then
  mkdir -p "$(dirname "${RENDERED}")"
  sed "s/{{MODEL_ID}}/${MODEL_ID}/g" "${TEMPLATE}" > "${RENDERED}"
elif [[ -f "${ROOT}/lora/train_config.yaml" ]]; then
  # Legacy single-model config (pre-registry layout) — use as-is.
  RENDERED="${ROOT}/lora/train_config.yaml"
  echo "Warning: ${TEMPLATE} missing, falling back to legacy ${RENDERED}" >&2
else
  echo "Missing config template: ${TEMPLATE}" >&2
  exit 2
fi

cd "${TOOLKIT}"
train_rc=0
"${TOOLKIT}/venv/bin/python" run.py "${RENDERED}" || train_rc=$?
if [[ "${train_rc}" -ne 0 ]]; then
  echo "Training '${MODEL_ID}' failed: run.py exited with code ${train_rc}." >&2
  exit "${train_rc}"
fi

copy_lora_output "${MODEL_ID}"
