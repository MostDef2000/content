#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLKIT="${ROOT}/ai-toolkit"

# Copy the newest *.safetensors checkpoint of the given arch from
# models/<model id>/lora/output/<model id>_<arch>_v1/ into the runtime slot
# where ComfyUI/manager load the character LoRA:
#   sdxl → runtime/models/loras/<model id>.safetensors
#   flux → runtime/models/loras/<model id>-flux.safetensors
# The templates name the training config exactly "<model id>_<arch>_v1", so
# only that checkpoint subfolder is searched (the neighbouring arch's run can
# never win the "newest file" race). Args:
#   copy_lora_output MODEL_ID ARCH [ROOT]
# Third arg overrides ROOT for fixture testing.
copy_lora_output() {
  local model_id="$1"
  local arch="$2"
  local root="${3:-${ROOT}}"
  local out_dir="${root}/models/${model_id}/lora/output"
  local config_dir="${out_dir}/${model_id}_${arch}_v1"
  local dest
  if [[ "${arch}" == "flux" ]]; then
    dest="${root}/runtime/models/loras/${model_id}-flux.safetensors"
  else
    dest="${root}/runtime/models/loras/${model_id}.safetensors"
  fi
  local newest
  newest="$(find "${config_dir}" -type f -name '*.safetensors' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -n 1 | cut -d' ' -f2- || true)"
  if [[ -z "${newest}" ]]; then
    echo "Training finished but no ${arch} *.safetensors checkpoint found under ${config_dir}; nothing to copy to ${dest}." >&2
    return 3
  fi
  mkdir -p "$(dirname "${dest}")"
  cp -f "${newest}" "${dest}"
  echo "LoRA ready (${arch}): ${newest} -> ${dest}"
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

# ARCH: second positional argument (after MODEL_ID); default "sdxl" for
# back-compat. Selects the training template and the runtime LoRA slot:
#   sdxl → lora/train_config.template.yaml      → runtime/train/<id>.yaml
#         → runtime/models/loras/<id>.safetensors
#   flux → lora/train_config.flux.template.yaml → runtime/train/<id>_flux.yaml
#         → runtime/models/loras/<id>-flux.safetensors
if [[ $# -ge 2 ]]; then
  ARCH="$2"
else
  ARCH="sdxl"
fi
if [[ ! "${ARCH}" =~ ^[a-z]+$ ]]; then
  echo "Invalid arch '${ARCH}': must be lowercase letters (^[a-z]+\$)." >&2
  exit 2
fi
if [[ "${ARCH}" != "sdxl" && "${ARCH}" != "flux" ]]; then
  echo "Invalid arch '${ARCH}': supported values are 'sdxl' and 'flux'." >&2
  exit 2
fi

if [[ "${ARCH}" == "flux" ]]; then
  TEMPLATE="${ROOT}/lora/train_config.flux.template.yaml"
  RENDERED="${ROOT}/runtime/train/${MODEL_ID}_flux.yaml"
else
  TEMPLATE="${ROOT}/lora/train_config.template.yaml"
  RENDERED="${ROOT}/runtime/train/${MODEL_ID}.yaml"
fi

DATASET_DIR="${ROOT}/models/${MODEL_ID}/dataset/images"

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

# Config template is mandatory — no legacy single-model fallback (which could
# silently train the wrong dataset). Fail loudly before any render attempt.
if [[ ! -f "${TEMPLATE}" ]]; then
  echo "Missing config template: ${TEMPLATE}" >&2
  exit 2
fi

mkdir -p "$(dirname "${RENDERED}")"
sed "s/{{MODEL_ID}}/${MODEL_ID}/g" "${TEMPLATE}" > "${RENDERED}"

# Test seam: SKIP_TRAIN=1 stops right after rendering without running training.
# Never set in production.
if [[ "${SKIP_TRAIN:-}" == "1" ]]; then
  echo "SKIP_TRAIN=1: rendered ${RENDERED}; not running training."
  exit 0
fi

cd "${TOOLKIT}"
train_rc=0
"${TOOLKIT}/venv/bin/python" run.py "${RENDERED}" || train_rc=$?
if [[ "${train_rc}" -ne 0 ]]; then
  echo "Training '${MODEL_ID}' failed: run.py exited with code ${train_rc}." >&2
  exit "${train_rc}"
fi

copy_lora_output "${MODEL_ID}" "${ARCH}"
