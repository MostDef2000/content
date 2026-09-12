#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLKIT="${ROOT}/ai-toolkit"

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

image_count="$(find "${ROOT}/dataset/images" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)"
caption_count="$(find "${ROOT}/dataset/images" -maxdepth 1 -type f -iname '*.txt' | wc -l)"
if [[ "${image_count}" -lt 15 ]] || [[ "${image_count}" -ne "${caption_count}" ]]; then
  echo "Training requires at least 15 images and one matching .txt caption per image." >&2
  exit 2
fi

cd "${TOOLKIT}"
exec "${TOOLKIT}/venv/bin/python" run.py "${ROOT}/lora/train_config.yaml"
