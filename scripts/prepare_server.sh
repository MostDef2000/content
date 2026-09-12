#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p \
  "${ROOT}/runtime/models/checkpoints" \
  "${ROOT}/runtime/models/loras" \
  "${ROOT}/runtime/input" \
  "${ROOT}/runtime/output" \
  "${ROOT}/runtime/user" \
  "${ROOT}/dataset/images" \
  "${ROOT}/lora/output" \
  "${ROOT}/posts" \
  "${ROOT}/reference/candidates"

if [[ ! -f "${ROOT}/.env" ]]; then
  cp "${ROOT}/.env.example" "${ROOT}/.env"
fi

echo "Directories are ready. Add HF_TOKEN to .env before downloading FLUX.1-dev."
