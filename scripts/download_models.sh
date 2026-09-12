#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ ! -f .env ]] || ! grep -Eq '^HF_TOKEN=.+$' .env; then
  echo "Set HF_TOKEN in ${ROOT}/.env first." >&2
  exit 2
fi

docker compose run --rm --entrypoint bash comfyui -lc '
  set -euo pipefail
  hf download Comfy-Org/flux1-dev flux1-dev-fp8.safetensors \
    --local-dir /opt/ComfyUI/models/checkpoints
  hf download Comfy-Org/flux1-kontext-dev_ComfyUI flux1-dev-kontext_fp8_scaled.safetensors \
    --local-dir /opt/ComfyUI/models/diffusion_models
  hf download comfyanonymous/flux_text_encoders t5xxl_fp8_e4m3fn_scaled.safetensors \
    --local-dir /opt/ComfyUI/models/text_encoders
  hf download comfyanonymous/flux_text_encoders clip_l.safetensors \
    --local-dir /opt/ComfyUI/models/text_encoders
  hf download Comfy-Org/Lumina_Image_2.0_Repackaged --include "split_files/vae/ae.safetensors" \
    --local-dir /tmp/lumina && cp /tmp/lumina/split_files/vae/ae.safetensors /opt/ComfyUI/models/vae/
'

echo "Models downloaded: flux1-dev-fp8, flux1-kontext fp8, text encoders, ae VAE."
