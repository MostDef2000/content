#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLKIT="${ROOT}/ai-toolkit"

if [[ ! -d "${TOOLKIT}/.git" ]]; then
  git clone https://github.com/ostris/ai-toolkit.git "${TOOLKIT}"
fi

python3 -m venv "${TOOLKIT}/venv"
"${TOOLKIT}/venv/bin/pip" install --upgrade pip
"${TOOLKIT}/venv/bin/pip" install --no-cache-dir \
  torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130
"${TOOLKIT}/venv/bin/pip" install -r "${TOOLKIT}/requirements.txt"

"${TOOLKIT}/venv/bin/python" -c \
  'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(), torch.cuda.get_arch_list())'
