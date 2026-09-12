#!/usr/bin/env bash
set -euo pipefail

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
docker version --format 'Docker server: {{.Server.Version}}'
docker compose version
docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi
