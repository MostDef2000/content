#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLKIT="${ROOT}/ai-toolkit"
# kohya sd-scripts checkout for the FLUX trainer (spec 010): bind-mounted at
# /workspace/kohya in the manager container (compose manager volumes).
KOHYA="${ROOT}/kohya"
# kohya venv lives OUTSIDE the checkout, on the persistent rw runtime mount:
# the kohya clone ships a HOST uv-venv (venv/) that is broken inside the
# container, and the checkout itself is host-owned — so the self-healing
# bootstrap needs a writable, persistent location of its own (spec 010).
KOHYA_VENV="${ROOT}/runtime/kohya-venv"

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

# ---- FLUX (kohya) helpers (spec 010) --------------------------------------

# resolve_existing CANDIDATE...: print the first candidate that exists as a
# regular file, non-zero exit when none does (callers fail fast with a
# per-component actionable message).
resolve_existing() {
  local cand
  for cand in "$@"; do
    if [[ -f "${cand}" ]]; then
      printf '%s\n' "${cand}"
      return 0
    fi
  done
  return 1
}

# require_flux_int NAME VALUE: numeric placeholders render as bare integers
# (the templates keep them quoted so the templates themselves stay valid
# TOML); reject anything non-numeric before the render.
require_flux_int() {
  local name="$1" value="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "Invalid ${name} '${value}': must be a non-negative integer." >&2
    exit 2
  fi
}

# render_kohya_toml TEMPLATE DEST: render a flux kohya TOML template. String
# placeholders keep their quotes; the quoted numeric placeholders ("{{STEPS}}",
# "{{SEED}}", "{{RESOLUTION}}", "{{BATCH}}") are replaced together with the
# quotes because sd-scripts expects bare integers there.
render_kohya_toml() {
  sed \
    -e "s/{{MODEL_ID}}/${MODEL_ID}/g" \
    -e "s|{{OUTPUT_DIR}}|${FLUX_OUTPUT_DIR}|g" \
    -e "s|{{DATASET_DIR}}|${DATASET_DIR}|g" \
    -e "s/\"{{STEPS}}\"/${FLUX_STEPS}/g" \
    -e "s/\"{{SEED}}\"/${FLUX_SEED}/g" \
    -e "s/\"{{RESOLUTION}}\"/${FLUX_RESOLUTION}/g" \
    -e "s/\"{{BATCH}}\"/${FLUX_BATCH}/g" \
    -e "s|{{FLUX_DIT}}|${FLUX_DIT}|g" \
    -e "s|{{FLUX_AE}}|${FLUX_AE}|g" \
    -e "s|{{FLUX_CLIP_L}}|${FLUX_CLIP_L}|g" \
    -e "s|{{FLUX_T5XXL}}|${FLUX_T5XXL}|g" \
    "$1" > "$2"
}

# Deploy the newest FLUX kohya checkpoint to the registry FLUX slot:
#   runtime/models/loras/<model id>-flux.safetensors
# The DASH form is the contract of management/main.py `_lora_name_for_engine`
# (registry slot "lora_flux") — do NOT switch to underscores. tmp + mv keeps
# the copy atomic so a concurrent ComfyUI load never sees a partial file.
# Args:
#   deploy_flux_lora MODEL_ID [ROOT]
deploy_flux_lora() {
  local model_id="$1"
  local root="${2:-${ROOT}}"
  local out_dir="${root}/models/${model_id}/lora/output/${model_id}_flux_v1"
  local dest
  dest="${root}/runtime/models/loras/${model_id}-flux.safetensors"
  local newest tmp
  newest="$(find "${out_dir}" -type f -name '*.safetensors' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -n 1 | cut -d' ' -f2- || true)"
  if [[ -z "${newest}" ]]; then
    echo "Training finished but no flux *.safetensors checkpoint found under ${out_dir}; nothing to copy to ${dest}." >&2
    return 3
  fi
  mkdir -p "$(dirname "${dest}")"
  tmp="$(mktemp "$(dirname "${dest}")/.${model_id}-flux.tmp.XXXXXX")"
  cp -f "${newest}" "${tmp}"
  # mktemp creates 0600 files; restore the usual 0644 so non-root readers on
  # the shared runtime mount are not locked out.
  chmod 644 "${tmp}"
  mv -f "${tmp}" "${dest}"
  echo "LoRA ready (flux): ${newest} -> ${dest}"
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
#   flux → lora/train_config_flux_kohya{,_dataset}.template.toml
#         → runtime/train/<id>_flux_kohya{,_dataset}.toml
#         → runtime/models/loras/<id>-flux.safetensors
# The flux branch renders TWO config files because of how sd-scripts consumes
# them: --config_file (script arguments; every TOML section is flattened into
# the argparse namespace) and --dataset_config (validated strictly by a
# voluptuous schema that rejects any non-dataset keys), so one merged file
# cannot serve both options (spec 010).
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
  TEMPLATE="${ROOT}/lora/train_config_flux_kohya.template.toml"
  TEMPLATE_DATASET="${ROOT}/lora/train_config_flux_kohya_dataset.template.toml"
  RENDERED="${ROOT}/runtime/train/${MODEL_ID}_flux_kohya.toml"
  RENDERED_DATASET="${ROOT}/runtime/train/${MODEL_ID}_flux_kohya_dataset.toml"
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

# ---- FLUX (kohya) branch (spec 010) -----------------------------------------
# Self-contained: preflight, dual render, SKIP_TRAIN seam, venv bootstrap,
# launch, deploy. Success ends with exit 0 and every failure path exits with
# its own code, so the untouched SDXL path below only ever runs for sdxl.
if [[ "${ARCH}" == "flux" ]]; then
  # Fail fast on a missing trainer checkout (same discipline as the
  # ai-toolkit guard): /workspace/kohya must be bind-mounted in the container.
  if [[ ! -f "${KOHYA}/flux_train_network.py" ]]; then
    echo "kohya sd-scripts not found at ${KOHYA} (flux_train_network.py missing); check the /workspace/kohya bind mount in compose.yaml." >&2
    exit 2
  fi
  if [[ ! -f "${TEMPLATE_DATASET}" ]]; then
    echo "Missing config template: ${TEMPLATE_DATASET}" >&2
    exit 2
  fi

  # Numeric render overrides ({{STEPS}}/{{SEED}}/{{RESOLUTION}}/{{BATCH}}):
  # the templates quote them to stay valid TOML, so the render must only ever
  # write bare integers — validate the env overrides before rendering.
  # Defaults target the 12 GB card; STEPS=20 makes a usable smoke run.
  FLUX_STEPS="${STEPS:-1500}"
  FLUX_SEED="${SEED:-1}"
  FLUX_RESOLUTION="${RESOLUTION:-768}"
  FLUX_BATCH="${BATCH:-1}"
  require_flux_int "STEPS" "${FLUX_STEPS}"
  require_flux_int "SEED" "${FLUX_SEED}"
  require_flux_int "RESOLUTION" "${FLUX_RESOLUTION}"
  require_flux_int "BATCH" "${FLUX_BATCH}"

  # The FLUX trainer loads four LOCAL single-file checkpoints (no HF
  # auto-download in sd-scripts), resolved from the shared ComfyUI runtime
  # layout that scripts/download_models.sh fills. Fail fast with the fix.
  MODELS_DIR="${ROOT}/runtime/models"
  FLUX_DIT="$(resolve_existing \
    "${MODELS_DIR}/checkpoints/flux1-dev-fp8.safetensors" \
    "${MODELS_DIR}/checkpoints/flux1-dev.safetensors")" || {
    echo "FLUX base checkpoint (DiT) not found: expected ${MODELS_DIR}/checkpoints/flux1-dev-fp8.safetensors (or flux1-dev.safetensors); run scripts/download_models.sh on the server first." >&2
    exit 2
  }
  FLUX_AE="$(resolve_existing "${MODELS_DIR}/vae/ae.safetensors")" || {
    echo "FLUX AutoEncoder not found: expected ${MODELS_DIR}/vae/ae.safetensors; run scripts/download_models.sh on the server first." >&2
    exit 2
  }
  FLUX_CLIP_L="$(resolve_existing "${MODELS_DIR}/text_encoders/clip_l.safetensors")" || {
    echo "CLIP-L text encoder not found: expected ${MODELS_DIR}/text_encoders/clip_l.safetensors; run scripts/download_models.sh on the server first." >&2
    exit 2
  }
  FLUX_T5XXL="$(resolve_existing \
    "${MODELS_DIR}/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors" \
    "${MODELS_DIR}/text_encoders/t5xxl_fp16.safetensors" \
    "${MODELS_DIR}/text_encoders/t5xxl_fp8_e4m3fn.safetensors")" || {
    echo "T5-XXL text encoder not found: expected ${MODELS_DIR}/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors (or t5xxl_fp16.safetensors); run scripts/download_models.sh on the server first." >&2
    exit 2
  }

  FLUX_OUTPUT_DIR="${ROOT}/models/${MODEL_ID}/lora/output/${MODEL_ID}_flux_v1"

  mkdir -p "${FLUX_OUTPUT_DIR}" "$(dirname "${RENDERED}")"
  render_kohya_toml "${TEMPLATE}" "${RENDERED}"
  render_kohya_toml "${TEMPLATE_DATASET}" "${RENDERED_DATASET}"

  # Test seam: SKIP_TRAIN=1 stops right after rendering without running
  # training (same contract as the SDXL seam below). Never set in production.
  if [[ "${SKIP_TRAIN:-}" == "1" ]]; then
    echo "SKIP_TRAIN=1: rendered ${RENDERED}; not running training."
    exit 0
  fi

  # Memory preflight (spec 010 D-006, bounded-memory rounds): train-1792249442
  # died to host SIGKILL because the box budget is < ~26GB — a 20-minute train
  # that dies half-way is worse than a 1-second refusal to start. Pure stdlib
  # (NO torch — this runs before the venv python), placed after the SKIP_TRAIN
  # seam and before the accelerate launch: measures the DiT checkpoint size
  # from its safetensors header (struct '<Q' + json data_offsets/nbytes, with
  # an absolute 256MB header ceiling and strict entry validation — the
  # trainer loads exactly this file as {{FLUX_DIT}} ->
  # --pretrained_model_name_or_path), reads MemAvailable + SwapTotal and the
  # cgroup state (v2 memory.max + memory.current, v1 fallback; seam root
  # MEM_PREFLIGHT_CGROUP_ROOT, default /sys/fs/cgroup), logs every value
  # under a PREFLIGHT marker, and aborts non-zero when the effective budget
  # < measured_ckpt_bytes + SLACK — budget = min(MemAvailable,
  # cgroup_limit - cgroup_usage) for a finite limit, and a finite limit with
  # unreadable usage fails CLOSED (never silently host-only).
  # SLACK defaults to 13GB (T5 9.4GB + CLIP 1.7GB + ~1.9GB headroom), must be
  # a non-negative integer, and is overridable via the
  # MEM_PREFLIGHT_SLACK_BYTES seam (tests set 0 — threshold = measured bytes
  # only).
  FLUX_DIT="${FLUX_DIT}" python3 - <<'PY'
import json
import os
import struct
import sys

MAX_HEADER_BYTES = 256 * 1024 * 1024


def cgroup_state():
    """(limit, usage, source): cgroup v2 under <root>/ (memory.max +
    memory.current), v1 fallback under <root>/memory/ (memory.limit_in_bytes
    + memory.usage_in_bytes). "max" (v2) or >= 2^60 (v1) = unlimited; a root
    that cannot be read at all -> (None, None, "unknown") = host-only, while
    a FINITE limit with unreadable usage returns usage=None and fails
    closed at the gate (never a silent host-only)."""
    root = os.environ.get("MEM_PREFLIGHT_CGROUP_ROOT", "/sys/fs/cgroup")
    try:
        with open(os.path.join(root, "memory.max")) as fh:
            raw = fh.read().strip()
        limit = None if raw == "max" else int(raw)
        try:
            with open(os.path.join(root, "memory.current")) as fh:
                usage = int(fh.read().strip())
        except (OSError, ValueError):
            usage = None
        return limit, usage, "cgroup-v2"
    except (OSError, ValueError):
        pass
    try:
        with open(os.path.join(root, "memory", "memory.limit_in_bytes")) as fh:
            raw = int(fh.read().strip())
        limit = None if raw >= (1 << 60) else raw
        try:
            with open(os.path.join(root, "memory", "memory.usage_in_bytes")) as fh:
                usage = int(fh.read().strip())
        except (OSError, ValueError):
            usage = None
        return limit, usage, "cgroup-v1"
    except (OSError, ValueError):
        return None, None, "unknown"


def meminfo():
    avail = swap = None
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) * 1024
            elif line.startswith("SwapTotal:"):
                swap = int(line.split()[1]) * 1024
    if avail is None:
        sys.exit("preflight: cannot read MemAvailable from /proc/meminfo")
    return avail, swap


def ckpt_tensor_bytes(path):
    # Corrupt/truncated checkpoint → clean, marked abort (word "preflight"),
    # never a traceback; header_len is bounded by the ABSOLUTE 256MB ceiling
    # FIRST, then by the file size, so garbage first-8-bytes can never
    # trigger a huge header read. Every entry must be a JSON object with
    # data_offsets = [start, end] (0 <= start <= end) or a non-negative
    # nbytes integer — anything malformed is a clean refusal, never a
    # silent undercount.
    file_size = os.path.getsize(path)
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            sys.exit(f"preflight: cannot parse checkpoint header for {path}: truncated (file {file_size} bytes)")
        (header_len,) = struct.unpack("<Q", raw)
        if header_len > MAX_HEADER_BYTES:
            sys.exit(f"preflight: cannot parse checkpoint header for {path}: header length {header_len} exceeds the {MAX_HEADER_BYTES} byte ceiling (file {file_size} bytes)")
        if header_len > file_size - 8:
            sys.exit(f"preflight: cannot parse checkpoint header for {path}: header length {header_len} exceeds file size {file_size} (file {file_size} bytes)")
        try:
            header = json.loads(fh.read(header_len))
            if not isinstance(header, dict):
                raise ValueError("header is not a JSON object")
        except (ValueError, UnicodeDecodeError) as e:
            sys.exit(f"preflight: cannot parse checkpoint header for {path}: {e} (file {file_size} bytes)")
    total = 0
    # data section = bytes after the 8-byte length and the JSON header; every
    # declared span must fit inside it (real checkpoints always do).
    data_section = file_size - 8 - header_len
    for key, value in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(value, dict):
            sys.exit(f"preflight: cannot parse checkpoint header for {path}: entry {key!r} is not a JSON object (file {file_size} bytes)")
        offsets = value.get("data_offsets")
        if offsets is not None:
            if (
                not isinstance(offsets, (list, tuple))
                or len(offsets) != 2
                or not all(type(n) is int for n in offsets)
                or not 0 <= offsets[0] <= offsets[1]
            ):
                sys.exit(f"preflight: cannot parse checkpoint header for {path}: malformed data_offsets for {key!r}: {offsets!r} (file {file_size} bytes)")
            if offsets[1] > data_section:
                sys.exit(f"preflight: data_offsets for '{key}' end {offsets[1]} exceeds the data section {data_section} (file {file_size} bytes)")
            total += offsets[1] - offsets[0]
        elif "nbytes" in value:
            nbytes = value["nbytes"]
            if type(nbytes) is not int or nbytes < 0:
                sys.exit(f"preflight: cannot parse checkpoint header for {path}: invalid nbytes for {key!r}: {nbytes!r} (file {file_size} bytes)")
            total += nbytes
        else:
            sys.exit(f"preflight: unsupported entry for '{key}': no data_offsets or nbytes (file {os.path.getsize(path)} bytes)")
    tensors = len(header) - (1 if "__metadata__" in header else 0)
    if tensors == 0:
        sys.exit(f"preflight: empty checkpoint header: no tensors (file {os.path.getsize(path)} bytes)")
    return total, header_len, tensors


ckpt_path = os.environ["FLUX_DIT"]
if not os.path.isfile(ckpt_path):
    sys.exit(f"preflight: flux checkpoint not found: {ckpt_path}")
ckpt_bytes, header_len, tensors = ckpt_tensor_bytes(ckpt_path)
limit, usage, cgroup_src = cgroup_state()
avail, swap = meminfo()
raw_slack = os.environ.get("MEM_PREFLIGHT_SLACK_BYTES", 13 * 1024**3)
try:
    slack = int(raw_slack)
except (TypeError, ValueError):
    sys.exit(f"preflight: invalid slack: MEM_PREFLIGHT_SLACK_BYTES={raw_slack!r} is not an integer")
if slack < 0:
    sys.exit(f"preflight: invalid slack: MEM_PREFLIGHT_SLACK_BYTES={raw_slack!r} is negative")
required = ckpt_bytes + slack
if limit is None:
    budget, remaining = avail, None
else:
    if usage is None:
        sys.exit(f"preflight: fail closed: cgroup limit {limit} but usage unreadable (source {cgroup_src})")
    remaining = limit - usage
    budget = min(avail, remaining)
print(f"PREFLIGHT ckpt={ckpt_path}")
print(
    f"PREFLIGHT ckpt_tensor_bytes={ckpt_bytes} "
    f"ckpt_file_bytes={os.path.getsize(ckpt_path)} "
    f"header_len={header_len} tensors={tensors}"
)
print(
    f"PREFLIGHT mem_available={avail} swap_total={swap} "
    f"cgroup_limit={limit} cgroup_usage={usage} cgroup_src={cgroup_src}"
)
print(
    f"PREFLIGHT slack={slack} required={required} "
    f"budget={budget} remaining={remaining}"
)
if budget < required:
    if limit is None:
        where = f"mem_available {avail}, cgroup unlimited"
    else:
        where = (
            f"mem_available {avail}, cgroup limit {limit} - usage {usage} "
            f"= remaining {remaining}"
        )
    sys.exit(
        "preflight: not enough memory to start the flux trainer: required "
        f"(checkpoint {ckpt_bytes} + slack {slack}) = {required} bytes, "
        f"budget {budget} bytes ({where})"
    )
print("PREFLIGHT ok")
PY

  # Venv bootstrap (first flux run, and self-heal on every broken state): the
  # host uv-venv shipped with the kohya clone does not work inside the
  # container (interpreter symlinks point at the host), so the venv is
  # (re)created from the container python3 at runtime/kohya-venv — NOT inside
  # the host-owned kohya checkout. ORDER MATTERS: install the cu130 torch
  # wheels FIRST, then requirements.txt — a plain "pip install -r
  # requirements.txt" first would pull a non-CUDA torch build. cu130 (not
  # cu124): the GPU box is Blackwell (sm_120) and the cu124 wheels carry no
  # sm_120 kernels, so the first conv2d dies with "CUDA error: no kernel
  # image is available for execution on the device" (prod case
  # train-1792158920); precedent: the ai-toolkit venv was rebaked to
  # torch 2.13.0+cu130.
  # rm -rf before venv --clear is the self-heal: `python3 -m venv` alone dies
  # with "Error: [Errno 2] No such file or directory" on a dir whose
  # bin/python3 is a broken symlink (the host uv-venv case, repro
  # train-1792131743); rm is fine because the container user writes to the
  # runtime mount, and --clear makes even a leftover partial venv harmless.
  # The probe is torch import + the accelerate binary (the final artifact of
  # the requirements install) + the .torch-flavor marker: torch imports fine
  # on CPU paths even from a venv built against a stale index, so the guard
  # also requires the marker to carry the current index slug
  # (TORCH_INDEX_SLUG) — no marker (pre-rebake venv) or a stale one (wrong
  # CUDA build for the current image) means a full rebuild, not a skip.
  TORCH_INDEX_SLUG="cu130"
  if ! "${KOHYA_VENV}/bin/python3" -c "import torch, torchvision" >/dev/null 2>&1 \
     || [[ ! -x "${KOHYA_VENV}/bin/accelerate" ]] \
     || [[ ! -f "${KOHYA_VENV}/.torch-flavor" ]] || ! grep -qx "${TORCH_INDEX_SLUG}" "${KOHYA_VENV}/.torch-flavor"; then
    echo "Bootstrapping kohya venv at ${KOHYA_VENV} (first flux run only; expect 10-20 minutes)..." >&2
    rm -rf "${KOHYA_VENV}"
    python3 -m venv --clear "${KOHYA_VENV}"
    "${KOHYA_VENV}/bin/python" -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/${TORCH_INDEX_SLUG}"
    # Flavor marker: torch imports on CPU paths even when the venv holds the
    # WRONG torch build (old cu124 has no Blackwell/sm_120 kernels), so the
    # guard needs a durable record of which index slug this venv was built
    # from. Written only after a successful torch pip (set -e kills the run
    # on failure, so the marker never lies about a half-install); normal
    # umask (644) is enough — it carries no secrets.
    echo "${TORCH_INDEX_SLUG}" > "${KOHYA_VENV}/.torch-flavor"
    # requirements.txt carries a relative editable install of sd-scripts
    # itself ("-e ."); pip resolves it against the CURRENT WORKING DIRECTORY,
    # so the install must run from inside the checkout — from the manager
    # cwd it fails with "Obtaining file:///workspace ... neither 'setup.py'
    # nor 'pyproject.toml' found" (prod case train-1792134096).
    ( cd "${KOHYA}" && "${KOHYA_VENV}/bin/python" -m pip install -r "${KOHYA}/requirements.txt" )
  fi

  # Compat shim (spec 010 prod crash): in Comfy-Org flux1-dev-fp8 the
  # unprefixed scaled_fp8 marker sits EARLY in key order (NOT first —
  # keys[0] is prefixed, else analyze_checkpoint_state would die on
  # max() before any load), so upstream's ComfyUI-prefix strip loop
  # (library/flux_utils.py load_flow_model) breaks on it mid-dict:
  # keys before it are renamed, the REMAINING prefixed keys stay
  # unrenamed — the model stays partially meta and training dies at
  # unet.to("cpu"); the ostris scaled-fp8 T5 also carries per-tensor
  # .scale_weight keys kohya ignores. The shim (a copy of
  # scripts/kohya_compat_shim.py in the venv's site-packages,
  # auto-imported by sitecustomize in EVERY venv python process) patches
  # library.flux_utils on first import to fix both.
  # Installed on EVERY flux run, OUTSIDE the rebuild guard above: a healthy
  # pre-shim venv (the prod state after W3d) never enters that branch, so a
  # rebuild-only install would never land on the box. The cp is
  # UNCONDITIONAL so future shim updates refresh installed venvs without a
  # 10-20 minute rebuild; a failing cp aborts via set -e.
  SP_DIR="$(echo "${KOHYA_VENV}"/lib/python*/site-packages)"
  cp "${ROOT}/scripts/kohya_compat_shim.py" "${SP_DIR}/kohya_compat_shim.py"
  # sitecustomize is auto-imported by the interpreter at startup; the gate
  # matches the ACTIVATION line (import + install()): a legacy import-only
  # sitecustomize (pre-F1 venv) without install() never registers the
  # meta_path finder, so it must be rewritten too — and a rewrite (not an
  # append) keeps the import list free of duplicates across runs.
  if ! grep -q "kohya_compat_shim.install" "${SP_DIR}/sitecustomize.py" 2>/dev/null; then
    printf 'import kohya_compat_shim\nkohya_compat_shim.install()\n' > "${SP_DIR}/sitecustomize.py"
  fi

  # Shared HF cache (FLUX.1-dev is pre-cached; CLIP/T5 tokenizer files download
  # on first run). HF_TOKEN is exported by the .env source block above.
  export HF_HOME="${ROOT}/runtime/hf-cache"
  mkdir -p "${HF_HOME}"

  cd "${KOHYA}"
  train_rc=0
  "${KOHYA_VENV}/bin/accelerate" launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    flux_train_network.py \
    --config_file "${RENDERED}" \
    --dataset_config "${RENDERED_DATASET}" || train_rc=$?
  if [[ "${train_rc}" -ne 0 ]]; then
    echo "Training '${MODEL_ID}' failed: flux_train_network.py exited with code ${train_rc}." >&2
    exit "${train_rc}"
  fi

  deploy_flux_lora "${MODEL_ID}"
  exit 0
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
