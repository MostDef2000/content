"""Contract + functional tests for the kohya sd-scripts FLUX branch of
scripts/train_lora.sh (spec 010) and the compose bind it depends on.

Three layers:
  1. Text contract of scripts/train_lora.sh: the FLUX deploy slot is strictly
     "<id>-flux.safetensors" (DASH — the contract of management/main.py
     _lora_name_for_engine / registry slot "lora_flux"; the underscore form is
     the regression this pins), accelerate launch carries --config_file +
     --dataset_config, the four local base checkpoints are resolved via
     resolve_existing, the missing-kohya guard and the ^[0-9]+$ numeric
     validation run BEFORE any render, and the dead ai-toolkit flux template
     is no longer read.
  2. Functional runs with SKIP_TRAIN=1 against an isolated fixture ROOT (the
     script derives ROOT from its own location, so the script + templates are
     copied into a tempdir — same isolation style as test_clear_dataset.py):
     both TOMLs render with BARE integers (max_train_steps == 20 at STEPS=20),
     zero "{{" leftovers, exit 0; negatives STEPS=abc / missing T5-XXL /
     missing kohya checkout each exit 2 without rendering; the untouched SDXL
     SKIP_TRAIN path still renders runtime/train/<id>.yaml and exits 0.
  3. deploy_flux_lora extracted from the script and run in bash against a
     fixture tree: empty output dir → rc 3; happy path → newest checkpoint
     atomically deployed to runtime/models/loras/<id>-flux.safetensors with
     mode 0644 and no tmp leftovers.
Plus compose.yaml: the manager service bind-mounts the kohya checkout at
/workspace/kohya (the ROOT/kohya the script guards), next to the #16
ai-toolkit bind.

pytest-compatible; pytest is not required.

Run either way:
    python -m pytest tests/test_train_lora_flux.py -q
    python tests/test_train_lora_flux.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # content/
SCRIPT = REPO_ROOT / "scripts" / "train_lora.sh"

MODEL_ID = "tst01"  # satisfies ^[A-Za-z0-9][A-Za-z0-9_-]*$
STEPS_SMOKE = "20"  # render override used by the happy-path fixtures
# Fixture stand-in for scripts/kohya_compat_shim.py (spec 010 prod-crash fix):
# _build_root always drops it so the bootstrap's shim-install lines can run in
# EVERY fixture test; the install test asserts the site-packages copy equals
# this exact content, pinning the copy source to ${ROOT}/scripts/.
SHIM_PLACEHOLDER = "# fixture placeholder for kohya_compat_shim\n"


# ---------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------


def _script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _build_root(
    *,
    with_kohya: bool = True,
    dit: str = "flux1-dev-fp8.safetensors",
    t5: str = "t5xxl_fp8_e4m3fn_scaled.safetensors",
) -> Path:
    """Isolated ROOT mirroring what train_lora.sh expects on disk:
    scripts/train_lora.sh (copy — the script resolves ROOT from its own
    location), the flux + sdxl templates, an executable ai-toolkit venv stub,
    the kohya checkout marker, a 15-image/15-caption dataset and the four
    local FLUX base checkpoints of the shared ComfyUI runtime layout
    (scripts/download_models.sh). Files are zero-byte stubs — the SKIP_TRAIN
    seam stops before anything reads them."""
    root = Path(tempfile.mkdtemp(prefix="train-lora-flux-"))
    (root / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, root / "scripts" / "train_lora.sh")
    (root / "scripts" / "kohya_compat_shim.py").write_text(SHIM_PLACEHOLDER, encoding="utf-8")
    (root / "lora").mkdir()
    for name in (
        "train_config_flux_kohya.template.toml",
        "train_config_flux_kohya_dataset.template.toml",
        "train_config.template.yaml",
    ):
        shutil.copy(REPO_ROOT / "lora" / name, root / "lora" / name)
    venv_py = root / "ai-toolkit" / "venv" / "bin" / "python"
    venv_py.parent.mkdir(parents=True)
    venv_py.touch()
    venv_py.chmod(0o755)
    if with_kohya:
        marker = root / "kohya" / "flux_train_network.py"
        marker.parent.mkdir(parents=True)
        marker.touch()
    dataset = root / "models" / MODEL_ID / "dataset" / "images"
    dataset.mkdir(parents=True)
    for i in range(15):  # the script requires >=15 images and captions 1:1
        (dataset / f"img{i:02d}.jpg").write_bytes(b"stub")
        (dataset / f"img{i:02d}.txt").write_text("caption", encoding="utf-8")
    models = root / "runtime" / "models"
    (models / "checkpoints").mkdir(parents=True)
    (models / "vae").mkdir()
    (models / "text_encoders").mkdir()
    (models / "checkpoints" / dit).write_bytes(b"d")
    (models / "vae" / "ae.safetensors").write_bytes(b"a")
    (models / "text_encoders" / "clip_l.safetensors").write_bytes(b"c")
    (models / "text_encoders" / t5).write_bytes(b"t")
    return root


def _cleanup(root: Path) -> None:
    shutil.rmtree(root, ignore_errors=True)


def _run_script(root: Path, *args: str, **env_overrides: str) -> subprocess.CompletedProcess:
    """Run the fixture copy of train_lora.sh with a minimal, controlled env
    (no inherited STEPS/SEED/... from the test process)."""
    env = {k: os.environ[k] for k in ("PATH", "HOME") if k in os.environ}
    env.update(env_overrides)
    return subprocess.run(
        ["bash", str(root / "scripts" / "train_lora.sh"), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=root,
    )


def _extract_function(name: str) -> str:
    """Pull a top-level bash function out of the script (definitions close
    with '}' at column 0) so it can be driven standalone in a fixture."""
    lines = _script_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"{name}() {{"))
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "}")
    return "\n".join(lines[start : end + 1]) + "\n"


def _run_deploy(root: Path) -> subprocess.CompletedProcess:
    driver = root / "deploy-driver.sh"
    driver.write_text(
        "set -uo pipefail\n" + _extract_function("deploy_flux_lora") + f'\ndeploy_flux_lora "{MODEL_ID}" "{root}"\n',
        encoding="utf-8",
    )
    return subprocess.run(["bash", str(driver)], capture_output=True, text=True, cwd=root)


# Emulates `python3 -m venv` for the bootstrap tests: a dangling-interpreter
# venv dir fails exactly like the real CPython repro (Errno 2 on
# bin/python3) unless --clear is passed; otherwise the venv bin gets the
# stub interpreter from _VENV_STUB_SH (see below). Anything not `-m venv`
# passes through to the real interpreter. Placeholders
# __LOG__/__STUBTEMPLATE__/__REAL_PYTHON__ are substituted when the shim is
# written.
_PYTHON3_VENV_SHIM = """#!/usr/bin/env bash
log="__LOG__"
echo "python3 $*" >> "$log"
if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then
  shift 2
  dir=""
  clear=0
  for a in "$@"; do
    case "$a" in
      --clear) clear=1 ;;
      -*) ;;
      *) dir="$a" ;;
    esac
  done
  if [ -n "$dir" ]; then
    if [ "$clear" -eq 0 ]; then
      for exe in python3 python; do
        if [ -L "$dir/bin/$exe" ] && [ ! -e "$dir/bin/$exe" ]; then
          echo "Error: [Errno 2] No such file or directory: '$dir/bin/python3'" >&2
          exit 1
        fi
      done
    else
      rm -rf "$dir"
    fi
    mkdir -p "$dir/bin" "$dir/lib/python3.12/site-packages"
    printf 'home = /usr/bin\nversion = 3.12\n' > "$dir/pyvenv.cfg"
    cat "__STUBTEMPLATE__" > "$dir/bin/python3"
    chmod 755 "$dir/bin/python3"
    ln -sf python3 "$dir/bin/python"
    exit 0
  fi
fi
exec "__REAL_PYTHON__" "$@"
"""

# The stub venv interpreter, written by the shim's `-m venv` emulation and
# by _seed_flavored_venv (one source of truth). It:
#   - logs every invocation WITH ITS CWD (the cwd contract tests read this);
#   - answers exit 0 to any `-c` probe (torch imports are always "fine");
#   - `pip install -r *requirements.txt` reads the mode file: "fail"
#     reproduces train-1792134096 (requirements die mid-install: exit 1, no
#     trainer binary), "ok" succeeds and creates a bin/accelerate stub
#     (exit 7 — a "real" trainer launch that fails without faking training
#     success), so a completed bootstrap always reaches the launch stage.
# It deliberately does NOT create the .torch-flavor marker: the marker is
# the SCRIPT's own artifact (echo "${TORCH_INDEX_SLUG}" > ...). A stub that
# materialized it from --index-url calls masked deletion of the writer line
# (pair-review gap F2) — with the emulation removed, a missing writer leaves
# no marker anywhere and the two-run test below goes red.
# Placeholders __LOG__/__MODEFILE__ are substituted per fixture root.
_VENV_STUB_SH = """#!/bin/sh
echo "venv-python cwd=$PWD $*" >> "__LOG__"
mode="$(cat "__MODEFILE__" 2>/dev/null || echo ok)"
has_r=0
has_req=0
for a in "$@"; do
  case "$a" in
    -r) has_r=1 ;;
    *requirements.txt) has_req=1 ;;
  esac
done
bin_dir="$(dirname "$0")"
if [ "$has_r" -eq 1 ] && [ "$has_req" -eq 1 ]; then
  if [ "$mode" = fail ]; then
    echo "pip-requirements FAILED cwd=$PWD" >> "__LOG__"
    exit 1
  fi
  printf '#!/bin/sh\\necho "accelerate $*" >> "%s"\\nexit 7\\n' "__LOG__" > "$bin_dir/accelerate"
  chmod 755 "$bin_dir/accelerate"
  echo "pip-requirements OK cwd=$PWD" >> "__LOG__"
  exit 0
fi
exit 0
"""

# The trainer-binary stub: logs the launch args and exits 7 (a genuine
# failure at the launch stage, never a faked success).
_ACCELERATE_STUB_SH = """#!/bin/sh
echo "accelerate $*" >> "__LOG__"
exit 7
"""


def _shim_paths(root: Path) -> tuple[Path, Path]:
    return root / "shim.log", root / "shim.mode"


def _stub_text(root: Path) -> str:
    log, mode = _shim_paths(root)
    return _VENV_STUB_SH.replace("__LOG__", str(log)).replace("__MODEFILE__", str(mode))


def _accelerate_stub_text(root: Path) -> str:
    log, _mode = _shim_paths(root)
    return _ACCELERATE_STUB_SH.replace("__LOG__", str(log))


def _install_python3_venv_shim(root: Path) -> tuple[Path, Path, Path]:
    """Put the venv-emulating `python3` shim first on PATH; returns
    (shim bin dir, log file, mode file). The mode file content selects the
    requirements-pip behaviour: "ok" (default) or "fail" (requirements die
    mid-install). Flip it between runs to script multi-run scenarios."""
    shim_dir = root / "fixture-bin"
    shim_dir.mkdir()
    log, mode = _shim_paths(root)
    mode.write_text("ok", encoding="utf-8")
    stub_template = shim_dir / "venv-stub.sh"
    stub_template.write_text(_stub_text(root), encoding="utf-8")
    script = (
        _PYTHON3_VENV_SHIM.replace("__LOG__", str(log))
        .replace("__STUBTEMPLATE__", str(stub_template))
        .replace("__REAL_PYTHON__", sys.executable)
    )
    shim = shim_dir / "python3"
    shim.write_text(script, encoding="utf-8")
    shim.chmod(0o755)
    return shim_dir, log, mode


def _seed_flavored_venv(root: Path, *, flavor: str | None = None, with_accelerate: bool = True) -> Path:
    """Pre-build a COMPLETE-looking kohya venv at the location the script
    guards (runtime/kohya-venv): stub interpreter (torch imports fine),
    executable accelerate stub, the site-packages dir a real venv always
    has — and — when flavor is given — a .torch-flavor marker with that
    index slug. The fixture for the flavor-marker guard tests: a matching
    marker must make the bootstrap skip, a missing or stale one must force
    a full rebuild. Returns the venv dir."""
    venv = root / "runtime" / "kohya-venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
    stub = venv / "bin" / "python3"
    stub.write_text(_stub_text(root), encoding="utf-8")
    stub.chmod(0o755)
    (venv / "bin" / "python").symlink_to("python3")
    if with_accelerate:
        acc = venv / "bin" / "accelerate"
        acc.write_text(_accelerate_stub_text(root), encoding="utf-8")
        acc.chmod(0o755)
    if flavor is not None:
        (venv / ".torch-flavor").write_text(f"{flavor}\n", encoding="utf-8")
    return venv


# ---------------------------------------------------------------------------
# 1. text contract of scripts/train_lora.sh
# ---------------------------------------------------------------------------


def test_flux_deploy_slot_is_dash_flux_safetensors():
    # <id>-flux.safetensors with a DASH is the contract of management/main.py
    # _lora_name_for_engine (registry slot "lora_flux", fallback
    # f"{model_id}-flux.safetensors"). Someone "fixing" the dash to an
    # underscore would silently orphan every deployed FLUX LoRA: the underscore
    # form must not appear anywhere in the script (the per-run output dir
    # "<id>_flux_v1" keeps its underscore and is a different string).
    text = _script_text()
    assert "${model_id}-flux.safetensors" in text, "dash slot form missing"
    assert "_flux.safetensors" not in text, "underscore flux slot form must never appear"
    # Cross-check against the manager-side contract literal.
    main_src = (REPO_ROOT / "management" / "main.py").read_text(encoding="utf-8")
    assert 'f"{model_id}-flux.safetensors"' in main_src


def test_flux_launches_accelerate_with_config_and_dataset_files():
    # sd-scripts consumes TWO rendered files: --config_file (script args) and
    # --dataset_config (strict voluptuous schema) — one merged file cannot
    # serve both, so both flags must be on the accelerate launch invocation.
    # The venv bin dir is matched as ${ANY_VAR}/bin/accelerate: the bootstrap
    # self-heal fix may relocate the venv off the host-owned kohya checkout;
    # the contract pinned here is the launch call shape, not its location.
    text = _script_text()
    launch = re.search(
        r'\$\{[A-Za-z_]+\}[^\s"]*/bin/accelerate"\s+launch(?P<body>.*?)\|\|\s+train_rc',
        text,
        re.DOTALL,
    )
    assert launch is not None, "accelerate launch invocation not found"
    body = launch.group("body")
    assert "flux_train_network.py" in body
    assert '--config_file "${RENDERED}"' in body
    assert '--dataset_config "${RENDERED_DATASET}"' in body


def test_flux_resolves_four_local_base_checkpoints():
    # The FLUX trainer loads four LOCAL single-file checkpoints (no HF
    # auto-download): DiT, AE (VAE), CLIP-L, T5-XXL — each via
    # resolve_existing against the download_models.sh runtime layout.
    text = _script_text()
    for var in ("FLUX_DIT", "FLUX_AE", "FLUX_CLIP_L", "FLUX_T5XXL"):
        assert re.search(rf'{var}="\$\(resolve_existing ', text), f"{var} not resolved via resolve_existing"
    for candidate in (
        "checkpoints/flux1-dev-fp8.safetensors",
        "vae/ae.safetensors",
        "text_encoders/clip_l.safetensors",
        "text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors",
    ):
        assert candidate in text, f"candidate path {candidate!r} missing"


def test_flux_guards_missing_kohya_checkout():
    # Fail-fast like the ai-toolkit venv guard: without the /workspace/kohya
    # bind mount the flux branch must exit 2 before rendering anything.
    text = _script_text()
    assert re.search(r'\[\[ ! -f "\$\{KOHYA\}/flux_train_network\.py" \]\]', text), "kohya guard missing"
    guard = text.index('flux_train_network.py" ]]; then')
    assert "exit 2" in text[guard : guard + 400], "kohya guard must exit 2"


def test_flux_numeric_env_validated_before_render():
    # {{STEPS}}/{{SEED}}/{{RESOLUTION}}/{{BATCH}} render as BARE integers, so
    # non-numeric env overrides would emit broken TOML — require_flux_int
    # (^[0-9]+$) must gate all four BEFORE the first render call.
    text = _script_text()
    assert re.search(r'require_flux_int\(\)[\s\S]{0,200}\^\[0-9\]\+\$', text), "numeric regex missing"
    for name in ("STEPS", "SEED", "RESOLUTION", "BATCH"):
        assert f'require_flux_int "{name}"' in text, f"{name} not validated"
    first_render = re.search(r'^\s+render_kohya_toml "\$\{TEMPLATE\}"', text, re.MULTILINE)
    assert first_render is not None, "render_kohya_toml call not found"
    for name in ("STEPS", "SEED", "RESOLUTION", "BATCH"):
        assert text.index(f'require_flux_int "{name}"') < first_render.start(), (
            f"require_flux_int {name} must run before the render"
        )


def test_flux_branch_does_not_read_dead_ai_toolkit_template():
    # lora/train_config.flux.template.yaml (ai-toolkit) is deleted with this
    # spec: the script must reference only the two kohya TOML templates.
    text = _script_text()
    assert "train_config.flux.template.yaml" not in text
    assert "train_config_flux_kohya.template.toml" in text
    assert "train_config_flux_kohya_dataset.template.toml" in text


def test_flux_script_writes_torch_flavor_marker_line():
    # Pair-review gap F2: the flavor guard skips the bootstrap only when
    # ${KOHYA_VENV}/.torch-flavor carries the current slug, and that file is
    # produced by exactly ONE writer line in the script. The test shim used
    # to materialize the marker from any --index-url pip call, so deleting
    # the writer line kept the suite green while prod rebuilt the 10-20
    # minute venv on EVERY flux run. The stub emulation is gone (marker is
    # the script's own artifact now); this textual pin plus
    # test_flux_second_run_skips_bootstrap_via_marker_written_by_script
    # below close the gap from both sides.
    text = _script_text()
    assert 'echo "${TORCH_INDEX_SLUG}" > "${KOHYA_VENV}/.torch-flavor"' in text, (
        "marker-writer line missing/broken: the bootstrap must record the "
        'index slug it installed from (echo "${TORCH_INDEX_SLUG}" > '
        '"${KOHYA_VENV}/.torch-flavor"), otherwise the flavor guard never '
        "finds a matching marker and rebuilds on every run"
    )


def test_flux_bootstrap_installs_compat_shim_text_contract():
    # Prod crash (spec 010, corrected D-006 wording): Comfy-Org
    # flux1-dev-fp8 carries the scaled_fp8 marker EARLY in key order — not
    # first (keys[0] is prefixed) — so upstream's strip loop (library/
    # flux_utils.py load_flow_model) breaks MID-DICT: the prefixed keys
    # after the marker stay unrenamed -> partially meta model ->
    # NotImplementedError at unet.to("cpu"); the ostris scaled-fp8 T5 also
    # needs its per-tensor .scale_weight pairs applied. The flux branch
    # must install scripts/kohya_compat_shim.py into the venv's
    # site-packages and drop an ACTIVATING sitecustomize, with a grep -q
    # gate so the write is idempotent.
    # Validator condition 1 (migration path) moved the install to EVERY-RUN
    # scope: a healthy cu130 venv predating the shim must still get it.
    # That PLACEMENT is pinned behaviorally (and much more precisely) by
    # test_flux_healthy_venv_still_gets_compat_shim_installed — the earlier
    # "inside the rebuild region" ordering pin contradicted the corrected
    # contract and is deliberately gone; here we keep only the mechanism
    # parts no behavioral test can see.
    text = _script_text()
    assert "kohya_compat_shim.py" in text, (
        "the flux branch must copy scripts/kohya_compat_shim.py into the venv"
    )
    assert "site-packages" in text, "the install must target lib/python3.*/site-packages"
    assert "sitecustomize" in text, "bootstrap must write a sitecustomize.py"
    assert re.search(r'grep -q[^\n]*(sitecustomize|kohya_compat_shim)', text), (
        "the sitecustomize write must be guarded by grep -q so a re-run does "
        "not duplicate the shim activation"
    )


def test_flux_templates_te_cache_excludes_caption_shuffle_and_dropout():
    # Cross-template contract with the trainer itself. kohya sd-scripts @
    # 690ea7f9, flux_train_network.py assert_extra_args (lines 70-73):
    #
    #     if args.cache_text_encoder_outputs:
    #         assert (
    #             train_dataset_group.is_text_encoder_output_cacheable()
    #         ), "when caching Text Encoder output, either caption_dropout_rate,
    #             shuffle_caption, token_warmup_step or caption_tag_dropout_rate
    #             cannot be used"
    #
    # (and cache_text_encoder_outputs_to_disk silently enables
    # cache_text_encoder_outputs a few lines above). So whenever the main
    # template turns the text-encoder output cache on, the dataset template
    # must NOT carry any of the four knobs — otherwise every flux training job
    # dies at startup on this assert. Read both templates AS TEXT (unrendered;
    # they are valid TOML as-is, the placeholder-quoting invariant).
    main_cfg = tomllib.loads(
        (REPO_ROOT / "lora" / "train_config_flux_kohya.template.toml").read_text(encoding="utf-8")
    )
    te_cache_on = bool(main_cfg.get("cache_text_encoder_outputs")) or bool(
        main_cfg.get("cache_text_encoder_outputs_to_disk")
    )
    if not te_cache_on:
        return  # the conflict only exists while text-encoder caching is on
    ds_cfg = tomllib.loads(
        (REPO_ROOT / "lora" / "train_config_flux_kohya_dataset.template.toml").read_text(encoding="utf-8")
    )

    def _scan(node):
        # sd-scripts accepts these knobs at [general]/[[datasets]]/
        # [[datasets.subsets]] alike — is_text_encoder_output_cacheable()
        # sees the merged dataset group, so scan every table, not just general.
        if isinstance(node, dict):
            for key, value in node.items():
                if key in FORBIDDEN and value:  # truthy = in effect; 0/False ok
                    yield f"{key} = {value!r}"
                yield from _scan(value)
        elif isinstance(node, list):
            for item in node:
                yield from _scan(item)

    FORBIDDEN = ("shuffle_caption", "caption_dropout_rate", "token_warmup_step", "caption_tag_dropout_rate")
    violations = sorted(set(_scan(ds_cfg)))
    assert not violations, (
        "flux_train_network.py @690ea7f9 aborts the run: 'when caching Text "
        "Encoder output, either caption_dropout_rate, shuffle_caption, "
        "token_warmup_step or caption_tag_dropout_rate cannot be used' — but "
        "the main template enables cache_text_encoder_outputs(_to_disk) while "
        f"the dataset template sets: {', '.join(violations)}"
    )


# ---------------------------------------------------------------------------
# 2. functional SKIP_TRAIN runs against the fixture ROOT
# ---------------------------------------------------------------------------


def test_flux_skip_train_renders_both_tomls_with_bare_numbers():
    root = _build_root()
    try:
        result = _run_script(root, MODEL_ID, "flux", SKIP_TRAIN="1", STEPS=STEPS_SMOKE)
        assert result.returncode == 0, result.stderr
        assert "SKIP_TRAIN=1" in result.stdout, result.stdout
        rendered = root / "runtime" / "train" / f"{MODEL_ID}_flux_kohya.toml"
        rendered_ds = root / "runtime" / "train" / f"{MODEL_ID}_flux_kohya_dataset.toml"
        assert rendered.is_file() and rendered_ds.is_file()
        main_text = rendered.read_text(encoding="utf-8")
        ds_text = rendered_ds.read_text(encoding="utf-8")
        # No placeholder residue in either render.
        assert "{{" not in main_text and "}}" not in main_text
        assert "{{" not in ds_text and "}}" not in ds_text
        # Both renders parse as TOML — and the numerics are BARE ints now.
        main_cfg = tomllib.loads(main_text)
        ds_cfg = tomllib.loads(ds_text)
        assert main_cfg["max_train_steps"] == 20 and isinstance(main_cfg["max_train_steps"], int)
        assert main_cfg["save_every_n_steps"] == 20
        assert main_cfg["seed"] == 1  # default SEED, unquoted
        assert main_cfg["output_name"] == f"{MODEL_ID}_flux_v1"
        assert main_cfg["output_dir"] == str(root / "models" / MODEL_ID / "lora" / "output" / f"{MODEL_ID}_flux_v1")
        assert main_cfg["network_module"] == "networks.lora_flux"
        assert main_cfg["fp8_base"] is True
        assert ds_cfg["datasets"][0]["resolution"] == 768 and isinstance(ds_cfg["datasets"][0]["resolution"], int)
        assert ds_cfg["datasets"][0]["batch_size"] == 1 and isinstance(ds_cfg["datasets"][0]["batch_size"], int)
        assert ds_cfg["datasets"][0]["subsets"][0]["image_dir"] == str(
            root / "models" / MODEL_ID / "dataset" / "images"
        )
        # The four model paths point at the fixture's runtime-layout files.
        for key in ("pretrained_model_name_or_path", "ae", "clip_l", "t5xxl"):
            assert Path(main_cfg[key]).is_file(), (key, main_cfg[key])
    finally:
        _cleanup(root)


def test_flux_model_resolution_falls_back_to_non_fp8_names():
    # resolve_existing candidate order: flux1-dev.safetensors when the fp8
    # checkpoint is absent, t5xxl_fp16.safetensors when the scaled fp8 is not
    # there — the render must carry the fallback paths that exist.
    root = _build_root(dit="flux1-dev.safetensors", t5="t5xxl_fp16.safetensors")
    try:
        result = _run_script(root, MODEL_ID, "flux", SKIP_TRAIN="1", STEPS=STEPS_SMOKE)
        assert result.returncode == 0, result.stderr
        main_cfg = tomllib.loads(
            (root / "runtime" / "train" / f"{MODEL_ID}_flux_kohya.toml").read_text(encoding="utf-8")
        )
        assert main_cfg["pretrained_model_name_or_path"].endswith("flux1-dev.safetensors")
        assert main_cfg["t5xxl"].endswith("t5xxl_fp16.safetensors")
    finally:
        _cleanup(root)


def test_flux_non_numeric_steps_exits_2_without_rendering():
    root = _build_root()
    try:
        result = _run_script(root, MODEL_ID, "flux", SKIP_TRAIN="1", STEPS="abc")
        assert result.returncode == 2, (result.returncode, result.stderr)
        assert "Invalid STEPS" in result.stderr, result.stderr
        assert not (root / "runtime" / "train" / f"{MODEL_ID}_flux_kohya.toml").exists()
        assert not (root / "runtime" / "train" / f"{MODEL_ID}_flux_kohya_dataset.toml").exists()
    finally:
        _cleanup(root)


def test_flux_missing_t5_encoder_exits_2_without_rendering():
    root = _build_root()
    try:
        (root / "runtime" / "models" / "text_encoders" / "t5xxl_fp8_e4m3fn_scaled.safetensors").unlink()
        result = _run_script(root, MODEL_ID, "flux", SKIP_TRAIN="1", STEPS=STEPS_SMOKE)
        assert result.returncode == 2, (result.returncode, result.stderr)
        assert "T5-XXL" in result.stderr, result.stderr
        assert not (root / "runtime" / "train" / f"{MODEL_ID}_flux_kohya.toml").exists()
    finally:
        _cleanup(root)


def test_flux_missing_kohya_checkout_exits_2_without_rendering():
    root = _build_root(with_kohya=False)
    try:
        result = _run_script(root, MODEL_ID, "flux", SKIP_TRAIN="1", STEPS=STEPS_SMOKE)
        assert result.returncode == 2, (result.returncode, result.stderr)
        assert "kohya sd-scripts not found" in result.stderr, result.stderr
        assert not (root / "runtime" / "train" / f"{MODEL_ID}_flux_kohya.toml").exists()
        # Fail fast also means no output dir was prepared for the run.
        assert not (root / "models" / MODEL_ID / "lora" / "output").exists()
    finally:
        _cleanup(root)


def test_sdxl_skip_train_path_unregressed_without_kohya():
    # AC-1: the untouched SDXL branch — `train_lora.sh <id>` (no arch) with
    # SKIP_TRAIN=1 renders runtime/train/<id>.yaml and exits 0 even when the
    # kohya checkout is absent (the flux guards must not leak into sdxl).
    root = _build_root(with_kohya=False)
    try:
        result = _run_script(root, MODEL_ID, SKIP_TRAIN="1")
        assert result.returncode == 0, result.stderr
        rendered = root / "runtime" / "train" / f"{MODEL_ID}.yaml"
        assert rendered.is_file()
        assert MODEL_ID in rendered.read_text(encoding="utf-8")
        assert not (root / "runtime" / "train" / f"{MODEL_ID}_flux_kohya.toml").exists()
    finally:
        _cleanup(root)


def test_flux_bootstrap_self_heals_broken_host_venv():
    # Prod bug (smoke train-1792131743): the kohya clone ships a HOST uv-venv
    # whose bin/python{,3} are dangling symlinks into the host filesystem.
    # `python3 -m venv` then dies with the real CPython repro "Error: [Errno
    # 2] No such file or directory: '<dir>/bin/python3'" and set -e kills the
    # whole flux run BEFORE training starts. The bootstrap must self-heal
    # (rm -rf / venv --clear; venv off the host-owned checkout): the run has
    # to get PAST the venv step and reach the trainer launch. SKIP_TRAIN=1
    # cannot cover this (the seam fires before the bootstrap), so the run
    # goes all the way with a PATH-shimmed python3: the shim emulates
    # `python3 -m venv`, the stub interpreter logs the pip installs and the
    # successful requirements install creates a bin/accelerate stub exiting
    # 7 — the canonical full-bootstrap signature is the launch of THAT stub
    # ("flux_train_network.py exited with code 7"), not the earlier
    # "accelerate binary missing" (rc 127) shape.
    root = _build_root()
    try:
        venv = root / "kohya" / "venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("home = /host/nonexistent/uv-python\n", encoding="utf-8")
        for exe in ("python", "python3"):
            os.symlink("/host/nonexistent/uv-python/bin/python3.12", venv / "bin" / exe)
        (root / "kohya" / "requirements.txt").write_text("accelerate\n-e .\n", encoding="utf-8")
        shim_dir, log, _mode = _install_python3_venv_shim(root)
        result = _run_script(
            root,
            MODEL_ID,
            "flux",
            STEPS=STEPS_SMOKE,
            PATH=f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
        )
        log_text = log.read_text(encoding="utf-8") if log.exists() else ""
        # 1) the venv step must no longer die on the broken host interpreter.
        assert "Errno 2" not in result.stderr, (result.returncode, result.stderr, log_text)
        # 2) the run reached the trainer launch and ran the trainer binary
        #    (the stub exits 7 — a real failure, no faked success).
        assert "flux_train_network.py exited with code 7" in result.stderr, (result.returncode, result.stderr, log_text)
        assert result.returncode == 7
        # 3) the bootstrap really ran through the shimmed python3 -m venv and
        #    then BOTH pip installs on the stub interpreter, in the required
        #    order (cu130 torch wheels first, then requirements.txt).
        assert "-m venv" in log_text, log_text
        pip_lines = [line for line in log_text.splitlines() if "-m pip install" in line]
        assert len(pip_lines) == 2, log_text
        assert "torch" in pip_lines[0] and "cu130" in pip_lines[0], pip_lines
        assert "requirements.txt" in pip_lines[1], pip_lines
    finally:
        _cleanup(root)


def test_flux_pip_requirements_runs_with_cwd_inside_kohya_checkout():
    # Prod bug (smoke train-1792134096): kohya's requirements.txt carries a
    # relative editable install of sd-scripts itself (line 58: "-e ."), and
    # pip resolves it against the CURRENT WORKING DIRECTORY — the job log
    # shows "Obtaining file:///workspace (from -r /workspace/kohya/
    # requirements.txt (line 58))" then "neither 'setup.py' nor
    # 'pyproject.toml' found". The script installs requirements BEFORE its
    # `cd "${KOHYA}"`, so the manager's cwd (/workspace) is used. The
    # requirements pip install must run with CWD = the kohya checkout.
    # The stub interpreter logs the CWD of every call; the red assertion
    # today is the logged cwd of the `-m pip install -r` line.
    root = _build_root()
    try:
        (root / "kohya" / "requirements.txt").write_text("accelerate\n-e .\n", encoding="utf-8")
        # the checkout really is a python project (pip's -e . target)
        (root / "kohya" / "pyproject.toml").write_text('[project]\nname = "sd-scripts"\n', encoding="utf-8")
        shim_dir, log, _mode = _install_python3_venv_shim(root)
        result = _run_script(
            root,
            MODEL_ID,
            "flux",
            STEPS=STEPS_SMOKE,
            PATH=f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
        )
        log_text = log.read_text(encoding="utf-8") if log.exists() else ""
        req_lines = [line for line in log_text.splitlines() if "-m pip install -r" in line]
        assert len(req_lines) == 1, log_text
        assert f"cwd={root / 'kohya'} " in req_lines[0], (
            "pip install -r requirements.txt must run with CWD = the kohya "
            f"checkout so the relative '-e .' resolves; the stub was invoked "
            f"from here instead: {req_lines[0]!r}"
        )
        # the run otherwise completes the bootstrap and launches the trainer
        # stub (exit 7) — the fix must not break the success path.
        assert "flux_train_network.py exited with code 7" in result.stderr, (result.returncode, result.stderr, log_text)
        assert result.returncode == 7
    finally:
        _cleanup(root)


def test_flux_guard_rebootstraps_after_partial_bootstrap():
    # Prod trap found on the same job (F1 gate): the bootstrap guard probes
    # only `import torch` inside the venv. When the cu130 wheels landed but
    # the requirements install died mid-way, the venv is HALF-BUILT (torch
    # importable, no bin/accelerate) — the next run skips the bootstrap and
    # fails on the missing trainer binary instead of healing. The guard must
    # also require ${KOHYA_VENV}/bin/accelerate so a partial bootstrap is
    # rebuilt from scratch.
    # Scenario: run 1 with the shim's requirements-pip in "fail" mode
    # (cu130 OK, requirements FAILED — the train-1792134096 shape); run 2
    # against the SAME venv dir with requirements now succeeding must re-run
    # the whole bootstrap (fresh venv + both pip installs) and launch the
    # trainer stub. Today run 2 skips the bootstrap (torch-only guard) and
    # dies on the missing accelerate — no venv/pip lines in its log and the
    # launch marker shows rc 127, not the trainer stub's 7.
    root = _build_root()
    try:
        (root / "kohya" / "requirements.txt").write_text("accelerate\n-e .\n", encoding="utf-8")
        (root / "kohya" / "pyproject.toml").write_text('[project]\nname = "sd-scripts"\n', encoding="utf-8")
        shim_dir, log, mode = _install_python3_venv_shim(root)
        path_env = f"{shim_dir}{os.pathsep}{os.environ['PATH']}"

        # (a) run 1: requirements die mid-install → no launch at all.
        mode.write_text("fail", encoding="utf-8")
        r1 = _run_script(root, MODEL_ID, "flux", STEPS=STEPS_SMOKE, PATH=path_env)
        log1 = log.read_text(encoding="utf-8") if log.exists() else ""
        assert r1.returncode != 0, (r1.returncode, r1.stderr)
        assert any("cu130" in line for line in log1.splitlines()), log1
        assert "pip-requirements FAILED" in log1, log1
        assert "flux_train_network.py exited with code" not in r1.stderr, r1.stderr
        # locate the venv the script created (location-agnostic: it is the
        # dir argument of the logged `python3 -m venv` call)
        venv_lines = [line for line in log1.splitlines() if line.startswith("python3 -m venv")]
        assert venv_lines, log1
        venv_dir = Path(venv_lines[-1].rsplit(None, 1)[-1])
        assert (venv_dir / "bin" / "python3").is_file(), venv_dir
        assert not (venv_dir / "bin" / "accelerate").exists(), "half-built venv must lack the trainer binary"

        # (b) run 2, same venv dir, requirements now succeed: the guard must
        #     notice the half-built venv and re-run the whole bootstrap.
        mode.write_text("ok", encoding="utf-8")
        log.unlink()
        r2 = _run_script(root, MODEL_ID, "flux", STEPS=STEPS_SMOKE, PATH=path_env)
        log2 = log.read_text(encoding="utf-8") if log.exists() else ""
        assert "-m venv" in log2, (
            "run 2 skipped the bootstrap on a half-built venv (torch-only "
            f"guard): no fresh venv creation in the log.\nlog2:\n{log2}\nstderr:\n{r2.stderr}"
        )
        pip2 = [line for line in log2.splitlines() if "-m pip install" in line]
        assert len(pip2) == 2, log2
        assert "flux_train_network.py exited with code 7" in r2.stderr, (r2.returncode, r2.stderr, log2)
        assert r2.returncode == 7
    finally:
        _cleanup(root)


def test_flux_guard_rebuilds_on_missing_or_stale_torch_flavor():
    # Torch-flavor rebake (cu124 → cu130): the bootstrap records the index
    # slug it installed from in ${KOHYA_VENV}/.torch-flavor, and the guard
    # skips the bootstrap only when the venv python imports torch+torchvision
    # AND the marker says "cu130" AND bin/accelerate is executable. A venv
    # that looks complete but carries NO marker or a STALE one ("cu124" —
    # built against the old index) is the wrong torch build for the new CUDA
    # image and must be rebuilt from scratch, with the marker rewritten to
    # cu130. RED today: the committed guard (e23dfb9) has no flavor notion —
    # torch+accelerate make it skip the bootstrap, so no rebuild happens and
    # the marker stays missing/stale; the pip install itself still says
    # cu124, so the fresh bootstrap of the first variant also fails the
    # cu130 contract.
    for stale in (None, "cu124"):
        root = _build_root()
        try:
            (root / "kohya" / "requirements.txt").write_text("accelerate\n-e .\n", encoding="utf-8")
            (root / "kohya" / "pyproject.toml").write_text('[project]\nname = "sd-scripts"\n', encoding="utf-8")
            venv = _seed_flavored_venv(root, flavor=stale)
            shim_dir, log, _mode = _install_python3_venv_shim(root)
            result = _run_script(
                root,
                MODEL_ID,
                "flux",
                STEPS=STEPS_SMOKE,
                PATH=f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
            )
            log_text = log.read_text(encoding="utf-8") if log.exists() else ""
            label = f"flavor={stale!r}: "
            # the half-recognized venv must be rebuilt end-to-end ...
            assert "-m venv" in log_text, (
                f"{label}guard skipped the bootstrap on a venv without a "
                f"matching cu130 marker.\nlog:\n{log_text}\nstderr:\n{result.stderr}"
            )
            pip_lines = [line for line in log_text.splitlines() if "-m pip install" in line]
            assert len(pip_lines) == 2, f"{label}{log_text}"
            assert "cu130" in pip_lines[0], f"{label}{pip_lines}"
            # ... and the marker must come out saying cu130 ...
            marker = venv / ".torch-flavor"
            assert marker.is_file(), f"{label}.torch-flavor marker not written by the rebuild"
            assert "cu130" in marker.read_text(encoding="utf-8"), f"{label}{marker.read_text(encoding='utf-8')!r}"
            # ... and the run still ends at the trainer launch (stub exit 7).
            assert "flux_train_network.py exited with code 7" in result.stderr, (
                f"{label}{result.returncode}\n{result.stderr}\n{log_text}"
            )
            assert result.returncode == 7
        finally:
            _cleanup(root)


def test_flux_guard_skips_bootstrap_on_matching_torch_flavor():
    # The other side of the flavor gate: a venv with torch importable,
    # executable accelerate AND .torch-flavor == "cu130" is fully current —
    # the bootstrap must be SKIPPED entirely (no python3 -m venv, no pip
    # torch install) and the launch must run the existing trainer binary.
    # Green today (the committed torch+accelerate guard also skips) and must
    # stay green after the flavor fix — the regression guard that the marker
    # check does not force needless 10-20 minute rebuilds on every run.
    root = _build_root()
    try:
        (root / "kohya" / "requirements.txt").write_text("accelerate\n-e .\n", encoding="utf-8")
        venv = _seed_flavored_venv(root, flavor="cu130")
        shim_dir, log, _mode = _install_python3_venv_shim(root)
        result = _run_script(
            root,
            MODEL_ID,
            "flux",
            STEPS=STEPS_SMOKE,
            PATH=f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
        )
        log_text = log.read_text(encoding="utf-8") if log.exists() else ""
        assert "-m venv" not in log_text, log_text
        assert not any("-m pip install" in line for line in log_text.splitlines()), log_text
        # the seeded marker and trainer binary survive untouched ...
        assert (venv / ".torch-flavor").read_text(encoding="utf-8").strip() == "cu130"
        # ... and the run goes straight to the launch of the seeded stub.
        assert "flux_train_network.py exited with code 7" in result.stderr, (result.returncode, result.stderr, log_text)
        assert result.returncode == 7
    finally:
        _cleanup(root)


def test_flux_second_run_skips_bootstrap_via_marker_written_by_script():
    # F2 behavioral half: .torch-flavor must be the SCRIPT's own artifact —
    # the shim stub deliberately never creates it. Two runs on ONE tree:
    #   run 1: venv location empty → full bootstrap (rm+venv+both pip) and
    #          the script's writer line leaves a cu130 marker;
    #   run 2: same tree, marker current → the guard skips the bootstrap
    #          ENTIRELY: no `python3 -m venv`, no pip calls, straight to the
    #          trainer launch (accelerate stub, exit 7).
    # Delete the writer line and run 2 finds no matching marker, rebuilds a
    # 10-20 minute venv on every run, and this test goes red — while every
    # single-run test (which the old stub-marker masked) would have stayed
    # green.
    root = _build_root()
    try:
        (root / "kohya" / "requirements.txt").write_text("accelerate\n-e .\n", encoding="utf-8")
        (root / "kohya" / "pyproject.toml").write_text('[project]\nname = "sd-scripts"\n', encoding="utf-8")
        shim_dir, log, _mode = _install_python3_venv_shim(root)
        path_env = f"{shim_dir}{os.pathsep}{os.environ['PATH']}"

        # run 1: full bootstrap on the empty venv location.
        r1 = _run_script(root, MODEL_ID, "flux", STEPS=STEPS_SMOKE, PATH=path_env)
        log1 = log.read_text(encoding="utf-8") if log.exists() else ""
        assert "-m venv" in log1, log1
        assert len([line for line in log1.splitlines() if "-m pip install" in line]) == 2, log1
        assert "flux_train_network.py exited with code 7" in r1.stderr, (r1.returncode, r1.stderr, log1)
        # the marker exists ONLY because the script wrote it (the stub does
        # not create it) and carries the slug the script installed from.
        marker = root / "runtime" / "kohya-venv" / ".torch-flavor"
        assert marker.is_file(), (
            "run 1 left no .torch-flavor marker: the script's marker-writer "
            f"line is missing/broken.\nlog1:\n{log1}\nstderr:\n{r1.stderr}"
        )
        assert "cu130" in marker.read_text(encoding="utf-8"), marker.read_text(encoding="utf-8")

        # run 2: SAME tree — the guard must skip the whole bootstrap.
        log.unlink()
        r2 = _run_script(root, MODEL_ID, "flux", STEPS=STEPS_SMOKE, PATH=path_env)
        log2 = log.read_text(encoding="utf-8") if log.exists() else ""
        assert "-m venv" not in log2, (
            "run 2 rebuilt the venv on a marker-current tree — without the "
            "script writing .torch-flavor during run 1 the guard can never "
            f"match the flavor and every run pays 10-20 minutes.\nlog2:\n{log2}\nstderr:\n{r2.stderr}"
        )
        assert not any("-m pip install" in line for line in log2.splitlines()), log2
        assert "flux_train_network.py exited with code 7" in r2.stderr, (r2.returncode, r2.stderr, log2)
        assert r2.returncode == 7
    finally:
        _cleanup(root)


def _site_packages(venv_root: Path) -> Path:
    """The single site-packages dir of a stub venv (glob python3.*)."""
    matches = sorted((venv_root / "lib").glob("python3.*/site-packages"))
    assert matches, f"no site-packages under {venv_root}/lib"
    return matches[0]


def _activation_probe(sp: Path) -> int:
    """F1 activation probe: run a fresh interpreter with the stub
    site-packages on PYTHONPATH — CPython auto-imports sitecustomize at
    startup, exactly like every venv python process — and count the
    library.flux_utils finders the shim registered. Isolated subprocess:
    the pytest process' sys.meta_path is never touched."""
    probe = (
        "import sys\n"
        "hits = [f for f in sys.meta_path\n"
        "        if getattr(type(f), '__module__', '') == 'kohya_compat_shim'\n"
        "        or getattr(f, 'target', None) == 'library.flux_utils']\n"
        "print(len(hits))\n"
    )
    env = {k: os.environ[k] for k in ("PATH", "HOME") if k in os.environ}
    env["PYTHONPATH"] = str(sp)
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, f"sitecustomize execution crashed: {proc.stderr}"
    assert proc.stdout.strip().isdigit(), f"probe printed {proc.stdout!r} / {proc.stderr!r}"
    return int(proc.stdout.strip())


def test_flux_bootstrap_installs_compat_shim_into_site_packages():
    # Behavioral half of the compat-shim install: after a full bootstrap the
    # venv's site-packages holds kohya_compat_shim.py — copied verbatim from
    # the repo file — plus a sitecustomize.py that ACTIVATES it. F1 (review
    # block): a sitecustomize that merely imports the module never calls
    # install(), sys.meta_path stays clean and the prod contract is dead.
    # Activation is pinned on the OBSERVABLE prod path — a python process
    # whose startup auto-imports the written sitecustomize must end up with
    # EXACTLY ONE library.flux_utils finder — regardless of which layer
    # (module-level install() in the shim, or an install() call line in the
    # sitecustomize) the fix chooses. Uses the REAL shim content in the
    # fixture scripts/ dir so the executed pair is the real production one.
    root = _build_root()
    try:
        real_shim = (REPO_ROOT / "scripts" / "kohya_compat_shim.py").read_text(encoding="utf-8")
        (root / "scripts" / "kohya_compat_shim.py").write_text(real_shim, encoding="utf-8")
        (root / "kohya" / "requirements.txt").write_text("accelerate\n-e .\n", encoding="utf-8")
        (root / "kohya" / "pyproject.toml").write_text('[project]\nname = "sd-scripts"\n', encoding="utf-8")
        shim_dir, log, _mode = _install_python3_venv_shim(root)
        result = _run_script(
            root,
            MODEL_ID,
            "flux",
            STEPS=STEPS_SMOKE,
            PATH=f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
        )
        log_text = log.read_text(encoding="utf-8") if log.exists() else ""
        # the bootstrap must complete (launch stub reached) even with the
        # install lines in place — a failing cp must not be swallowed.
        assert "flux_train_network.py exited with code 7" in result.stderr, (result.returncode, result.stderr, log_text)
        sp = _site_packages(root / "runtime" / "kohya-venv")
        installed = sp / "kohya_compat_shim.py"
        assert installed.is_file(), f"bootstrap did not install the compat shim into {sp}"
        assert installed.read_text(encoding="utf-8") == real_shim, (
            "the installed shim must be a verbatim copy of scripts/kohya_compat_shim.py"
        )
        sitecustomize = sp / "sitecustomize.py"
        assert sitecustomize.is_file(), f"no sitecustomize.py in {sp}"
        assert "import kohya_compat_shim" in sitecustomize.read_text(encoding="utf-8")

        found = _activation_probe(sp)
        assert found == 1, (
            f"executing the bootstrap-written sitecustomize must activate the shim "
            f"(exactly one library.flux_utils finder); got {found}. sitecustomize: "
            f"{sitecustomize.read_text(encoding='utf-8')!r}"
        )
    finally:
        _cleanup(root)


def test_flux_healthy_venv_still_gets_compat_shim_installed():
    # Validator condition 1 (migration path): W3f put the shim install and
    # the sitecustomize write INSIDE the rebuild branch. The exact prod
    # state after W3d is a HEALTHY cu130 venv built before the shim existed
    # — torch imports, accelerate is executable, the marker says cu130 — so
    # the flavor guard passes, the branch is skipped and the fix NEVER
    # LANDS ON THE BOX. Corrected contract: the shim install must run on
    # EVERY flux run (fresh AND healthy venv), the written sitecustomize
    # must carry BOTH the import and the install() activation line, and the
    # activation probe must find exactly one finder — while the healthy
    # venv is still NOT rebuilt (the guard's skip path must stay intact).
    # RED today: the healthy path skips the whole install block.
    root = _build_root()
    try:
        real_shim = (REPO_ROOT / "scripts" / "kohya_compat_shim.py").read_text(encoding="utf-8")
        (root / "scripts" / "kohya_compat_shim.py").write_text(real_shim, encoding="utf-8")
        (root / "kohya" / "requirements.txt").write_text("accelerate\n-e .\n", encoding="utf-8")
        (root / "kohya" / "pyproject.toml").write_text('[project]\nname = "sd-scripts"\n', encoding="utf-8")
        venv = _seed_flavored_venv(root, flavor="cu130")  # healthy: stub torch + accelerate + marker
        sp = _site_packages(venv)
        assert not (sp / "kohya_compat_shim.py").exists(), "fixture must start shim-free (the W3d prod state)"
        shim_dir, log, _mode = _install_python3_venv_shim(root)
        result = _run_script(
            root,
            MODEL_ID,
            "flux",
            STEPS=STEPS_SMOKE,
            PATH=f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
        )
        log_text = log.read_text(encoding="utf-8") if log.exists() else ""
        # migration must not weaken the guard: the healthy venv stays un-rebuilt
        assert "-m venv" not in log_text, log_text
        assert not any("-m pip install" in line for line in log_text.splitlines()), log_text
        assert "flux_train_network.py exited with code 7" in result.stderr, (result.returncode, result.stderr, log_text)
        # ... and STILL installs + activates the shim
        installed = sp / "kohya_compat_shim.py"
        assert installed.is_file(), (
            "the compat shim must be installed on EVERY flux run — a healthy "
            "cu130 venv predating the shim (the prod state after W3d) gets no "
            f"fix at all.\nlog:\n{log_text}\nstderr:\n{result.stderr}"
        )
        assert installed.read_text(encoding="utf-8") == real_shim, (
            "the installed shim must be a verbatim copy of scripts/kohya_compat_shim.py"
        )
        sitecustomize = sp / "sitecustomize.py"
        assert sitecustomize.is_file(), f"no sitecustomize.py in {sp} on the healthy path"
        content = sitecustomize.read_text(encoding="utf-8")
        assert "import kohya_compat_shim" in content, content
        assert "kohya_compat_shim.install()" in content, content
        found = _activation_probe(sp)
        assert found == 1, (
            f"executing the sitecustomize must activate the shim (exactly one "
            f"library.flux_utils finder); got {found}. sitecustomize: {content!r}"
        )
    finally:
        _cleanup(root)


def test_flux_healthy_venv_shim_copy_refreshes_stale_file():
    # Validator condition 1, second half: the install must REFRESH the copy
    # (cp overwrites, no exists-guard on the shim file itself) — a STALE
    # shim already sitting in a healthy venv's site-packages is replaced by
    # the current repo file on every run, so future shim updates reach
    # installed venvs without paying a 10-20 minute rebuild. The
    # sitecustomize here is pre-seeded already-activated, so the run's only
    # observable obligation is the refresh. RED today: the healthy path
    # never copies.
    root = _build_root()
    try:
        real_shim = (REPO_ROOT / "scripts" / "kohya_compat_shim.py").read_text(encoding="utf-8")
        (root / "scripts" / "kohya_compat_shim.py").write_text(real_shim, encoding="utf-8")
        (root / "kohya" / "requirements.txt").write_text("accelerate\n-e .\n", encoding="utf-8")
        (root / "kohya" / "pyproject.toml").write_text('[project]\nname = "sd-scripts"\n', encoding="utf-8")
        venv = _seed_flavored_venv(root, flavor="cu130")
        sp = _site_packages(venv)
        stale = sp / "kohya_compat_shim.py"
        stale.write_text("# STALE SHIM FROM AN OLDER INSTALL\n", encoding="utf-8")
        (sp / "sitecustomize.py").write_text(
            "import kohya_compat_shim\nkohya_compat_shim.install()\n", encoding="utf-8"
        )
        shim_dir, log, _mode = _install_python3_venv_shim(root)
        result = _run_script(
            root,
            MODEL_ID,
            "flux",
            STEPS=STEPS_SMOKE,
            PATH=f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
        )
        log_text = log.read_text(encoding="utf-8") if log.exists() else ""
        assert "-m venv" not in log_text, log_text  # healthy venv: no rebuild
        assert "flux_train_network.py exited with code 7" in result.stderr, (result.returncode, result.stderr, log_text)
        assert stale.read_text(encoding="utf-8") == real_shim, (
            "the shim copy must be REFRESHED on every flux run — the stale "
            "file survived, so shim updates could never reach an installed "
            "healthy venv"
        )
        # refresh must not duplicate the activation either
        lines = [ln for ln in (sp / "sitecustomize.py").read_text(encoding="utf-8").splitlines() if "import kohya_compat_shim" in ln]
        assert len(lines) == 1, lines
    finally:
        _cleanup(root)


def test_flux_sitecustomize_shim_import_not_duplicated_across_rebuilds():
    # The grep -q guard must keep the sitecustomize import idempotent: one
    # bootstrap leaves exactly ONE `import kohya_compat_shim` line; forcing a
    # rebuild (delete the flavor marker → guard re-arms) must still leave
    # one, not a stack of duplicated imports.
    root = _build_root()
    try:
        (root / "kohya" / "requirements.txt").write_text("accelerate\n-e .\n", encoding="utf-8")
        (root / "kohya" / "pyproject.toml").write_text('[project]\nname = "sd-scripts"\n', encoding="utf-8")
        shim_dir, log, _mode = _install_python3_venv_shim(root)
        path_env = f"{shim_dir}{os.pathsep}{os.environ['PATH']}"

        r1 = _run_script(root, MODEL_ID, "flux", STEPS=STEPS_SMOKE, PATH=path_env)
        assert r1.returncode == 7, (r1.returncode, r1.stderr)
        sp = _site_packages(root / "runtime" / "kohya-venv")
        sitecustomize = sp / "sitecustomize.py"
        assert sitecustomize.is_file(), sp
        imports1 = [ln for ln in sitecustomize.read_text(encoding="utf-8").splitlines() if "import kohya_compat_shim" in ln]
        assert len(imports1) == 1, imports1

        # force a rebuild: the marker disappears → the guard re-arms.
        (root / "runtime" / "kohya-venv" / ".torch-flavor").unlink()
        log.unlink()
        r2 = _run_script(root, MODEL_ID, "flux", STEPS=STEPS_SMOKE, PATH=path_env)
        assert r2.returncode == 7, (r2.returncode, r2.stderr)
        sp = _site_packages(root / "runtime" / "kohya-venv")
        sitecustomize = sp / "sitecustomize.py"
        imports2 = [ln for ln in sitecustomize.read_text(encoding="utf-8").splitlines() if "import kohya_compat_shim" in ln]
        assert len(imports2) == 1, f"rebuild duplicated the sitecustomize import: {imports2}"
        assert (sp / "kohya_compat_shim.py").is_file()
    finally:
        _cleanup(root)


# ---------------------------------------------------------------------------
# 3. deploy_flux_lora (function extracted from the script, bash-driven)
# ---------------------------------------------------------------------------


def test_deploy_flux_lora_empty_output_dir_returns_3():
    root = _build_root()
    try:
        out_dir = root / "models" / MODEL_ID / "lora" / "output" / f"{MODEL_ID}_flux_v1"
        out_dir.mkdir(parents=True)  # training "finished" but produced nothing
        result = _run_deploy(root)
        assert result.returncode == 3, (result.returncode, result.stderr)
        assert "no flux *.safetensors checkpoint found" in result.stderr, result.stderr
        assert not (root / "runtime" / "models" / "loras" / f"{MODEL_ID}-flux.safetensors").exists()
    finally:
        _cleanup(root)


def test_deploy_flux_lora_happy_path_dash_slot_0644_newest_wins():
    root = _build_root()
    try:
        out_dir = root / "models" / MODEL_ID / "lora" / "output" / f"{MODEL_ID}_flux_v1"
        out_dir.mkdir(parents=True)
        older = out_dir / "run_a.safetensors"
        newer = out_dir / "run_b.safetensors"
        older.write_bytes(b"OLD")
        newer.write_bytes(b"NEW")
        os.utime(older, (1_700_000_000, 1_700_000_000))
        os.utime(newer, (1_800_000_000, 1_800_000_000))
        result = _run_deploy(root)
        assert result.returncode == 0, (result.returncode, result.stderr)
        assert "LoRA ready (flux)" in result.stdout, result.stdout
        dest = root / "runtime" / "models" / "loras" / f"{MODEL_ID}-flux.safetensors"
        assert dest.is_file(), dest
        assert dest.read_bytes() == b"NEW", "the newest checkpoint must win"
        # mktemp starts at 0600; the script must chmod back to 0644 so the
        # shared runtime mount stays readable.
        assert (dest.stat().st_mode & 0o777) == 0o644, oct(dest.stat().st_mode & 0o777)
        # tmp+mv leaves no scratch file behind.
        leftovers = [p.name for p in dest.parent.iterdir() if ".tmp." in p.name]
        assert leftovers == []
    finally:
        _cleanup(root)


# ---------------------------------------------------------------------------
# 4. compose.yaml bind mount (manager must see the kohya checkout)
# ---------------------------------------------------------------------------


def test_compose_manager_binds_kohya_checkout():
    # The script resolves KOHYA="${ROOT}/kohya" = /workspace/kohya in the
    # manager container; without this bind every flux run exits 2. Fix #16
    # style check: the exact bind string in manager.volumes, and the
    # ai-toolkit bind it was added next to stays (SDXL trainer).
    import yaml

    doc = yaml.safe_load((REPO_ROOT / "compose.yaml").read_text(encoding="utf-8"))
    volumes = doc["services"]["manager"]["volumes"]
    assert "/opt/sea-speed-worker/valery/kohya:/workspace/kohya" in volumes, volumes
    assert "/opt/sea-speed-worker/valery/ai-toolkit:/workspace/ai-toolkit" in volumes, volumes


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if not (name.startswith("test_") and callable(function)):
            continue
        try:
            function()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}: {exc!r}")
        else:
            print(f"PASS {name}")
    raise SystemExit(1 if failures else 0)
