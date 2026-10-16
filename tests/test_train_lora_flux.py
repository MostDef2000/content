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


# Emulates `python3 -m venv` for the bootstrap test: a dangling-interpreter
# venv dir fails exactly like the real CPython repro (Errno 2 on
# bin/python3) unless --clear is passed; otherwise an executable stub
# interpreter is created that logs every invocation (guard -c checks and
# both pip installs) to the shim log. Anything not `-m venv` passes through
# to the real interpreter. Placeholders __LOG__/__REAL_PYTHON__ are
# substituted when the shim is written.
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
    mkdir -p "$dir/bin"
    printf '#!/bin/sh\\necho "venv-python $*" >> "%s"\\nexit 0\\n' "$log" > "$dir/bin/python3"
    chmod 755 "$dir/bin/python3"
    ln -sf python3 "$dir/bin/python"
    exit 0
  fi
fi
exec "__REAL_PYTHON__" "$@"
"""


def _install_python3_venv_shim(root: Path) -> tuple[Path, Path]:
    """Put the venv-emulating `python3` shim first on PATH; returns
    (shim bin dir, log file)."""
    shim_dir = root / "fixture-bin"
    shim_dir.mkdir()
    log = root / "shim.log"
    script = _PYTHON3_VENV_SHIM.replace("__LOG__", str(log)).replace("__REAL_PYTHON__", sys.executable)
    shim = shim_dir / "python3"
    shim.write_text(script, encoding="utf-8")
    shim.chmod(0o755)
    return shim_dir, log


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
    # `python3 -m venv ${KOHYA}/venv` then dies with the real CPython repro
    # "Error: [Errno 2] No such file or directory: '<dir>/bin/python3'" and
    # set -e kills the whole flux run BEFORE training starts. The bootstrap
    # must self-heal (rm -rf / venv --clear; venv off the host-owned
    # checkout): the run has to get PAST the venv step and reach the trainer
    # launch. SKIP_TRAIN=1 cannot cover this (the seam fires before the
    # bootstrap), so the run goes all the way with a PATH-shimmed python3:
    # the shim emulates `python3 -m venv` and the stub interpreter logs the
    # pip installs; the run is then expected to fail ONLY at the missing
    # accelerate binary inside the stub venv (no network, no GPU touched).
    root = _build_root()
    try:
        venv = root / "kohya" / "venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("home = /host/nonexistent/uv-python\n", encoding="utf-8")
        for exe in ("python", "python3"):
            os.symlink("/host/nonexistent/uv-python/bin/python3.12", venv / "bin" / exe)
        (root / "kohya" / "requirements.txt").write_text("torch\n", encoding="utf-8")
        shim_dir, log = _install_python3_venv_shim(root)
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
        # 2) the run must reach the trainer launch: the stub venv ships no
        #    accelerate binary, so the script's own launch-failure marker is
        #    the proof of arrival (and rc != 0 — the run must not fake success).
        assert "flux_train_network.py exited with code" in result.stderr, (result.returncode, result.stderr, log_text)
        assert result.returncode != 0
        # 3) the bootstrap really ran through the shimmed python3 -m venv and
        #    then BOTH pip installs on the stub interpreter, in the required
        #    order (cu124 torch wheels first, then requirements.txt).
        assert "-m venv" in log_text, log_text
        pip_lines = [line for line in log_text.splitlines() if "-m pip install" in line]
        assert len(pip_lines) == 2, log_text
        assert "torch" in pip_lines[0] and "cu124" in pip_lines[0], pip_lines
        assert "requirements.txt" in pip_lines[1], pip_lines
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
