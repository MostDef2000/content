"""Tests for scripts/kohya_compat_shim.py — the sd-scripts flux-loader
compat shim the kohya venv bootstrap installs (spec 010 prod crash).

Root cause (verified from prod logs + upstream kohya-ss/sd-scripts @
690ea7f9, corrected per spec D-006): library/flux_utils.py load_flow_model
strips the ComfyUI `model.diffusion_model.` prefix with a loop that BREAKS
on the first key without the prefix. In Comfy-Org's
flux1-dev-fp8.safetensors the unprefixed scaled-fp8 marker `scaled_fp8`
sits EARLY in key order — NOT first (keys[0] is prefixed; a marker-first
file would have died in analyze_checkpoint_state on max() before any
load). The loop therefore breaks MID-DICT: prefixed keys before the
marker do get renamed, the REMAINING `model.diffusion_model.*` keys stay
unrenamed and unexpected, the model built under torch.device("meta")
receives only part of its weights (PARTIALLY meta) and training dies with
"NotImplementedError: Cannot copy out of meta tensor" at
unet.to("cpu") (flux_train_network.py:241). The ostris scaled-fp8 T5
(t5xxl_fp8_e4m3fn_scaled.safetensors) additionally carries 168 per-tensor
`*.scale_weight` keys kohya ignores, assigning raw fp8 values as weights —
numerically wrong text-encoder outputs. CLIP-L and AE are fine.

Two test layers:
  * STRUCTURAL (unconditional): AST/text pins of the shim source — the
    required function set, the meta_path finder class, the NO-BREAK strip
    loop (the exact upstream bug), the apply_sd_fixups composition and the
    patch_flux_utils wrap/restore shape. These run everywhere, including CI
    (torch is NOT installed there — verified).
  * BEHAVIORAL (skipif torch missing): real fp8 math — scalar and
    per-channel `.scale_weight` dequant to bf16, marker/scale-key drops,
    plain-fp8 cast, non-fp8 pass-through, the composed ComfyUI-style
    pipeline, the patch_flux_utils fake-module wrap/restore for BOTH
    loaders (flow AND t5xxl — reviewer F2: upstream loads the T5 after
    load_flow_model returned) and the idempotent meta_path install. They
    assert SHIM_PATH exists FIRST, so they fail (not skip) while the shim
    file is missing. The strip test is pure dict-key logic (torch is a
    lazy import in the shim) and runs UNGATED in CI (reviewer F4).
    Memory-fix pins (train-1792246674 host OOM): strip supports an
    UNGATED in_place rename (same dict object), dequant an in_place
    CONSUMING mode (gated; bit-exact vs copy mode), the default modes
    keep their copy/non-consuming semantics, and the wrapper must pass
    in_place through (structural, CI-visible).
    RAM-policy pins (train-1792249442, <~26GB usable — even the chunked
    bf16 result does not fit): dequant takes a target parameter, the
    DEFAULT target=None re-quantizes paired tensors back into their
    input fp8 dtype, target=torch.bfloat16 is the explicit old contract,
    and the wrapper must use fp8-preserve for load_flow_model vs
    explicit bf16 for load_t5xxl. The composed/consumes/flow-wrapper
    dtype expectations follow the preserve-fp8 default (the target
    implementation has since landed in the working tree — these pins now
    VERIFY it; they would have failed against the old bf16-forcing code);
    the bit-exact pin stays target-agnostic.

The sitecustomize ACTIVATION defect (reviewer F1 — import alone never
calls install()) is pinned behaviorally on the prod path in
tests/test_train_lora_flux.py::test_flux_bootstrap_installs_compat_shim_into_site_packages.

pytest-compatible; pytest is not required.

Run either way:
    python -m pytest tests/test_kohya_compat_shim.py -q
    python tests/test_kohya_compat_shim.py
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # content/
SHIM_PATH = REPO_ROOT / "scripts" / "kohya_compat_shim.py"

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

try:
    import pytest
except ModuleNotFoundError:  # direct runner: pytest is optional
    pytest = None


class _Skipped(Exception):
    """Direct-runner skip signal (pytest.skip is used when pytest exists)."""


def _require_torch() -> None:
    if TORCH_AVAILABLE:
        return
    reason = "torch not installed in this sandbox/CI; fp8 dtypes (torch.float8_e4m3fn) require torch>=2.1"
    if pytest is not None:
        pytest.skip(reason)
    raise _Skipped(reason)


def _read_shim_source() -> str:
    assert SHIM_PATH.is_file(), f"{SHIM_PATH} missing — the compat shim is not implemented yet"
    return SHIM_PATH.read_text(encoding="utf-8")


def _load_shim():
    spec = importlib.util.spec_from_file_location("kohya_compat_shim_under_test", SHIM_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _top_level_func(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


# ---------------------------------------------------------------------------
# STRUCTURAL pins (unconditional — carry the contract in CI without torch)
# ---------------------------------------------------------------------------


def test_shim_source_defines_contract_functions_and_finder():
    tree = ast.parse(_read_shim_source())
    funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for required in (
        "strip_comfyui_prefix",
        "dequant_scaled_fp8_sd",
        "apply_sd_fixups",
        "patch_flux_utils",
        "install",
    ):
        assert required in funcs, f"shim must define top-level {required}(); found {sorted(funcs)}"
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    assert any(
        any(isinstance(m, ast.FunctionDef) and m.name == "find_spec" for m in cls.body) for cls in classes
    ), "shim must define a sys.meta_path finder class exposing find_spec"


def test_strip_comfyui_prefix_loop_has_no_early_break():
    # The exact upstream bug: `if new_key == key: break` — the EARLY (not
    # first) scaled_fp8 marker aborts the rename MID-DICT, so the prefixed
    # keys after it stay unrenamed (partially meta model). The shim's strip
    # must iterate every key; no break may hide inside any loop of the
    # function.
    fn = _top_level_func(ast.parse(_read_shim_source()), "strip_comfyui_prefix")
    assert fn is not None, "strip_comfyui_prefix missing"
    for node in ast.walk(fn):
        if isinstance(node, (ast.For, ast.While)):
            for inner in ast.walk(node):
                assert not isinstance(inner, ast.Break), (
                    "strip_comfyui_prefix must not break early — "
                    "break-on-first-unprefixed-key is the prod crash itself"
                )


def test_apply_sd_fixups_composes_dequant_and_strip():
    fn = _top_level_func(ast.parse(_read_shim_source()), "apply_sd_fixups")
    assert fn is not None, "apply_sd_fixups missing"
    called = {n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert {"strip_comfyui_prefix", "dequant_scaled_fp8_sd"} <= called, (
        f"apply_sd_fixups must compose both fixups, calls seen: {sorted(called)}"
    )


def test_patch_flux_utils_shape_wrap_load_safetensors_restore_in_finally():
    # Definitive contract: patch_flux_utils(mod) wraps BOTH loaders —
    # mod.load_flow_model AND mod.load_t5xxl (F2: upstream loads the T5 via
    # load_t5xxl AFTER load_flow_model returned, so wrapping only the flow
    # path leaves the scaled-fp8 T5 sd unfixed) — each wrapper temporarily
    # replaces mod.load_safetensors with one that applies apply_sd_fixups to
    # its return value and RESTORES the original in a finally block (the
    # inner loaders are NOT re-implemented).
    fn = _top_level_func(ast.parse(_read_shim_source()), "patch_flux_utils")
    assert fn is not None, "patch_flux_utils missing"
    # bare names AND attribute names (mod.load_flow_model is an Attribute,
    # not a Name — both spellings of the contract symbols must count)
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)
    }
    for required in ("load_flow_model", "load_t5xxl", "load_safetensors", "apply_sd_fixups"):
        assert required in names, f"patch_flux_utils must reference {required}"
    trys = [n for n in ast.walk(fn) if isinstance(n, ast.Try)]
    assert any(t.finalbody for t in trys), (
        "patch_flux_utils must restore the original load_safetensors in a finally block"
    )
    # Memory fix (train-1792246674, host OOM SIGKILL between the loader's
    # "Loading state dict" line and any traceback): the wrapper must hand
    # the loader's dict to the fixups in CONSUMING mode — an in_place
    # reference at the apply_sd_fixups call site inside patch_flux_utils
    # (keyword form; a positional True does not pin the intent). Without
    # it the raw ~17GB sd stays fully referenced while the fp32/bf16
    # intermediates pile up (~90GB peak) and the cgroup kills the run.
    call_kw = {
        kw.arg for n in ast.walk(fn) if isinstance(n, ast.Call) for kw in n.keywords if kw.arg
    }
    assert "in_place" in call_kw | names, (
        "patch_flux_utils must pass in_place through to apply_sd_fixups — "
        "consuming mode is the OOM fix"
    )


# ---------------------------------------------------------------------------
# BEHAVIORAL tests (torch-gated; file-existence asserted first so they are
# RED while the shim is missing, SKIPPED only in a torch-less CI with the
# shim present)
# ---------------------------------------------------------------------------


def test_strip_prefix_removes_prefix_from_all_keys():
    # F4: strip_comfyui_prefix is PURE dict-key logic (the shim imports torch
    # lazily only where dtypes are needed), so this runs in CI without torch.
    # Values are opaque sentinels: strip must rename keys and pass values
    # through by identity. Un-gating is a coverage move — this test is
    # expected GREEN immediately, it pins no new fix.
    assert SHIM_PATH.is_file()
    shim = _load_shim()
    marker, a, b, emb = object(), object(), object(), object()
    sd = {
        # unprefixed key FIRST here — the prod trap (early-but-not-first
        # scaled_fp8 marker breaking the loop mid-dict) taken to its
        # extreme: a breaking loop would rename nothing at all.
        "text_embedder.linear": emb,
        "model.diffusion_model.a": a,
        "scaled_fp8": marker,
        "model.diffusion_model.b.block": b,
    }
    out = shim.strip_comfyui_prefix(sd)
    assert set(out) == {"text_embedder.linear", "a", "scaled_fp8", "b.block"}
    # every prefixed key renamed despite the unprefixed keys coming first,
    # values passed through untouched (identity), input not mutated
    assert out["a"] is a
    assert out["b.block"] is b
    assert out["scaled_fp8"] is marker
    assert out["text_embedder.linear"] is emb
    assert set(sd) == {"text_embedder.linear", "model.diffusion_model.a", "scaled_fp8", "model.diffusion_model.b.block"}


def test_strip_prefix_in_place_returns_and_mutates_same_dict():
    # Memory fix (train-1792246674, host OOM SIGKILL): the wrapper must be
    # able to rename the loader's ~17GB dict WITHOUT copying it —
    # in_place=True mutates the input and returns THE SAME dict object,
    # with the same key-set semantics as copy mode (every prefixed key
    # renamed, unprefixed keys untouched, values passed by identity).
    # Pure dict-key logic — runs without torch, so this pin is GREEN in CI
    # today (the landed strip_comfyui_prefix accepts in_place; it was RED
    # before the parameter existed → TypeError).
    assert SHIM_PATH.is_file()
    shim = _load_shim()
    marker, a, b, emb = object(), object(), object(), object()
    sd = {
        "text_embedder.linear": emb,
        "model.diffusion_model.a": a,
        "scaled_fp8": marker,
        "model.diffusion_model.b.block": b,
    }
    out = shim.strip_comfyui_prefix(sd, in_place=True)
    assert out is sd, "in_place mode must return the SAME dict object (no copy)"
    assert set(out) == {"text_embedder.linear", "a", "scaled_fp8", "b.block"}
    assert out["a"] is a
    assert out["b.block"] is b
    assert out["scaled_fp8"] is marker
    assert out["text_embedder.linear"] is emb
    assert "model.diffusion_model.a" not in out
    assert "model.diffusion_model.b.block" not in out


def test_dequant_scalar_scale():
    # The bf16 result is now an EXPLICIT opt-in (target=torch.bfloat16):
    # the RAM reality (train-1792249442, <~26GB usable) moved the DEFAULT
    # to preserving the input fp8 dtype — pinned by
    # test_dequant_target_none_preserves_input_fp8_dtype.
    # torch-gated: SKIPPED in this sandbox; on a torch machine this
    # exercises the landed explicit-target opt-in.
    assert SHIM_PATH.is_file()
    _require_torch()
    import torch

    shim = _load_shim()
    t = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float8_e4m3fn)
    scale = torch.tensor(0.25)  # 0-dim scalar
    out = shim.dequant_scaled_fp8_sd({"x": t, "x.scale_weight": scale}, target=torch.bfloat16)
    assert out["x"].dtype == torch.bfloat16
    expected = (t.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)
    assert torch.equal(out["x"], expected)


def test_dequant_per_channel_scale():
    # bf16 explicit opt-in at the per-channel shape (same note as above).
    assert SHIM_PATH.is_file()
    _require_torch()
    import torch

    shim = _load_shim()
    t = torch.ones(4, 3).to(torch.float8_e4m3fn)
    scale = torch.tensor([[2.0], [0.5], [1.0], [4.0]])  # (out, 1) per-row broadcast
    out = shim.dequant_scaled_fp8_sd({"w": t, "w.scale_weight": scale}, target=torch.bfloat16)
    expected = (t.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)
    assert out["w"].shape == (4, 3)
    assert out["w"].dtype == torch.bfloat16
    assert torch.equal(out["w"], expected)


def test_dequant_drops_scale_keys_and_marker():
    assert SHIM_PATH.is_file()
    _require_torch()
    import torch

    shim = _load_shim()
    sd = {
        "scaled_fp8": torch.tensor(0, dtype=torch.uint8),  # bare marker, any dtype
        "a": torch.ones(2).to(torch.float8_e4m3fn),
        "a.scale_weight": torch.tensor(0.1),
        "b": torch.ones(2, dtype=torch.bfloat16),
    }
    out = shim.dequant_scaled_fp8_sd(sd)
    assert set(out) == {"a", "b"}
    assert not any(k.endswith(".scale_weight") for k in out)
    assert "scaled_fp8" not in out


def test_plain_fp8_without_scale_cast_to_bf16():
    # With an EXPLICIT bf16 target a stray fp8 tensor (no scale) is cast
    # through fp32; under the new default (target=None) it PRESERVES its
    # fp8 dtype instead — pinned in test_dequant_target_none... and in the
    # consumes-input test.
    assert SHIM_PATH.is_file()
    _require_torch()
    import torch

    shim = _load_shim()
    t = torch.tensor([1.0, -0.5], dtype=torch.float8_e4m3fn)
    out = shim.dequant_scaled_fp8_sd({"z": t}, target=torch.bfloat16)
    assert out["z"].dtype == torch.bfloat16
    assert torch.equal(out["z"], t.to(torch.float32).to(torch.bfloat16))


def test_non_fp8_tensors_pass_through():
    assert SHIM_PATH.is_file()
    _require_torch()
    import torch

    shim = _load_shim()
    a = torch.randn(3, 2, dtype=torch.bfloat16)
    b = torch.randn(2, 3, dtype=torch.float16)
    out = shim.dequant_scaled_fp8_sd({"a": a, "b": b})
    assert out["a"].dtype == torch.bfloat16
    assert torch.equal(out["a"], a)
    assert out["b"].dtype == torch.float16
    assert torch.equal(out["b"], b)


def test_dequant_target_none_preserves_input_fp8_dtype():
    # RAM reality (train-1792249442: SIGKILL right after "Loading state
    # dict from flux1-dev-fp8.safetensors" even with chunked+consuming
    # dequant — the box has < ~26GB usable, a ~24GB bf16 result cannot
    # fit): the DEFAULT target=None must keep every paired tensor in its
    # INPUT fp8 dtype — dequant fp32 × scale, re-quantize back. End-state
    # equivalence (corrected per review): this is what kohya's plain-fp8
    # checkpoint flow under fp8_base consumes — a SINGLE fp32→fp8
    # rounding. It is NOT bit-identical to the old bf16→kohya-cast-to-fp8
    # design (that path double-rounds, fp32→bf16→fp8, and can differ by
    # ≤1 fp8 ulp on ties; it also never completed a run — it died OOM),
    # so no "deployed end-state" exists to match. Values/scales here are
    # exact powers of two, so the assertions stay exact under either
    # reading. torch-gated: SKIPPED in this sandbox; on a torch machine
    # this verifies the landed target=None default.
    assert SHIM_PATH.is_file()
    _require_torch()
    import torch

    shim = _load_shim()
    t4 = torch.ones(2).to(torch.float8_e4m3fn)
    s4 = torch.tensor(0.5)  # scalar branch
    t5 = torch.ones(3, 2).to(torch.float8_e5m2)
    s5 = torch.tensor([[2.0], [0.5], [1.0]])  # per-channel (out, 1) branch
    sd = {
        "scaled_fp8": torch.tensor(0, dtype=torch.uint8),
        "a": t4,
        "a.scale_weight": s4,
        "b": t5,
        "b.scale_weight": s5,
    }
    before = set(sd)
    out = shim.dequant_scaled_fp8_sd(sd)  # default target=None = preserve
    assert out is not sd and set(sd) == before, "in_place=False must not consume"
    assert out["a"].dtype == torch.float8_e4m3fn
    assert out["b"].dtype == torch.float8_e5m2
    exp_a = (t4.to(torch.float32) * s4.to(torch.float32)).to(torch.float8_e4m3fn)
    exp_b = (t5.to(torch.float32) * s5.to(torch.float32)).to(torch.float8_e5m2)
    assert torch.equal(out["a"].to(torch.float32), exp_a.to(torch.float32))
    assert torch.equal(out["b"].to(torch.float32), exp_b.to(torch.float32))


def test_dequant_target_bfloat16_explicit():
    # The OLD contract as an explicit opt-in: target=torch.bfloat16 must
    # produce bf16 for both paired tensors (this is what the wrapper will
    # request for load_t5xxl). torch-gated: SKIPPED here; on a torch machine
    # this verifies the landed target parameter's explicit-bf16 branch.
    assert SHIM_PATH.is_file()
    _require_torch()
    import torch

    shim = _load_shim()
    t4 = torch.ones(2).to(torch.float8_e4m3fn)
    s4 = torch.tensor(0.5)
    t5 = torch.ones(3, 2).to(torch.float8_e5m2)
    s5 = torch.tensor([[2.0], [0.5], [1.0]])
    out = shim.dequant_scaled_fp8_sd(
        {
            "scaled_fp8": torch.tensor(0, dtype=torch.uint8),
            "a": t4,
            "a.scale_weight": s4,
            "b": t5,
            "b.scale_weight": s5,
        },
        target=torch.bfloat16,
    )
    assert out["a"].dtype == torch.bfloat16
    assert out["b"].dtype == torch.bfloat16
    exp_a = (t4.to(torch.float32) * s4.to(torch.float32)).to(torch.bfloat16)
    exp_b = (t5.to(torch.float32) * s5.to(torch.float32)).to(torch.bfloat16)
    assert torch.equal(out["a"], exp_a)
    assert torch.equal(out["b"], exp_b)


def test_dequant_in_place_consumes_input_dict():
    # Memory fix, core half (train-1792246674): in_place=True must POP from
    # the input as it goes so the fp8 memory is freed progressively — the
    # copy mode's full-fp32 intermediates on top of the still-referenced
    # raw sd peaked at ~90GB and the cgroup SIGKILLed the run (no child
    # traceback, rc 1). Contract: correct result AND after the call the
    # input holds NONE of the consumed keys (paired bases, their scales,
    # the marker); the orphan .scale_weight must not be returned either.
    # torch-gated: SKIPPED in this sandbox; on a torch machine this verifies
    # the landed preserve-fp8 default (it would have failed against the old
    # bf16-forcing code).
    assert SHIM_PATH.is_file()
    _require_torch()
    import torch

    shim = _load_shim()
    t1 = torch.ones(2).to(torch.float8_e4m3fn)
    s1 = torch.tensor(0.5)
    t2 = torch.ones(3).to(torch.float8_e4m3fn)
    s2 = torch.tensor([[2.0], [0.5], [1.0]])
    # (1,) 1-D scale — the scalar-branch shape the shim docstring claims
    # ("0-dim, (1,) or (1,1)"): the CONSUMING path must see it too, not
    # only the bit-exact comparison.
    t3 = torch.ones(2).to(torch.float8_e4m3fn)
    s3 = torch.tensor([0.5])
    plain = torch.ones(2, dtype=torch.bfloat16)
    stray = torch.ones(1).to(torch.float8_e4m3fn)
    sd = {
        "scaled_fp8": torch.tensor(0, dtype=torch.uint8),
        "a": t1,
        "a.scale_weight": s1,
        "b": t2,
        "b.scale_weight": s2,
        "c": t3,
        "c.scale_weight": s3,
        "orphan.scale_weight": torch.tensor(1.0),
        "plain": plain,
        "stray": stray,
    }
    out = shim.dequant_scaled_fp8_sd(sd, in_place=True)
    # NEW default target policy (train-1792249442 RAM pin): paired fp8
    # tensors are re-quantized back into their INPUT dtype (all products
    # here are exact powers of two), and a stray fp8 tensor without a
    # scale PRESERVES its fp8 dtype (no bf16 inflation) — the landed default
    # this pin verifies (the old code forced bf16 for both).
    exp_a = (t1.to(torch.float32) * s1.to(torch.float32)).to(torch.float8_e4m3fn)
    exp_b = (t2.to(torch.float32) * s2.to(torch.float32)).to(torch.float8_e4m3fn)
    exp_c = (t3.to(torch.float32) * s3.to(torch.float32)).to(torch.float8_e4m3fn)
    assert set(out) == {"a", "b", "c", "plain", "stray"}, set(out)
    assert out["a"].dtype == torch.float8_e4m3fn
    assert out["b"].dtype == torch.float8_e4m3fn
    assert out["c"].dtype == torch.float8_e4m3fn
    assert torch.equal(out["a"].to(torch.float32), exp_a.to(torch.float32))
    assert torch.equal(out["b"].to(torch.float32), exp_b.to(torch.float32))
    assert torch.equal(out["c"].to(torch.float32), exp_c.to(torch.float32))
    assert torch.equal(out["plain"], plain)
    assert out["stray"].dtype == torch.float8_e4m3fn
    for key in ("scaled_fp8", "a", "a.scale_weight", "b", "b.scale_weight", "c", "c.scale_weight"):
        assert key not in sd, f"in_place mode must consume {key!r} out of the input dict"


def test_dequant_in_place_matches_copy_mode_bit_exact():
    # Numerics guard for the chunked fp32 refactor: the SAME synthetic sd
    # run through copy mode and through in_place mode must produce
    # bit-identical tensors (chunk boundaries must not shift rounding).
    # Target-AGNOSTIC by design (compares the two modes to each other):
    # green under the old bf16-forcing code AND the landed preserve-fp8
    # default — it pins mode equality, never the target policy (that is
    # tests 1-3's job). torch-gated: SKIPPED in this sandbox.
    assert SHIM_PATH.is_file()
    _require_torch()
    import torch

    shim = _load_shim()

    def synth():
        return {
            "scaled_fp8": torch.tensor(0, dtype=torch.uint8),
            "a": torch.linspace(-1.0, 1.0, 6).reshape(3, 2).to(torch.float8_e4m3fn),
            "a.scale_weight": torch.tensor([[2.0], [0.5], [1.0]]),
            "b": torch.ones(4).to(torch.float8_e4m3fn),
            "b.scale_weight": torch.tensor(0.25),
            # the scalar-branch shapes the shim docstring claims to handle
            # ("0-dim, (1,) or (1,1)") — b covers 0-dim; these two pin the
            # 1-element 1-D and (1,1) encodings, previously comment-only.
            "e": torch.ones(3).to(torch.float8_e4m3fn),
            "e.scale_weight": torch.tensor([0.5]),
            "f": torch.linspace(0.0, 3.0, 4).to(torch.float8_e4m3fn),
            "f.scale_weight": torch.tensor([[0.25]]),
            "c": torch.ones(2, dtype=torch.float16),
            "d": torch.ones(1).to(torch.float8_e5m2),
        }

    copy_out = shim.dequant_scaled_fp8_sd(synth())
    inplace_out = shim.dequant_scaled_fp8_sd(synth(), in_place=True)
    # Mode-EQUALITY only: the two modes must agree on keys, on each
    # tensor's dtype and on its values (fp32 view — exact for both fp8 and
    # bf16 results). The dtype POLICY itself (preserve-fp8 default vs
    # explicit bf16) is pinned by the target tests, NOT here — this pin
    # stays green under whichever target the default uses, guarding only
    # that the chunked/consuming refactor cannot shift rounding.
    assert set(copy_out) == set(inplace_out)
    for key in copy_out:
        assert copy_out[key].dtype == inplace_out[key].dtype, key
        assert torch.equal(
            copy_out[key].to(torch.float32), inplace_out[key].to(torch.float32)
        ), key


def test_dequant_default_mode_does_not_consume_input():
    # Back-compat pin for the memory fix: the DEFAULT (copy) mode keeps its
    # non-consuming semantics — a new dict out, every input key still
    # there. GREEN today and after the fix (regression guard, not a red
    # pin). torch-gated: SKIPPED in this sandbox.
    assert SHIM_PATH.is_file()
    _require_torch()
    import torch

    shim = _load_shim()
    t = torch.ones(2).to(torch.float8_e4m3fn)
    sd = {
        "scaled_fp8": torch.tensor(0, dtype=torch.uint8),
        "a": t,
        "a.scale_weight": torch.tensor(0.5),
    }
    before = set(sd)
    out = shim.dequant_scaled_fp8_sd(sd)
    assert set(sd) == before, "default copy mode must not consume the input"
    assert out is not sd
    assert set(out) == {"a"}


def test_apply_sd_fixups_composed():
    # Representative ComfyUI-style scaled-fp8 state dict: marker EARLY
    # (first in this fixture — stricter than prod's early-but-not-first),
    # prefixed tensors with per-channel .scale_weight pairs, one unprefixed
    # key and one plain fp16 tensor. The composed pipeline must land on:
    # stripped keys, no markers, no scale keys — and under the NEW default
    # target policy (train-1792249442 RAM pin) the paired fp8 tensor is
    # re-quantized back into its INPUT fp8 dtype (products here are exact
    # powers of two) — the landed default this pin verifies.
    assert SHIM_PATH.is_file()
    _require_torch()
    import torch

    shim = _load_shim()
    t = torch.ones(3, 2).to(torch.float8_e4m3fn)
    scale = torch.tensor([[2.0], [0.5], [1.0]])
    plain = torch.ones(2, dtype=torch.float16)
    emb = torch.ones(1, dtype=torch.bfloat16)
    sd = {
        "scaled_fp8": torch.tensor(0, dtype=torch.uint8),
        "model.diffusion_model.blocks.0.attn.qkv.weight": t,
        "model.diffusion_model.blocks.0.attn.qkv.weight.scale_weight": scale,
        "model.diffusion_model.final.linear": plain,
        "text_embedder.emb": emb,
    }
    out = shim.apply_sd_fixups(sd)
    assert set(out) == {"blocks.0.attn.qkv.weight", "final.linear", "text_embedder.emb"}
    qkv = out["blocks.0.attn.qkv.weight"]
    assert qkv.dtype == torch.float8_e4m3fn, (
        "default composed policy: preserve the input fp8 dtype (target=None)"
    )
    expected = (t.to(torch.float32) * scale.to(torch.float32)).to(torch.float8_e4m3fn)
    assert torch.equal(qkv.to(torch.float32), expected.to(torch.float32))
    assert torch.equal(out["final.linear"], plain)
    assert torch.equal(out["text_embedder.emb"], emb)


def test_patch_flux_utils_wraps_load_safetensors_and_restores():
    # The definitive fake-module contract from the fix design: a module
    # whose load_flow_model internally calls the module-global
    # load_safetensors; after patch_flux_utils, calling load_flow_model
    # returns the FIXED sd, and the module's load_safetensors is the
    # ORIGINAL again afterwards.
    # Validator condition 2: the fake must mirror the REAL module's
    # entry-point set — production patch_flux_utils wraps load_flow_model
    # AND load_t5xxl unconditionally, so a fake without load_t5xxl dies on
    # AttributeError on a torch machine and never exercises the assertions.
    assert SHIM_PATH.is_file()
    _require_torch()
    import torch

    shim = _load_shim()
    mod = types.ModuleType("library.flux_utils")
    exec(  # noqa: S102 - building the fake module namespace on purpose
        "def load_safetensors(path):\n"
        "    return dict(RAW)\n"
        "def load_flow_model(sd_path, dtype=None, device=None):\n"
        "    sd = load_safetensors(sd_path)\n"
        "    return sd\n"
        "def load_t5xxl(dtype, device=None):\n"
        "    sd = load_safetensors(T5_PATH)\n"
        "    return sd\n",
        mod.__dict__,
    )
    t = torch.ones(2).to(torch.float8_e4m3fn)
    scale = torch.tensor(0.5)
    mod.RAW = {
        "scaled_fp8": torch.tensor(0, dtype=torch.uint8),
        "model.diffusion_model.a": t,
        "model.diffusion_model.a.scale_weight": scale,
    }
    mod.T5_PATH = "t5xxl_fp8_e4m3fn_scaled.safetensors"
    original_loader = mod.load_safetensors
    original_flow = mod.load_flow_model

    shim.patch_flux_utils(mod)
    assert mod.load_flow_model is not original_flow, "patch_flux_utils must wrap load_flow_model"

    out = mod.load_flow_model("fake.safetensors")
    assert set(out) == {"a"}, out.keys()
    # RAM policy (train-1792249442): the wrapper must run the DiT sd with
    # target=None — the fp8 pair STAYS fp8 (bf16 result ~24GB does not fit
    # the box) — the landed wrapper behavior this pin verifies.
    assert out["a"].dtype == torch.float8_e4m3fn
    expected = (t.to(torch.float32) * scale.to(torch.float32)).to(torch.float8_e4m3fn)
    assert torch.equal(out["a"].to(torch.float32), expected.to(torch.float32))
    # the temporary load_safetensors patch must be restored after the call
    assert mod.load_safetensors is original_loader


def test_patch_flux_utils_also_fixes_load_t5xxl_path():
    # F2: upstream loads the T5 via flux_utils.load_t5xxl AFTER
    # load_flow_model returned — if only the flow path is wrapped, the
    # temporary load_safetensors swap is already restored by then and the
    # scaled-fp8 T5 state dict reaches the trainer raw (numerically wrong TE
    # outputs). patch_flux_utils must wrap BOTH loaders with the same
    # swap-and-restore pattern, and both wrappers must coexist on one
    # patched module.
    assert SHIM_PATH.is_file()
    _require_torch()
    import torch

    shim = _load_shim()
    mod = types.ModuleType("library.flux_utils")
    exec(  # noqa: S102 - building the fake module namespace on purpose
        "def load_safetensors(path):\n"
        "    return dict(RAW)\n"
        "def load_flow_model(sd_path, dtype=None, device=None):\n"
        "    sd = load_safetensors(sd_path)\n"
        "    return sd\n"
        "def load_t5xxl(dtype, device=None):\n"
        "    sd = load_safetensors(T5_PATH)\n"
        "    return sd\n",
        mod.__dict__,
    )
    t_flow = torch.ones(2).to(torch.float8_e4m3fn)
    s_flow = torch.tensor(0.5)
    t_t5 = torch.ones(3).to(torch.float8_e4m3fn)
    s_t5 = torch.tensor(0.25)
    mod.RAW = {
        "scaled_fp8": torch.tensor(0, dtype=torch.uint8),
        "model.diffusion_model.a": t_flow,
        "model.diffusion_model.a.scale_weight": s_flow,
        "t5_block.weight": t_t5,
        "t5_block.weight.scale_weight": s_t5,
    }
    mod.T5_PATH = "t5xxl_fp8_e4m3fn_scaled.safetensors"
    original_loader = mod.load_safetensors
    original_t5 = mod.load_t5xxl

    shim.patch_flux_utils(mod)
    assert mod.load_t5xxl is not original_t5, "patch_flux_utils must wrap load_t5xxl too"

    # per-path target policy (train-1792249442): the flow path preserves
    # the fp8 dtypes (target=None), the T5 path gets explicit bf16; the
    # products here are exact powers of two in both dtypes.
    exp_flow_a = (t_flow.to(torch.float32) * s_flow.to(torch.float32)).to(torch.float8_e4m3fn)
    exp_flow_t5 = (t_t5.to(torch.float32) * s_t5.to(torch.float32)).to(torch.float8_e4m3fn)
    exp_t5_a = (t_flow.to(torch.float32) * s_flow.to(torch.float32)).to(torch.bfloat16)
    exp_t5_t5 = (t_t5.to(torch.float32) * s_t5.to(torch.float32)).to(torch.bfloat16)

    # the upstream sequence: flow first, T5 loaded AFTER it returned —
    # both paths must still hand out the FIXED sd, and the module-global
    # loader must be restored after EACH call (no swap leak either way).
    out_flow = mod.load_flow_model("flux.safetensors")
    assert mod.load_safetensors is original_loader, "flow call must restore the loader"
    out_t5 = mod.load_t5xxl("bf16")
    assert mod.load_safetensors is original_loader, "t5 call must restore the loader too"

    for out in (out_flow, out_t5):
        assert "scaled_fp8" not in out, out.keys()
        assert not any(str(k).endswith(".scale_weight") for k in out), out.keys()
        assert "model.diffusion_model.a" not in out, out.keys()

    # flow path: fp8 preserved (the landed default; old code forced bf16)
    assert out_flow["a"].dtype == torch.float8_e4m3fn
    assert out_flow["t5_block.weight"].dtype == torch.float8_e4m3fn
    assert torch.equal(out_flow["a"].to(torch.float32), exp_flow_a.to(torch.float32))
    assert torch.equal(
        out_flow["t5_block.weight"].to(torch.float32), exp_flow_t5.to(torch.float32)
    )
    # T5 path: explicit bf16 (TE-cache quality must not degrade to fp8)
    assert out_t5["a"].dtype == torch.bfloat16
    assert out_t5["t5_block.weight"].dtype == torch.bfloat16
    assert torch.equal(out_t5["a"], exp_t5_a)
    assert torch.equal(out_t5["t5_block.weight"], exp_t5_t5)


def test_wrapper_loaders_use_distinct_targets():
    # The wrapper's per-loader target policy (train-1792249442 design):
    # load_flow_model runs the fixups with target=None (DiT sd stays in
    # its ~17GB fp8 form — the only footprint the box can hold), while
    # load_t5xxl requests an explicit bf16 target (TE-cache quality).
    # One fake module, BOTH loaders calling the module-global
    # load_safetensors; after patch_flux_utils the two paths must return
    # the same pair in DIFFERENT dtypes. torch-gated: SKIPPED here; on a
    # torch machine this verifies the landed per-loader target split (the
    # old wrapper forced bf16 for both).
    assert SHIM_PATH.is_file()
    _require_torch()
    import torch

    shim = _load_shim()
    mod = types.ModuleType("library.flux_utils")
    exec(  # noqa: S102 - building the fake module namespace on purpose
        "def load_safetensors(path):\n"
        "    return dict(RAW)\n"
        "def load_flow_model(sd_path, dtype=None, device=None):\n"
        "    sd = load_safetensors(sd_path)\n"
        "    return sd\n"
        "def load_t5xxl(dtype, device=None):\n"
        "    sd = load_safetensors(T5_PATH)\n"
        "    return sd\n",
        mod.__dict__,
    )
    t = torch.ones(2).to(torch.float8_e4m3fn)
    scale = torch.tensor(0.5)
    mod.RAW = {
        "scaled_fp8": torch.tensor(0, dtype=torch.uint8),
        "a": t,
        "a.scale_weight": scale,
    }
    mod.T5_PATH = "t5xxl_fp8_e4m3fn_scaled.safetensors"
    original_loader = mod.load_safetensors

    shim.patch_flux_utils(mod)

    flow_out = mod.load_flow_model("flux1-dev-fp8.safetensors")
    assert set(flow_out) == {"a"}, flow_out.keys()
    assert flow_out["a"].dtype == torch.float8_e4m3fn, (
        "the flow path must PRESERVE fp8 (target=None) — the bf16 result "
        "does not fit the box's <~26GB RAM (train-1792249442)"
    )
    assert mod.load_safetensors is original_loader

    t5_out = mod.load_t5xxl("bf16")
    assert set(t5_out) == {"a"}, t5_out.keys()
    assert t5_out["a"].dtype == torch.bfloat16, (
        "the T5 path must get the explicit bf16 target (TE-cache quality)"
    )
    assert mod.load_safetensors is original_loader


def test_install_meta_path_hook_is_idempotent():
    assert SHIM_PATH.is_file()
    _require_torch()

    # snapshot BEFORE loading: a module-level install() (one of the allowed
    # F1 activations) already registers its finder during exec_module — it
    # must be counted and cleaned exactly like the explicit calls.
    shim_name = "kohya_compat_shim_under_test"
    before = list(sys.meta_path)
    try:
        shim = _load_shim()
        shim.install()
        shim.install()
        mine = [f for f in sys.meta_path if getattr(type(f), "__module__", "") == shim_name]
        assert len(mine) == 1, f"install() must be idempotent; found {len(mine)} finders from the shim"
    finally:
        for finder in [f for f in sys.meta_path if getattr(type(f), "__module__", "") == shim_name]:
            sys.meta_path.remove(finder)
    assert list(sys.meta_path) == before, "test must leave sys.meta_path untouched"


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if not (name.startswith("test_") and callable(function)):
            continue
        try:
            function()
        except _Skipped as exc:
            print(f"SKIP {name} ({exc})")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}: {exc!r}")
        else:
            print(f"PASS {name}")
    raise SystemExit(1 if failures else 0)
