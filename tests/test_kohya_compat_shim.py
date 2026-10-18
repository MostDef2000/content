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

Dtype-hygiene pins (train-1792332693, probe #5): the ostris scaled-fp8
T5 stores SOME tensors non-fp8 — notably the token embedding in fp16 —
and Pass 2's else-branch passed every non-fp8 tensor through regardless
of target, so the T5 path (load_t5xxl, dtype=None + assign=True, no
post-cast) produced a mixed-dtype model: Half activations against
dequanted bf16 weights crash the first T5 attention
(modeling_t5.py:304 via strategy_flux). Fix contract: with target NOT
None, floating-point non-fp8 tensors are cast to the target; int/bool/
uint8 are NEVER cast; target=None (the flow path) keeps byte-identical
passthrough. P7/P8 run on a FAKE torch with dtype-tracking tensors
(no numerics) so they are CI-visible without torch — P7 pins the target
cast (fp16 embed must land bf16), P8 pins the gating (target=None must
cast NOTHING — the D-006/D-007 fp8-preserve defense).

The sitecustomize ACTIVATION defect (reviewer F1 — import alone never
calls install()) is pinned behaviorally on the prod path in
tests/test_train_lora_flux.py::test_flux_bootstrap_installs_compat_shim_into_site_packages.

D-007 TE-SKIP pins (two-pass memory design, Valery W3): upstream
flux_train_network.py load_target_model loads CLIP-L (:141), T5XXL (:153)
and AE (:162) UNCONDITIONALLY — no cache-existence skip — so with a
complete on-disk text-encoder cache the trainer must NOT pay for the TEs
at all. With env KOHYA_SKIP_TEXT_ENCODERS present and NON-EMPTY,
patch_flux_utils must additionally wrap flux_utils.load_clip_l and
flux_utils.load_t5xxl to return a STUB instead of loading from disk: real
.dtype (torch.bfloat16 — the fp8 dtype check at flux_train_network.py
:157-160 reads it), eval()/to()/train() no-ops returning self (the stub
surface the cache path touches at :144/:154/:246-247/:284-286), and any
encode/forward call raising RuntimeError containing "text encoder cache"
(guard against an incomplete cache / stub misuse). Since train-1792334532
the stub ALSO exposes the network-creation traversal surface —
named_modules/named_parameters/named_buffers/buffers yield NOTHING
(the LoRA-targeting surface: zero yielded modules = zero TE LoRA = the
designed skip; pinned by P9) and state_dict() == {} / requires_grad_
returns self — because networks/lora_flux.py create_modules (:819,
reached from :947 "create LoRA for Text Encoder 1") TRAVERSES the text
encoder even under network_train_unet_only. REVISED by train-1792365662:
parameters() must yield >=3 bf16-dtype placeholder params (the
training-loop start log at train_network.py:1521-1525 reads the third),
and the stub carries .device with .type == "cpu" (the :1263 gate that
skips the fp8 cast block for a weightless stub; pinned by P10). With the
env absent or
EMPTY the real loaders pass through untouched; the DiT path
(load_flow_model) stays wrapped and live under the gate either way. The
behavioral pins inject a FAKE torch module via sys.modules — the gate
path needs no real tensor math — so they are RED/GREEN in torch-less CI
instead of skipping (same lazy-import seam the strip tests rely on).

pytest-compatible; pytest is not required.

Run either way:
    python -m pytest tests/test_kohya_compat_shim.py -q
    python tests/test_kohya_compat_shim.py
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import os
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
# D-007 harness: env mechanism + fake-torch injection for the TE-skip gate
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _env(**assignments: str | None):
    """Harness env mechanism: set each NAME to its value (None pops the key),
    restore the previous state in finally. Lets the TE-skip pins flip
    KOHYA_SKIP_TEXT_ENCODERS deterministically inside the test process."""
    saved = {}
    for key, value in assignments.items():
        saved[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def _fake_torch_installed():
    """Inject a minimal fake `torch` into sys.modules for the duration of the
    block (restored in finally, real torch untouched when installed). The
    TE-skip gate path performs NO real tensor math — the stub carries the
    dtype sentinel by reference and the sd-fixup passthrough only does
    dict-key work on non-tensor values — so these pins run (and can be RED)
    in torch-less CI instead of skipping, exactly like the strip tests rely
    on the shim's lazy torch import."""
    fake = types.ModuleType("torch")

    class _FakeDtype:
        def __init__(self, name: str) -> None:
            self.name = name

        def __repr__(self) -> str:
            return f"torch.{self.name}"

    class _FakeTensor:  # isinstance target only — never constructed here
        pass

    class _FakeDevice:
        """torch.device(...) stand-in (train-1792365662 stub .device pin):
        additive so a fix may legitimately build the device via
        torch.device("cpu") under the fake; existing pins never touch it."""

        def __init__(self, spec):
            self.type = str(spec).split(":")[0]

        def __repr__(self):
            return f"device(type='{self.type}')"

    fake.bfloat16 = _FakeDtype("bfloat16")
    fake.float32 = _FakeDtype("float32")
    fake.float8_e4m3fn = _FakeDtype("float8_e4m3fn")
    fake.float8_e5m2 = _FakeDtype("float8_e5m2")
    fake.Tensor = _FakeTensor
    fake.device = _FakeDevice
    saved = sys.modules.get("torch")
    sys.modules["torch"] = fake
    try:
        yield fake
    finally:
        if saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved


# Upstream signatures (library/flux_utils.py @690ea7f9): load_clip_l(ckpt_path,
# dtype, device, disable_mmap=False, state_dict=None) at :248 and
# load_t5xxl(ckpt_path, dtype, device, disable_mmap=False, state_dict=None) at
# :358 — the fakes mirror them and count invocations per loader, so "the
# original loader was NOT invoked" is observable rather than assumed.
_FAKE_FLUX_UTILS_SRC = (
    "def load_safetensors(path):\n"
    "    CALLS['safetensors'] += 1\n"
    "    return dict(RAW)\n"
    "def load_flow_model(ckpt_path, dtype=None, device=None, disable_mmap=False, state_dict=None):\n"
    "    CALLS['flow'] += 1\n"
    "    return load_safetensors(ckpt_path)\n"
    "def load_clip_l(ckpt_path, dtype, device, disable_mmap=False, state_dict=None):\n"
    "    CALLS['clip_l'] += 1\n"
    "    return {'clip_l_original': ckpt_path}\n"
    "def load_t5xxl(ckpt_path, dtype, device, disable_mmap=False, state_dict=None):\n"
    "    CALLS['t5xxl'] += 1\n"
    "    return {'t5xxl_original': ckpt_path}\n"
)


def _fake_flux_utils_with_counters():
    """library.flux_utils fake whose loaders increment per-loader counters
    (harness stub counters for the TE-skip gate pins). RAW carries no
    scaled-fp8 marker, no scale pairs and no tensors, so the DiT fixup
    wrapper passes it through with pure dict-key logic under the fake torch."""
    mod = types.ModuleType("library.flux_utils")
    exec(_FAKE_FLUX_UTILS_SRC, mod.__dict__)  # noqa: S102 - fake module namespace
    mod.CALLS = {"safetensors": 0, "flow": 0, "clip_l": 0, "t5xxl": 0}
    mod.RAW = {"text_embedder.linear": object()}
    return mod


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


def test_shim_source_has_skip_te_gate_and_stub_surface():
    # D-007 two-pass memory design, STRUCTURAL half (CI-visible, unconditional):
    # the shim must gate on KOHYA_SKIP_TEXT_ENCODERS (read from the
    # environment), define a TE stub class exposing the surface upstream
    # touches on the skipped TEs — .dtype (flux_train_network.py:157-160 fp8
    # dtype check), .eval() (:144/:154), .to() (:246-247/:284-286), .train()
    # — plus an encode/forward guard raising RuntimeError with the literal
    # "text encoder cache", and reference load_clip_l (the second TE loader
    # that must be stubbed alongside load_t5xxl). The gate has landed: this
    # pin is the STRUCTURAL REGRESSION GUARD for the D-007 gate — CI-visible
    # without torch, it fails if the env gate, the stub surface, or the
    # tripwire message is ever removed or renamed in the shim source (the
    # behavioral half below is torch-gated, so this is the only
    # unconditional protection of the design).
    src = _read_shim_source()
    assert "KOHYA_SKIP_TEXT_ENCODERS" in src, (
        "shim must gate the TE stubbing on the KOHYA_SKIP_TEXT_ENCODERS env var"
    )
    assert "text encoder cache" in src, (
        'the stub guard message must contain the literal "text encoder cache"'
    )
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    literals = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert "environ" in attrs or "getenv" in attrs, (
        "the gate must read the environment (os.environ / os.getenv)"
    )
    assert "load_clip_l" in names | attrs | literals, (
        "the shim must wrap flux_utils.load_clip_l under the gate (both TEs: "
        "clip_l AND t5xxl — upstream loads them unconditionally at :141/:153)"
    )
    # TE stub surface: a class with eval/to/train no-ops and an encode/
    # forward (or __getattr__) guard.
    stub_cls = None
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        methods = {m.name for m in cls.body if isinstance(m, ast.FunctionDef)}
        if {"eval", "to", "train"} <= methods and ({"encode", "forward", "__getattr__"} & methods):
            stub_cls = cls
            break
    assert stub_cls is not None, (
        "shim must define a TE stub class exposing eval/to/train and an "
        "encode/forward (or __getattr__) guard"
    )
    members = {m.name for m in stub_cls.body if isinstance(m, ast.FunctionDef)}
    dtype_ok = "dtype" in members or any(
        isinstance(n, ast.Attribute) and n.attr == "dtype" for n in ast.walk(stub_cls)
    )
    assert dtype_ok, "the TE stub must expose .dtype (a real torch dtype — bf16)"
    guard_fns = [
        m for m in stub_cls.body
        if isinstance(m, ast.FunctionDef) and m.name in {"encode", "forward", "__getattr__", "__call__"}
    ]
    raises_runtime_error = any(
        isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "RuntimeError"
        for m in guard_fns
        for node in ast.walk(m)
    )
    assert raises_runtime_error, (
        "the TE stub encode/forward guard must raise RuntimeError (cache-incomplete tripwire)"
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


# ---------------------------------------------------------------------------
# train-1792332693 dtype pins — fake torch WITH dtype-tracking tensors, so
# these run (and can be RED) in torch-less CI, unlike the torch-gated
# dequant pins above (which skip here). The pins assert dtypes only, never
# numerics, so the fake carries shape/dtype bookkeeping and no math.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _fake_torch_with_tensors():
    """P7/P8 harness: the shared _fake_torch_installed never constructs
    tensors; the dequant paths touch torch.empty and tensor surfaces
    (reshape/view/numel/dim/getitem/setitem/mul/to). This extended fake
    tracks ONLY dtype and shape — every derived tensor keeps the dtype
    chain honest, values are irrelevant — with the same
    sys.modules injection/restore discipline, so the shim's lazy
    `import torch` resolves the fake inside dequant_scaled_fp8_sd."""
    fake = types.ModuleType("torch")

    class _FakeDtype:
        def __init__(self, name: str) -> None:
            self.name = name

        def __repr__(self) -> str:
            return f"torch.{self.name}"

    for _name in (
        "bfloat16", "float32", "float64", "float16", "int64", "uint8",
        "bool", "float8_e4m3fn", "float8_e5m2",
    ):
        setattr(fake, _name, _FakeDtype(_name))

    class _FakeTensor:
        def __init__(self, dtype, shape):
            self.dtype = dtype
            self.shape = tuple(shape)

        def numel(self):
            n = 1
            for s in self.shape:
                n *= s
            return n

        def dim(self):
            return len(self.shape)

        def is_floating_point(self):
            # torch semantics: fp8 dtypes ARE floating point (the shim's
            # fp8 branch handles them before any float-cast gate; the
            # method exists so a method-based fix implementation is not
            # forced into a dtype-set comparison by the harness)
            return self.dtype in (
                fake.bfloat16, fake.float32, fake.float64, fake.float16,
                fake.float8_e4m3fn, fake.float8_e5m2,
            )

        def to(self, dtype):
            return _FakeTensor(dtype, self.shape)

        def reshape(self, *args):
            return _FakeTensor(self.dtype, (self.numel(),))

        def view(self, *args):
            return _FakeTensor(self.dtype, (self.numel(),))

        def __getitem__(self, index):
            return _FakeTensor(self.dtype, self.shape)

        def __mul__(self, other):
            return _FakeTensor(self.dtype, self.shape)

        def __setitem__(self, index, value):
            pass  # dtype-tracking fake: element writes are irrelevant

    fake.Tensor = _FakeTensor
    fake.empty = lambda shape, dtype=None: _FakeTensor(dtype, shape)
    saved = sys.modules.get("torch")
    sys.modules["torch"] = fake
    try:
        yield fake
    finally:
        if saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved


def _mixed_scaled_fp8_sd(torch) -> dict:
    """The ostris t5xxl_fp8_e4m3fn_scaled shape (train-1792332693):
    paired fp8 tensors with scalar scales (e4m3fn AND e5m2), a STRAY
    unpaired fp8, the fp16 token-embedding stand-in (the defect tensor),
    a bf16 attention weight, an int64 step counter, a bool mask and the
    top-level scaled_fp8 marker."""
    T = torch.Tensor
    return {
        "qkv": T(torch.float8_e4m3fn, (4,)),
        "qkv.scale_weight": T(torch.float32, ()),
        "mlp": T(torch.float8_e5m2, (4,)),
        "mlp.scale_weight": T(torch.float32, ()),
        "stray_fp8": T(torch.float8_e4m3fn, (4,)),
        "embed_tokens": T(torch.float16, (4,)),
        "attn_bf16": T(torch.bfloat16, (4,)),
        "shared_step": T(torch.int64, (4,)),
        "bool_mask": T(torch.bool, (4,)),
        "scaled_fp8": T(torch.uint8, ()),
    }


def test_dequant_target_bf16_casts_stray_floats_to_target():
    # P7 (prod train-1792332693, probe #5 — landed RED-first against the
    # pre-fix shim; PERMANENT contract guard): the cache pass cleared
    # CLIP (encode complete) and died at the T5's first attention —
    # "expected mat1 and mat2 to have the same dtype, but got: c10::Half
    # != c10::BFloat16" (modeling_t5.py:304 via strategy_flux T5 encode).
    # Mechanism: the ostris scaled-fp8 checkpoint stores SOME tensors
    # non-fp8 — notably the token embedding in fp16 — and Pass 2's
    # else-branch passed every non-fp8 tensor through UNCHANGED
    # regardless of target; load_t5xxl runs with dtype=None and
    # assign=True (no post-cast), so the model came out mixed-dtype:
    # fp16 embedding -> Half activations against bf16 dequanted weights.
    # Fix contract: with target NOT None, floating-point non-fp8 tensors
    # are cast to the target (value.to(target)); int/bool/uint8 are NEVER
    # cast; both modes (copy + in_place) behave identically. Red
    # signature if the cast branch is ever removed: the fp16 embedding
    # comes back fp16.
    assert SHIM_PATH.is_file()
    with _fake_torch_with_tensors() as torch:
        shim = _load_shim()

        sd = _mixed_scaled_fp8_sd(torch)
        out = shim.dequant_scaled_fp8_sd(sd, target=torch.bfloat16)
        # fp8 handling (paired + stray) — already target-cast, unchanged:
        assert out["qkv"].dtype is torch.bfloat16
        assert out["mlp"].dtype is torch.bfloat16
        assert out["stray_fp8"].dtype is torch.bfloat16
        # THE FIX HALF — non-fp8 floats follow the target too:
        assert out["embed_tokens"].dtype is torch.bfloat16, (
            "train-1792332693: a stray fp16 float tensor must be cast to "
            "the bf16 target (the ostris T5 stores the token embedding in "
            "fp16; Half activations against dequanted bf16 weights crash "
            f"the first T5 attention); got: {out['embed_tokens'].dtype}"
        )
        assert out["attn_bf16"].dtype is torch.bfloat16
        # never-cast classes keep their dtype:
        assert out["shared_step"].dtype is torch.int64
        assert out["bool_mask"].dtype is torch.bool
        # housekeeping: marker and scale keys gone; copy mode leaves the
        # input dict untouched (the fp16 entry still fp16 there):
        assert "scaled_fp8" not in out
        assert not any(k.endswith(".scale_weight") for k in out)
        assert sd["embed_tokens"].dtype is torch.float16, (
            "copy mode must not mutate the input tensors"
        )

        # in_place mode: SAME outcome, input consumed as before:
        sd2 = _mixed_scaled_fp8_sd(torch)
        out2 = shim.dequant_scaled_fp8_sd(sd2, in_place=True, target=torch.bfloat16)
        assert out2["embed_tokens"].dtype is torch.bfloat16, (
            f"in_place mode must cast identically to copy mode; got: "
            f"{out2['embed_tokens'].dtype}"
        )
        assert out2["shared_step"].dtype is torch.int64
        assert out2["bool_mask"].dtype is torch.bool
        assert set(sd2) == set(), (
            f"in_place must consume the input as before; left: {set(sd2)}"
        )


def test_dequant_target_none_never_casts_anything():
    # P8 (the GATING half of the fix — green-by-construction against the
    # pre-fix shim, disclosed the way P4/P6 were; its bite is against the
    # NEW branch): target=None is the D-006/D-007 flow-path contract
    # (fp8-preserve, train-1792249442 RAM reality) — NOTHING may be cast:
    # paired fp8 re-quantizes into its OWN fp8 dtype (e4m3fn AND e5m2
    # alike), the stray fp8 stays fp8, the fp16 embedding stays fp16, the
    # bf16 stays bf16, int/bool untouched. Discriminator: a fix that
    # casts non-fp8 floats UNCONDITIONALLY (dropping the
    # `target is not None` gate) flips the fp16/bf16 passthrough
    # assertions here — silently dragging the flow checkpoint (whose
    # fp8-preserve end-state kohya consumes directly) into a cast world
    # it must never enter. This pin defends the gate; P7 defends the
    # cast.
    assert SHIM_PATH.is_file()
    with _fake_torch_with_tensors() as torch:
        shim = _load_shim()
        sd = _mixed_scaled_fp8_sd(torch)
        out = shim.dequant_scaled_fp8_sd(sd)  # default target=None
        assert out["qkv"].dtype is torch.float8_e4m3fn
        assert out["mlp"].dtype is torch.float8_e5m2
        assert out["stray_fp8"].dtype is torch.float8_e4m3fn
        assert out["embed_tokens"].dtype is torch.float16, (
            "target=None (flow path) must NOT cast the fp16 embedding — "
            "the D-006/D-007 fp8-preserve contract; got: "
            f"{out['embed_tokens'].dtype}"
        )
        assert out["attn_bf16"].dtype is torch.bfloat16
        assert out["shared_step"].dtype is torch.int64
        assert out["bool_mask"].dtype is torch.bool
        assert "scaled_fp8" not in out
        assert not any(k.endswith(".scale_weight") for k in out)


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


def test_skip_te_env_wraps_clip_l_and_t5xxl_with_stubs():
    # D-007 two-pass memory design, BEHAVIORAL half (fake torch — runs and
    # is RED in torch-less CI). With KOHYA_SKIP_TEXT_ENCODERS=1 the wrapped
    # load_clip_l / load_t5xxl must return STUB objects WITHOUT invoking the
    # original loaders at all (harness stub counters: the loaders and the
    # module-global load_safetensors stay untouched — that is the whole
    # memory point: upstream flux_train_network.py loads both TEs
    # unconditionally at :141/:153, and with a complete cache +
    # network_train_unet_only LoRA get_models_for_text_encoding (:206-213)
    # returns None, so the TEs are never encoded through in the training
    # loop). Stub contract: real .dtype = torch.bfloat16 (the fp8 dtype
    # check at :157-160 reads it), eval()/to()/train() no-ops returning
    # self (the surface touched at :144/:154 and :246-247/:284-286), and
    # encode/forward raising RuntimeError containing "text encoder cache"
    # (clear guard if the cache is incomplete or the stub is misused). The
    # DiT path stays live: load_flow_model must still run the original
    # loader through the sd-fixup wrapper.
    assert SHIM_PATH.is_file()
    with _env(KOHYA_SKIP_TEXT_ENCODERS="1"), _fake_torch_installed() as fake_torch:
        shim = _load_shim()
        mod = _fake_flux_utils_with_counters()
        shim.patch_flux_utils(mod)

        clip = mod.load_clip_l("clip_l.safetensors", fake_torch.bfloat16, "cpu")
        assert mod.CALLS["clip_l"] == 0, (
            "KOHYA_SKIP_TEXT_ENCODERS=1: load_clip_l must return a stub "
            "WITHOUT invoking the original loader"
        )
        t5 = mod.load_t5xxl(
            "t5xxl_fp8_e4m3fn_scaled.safetensors", None, "cpu", disable_mmap=False
        )
        assert mod.CALLS["t5xxl"] == 0, (
            "KOHYA_SKIP_TEXT_ENCODERS=1: load_t5xxl must return a stub "
            "WITHOUT invoking the original loader"
        )
        assert mod.CALLS["safetensors"] == 0, (
            "stubbed TE loaders must not read anything from disk"
        )

        for label, stub in (("clip_l", clip), ("t5xxl", t5)):
            assert stub is not None, f"{label}: stub must be returned"
            assert stub.dtype is fake_torch.bfloat16, (
                f"{label}: stub .dtype must be the real bf16 dtype "
                "(flux_train_network.py:157-160 dtype check)"
            )
            assert stub.eval() is stub, f"{label}: eval() must be a self-returning no-op (:144/:154)"
            assert stub.to("cpu") is stub, f"{label}: to() must be a self-returning no-op (:246-247)"
            assert stub.train() is stub, f"{label}: train() must be a self-returning no-op"

        for label, stub in (("clip_l", clip), ("t5xxl", t5)):
            for method in ("encode", "forward"):
                try:
                    getattr(stub, method)("some text")
                except RuntimeError as exc:
                    assert "text encoder cache" in str(exc), (
                        f"{label}.{method}() must raise RuntimeError mentioning "
                        f"'text encoder cache'; got: {exc!r}"
                    )
                else:
                    raise AssertionError(
                        f"{label}.{method}() must raise RuntimeError — a stub that "
                        "silently 'encodes' would train on garbage instead of tripping "
                        "on an incomplete cache"
                    )

        # the gate must not over-stub: the DiT loader still runs (through the
        # sd-fixup wrapper) even with the env set.
        flow = mod.load_flow_model("flux1-dev-fp8.safetensors")
        assert mod.CALLS["flow"] == 1, (
            "KOHYA_SKIP_TEXT_ENCODERS stubs the TEXT ENCODERS only — the DiT "
            "(load_flow_model) must still load (the trainer holds DiT + ~3GB, D-007)"
        )
        assert "text_embedder.linear" in flow, flow


def test_skip_te_stub_exposes_empty_network_traversal_surface():
    # P9 (prod train-1792334532 — the FIRST time the trainer ever ran,
    # because the TE cache pass finally completed rc=0 with 53/53 npz):
    # kohya's network creation TRAVERSES the text encoder even under
    # network_train_unet_only — networks/lora_flux.py create_modules :819
    # runs `for name, module in root_module.named_modules():`, called
    # from :947 ("create LoRA for Text Encoder 1") — and the D-007 stub
    # exposed only dtype/eval/to/train/encode/forward, so the run died:
    # AttributeError: '_SkippedTextEncoder' object has no attribute
    # 'named_modules'. Intent: with a real nn.Module the traversal yields
    # modules and the TE-target match list decides which get LoRA; the
    # designed skip under network_train_unet_only means ZERO TE LoRA
    # modules, so the stub's traversal must yield NOTHING across the
    # LOAD-BEARING named_* surface (named_modules/named_parameters/
    # named_buffers/buffers return EMPTY iterables, state_dict() == {},
    # requires_grad_(...) returns self) — on BOTH stubs (clip_l and t5xxl
    # are traversed the same way). The
    # encode/forward tripwires stay exactly as pinned by
    # test_skip_te_env_wraps_clip_l_and_t5xxl_with_stubs — this pin ADDS
    # traversal surface, and re-checks one tripwire to prove the addition
    # did not swallow it. Landed RED-first against the pre-traversal
    # stub; PERMANENT guard — its red signature if the surface is ever
    # removed: AttributeError on named_modules, the exact prod frame.
    #
    # CONTRACT REVISION (prod train-1792365662, probe #7): the original
    # "parameters() EMPTY" assertion was WRONG — it predated the
    # training-loop evidence. train_network.py:1521-1525 (training-loop
    # start log) does params_itr = t_enc.parameters(); next(); next()
    # (skipping two "embeddings"); param_3rd = next(); then reads
    # param_3rd.dtype — so parameters() must yield AT LEAST 3 placeholder
    # params, each carrying .dtype == torch.bfloat16 (StopIteration here
    # kills the run at loop start). The security-relevant emptiness is
    # the named_* traversal (zero TE LoRA targets — "0 modules" CONFIRMED
    # in prod) and stays UNTOUCHED; buffers()/named_buffers() emptiness
    # also stands (the loop reads only parameters()).
    assert SHIM_PATH.is_file()
    with _env(KOHYA_SKIP_TEXT_ENCODERS="1"), _fake_torch_installed() as fake_torch:
        shim = _load_shim()
        mod = _fake_flux_utils_with_counters()
        shim.patch_flux_utils(mod)

        clip = mod.load_clip_l("clip_l.safetensors", fake_torch.bfloat16, "cpu")
        t5 = mod.load_t5xxl(
            "t5xxl_fp8_e4m3fn_scaled.safetensors", None, "cpu", disable_mmap=False
        )

        for label, stub in (("clip_l", clip), ("t5xxl", t5)):
            # the exact crashing frame (lora_flux.py:819): named_modules()
            # must be iterable AND yield nothing
            named_modules = stub.named_modules()
            assert iter(named_modules) is not None, (
                f"{label}: named_modules() must return an iterable "
                "(lora_flux.py:819 iterates it directly)"
            )
            assert list(named_modules) == [], (
                f"{label}: the traversal must yield NOTHING — under "
                "network_train_unet_only the designed skip means zero TE "
                f"LoRA modules; got: {list(stub.named_modules())}"
            )
            assert list(stub.named_parameters()) == [], label
            assert list(stub.named_buffers()) == [], label
            # REVISED by train-1792365662: parameters() must yield >=3
            # placeholder params with the bf16 dtype sentinel — the
            # training-loop start log reads the THIRD (see docstring).
            params = list(stub.parameters())
            assert len(params) >= 3, (
                f"{label}: parameters() must yield >=3 placeholder params — "
                "train_network.py:1521-1525 skips two 'embeddings' and "
                f"logs the third's dtype; got {len(params)} (fewer -> "
                "StopIteration kills the run at training-loop start)"
            )
            for i, p in enumerate(params[:3]):
                assert p.dtype is fake_torch.bfloat16, (
                    f"{label}: placeholder param #{i} must carry the real "
                    "bf16 dtype sentinel (read at train_network.py:1525); "
                    f"got: {getattr(p, 'dtype', None)!r}"
                )
            assert list(stub.buffers()) == [], label
            assert stub.state_dict() == {}, label
            assert stub.requires_grad_(False) is stub, (
                f"{label}: requires_grad_ must be a self-returning no-op "
                "(network creation freezes/presets grads across traversed modules)"
            )

        # tripwires UNCHANGED by the surface addition (one re-check per
        # method family; full tripwire contract stays pinned by the
        # stub-behavior test above):
        for label, stub in (("clip_l", clip), ("t5xxl", t5)):
            for method in ("encode", "forward"):
                try:
                    getattr(stub, method)("some text")
                except RuntimeError as exc:
                    assert "text encoder cache" in str(exc), (
                        f"{label}.{method}() tripwire must survive the "
                        f"traversal-surface addition; got: {exc!r}"
                    )
                else:
                    raise AssertionError(
                        f"{label}.{method}() must still raise RuntimeError "
                        "after the traversal surface was added"
                    )


def test_skip_te_stub_exposes_cpu_device_for_fp8_cast_gate():
    # P10 (prod train-1792365662, probe #7 — landed RED-first against the pre-device stub: AttributeError on
    # .device): this run MILESTONED — TE LoRA "0 modules" as designed
    # (the empty named_* traversal holds), DiT "create LoRA for FLUX all
    # blocks: 304 modules", optimizer prep started — then
    # train_network.py:1263 died on
    #     if t_enc.device.type != "cpu" and self.cast_text_encoder(args):
    # because the stub exposes no .device. Contract: the stub must expose
    # a device whose .type == "cpu" — the gate then skips the WHOLE fp8
    # cast + prepare_text_encoder_fp8 block, which is exactly right for a
    # weightless stub (nothing to cast; the on-disk cache already holds
    # the TE outputs). A real torch.device("cpu") is an acceptable
    # implementation (the fake torch carries a device stand-in for
    # it) — the pin asserts only the observable stub.device.type == "cpu"
    # on BOTH stubs. Landed RED-first; PERMANENT guard — red signature if
    # removed: the exact prod AttributeError on .device at :1263.
    assert SHIM_PATH.is_file()
    with _env(KOHYA_SKIP_TEXT_ENCODERS="1"), _fake_torch_installed():
        shim = _load_shim()
        mod = _fake_flux_utils_with_counters()
        shim.patch_flux_utils(mod)

        clip = mod.load_clip_l("clip_l.safetensors", None, "cpu")
        t5 = mod.load_t5xxl(
            "t5xxl_fp8_e4m3fn_scaled.safetensors", None, "cpu", disable_mmap=False
        )

        for label, stub in (("clip_l", clip), ("t5xxl", t5)):
            dev = stub.device
            assert getattr(dev, "type", None) == "cpu", (
                f"{label}: stub.device.type must be 'cpu' — "
                "train_network.py:1263 gates the fp8 cast / "
                "prepare_text_encoder_fp8 block on it (cpu -> skipped, the "
                "correct path for a weightless stub); got: "
                f"{getattr(dev, 'type', None)!r} (device: {dev!r})"
            )


def test_skip_te_env_absent_or_empty_uses_real_loaders():
    # D-007 gate polarity (permanent regression guard for the stubbing):
    # with the env ABSENT or EMPTY the real loaders must pass through
    # untouched — load_clip_l is not stubbed (invoked, original return
    # value), and load_t5xxl keeps the existing sd-fixup behavior (original
    # invoked). This pins that the gate reads "present AND non-empty", so
    # no ambient empty variable can silently stub the TEs on a non-two-pass
    # run. The gate now EXISTS in scripts/kohya_compat_shim.py, so this
    # polarity is enforced BY THE GATE, not by construction: env absent or
    # empty must NEVER stub the TEs — any future widening of the trigger
    # (e.g. "key present at all", or a truthiness-flipped check) flips
    # this pin red.
    assert SHIM_PATH.is_file()
    for env_value in (None, ""):
        with _env(KOHYA_SKIP_TEXT_ENCODERS=env_value), _fake_torch_installed() as fake_torch:
            shim = _load_shim()
            mod = _fake_flux_utils_with_counters()
            shim.patch_flux_utils(mod)

            clip = mod.load_clip_l("clip_l.safetensors", fake_torch.bfloat16, "cpu")
            assert mod.CALLS["clip_l"] == 1, (
                f"KOHYA_SKIP_TEXT_ENCODERS={env_value!r}: the original "
                "load_clip_l must be invoked (no stubbing without the gate)"
            )
            assert clip == {"clip_l_original": "clip_l.safetensors"}, clip

            t5 = mod.load_t5xxl("t5.safetensors", None, "cpu")
            assert mod.CALLS["t5xxl"] == 1, (
                f"KOHYA_SKIP_TEXT_ENCODERS={env_value!r}: the original "
                "load_t5xxl must be invoked (real loaders pass through)"
            )
            assert t5 == {"t5xxl_original": "t5.safetensors"}, t5


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
