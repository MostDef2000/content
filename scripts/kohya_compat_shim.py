"""kohya sd-scripts compat shim (spec 010 prod crash).

Root cause (prod logs + upstream kohya-ss/sd-scripts @690ea7f9,
library/flux_utils.py load_flow_model): the ComfyUI `model.diffusion_model.`
prefix strip loop BREAKS on the first key without the prefix. In Comfy-Org's
flux1-dev-fp8.safetensors the unprefixed scaled-fp8 marker `scaled_fp8` sits
EARLY in key order — not first (keys[0] is prefixed: analyze_checkpoint_state
matched its bare double_blocks./single_blocks. patterns and proceeded; a
marker-first file would have died with "max() arg is an empty sequence"
before any load). The loop therefore breaks mid-dict: keys before the marker
get renamed, the REMAINING prefixed keys stay unrenamed, the model built
under torch.device("meta") receives only part of its weights (stays
partially meta) and training dies with "NotImplementedError: Cannot copy out
of meta tensor" at unet.to("cpu").
The ostris scaled-fp8 T5 (t5xxl_fp8_e4m3fn_scaled.safetensors) additionally
carries per-tensor `*.scale_weight` keys kohya ignores — raw fp8 values get
assigned as weights (numerically wrong text-encoder outputs).

Memory profile (two prod SIGKILLs, both silent deaths by the OOM signature):
train-1792246674 — full-tensor fp32 intermediates on the ~17GB fp8 sd piled
up ~90GB peak; fixed by bounded chunks (fp32 transient ~16MB flat /
~256-row per-channel) plus CONSUMING mode (`in_place=True`: tensors are
popped out of the loader's dict as processed, fp8 memory frees
progressively). train-1792249442 — even the ~24GB bf16 RESULT cannot fit
the box budget (< ~26GB): fixed by REAL re-quantization — the flow model
dequantizes back into the checkpoint's own fp8 dtype (1:1 fp8→fp8 with
progressive release, flow-phase peak ≈ the DiT-sd size, 1:1), while the T5 runs
explicitly bf16 (T5-phase peak ≈ DiT-sd + ~1.7GB CLIP + ~9.4GB T5 — ~23.1GB
for a ~12GB DiT-sd, ~28.1GB for ~17GB; estimates — the trainer's memory
preflight measures the real checkpoint bytes and available memory). The flow end-state is the correctly-rounded SINGLE-rounding
fp8 quantization of the true values — the same end-state as kohya's
PLAIN-fp8 fp8_base flow for a raw fp8 checkpoint. It is NOT bit-identical
to the old bf16-materialize + kohya-cast path (which never completed a run
anyway): that was double rounding e4m3(bf16(x)) vs this single rounding
e4m3(x) — the two differ by ≤1 fp8 ulp only in double-rounding ties
(~1/32 of normal-binade elements). The kohya cast at
flux_train_network.py:129, if it runs at all, touches only the non-fp8
remainder (same-dtype Tensor.to() allocates nothing — worst case ≈
+~2.5GB, not a full-DiT spike).

D-007 TE-skip gate (two-pass memory design): the trainer invocation sets
KOHYA_SKIP_TEXT_ENCODERS=1 (train_lora.sh env-prefixes the accelerate
launch; the TE cache pass runs WITHOUT it). With the env present and
NON-EMPTY, patch_flux_utils additionally wraps flux_utils.load_clip_l and
flux_utils.load_t5xxl to return a STUB instead of loading from disk —
upstream flux_train_network.py loads CLIP-L (:141) and T5XXL (:153)
unconditionally even with a complete on-disk cache, and with
network_train_unet_only LoRA get_models_for_text_encoding (:206-213)
returns None, so the TEs are pure dead weight in the trainer (~11GB RAM).
The stub carries the surface the cache path touches (real bf16 .dtype,
self-returning eval/to/train) and a loud encode/forward tripwire; the DiT
loader (load_flow_model) stays wrapped and live under the gate either way.

This shim is installed into the kohya venv's site-packages by the bootstrap
in scripts/train_lora.sh; a sitecustomize.py next to it imports this module
and calls install() in EVERY python process of the venv, and install()
registers a sys.meta_path finder that patches library.flux_utils right after
its first import — wrapping BOTH sd-loading entry points (load_flow_model
and load_t5xxl; clip_l/ae are untouched outside the D-007 gate) so every
loader hands out a fixed state dict.

torch is imported lazily (inside the functions that need it): CI parses and
imports this file without torch installed.
"""

from __future__ import annotations

import importlib.util
import os
import sys

__all__ = [
    "strip_comfyui_prefix",
    "dequant_scaled_fp8_sd",
    "apply_sd_fixups",
    "patch_flux_utils",
    "install",
]

# ComfyUI convention: diffusion weights are stored under this prefix; the
# bare kohya model expects the suffix names.
_COMFYUI_PREFIX = "model.diffusion_model."

# ComfyUI scaled-fp8 marker key (top-level; value dtype varies between
# checkpoints — uint8/float/bfloat16 — and is irrelevant).
_SCALED_FP8_MARKER = "scaled_fp8"

# Per-tensor scale suffix paired with a fp8 tensor key.
_SCALE_SUFFIX = ".scale_weight"

# Dequant target dtype: the trainer runs bf16 mixed precision, and kohya
# builds its model in bf16 — hand it dequantized bf16 tensors.
_TARGET_DTYPE_NAME = "bfloat16"

# Chunked fp32 math (memory fix, train-1792246674): the fp32 transient must
# stay bounded — ~4M flat elements = 16MB per chunk, ~256 rows per chunk for
# per-channel (out, 1) scales. Numerics are identical to the full-tensor
# fp32 roundtrip: multiply and cast are elementwise ops.
_DEQUANT_CHUNK_ELEMS = 4_000_000
_DEQUANT_CHUNK_ROWS = 256

# D-007 TE-skip gate (two-pass memory design): the TRAINER invocation sets
# this env var (train_lora.sh env-prefixes KOHYA_SKIP_TEXT_ENCODERS=1 onto
# the accelerate launch); the TE cache pass runs WITHOUT it and needs the
# REAL loaders. Read at patch time (call time), not import time.
_SKIP_TEXT_ENCODERS_ENV = "KOHYA_SKIP_TEXT_ENCODERS"


def _skip_text_encoders():
    """True when the gate is present AND non-empty: an ambient EMPTY variable
    must never stub the TEs on a non-two-pass run (polarity pinned by
    tests/test_kohya_compat_shim.py::test_skip_te_env_absent_or_empty_uses_real_loaders)."""
    return bool(os.environ.get(_SKIP_TEXT_ENCODERS_ENV, ""))


class _SkippedTextEncoder:
    """D-007 TE-skip stub: stands in for a text encoder the two-pass trainer
    run never loads. Upstream flux_train_network.py still LOADS both TEs
    unconditionally (:141 CLIP-L, :153 T5XXL) even with a complete on-disk
    cache; with network_train_unet_only LoRA they are never encoded through
    (get_models_for_text_encoding :206-213 returns None), so the stub exposes
    exactly the surface the cache path touches:
      * .dtype — a REAL torch dtype (bfloat16): the fp8 dtype check at
        flux_train_network.py:157-160 reads it;
      * eval()/to()/train() — no-ops returning self (:144/:154 and
        :246-247/:284-286);
      * encode()/forward() — LOUD tripwire: RuntimeError mentioning the
        text encoder cache, so an incomplete cache or stub misuse fails the
        run instead of silently training on garbage.
    torch is imported lazily (this module must stay importable without it)."""

    def __init__(self, name):
        import torch  # lazy: real bf16 dtype sentinel

        self.name = name
        self.dtype = torch.bfloat16

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def train(self, mode=True):
        return self

    def encode(self, *args, **kwargs):
        raise RuntimeError(
            f"KOHYA_SKIP_TEXT_ENCODERS: {self.name} is stubbed — the text encoder "
            "cache must be complete before training (run the TE cache pass "
            "tools/cache_text_encoder_outputs.py first, or unset "
            "KOHYA_SKIP_TEXT_ENCODERS to load the real encoders)"
        )

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            f"KOHYA_SKIP_TEXT_ENCODERS: {self.name} is stubbed — the text encoder "
            "cache must be complete before training (run the TE cache pass "
            "tools/cache_text_encoder_outputs.py first, or unset "
            "KOHYA_SKIP_TEXT_ENCODERS to load the real encoders)"
        )


def strip_comfyui_prefix(sd, in_place=False):
    """Strip the ComfyUI `model.diffusion_model.` prefix from EVERY key that
    carries it; keys without the prefix pass through unchanged.

    Upstream's loop breaks on the first unprefixed key it hits (in the
    Comfy-Org checkpoint that is the scaled_fp8 marker, sitting early in
    key order — some earlier keys DO get renamed, the rest stay unrenamed,
    which is exactly why the prod model ended up partially meta) — here
    every key is inspected independently and no loop ever breaks early.
    Default returns a NEW dict (input not mutated); in_place=True renames
    keys in the SAME dict object (pop old key, assign new) and returns it —
    the memory fix's consuming mode for the loader's ~17GB dict.
    """
    if in_place:
        for key in [k for k in sd if k.startswith(_COMFYUI_PREFIX)]:
            sd[key[len(_COMFYUI_PREFIX):]] = sd.pop(key)
        return sd
    out = {}
    for key, value in sd.items():
        if key.startswith(_COMFYUI_PREFIX):
            out[key[len(_COMFYUI_PREFIX):]] = value
        else:
            out[key] = value
    return out


def _fp8_dtypes(torch):
    """The two fp8 dtypes torch ships (module passed in to keep the torch
    import lazy at the call site)."""
    return (torch.float8_e4m3fn, torch.float8_e5m2)


def _to_target(torch, tensor, target=None):
    """Cast a stray fp8 tensor (no paired scale — its stored values ARE the
    true values): with an explicit torch dtype, fp32-roundtrip into it (fp8
    -> bf16 directly can round differently than through fp32); with
    target=None (fp8-preserve mode) the tensor keeps its own fp8 dtype —
    nothing to re-quantize, returned as-is."""
    if target is None:
        return tensor
    return tensor.to(torch.float32).to(target)


def _dequant_tensor(torch, tensor, scale, target=None):
    """Dequantize ONE fp8 tensor by its scale into a preallocated output
    tensor of the input's shape, in bounded chunks for the scalar and
    (out, 1) scale shapes. target=None re-quantizes into the INPUT's own
    dtype (flow path: each tensor keeps its own fp8 dtype — e4m3fn AND
    e5m2 inputs alike); a torch dtype object (e.g. torch.bfloat16 for the
    T5 path) casts the result there instead. Numerics are identical per
    element to the full-tensor `(tensor.fp32 * scale.fp32).cast` roundtrip
    (multiply and cast are elementwise — chunk boundaries cannot shift
    rounding), but the fp32 transient is ~16MB instead of a full-tensor
    copy (train-1792246674 OOM). EXOTIC scale shapes fall through to the
    plain full-tensor formula — explicitly NOT chunked (correctness first,
    they do not occur in the shipped checkpoints)."""
    out_dtype = tensor.dtype if target is None else target
    scale_fp32 = scale.to(torch.float32)
    out = torch.empty(tensor.shape, dtype=out_dtype)
    if scale_fp32.numel() == 1:
        # scalar scale (0-dim, (1,) or (1,1) — all encode one per-tensor
        # multiplier): flat chunks; reshape(()) normalizes every 1-element
        # shape to a true scalar, and the elementwise product is identical
        # to the old full-tensor broadcast in all three shapes.
        flat = tensor.reshape(-1)
        out_flat = out.view(-1)
        scalar = scale_fp32.reshape(())
        for r0 in range(0, flat.numel(), _DEQUANT_CHUNK_ELEMS):
            r1 = min(r0 + _DEQUANT_CHUNK_ELEMS, flat.numel())
            out_flat[r0:r1] = (flat[r0:r1].to(torch.float32) * scalar).to(out_dtype)
        return out
    if (
        tensor.dim() == 2
        and scale_fp32.dim() == 2
        and scale_fp32.shape[1] == 1
        and scale_fp32.shape[0] > 1
        and scale_fp32.shape[0] == tensor.shape[0]
    ):
        # per-channel (out, 1): row chunks; a (r, 1) slice broadcasts
        # against the (r, C) row block directly — elementwise identical to
        # the old full-tensor formula.
        for r0 in range(0, tensor.shape[0], _DEQUANT_CHUNK_ROWS):
            r1 = min(r0 + _DEQUANT_CHUNK_ROWS, tensor.shape[0])
            out[r0:r1] = (
                tensor[r0:r1].to(torch.float32) * scale_fp32[r0:r1]
            ).to(out_dtype)
        return out
    # any other broadcast shape: keep the plain (non-chunked) formula
    return (tensor.to(torch.float32) * scale_fp32).to(out_dtype)


def dequant_scaled_fp8_sd(sd, in_place=False, target=None):
    """Apply the scaled-fp8 fixups to a state dict.

    * every `<tensor>.scale_weight` pair: dequantize
      `(tensor.fp32 * scale.fp32).cast` under the tensor key (scales may be
      0-dim scalars or per-channel shapes like (out, 1) — broadcast
      semantics preserved exactly; math runs in bounded fp32 chunks).
      target=None re-quantizes into the INPUT tensor's own dtype (flow
      path: fp8-preserve, e4m3fn and e5m2 alike); a torch dtype object
      casts the result there (T5 path: torch.bfloat16);
    * the top-level `scaled_fp8` marker is dropped regardless of its value;
    * a pure fp8 tensor WITHOUT a paired scale follows the same target:
      kept in its own fp8 dtype when target=None, fp32-roundtripped into
      the target dtype otherwise (ComfyUI-style checkpoints store some
      tensors unscaled);
    * everything else passes through unchanged.

    Default (copy) mode returns a NEW dict, input untouched. in_place=True
    is the consuming mode (train-1792246674 OOM fix): each tensor and its
    scale are POPPED from the input as processed — fp8 memory frees
    progressively; the marker and orphan scale keys are dropped from the
    input too, and plain/stray-fp8 remainder tensors move into the result.
    """
    import torch  # lazy: this module must import (and be installed) without torch

    out = {}

    # Pass 1: dequantize every paired tensor, drop its scale key.
    paired_bases = set()
    for key in [k for k in sd if k.endswith(_SCALE_SUFFIX)]:
        base = key[: -len(_SCALE_SUFFIX)]
        if base not in sd:
            if in_place:
                del sd[key]  # unpaired scale key has nothing to scale — drop it
            continue
        scale = sd.pop(key) if in_place else sd[key]
        tensor = sd.pop(base) if in_place else sd[base]
        out[base] = _dequant_tensor(torch, tensor, scale, target)
        paired_bases.add(base)

    # Pass 2: every other key — drop the marker, handle stray fp8, pass the
    # rest through untouched.
    for key in list(sd):
        value = sd.pop(key) if in_place else sd[key]
        if key == _SCALED_FP8_MARKER or key.endswith(_SCALE_SUFFIX) or key in paired_bases:
            continue
        if isinstance(value, torch.Tensor) and value.dtype in _fp8_dtypes(torch):
            out[key] = _to_target(torch, value, target)
        else:
            out[key] = value
    return out


def apply_sd_fixups(sd, in_place=False, target=None):
    """The full sd fixup pipeline for ComfyUI-style checkpoints: prefix strip
    FIRST (tensor and its .scale_weight pair share the prefix, so they stay
    paired under the final name), then the scaled-fp8 dequant. in_place=True
    threads the consuming mode through the chain (memory fix); target=None
    re-quantizes fp8 into each tensor's own dtype (flow path), a torch dtype
    object (torch.bfloat16) casts results there (T5 path)."""
    return dequant_scaled_fp8_sd(
        strip_comfyui_prefix(sd, in_place=in_place), in_place=in_place, target=target
    )


def patch_flux_utils(mod):
    """Patch a freshly imported library.flux_utils module in place.

    BOTH sd-loading entry points are wrapped with the same swap-and-restore
    pattern: mod.load_flow_model AND mod.load_t5xxl (upstream loads the T5
    via load_t5xxl AFTER load_flow_model returned, so wrapping only the flow
    path would leave the scaled-fp8 T5 state dict raw). For the duration of
    each call, mod.load_safetensors is replaced by a shim that runs its
    return value through apply_sd_fixups — the REAL (unmodified) kohya
    loaders receive an already-fixed state dict. The fixups run in CONSUMING
    mode (explicit `in_place=True` keyword) with DISTINCT targets per
    loader:
      * load_flow_model -> target=None: REAL re-quantization into the
        checkpoint's own fp8 dtype — the correctly-rounded SINGLE-rounding
        quantization of the true values, i.e. the same end-state as
        kohya's PLAIN-fp8 fp8_base flow for a raw fp8 checkpoint (NOT the
        old bf16-materialize + kohya-cast path: that was double rounding
        e4m3(bf16(x)) vs this single rounding e4m3(x) — ≤1 fp8 ulp apart
        only in double-rounding ties, ~1/32 of normal-binade elements, and
        that old path never completed a run). The kohya cast at
        flux_train_network.py:129, if it runs at all, touches only the
        non-fp8 remainder (same-dtype Tensor.to() allocates nothing); this
        eliminates the ~24GB bf16 materialization — flow-phase
        peak ≈ the DiT-sd size (1:1 fp8→fp8, progressive release;
        train-1792249442: the box budget is < ~26GB, a 24GB bf16 result
        cannot fit);
      * load_t5xxl -> torch.bfloat16: the TE cache must hold bf16 caption
        embeddings — an fp8 T5 would bake fp8 quantization noise into them
        (T5-phase peak ≈ DiT-sd + ~1.7GB CLIP + ~9.4GB T5 — ~23.1GB for a
        ~12GB DiT-sd, ~28.1GB for ~17GB; estimates — the trainer's memory
        preflight measures the real checkpoint bytes and available memory).
    The original load_safetensors is restored in a finally block, win or
    lose; load_clip_l and load_ae stay untouched (no scaled-fp8 there) —
    except under the D-007 gate below, which stubs BOTH text-encoder loaders.

    D-007 TE-skip gate: when KOHYA_SKIP_TEXT_ENCODERS is present and
    NON-EMPTY (the two-pass trainer invocation sets it; the TE cache pass
    runs WITHOUT it and gets the real loaders), load_clip_l AND load_t5xxl
    are additionally wrapped to return a _SkippedTextEncoder WITHOUT
    invoking the underlying loaders at all — nothing is read from disk, no
    sd-fixup swap happens for the TEs. The DiT path (load_flow_model) stays
    wrapped and live either way: the trainer still holds DiT + ~3GB (D-007).
    """
    import torch  # lazy: torch.bfloat16 target for the T5 path

    original_loader = mod.load_safetensors

    def _make_patched(original, target):
        def _fixed_load_safetensors(*args, **kwargs):
            return apply_sd_fixups(original_loader(*args, **kwargs), in_place=True, target=target)

        def _patched(*args, **kwargs):
            mod.load_safetensors = _fixed_load_safetensors
            try:
                return original(*args, **kwargs)
            finally:
                mod.load_safetensors = original_loader

        return _patched

    # flow: fp8-preserve (re-quantize into the checkpoint's own fp8 grid);
    # T5: explicit bf16 (TE-cache quality).
    mod.load_flow_model = _make_patched(mod.load_flow_model, None)
    mod.load_t5xxl = _make_patched(mod.load_t5xxl, torch.bfloat16)

    if _skip_text_encoders():
        # D-007 two-pass memory design: with the complete TE cache from pass 1
        # (tools/cache_text_encoder_outputs.py) the trainer must not pay for
        # the text encoders — upstream loads them unconditionally (:141/:153)
        # and with network_train_unet_only LoRA never encodes through them.
        # Both TE loaders return the stub WITHOUT touching disk (and without
        # the sd-fixup swap); the DiT loader above stays live.
        def _make_te_stub_loader(name):
            def _stub_loader(*args, **kwargs):
                return _SkippedTextEncoder(name)

            return _stub_loader

        mod.load_clip_l = _make_te_stub_loader("clip_l")
        mod.load_t5xxl = _make_te_stub_loader("t5xxl")
    return mod


class _FluxUtilsPatchingLoader:
    """Loader wrapper: run the REAL exec_module, then patch the module."""

    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        create = getattr(self._inner, "create_module", None)
        if create is None:
            return None
        return create(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        patch_flux_utils(module)


class _FluxUtilsPatchFinder:
    """sys.meta_path finder intercepting exactly library.flux_utils: the
    real spec is produced by the rest of the meta path, its loader wrapped
    so the module is patched immediately after it executes."""

    target = "library.flux_utils"

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.target:
            return None
        real_spec = None
        for finder in sys.meta_path:
            if finder is self:
                continue  # do not recurse into ourselves
            found = finder.find_spec(fullname, path, target)
            if found is not None:
                real_spec = found
                break
        if real_spec is None or real_spec.loader is None:
            return real_spec  # namespace/no-loader: let normal machinery run
        real_spec.loader = _FluxUtilsPatchingLoader(real_spec.loader)
        return real_spec


_installed = False


def install():
    """Register the meta_path finder (idempotent: repeated calls must not
    stack duplicate finders)."""
    global _installed
    if _installed:
        return
    if not any(isinstance(finder, _FluxUtilsPatchFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _FluxUtilsPatchFinder())
    _installed = True
