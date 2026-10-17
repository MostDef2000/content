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

This shim is installed into the kohya venv's site-packages by the bootstrap
in scripts/train_lora.sh; a sitecustomize.py next to it imports this module
and calls install() in EVERY python process of the venv, and install()
registers a sys.meta_path finder that patches library.flux_utils right after
its first import — wrapping BOTH sd-loading entry points (load_flow_model
and load_t5xxl; clip_l/ae are untouched) so every loader hands out a fixed
state dict.

torch is imported lazily (inside the functions that need it): CI parses and
imports this file without torch installed.
"""

from __future__ import annotations

import importlib.util
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


def strip_comfyui_prefix(sd):
    """Strip the ComfyUI `model.diffusion_model.` prefix from EVERY key that
    carries it; keys without the prefix pass through unchanged.

    Upstream's loop breaks on the first unprefixed key it hits (in the
    Comfy-Org checkpoint that is the scaled_fp8 marker, sitting early in
    key order — some earlier keys DO get renamed, the rest stay unrenamed,
    which is exactly why the prod model ended up partially meta) — here
    every key is inspected independently and no loop ever breaks early.
    Returns a NEW dict; the input is not mutated.
    """
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


def _to_target(torch, tensor):
    """fp32-roundtrip cast to the trainer dtype (fp8 -> bf16 directly can
    round differently than through fp32; go through fp32 explicitly)."""
    return tensor.to(torch.float32).to(getattr(torch, _TARGET_DTYPE_NAME))


def dequant_scaled_fp8_sd(sd):
    """Apply the scaled-fp8 fixups to a state dict.

    * every `<tensor>.scale_weight` pair: dequantize
      `(tensor.fp32 * scale.fp32).bf16` under the tensor key (scales may be
      0-dim scalars or per-channel shapes like (out, 1) — plain torch
      broadcast rules apply);
    * the top-level `scaled_fp8` marker is dropped regardless of its value;
    * a pure fp8 tensor WITHOUT a paired scale is cast to bf16 as-is
      (ComfyUI-style checkpoints store some tensors unscaled);
    * everything else passes through unchanged.

    Returns a NEW dict; the input is not mutated.
    """
    import torch  # lazy: this module must import (and be installed) without torch

    target = getattr(torch, _TARGET_DTYPE_NAME)
    out = {}

    # Pass 1: dequantize every paired tensor, drop its scale key.
    paired_bases = set()
    for key, value in sd.items():
        if not key.endswith(_SCALE_SUFFIX):
            continue
        base = key[: -len(_SCALE_SUFFIX)]
        if base in sd:
            tensor = sd[base]
            out[base] = (tensor.to(torch.float32) * value.to(torch.float32)).to(target)
            paired_bases.add(base)
        # An unpaired .scale_weight key has nothing to scale — drop it.

    # Pass 2: every other key — drop the marker, cast stray fp8, pass the
    # rest through untouched.
    for key, value in sd.items():
        if key == _SCALED_FP8_MARKER:
            continue
        if key.endswith(_SCALE_SUFFIX):
            continue
        if key in paired_bases:
            continue
        if isinstance(value, torch.Tensor) and value.dtype in _fp8_dtypes(torch):
            out[key] = _to_target(torch, value)
        else:
            out[key] = value
    return out


def apply_sd_fixups(sd):
    """The full sd fixup pipeline for ComfyUI-style checkpoints: prefix strip
    FIRST (tensor and its .scale_weight pair share the prefix, so they stay
    paired under the final name), then the scaled-fp8 dequant."""
    return dequant_scaled_fp8_sd(strip_comfyui_prefix(sd))


def patch_flux_utils(mod):
    """Patch a freshly imported library.flux_utils module in place.

    BOTH sd-loading entry points are wrapped with the same swap-and-restore
    pattern: mod.load_flow_model AND mod.load_t5xxl (upstream loads the T5
    via load_t5xxl AFTER load_flow_model returned, so wrapping only the flow
    path would leave the scaled-fp8 T5 state dict raw). For the duration of
    each call, mod.load_safetensors is replaced by a shim that runs its
    return value through apply_sd_fixups — the REAL (unmodified) kohya
    loaders receive an already-fixed state dict. The original
    load_safetensors is restored in a finally block, win or lose; load_clip_l
    and load_ae stay untouched (no scaled-fp8 there).
    """
    original_loader = mod.load_safetensors

    def _fixed_load_safetensors(*args, **kwargs):
        return apply_sd_fixups(original_loader(*args, **kwargs))

    def _make_patched(original):
        def _patched(*args, **kwargs):
            mod.load_safetensors = _fixed_load_safetensors
            try:
                return original(*args, **kwargs)
            finally:
                mod.load_safetensors = original_loader

        return _patched

    mod.load_flow_model = _make_patched(mod.load_flow_model)
    mod.load_t5xxl = _make_patched(mod.load_t5xxl)
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
