#!/usr/bin/env python3
"""Cache-pass wrapper for the sd-scripts FLUX TE cache tool.

Spec 010 D-007 follow-up; production failure train-1792313236.

WHY THIS EXISTS: upstream tools/cache_text_encoder_outputs.py (kohya-ss/
sd-scripts @690ea7f9) opens cache_to_disk with

    assert is_sdxl or args.weighted_captions is None   # :62

while its own parser spells the flag as `--weighted_captions` with
action="store_true", default=False (~:217). For the FLUX branch (--flux,
not --sdxl) that assert can therefore NEVER pass from the CLI or a TOML
config: parse_args yields False (a store_true toggle is a plain bool; TOML
has no null), so the cache pass dies rc=1 before writing anything — the
observed prod failure train-1792313236. This is an upstream defect; the
repo-side fix is this wrapper (NO upstream files are modified).

WHAT IT DOES: replicates the tool's __main__ flow —

    parser = setup_parser(); args = parser.parse_args(tool_flags)
    args = args_util.read_config_from_file(args, parser)   # tool: import library.args as args_util (module top, :23)
    cache_to_disk(args)

— with the weighted_captions insertion between read_config_from_file and
cache_to_disk (`args.weighted_captions = None`, the only FLUX-passable
value for the :62 assert — the FIRST upstream defect) plus the compat
patch for the SECOND upstream defect and the loader wraps for the THIRD
upstream defect, both documented below. Downstream
is_weighted consumers are truthiness-only checks (strategy_sdxl.py:270,
strategy_lumina.py:241 @690ea7f9), so None is safe.

SECOND upstream defect (prod train-1792323801 — the weighted_captions fix
let the cache pass reach the NEXT landmine): the tool's helper
tools/cache_latents.py:33 calls
`flux_utils.check_flux_state_dict_diffusers_schnell(path)` and 3-unpacks
the result — an OLD API the tool predates the rename of: library/flux_utils
@690ea7f9 defines only `analyze_checkpoint_state(path)` returning a 4-tuple
(is_diffusers, is_schnell, (n_double, n_single), checkpoint paths), the
spelling the trainer itself uses (flux_train_network.py:171). The wrapper
therefore compat-patches the loaded tool's flux_utils module object right after
exec_module: if it lacks `check_flux_state_dict_diffusers_schnell`, an
adapter `path -> analyze_checkpoint_state(path)[:3]` is added onto THAT
object (module-attribute patch — visible to cache_latents through the same
module object; the [:3] slice restores the old 3-tuple contract, so
is_schnell stays at index 1). ONLY-IF-MISSING: an existing attribute is
never overridden — if upstream someday fixes the rename, the tool's own
implementation must win.

THIRD upstream defect (prod train-1792329792 — the rename adapter let the
cache pass reach the NEXT landmine, the first CLIP forward): the tool loads
the text encoders via `flux_utils.load_clip_l(...)`/`load_t5xxl(...)`
(module-attribute calls, tools/cache_text_encoder_outputs.py:134/:138);
library/flux_utils @690ea7f9 builds both models under `init_empty_weights()`
(:345/:400), which does NOT meta-ify non-persistent buffers — position_ids
is created on CPU — then `load_state_dict(..., assign=True)` (:353/:408)
places the PARAMETERS on accelerator.device, and no module `.to()` follows:
the CPU buffer survives into the forward pass and dies on a device mismatch.
The wrapper therefore sanitizes at load time: it wraps the tool's
flux_utils load_clip_l/load_t5xxl attributes (when present) so every
returned model gets its mismatched non-meta buffers re-materialized onto
the parameter device (`ref = next(model.parameters()).device`;
`m._buffers[name] = buf.to(ref)`). ALWAYS-ACTIVE-WHEN-PRESENT (unlike the
only-if-missing defect-#2 adapter): it is a wrap of an existing function
and a harmless no-op if upstream ever fixes the buffer placement (buffers
already on the parameter device are left untouched — the _buffers entry is
not rewritten).

USAGE:
    python3 cache_pass_wrapper.py TOOL_PATH [TOOL_FLAGS...]

The tool is loaded as a module (importlib.util.spec_from_file_location) —
never executed as a script — and its directory is placed at sys.path[0]
first, mirroring the environment of a direct `python tools/<tool>.py` run
(the tool does a bare `from cache_latents import ...`, which resolves
against the script's own directory).

FAILURE SEMANTICS (the train_lora.sh fail-fast depends on both):
  * invoked with NO arguments → usage message on stderr, non-zero exit,
    FAIL FAST: no tool module is imported or executed;
  * exceptions from cache_to_disk are deliberately NOT caught — they
    propagate (visible traceback, non-zero exit), so an incomplete/failed
    cache pass can never be mistaken for a success and the trainer is
    never launched on a broken cache.

stdlib only; no torch import at module level (the tool imports torch
itself — run this wrapper with the kohya venv python).
"""

from __future__ import annotations

import importlib.util
import os
import sys

_TOOL_MODULE_NAME = "_sdscripts_cache_tool"

_USAGE = (
    "usage: cache_pass_wrapper.py TOOL_PATH [TOOL_FLAGS...]\n"
    "Loads the sd-scripts TE cache tool TOOL_PATH as a module, replicates "
    "its __main__ flow (setup_parser -> parse_args -> "
    "args_util.read_config_from_file), forces args.weighted_captions = None "
    "(upstream defect @690ea7f9: --weighted_captions is store_true/"
    "default=False while cache_to_disk asserts `is_sdxl or "
    "args.weighted_captions is None` at :62 — the FLUX branch is "
    "unreachable from CLI/TOML otherwise; train-1792313236) and calls "
    "cache_to_disk(args)."
)


def _sanitize_module_buffers(model):
    """Defect-#3 contract (train-1792329792): re-materialize every
    non-meta buffer that sits off the parameter device onto that device.
    Matching buffers are left untouched (the _buffers entry is NOT
    rewritten), so an already-correct model is a strict no-op (P6)."""
    ref = next(model.parameters()).device
    for m in model.modules():
        for name, buf in list(m._buffers.items()):
            if buf is not None and not buf.is_meta and buf.device != ref:
                m._buffers[name] = buf.to(ref)


def _wrap_loader_with_buffer_sanitize(orig):
    """Defect-#3 wrap (train-1792329792): call the original flux_utils
    loader, then sanitize the returned model's buffers onto the parameter
    device before the tool ever sees it."""

    def _wrapped_loader(*args, **kwargs):
        model = orig(*args, **kwargs)
        _sanitize_module_buffers(model)
        return model

    return _wrapped_loader


def main(argv):
    # P1c fail-fast: validate the invocation BEFORE touching any tool
    # module — a no-argument run must not import/exec anything.
    if len(argv) < 2:
        print(_USAGE, file=sys.stderr)
        return 2

    tool_path = argv[1]
    tool_flags = argv[2:]

    # The tool's own directory at sys.path[0]: a direct
    # `python tools/<tool>.py` run puts tools/ first on sys.path, and the
    # tool relies on it (bare `from cache_latents import ...`). The
    # editable sd-scripts install exposes the repo root (library/...), not
    # tools/, so the wrapper must restore that lookup path itself.
    tool_dir = os.path.dirname(os.path.abspath(tool_path))
    if tool_dir and tool_dir not in sys.path:
        sys.path.insert(0, tool_dir)

    spec = importlib.util.spec_from_file_location(_TOOL_MODULE_NAME, tool_path)
    if spec is None or spec.loader is None:
        print(f"cache_pass_wrapper: cannot load tool module from {tool_path}", file=sys.stderr)
        return 2
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)

    # SECOND upstream defect compat patch (prod train-1792323801): the
    # tool's helper tools/cache_latents.py:33 calls
    # `flux_utils.check_flux_state_dict_diffusers_schnell(path)` — a
    # 3-tuple API (is_diffusers, is_schnell, (n_double, n_single)) that the
    # tool predates: library/flux_utils @690ea7f9 only defines the RENAMED
    # `analyze_checkpoint_state(path)` returning a 4-tuple
    # (is_diffusers, is_schnell, (n_double, n_single), checkpoint paths),
    # the spelling the trainer itself uses (flux_train_network.py:171).
    # Only-if-missing: if upstream someday restores/fixes the rename, the
    # existing attribute must NOT be overridden (P4).
    fu = getattr(tool, "flux_utils", None)
    if fu is not None and not hasattr(fu, "check_flux_state_dict_diffusers_schnell"):

        def _check_flux_state_dict_diffusers_schnell(path):
            # Old 3-tuple contract: the cache_latents.py:33 call site
            # 3-unpacks, so the [:3] slice is LOAD-BEARING (an unsliced
            # 4-tuple breaks it; is_schnell stays at index 1). Delegating
            # through the module object `fu` — the same object the tool's
            # call sites resolve — keeps the patch visible to cache_latents.
            return fu.analyze_checkpoint_state(path)[:3]

        fu.check_flux_state_dict_diffusers_schnell = _check_flux_state_dict_diffusers_schnell

    # THIRD upstream defect wrap (prod train-1792329792): the tool loads
    # the text encoders via `flux_utils.load_clip_l(...)`/`load_t5xxl(...)`
    # — module-attribute calls (tools/cache_text_encoder_outputs.py:134/
    # :138) — and library/flux_utils @690ea7f9 builds those models under
    # init_empty_weights() (:345/:400), which does NOT meta-ify non-
    # persistent buffers (position_ids is created on CPU), then
    # load_state_dict(..., assign=True) (:353/:408) puts only the
    # PARAMETERS on accelerator.device; no module `.to()` follows, so the
    # CPU buffer survives into the first CLIP forward and dies on a device
    # mismatch. Wrap both loaders when present — ALWAYS-ACTIVE (a wrap of
    # an existing function, NOT only-if-missing like defect #2): returned
    # models get their mismatched non-meta buffers re-materialized onto
    # the parameter device; already-matching buffers are left untouched
    # (harmless no-op if upstream ever fixes the placement, P6).
    if fu is not None:
        if hasattr(fu, "load_clip_l"):
            fu.load_clip_l = _wrap_loader_with_buffer_sanitize(fu.load_clip_l)
        if hasattr(fu, "load_t5xxl"):
            fu.load_t5xxl = _wrap_loader_with_buffer_sanitize(fu.load_t5xxl)

    # Exact replication of the upstream __main__ block (tool @690ea7f9
    # lines 226-229): setup_parser -> parse_args -> read_config_from_file.
    parser = tool.setup_parser()
    args = parser.parse_args(tool_flags)
    args = tool.args_util.read_config_from_file(args, parser)

    # THE upstream-defect workaround (train-1792313236): the tool's parser
    # spells --weighted_captions as store_true/default=False (~:217) while
    # cache_to_disk asserts `is_sdxl or args.weighted_captions is None`
    # (:62) — parse_args can only ever yield False here, so the FLUX branch
    # is unreachable via CLI/TOML (TOML has no null). None is the only
    # FLUX-passable value and is safe downstream (is_weighted consumers are
    # truthiness-only checks). sd-scripts @690ea7f9; no upstream files
    # modified.
    args.weighted_captions = None

    # Exceptions deliberately propagate (no try/except): a failing
    # cache_to_disk must yield a non-zero exit with a visible traceback so
    # the script-level fail-fast aborts the run before the trainer starts.
    tool.cache_to_disk(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
