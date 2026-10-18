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
patch for the SECOND upstream defect documented below. Downstream
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
