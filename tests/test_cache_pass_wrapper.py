"""Contract tests for scripts/cache_pass_wrapper.py (spec 010, D-007 follow-up;
production failure train-1792313236).

The flux TE cache pass invokes upstream sd-scripts tools/
cache_text_encoder_outputs.py, whose cache_to_disk opens with
`assert is_sdxl or args.weighted_captions is None` (:62) while the parser
spells --weighted_captions as `action="store_true", default=False` (:217).
For FLUX (--flux, not --sdxl) that assert can NEVER pass from CLI/TOML:
parse_args yields False (store_true toggles a bool; TOML has no null), so
the cache pass dies rc=1 before writing anything. The fix IS the repo-side
wrapper scripts/cache_pass_wrapper.py: load the tool as a module
(importlib.util.spec_from_file_location), replicate its __main__ flow
(setup_parser → parse_args(tool_args) → tool.args_util
.read_config_from_file(args, parser) — the tool does `import library.args
as args_util` at top, so the attribute exists), force
args.weighted_captions = None (the only FLUX-passable value; downstream
is_weighted consumers are truthiness checks, None is safe), then call
tool.cache_to_disk(args).

The wrapper has landed: these pins are its PERMANENT CONTRACT REGRESSION
GUARD — CI-visible without torch, they fail if the force-None, the failure
propagation, or the fast usage guard is ever removed or weakened (they
landed RED-first against the pre-fix tree and are kept green as regression
protection).

Wrapper contract pinned here (argv: python cache_pass_wrapper.py
<tool_path> <tool_flags...>):
  P1a happy path — rc 0; the tool ran with weighted_captions FORCED to
      None (json null in the marker line), flux True, and the flags
      forwarded untouched;
  P1b failure propagation — a tool whose cache_to_disk raises makes the
      wrapper exit non-zero (the script's fail-fast on the cache pass
      depends on it: the trainer must never launch on a broken cache);
  P1c usage guard — invoked with no arguments the wrapper exits non-zero
      and fails FAST: the tool module is never imported/executed.

Second upstream defect (train-1792323801 — the weighted_captions fix let
the cache pass reach the NEXT landmine): tools/cache_latents.py:33 calls
flux_utils.check_flux_state_dict_diffusers_schnell(path), an attribute
that does NOT exist in library/flux_utils @690ea7f9 (renamed API: the
trainer uses the 4-tuple flux_utils.analyze_checkpoint_state(path) ->
(is_diffusers, is_schnell, (n_double, n_single), keys),
flux_train_network.py:171). The compat patch HAS LANDED in the wrapper:
after exec_module, if the loaded tool module has a flux_utils attribute
whose object LACKS check_flux_state_dict_diffusers_schnell, the wrapper
adds an adapter lambda path: flux_utils.analyze_checkpoint_state(path)[:3]
onto THAT object (module-attribute patch — visible to cache_latents
through the same module object; the old 3-tuple contract keeps is_schnell
at index 1 and the call site 3-unpacks, so the [:3] slice is
load-bearing). These pins are the patch's PERMANENT CONTRACT GUARD:
  P3 adapter applied when missing — the call-site succeeds through the
      adapter, delegating to analyze_checkpoint_state with the exact [:3]
      slice (rc 0, marker proves injection);
  P4 existing attr preserved — when the tool's flux_utils ALREADY
      defines check_flux_state_dict_diffusers_schnell (upstream someday
      fixes the rename), the wrapper must NOT override it (sentinel
      returned unchanged, analyze never touched through an adapter).

Third upstream defect (train-1792329792 — probe #3): with the adapter in
place the cache pass reaches the first CLIP forward and dies there —
RuntimeError "Expected all tensors to be on the same device, but got
index is on cpu, different from other tensors on cuda:0"
(strategy_flux.py:71 -> clip_text_model.py:53 -> HF position_embedding).
Mechanism (@690ea7f9): tools/cache_text_encoder_outputs.py:134/:138 call
flux_utils.load_clip_l(args.clip_l, weight_dtype, accelerator.device) /
load_t5xxl(args.t5xxl, None, accelerator.device) (module-attribute
access); inside load_clip_l, `with init_empty_weights():` creates the
NON-PERSISTENT position_ids buffer on CPU (init_empty_weights does not
meta-ify buffers by default), load_state_dict(..., strict=False,
assign=True) puts the parameters on accelerator.device, and there is NO
module .to() afterwards — the buffer stays CPU forever. Fix contract:
the wrapper sanitizes the loaders it wraps — after the original returns,
ref = next(model.parameters()).device, and every submodule buffer that
is not None, not is_meta and sits on a different device is
re-materialized with buf.to(ref) into m._buffers; matching buffers are
left as the SAME objects. Pins:
  P5  clip_l sanitizer — the cpu position_ids buffers at the ROOT and in
      a nested CHILD module (the real defect lives in the HF
      position_embedding submodule) must land on the parameter device
      ("cuda:0") through the wrapped load_clip_l — pins the modules()
      walk, not just a root _buffers poke;
  P5b t5xxl sanitizer — same through the wrapped load_t5xxl;
  P6  matching-buffer polarity — buffers already on the parameter device
      (root AND nested child) must NOT be re-materialized (object
      identity preserved; the fake
      tags every .to(), so a rewrite-everything sanitizer flips the tag).

Sandbox-safe and torch-free BY DESIGN: a FAKE tool module (written to a
tmp dir by the tests) mirrors exactly the upstream surface the wrapper
touches — including the store_true/default=False spelling of
--weighted_captions that makes the real tool unfixable from the CLI. All
tool-side observables land in a marker file whose path the fake reads from
an env var the test sets; the fake also touches a '<marker>.imported'
sentinel at import time, so P1c can prove the module was never executed.
No real network, no torch, no kohya checkout.

pytest-compatible; pytest is not required.

Run either way:
    python -m pytest tests/test_cache_pass_wrapper.py -q
    python tests/test_cache_pass_wrapper.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # content/
WRAPPER = REPO_ROOT / "scripts" / "cache_pass_wrapper.py"

# Fake stand-in for upstream tools/cache_text_encoder_outputs.py — ONLY the
# surface the wrapper contract touches, spelled EXACTLY like upstream where
# the bug lives: --weighted_captions store_true/default=False (the :217
# trap), a module-level args_util namespace with read_config_from_file
# (identity — the real one merges a TOML; the wrapper must call it anyway),
# and cache_to_disk. cache_to_disk appends its observed args as one JSON
# line to the marker BEFORE optionally raising (FAKE_TOOL_RAISE set), so
# P1b can distinguish "the wrapper ran the tool and propagated its failure"
# from "the wrapper never ran at all" — a non-zero exit alone cannot tell
# the two apart.
_FAKE_TOOL_SRC = '''"""Fake cache_text_encoder_outputs.py (contract harness)."""
import argparse
import json
import os
from types import SimpleNamespace

MARKER = os.environ["FAKE_TOOL_MARKER"]

# import-time sentinel: exists iff the module was imported/executed
with open(MARKER + ".imported", "a", encoding="utf-8"):
    pass


def setup_parser():
    parser = argparse.ArgumentParser(prog="fake_cache_text_encoder_outputs")
    parser.add_argument("--flux", action="store_true")
    parser.add_argument("--weighted_captions", action="store_true", default=False)
    parser.add_argument("--dataset_config", type=str, default=None)
    return parser


def _read_config_from_file(args, parser):
    return args


args_util = SimpleNamespace(read_config_from_file=_read_config_from_file)


def _forwarded_flags():
    """This process's argv AFTER the tool path — robust to the wrapper
    either keeping its own argv (python wrapper <tool> <flags>) or
    rewriting sys.argv to emulate the tool's __main__ ([<tool>, <flags>])."""
    argv = list(os.sys.argv)
    me = os.path.abspath(__file__)
    for i, a in enumerate(argv):
        try:
            if os.path.abspath(a) == me:
                return argv[i + 1:]
        except OSError:
            continue
    return argv[1:]


def cache_to_disk(args):
    with open(MARKER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "weighted_captions": args.weighted_captions,
            "flux": args.flux,
            "argv_rest": _forwarded_flags(),
        }) + "\\n")
    if os.environ.get("FAKE_TOOL_RAISE"):
        raise RuntimeError("fake cache_to_disk failed (train-1792313236 stand-in)")
'''

# P3 fake — train-1792323801 call-site stand-in: module-global flux_utils
# EXPOSES ONLY the renamed 4-tuple API (analyze_checkpoint_state, the
# exact shape of library/flux_utils @690ea7f9 per flux_train_network.py:171)
# and LACKS check_flux_state_dict_diffusers_schnell. cache_to_disk performs
# the tools/cache_latents.py:33 call shape and records the RAW result list,
# so the marker discriminates: no adapter -> AttributeError (nothing
# recorded); correct adapter -> [False, True, [2, 3]] ([:3] slice,
# is_schnell at index 1); unsliced 4-tuple adapter -> a fourth element
# appears and the equality fails.
_FAKE_TOOL_MISSING_CHECK_SRC = '''"""Fake tool whose flux_utils lacks check_flux_state_dict_diffusers_schnell."""
import argparse
import json
import os
from types import SimpleNamespace

MARKER = os.environ["FAKE_TOOL_MARKER"]


def setup_parser():
    parser = argparse.ArgumentParser(prog="fake_cache_text_encoder_outputs")
    parser.add_argument("--flux", action="store_true")
    parser.add_argument("--weighted_captions", action="store_true", default=False)
    parser.add_argument("--dataset_config", type=str, default=None)
    return parser


def _read_config_from_file(args, parser):
    return args


args_util = SimpleNamespace(read_config_from_file=_read_config_from_file)


def _analyze_checkpoint_state(path):
    # renamed-API shape @690ea7f9: (is_diffusers, is_schnell,
    # (n_double, n_single), keys) — distinctive values so a slice is visible
    return (False, True, (2, 3), ["k"])


flux_utils = SimpleNamespace(analyze_checkpoint_state=_analyze_checkpoint_state)


def cache_to_disk(args):
    # tools/cache_latents.py:33 call shape: module-global flux_utils name
    result = flux_utils.check_flux_state_dict_diffusers_schnell("ignored-path")
    with open(MARKER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "called": True,
            "is_schnell": result[1],
            "result": list(result),
        }) + "\\n")
'''

# P4 fake — polarity half: flux_utils ALREADY defines
# check_flux_state_dict_diffusers_schnell (sentinel tuple, the "upstream
# someday fixes the rename" world). analyze_checkpoint_state COUNTS its
# calls into the marker record: an over-riding wrapper (unconditional
# adapter) would replace the sentinel AND reach analyze through the
# adapter — both observable. A correct no-override leaves result == the
# sentinel and analyze_calls == 0.
_FAKE_TOOL_SENTINEL_CHECK_SRC = '''"""Fake tool whose flux_utils already provides the check attr."""
import argparse
import json
import os
from types import SimpleNamespace

MARKER = os.environ["FAKE_TOOL_MARKER"]
ANALYZE_CALLS = 0


def setup_parser():
    parser = argparse.ArgumentParser(prog="fake_cache_text_encoder_outputs")
    parser.add_argument("--flux", action="store_true")
    parser.add_argument("--weighted_captions", action="store_true", default=False)
    parser.add_argument("--dataset_config", type=str, default=None)
    return parser


def _read_config_from_file(args, parser):
    return args


args_util = SimpleNamespace(read_config_from_file=_read_config_from_file)


def _analyze_checkpoint_state(path):
    global ANALYZE_CALLS
    ANALYZE_CALLS += 1
    return (False, True, (2, 3), ["k"])


def _check(path):
    return ("sentinel", "preserved", "upstream")


flux_utils = SimpleNamespace(
    analyze_checkpoint_state=_analyze_checkpoint_state,
    check_flux_state_dict_diffusers_schnell=_check,
)


def cache_to_disk(args):
    raw = flux_utils.check_flux_state_dict_diffusers_schnell("ignored-path")
    with open(MARKER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "called": True,
            "result": list(raw),
            "analyze_calls": ANALYZE_CALLS,
        }) + "\\n")
'''


# P5/P5b/P6 fake — loader-device stand-in for the train-1792329792 split:
# flux_utils exposes load_clip_l/load_t5xxl with the REAL signature
# (path, dtype, device, disable_mmap kwarg); each returns a FakeModel whose
# PARAMETERS sit on the loader's device (assign=True semantics) while the
# non-persistent position_ids BUFFER lands on FAKE_BUFFER_DEVICE ("cpu" by
# default — init_empty_weights leaves buffers on CPU; P6 sets "cuda:0" to
# model the already-matching case) — at the ROOT and in a nested CHILD
# module (the real defect class: the HF position_embedding submodule holds
# position_ids), so the sanitizer's modules() walk is pinned, not just a
# root _buffers poke. FakeTensor.to() returns a NEW object
# with a ".moved" tag: the marker record's tag distinguishes "sanitizer
# skipped a matching buffer" (tag stays "orig", same object) from
# "sanitizer re-materialized it" (tag ".moved"). cache_to_disk performs the
# real flux-branch call shape (module-attribute access, :134/:138) and
# records loader/param_device/buffer_device/buffer_tag per loader.
_FAKE_TOOL_LOADER_DEVICE_SRC = '''"""Fake tool exercising the flux_utils loader call sites (:134/:138)."""
import argparse
import json
import os
from types import SimpleNamespace

MARKER = os.environ["FAKE_TOOL_MARKER"]
BUFFER_DEVICE = os.environ.get("FAKE_BUFFER_DEVICE", "cpu")


class FakeTensor:
    def __init__(self, device, tag="orig"):
        self.device = device
        self.tag = tag
        self.is_meta = False

    def to(self, device):
        return FakeTensor(device, tag=self.tag + ".moved")


class FakeModule:
    """Nested submodule with its OWN non-persistent buffer — the
    train-1792329792 defect class lives in a SUBMODULE (the HF
    position_embedding holds position_ids), so the sanitizer's modules()
    walk is discriminatively pinned: a root-only variant leaves THIS
    buffer on cpu and fails P5/P5b."""

    def __init__(self, buffer_device):
        self._buffers = {"position_ids": FakeTensor(buffer_device)}

    def modules(self):
        return [self]


class FakeModel:
    def __init__(self, param_device, buffer_device):
        self._params = [FakeTensor(param_device, tag="param")]
        self._buffers = {"position_ids": FakeTensor(buffer_device)}
        self.child = FakeModule(buffer_device)

    def parameters(self):
        return iter(self._params)

    def modules(self):
        return [self, self.child]

    def named_modules(self):
        return [("", self), ("child", self.child)]

    def children(self):
        return [self.child]

    def buffers(self):
        return iter(list(self._buffers.values()) + list(self.child._buffers.values()))


def setup_parser():
    parser = argparse.ArgumentParser(prog="fake_cache_text_encoder_outputs")
    parser.add_argument("--flux", action="store_true")
    parser.add_argument("--weighted_captions", action="store_true", default=False)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--clip_l", type=str, default=None)
    parser.add_argument("--t5xxl", type=str, default=None)
    return parser


def _read_config_from_file(args, parser):
    return args


args_util = SimpleNamespace(read_config_from_file=_read_config_from_file)


def _load_clip_l(path, dtype, device, disable_mmap=None):
    return FakeModel(device, BUFFER_DEVICE)


def _load_t5xxl(path, dtype, device, disable_mmap=None):
    return FakeModel(device, BUFFER_DEVICE)


flux_utils = SimpleNamespace(load_clip_l=_load_clip_l, load_t5xxl=_load_t5xxl)


def cache_to_disk(args):
    # real call shape: module-attribute access + disable_mmap kwarg
    clip = flux_utils.load_clip_l(args.clip_l, None, "cuda:0", disable_mmap=False)
    t5 = flux_utils.load_t5xxl(args.t5xxl, None, "cuda:0", disable_mmap=False)
    with open(MARKER, "a", encoding="utf-8") as fh:
        for name, model in (("clip_l", clip), ("t5xxl", t5)):
            buf = model._buffers["position_ids"]
            child_buf = model.child._buffers["position_ids"]
            fh.write(json.dumps({
                "loader": name,
                "param_device": next(model.parameters()).device,
                "buffer_device": buf.device,
                "buffer_tag": buf.tag,
                "child_buffer_device": child_buf.device,
                "child_buffer_tag": child_buf.tag,
            }) + "\\n")
'''


def _make_fixture(root: Path, src: str = _FAKE_TOOL_SRC) -> tuple[Path, Path]:
    tool = root / "fake_cache_text_encoder_outputs.py"
    tool.write_text(src, encoding="utf-8")
    return tool, root / "marker.jsonl"


def _run_wrapper(
    tool: Path | None,
    marker: Path,
    flags: list[str],
    *,
    raise_: bool = False,
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run the wrapper in a fresh interpreter with a minimal, controlled
    env (no inherited state; the marker path is the fake tool's only
    channel). tool=None models the no-arguments usage-guard call."""
    env = {k: os.environ[k] for k in ("PATH", "HOME") if k in os.environ}
    env["FAKE_TOOL_MARKER"] = str(marker)
    if raise_:
        env["FAKE_TOOL_RAISE"] = "1"
    if env_extra:
        env.update(env_extra)
    argv = [sys.executable, str(WRAPPER)]
    if tool is not None:
        argv += [str(tool), *flags]
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(marker.parent),
    )


def _marker_records(marker: Path) -> list[dict]:
    return [json.loads(ln) for ln in marker.read_text(encoding="utf-8").splitlines() if ln]


def test_wrapper_forces_weighted_captions_none_and_forwards_flags():
    # P1a: the whole point of the wrapper — the tool's :62 assert
    # `is_sdxl or args.weighted_captions is None` can only pass for FLUX
    # with weighted_captions FORCED to None after parsing (the parser's
    # store_true/default=False spelling, mirrored exactly in the fake,
    # otherwise yields False). rc 0; the marker line shows json null,
    # flux True, and the flags forwarded to the tool untouched.
    root = Path(tempfile.mkdtemp(prefix="cache-pass-wrapper-"))
    try:
        tool, marker = _make_fixture(root)
        result = _run_wrapper(tool, marker, ["--flux", "--dataset_config", "X"])
        assert result.returncode == 0, (
            f"wrapper must exit 0 on a healthy tool; rc={result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert marker.is_file(), (
            "cache_to_disk must have run (marker line expected); "
            f"rc={result.returncode}\nstderr:\n{result.stderr}"
        )
        records = _marker_records(marker)
        assert len(records) == 1, records
        rec = records[0]
        assert rec["weighted_captions"] is None, (
            "the wrapper must force args.weighted_captions = None before "
            "cache_to_disk (json null) — the tool's :62 assert is only "
            f"FLUX-passable with None; store_true default False is the "
            f"train-1792313236 failure; got: {rec['weighted_captions']!r}"
        )
        assert rec["flux"] is True, (
            f"--flux must reach the tool's parser as store_true; got: {rec}"
        )
        assert rec["argv_rest"] == ["--flux", "--dataset_config", "X"], (
            "the flags after the tool path must be forwarded to the tool "
            f"untouched; got: {rec['argv_rest']}"
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_wrapper_propagates_cache_to_disk_failure():
    # P1b: the script fail-fasts on a non-zero cache pass (the trainer must
    # never launch on an incomplete cache), so the wrapper must exit
    # non-zero when the tool's cache_to_disk raises — AND the marker line
    # must exist: this defeats the rc≠0-without-running tautology (a
    # missing file, an import error, an argparse abort all exit non-zero
    # without ever running the wrapper's contract). The marker must prove
    # cache_to_disk was actually REACHED and its failure propagated.
    root = Path(tempfile.mkdtemp(prefix="cache-pass-wrapper-"))
    try:
        tool, marker = _make_fixture(root)
        result = _run_wrapper(tool, marker, ["--flux"], raise_=True)
        assert result.returncode != 0, (
            f"a raising cache_to_disk must make the wrapper exit non-zero; "
            f"rc={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert marker.is_file(), (
            "the marker must show cache_to_disk was REACHED before the "
            "failure — otherwise this pin is satisfied by a wrapper that "
            f"never ran; rc={result.returncode}\nstderr:\n{result.stderr}"
        )
        records = _marker_records(marker)
        assert len(records) == 1, records
        assert records[0]["weighted_captions"] is None, records[0]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_wrapper_usage_guard_no_args_fails_fast_without_importing_tool():
    # P1c: invoked with NO arguments the wrapper must exit non-zero and
    # fail FAST — before touching any tool module: the guard must reject
    # BEFORE importing/executing any tool, so the marker AND the import
    # sentinel must both stay absent (the fake's module-level code would
    # create the sentinel on any import/exec). The "can't open file"
    # exclusion proves the rejection comes from the WRAPPER's own usage
    # guard, not from the interpreter failing to find the script — an
    # absent wrapper is NOT a usage guard.
    root = Path(tempfile.mkdtemp(prefix="cache-pass-wrapper-"))
    try:
        tool, marker = _make_fixture(root)
        result = _run_wrapper(None, marker, [])
        assert result.returncode != 0, (
            f"no-argument invocation must exit non-zero; rc={result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert "can't open file" not in result.stderr, (
            "the usage guard must be the WRAPPER's own fast rejection — "
            "'can't open file' means the wrapper script itself is missing "
            f"and no guard ran; stderr:\n{result.stderr}"
        )
        assert not marker.exists(), (
            f"the tool must never be reached without args; marker file "
            f"{marker.name} was created; stderr:\n{result.stderr}"
        )
        assert not Path(str(marker) + ".imported").exists(), (
            "fail-fast means the tool module is NOT imported/executed — "
            f"the import sentinel {marker.name}.imported exists; "
            f"stderr:\n{result.stderr}"
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_wrapper_injects_flux_check_adapter_when_missing():
    # P3 (train-1792323801, adapter-contract guard): the weighted_
    # captions fix let the cache pass reach the NEXT upstream defect —
    # tools/cache_latents.py:33 calls flux_utils.check_flux_state_dict_
    # diffusers_schnell(path), which library/flux_utils @690ea7f9 does NOT
    # define (renamed API: analyze_checkpoint_state -> 4-tuple
    # (is_diffusers, is_schnell, (n_double, n_single), keys), the spelling
    # the trainer itself uses at flux_train_network.py:171). The landed
    # compat patch: after exec_module, if the tool module has a
    # flux_utils attribute whose object LACKS the check, the wrapper adds
    # lambda path: flux_utils.analyze_checkpoint_state(path)[:3] onto that
    # object (module-attribute patch — the same object the tool's call
    # sites resolve, so cache_latents sees it). Fake: flux_utils exposes
    # ONLY the 4-tuple analyze; cache_to_disk performs the real call shape
    # and records the RAW result. This pin guards the adapter contract:
    # injection when missing, delegation to analyze_checkpoint_state with
    # the load-bearing [:3] slice (the call site 3-unpacks). Its red
    # signature if the patch is ever removed: AttributeError inside
    # cache_to_disk -> rc != 0, marker never written. The raw-result
    # assertion pins the slice semantics: is_schnell stays at index 1
    # (True here) and an unsliced 4-tuple adapter is caught by the extra
    # element.
    root = Path(tempfile.mkdtemp(prefix="cache-pass-wrapper-"))
    try:
        tool, marker = _make_fixture(root, src=_FAKE_TOOL_MISSING_CHECK_SRC)
        result = _run_wrapper(tool, marker, ["--flux"])
        assert result.returncode == 0, (
            "the wrapper must compat-patch the missing check so the tool's "
            "cache_latents.py:33 call-site succeeds; rc="
            f"{result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert marker.is_file(), (
            "the marker must prove the adapter was injected and the call "
            f"succeeded; rc={result.returncode}\nstderr:\n{result.stderr}"
        )
        records = _marker_records(marker)
        assert len(records) == 1, records
        rec = records[0]
        assert rec["called"] is True, rec
        assert rec["is_schnell"] is True, (
            "the adapter's result must keep is_schnell at index 1 (the old "
            f"3-tuple contract); got: {rec}"
        )
        assert rec["result"] == [False, True, [2, 3]], (
            "the adapter must delegate to analyze_checkpoint_state and slice "
            f"[:3] — (is_diffusers, is_schnell, (n_double, n_single)); an "
            f"unsliced 4-tuple is a contract break; got: {rec['result']}"
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_wrapper_preserves_existing_flux_check_attribute():
    # P4 (polarity half of the compat patch): when the loaded tool's
    # flux_utils ALREADY defines check_flux_state_dict_diffusers_schnell
    # (upstream someday fixes the rename), the wrapper must NOT override
    # it — the sentinel must survive untouched. The fake's
    # analyze_checkpoint_state counts calls into the marker record, so an
    # over-riding implementation (unconditional adapter) is caught twice:
    # the sentinel AND analyze_calls flips to 1. The compat patch landed
    # WITH the missing-check guard, so this pin is the PERMANENT polarity
    # guard against the over-riding regression (an unconditional setattr
    # would replace the sentinel and route analyze) — the no-override
    # half of the P3/P4 pair, landed together as the train-1792323801
    # fix.
    root = Path(tempfile.mkdtemp(prefix="cache-pass-wrapper-"))
    try:
        tool, marker = _make_fixture(root, src=_FAKE_TOOL_SENTINEL_CHECK_SRC)
        result = _run_wrapper(tool, marker, ["--flux"])
        assert result.returncode == 0, (
            f"a tool whose flux_utils provides the check must run untouched; "
            f"rc={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert marker.is_file(), (
            f"cache_to_disk must have recorded the raw check result; "
            f"rc={result.returncode}\nstderr:\n{result.stderr}"
        )
        records = _marker_records(marker)
        assert len(records) == 1, records
        rec = records[0]
        assert rec["result"] == ["sentinel", "preserved", "upstream"], (
            "the wrapper must NOT override an existing "
            f"check_flux_state_dict_diffusers_schnell — the sentinel must be "
            f"returned unchanged; got: {rec['result']}"
        )
        assert rec["analyze_calls"] == 0, (
            "the existing check must be called directly, never routed "
            f"through an adapter onto analyze_checkpoint_state; got: {rec}"
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_wrapper_sanitizes_clip_l_cpu_buffer_onto_param_device():
    # P5 (train-1792323801 adapter worked; train-1792329792 is the NEXT
    # landmine — landed RED-first against the pre-wrap tree; PERMANENT
    # guard for the loader wrap): the cache pass now reaches the first
    # CLIP forward and dies
    # with "Expected all tensors to be on the same device, but got index
    # is on cpu, different from other tensors on cuda:0"
    # (strategy_flux.py:71 -> clip_text_model.py:53 -> HF
    # position_embedding). Mechanism (@690ea7f9):
    # tools/cache_text_encoder_outputs.py:134 calls
    # flux_utils.load_clip_l(args.clip_l, weight_dtype, accelerator.device)
    # via MODULE-ATTRIBUTE access; inside, `with init_empty_weights():`
    # leaves the non-persistent position_ids buffer on CPU
    # (init_empty_weights does not meta-ify buffers by default),
    # load_state_dict(..., assign=True) puts the parameters on
    # accelerator.device, and NO module .to() follows — the buffer stays
    # CPU forever. Fix contract: the wrapper sanitizes the wrapped
    # load_clip_l — after the original returns, ref =
    # next(model.parameters()).device; every submodule buffer that is not
    # None, not is_meta and sits on a different device is re-materialized
    # with buf.to(ref) into m._buffers. Fake: load_clip_l returns a model
    # with cuda:0 params and a cpu position_ids buffer at the ROOT and in
    # a nested CHILD module (the real defect lives in the HF
    # position_embedding SUBMODULE, so the modules() walk is pinned, not
    # just a root _buffers poke); cache_to_disk performs the real call
    # shape and records both buffers. Red signature if the wrap is ever
    # removed: buffers stay "cpu" (root and child alike). The root-buffer
    # assertion pins the mismatch re-materialization; the child-buffer
    # assertion pins the submodule walk.
    root = Path(tempfile.mkdtemp(prefix="cache-pass-wrapper-"))
    try:
        tool, marker = _make_fixture(root, src=_FAKE_TOOL_LOADER_DEVICE_SRC)
        result = _run_wrapper(
            tool, marker,
            ["--flux", "--clip_l", "clip_l.safetensors", "--t5xxl", "t5xxl.safetensors"],
        )
        assert result.returncode == 0, (
            "the loader call sites must run through the wrapper; "
            f"rc={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        records = {rec["loader"]: rec for rec in _marker_records(marker)}
        assert set(records) == {"clip_l", "t5xxl"}, records
        clip = records["clip_l"]
        assert clip["param_device"] == "cuda:0", clip
        assert clip["buffer_device"] == "cuda:0", (
            "the sanitizer must re-materialize the mismatched position_ids "
            "buffer onto the parameter device (train-1792329792: a CPU "
            "buffer against cuda:0 params crashes the first CLIP forward "
            "at the HF position-embedding lookup); got: "
            f"{clip}\nthis is the pin's red signature if the load_clip_l "
            "wrap is ever removed."
        )
        assert clip["child_buffer_device"] == "cuda:0", (
            "the sanitizer must WALK SUBMODULES (model.modules()) — the "
            "real position_ids lives in the nested HF position_embedding, "
            "so a root-only sanitizer variant leaves the child buffer on "
            f"cpu and the prod crash survives; got: {clip}"
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_wrapper_sanitizes_t5xxl_cpu_buffer_onto_param_device():
    # P5b: the same sanitizer contract on the SECOND loader —
    # tools/cache_text_encoder_outputs.py:138 calls
    # flux_utils.load_t5xxl(args.t5xxl, None, accelerator.device) and the
    # T5 position_ids buffer splits exactly like CLIP's (assign=True
    # params on cuda:0, init_empty_weights CPU buffer, no post-.to()).
    # Landed RED-first against the pre-wrap tree; PERMANENT guard — its
    # red signature if the load_t5xxl wrap is ever removed is both
    # buffers (root and nested child) staying "cpu".
    root = Path(tempfile.mkdtemp(prefix="cache-pass-wrapper-"))
    try:
        tool, marker = _make_fixture(root, src=_FAKE_TOOL_LOADER_DEVICE_SRC)
        result = _run_wrapper(
            tool, marker,
            ["--flux", "--clip_l", "clip_l.safetensors", "--t5xxl", "t5xxl.safetensors"],
        )
        assert result.returncode == 0, (
            f"rc={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        records = {rec["loader"]: rec for rec in _marker_records(marker)}
        assert set(records) == {"clip_l", "t5xxl"}, records
        t5 = records["t5xxl"]
        assert t5["param_device"] == "cuda:0", t5
        assert t5["buffer_device"] == "cuda:0", (
            "the sanitizer must re-materialize the mismatched T5 position_ids "
            "buffer onto the parameter device too (load_t5xxl is wrapped "
            f"under the same contract as load_clip_l); got: {t5}\n"
            "this is the pin's red signature if the load_t5xxl wrap is "
            "ever removed."
        )
        assert t5["child_buffer_device"] == "cuda:0", (
            "the T5 sanitizer must WALK SUBMODULES too — a root-only "
            "variant leaves the nested child buffer on cpu; got: "
            f"{t5}"
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_wrapper_leaves_matching_buffers_untouched():
    # P6 (polarity guard for the sanitizer): only MISMATCHED, non-meta
    # buffers may be re-materialized; a buffer already on the parameter
    # device must survive as the SAME object — at the ROOT and in the
    # nested CHILD module. The fake tags every .to() materialization
    # ".moved", so a rewrite-everything sanitizer (unconditional
    # buf.to(ref)) flips the tag even though the device already matches —
    # that is this pin's bite. The sanitizer landed WITH the mismatch
    # guard, so this is the PERMANENT polarity guard against the
    # rewrite-everything regression (it was green-by-construction against
    # the pre-wrap tree, disclosed the way P4 was, and landed alongside
    # P5/P5b as the no-touch half of the trio).
    root = Path(tempfile.mkdtemp(prefix="cache-pass-wrapper-"))
    try:
        tool, marker = _make_fixture(root, src=_FAKE_TOOL_LOADER_DEVICE_SRC)
        result = _run_wrapper(
            tool, marker,
            ["--flux", "--clip_l", "clip_l.safetensors", "--t5xxl", "t5xxl.safetensors"],
            env_extra={"FAKE_BUFFER_DEVICE": "cuda:0"},
        )
        assert result.returncode == 0, (
            f"rc={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        records = {rec["loader"]: rec for rec in _marker_records(marker)}
        assert set(records) == {"clip_l", "t5xxl"}, records
        for name in ("clip_l", "t5xxl"):
            rec = records[name]
            assert rec["buffer_device"] == "cuda:0", rec
            assert rec["buffer_tag"] == "orig", (
                f"{name}: a buffer already on the param device must be left "
                "alone — object identity preserved (a re-materialized "
                f"buffer carries the '.moved' tag); got: {rec}"
            )
            assert rec["child_buffer_device"] == "cuda:0", rec
            assert rec["child_buffer_tag"] == "orig", (
                f"{name}: the MATCHING nested-submodule buffer must be left "
                "alone too — walking submodules must not turn into "
                "rewriting everything; got: "
                f"{rec}"
            )
    finally:
        shutil.rmtree(root, ignore_errors=True)


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
