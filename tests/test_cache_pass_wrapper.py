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


def _make_fixture(root: Path) -> tuple[Path, Path]:
    tool = root / "fake_cache_text_encoder_outputs.py"
    tool.write_text(_FAKE_TOOL_SRC, encoding="utf-8")
    return tool, root / "marker.jsonl"


def _run_wrapper(
    tool: Path | None,
    marker: Path,
    flags: list[str],
    *,
    raise_: bool = False,
) -> subprocess.CompletedProcess:
    """Run the wrapper in a fresh interpreter with a minimal, controlled
    env (no inherited state; the marker path is the fake tool's only
    channel). tool=None models the no-arguments usage-guard call."""
    env = {k: os.environ[k] for k in ("PATH", "HOME") if k in os.environ}
    env["FAKE_TOOL_MARKER"] = str(marker)
    if raise_:
        env["FAKE_TOOL_RAISE"] = "1"
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
