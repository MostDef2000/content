"""Cross-device trash pins (W3 teardown prod blocker, train-1792396861).

On the prod box models/ and runtime/trash/ live on DIFFERENT mounts, and
the soft-delete paths moved with os.replace — atomic only on the SAME
filesystem — so every teardown path 500s (all four now routed through
the shared _move_to_trash helper — first three landed RED-first, the
fourth likewise):

  * delete_model (management/main.py:797)          -> 500 «could not move
    model to trash: [Errno 18] Invalid cross-device link:
    '/workspace/models/valery23' -> '/workspace/runtime/trash/valery23-1792396861'»
  * wipe_model_dataset (main.py:1115)              -> 500 «could not move
    dataset to trash: ... valery23/dataset/images ->
    /workspace/runtime/trash/dataset/valery23-wipe-...» (reproduced live)
  * delete_model_post (main.py:438)                -> 500 «could not move
    post to trash: ...» (reproduced: POST
    /api/models/valery23/posts/{name}?confirm={name}&kind=private)
  * 4th same-class site flagged by the code worker (now routed too,
    prod train-1792396861 class): delete_model_dataset_image
    (main.py:1187, DELETE /api/models/{id}/dataset/{filename}) — TWO raw
    os.replace calls (image :1203 + paired caption :1205) into
    runtime/trash/dataset/<id>-<ts>/ across the same two prod mounts,
    500 «could not move image to trash: ...»

Fix direction (code worker): a SHARED trash-move helper — try os.replace
first (keep the same-fs atomic fast path), on OSError fall back to
shutil.move (copy+delete across devices). All four call sites go through
it; the existing semantics (collision "-2"/"-3" suffixes, parents
creation, manifest copy, empty-dataset no-op) stay unchanged.

These pins simulate the prod topology by patching os.replace to raise
errno.EXDEV ONLY for destinations under the trash dir — the registry and
manifest atomic saves (tmp+os.replace on their own mount) keep working,
the fake delegates everything else to the real os.replace. The first
three pins landed RED-first against the raw-os.replace tree and are now
PERMANENT guards for the routed _move_to_trash helper (assertions prove
the fallback preserved the semantics: model dir gone + copy under trash +
registry entry removed; images/ recreated empty + manifest reset + trash
copy incl. manifest.json; post folder moved to trash). The fourth pin
(single-image dataset delete) landed RED-first like the others —
PERMANENT guard for its both-moves routing.

Validator-narrowing round (release review on the delivered helper):
pin A (test_trash_move_fallback_copies_across_devices) is a COVERAGE pin,
green by design — it forces BOTH os.replace AND os.rename to EXDEV so
shutil.move must take the genuine copytree/copy2+rmtree branch (the
pre-A pins could not distinguish that from a rename shortcut) and proves
byte-exact content + source removal end-to-end. Pins B and C landed
RED-first against the round-18 broad-fallback helper (shutil.move on ANY
OSError); after the delivered EXDEV-only narrowing they are PERMANENT
guards — a permission-class error (EACCES) propagates to the 500 wrapper
with the source intact and no partial destination (B at the pure helper,
C at the two-move dataset-image site where a half-completed pair would
silently break image_count == caption_count). Pin D
(test_existing_dst_is_rejected_loudly) is a second COVERAGE pin, green by
design, for the validator's existing-dst rejection: the helper raises
FileExistsError(EEXIST) where shutil.move would otherwise silently NEST
the source into a pre-existing destination. Pin E
(test_exdev_copy_failure_surfaces_with_source_intact) is a third COVERAGE
pin, green by design, for validator condition 1: when the EXDEV copy
branch itself fails mid-way (simulated EIO after the first file), the
failure surfaces loud (500), the source stays byte-intact, the manifest
reset never runs, and the PARTIAL destination remains identified in the
trash — the pinned non-transactional property (no self-clean).

pytest-compatible; pytest is not required.

Run either way:
    python -m pytest tests/test_trash_crossdevice.py -q
    python tests/test_trash_crossdevice.py
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # content/
sys.path.insert(0, str(REPO_ROOT))

_TMP_WORKSPACE = tempfile.TemporaryDirectory(prefix="xdev-ws-")
# StaticFiles(directory=...) is mounted at import time and requires the dir.
Path(_TMP_WORKSPACE.name, "management", "static").mkdir(parents=True, exist_ok=True)

_PREV_WORKSPACE = os.environ.get("WORKSPACE")
os.environ["WORKSPACE"] = _TMP_WORKSPACE.name
try:
    import management.main as main  # noqa: E402
finally:
    if _PREV_WORKSPACE is None:
        os.environ.pop("WORKSPACE", None)
    else:
        os.environ["WORKSPACE"] = _PREV_WORKSPACE


# In a shared pytest process management.main's module-level paths freeze at
# first import (first file's temp WORKSPACE). Re-point them at THIS file's
# temp WORKSPACE before the seeds below and re-bind per module at run time.
def _bind_paths() -> None:
    ws = Path(_TMP_WORKSPACE.name)
    main.WORKSPACE = ws
    main.JOBS_DIR = ws / "runtime" / "jobs"
    main.MODELS_DIR = ws / "models"
    main.REGISTRY_PATH = ws / "models" / "registry.json"
    main.LIBRARY_PATH = ws / "models" / "library.json"
    main.LORAS_DIR = ws / "runtime" / "models" / "loras"
    main.TRASH_DIR = ws / "runtime" / "trash"
    main.DATASET_TRASH = main.TRASH_DIR / "dataset"


_bind_paths()  # direct runner: single file per process

try:
    import pytest

    @pytest.fixture(scope="module", autouse=True)
    def _bind_file_paths():
        _bind_paths()
        yield
except ModuleNotFoundError:  # direct runner: pytest is optional
    pass

# Distinct ids (no collision with tst01/postmodel job pollution from other
# files): xdev01 active (wipe/post target), xdev02 inactive (delete target).
MODEL_ID = "xdev01"
GONE_ID = "xdev02"

OLD_MANIFEST = {
    "version": 1,
    "images": [
        {"filename": "a.png", "tag": "tasteful", "caption": "caption a",
         "created": "2026-01-01T00:00:00Z"},
    ],
}


def _run(coro):
    return asyncio.run(coro)


def _write_registry() -> None:
    """Fresh two-model registry: xdev01 active, xdev02 inactive (delete_model
    refuses the active one). Re-written at every pin start so pins are
    order-independent even after a successful delete removed xdev02."""
    main.REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    main.REGISTRY_PATH.write_text(
        json.dumps(
            {
                "version": 1,
                "models": [
                    {
                        "id": MODEL_ID,
                        "name": "Xdev",
                        "age": 23,
                        "height_cm": 178,
                        "tags": [],
                        "active": True,
                        "created": "2026-01-01T00:00:00Z",
                        "lora": f"{MODEL_ID}.safetensors",
                        "groups": [],
                    },
                    {
                        "id": GONE_ID,
                        "name": "Gone",
                        "age": 24,
                        "height_cm": 170,
                        "tags": [],
                        "active": False,
                        "created": "2026-01-01T00:00:00Z",
                        "lora": f"{GONE_ID}.safetensors",
                        "groups": [],
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _seed_model_dir(model_id: str) -> Path:
    model_dir = main.MODELS_DIR / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "character.json").write_text(
        json.dumps({"name": "Gone", "age": 24}), encoding="utf-8"
    )
    return model_dir


def _seed_dataset() -> None:
    """Dataset images/ with a.png + a.txt and a 1-entry manifest."""
    image_dir = main._dataset_image_dir(MODEL_ID)
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "a.png").write_bytes(b"fake-png-a")
    (image_dir / "a.txt").write_bytes(b"caption a")
    main._save_manifest(MODEL_ID, OLD_MANIFEST)


@contextlib.contextmanager
def _simulate_cross_device_trash(error: int = errno.EXDEV, patch_rename: bool = False):
    """Prod topology: os.replace raises `error` (EXDEV by default) for every
    destination under the trash dir (models/ and runtime/trash/ on different
    mounts), and DELEGATES to the real os.replace for everything else — the
    registry and manifest atomic saves (tmp+replace on their own mount) must
    keep working, exactly like prod. Patched on the os MODULE (main calls
    os.replace via the module attribute); restored in finally.

    patch_rename=True additionally raises the same error for trash-destined
    os.rename calls — shutil.move's FIRST internal attempt is os.rename,
    which would otherwise succeed on the same-fs test workspace and mask
    the real copy+delete branch (pin A forces that branch)."""
    real_replace = os.replace
    real_rename = os.rename
    trash_root = str(main.TRASH_DIR)

    def _dispatch(real_fn, src, dst, args, kwargs):
        if str(dst).startswith(trash_root):
            raise OSError(error, os.strerror(error), str(src), str(dst))
        return real_fn(src, dst, *args, **kwargs)

    def fake_replace(src, dst, *args, **kwargs):
        return _dispatch(real_replace, src, dst, args, kwargs)

    def fake_rename(src, dst, *args, **kwargs):
        return _dispatch(real_rename, src, dst, args, kwargs)

    os.replace = fake_replace
    if patch_rename:
        os.rename = fake_rename
    try:
        yield
    finally:
        os.replace = real_replace
        if patch_rename:
            os.rename = real_rename


@contextlib.contextmanager
def _fail_copytree_after_first_file():
    """Pin E injection: replaces shutil.copytree (the module attribute —
    shutil.move resolves copytree from its own module globals, and main
    calls shutil.move through the same module object) with a PARTIAL copy:
    creates the real destination tree, copies the first sorted entry, then
    raises OSError(EIO) mid-copy. The partial destination therefore
    actually exists and is identifiable when the failure surfaces."""
    real_copytree = shutil.copytree

    def failing(src, dst, *args, **kwargs):
        os.makedirs(dst)
        first = sorted(os.listdir(src))[0]
        shutil.copy2(os.path.join(src, first), os.path.join(dst, first))
        raise OSError(errno.EIO, "simulated mid-copy failure")

    shutil.copytree = failing
    try:
        yield
    finally:
        shutil.copytree = real_copytree


# --------------------------------------------------------------------------- #
# RED pins — one per broken call site
# --------------------------------------------------------------------------- #


def test_delete_model_survives_cross_device_trash():
    # delete_model (main.py:797): prod 500 «could not move model to trash:
    # [Errno 18] Invalid cross-device link: '/workspace/models/valery23'
    # -> '/workspace/runtime/trash/valery23-1792396861'». Fix contract: the
    # trash move falls back to shutil.move on OSError (os.replace kept as
    # the same-fs fast path). Landed RED-first against the raw-os.replace
    # tree (red signature: EXDEV on the os.replace call -> HTTPException
    # 500, the exact prod frame). Against the routed helper this pin also
    # proves the fallback preserved the semantics: model dir gone, folder
    # landed in runtime/trash/ with its content, registry entry removed.
    _write_registry()
    model_dir = _seed_model_dir(GONE_ID)
    with _simulate_cross_device_trash():
        result = _run(main.delete_model(GONE_ID, confirm=GONE_ID))
    assert result == {"ok": True}, result
    assert not model_dir.exists(), "the model dir must be gone from models/"
    trash_entries = sorted(
        p for p in main.TRASH_DIR.iterdir() if p.name.startswith(f"{GONE_ID}-")
    )
    assert trash_entries, (
        "the model folder must land under runtime/trash/ via the "
        f"cross-device fallback; trash dir holds: {sorted(p.name for p in main.TRASH_DIR.iterdir())}"
    )
    assert (trash_entries[0] / "character.json").is_file(), (
        "the fallback must MOVE the folder content, not just create a stub"
    )
    reg = json.loads(main.REGISTRY_PATH.read_text(encoding="utf-8"))
    assert GONE_ID not in {str(m.get("id")) for m in reg["models"]}, (
        "the registry entry must be removed (same-fs atomic save kept working "
        "while the trash move used the fallback)"
    )
    assert MODEL_ID in {str(m.get("id")) for m in reg["models"]}, "the other model stays"


def test_wipe_model_dataset_survives_cross_device_trash():
    # wipe_model_dataset (main.py:1115): prod 500 «could not move dataset to
    # trash: [Errno 18] ... valery23/dataset/images ->
    # /workspace/runtime/trash/dataset/valery23-wipe-...» (reproduced live).
    # Fix contract: same shared helper; the manifest copy (shutil.copy2 into
    # the trash dir — NOT os.replace) and the manifest reset must keep their
    # semantics. Landed RED-first against the raw-os.replace tree (red
    # signature: HTTPException 500 on the simulated EXDEV).
    _write_registry()
    _seed_dataset()
    image_dir = main._dataset_image_dir(MODEL_ID)
    with _simulate_cross_device_trash():
        result = _run(main.wipe_model_dataset(MODEL_ID, confirm=MODEL_ID))
    assert result["ok"] is True, result
    assert result["trash_path"] is not None, "the wipe must report the trash path"
    assert image_dir.is_dir() and list(image_dir.iterdir()) == [], (
        "images/ must be recreated EMPTY after the fallback move"
    )
    manifest = json.loads(main._dataset_manifest_path(MODEL_ID).read_text(encoding="utf-8"))
    assert manifest["images"] == [], "manifest reset (atomic save worked through the fake)"
    trash_dir = Path(result["trash_path"])
    assert trash_dir.is_dir() and trash_dir.parent == main.DATASET_TRASH
    assert (trash_dir / "a.png").is_file(), "the dataset content must land in trash"
    assert (trash_dir / "manifest.json").is_file(), (
        "the manifest copy-along must survive the fallback move"
    )


def test_delete_model_post_survives_cross_device_trash():
    # delete_model_post (main.py:438, pure helper — directly importable):
    # prod 500 «could not move post to trash: [Errno 18] ...» (reproduced
    # via POST /api/models/valery23/posts/{name}?confirm={name}&kind=private).
    # Fix contract: same shared helper; the collision "-2"/"-3" suffixing
    # and parents creation stay unchanged. Landed RED-first against the
    # raw-os.replace tree (red signature: HTTPException 500).
    _write_registry()
    post_dir = main.WORKSPACE / "models" / MODEL_ID / "posts" / "post_x"
    post_dir.mkdir(parents=True, exist_ok=True)
    (post_dir / "index.json").write_text("{}", encoding="utf-8")
    trash_posts = main.WORKSPACE / "runtime" / "trash" / "posts"
    with _simulate_cross_device_trash():
        name = main.delete_model_post(main.WORKSPACE, MODEL_ID, "post_x", "public")
    assert name == "post_x", name
    assert not post_dir.exists(), "the post folder must be gone"
    # trash naming is "{kind}-{name}-{ts}" (same as the landed post-delete
    # pins' _trash_matches("public-<name>") idiom)
    moved = sorted(p for p in trash_posts.iterdir() if p.name.startswith("public-post_x-"))
    assert len(moved) == 1, (
        "the post folder must land in runtime/trash/posts/ via the fallback; "
        f"trash posts holds: {sorted(p.name for p in trash_posts.iterdir())}"
    )
    assert (moved[0] / "index.json").is_file(), "moved folder keeps its content"


def test_delete_dataset_image_survives_cross_device_trash():
    # 4th same-class EXDEV site (code-worker flag; prod train-1792396861
    # class, single-image path): delete_model_dataset_image (main.py:1187,
    # DELETE /api/models/{id}/dataset/{filename}) used TWO RAW os.replace
    # calls — the image (:1203) and its paired .txt caption (:1205) — into
    # runtime/trash/dataset/<id>-<ts>/ across the same two prod mounts.
    # Landed RED-first against the pre-fix tree; PERMANENT guard: red
    # signature (HTTPException 500 «could not move image to trash: [Errno 18]...»)
    # if either move is un-routed from the shared _move_to_trash helper.
    # Fix contract: route BOTH moves through the helper (os.replace fast
    # path + shutil.move fallback); the manifest entry removal and the
    # image+caption pairing semantics (image_count == caption_count stays
    # balanced in images/) are unchanged. After the fix this pin also
    # proves the fallback moved BOTH files with content intact and
    # updated the manifest on disk.
    _write_registry()
    _seed_dataset()
    image_dir = main._dataset_image_dir(MODEL_ID)
    with _simulate_cross_device_trash():
        result = _run(main.delete_model_dataset_image(MODEL_ID, "a.png"))
    assert result["ok"] is True, result
    assert result["manifest"]["images"] == [], result
    manifest = json.loads(
        main._dataset_manifest_path(MODEL_ID).read_text(encoding="utf-8")
    )
    assert manifest["images"] == [], (
        "the deleted filename's manifest entry must be gone on disk too "
        "(the same-fs atomic manifest save kept working through the fake)"
    )
    assert not (image_dir / "a.png").exists() and not (image_dir / "a.txt").exists(), (
        "the image AND its paired caption must leave images/ — both raw "
        "os.replace calls need the fallback"
    )
    single_trash = [
        p
        for p in main.DATASET_TRASH.iterdir()
        if p.name.startswith(f"{MODEL_ID}-") and "-wipe-" not in p.name
    ]
    assert len(single_trash) == 1, (
        "expected exactly one single-image trash dir under "
        f"runtime/trash/dataset/; got: {[p.name for p in main.DATASET_TRASH.iterdir()]}"
    )
    assert (single_trash[0] / "a.png").read_bytes() == b"fake-png-a", (
        "the fallback must move the image content intact"
    )
    assert (single_trash[0] / "a.txt").read_bytes() == b"caption a", (
        "the paired caption must land in the same trash dir"
    )


# --------------------------------------------------------------------------- #
# Validator-narrowing pins (release review on the round-18 helper, which
# fell back to shutil.move on ANY OSError; the delivered helper implements
# the required EXDEV-only fallback + existing-dst rejection). Pin A is a
# COVERAGE pin (green by design, disclosed below): the existing pins patch
# only os.replace, and shutil.move's internal os.rename succeeds on the
# same-fs test workspace, so the REAL copy+delete branch was never
# exercised. Pins B and C landed RED-first against the broad-fallback
# helper; after the narrowing they are PERMANENT guards. Pin D is a second
# COVERAGE pin (green by design) for the existing-dst rejection, pin E a
# third (green by design) for the mid-copy failure's loud-500 +
# source-intact + partial-dst-identified non-transactional property.
# --------------------------------------------------------------------------- #


def test_trash_move_fallback_copies_across_devices():
    # PIN A — COVERAGE pin, green by design against the delivered helper
    # (disclosed like P6/P8): forces BOTH os.replace and os.rename to raise
    # EXDEV for trash-destined moves, so shutil.move cannot shortcut via
    # rename and MUST take the genuine cross-device branch
    # (copytree/copy2 + rmtree/unlink). The pre-A pins could not tell a
    # working copy branch from a rename-succeeded one; this pin proves the
    # prod failure mode end-to-end: byte-exact content lands in trash, the
    # source is gone, on the pure helper (delete_model_post, directory ->
    # copytree) AND an endpoint (wipe_model_dataset, directory + manifest
    # copy2 alongside). Stays green through the EXDEV-only narrowing —
    # this scenario IS the EXDEV case.
    _write_registry()

    # scenario 1: pure helper, directory move via the copy branch
    post_dir = main.WORKSPACE / "models" / MODEL_ID / "posts" / "post_a"
    post_dir.mkdir(parents=True, exist_ok=True)
    (post_dir / "photo.jpg").write_bytes(b"PHOTOBYTES-0123456789")
    (post_dir / "caption.txt").write_bytes(b"a caption\n")
    trash_posts = main.WORKSPACE / "runtime" / "trash" / "posts"
    with _simulate_cross_device_trash(patch_rename=True):
        name = main.delete_model_post(main.WORKSPACE, MODEL_ID, "post_a", "public")
    assert name == "post_a", name
    assert not post_dir.exists(), "the copy+delete branch must remove the source"
    moved = sorted(p for p in trash_posts.iterdir() if p.name.startswith("public-post_a-"))
    assert len(moved) == 1, f"expected one trash copy, got {[p.name for p in moved]}"
    assert (moved[0] / "photo.jpg").read_bytes() == b"PHOTOBYTES-0123456789", (
        "byte-exact image content through the real copy branch"
    )
    assert (moved[0] / "caption.txt").read_bytes() == b"a caption\n"

    # scenario 2: endpoint, dataset images/ dir via the copy branch
    _seed_dataset()
    image_dir = main._dataset_image_dir(MODEL_ID)
    with _simulate_cross_device_trash(patch_rename=True):
        result = _run(main.wipe_model_dataset(MODEL_ID, confirm=MODEL_ID))
    assert result["ok"] is True and result["trash_path"] is not None, result
    wipe_dir = Path(result["trash_path"])
    assert (wipe_dir / "a.png").read_bytes() == b"fake-png-a"
    assert (wipe_dir / "a.txt").read_bytes() == b"caption a"
    assert (wipe_dir / "manifest.json").is_file(), (
        "the manifest copy-along must survive the forced copy branch"
    )
    assert image_dir.is_dir() and list(image_dir.iterdir()) == [], (
        "source emptied by the copy+delete branch"
    )


def test_trash_move_non_exdev_error_does_not_fall_back():
    # PIN B — landed RED-first against the broad-fallback helper (round
    # 18); PERMANENT guard for the validator's EXDEV-only narrowing at the
    # pure helper. The round-18 _move_to_trash caught OSError GENERALLY,
    # so a permission error (EACCES — not a cross-device condition)
    # silently triggered the copy+delete fallback: shutil.move's internal
    # os.rename (unpatched here, same-fs) succeeded and the delete
    # COMPLETED anyway — masking an operator-visible failure. Required and
    # delivered: fall back ONLY when exc.errno == EXDEV; anything else
    # propagates to the call site's 500 wrapper. Assertions: 500 «could
    # not move post to trash», source intact, no trash destination
    # created. The pin's red signature against the round-18 helper was the
    # else-branch AssertionError (move completed).
    _write_registry()
    post_dir = main.WORKSPACE / "models" / MODEL_ID / "posts" / "post_b"
    post_dir.mkdir(parents=True, exist_ok=True)
    (post_dir / "photo.jpg").write_bytes(b"PHOTO-B")
    trash_posts = main.WORKSPACE / "runtime" / "trash" / "posts"
    with _simulate_cross_device_trash(error=errno.EACCES):
        try:
            name = main.delete_model_post(main.WORKSPACE, MODEL_ID, "post_b", "public")
        except main.HTTPException as exc:
            assert exc.status_code == 500, exc.status_code
            assert "could not move post to trash" in exc.detail, exc.detail
        else:
            raise AssertionError(
                "EACCES must NOT trigger the cross-device fallback "
                "(EXDEV-only narrowing) — the delete completed anyway: "
                f"{name!r}"
            )
    assert post_dir.is_dir() and (post_dir / "photo.jpg").is_file(), (
        "a refused move leaves the source intact"
    )
    leftovers = [p for p in trash_posts.iterdir() if p.name.startswith("public-post_b-")]
    assert leftovers == [], (
        f"no trash destination may be created for a refused move; got: {leftovers}"
    )


def test_trash_move_non_exdev_error_keeps_dataset_unsplit():
    # PIN C — landed RED-first against the broad-fallback helper (round
    # 18); PERMANENT split-state guard of the same narrowing, on the
    # TWO-move site: delete_model_dataset_image moves the image AND its
    # paired caption (:1203/:1205). A simulated I/O error that is NOT
    # EXDEV (EACCES) must 500 with BOTH files still in images/, the
    # manifest entry intact and no partial content in the trash dir. The
    # pin's red signature against the round-18 helper: the broad fallback
    # moved both files and returned ok:true -> no 500 -> the else-branch
    # AssertionError. (The literal mid-copy failure injection — EXDEV + a
    # raising copytree — passed even against the broad helper because
    # shutil.move's copytree error propagates to the 500 wrapper; the
    # non-EXDEV error is the variant that bit against the broad fallback
    # and now pins the delivered narrowing, including the pairing
    # invariant image_count == caption_count.)
    _write_registry()
    _seed_dataset()
    image_dir = main._dataset_image_dir(MODEL_ID)
    # snapshot pre-existing single-image trash dirs (pin 4 leaves a FULL
    # legitimate copy in the same workspace) — only dirs created by THIS
    # attempt may be checked for partial content
    def _single_img_dirs():
        if not main.DATASET_TRASH.is_dir():
            return set()
        return {
            p
            for p in main.DATASET_TRASH.iterdir()
            if p.name.startswith(f"{MODEL_ID}-") and "-wipe-" not in p.name
        }

    before_dirs = _single_img_dirs()
    with _simulate_cross_device_trash(error=errno.EACCES):
        try:
            result = _run(main.delete_model_dataset_image(MODEL_ID, "a.png"))
        except main.HTTPException as exc:
            assert exc.status_code == 500, exc.status_code
            assert "could not move image to trash" in exc.detail, exc.detail
        else:
            raise AssertionError(
                "a non-EXDEV I/O error must 500 without the fallback; the "
                f"delete completed anyway: {result}"
            )
    assert (image_dir / "a.png").is_file() and (image_dir / "a.txt").is_file(), (
        "image AND paired caption must stay in images/ — no split state"
    )
    manifest = json.loads(
        main._dataset_manifest_path(MODEL_ID).read_text(encoding="utf-8")
    )
    assert [e["filename"] for e in manifest["images"]] == ["a.png"], (
        f"the manifest entry must be untouched on a refused move; got: {manifest}"
    )
    for d in _single_img_dirs() - before_dirs:
        assert list(d.iterdir()) == [], (
            f"no partial destination content may be left by a refused move: {d}"
        )


def test_existing_dst_is_rejected_loudly():
    # PIN D — COVERAGE pin, green by design (disclosed like pin A): the
    # validator's requirement 2, pinned at the helper level. A
    # pre-existing trash destination must be REJECTED LOUDLY
    # (FileExistsError, errno EEXIST) — without the guard shutil.move
    # would silently NEST the source into the existing directory (and
    # os.replace onto a non-empty dir dies with a confusing ENOTEMPTY).
    # Its value: deleting the guard re-opens the silent-nesting hole.
    # Direct helper call: small real src file, dst dir pre-created under
    # TRASH_DIR with an occupant; after the refused move the source is
    # intact and the destination unchanged (no nesting, no overwrite).
    src = main.WORKSPACE / "models" / MODEL_ID / "dst-src.txt"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"PAYLOAD")
    dst = main.TRASH_DIR / "existing-dst"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "occupant.txt").write_bytes(b"OCCUPANT")
    try:
        main._move_to_trash(src, dst)
    except FileExistsError as exc:
        assert exc.errno == errno.EEXIST, exc
    else:
        raise AssertionError(
            "an existing trash destination must be rejected loudly "
            "(FileExistsError/EEXIST) — silent nesting is the hole the "
            "guard closes"
        )
    assert src.is_file() and src.read_bytes() == b"PAYLOAD", (
        "a refused move must leave the source intact"
    )
    assert [p.name for p in dst.iterdir()] == ["occupant.txt"], (
        "the pre-existing destination must be unchanged — no nesting, no overwrite"
    )
    src.unlink()
    (dst / "occupant.txt").unlink()
    dst.rmdir()


def test_exdev_copy_failure_surfaces_with_source_intact():
    # PIN E — COVERAGE pin, green by design (disclosed like A/D; validator
    # condition 1): EXDEV enters the REAL copy branch and the copy FAILS
    # mid-way -> the failure must surface LOUD (500), the source bytes stay
    # intact, the manifest stays unchanged, and the PARTIAL destination is
    # IDENTIFIED (exists, holding exactly the first copied file). Site
    # choice (disclosed): wipe_model_dataset — the richest observable state
    # (its manifest reset runs only AFTER a successful move) and a
    # deterministic two-file dir for mid-copy injection (sorted: a.png
    # lands in the partial tree, a.txt never copied). delete_model_post
    # would work too but adds no observable state beyond the post folder.
    # Injection: replace+rename EXDEV (genuine copytree branch, as pin A)
    # + a copytree that copies the first file then raises EIO. The error
    # propagates shutil.move -> _move_to_trash -> the endpoint's
    # except-OSError -> 500 «could not move dataset to trash».
    # NON-TRANSACTIONAL property pinned (not accidental): NO self-clean of
    # the partial duplicate — the trash keeps a <id>-wipe-<ts>/ dir with
    # only a.png; an operator sees and removes it. (Wipe's collision loop
    # means a retry moves into a NEW dst; delete_model, which has no
    # suffix loop, would trip the EEXIST guard on a same-second retry —
    # both consequences follow from this pinned state.) Value: deleting
    # the EXDEV-only gate or the raise would change the behavior — the
    # source-intact + manifest-unchanged + partial-dst-identified triple
    # is the contract.
    _write_registry()
    _seed_dataset()
    image_dir = main._dataset_image_dir(MODEL_ID)

    def _wipe_dirs():
        if not main.DATASET_TRASH.is_dir():
            return set()
        return {
            p
            for p in main.DATASET_TRASH.iterdir()
            if p.name.startswith(f"{MODEL_ID}-wipe-")
        }

    before = _wipe_dirs()
    with _simulate_cross_device_trash(patch_rename=True), _fail_copytree_after_first_file():
        try:
            result = _run(main.wipe_model_dataset(MODEL_ID, confirm=MODEL_ID))
        except main.HTTPException as exc:
            assert exc.status_code == 500, exc.status_code
            assert "could not move dataset to trash" in exc.detail, exc.detail
        else:
            raise AssertionError(
                f"a mid-copy EIO must surface as a loud 500; wipe returned: {result}"
            )
    assert (image_dir / "a.png").read_bytes() == b"fake-png-a", (
        "the source must stay byte-intact when the copy fails mid-way"
    )
    assert (image_dir / "a.txt").read_bytes() == b"caption a"
    manifest = json.loads(
        main._dataset_manifest_path(MODEL_ID).read_text(encoding="utf-8")
    )
    assert [e["filename"] for e in manifest["images"]] == ["a.png"], (
        f"the manifest reset must NOT have run (it follows the move); got: {manifest}"
    )
    new_dirs = _wipe_dirs() - before
    assert len(new_dirs) == 1, (
        f"expected exactly one partial destination, got: {[str(p) for p in new_dirs]}"
    )
    partial = next(iter(new_dirs))
    assert [p.name for p in partial.iterdir()] == ["a.png"], (
        "non-transactional: the partial duplicate stays IDENTIFIED in trash "
        "(first copied file only) — no silent self-clean"
    )
    assert (partial / "a.png").read_bytes() == b"fake-png-a"


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
