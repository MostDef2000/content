"""Import-safe tests for the Phase 3 group-composite helpers in
scripts/queue_workflow.py (feature 007, group-photos): the pure
_cover_crop (scale-to-cover + center-crop) and _compose_grid (the first
image defines the cell size, unequal-size members are cover-cropped,
unfilled tiles keep the neutral (230, 230, 230) canvas fill, canvas is
columns*cell_w x rows*cell_h with rows = ceil(N / columns)).

All fixtures are in-memory PIL images built with Image.new; nothing is
written to disk. pytest-compatible; pytest is not required.

Run either way:
    python -m pytest tests/test_group_composite.py -q
    python tests/test_group_composite.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.queue_workflow import _compose_grid, _cover_crop  # noqa: E402


A_COLOR = (10, 20, 90)
B_COLOR = (90, 20, 10)
C_COLOR = (10, 90, 20)
BG_COLOR = (30, 40, 50)
MARKER = (200, 100, 50)


def _tiled(size, color, marker=None) -> Image.Image:
    """Flat-color in-memory image, optionally with a centred marker square."""
    image = Image.new("RGB", size, color)
    if marker is not None:
        mark_size, mark_color = marker
        image.paste(
            Image.new("RGB", mark_size, mark_color),
            ((size[0] - mark_size[0]) // 2, (size[1] - mark_size[1]) // 2),
        )
    return image


def test_cover_crop_scales_and_centers():
    # 480x480 square with a 40x40 marker dead centre → 640x960 cover crop
    # is a uniform 2x scale with 160px cropped off each horizontal side.
    source = _tiled((480, 480), BG_COLOR, marker=((40, 40), MARKER))
    result = _cover_crop(source, 640, 960)
    assert result.size == (640, 960)
    # The centred marker lands at the result centre (2x of 240,240 minus
    # the 160px left crop).
    assert result.getpixel((320, 480)) == MARKER, "centre marker not preserved"
    # (350, 480) is inside the 2x-scaled marker (x 280..359) but outside the
    # x-range (293..346) a non-uniform stretch to 640x960 would leave — this
    # distinguishes scale-to-cover from stretching.
    assert result.getpixel((350, 480)) == MARKER, "not a uniform scale-to-cover"
    # Background outside the marker survives at the canvas corner.
    assert result.getpixel((5, 5)) == BG_COLOR


def test_compose_row_two_members():
    a = Image.new("RGB", (100, 200), A_COLOR)
    b = Image.new("RGB", (100, 200), B_COLOR)
    canvas = _compose_grid([a, b], 1)
    assert canvas.size == (100, 400)
    assert canvas.getpixel((50, 50)) == A_COLOR
    assert canvas.getpixel((50, 250)) == B_COLOR


def test_compose_grid2_three_members_empty_tile():
    a = Image.new("RGB", (100, 200), A_COLOR)
    b = Image.new("RGB", (100, 200), B_COLOR)
    c = Image.new("RGB", (100, 200), C_COLOR)
    canvas = _compose_grid([a, b, c], 2)
    # 3 members in 2 columns → 2 rows → 200x400 canvas.
    assert canvas.size == (200, 400)
    assert canvas.getpixel((50, 50)) == A_COLOR  # tile 1 at (0, 0)
    assert canvas.getpixel((150, 50)) == B_COLOR  # tile 2 at (100, 0)
    assert canvas.getpixel((50, 250)) == C_COLOR  # tile 3 at (0, 200)
    # tile 4 at (100, 200) is unfilled → neutral canvas fill
    assert canvas.getpixel((150, 300)) == (230, 230, 230)


def test_compose_mixed_sizes_cover_cropped():
    # The first member defines the cell (100x200); B is 3x larger in both
    # dimensions, so it must be scaled to cover the cell and centred — not
    # stretched and not pasted raw.
    a = Image.new("RGB", (100, 200), A_COLOR)
    b = _tiled((300, 600), (60, 70, 80), marker=((60, 60), MARKER))
    canvas = _compose_grid([a, b], 1)
    assert canvas.size == (100, 400)
    assert canvas.getpixel((50, 50)) == A_COLOR
    # B's centred 60x60 marker scales to 20x20 centred in the second tile;
    # the tile centre must be dominated by B's centre, not its edges.
    assert canvas.getpixel((50, 300)) == MARKER, "B centre must dominate its tile"
    # Tile edges keep B's background (content from B's border region).
    assert canvas.getpixel((5, 300)) == (60, 70, 80)
    assert canvas.getpixel((50, 205)) == (60, 70, 80)


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
