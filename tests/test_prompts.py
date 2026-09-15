"""Import-safe tests for prompts.py (pytest-compatible; pytest not required).

Run either way:
    python -m pytest tests/ -q
    python tests/test_prompts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import prompts  # noqa: E402


def test_guardrail_phrase_injected_in_all_modes():
    for mode in prompts.MODES:
        text = prompts.build_positive(name="Valery", age=23, mode=mode)
        assert "23-year-old fictional" in text, f"mode={mode}: {text}"


def test_age_floor():
    assert prompts.guardrail_phrase(17) == "23-year-old fictional"
    assert prompts.guardrail_phrase(23) == "23-year-old fictional"
    assert prompts.guardrail_phrase(29) == "29-year-old fictional"


def test_validate_positive_flags_blocked_terms():
    assert prompts.validate_positive("teen girl"), "expected violations for 'teen girl'"
    assert not prompts.validate_positive("a healthy woman")


def test_validate_positive_word_boundary_no_false_positives():
    assert prompts.validate_positive("kidney bean") == []
    assert prompts.validate_positive("canteen") == []
    assert prompts.validate_positive("realistic anatomy") == []


def test_validate_positive_blocks_minors_age_and_words():
    for text in ("18-year-old girl", "16-year-old", "preteen", "school girl"):
        hits = prompts.validate_positive(text)
        assert hits, f"expected violations for {text!r}"
    # adult ages (>= AGE_FLOOR) and clean prompts must not be rejected
    assert prompts.validate_positive("23-year-old fictional woman") == []
    assert prompts.validate_positive("29-year-old") == []
    assert prompts.validate_positive("a confident woman") == []


def test_build_negative_keeps_canonical_terms():
    base = prompts.build_negative()
    assert "real person" in base
    assert "child" in base
    with_user = prompts.build_negative("some user text")
    assert "real person" in with_user
    assert "child" in with_user
    # user negative cannot subtract canonical terms
    for term in (t.strip() for t in prompts.CANONICAL_NEGATIVE.split(",")):
        assert term in with_user, f"canonical term lost: {term}"
    # word-level dedupe: "text" is already canonical, so only new words survive
    assert "some user" in with_user
    assert "some user text" not in with_user


def test_build_negative_word_dedupe_preserves_order():
    merged = prompts.build_negative("blurry, extra fingers, soft focus")
    assert merged.index("child") < merged.index("real person")
    assert "soft focus" in merged
    # duplicate words from the user side are dropped, canonical terms stay first
    assert merged.index("child") < merged.index("soft focus")
    tail = merged.split("soft focus", 1)[1]
    assert "blurry" not in tail and "extra fingers" not in tail


def test_build_positive_order_and_guardrail():
    text = prompts.build_positive(
        name="Valery",
        age=23,
        face="oval face",
        body="athletic build",
        style="fitness fashion",
        mode="post",
        scene="gym",
    )
    assert "23-year-old fictional" in text
    for needle in ("oval face", "athletic build", "Instagram fashion photograph", "fitness fashion", "gym"):
        assert needle in text, needle
    # order: identity -> face -> body -> mode frame -> style -> scene -> tail
    assert text.index("23-year-old fictional") < text.index("oval face")
    assert text.index("oval face") < text.index("athletic build")
    assert text.index("athletic build") < text.index("Instagram fashion photograph")
    assert text.index("Instagram fashion photograph") < text.index("fitness fashion")
    assert text.index("fitness fashion") < text.index("gym")
    assert text.endswith(prompts.POSITIVE_TAIL)


def test_build_positive_rejects_blocked_input():
    try:
        prompts.build_positive(name="Valery", age=23, mode="post", scene="teen girl")
    except prompts.GuardrailError:
        pass
    else:
        raise AssertionError("GuardrailError expected for blocked scene text")


def test_unknown_mode_rejected():
    for call in (
        lambda: prompts.build_positive(name="Valery", age=23, mode="video"),
        lambda: prompts.build_negative(mode="video"),
        lambda: prompts.scene_presets("video"),
    ):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("ValueError expected for unknown mode")


def test_default_profile_shape():
    character = {
        "appearance": {"face": "oval face", "build": "athletic build"},
        "style": "editorial",
        "body": "athletic build",
    }
    profile = prompts.default_profile(character)
    assert profile["version"] == 1
    assert profile["face"] == "oval face"
    assert profile["body"] == "athletic build"
    assert profile["style"] == "editorial"
    assert profile["negative"] == ""
    assert "casting" in profile["scenes"]


def test_scene_presets_nonempty_with_expand():
    presets = prompts.scene_presets()
    assert presets, "scene library must not be empty"
    assert any(preset["mode"] == "expand" for preset in presets)
    assert all(set(preset) == {"id", "name", "mode", "text", "tags"} for preset in presets)
    expand_only = prompts.scene_presets("expand")
    assert expand_only and all(preset["mode"] == "expand" for preset in expand_only)


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
