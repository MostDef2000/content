"""Prompt assembly for the content project — pure module, no I/O.

Hard guardrail everywhere (Phase 1 redesign, specs/003): every positive prompt
contains the phrase ``{max(AGE_FLOOR, age)}-year-old fictional`` and all
positive inputs plus the final result are scanned for blocked terms (minors,
real people). Only adult fictional content is allowed: no real people, no
minors. The canonical negative prompt can never be subtracted by user input.
"""

from __future__ import annotations

import re

MODES = ("candidates", "expand", "post")
AGE_FLOOR = 23


class GuardrailError(ValueError):
    """Raised when a prompt would violate the adult-fictional guardrail."""


# Blocked positive-prompt terms: word-boundary regex, case-insensitive.
# Word boundaries prevent false positives like "kidney" (kid), "canteen" (teen).
GUARDRAIL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bchildren?\b", re.IGNORECASE),
    re.compile(r"\bkids?\b", re.IGNORECASE),
    re.compile(r"\bteens?\b", re.IGNORECASE),
    re.compile(r"\bteenage(?:r)?\b", re.IGNORECASE),
    re.compile(r"\bunderage\b", re.IGNORECASE),
    re.compile(r"\bminors?\b", re.IGNORECASE),
    re.compile(r"\bloli(?:con)?\b", re.IGNORECASE),
    re.compile(r"\bshota(?:con)?\b", re.IGNORECASE),
    re.compile(r"\bschoolgirls?\b", re.IGNORECASE),
    re.compile(r"\bschoolboys?\b", re.IGNORECASE),
    re.compile(r"\bpreteens?\b", re.IGNORECASE),
    # Adult content only: bare "girl"/"boy" are blocked ("schoolgirl" is covered
    # by its own pattern above — word boundaries prevent a double match).
    re.compile(r"\bgirls?\b", re.IGNORECASE),
    re.compile(r"\bboys?\b", re.IGNORECASE),
    re.compile(r"\breal\s+(?:persons?|people|womans?|mans?|humans?)\b", re.IGNORECASE),
    re.compile(r"\bcelebrit(?:y|ies)\b", re.IGNORECASE),
    re.compile(r"\bpoliticians?\b", re.IGNORECASE),
)

# Numeric age mentions below AGE_FLOOR ("18-year-old", "16 yo") are violations.
# The injected guardrail phrase itself ("23-year-old fictional") is >= AGE_FLOOR
# and never trips these.
AGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(\d{1,2})\s*(?:-| )?years?\s*-?\s*old\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,2})\s*yo\b", re.IGNORECASE),
)

# Mode frames (constant scaffolding, mirrors comfy/workflow_*.json semantics).
MODE_FRAMES = {
    "candidates": (
        "casting portrait, chest-up, neutral warm-gray studio, "
        "soft frontal beauty lighting, 85mm lens, eye-level camera, "
        "centered composition"
    ),
    "expand": (
        "same woman, identical face and identity, new outfit and pose, "
        "natural studio lighting, photorealistic"
    ),
    "post": "Instagram fashion photograph, premium editorial lighting",
}

POSITIVE_TAIL = "realistic anatomy, natural skin texture, no text, no logo, no other person"

CANONICAL_NEGATIVE = (
    "child, teen, teenage, underage, minor, loli, lolicon, shota, kid, "
    "young girl, young boy, real person, real people, celebrity, politician, "
    "photograph of a real human"
)

NEGATIVE_ALWAYS = "text, logo, watermark, extra fingers, deformed hands, blurry"

NEGATIVE_MODE_ADDITIONS = {
    "candidates": "jewelry, sunglasses, hat",
    "expand": "",
    "post": "oversaturated colors",
}


def guardrail_phrase(age: int) -> str:
    """Guardrail phrase with the age floor applied (23)."""
    return f"{max(AGE_FLOOR, int(age))}-year-old fictional"


def validate_positive(text: str) -> list[str]:
    """Return blocked terms found in ``text`` (empty list = OK)."""
    if not text:
        return []
    found: list[str] = []
    for pattern in GUARDRAIL_PATTERNS:
        for match in pattern.finditer(text):
            term = match.group(0).lower()
            if term not in found:
                found.append(term)
    # Numeric ages below the floor: report as "age N".
    for pattern in AGE_PATTERNS:
        for match in pattern.finditer(text):
            if int(match.group(1)) < AGE_FLOOR:
                term = f"age {match.group(1)}"
                if term not in found:
                    found.append(term)
    return found


def build_positive(
    *,
    name: str,
    age: int,
    face: str = "",
    body: str = "",
    style: str = "",
    mode: str,
    scene: str = "",
) -> str:
    """Assemble the positive prompt; always inject the guardrail phrase."""
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode!r}; expected one of {MODES}")
    if not name.strip():
        raise ValueError("name must not be empty")

    violations: list[str] = []
    for field, value in (
        ("name", name),
        ("face", face),
        ("body", body),
        ("style", style),
        ("scene", scene),
    ):
        hits = validate_positive(value)
        if hits:
            violations.append(f"{field}: {', '.join(hits)}")
    if violations:
        raise GuardrailError("blocked terms in positive prompt inputs — " + "; ".join(violations))

    phrase = guardrail_phrase(age)
    parts = [f"Photorealistic portrait of {name.strip()}, one {phrase} woman"]
    for value in (face, body):
        value = value.strip()
        if value:
            parts.append(value)
    parts.append(MODE_FRAMES[mode])
    for value in (style, scene):
        value = value.strip()
        if value:
            parts.append(value)
    parts.append(POSITIVE_TAIL)
    prompt = ", ".join(parts)

    # Final self-check: guardrail phrase present, result itself clean.
    if phrase not in prompt or validate_positive(prompt):
        raise GuardrailError("final positive prompt failed the adult-fictional guardrail")
    return prompt


def _terms(text: str) -> list[str]:
    return [term.strip() for term in text.split(",") if term.strip()]


def build_negative(user_negative: str = "", *, mode: str = "post") -> str:
    """Canonical negative + mode additions + user terms (word-level dedupe).

    Canonical protective terms always come first and cannot be removed or
    reordered by ``user_negative``.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode!r}; expected one of {MODES}")

    ordered: list[str] = [CANONICAL_NEGATIVE]
    seen: set[str] = set()
    for term in _terms(CANONICAL_NEGATIVE) + _terms(NEGATIVE_ALWAYS) + _terms(
        NEGATIVE_MODE_ADDITIONS.get(mode, "")
    ):
        if term.lower() not in seen:
            seen.add(term.lower())
            ordered.append(term)

    for term in _terms(user_negative):
        words = [word for word in term.split() if word.lower() not in seen]
        if not words:
            continue
        seen.update(word.lower() for word in words)
        ordered.append(" ".join(words))
    return ", ".join(ordered)


def default_profile(character: dict) -> dict:
    """Build the tracked prompt profile skeleton from a character dict."""
    return {
        "version": 1,
        "face": character["appearance"]["face"],
        "body": character.get("body", character["appearance"]["build"]),
        "style": character["style"],
        "negative": "",
        "scenes": {
            "casting": "Chest-up portrait, neutral warm-gray studio, soft frontal beauty lighting",
        },
    }


# Scene library (Phase 1): expand texts mirror VARIATION_PROMPTS in
# scripts/queue_workflow.py; candidates/post presets mirror the workflow JSON style.
_SCENE_PRESETS: list[dict] = [
    {"id": "expand-head-turn", "name": "Head turn, soft smile", "mode": "expand",
     "text": ("same fictional adult woman, identical face, slight head turn, soft smile, "
              "fitted olive athletic top, neutral studio, photorealistic portrait"),
     "tags": ["athletic", "studio"]},
    {"id": "expand-white-tee", "name": "White tee, daylight", "mode": "expand",
     "text": ("same woman, identical face, hair slightly pulled back, white fitted tee, "
              "bright daylight, casual confident pose"),
     "tags": ["casual", "daylight"]},
    {"id": "expand-gym", "name": "Gym athletic", "mode": "expand",
     "text": ("same woman, identical face, gym setting, black sports bra and leggings, "
              "tasteful athletic framing, strong posture"),
     "tags": ["gym", "athletic"]},
    {"id": "expand-rooftop", "name": "Evening rooftop", "mode": "expand",
     "text": ("same woman, identical face, evening rooftop, fitted dark fashion top, "
              "city lights bokeh, confident gaze"),
     "tags": ["evening", "urban"]},
    {"id": "expand-beauty", "name": "Beauty close-up", "mode": "expand",
     "text": ("same woman, identical face, close-up beauty shot, dewy skin, "
              "minimal makeup, shallow depth of field"),
     "tags": ["beauty", "close-up"]},
    {"id": "expand-street", "name": "Urban street style", "mode": "expand",
     "text": ("same woman, identical face, leaning against wall, beige crop top, "
              "relaxed fit, urban street style"),
     "tags": ["street", "urban"]},
    {"id": "expand-mirror", "name": "Fitness studio mirror", "mode": "expand",
     "text": ("same woman, identical face, fitness studio mirror, athletic wear, "
              "dynamic stance, professional lighting"),
     "tags": ["fitness", "studio"]},
    {"id": "expand-cafe", "name": "Outdoor cafe lifestyle", "mode": "expand",
     "text": ("same woman, identical face, outdoor cafe, light summer dress, "
              "natural smile, lifestyle editorial"),
     "tags": ["lifestyle", "editorial"]},
    {"id": "candidates-studio", "name": "Casting studio portrait", "mode": "candidates",
     "text": ("chest-up casting portrait, neutral warm-gray studio, soft frontal beauty "
              "lighting, 85mm lens, eye-level camera, centered symmetrical composition"),
     "tags": ["casting", "studio"]},
    {"id": "candidates-beauty", "name": "Casting beauty gaze", "mode": "candidates",
     "text": ("minimal luminous makeup, direct calm gaze, soft frontal lighting, "
              "shallow depth of field, premium editorial realism"),
     "tags": ["casting", "beauty"]},
    {"id": "post-instagram-fashion", "name": "Instagram fashion editorial", "mode": "post",
     "text": ("Instagram fashion photograph, premium editorial lighting, confident "
              "contemporary fitness fashion, tasteful and non-explicit"),
     "tags": ["post", "fashion"]},
    {"id": "post-gym-athletic", "name": "Gym editorial", "mode": "post",
     "text": ("gym setting, black sports bra and leggings, tasteful athletic framing, "
              "strong posture, premium editorial lighting"),
     "tags": ["post", "gym", "athletic"]},
]


def scene_presets(mode: str | None = None) -> list[dict]:
    """Scene library entries ({id,name,mode,text,tags}), optionally filtered by mode."""
    if mode is not None and mode not in MODES:
        raise ValueError(f"unknown mode: {mode!r}; expected one of {MODES}")
    if mode is None:
        return [dict(preset) for preset in _SCENE_PRESETS]
    return [dict(preset) for preset in _SCENE_PRESETS if preset["mode"] == mode]
