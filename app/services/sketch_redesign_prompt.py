from __future__ import annotations

"""
app/services/sketch_redesign_prompt.py

RENDEREXPO AI STUDIO
MistoLine-based Sketch to Redesign prompt builder

PURPOSE
-------
This file builds the prompt package for the NEW Sketch to Redesign mode.

IMPORTANT PRODUCT RULE
----------------------
- Sketch to Render stays untouched.
- Sketch to Redesign is a SEPARATE mode.
- Both modes can live in the same MistoLine / SDXL family.
- Redesign is more creative and less strict than Render.
- Redesign still starts from the uploaded sketch as the base structure,
  but it allows more reinterpretation in:
    * facade language
    * materials
    * balcony/detail style
    * architectural character
    * presentation mood

THIS FILE DOES NOT:
-------------------
- load any models
- call ComfyUI
- dispatch jobs
- write files
- alter the working Sketch to Render flow

THIS FILE ONLY:
---------------
- accepts allowed client-facing redesign inputs
- builds a locked redesign-oriented prompt
- builds a stable negative prompt
- returns a clean prompt package for downstream services/router logic
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional


# ---------------------------------------------------------------------
# Locked public warning / product text
# ---------------------------------------------------------------------

SKETCH_REDESIGN_WARNING_TEXT = (
    "Sketch to Redesign is more creative and less strict than Sketch to Render. "
    "It uses your sketch as a starting point, but exact structure preservation is not guaranteed."
)

SKETCH_REDESIGN_PRODUCT_PROMISE = (
    "Uses your sketch as a base while exploring alternative facade, material, "
    "and architectural design directions."
)


# ---------------------------------------------------------------------
# Style preset map
# These are redesign-facing style families.
# They affect tone, material language, and design character,
# not raw backend engine settings.
# ---------------------------------------------------------------------

STYLE_PRESET_BLOCKS: Dict[str, str] = {
    "contemporary_minimal": (
        "contemporary minimal architectural redesign, clean planar facade language, "
        "restrained detailing, elegant material transitions, premium modern residential character"
    ),
    "warm_contemporary": (
        "warm contemporary architectural redesign, balanced natural materials, "
        "soft premium facade expression, refined modern residential character"
    ),
    "luxury_modern": (
        "luxury modern architectural redesign, premium facade composition, "
        "high-end residential detailing, elegant proportions, sophisticated material palette"
    ),
    "clean_urban_residential": (
        "clean urban residential redesign, crisp facade articulation, "
        "developer-grade premium realism, orderly composition, modern city-residential character"
    ),
    "soft_scandinavian": (
        "soft Scandinavian-inspired redesign, clean modern simplicity, "
        "calm material palette, bright restrained architectural character"
    ),
    "natural_stone_contemporary": (
        "natural stone contemporary redesign, elegant stone-led facade language, "
        "premium textured realism, modern refined residential character"
    ),
    "premium_developer_marketing": (
        "premium developer-marketing architectural redesign, polished facade presentation, "
        "commercially appealing realism, clean upscale residential image"
    ),
    "daylight_real_estate_standard": (
        "high-quality daylight real-estate redesign, realistic facade presentation, "
        "clear market-ready residential visual language"
    ),
}


# ---------------------------------------------------------------------
# Locked base and negative prompt blocks
# ---------------------------------------------------------------------

LOCKED_BASE_REDESIGN_PROMPT = (
    "architectural redesign concept render based on the uploaded sketch, "
    "photorealistic architectural visualization, realistic facade materialization, "
    "credible massing and perspective continuity from the source sketch, "
    "high-quality architectural presentation, realistic daylight, site realism, "
    "refined residential design expression"
)

LOCKED_REDESIGN_FREEDOM_BLOCK = (
    "allow reinterpretation of facade style, balcony language, material palette, "
    "architectural detailing, presentation mood, and design character while still using "
    "the uploaded sketch as the structural starting point"
)

LOCKED_NEGATIVE_PROMPT = (
    "line drawing, sketch look, blueprint look, monochrome sketch, unfinished drawing, "
    "raw line art, painterly, cartoon, lowres, blurry, smeared facade, distorted geometry, "
    "warped windows, melted balconies, deformed building form, broken perspective, "
    "messy composition, chaotic massing, ugly facade, unrealistic materials, noisy image, "
    "ghosting, duplicated elements, extra floors, extra windows, missing openings, "
    "uncontrolled redesign, random structure collapse"
)


# ---------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SketchRedesignPromptPackage:
    mode: str
    product_promise: str
    warning_text: str
    prompt: str
    negative_prompt: str
    style_preset: str
    style_preset_applied: bool
    allowed_client_fields: Dict[str, str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _clean(value: Optional[str]) -> str:
    return " ".join(str(value or "").strip().split())


def _is_blank(value: Optional[str]) -> bool:
    return _clean(value) == ""


def _normalize_style_preset(value: Optional[str]) -> str:
    raw = _clean(value).lower()
    if raw in STYLE_PRESET_BLOCKS:
        return raw
    return "contemporary_minimal"


def _join_blocks(*parts: str) -> str:
    cleaned = []
    for part in parts:
        text = _clean(part)
        if text:
            cleaned.append(text.rstrip(", "))
    return ", ".join(cleaned)


# ---------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------

def build_sketch_redesign_prompt_package(
    *,
    style_preset: Optional[str] = None,
    materials_notes: Optional[str] = None,
    atmosphere_notes: Optional[str] = None,
    background_notes: Optional[str] = None,
    mood_notes: Optional[str] = None,
    style_notes: Optional[str] = None,
    aesthetic_notes: Optional[str] = None,
) -> SketchRedesignPromptPackage:
    """
    Build the redesign-mode prompt package.

    Client-facing allowed inputs for redesign:
    - style_preset
    - materials_notes
    - atmosphere_notes
    - background_notes
    - mood_notes
    - style_notes
    - aesthetic_notes

    Notes:
    - Redesign is intentionally looser than Sketch to Render.
    - This prompt should encourage reinterpretation, not strict 1:1 preservation.
    - The uploaded sketch is still the structural starting point, but exact fidelity
      is not guaranteed in this mode.
    """
    normalized_preset = _normalize_style_preset(style_preset)
    preset_block = STYLE_PRESET_BLOCKS[normalized_preset]

    user_materials = _clean(materials_notes)
    user_atmosphere = _clean(atmosphere_notes)
    user_background = _clean(background_notes)
    user_mood = _clean(mood_notes)
    user_style = _clean(style_notes)
    user_aesthetic = _clean(aesthetic_notes)

    prompt = _join_blocks(
        LOCKED_BASE_REDESIGN_PROMPT,
        preset_block,
        LOCKED_REDESIGN_FREEDOM_BLOCK,
        f"materials direction: {user_materials}" if user_materials else "",
        f"atmosphere direction: {user_atmosphere}" if user_atmosphere else "",
        f"background/site direction: {user_background}" if user_background else "",
        f"mood/lighting direction: {user_mood}" if user_mood else "",
        f"style direction: {user_style}" if user_style else "",
        f"aesthetic direction: {user_aesthetic}" if user_aesthetic else "",
    )

    return SketchRedesignPromptPackage(
        mode="sketch_to_redesign",
        product_promise=SKETCH_REDESIGN_PRODUCT_PROMISE,
        warning_text=SKETCH_REDESIGN_WARNING_TEXT,
        prompt=prompt,
        negative_prompt=LOCKED_NEGATIVE_PROMPT,
        style_preset=normalized_preset,
        style_preset_applied=True,
        allowed_client_fields={
            "style_preset": normalized_preset,
            "materials_notes": user_materials,
            "atmosphere_notes": user_atmosphere,
            "background_notes": user_background,
            "mood_notes": user_mood,
            "style_notes": user_style,
            "aesthetic_notes": user_aesthetic,
        },
    )