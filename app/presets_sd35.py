# app/presets_sd35.py
"""
RENDEREXPO AI STUDIO - SD3.5 Presets

Goals:
- ONE centralized preset system that all planner routers can use.
- Presets lock: steps, CFG, LyCORIS PRO 2.1 multiplier, GEO multiplier, base resolution.
- Upscale is OPTIONAL:
    * preset default, OR
    * overridden per request via upscale_2x true/false.
- This file is planner-side only.
- Planner writes meta; GPU runtime executes meta.

IMPORTANT:
- This file does NOT load models.
- This file does NOT decide ports.
- This file does NOT perform inference.
- This file should not silently destroy img2img behavior.

CRITICAL RULE:
- Do NOT globally force denoise/strength to 0.0 here.
- Text2img does not use denoise.
- Img2img/reclad depends on strength, and that must remain router-controlled.
- Only deterministic upscale uses denoise=0.0 because it is non-diffusion resize.

Canonical profile names:
- r1_wide_hero
- r1_close_detail
- luxury_interior_heavy_detail

Backward-compatible category/shot aliases are also supported:
- urban:wide
- urban:close
- suburban:wide
- suburban:close
- interior:wide
- interior:close
- wide_hero:wide

RENDEREXPO img2img rule:
- Presets still define the DEFAULT working resolution.
- But planner/runtime may follow INPUT aspect ratio for img2img unless caller explicitly specifies otherwise.
- Therefore this file should mark preset resolution as DEFAULT/INHERITED, not as a forced user override.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional


Category = Literal["urban", "suburban", "interior", "wide_hero"]
Shot = Literal["wide", "close"]


@dataclass(frozen=True)
class SD35Preset:
    profile_name: str
    category: str
    shot: str

    # Core generation controls (LOCKED)
    width: int
    height: int
    steps: int
    cfg: float

    # LyCORIS (PRO 2.1) controls (LOCKED)
    lycoris_path: str
    lycoris_multiplier: float

    # GEO controls (LOCKED)
    geo_path: str
    geo_multiplier: float

    # Upscale behavior (OPTIONAL)
    upscale_default: bool
    upscale_factor: int = 2
    upscale_method: str = "lanczos"


# ---------------------------------------------------------------------
# LOCKED PRESET CONSTANTS
# ---------------------------------------------------------------------

DEFAULT_LYCORIS_PATH = "models/lycoris/RENDEREXPO_PRO21.safetensors"
DEFAULT_GEO_PATH = "models/geo/RENDEREXPO_GEO.safetensors"

DEFAULT_LYCORIS_MULT = 0.05
DEFAULT_GEO_MULT = 0.010

# Locked Doc 18 / later production values
WIDE_STEPS = 46
WIDE_CFG = 5.6

CLOSE_STEPS = 48
CLOSE_CFG = 6.0

HEAVY_INTERIOR_STEPS = 60
HEAVY_INTERIOR_CFG = 6.0

BASE_W = 1024
BASE_H = 1024


# ---------------------------------------------------------------------
# CANONICAL PROFILES
# ---------------------------------------------------------------------

PROFILES: Dict[str, SD35Preset] = {
    "r1_wide_hero": SD35Preset(
        profile_name="r1_wide_hero",
        category="wide_hero",
        shot="wide",
        width=BASE_W,
        height=BASE_H,
        steps=WIDE_STEPS,
        cfg=WIDE_CFG,
        lycoris_path=DEFAULT_LYCORIS_PATH,
        lycoris_multiplier=DEFAULT_LYCORIS_MULT,
        geo_path=DEFAULT_GEO_PATH,
        geo_multiplier=DEFAULT_GEO_MULT,
        upscale_default=True,
        upscale_factor=2,
        upscale_method="lanczos",
    ),
    "r1_close_detail": SD35Preset(
        profile_name="r1_close_detail",
        category="urban",
        shot="close",
        width=BASE_W,
        height=BASE_H,
        steps=CLOSE_STEPS,
        cfg=CLOSE_CFG,
        lycoris_path=DEFAULT_LYCORIS_PATH,
        lycoris_multiplier=DEFAULT_LYCORIS_MULT,
        geo_path=DEFAULT_GEO_PATH,
        geo_multiplier=DEFAULT_GEO_MULT,
        upscale_default=False,
        upscale_factor=2,
        upscale_method="lanczos",
    ),
    "luxury_interior_heavy_detail": SD35Preset(
        profile_name="luxury_interior_heavy_detail",
        category="interior",
        shot="close",
        width=BASE_W,
        height=BASE_H,
        steps=HEAVY_INTERIOR_STEPS,
        cfg=HEAVY_INTERIOR_CFG,
        lycoris_path=DEFAULT_LYCORIS_PATH,
        lycoris_multiplier=DEFAULT_LYCORIS_MULT,
        geo_path=DEFAULT_GEO_PATH,
        geo_multiplier=DEFAULT_GEO_MULT,
        upscale_default=True,
        upscale_factor=2,
        upscale_method="lanczos",
    ),
}


# ---------------------------------------------------------------------
# BACKWARD-COMPATIBLE ALIASES
# ---------------------------------------------------------------------

ALIASES: Dict[str, str] = {
    # Wide behavior
    "urban:wide": "r1_wide_hero",
    "suburban:wide": "r1_wide_hero",
    "interior:wide": "r1_wide_hero",
    "wide_hero:wide": "r1_wide_hero",

    # Close/detail behavior
    "urban:close": "r1_close_detail",
    "suburban:close": "r1_close_detail",
    "interior:close": "r1_close_detail",
}


def resolve_preset(
    category: Optional[str] = None,
    shot: Optional[str] = None,
    profile: Optional[str] = None,
) -> SD35Preset:
    """
    Resolve a preset by canonical profile name or by category/shot alias.

    Preferred:
        resolve_preset(profile="r1_wide_hero")

    Backward compatible:
        resolve_preset(category="urban", shot="wide")
    """
    if profile:
        key = str(profile).strip()
        if key in PROFILES:
            return PROFILES[key]
        raise ValueError(
            f"Unknown profile '{key}'. Valid profiles: {sorted(PROFILES.keys())}"
        )

    if not category or not shot:
        raise ValueError("Either 'profile' or both 'category' and 'shot' must be provided.")

    alias = f"{category}:{shot}"
    mapped = ALIASES.get(alias)
    if not mapped:
        raise ValueError(
            f"Unknown preset alias '{alias}'. Valid aliases: {sorted(ALIASES.keys())}"
        )

    return PROFILES[mapped]


def _set_default_resolution_metadata(meta: Dict[str, Any], preset: SD35Preset) -> None:
    """
    Record preset resolution as the DEFAULT planner working size.

    IMPORTANT:
    - This is not the same thing as an explicit user override.
    - Img2img/inpaint runtime may auto-follow the input aspect ratio when
      dimensions were not explicitly requested by caller.
    """
    meta["width"] = int(preset.width)
    meta["height"] = int(preset.height)

    # Traceability flags for routers/runtime.
    meta["resolution_policy"] = "preset_default"
    meta["preset_resolution"] = {
        "width": int(preset.width),
        "height": int(preset.height),
        "source": "preset_default",
    }

    # Preserve any router decision if already present; otherwise default to False.
    if "explicit_dimensions" not in meta:
        meta["explicit_dimensions"] = False

    # Preserve any router decision if already present; otherwise default to True
    # because default preset size for img2img should be treated as follow-input
    # unless caller explicitly overrides.
    if "preserve_input_aspect_ratio" not in meta:
        meta["preserve_input_aspect_ratio"] = True


def apply_preset_to_meta(
    meta: Dict[str, Any],
    category: Optional[str] = None,
    shot: Optional[str] = None,
    profile: Optional[str] = None,
    upscale_2x: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Mutates + returns meta.

    Injects locked values into meta:
    - width/height (as DEFAULT preset resolution, not implicit user override)
    - num_inference_steps
    - guidance_scale
    - lora_config (LyCORIS PRO 2.1)
    - geo_config (GEO)
    - upscale config (enabled optional)

    IMPORTANT:
    - This function does NOT overwrite img2img strength/denoise.
    - Routers must preserve strength for reclad/img2img flows.
    - This function should not silently force square output for img2img when
      caller did not explicitly request square dimensions.
    """
    p = resolve_preset(category=category, shot=shot, profile=profile)

    upscale_enabled = p.upscale_default if upscale_2x is None else bool(upscale_2x)

    # ------------------------------------------------------------------
    # LOCKED core generation controls
    # ------------------------------------------------------------------
    _set_default_resolution_metadata(meta, p)
    meta["num_inference_steps"] = int(p.steps)
    meta["guidance_scale"] = float(p.cfg)

    # ------------------------------------------------------------------
    # Locked adapter configs
    # ------------------------------------------------------------------
    meta["lora_config"] = {
        "path": p.lycoris_path,
        "strength": float(p.lycoris_multiplier),
        "scale": float(p.lycoris_multiplier),
        "label": "LYCORIS_PRO21",
    }

    meta["geo_config"] = {
        "path": p.geo_path,
        "strength": float(p.geo_multiplier),
        "scale": float(p.geo_multiplier),
        "label": "GEO",
    }

    # ------------------------------------------------------------------
    # Upscale (OPTIONAL, deterministic resize)
    # ------------------------------------------------------------------
    meta["upscale"] = {
        "enabled": bool(upscale_enabled),
        "factor": int(p.upscale_factor),
        "method": str(p.upscale_method),
    }

    # ------------------------------------------------------------------
    # Helpful traceability
    # ------------------------------------------------------------------
    meta["preset"] = {
        "profile_name": p.profile_name,
        "category": p.category,
        "shot": p.shot,
        "steps": p.steps,
        "cfg": p.cfg,
        "lycoris_multiplier": p.lycoris_multiplier,
        "geo_multiplier": p.geo_multiplier,
        "upscale_default": p.upscale_default,
        "upscale_enabled": bool(upscale_enabled),
        "width": p.width,
        "height": p.height,
    }

    return meta