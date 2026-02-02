# app/presets_sd35.py
"""
RENDEREXPO AI STUDIO - SD3.5 Presets (Doc 18 locked logic + optional upscale switch)

Goals:
- ONE centralized preset system that all routers can use.
- Presets lock: steps, CFG, LyCORIS PRO 2.1 multiplier, GEO multiplier, base resolution.
- Denoise is ALWAYS 0.0 (no denoise anywhere in any shot).
- Upscale is OPTIONAL:
    * preset default, OR
    * overridden per request via upscale_2x true/false.

How to use in any router:
1) Build meta dict (prompt, type, etc.)
2) Call:
       apply_preset_to_meta(meta, category=..., shot=..., upscale_2x=...)
3) Save meta.json
4) GPU runtime uses meta.json to run real inference later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Literal, Any


Category = Literal["urban", "suburban", "interior", "wide_hero"]
Shot = Literal["wide", "close"]


@dataclass(frozen=True)
class SD35Preset:
    category: Category
    shot: Shot

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

    # Hard lock: NO denoise anywhere
    denoise: float = 0.0


# ---------------------------------------------------------------------
# LOCKED PRESETS (Doc 18)
#
# NOTE:
# - "PRO" means PRO 2.1 (as you confirmed).
# - Paths below are placeholders unless you already have real files:
#     models/lycoris/RENDEREXPO_PRO21.safetensors
#     models/geo/RENDEREXPO_GEO.safetensors
#
# If your actual files are named differently, update only the PATHS,
# NOT the multipliers/steps/cfg unless Doc 18 says so.
# ---------------------------------------------------------------------

DEFAULT_LYCORIS_PATH = "models/lycoris/RENDEREXPO_PRO21.safetensors"
DEFAULT_GEO_PATH = "models/geo/RENDEREXPO_GEO.safetensors"

# If Doc 18 has different multipliers per category, set them per preset key.
DEFAULT_LYCORIS_MULT = 0.05
DEFAULT_GEO_MULT = 0.010

# Doc 18 locked steps/cfg targets:
# - Wide/Hero: CFG 5.6, Steps 44–46 (lock 46)
# - Close/Detail: CFG 6.0, Steps 46–50 (lock 48)
WIDE_STEPS = 46
WIDE_CFG = 5.6
CLOSE_STEPS = 48
CLOSE_CFG = 6.0

# Base resolution (locked for now)
BASE_W = 1024
BASE_H = 1024

PRESETS: Dict[str, SD35Preset] = {
    # URBAN
    "urban:wide": SD35Preset(
        category="urban",
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
    ),
    "urban:close": SD35Preset(
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
    ),

    # SUBURBAN
    "suburban:wide": SD35Preset(
        category="suburban",
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
    ),
    "suburban:close": SD35Preset(
        category="suburban",
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
    ),

    # INTERIOR
    "interior:wide": SD35Preset(
        category="interior",
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
    ),
    "interior:close": SD35Preset(
        category="interior",
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
    ),

    # WIDE HERO (always wide behavior)
    "wide_hero:wide": SD35Preset(
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
    ),
}


def resolve_preset(category: str, shot: str) -> SD35Preset:
    key = f"{category}:{shot}"
    if key not in PRESETS:
        raise ValueError(
            f"Unknown preset key '{key}'. Valid keys: {sorted(PRESETS.keys())}"
        )
    return PRESETS[key]


def apply_preset_to_meta(
    meta: Dict[str, Any],
    category: str,
    shot: str,
    upscale_2x: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Mutates + returns meta.

    Injects locked Doc 18 values into meta:
    - width/height
    - num_inference_steps
    - guidance_scale
    - denoise = 0.0
    - lora_config (LyCORIS PRO 2.1)
    - geo_config (GEO)
    - upscale config (enabled optional)
    """
    p = resolve_preset(category, shot)

    # Upscale decision: default per preset unless explicitly overridden
    upscale_enabled = p.upscale_default if upscale_2x is None else bool(upscale_2x)

    # ------------------------------------------------------------------
    # LOCKED core generation controls
    # ------------------------------------------------------------------
    meta["width"] = int(p.width)
    meta["height"] = int(p.height)
    meta["num_inference_steps"] = int(p.steps)
    meta["guidance_scale"] = float(p.cfg)

    # ------------------------------------------------------------------
    # HARD LOCK: NO denoise anywhere
    # ------------------------------------------------------------------
    meta["denoise"] = 0.0

    # Also lock any known denoise fields commonly used elsewhere
    # (safety: if other routers use these keys)
    if "strength" in meta:
        # strength is NOT denoise. Keep strength untouched.
        pass

    # ------------------------------------------------------------------
    # LyCORIS (PRO 2.1) + GEO configs
    # NOTE: runtime must actually apply geo_config too (we will wire it).
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
    # Upscale (OPTIONAL, deterministic: no diffusion denoise)
    # ------------------------------------------------------------------
    meta["upscale"] = {
        "enabled": bool(upscale_enabled),
        "factor": int(p.upscale_factor),
        "denoise": 0.0,
        "method": "lanczos",
    }

    # Helpful for traceability
    meta["preset"] = {
        "category": p.category,
        "shot": p.shot,
        "steps": p.steps,
        "cfg": p.cfg,
        "lycoris_multiplier": p.lycoris_multiplier,
        "geo_multiplier": p.geo_multiplier,
        "upscale_default": p.upscale_default,
        "upscale_enabled": bool(upscale_enabled),
        "denoise_locked": 0.0,
    }

    return meta
