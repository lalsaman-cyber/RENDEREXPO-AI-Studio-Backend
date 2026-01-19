# app/presets_sd35.py
"""
RENDEREXPO AI STUDIO - SD3.5 Presets (Doc 18 locked logic + optional upscale switch)

Key rules:
- Presets lock: steps, CFG, LyCORIS multiplier, GEO multiplier.
- Denoise is ALWAYS 0.0 (no denoise anywhere).
- Upscale is OPTIONAL and can be overridden per request (upscale_2x true/false).
- This module must be used by ALL features: txt2img, img2img, sketch, etc.

Design:
- Routers create a "meta" dict (type-specific fields).
- Then call apply_preset_to_meta(meta, category, shot, upscale_2x=?).
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

    # Core generation controls
    steps: int
    cfg: float

    # LyCORIS / GEO controls (Doc 18)
    lycoris_path: str
    lycoris_multiplier: float
    geo_path: str
    geo_multiplier: float

    # Upscale behavior (optional)
    upscale_default: bool
    upscale_factor: int = 2

    # Hard lock: NO denoise anywhere
    denoise: float = 0.0


# ---------------------------------------------------------------------
# IMPORTANT:
# Replace lycoris_path / geo_path with your real files if needed.
# Multipliers below must match Doc 18 locked values.
# ---------------------------------------------------------------------

PRESETS: Dict[str, SD35Preset] = {
    # URBAN
    "urban:wide": SD35Preset(
        category="urban",
        shot="wide",
        steps=46,
        cfg=5.6,
        lycoris_path="models/lycoris/RENDEREXPO_PRO21.safetensors",
        lycoris_multiplier=0.05,
        geo_path="models/geo/RENDEREXPO_GEO.safetensors",
        geo_multiplier=0.010,
        upscale_default=True,
    ),
    "urban:close": SD35Preset(
        category="urban",
        shot="close",
        steps=48,
        cfg=6.0,
        lycoris_path="models/lycoris/RENDEREXPO_PRO21.safetensors",
        lycoris_multiplier=0.05,
        geo_path="models/geo/RENDEREXPO_GEO.safetensors",
        geo_multiplier=0.010,
        upscale_default=False,
    ),

    # SUBURBAN
    "suburban:wide": SD35Preset(
        category="suburban",
        shot="wide",
        steps=46,
        cfg=5.6,
        lycoris_path="models/lycoris/RENDEREXPO_PRO21.safetensors",
        lycoris_multiplier=0.05,
        geo_path="models/geo/RENDEREXPO_GEO.safetensors",
        geo_multiplier=0.010,
        upscale_default=True,
    ),
    "suburban:close": SD35Preset(
        category="suburban",
        shot="close",
        steps=48,
        cfg=6.0,
        lycoris_path="models/lycoris/RENDEREXPO_PRO21.safetensors",
        lycoris_multiplier=0.05,
        geo_path="models/geo/RENDEREXPO_GEO.safetensors",
        geo_multiplier=0.010,
        upscale_default=False,
    ),

    # INTERIOR
    "interior:wide": SD35Preset(
        category="interior",
        shot="wide",
        steps=46,
        cfg=5.6,
        lycoris_path="models/lycoris/RENDEREXPO_PRO21.safetensors",
        lycoris_multiplier=0.05,
        geo_path="models/geo/RENDEREXPO_GEO.safetensors",
        geo_multiplier=0.010,
        upscale_default=True,
    ),
    "interior:close": SD35Preset(
        category="interior",
        shot="close",
        steps=48,
        cfg=6.0,
        lycoris_path="models/lycoris/RENDEREXPO_PRO21.safetensors",
        lycoris_multiplier=0.05,
        geo_path="models/geo/RENDEREXPO_GEO.safetensors",
        geo_multiplier=0.010,
        upscale_default=False,
    ),

    # WIDE HERO
    "wide_hero:wide": SD35Preset(
        category="wide_hero",
        shot="wide",
        steps=46,
        cfg=5.6,
        lycoris_path="models/lycoris/RENDEREXPO_PRO21.safetensors",
        lycoris_multiplier=0.05,
        geo_path="models/geo/RENDEREXPO_GEO.safetensors",
        geo_multiplier=0.010,
        upscale_default=True,
    ),
}


def resolve_preset(category: str, shot: str) -> SD35Preset:
    c = (category or "urban").strip().lower()
    s = (shot or "wide").strip().lower()
    key = f"{c}:{s}"

    if key not in PRESETS:
        raise ValueError(f"Unknown preset key '{key}'. Valid keys: {list(PRESETS.keys())}")

    return PRESETS[key]


def apply_preset_to_meta(
    meta: Dict[str, Any],
    *,
    category: str,
    shot: str,
    upscale_2x: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Mutates meta to inject locked preset logic.

    What it sets/overrides:
    - meta["num_inference_steps"]
    - meta["guidance_scale"]
    - meta["denoise"] = 0.0
    - meta["lora_config"] (LyCORIS PRO 2.1)
    - meta["geo_config"]
    - meta["upscale"] (optional; denoise forced 0.0)

    It DOES NOT force type/width/height/strength/etc. Those are feature-specific.
    """
    p = resolve_preset(category, shot)

    upscale_enabled = p.upscale_default if upscale_2x is None else bool(upscale_2x)

    # Debug/telemetry: lock record
    meta["preset"] = {
        "engine": "sd3.5-large",
        "pro": "PRO 2.1",
        "category": p.category,
        "shot": p.shot,
        "steps": p.steps,
        "cfg": p.cfg,
        "lycoris_multiplier": p.lycoris_multiplier,
        "geo_multiplier": p.geo_multiplier,
        "upscale_default": p.upscale_default,
    }

    # Locked core values
    meta["num_inference_steps"] = int(p.steps)
    meta["guidance_scale"] = float(p.cfg)

    # Locked: NO denoise anywhere
    meta["denoise"] = 0.0

    # Locked LyCORIS + GEO
    meta["lora_config"] = {
        "path": p.lycoris_path,
        "strength": p.lycoris_multiplier,
        "scale": p.lycoris_multiplier,
        "label": "LYCORIS_PRO21",
    }
    meta["geo_config"] = {
        "path": p.geo_path,
        "strength": p.geo_multiplier,
        "scale": p.geo_multiplier,
        "label": "GEO",
    }

    # Optional upscale (deterministic resize; no diffusion)
    meta["upscale"] = {
        "enabled": upscale_enabled,
        "factor": int(p.upscale_factor),
        "denoise": 0.0,
        "method": "lanczos",
    }

    return meta
