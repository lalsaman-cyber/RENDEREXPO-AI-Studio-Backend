"""
pipelines/presets.py

Central library of style / lighting / camera / mood / scene presets
for RENDEREXPO AI STUDIO.

These are *prompt helpers* only:
- The backend never forces them.
- The frontend can show them as dropdowns, pills, or cards.
- /v1/txt2img, /v1/img2img, /v1/architecture/scene, /v1/selective-edit
  may accept `style`, `lighting`, `camera`, `mood`, etc. and append the
  corresponding prompt snippets to the user prompt.

Everything here is 100% "text prompt" level — no models/licenses involved.
"""

from typing import Dict, Any


def _architecture_styles() -> Dict[str, Dict[str, Any]]:
    """
    High-level architectural style presets.
    Keys are meant to be stable API IDs (no spaces, lowercase).
    """
    return {
        "modern_minimal": {
            "label": "Modern Minimal",
            "description": "Clean lines, large openings, neutral palette, minimal ornament",
            "prompt_snippet": (
                "modern minimal architecture, clean lines, large floor-to-ceiling glazing, "
                "thin profiles, neutral palette, minimal ornament, integrated lighting"
            ),
            "negative_snippet": "cluttered, busy, ornate, low-detail, low-quality",
            "tags": ["architecture", "exterior", "interior", "minimal"],
        },
        "scandinavian": {
            "label": "Scandinavian Warm",
            "description": "Light woods, soft textiles, bright and cozy interiors",
            "prompt_snippet": (
                "Scandinavian interior, light oak floors, white walls, simple furniture, "
                "soft textiles, warm and cozy atmosphere"
            ),
            "negative_snippet": "dark gothic, heavy ornament, clutter, low-detail",
            "tags": ["architecture", "interior", "warm", "residential"],
        },
        "brutalist": {
            "label": "Brutalist Concrete",
            "description": "Exposed concrete, bold geometric forms, dramatic shadows",
            "prompt_snippet": (
                "brutalist architecture, exposed board-form concrete, massive volumes, "
                "dramatic shadows, minimal glazing"
            ),
            "negative_snippet": "cute, soft, colorful, cartoonish",
            "tags": ["architecture", "exterior", "bold", "concrete"],
        },
        "tropical_villa": {
            "label": "Tropical Villa",
            "description": "Open-plan villa with greenery, pools, and warm wood",
            "prompt_snippet": (
                "tropical resort villa, open plan, lush greenery, infinity pool, "
                "warm wood ceilings, outdoor-indoor living"
            ),
            "negative_snippet": "cold, sterile, snow, winter",
            "tags": ["architecture", "exterior", "landscape", "luxury"],
        },
        "parametric": {
            "label": "Parametric / Futuristic",
            "description": "Fluid, sculptural forms and bold futuristic details",
            "prompt_snippet": (
                "parametric architecture, fluid sculptural forms, futuristic facade, "
                "complex curved geometry, high-tech materials"
            ),
            "negative_snippet": "traditional, rustic, low-tech, simple box",
            "tags": ["architecture", "futuristic", "exterior", "concept"],
        },
        "heritage_classic": {
            "label": "Heritage Classic",
            "description": "Refined classical details and balanced proportions",
            "prompt_snippet": (
                "heritage classical architecture, balanced proportions, elegant moldings, "
                "stone or stucco facade, detailed window surrounds"
            ),
            "negative_snippet": "sci-fi, brutalist, hyper-modern",
            "tags": ["architecture", "exterior", "heritage"],
        },
    }


def _interior_styles() -> Dict[str, Dict[str, Any]]:
    return {
        "japandi": {
            "label": "Japandi Calm Interior",
            "description": "Blend of Japanese and Scandinavian minimalism.",
            "prompt_snippet": (
                "Japandi interior, low furniture, natural materials, light oak and linen, "
                "muted earthy tones, minimal decoration, calm atmosphere"
            ),
            "negative_snippet": "neon, cluttered, maximalist, cartoon",
            "tags": ["interior", "residential", "minimal", "warm"],
        },
        "luxury_hotel": {
            "label": "Luxury Hotel Lobby",
            "description": "High-end hospitality look with premium finishes.",
            "prompt_snippet": (
                "luxury hotel lobby, double-height space, premium stone flooring, "
                "feature lighting, sculptural furniture, refined detailing"
            ),
            "negative_snippet": "cheap, low-res, noisy, cluttered",
            "tags": ["interior", "hospitality", "luxury"],
        },
        "loft_industrial": {
            "label": "Industrial Loft",
            "description": "Exposed structure, brick, concrete, metal.",
            "prompt_snippet": (
                "industrial loft interior, exposed brick, exposed concrete ceiling, "
                "visible ductwork, large factory windows, raw materials"
            ),
            "negative_snippet": "plaster ornament, overly polished, clinical",
            "tags": ["interior", "residential", "industrial"],
        },
        "workspace_minimal": {
            "label": "Minimal Workspace",
            "description": "Clean office / studio environment.",
            "prompt_snippet": (
                "minimal contemporary workspace, clean desks, cable management, "
                "soft task lighting, neutral palette, focused environment"
            ),
            "negative_snippet": "messy, cluttered, chaotic layout",
            "tags": ["interior", "office", "workspace"],
        },
        "retail_gallery": {
            "label": "Gallery / Retail",
            "description": "Neutral shell highlighting products or artwork.",
            "prompt_snippet": (
                "gallery-like retail interior, white walls, accent lighting, "
                "clean display plinths, minimal branding"
            ),
            "negative_snippet": "overcrowded, noisy graphics, chaotic signage",
            "tags": ["interior", "retail", "gallery"],
        },
    }


def _landscape_site() -> Dict[str, Dict[str, Any]]:
    return {
        "desert_site": {
            "label": "Desert Site",
            "description": "Arid landscape, dunes or rocky terrain, sparse vegetation.",
            "prompt_snippet": (
                "desert landscape, sand dunes and rocky terrain, sparse vegetation, "
                "low shrubs, warm dry atmosphere"
            ),
            "negative_snippet": "dense forest, heavy rain, lush lawn",
            "tags": ["landscape", "site", "desert"],
        },
        "forest_cabin": {
            "label": "Forest Clearing",
            "description": "Evergreen or mixed forest clearing with tall trees.",
            "prompt_snippet": (
                "forest clearing, tall pines and mixed trees, filtered sunlight, "
                "soft moss, natural ground cover"
            ),
            "negative_snippet": "urban street, highway, desert",
            "tags": ["landscape", "site", "forest"],
        },
        "coastal_cliff": {
            "label": "Coastal Cliff",
            "description": "Cliffside site overlooking the sea, dramatic views.",
            "prompt_snippet": (
                "coastal cliff site, ocean horizon, rocky edge, dramatic drop, "
                "salt-tolerant vegetation"
            ),
            "negative_snippet": "city block, flat farmland",
            "tags": ["landscape", "site", "coastal"],
        },
        "urban_rooftop": {
            "label": "Urban Rooftop",
            "description": "Rooftop terrace overlooking city skyline.",
            "prompt_snippet": (
                "urban rooftop terrace, city skyline in background, parapet walls, "
                "planters, outdoor seating"
            ),
            "negative_snippet": "rural field, forest, mountain cabin",
            "tags": ["landscape", "urban", "roof"],
        },
        "courtyard": {
            "label": "Inner Courtyard",
            "description": "Enclosed courtyard with planting and seating.",
            "prompt_snippet": (
                "architectural inner courtyard, surrounding building facades, trees, "
                "paving patterns, integrated seating"
            ),
            "negative_snippet": "open field, generic park",
            "tags": ["landscape", "courtyard", "urban"],
        },
    }


def _lighting_presets() -> Dict[str, Dict[str, Any]]:
    return {
        "day_soft": {
            "label": "Soft Daylight",
            "description": "Overcast or softly diffused daylight.",
            "prompt_snippet": (
                "soft overcast daylight, gentle shadows, realistic global illumination"
            ),
            "negative_snippet": "harsh contrast, blown-out highlights",
            "tags": ["lighting", "day"],
        },
        "golden_hour": {
            "label": "Golden Hour",
            "description": "Warm low-angle sunlight, long shadows.",
            "prompt_snippet": (
                "golden hour lighting, warm low sun, long soft shadows, glowing sky"
            ),
            "negative_snippet": "flat lighting, midday harsh sun",
            "tags": ["lighting", "sunset", "exterior"],
        },
        "evening_warm": {
            "label": "Evening Interior Warm",
            "description": "Warm artificial lights, cozy indoor feel.",
            "prompt_snippet": (
                "evening interior lighting, warm color temperature, soft indirect lights, "
                "accent lamps, cozy atmosphere"
            ),
            "negative_snippet": "cold clinical white light, no shadows",
            "tags": ["lighting", "interior", "evening"],
        },
        "night_moody": {
            "label": "Night Moody",
            "description": "Low light, strong contrast, dramatic mood.",
            "prompt_snippet": (
                "night scene, moody lighting, deep shadows, focused highlights, "
                "cinematic contrast"
            ),
            "negative_snippet": "flat, evenly lit, washed out",
            "tags": ["lighting", "night", "cinematic"],
        },
        "studio_product": {
            "label": "Studio Product Lighting",
            "description": "Softbox-style clean lighting for objects or furniture.",
            "prompt_snippet": (
                "studio lighting, softbox, clean background, subtle reflections, "
                "high-quality product photography"
            ),
            "negative_snippet": "harsh flash, noisy background, clutter",
            "tags": ["lighting", "product", "furniture"],
        },
    }


def _camera_presets() -> Dict[str, Dict[str, Any]]:
    return {
        "eye_level": {
            "label": "Eye-Level View",
            "description": "Standard human eye-level camera for realistic views.",
            "prompt_snippet": "eye-level camera view, natural perspective",
            "tags": ["camera", "exterior", "interior"],
        },
        "aerial_3_4": {
            "label": "Aerial 3/4 View",
            "description": "High angle, looking down at the building and site.",
            "prompt_snippet": "aerial 3/4 view, high angle, showing building and site context",
            "tags": ["camera", "exterior", "site"],
        },
        "worm_eye": {
            "label": "Worm's Eye / Hero",
            "description": "Low-angle heroic view emphasizing height and drama.",
            "prompt_snippet": "low-angle hero shot, worm's eye view, dynamic composition",
            "tags": ["camera", "exterior", "dramatic"],
        },
        "wide_interior": {
            "label": "Wide Interior View",
            "description": "Wide-angle lens to show the whole room.",
            "prompt_snippet": "wide-angle interior photograph, showing the whole room",
            "tags": ["camera", "interior"],
        },
        "detail_closeup": {
            "label": "Detail Close-Up",
            "description": "Tight crop focusing on materials and junctions.",
            "prompt_snippet": "close-up detail view, focus on materials and junctions",
            "tags": ["camera", "detail", "materials"],
        },
    }


def _mood_presets() -> Dict[str, Dict[str, Any]]:
    return {
        "calm_residential": {
            "label": "Calm Residential",
            "description": "Quiet, livable mood, soft and welcoming.",
            "prompt_snippet": (
                "calm and welcoming residential atmosphere, soft and livable, "
                "subtle styling, no clutter"
            ),
            "tags": ["mood", "residential"],
        },
        "dramatic_concept": {
            "label": "Dramatic Concept",
            "description": "Bold, conceptual, eye-catching mood for competitions.",
            "prompt_snippet": (
                "dramatic concept rendering, bold composition, strong contrasts, "
                "competition-winning visual"
            ),
            "tags": ["mood", "concept", "competition"],
        },
        "cozy_evening": {
            "label": "Cozy Evening",
            "description": "Warm, intimate, comfortable atmosphere.",
            "prompt_snippet": (
                "cozy evening mood, warm light pools, intimate and comfortable feel"
            ),
            "tags": ["mood", "interior", "residential"],
        },
        "sleek_corporate": {
            "label": "Sleek Corporate",
            "description": "Clean, professional, polished visuals for commercial.",
            "prompt_snippet": (
                "sleek corporate mood, professional and polished, "
                "subtle branding, refined materials"
            ),
            "tags": ["mood", "office", "corporate"],
        },
    }


def _furniture_styles() -> Dict[str, Dict[str, Any]]:
    return {
        "contemporary_clean": {
            "label": "Contemporary Clean",
            "description": "Simple, modern furniture with clean lines.",
            "prompt_snippet": (
                "contemporary furniture, clean lines, neutral upholstery, "
                "simple and elegant"
            ),
            "tags": ["furniture", "interior"],
        },
        "midcentury": {
            "label": "Mid-century Modern",
            "description": "Iconic mid-century furniture pieces.",
            "prompt_snippet": (
                "mid-century modern furniture, tapered legs, warm wood, "
                "iconic designer pieces"
            ),
            "tags": ["furniture", "interior"],
        },
        "soft_lounge": {
            "label": "Soft Lounge",
            "description": "Deep sofas and soft lounge seating.",
            "prompt_snippet": (
                "soft lounge seating, deep sofas, comfortable cushions, "
                "relaxed layout"
            ),
            "tags": ["furniture", "interior", "lounge"],
        },
    }


def get_presets() -> Dict[str, Any]:
    """
    Master dictionary returned by /v1/presets.
    Frontend can read this and build dropdowns / pills from it.
    """
    return {
        "architecture_styles": _architecture_styles(),
        "interior_styles": _interior_styles(),
        "landscape_site": _landscape_site(),
        "lighting": _lighting_presets(),
        "camera": _camera_presets(),
        "mood": _mood_presets(),
        "furniture_styles": _furniture_styles(),
    }


# Optional alias if main.py imports a different name
def list_presets() -> Dict[str, Any]:
    return get_presets()


__all__ = ["get_presets", "list_presets"]
