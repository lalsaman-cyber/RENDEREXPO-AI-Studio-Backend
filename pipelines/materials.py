"""
pipelines/materials.py

Central material preset library for RENDEREXPO AI STUDIO.

These are *prompt helpers* only:
- No models, no weights, no extra licenses.
- Each material is just a text snippet + metadata.
- Frontend can show these as selectable chips, cards, dropdowns.
- Backend simply appends the `prompt_snippet` to the user prompt when
  `material_key` is provided in /v1/txt2img or /v1/img2img (via main.py).

Everything here is safe, commercial-friendly, and model-agnostic.
"""

from typing import Dict, Any

# Master dict: MATERIAL_LIBRARY
# Keys are stable API keys (no spaces, lowercase, snake_case).
MATERIAL_LIBRARY: Dict[str, Dict[str, Any]] = {
    # -----------------------------
    # WOOD
    # -----------------------------
    "dark_walnut": {
        "label": "Dark Walnut Wood",
        "category": "wood",
        "description": "Rich dark walnut with visible grain and semi-gloss finish.",
        "prompt_snippet": (
            "dark walnut wood, rich brown tone, visible linear grain, "
            "semi-gloss finish, high-end millwork quality"
        ),
        "tags": ["wood", "interior", "luxury", "warm"],
    },
    "light_oak": {
        "label": "Light Oak Wood",
        "category": "wood",
        "description": "Natural light oak with matte finish and subtle grain.",
        "prompt_snippet": (
            "light oak wood, natural matte finish, subtle grain, "
            "Scandinavian-style warmth"
        ),
        "tags": ["wood", "interior", "scandinavian", "minimal"],
    },
    "black_ash": {
        "label": "Black Stained Ash",
        "category": "wood",
        "description": "Deep black stained timber with visible wood texture.",
        "prompt_snippet": (
            "black stained ash wood, deep black tone, visible wood texture, "
            "contemporary minimal joinery"
        ),
        "tags": ["wood", "interior", "modern", "dark"],
    },

    # -----------------------------
    # STONE / TILES
    # -----------------------------
    "white_marble": {
        "label": "White Marble",
        "category": "stone",
        "description": "White marble with subtle grey veining, polished finish.",
        "prompt_snippet": (
            "white marble stone, subtle grey veins, polished reflective surface, "
            "luxury high-end finish"
        ),
        "tags": ["stone", "luxury", "interior", "lobby"],
    },
    "travertine_honed": {
        "label": "Honed Travertine",
        "category": "stone",
        "description": "Warm beige travertine with honed matte finish.",
        "prompt_snippet": (
            "honed travertine stone, warm beige tone, horizontal veining, "
            "matte finish, gallery-like calm atmosphere"
        ),
        "tags": ["stone", "interior", "gallery", "warm"],
    },
    "concrete_polished": {
        "label": "Polished Concrete",
        "category": "concrete",
        "description": "Smooth polished concrete floor with soft reflections.",
        "prompt_snippet": (
            "polished concrete floor, smooth surface, subtle reflections, "
            "contemporary industrial aesthetic"
        ),
        "tags": ["concrete", "floor", "interior", "industrial"],
    },
    "concrete_boardform": {
        "label": "Board-Form Concrete",
        "category": "concrete",
        "description": "Cast-in-place concrete with board-form texture.",
        "prompt_snippet": (
            "board-formed cast-in-place concrete, visible wood grain imprint, "
            "vertical plank texture, brutalist character"
        ),
        "tags": ["concrete", "exterior", "brutalist"],
    },
    "terracotta_tiles": {
        "label": "Terracotta Tiles",
        "category": "tile",
        "description": "Warm terracotta floor tiles with slight imperfections.",
        "prompt_snippet": (
            "terracotta floor tiles, warm earthy red-orange tone, "
            "slight imperfections, handcrafted feeling"
        ),
        "tags": ["tile", "warm", "mediterranean", "interior", "exterior"],
    },

    # -----------------------------
    # METAL
    # -----------------------------
    "brushed_steel": {
        "label": "Brushed Steel",
        "category": "metal",
        "description": "Brushed stainless steel with linear texture.",
        "prompt_snippet": (
            "brushed stainless steel, soft linear highlights, "
            "refined industrial look"
        ),
        "tags": ["metal", "cool", "modern"],
    },
    "black_metal": {
        "label": "Matte Black Metal",
        "category": "metal",
        "description": "Matte black metal for frames or fixtures.",
        "prompt_snippet": (
            "matte black metal, powder-coated finish, slim profiles, "
            "minimalist detailing"
        ),
        "tags": ["metal", "frames", "windows", "fixtures"],
    },
    "bronze_patina": {
        "label": "Bronze with Patina",
        "category": "metal",
        "description": "Aged bronze metal with rich patina.",
        "prompt_snippet": (
            "aged bronze metal, subtle patina, rich dark bronze tone, "
            "high-end hardware and detailing"
        ),
        "tags": ["metal", "luxury", "warm"],
    },

    # -----------------------------
    # GLASS
    # -----------------------------
    "clear_glass": {
        "label": "Clear Low-Iron Glass",
        "category": "glass",
        "description": "Clear low-iron glass with minimal green tint.",
        "prompt_snippet": (
            "clear low-iron glass, minimal green tint, high transparency, "
            "slim framing"
        ),
        "tags": ["glass", "facade", "windows"],
    },
    "frosted_glass": {
        "label": "Frosted Glass",
        "category": "glass",
        "description": "Soft frosted glass for privacy and diffuse light.",
        "prompt_snippet": (
            "frosted glass panels, diffused light, soft glow, "
            "privacy with gentle translucency"
        ),
        "tags": ["glass", "privacy", "interior"],
    },

    # -----------------------------
    # FABRIC / SOFT MATERIALS
    # -----------------------------
    "warm_fabric": {
        "label": "Warm Neutral Fabric",
        "category": "fabric",
        "description": "Warm neutral upholstery fabric with soft texture.",
        "prompt_snippet": (
            "warm neutral upholstery fabric, soft woven texture, "
            "comfortable and inviting seating"
        ),
        "tags": ["fabric", "sofa", "chairs", "interior"],
    },
    "cool_fabric": {
        "label": "Cool Grey Fabric",
        "category": "fabric",
        "description": "Cool grey fabric with smooth weave.",
        "prompt_snippet": (
            "cool grey fabric, smooth weave, contemporary minimal upholstery"
        ),
        "tags": ["fabric", "sofa", "chairs", "modern"],
    },

    # -----------------------------
    # EXTERIOR / LANDSCAPE SURFACES
    # -----------------------------
    "decking_timber": {
        "label": "Exterior Timber Decking",
        "category": "exterior",
        "description": "Exterior-rated timber decking with subtle gaps.",
        "prompt_snippet": (
            "exterior timber decking, linear planks, subtle gaps, "
            "weather-resistant finish"
        ),
        "tags": ["exterior", "deck", "wood", "landscape"],
    },
    "gravel_light": {
        "label": "Light Gravel",
        "category": "exterior",
        "description": "Light stone gravel for outdoor surfaces.",
        "prompt_snippet": (
            "light-colored gravel ground, small stones, clean and minimal landscape"
        ),
        "tags": ["exterior", "landscape", "ground"],
    },
    "asphalt_dark": {
        "label": "Dark Asphalt",
        "category": "exterior",
        "description": "Dark asphalt surface for roads and parking.",
        "prompt_snippet": (
            "dark asphalt surface, smooth road texture, subtle specular highlights"
        ),
        "tags": ["exterior", "road", "parking"],
    },
}


def get_materials() -> Dict[str, Dict[str, Any]]:
    """
    Optional helper if we want a function to return the library.
    main.py currently imports MATERIAL_LIBRARY directly.
    """
    return MATERIAL_LIBRARY


__all__ = ["MATERIAL_LIBRARY", "get_materials"]
