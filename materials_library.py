"""
Material library for RENDEREXPO AI STUDIO.

- Pure Python, no external dependencies.
- Architecture-focused, commercial-safe descriptions.
- Used by /v1/materials and /v1/selective-edit.
"""

from typing import Dict, List, Optional, Any

# -------------------------------------------------------------------
# Core data structure
# -------------------------------------------------------------------

# You can extend this over time. Keep it brand-free and generic.
MATERIAL_LIBRARY: Dict[str, Any] = {
    "categories": [
        {
            "id": "wood",
            "name": "Wood",
            "description": "Timber finishes for floors, walls, ceilings, and joinery.",
            "items": [
                {
                    "id": "wood_oak_light",
                    "label": "Light Oak",
                    "tags": ["warm", "modern", "interior"],
                    "prompt": (
                        "light oak wood, fine grain, warm tone, matte finish, "
                        "high-end interior joinery, soft reflections"
                    ),
                },
                {
                    "id": "wood_walnut_dark",
                    "label": "Dark Walnut",
                    "tags": ["luxury", "rich", "interior"],
                    "prompt": (
                        "dark walnut wood, rich brown tone, visible grain, "
                        "semi-gloss finish, premium millwork"
                    ),
                },
                {
                    "id": "wood_ash_bleached",
                    "label": "Bleached Ash",
                    "tags": ["minimal", "scandinavian"],
                    "prompt": (
                        "bleached ash wood, pale neutral tone, minimal grain, "
                        "smooth matte finish, nordic interior"
                    ),
                },
            ],
        },
        {
            "id": "stone",
            "name": "Stone & Tiles",
            "description": "Natural stone, tiles, and masonry surfaces.",
            "items": [
                {
                    "id": "stone_travertine_warm",
                    "label": "Warm Travertine",
                    "tags": ["interior", "wall", "floor"],
                    "prompt": (
                        "warm travertine stone, soft beige color, subtle veining, "
                        "honed finish, contemporary luxury interior"
                    ),
                },
                {
                    "id": "stone_concrete_raw",
                    "label": "Raw Cast Concrete",
                    "tags": ["brutalist", "facade", "structural"],
                    "prompt": (
                        "raw cast-in-place concrete, slight imperfections, "
                        "formwork lines, brutalist architecture, matte surface"
                    ),
                },
                {
                    "id": "tile_porcelain_large_format",
                    "label": "Large Porcelain Slab",
                    "tags": ["floor", "wall", "minimal"],
                    "prompt": (
                        "large-format porcelain tile, minimal joints, "
                        "matte neutral gray, clean contemporary interior"
                    ),
                },
            ],
        },
        {
            "id": "metal",
            "name": "Metals",
            "description": "Metal panels, trims, and details.",
            "items": [
                {
                    "id": "metal_brushed_aluminum",
                    "label": "Brushed Aluminum",
                    "tags": ["facade", "modern", "tech"],
                    "prompt": (
                        "brushed aluminum, linear texture, cool metallic sheen, "
                        "clean modern detailing, precise reflections"
                    ),
                },
                {
                    "id": "metal_black_steel",
                    "label": "Blackened Steel",
                    "tags": ["industrial", "loft", "frames"],
                    "prompt": (
                        "blackened steel, slightly reflective, deep dark tone, "
                        "industrial loft aesthetic"
                    ),
                },
                {
                    "id": "metal_bronze_warm",
                    "label": "Warm Bronze",
                    "tags": ["luxury", "decorative"],
                    "prompt": (
                        "warm bronze metal, subtle patina, soft reflections, "
                        "high-end architectural detailing"
                    ),
                },
            ],
        },
        {
            "id": "glass",
            "name": "Glass",
            "description": "Glazing and glass partitions.",
            "items": [
                {
                    "id": "glass_clear_low_iron",
                    "label": "Clear Low-Iron Glass",
                    "tags": ["facade", "interior", "transparent"],
                    "prompt": (
                        "clear low-iron glass, minimal green tint, high transparency, "
                        "crisp reflections, modern glazing"
                    ),
                },
                {
                    "id": "glass_frosted_privacy",
                    "label": "Frosted Glass",
                    "tags": ["partition", "privacy"],
                    "prompt": (
                        "frosted glass, diffused light, semi-opaque, "
                        "soft privacy effect, clean modern interior"
                    ),
                },
            ],
        },
        {
            "id": "fabric",
            "name": "Fabric & Soft Surfaces",
            "description": "Upholstery and soft finishes.",
            "items": [
                {
                    "id": "fabric_linen_neutral",
                    "label": "Neutral Linen",
                    "tags": ["sofa", "curtain", "soft"],
                    "prompt": (
                        "neutral linen fabric, fine weave, soft texture, "
                        "matte finish, warm and inviting"
                    ),
                },
                {
                    "id": "fabric_velvet_deep",
                    "label": "Deep Velvet",
                    "tags": ["accent", "luxury"],
                    "prompt": (
                        "deep velvet fabric, rich color, soft sheen, "
                        "luxurious tactile surface"
                    ),
                },
            ],
        },
        {
            "id": "floor",
            "name": "Flooring",
            "description": "Floor finishes for interior and exterior.",
            "items": [
                {
                    "id": "floor_oak_planks",
                    "label": "Oak Plank Flooring",
                    "tags": ["interior", "warm", "wood"],
                    "prompt": (
                        "wide oak wood planks, subtle bevel edges, warm tone, "
                        "matte finish, contemporary interior flooring"
                    ),
                },
                {
                    "id": "floor_polished_concrete",
                    "label": "Polished Concrete Floor",
                    "tags": ["industrial", "loft", "cool"],
                    "prompt": (
                        "polished concrete floor, smooth surface, soft reflections, "
                        "industrial loft aesthetic"
                    ),
                },
            ],
        },
    ]
}


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def list_materials() -> Dict[str, Any]:
    """
    Return the entire material library as-is.
    """
    return MATERIAL_LIBRARY


def find_material_by_id(material_id: str) -> Optional[Dict[str, Any]]:
    """
    Linear search through all categories and items to find a material
    with the given id. Returns the item dict or None.
    """
    for category in MATERIAL_LIBRARY.get("categories", []):
        for item in category.get("items", []):
            if item.get("id") == material_id:
                return item
    return None


def build_material_prompt(material_id: str) -> Optional[str]:
    """
    Return a detailed, architecture-focused prompt fragment for a material.
    If the material_id is unknown, return None.
    """
    material = find_material_by_id(material_id)
    if not material:
        return None

    base_prompt = material.get("prompt", "")
    label = material.get("label", "")
    tags = material.get("tags", [])

    extra_bits = []

    if label:
        extra_bits.append(f"material: {label}")
    if tags:
        extra_bits.append("tags: " + ", ".join(tags))

    extra_text = ". ".join(extra_bits) if extra_bits else ""

    if extra_text:
        return f"{base_prompt}. {extra_text}"
    return base_prompt


__all__ = [
    "MATERIAL_LIBRARY",
    "list_materials",
    "find_material_by_id",
    "build_material_prompt",
]
