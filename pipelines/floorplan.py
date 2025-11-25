"""
pipelines/floorplan.py

CPU-friendly stubs for SYNAPS-style floorplan workflows:

- generate_floorplan_from_text:
    Takes a natural-language description and produces a JSON "floorplan"
    with extremely simple room rectangles. This is a *placeholder* for a
    future ML-based system. 100% text + JSON, no model calls.

- render_floorplan_view:
    Takes a floorplan JSON path plus a camera/view label and produces a
    very simple 2D schematic PNG. This is enough for the frontend to
    wire UI and for us to keep everything license-safe and CPU-only
    until we move to real models.

Both functions write their outputs under OUTPUTS_DIR (from config.py).
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from PIL import Image, ImageDraw

from config import OUTPUTS_DIR
from file_utils import generate_output_filename


def _make_dummy_rooms(levels: int, approx_area: Optional[float]) -> List[Dict[str, Any]]:
    """
    Produce a very simple list of rectangular "rooms" for the stub floorplan.
    This is *not* geometric truth, just structured data for the frontend.

    The goal: give you a stable JSON structure to build UI around,
    and later we can plug a real ML or rules-based generator behind the same API.
    """
    base_rooms = [
        {"name": "Living Room", "width": 5.0, "depth": 4.0},
        {"name": "Kitchen", "width": 3.0, "depth": 3.0},
        {"name": "Bedroom 1", "width": 3.5, "depth": 3.0},
        {"name": "Bedroom 2", "width": 3.0, "depth": 3.0},
        {"name": "Bathroom", "width": 2.0, "depth": 2.0},
    ]

    rooms: List[Dict[str, Any]] = []
    level_index = 0
    x_cursor = 0.0
    y_cursor = 0.0

    for i in range(levels):
        for room in base_rooms:
            rooms.append(
                {
                    "level": level_index,
                    "name": room["name"],
                    "x": x_cursor,
                    "y": y_cursor,
                    "width": room["width"],
                    "depth": room["depth"],
                }
            )
            x_cursor += room["width"] + 0.5

        # next level stacked above, reset x and shift y
        level_index += 1
        x_cursor = 0.0
        y_cursor += 6.0

    return rooms


def generate_floorplan_from_text(
    description: str,
    unit_system: str = "metric",
    levels: int = 1,
    approx_area: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Stub: text → floorplan JSON.

    - No ML.
    - No geometry guarantee.
    - Pure JSON structure for the frontend to start working with.

    Returns a dict with:
        - floorplan_path: path to saved JSON
        - metadata: high-level info
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    filename = generate_output_filename(prefix="floorplan", ext="json")
    floorplan_path = OUTPUTS_DIR / filename

    rooms = _make_dummy_rooms(levels=levels, approx_area=approx_area)

    data = {
        "type": "floorplan_stub",
        "description": description,
        "unit_system": unit_system,
        "levels": levels,
        "approx_area": approx_area,
        "rooms": rooms,
        "notes": (
            "This is a CPU-only stub floorplan. "
            "Later we can swap this implementation for a true ML or "
            "rules-based generator without changing the API."
        ),
    }

    with floorplan_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return {
        "floorplan_path": str(floorplan_path),
        "unit_system": unit_system,
        "levels": levels,
        "approx_area": approx_area,
        "engine": "stub_text_to_floorplan_v1",
        "rooms_count": len(rooms),
    }


def render_floorplan_view(
    floorplan_file_path: str,
    camera_label: str = "view_a",
    width: int = 1024,
    height: int = 768,
) -> Dict[str, Any]:
    """
    Stub: floorplan JSON → simple PNG.

    Loads the floorplan JSON, draws rectangles for rooms, and saves a very
    basic schematic image. This gives the frontend something visual to show
    while we stay within CPU + licensing limits.
    """
    floorplan_path = Path(floorplan_file_path)
    if not floorplan_path.is_file():
        raise FileNotFoundError(f"Floorplan file does not exist: {floorplan_file_path}")

    with floorplan_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rooms = data.get("rooms", [])

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_name = generate_output_filename(prefix="floorplan_view", ext="png")
    out_path = OUTPUTS_DIR / out_name

    img = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    # Simple scaling from "meters" to pixels
    scale = 40.0

    for room in rooms:
        x = room.get("x", 0.0) * scale + 50
        y = room.get("y", 0.0) * scale + 50
        w = room.get("width", 3.0) * scale
        d = room.get("depth", 3.0) * scale

        rect = [x, y, x + w, y + d]
        draw.rectangle(rect, outline=(0, 0, 0), width=2)

        name = room.get("name", "Room")
        draw.text((x + 5, y + 5), name, fill=(0, 0, 0))

    # Camera label in the corner so the frontend can map UI → image
    draw.text((10, 10), f"Camera: {camera_label}", fill=(50, 50, 50))

    img.save(out_path)

    return {
        "image_path": str(out_path),
        "camera_label": camera_label,
        "engine": "stub_floorplan_view_renderer_v1",
        "source_floorplan": str(floorplan_path),
    }
