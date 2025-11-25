"""
pipelines/space_capture_stub.py

CPU-only stub for Marble-style space capture / VR world generation.

- Takes a list of image paths (uploads / outputs)
- Does NOT run NeRF / Gaussian Splatting / 3D ML (to keep it CPU + license safe)
- Writes a JSON "bundle" describing a fake navigable space
- Returns paths that the frontend (or a future 3D viewer) can use

Later, on GPU, we can replace this with a real reconstruction pipeline while
keeping the API stable.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from config import OUTPUTS_DIR
from file_utils import generate_output_filename

logger = logging.getLogger(__name__)


def generate_space_reconstruction_stub(image_paths: List[str]) -> dict:
    """
    Create a stub "space bundle" JSON under /app/outputs.

    The bundle describes:
    - which input images were used
    - some fake navigation points (like 'poi_1', 'poi_2')
    - a placeholder mesh path (no real mesh is created yet)

    Returns a dict with:
    - bundle_path: JSON file path
    - mesh_path: placeholder mesh file path (e.g. .obj)
    - engine: 'stub'
    - notes: human-readable description
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON file name
    bundle_filename = generate_output_filename(prefix="space_bundle", ext="json")
    bundle_path = OUTPUTS_DIR / bundle_filename

    # Placeholder mesh path (no actual mesh file is created here)
    mesh_placeholder = OUTPUTS_DIR / (bundle_filename.replace(".json", ".obj"))

    # Basic fake navigation points (front / left / right / back)
    navigation_points = [
        {
            "id": "poi_front",
            "label": "Front view",
            "approx_position": [0.0, 0.0, 0.0],
            "related_image": image_paths[0] if image_paths else None,
        },
        {
            "id": "poi_left",
            "label": "Left view",
            "approx_position": [-1.0, 0.0, 0.5],
            "related_image": image_paths[1] if len(image_paths) > 1 else None,
        },
        {
            "id": "poi_right",
            "label": "Right view",
            "approx_position": [1.0, 0.0, 0.5],
            "related_image": image_paths[2] if len(image_paths) > 2 else None,
        },
    ]

    payload = {
        "type": "space_capture_stub",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "engine": "stub",
        "note": (
            "This is a CPU-only placeholder bundle. No real 3D reconstruction "
            "has been performed yet. On GPU we can plug in a NeRF / 3DGS pipeline "
            "while keeping this JSON contract stable."
        ),
        "inputs": {
            "images": image_paths,
        },
        "navigation": {
            "points": navigation_points,
        },
        "outputs": {
            "mesh_placeholder": str(mesh_placeholder),
        },
    }

    try:
        with bundle_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Space capture stub bundle written to %s", bundle_path)
    except Exception as e:
        logger.exception("Failed to write space capture stub bundle")
        raise

    return {
        "bundle_path": str(bundle_path),
        "mesh_path": str(mesh_placeholder),
        "engine": "stub",
        "notes": "Space capture stub; ready to be replaced by real 3D pipeline later.",
    }
