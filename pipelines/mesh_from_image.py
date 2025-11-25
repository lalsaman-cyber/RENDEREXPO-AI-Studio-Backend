"""
pipelines/mesh_from_image.py

CPU-friendly stub for "mesh from image" for RENDEREXPO AI STUDIO.

What it does (for now, on CPU only):
- Validates the input image exists.
- Writes a small placeholder 3D mesh file into OUTPUTS_DIR.
- For format "obj": we write a *real* minimal OBJ with a simple cube.
- For formats "glb" or "fbx": we write a placeholder text file with the
  correct extension (not a real GLB/FBX yet, but a valid path for download).

Later, on GPU:
- Replace internals with a real image-to-3D model.
- Keep the same function signature so the API contract stays stable.
"""

import logging
from pathlib import Path
from typing import Dict, Any

from config import OUTPUTS_DIR
from file_utils import validate_image_file, generate_output_filename

logger = logging.getLogger(__name__)


def _write_placeholder_obj_cube(path: Path) -> None:
    """
    Write a tiny, valid OBJ file defining a simple cube.

    This is enough for most DCC / 3D viewers to open and show "something".
    """
    obj_text = """# RENDEREXPO AI STUDIO placeholder mesh
# Simple unit cube mesh

o PlaceholderCube

v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
v 0.0 0.0 1.0
v 1.0 0.0 1.0
v 1.0 1.0 1.0
v 0.0 1.0 1.0

# Faces (1-based indices)
f 1 2 3 4
f 5 6 7 8
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
"""
    path.write_text(obj_text, encoding="utf-8")


def generate_mesh_from_image(input_image_path: str, fmt: str = "obj") -> Dict[str, Any]:
    """
    Public API used by main.py (/v1/mesh/from-image).

    Parameters
    ----------
    input_image_path : str
        Path to the input image (/app/uploads/... or /app/outputs/...).
    fmt : str
        Desired mesh format: "obj", "glb", "fbx".
        - obj → real minimal OBJ cube.
        - glb/fbx → placeholder text file with the correct extension.

    Returns
    -------
    dict
        {
          "engine": "stub_mesh_v1",
          "input_image": "...",
          "mesh_format": "obj",
          "output_mesh_path": "...",
          "notes": "...",
        }
    """
    image_path = Path(input_image_path)
    validate_image_file(image_path)

    fmt = (fmt or "obj").lower()
    if fmt not in {"obj", "glb", "fbx"}:
        logger.warning("Unsupported mesh format '%s', falling back to obj.", fmt)
        fmt = "obj"

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = generate_output_filename(prefix="mesh_from_image", ext=fmt)
    mesh_path = OUTPUTS_DIR / filename

    if fmt == "obj":
        _write_placeholder_obj_cube(mesh_path)
        notes = "Minimal OBJ cube placeholder generated."
    else:
        # For now, write a simple placeholder text file with the right extension.
        placeholder_text = (
            f"RENDEREXPO AI STUDIO placeholder mesh file ({fmt.upper()}).\n"
            f"This is not yet a real {fmt.upper()} binary; will be replaced by a "
            f"true image-to-3D pipeline in a future GPU phase.\n"
        )
        mesh_path.write_text(placeholder_text, encoding="utf-8")
        notes = f"Placeholder {fmt.upper()} file created (not a real 3D binary yet)."

    logger.info("Generated mesh file at: %s", mesh_path)

    return {
        "engine": "stub_mesh_v1",
        "input_image": str(image_path),
        "mesh_format": fmt,
        "output_mesh_path": str(mesh_path),
        "notes": notes,
    }


__all__ = ["generate_mesh_from_image"]
