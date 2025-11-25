"""
pipelines/cad_from_image.py

CPU-friendly stub for "CAD from image" for RENDEREXPO AI STUDIO.

What it does (for now, on CPU only):
- Validates the input image exists.
- Writes a small placeholder CAD file into OUTPUTS_DIR.
- For format "dxf": we write a *real* minimal DXF with a simple rectangle / room outline.
- For formats "dwg" or "ifc": we write a placeholder text file with the
  correct extension (not a real DWG/IFC yet, but a valid path for download).

Later, in GPU / advanced phases:
- Replace internals with a real depth + edge + vectorization pipeline.
- Keep the same function signature so the API contract stays stable.
"""

import logging
from pathlib import Path
from typing import Dict, Any

from config import OUTPUTS_DIR
from file_utils import validate_image_file, generate_output_filename

logger = logging.getLogger(__name__)


def _write_placeholder_dxf_room(path: Path) -> None:
    """
    Write a tiny, valid DXF file defining a simple rectangular room outline
    using LINE entities.

    This should open in most CAD viewers as a simple box in model space.
    """
    # Minimal DXF: ENTITIES section with four LINEs
    dxf_text = """0
SECTION
2
ENTITIES
0
LINE
8
0
10
0.0
20
0.0
30
0.0
11
10.0
21
0.0
31
0.0
0
LINE
8
0
10
10.0
20
0.0
30
0.0
11
10.0
21
6.0
31
0.0
0
LINE
8
0
10
10.0
20
6.0
30
0.0
11
0.0
21
6.0
31
0.0
0
LINE
8
0
10
0.0
20
6.0
30
0.0
11
0.0
21
0.0
31
0.0
0
ENDSEC
0
EOF
"""
    path.write_text(dxf_text, encoding="utf-8")


def generate_cad_from_image(input_image_path: str, fmt: str = "dxf") -> Dict[str, Any]:
    """
    Public API used by main.py (/v1/cad/from-image).

    Parameters
    ----------
    input_image_path : str
        Path to the input image (/app/uploads/... or /app/outputs/...).
    fmt : str
        Desired CAD format: "dxf", "dwg", "ifc".
        - dxf → real minimal DXF with a simple room outline.
        - dwg/ifc → placeholder text file with the correct extension.

    Returns
    -------
    dict
        {
          "engine": "stub_cad_v1",
          "input_image": "...",
          "cad_format": "dxf",
          "output_cad_path": "...",
          "notes": "...",
        }
    """
    image_path = Path(input_image_path)
    validate_image_file(image_path)

    fmt = (fmt or "dxf").lower()
    if fmt not in {"dxf", "dwg", "ifc"}:
        logger.warning("Unsupported CAD format '%s', falling back to dxf.", fmt)
        fmt = "dxf"

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = generate_output_filename(prefix="cad_from_image", ext=fmt)
    cad_path = OUTPUTS_DIR / filename

    if fmt == "dxf":
        _write_placeholder_dxf_room(cad_path)
        notes = "Minimal DXF room-outline placeholder generated."
    else:
        # For now, write a simple placeholder text file with the right extension.
        placeholder_text = (
            f"RENDEREXPO AI STUDIO placeholder CAD file ({fmt.upper()}).\n"
            f"This is not yet a real {fmt.upper()} model; it will be replaced by a "
            f"true CAD pipeline in a future phase.\n"
        )
        cad_path.write_text(placeholder_text, encoding="utf-8")
        notes = f"Placeholder {fmt.upper()} file created (not a real CAD model yet)."

    logger.info("Generated CAD file at: %s", cad_path)

    return {
        "engine": "stub_cad_v1",
        "input_image": str(image_path),
        "cad_format": fmt,
        "output_cad_path": str(cad_path),
        "notes": notes,
    }


__all__ = ["generate_cad_from_image"]
