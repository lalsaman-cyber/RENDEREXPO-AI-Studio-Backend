# app/api/gpu/cad/export_dxf.py
from __future__ import annotations

from typing import Dict, List, Tuple, Any

import ezdxf


Polyline = List[Tuple[float, float]]


def write_dxf(
    dxf_path: str,
    vectors: Dict[str, Any],
    meters_per_pixel: float,
    layer_profile: str = "standard",
    no_dimensions: bool = True,
) -> None:
    """
    REAL DXF writer:
    - Writes polylines to named layers
    - Applies real-world scaling using meters_per_pixel
    - Keeps geometry snapping-friendly

    Units: meters (by scaling).
    """
    doc = ezdxf.new(dxfversion="R2018")
    msp = doc.modelspace()

    # Create layers
    layer_names = [k for k in vectors.keys() if not k.startswith("_")]
    for name in layer_names:
        if name not in doc.layers:
            doc.layers.new(name=name)

    # Convert pixels to meters; DXF coordinates will be in meters
    scale = float(meters_per_pixel) if meters_per_pixel and meters_per_pixel > 0 else 1.0

    for layer, polys in vectors.items():
        if layer.startswith("_"):
            continue
        for pl in polys:
            if len(pl) < 2:
                continue

            pts = [(p[0] * scale, -p[1] * scale) for p in pl]  # flip Y for CAD-like coords
            # Use LWPolyline for snapping friendliness
            msp.add_lwpolyline(pts, dxfattribs={"layer": layer})

    # No dimensions: we simply do not create DIM entities.
    # If later needed: add optional but default off.

    doc.saveas(dxf_path)
