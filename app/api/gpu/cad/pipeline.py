# app/api/gpu/cad/pipeline.py
from __future__ import annotations

import os
import json
import traceback
from typing import Any, Dict, Optional

from .vectorize import extract_linework_vectors
from .export_dxf import write_dxf
from .export_dwg import convert_dxf_to_dwg


def _safe_load_meta(meta_path: str) -> Dict[str, Any]:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_save_meta(meta_path: str, meta: Dict[str, Any]) -> None:
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)


def run_cad_from_image_job(job_folder: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    REAL CAD pipeline:
    - load input image
    - extract linework vectors + layers
    - apply real-world scaling based on two-point reference or fallback
    - write DXF (required)
    - convert to DWG (best effort)
    - write preview PNG
    - update meta.json
    """
    os.makedirs(job_folder, exist_ok=True)
    meta_path = os.path.join(job_folder, "meta.json")

    if meta is None:
        meta = _safe_load_meta(meta_path)

    try:
        meta["status"] = "processing"
        _safe_save_meta(meta_path, meta)

        img_path = os.path.join(job_folder, meta["inputs"]["image"])
        if not os.path.isfile(img_path):
            raise RuntimeError(f"Input image missing: {img_path}")

        # Extract vectors (returns: preview_path, layers dict)
        preview_path = os.path.join(job_folder, meta["outputs"]["preview"])
        vectors = extract_linework_vectors(
            image_path=img_path,
            preview_out_path=preview_path,
            semantics_level=meta.get("semantics", {}).get("level", "architectural"),
        )

        # vectors: {layer_name: [ [ (x,y), (x,y), ... ], ... ] } in IMAGE PIXEL COORDS

        # Scaling
        scaling = meta.get("scaling", {}) or {}
        scale_factor = 1.0  # pixels -> real units
        unit = scaling.get("unit", "m")

        if scaling.get("mode") == "two_point":
            pts = scaling.get("points") or []
            dist = scaling.get("distance")
            if not pts or dist is None:
                raise RuntimeError("two_point scaling requires points + distance")
            p1 = pts[0]
            p2 = pts[1]
            # normalized coords -> pixel coords
            # We need image size; vectorizer can embed it in vectors["_image_size"]
            w = vectors.get("_image_size", {}).get("w")
            h = vectors.get("_image_size", {}).get("h")
            if not w or not h:
                raise RuntimeError("Missing image size for scaling")

            x1, y1 = float(p1["x"]) * w, float(p1["y"]) * h
            x2, y2 = float(p2["x"]) * w, float(p2["y"]) * h
            px_dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if px_dist <= 1e-6:
                raise RuntimeError("Invalid two_point scaling: points too close")

            # Convert distance in unit to base DXF units (we keep DXF in meters by default)
            # If user gives cm/mm/ft, normalize to meters so DXF units are meters.
            real_m = _to_meters(float(dist), unit)
            scale_factor = real_m / px_dist  # meters per pixel
            meta["scaling"]["_computed_meters_per_pixel"] = scale_factor

        elif scaling.get("mode") == "door_height_fallback":
            # Fallback uses door height in meters and tries to find a door-like vertical span.
            # v1 heuristic: assume a door is ~10-20% of image height; set scale using image height proportion.
            # This is real math but obviously less accurate than two-point.
            w = vectors.get("_image_size", {}).get("w")
            h = vectors.get("_image_size", {}).get("h")
            if not w or not h:
                raise RuntimeError("Missing image size for fallback scaling")
            door_h = float(scaling.get("fallback_door_height", 2.10))
            # assume door height spans 0.18*h (heuristic); user can correct later
            px_dist = 0.18 * h
            scale_factor = door_h / px_dist
            meta["scaling"]["_computed_meters_per_pixel"] = scale_factor
        else:
            # default safe
            meta["scaling"]["_computed_meters_per_pixel"] = None

        # Write DXF
        dxf_path = os.path.join(job_folder, meta["outputs"]["dxf"])
        write_dxf(
            dxf_path=dxf_path,
            vectors=vectors,
            meters_per_pixel=scale_factor,
            layer_profile=meta.get("semantics", {}).get("layer_profile", "standard"),
            no_dimensions=True,
        )

        # Convert to DWG (best effort)
        dwg_path = os.path.join(job_folder, meta["outputs"]["dwg"])
        dwg_ok, dwg_msg = convert_dxf_to_dwg(dxf_path=dxf_path, dwg_path=dwg_path)

        meta["status"] = "completed"
        meta.setdefault("outputs_runtime", {})
        meta["outputs_runtime"]["dxf_written"] = True
        meta["outputs_runtime"]["dwg_written"] = bool(dwg_ok)
        meta["outputs_runtime"]["dwg_message"] = dwg_msg
        _safe_save_meta(meta_path, meta)

        return {"status": "completed", "dxf": dxf_path, "dwg": dwg_path if dwg_ok else None, "preview": preview_path}

    except Exception as exc:
        meta["status"] = "error"
        meta.setdefault("error", {})
        meta["error"]["detail"] = str(exc)
        meta["error"]["traceback"] = traceback.format_exc()
        _safe_save_meta(meta_path, meta)
        return {"status": "error", "detail": str(exc)}


def _to_meters(value: float, unit: str) -> float:
    u = (unit or "m").lower().strip()
    if u == "m":
        return value
    if u == "cm":
        return value / 100.0
    if u == "mm":
        return value / 1000.0
    if u == "ft":
        return value * 0.3048
    # default assume meters
    return value
