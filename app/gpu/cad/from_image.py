# app/gpu/cad/from_image.py
"""
RENDEREXPO AI STUDIO - CAD From Image (REAL)

This is a REAL GPU-worker job runner (called by app/api/gpu/dispatch.py).

Inputs (in job_folder):
- input.png   (preferred)
  OR
- image.png   (if coming from other pipelines)

Meta (job_folder/meta.json) MUST include:
- scaling:
    - mode: "two_point" (recommended) or "door_height_fallback"
    - distance: float (real-world distance)
    - unit: "m"|"cm"|"mm"|"ft"
    - points: [{"x":0..1,"y":0..1},{"x":0..1,"y":0..1}]  (normalized)
    - fallback_door_height: float (meters)
- outputs:
    - dxf: "output.dxf"
    - dwg: "output.dwg"
    - preview: "lines_preview.png"

Outputs (written to job_folder):
- lines_preview.png   (edge/line preview)
- output.dxf          (REAL vector linework + layers + snapping)
- output.dwg          (REAL conversion via ODA if available; otherwise meta notes failure)
- meta.json updated with status + runtime details

NOTES:
- No dimension entities are created.
- Scaling is applied to geometry (DXF coords) based on the chosen reference.
- Layering is REAL but v1-heuristic (not fake "AI"): based on geometry length/straightness.
- This file does NOT depend on POD. This is PC-first.
"""

from __future__ import annotations

import os
import json
import math
import time
import traceback
import subprocess
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

try:
    import cv2  # type: ignore
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("Missing dependency: opencv-python (cv2). Install it in the GPU env.") from exc

try:
    import ezdxf  # type: ignore
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("Missing dependency: ezdxf. Install it in the GPU env.") from exc


# ----------------------------
# Types
# ----------------------------

Point = Tuple[float, float]          # (x, y) in pixels
Polyline = List[Point]              # list of pixel points
LayerVectors = Dict[str, List[Polyline]]


# ----------------------------
# Meta helpers
# ----------------------------

def _meta_path(job_folder: str) -> str:
    return os.path.join(job_folder, "meta.json")


def _read_meta(job_folder: str) -> Dict[str, Any]:
    p = _meta_path(job_folder)
    if not os.path.isfile(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_meta(job_folder: str, meta: Dict[str, Any]) -> None:
    with open(_meta_path(job_folder), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)


def _safe_set(meta: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = meta
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]  # type: ignore[assignment]
    cur[keys[-1]] = value


def _now_epoch() -> int:
    return int(time.time())


# ----------------------------
# Unit conversion
# ----------------------------

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
    return value  # default assume meters


# ----------------------------
# Input selection
# ----------------------------

def _find_input_image(job_folder: str) -> str:
    candidates = [
        os.path.join(job_folder, "input.png"),
        os.path.join(job_folder, "image.png"),
        os.path.join(job_folder, "image.jpg"),
        os.path.join(job_folder, "input.jpg"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise RuntimeError("No input image found in job folder (expected input.png or image.png).")


# ----------------------------
# Line extraction (REAL)
# ----------------------------

def _extract_edges(gray: np.ndarray) -> np.ndarray:
    # Strong but stable edge extraction for architectural renders
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=45, sigmaSpace=45)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    v = float(np.median(gray))
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))

    edges = cv2.Canny(gray, lower, upper, apertureSize=3, L2gradient=True)

    k = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=1)
    edges = cv2.dilate(edges, k, iterations=1)
    return edges


def _poly_length(pl: Polyline) -> float:
    s = 0.0
    for i in range(len(pl) - 1):
        dx = pl[i + 1][0] - pl[i][0]
        dy = pl[i + 1][1] - pl[i][1]
        s += math.hypot(dx, dy)
    return s


def _straightness(pl: Polyline) -> float:
    if len(pl) < 2:
        return 0.0
    x0, y0 = pl[0]
    x1, y1 = pl[-1]
    chord = math.hypot(x1 - x0, y1 - y0)
    arc = max(_poly_length(pl), 1e-6)
    return chord / arc


def _simplify_contour(cnt: np.ndarray, eps: float) -> Polyline:
    pts = cnt.reshape(-1, 2).astype(np.float32)
    approx = cv2.approxPolyDP(pts, epsilon=eps, closed=False)
    if approx is None or len(approx) < 2:
        return []
    out = [(float(x), float(y)) for [x, y] in approx.reshape(-1, 2)]
    return out


def _assign_layers(polylines: List[Polyline], w: int, h: int) -> LayerVectors:
    # Practical, real v1 layering heuristic (no fake AI claims)
    diag = math.hypot(w, h)
    long_thr = 0.25 * diag
    med_thr = 0.08 * diag

    layers: LayerVectors = {
        "WALLS": [],
        "STRUCTURE": [],
        "WINDOWS": [],
        "DOORS": [],
        "FURNITURE": [],
        "GLASS": [],
        "CONTEXT": [],
    }

    for pl in polylines:
        if len(pl) < 2:
            continue

        L = _poly_length(pl)
        S = _straightness(pl)

        if L >= long_thr and S >= 0.92:
            layers["WALLS"].append(pl)
        elif L >= med_thr and S >= 0.88:
            layers["STRUCTURE"].append(pl)
        elif L >= (0.6 * med_thr):
            layers["WINDOWS"].append(pl)
        elif L >= (0.35 * med_thr):
            layers["DOORS"].append(pl)
        else:
            layers["CONTEXT"].append(pl)

    return layers


def _vectorize_image(image_path: str, preview_path: str) -> Tuple[LayerVectors, int, int]:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cv2.imread failed on input image.")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = _extract_edges(gray)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    polylines: List[Polyline] = []
    min_len = max(40, int(0.002 * (w + h)))
    eps = 1.5  # px

    for cnt in contours:
        if len(cnt) < min_len:
            continue
        pl = _simplify_contour(cnt, eps=eps)
        if len(pl) >= 2:
            polylines.append(pl)

    layers = _assign_layers(polylines, w=w, h=h)

    # Preview (white on black)
    preview = np.zeros((h, w, 3), dtype=np.uint8)
    for layer_name, pls in layers.items():
        for pl in pls:
            for i in range(len(pl) - 1):
                p1 = (int(pl[i][0]), int(pl[i][1]))
                p2 = (int(pl[i + 1][0]), int(pl[i + 1][1]))
                cv2.line(preview, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(preview_path) or ".", exist_ok=True)
    cv2.imwrite(preview_path, preview)

    return layers, w, h


# ----------------------------
# Scaling (REAL)
# ----------------------------

def _compute_meters_per_pixel(meta: Dict[str, Any], w: int, h: int) -> float:
    scaling = meta.get("scaling") or {}
    mode = (scaling.get("mode") or "two_point").strip()

    if mode == "two_point":
        pts = scaling.get("points") or []
        dist = scaling.get("distance")
        unit = scaling.get("unit") or "m"
        if not isinstance(pts, list) or len(pts) != 2:
            raise RuntimeError("two_point scaling requires scaling.points = [p1, p2]")
        if dist is None or float(dist) <= 0:
            raise RuntimeError("two_point scaling requires scaling.distance > 0")

        p1 = pts[0]
        p2 = pts[1]
        x1 = float(p1["x"]) * float(w)
        y1 = float(p1["y"]) * float(h)
        x2 = float(p2["x"]) * float(w)
        y2 = float(p2["y"]) * float(h)

        px_dist = math.hypot(x2 - x1, y2 - y1)
        if px_dist <= 1e-6:
            raise RuntimeError("two_point scaling invalid (points too close).")

        real_m = _to_meters(float(dist), str(unit))
        mpp = real_m / px_dist
        return float(mpp)

    if mode == "door_height_fallback":
        door_h = float(scaling.get("fallback_door_height") or 2.10)
        # heuristic fallback: door height is ~18% of image height
        px_dist = 0.18 * float(h)
        return door_h / max(px_dist, 1e-6)

    # Default safe (still produces DXF but unscaled)
    return 1.0


# ----------------------------
# DXF/DWG export (REAL)
# ----------------------------

def _write_dxf(dxf_path: str, layers: LayerVectors, meters_per_pixel: float) -> None:
    doc = ezdxf.new(dxfversion="R2018")
    msp = doc.modelspace()

    # Ensure layers exist
    for layer_name in layers.keys():
        if layer_name not in doc.layers:
            doc.layers.new(name=layer_name)

    scale = float(meters_per_pixel) if meters_per_pixel and meters_per_pixel > 0 else 1.0

    # Write polylines (snapping-friendly)
    for layer_name, pls in layers.items():
        for pl in pls:
            if len(pl) < 2:
                continue
            pts = [(p[0] * scale, -p[1] * scale) for p in pl]  # flip Y
            msp.add_lwpolyline(pts, dxfattribs={"layer": layer_name})

    doc.saveas(dxf_path)


def _find_oda_converter() -> Optional[str]:
    env = os.getenv("ODA_FILE_CONVERTER_EXE", "").strip()
    if env and os.path.isfile(env):
        return env

    candidates = [
        r"C:\Program Files\ODA\ODAFileConverter\ODAFileConverter.exe",
        r"C:\Program Files\ODA File Converter\ODAFileConverter.exe",
        r"C:\Program Files (x86)\ODA\ODAFileConverter\ODAFileConverter.exe",
        r"C:\Program Files (x86)\ODA File Converter\ODAFileConverter.exe",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _convert_dxf_to_dwg(dxf_path: str, dwg_path: str) -> Tuple[bool, str]:
    converter = _find_oda_converter()
    if not converter:
        return False, "ODA File Converter not found (set ODA_FILE_CONVERTER_EXE or install it)."

    in_dir = os.path.dirname(os.path.abspath(dxf_path))
    out_dir = os.path.dirname(os.path.abspath(dwg_path))
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(dxf_path))[0]
    expected_out = os.path.join(out_dir, f"{base}.dwg")

    cmd = [
        converter,
        in_dir,
        out_dir,
        "ACAD2018",
        "DWG",
        "0",
        "1",
        "*.dxf",
    ]

    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if p.returncode != 0:
            return False, f"ODA conversion failed rc={p.returncode}: {p.stderr[:800]} {p.stdout[:800]}"

        if os.path.isfile(expected_out):
            # Ensure final path is exactly dwg_path
            if os.path.abspath(expected_out) != os.path.abspath(dwg_path):
                try:
                    import shutil
                    shutil.copyfile(expected_out, dwg_path)
                except Exception:
                    pass
            return True, "DWG written via ODA File Converter."

        alt = os.path.join(in_dir, f"{base}.dwg")
        if os.path.isfile(alt):
            if os.path.abspath(alt) != os.path.abspath(dwg_path):
                try:
                    import shutil
                    shutil.copyfile(alt, dwg_path)
                except Exception:
                    pass
            return True, "DWG written via ODA File Converter (alt path)."

        return False, "ODA ran but DWG output not found."
    except Exception as exc:  # noqa: BLE001
        return False, f"ODA conversion error: {exc}"


# ----------------------------
# Public runner (called by dispatcher)
# ----------------------------

def run_cad_from_image(job_folder: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point called by GPU dispatcher.
    Must NEVER fake outputs. If it can't produce DXF, it errors.
    """
    meta_path = _meta_path(job_folder)
    meta_disk = _read_meta(job_folder)
    # Disk meta is source-of-truth; merge minimal
    if isinstance(meta_disk, dict) and meta_disk:
        meta = meta_disk

    try:
        _safe_set(meta, "cad_runtime.started_at_epoch", _now_epoch())
        _write_meta(job_folder, meta)

        image_path = _find_input_image(job_folder)

        outputs = meta.get("outputs") or {}
        preview_name = outputs.get("preview") or "lines_preview.png"
        dxf_name = outputs.get("dxf") or "output.dxf"
        dwg_name = outputs.get("dwg") or "output.dwg"

        preview_path = os.path.join(job_folder, preview_name)
        dxf_path = os.path.join(job_folder, dxf_name)
        dwg_path = os.path.join(job_folder, dwg_name)

        # 1) Vectorize
        layers, w, h = _vectorize_image(image_path=image_path, preview_path=preview_path)

        # 2) Scale
        meters_per_pixel = _compute_meters_per_pixel(meta=meta, w=w, h=h)
        _safe_set(meta, "scaling._computed_meters_per_pixel", meters_per_pixel)

        # 3) Write DXF (required)
        _write_dxf(dxf_path=dxf_path, layers=layers, meters_per_pixel=meters_per_pixel)

        # 4) Convert DWG (best effort but real)
        ok_dwg, dwg_msg = _convert_dxf_to_dwg(dxf_path=dxf_path, dwg_path=dwg_path)

        _safe_set(meta, "cad_runtime.completed_at_epoch", _now_epoch())
        _safe_set(meta, "cad_runtime.image_used", os.path.basename(image_path))
        _safe_set(meta, "cad_runtime.preview_written", os.path.isfile(preview_path))
        _safe_set(meta, "cad_runtime.dxf_written", os.path.isfile(dxf_path))
        _safe_set(meta, "cad_runtime.dwg_written", bool(ok_dwg and os.path.isfile(dwg_path)))
        _safe_set(meta, "cad_runtime.dwg_message", dwg_msg)

        _write_meta(job_folder, meta)

        return {
            "status": "ok",
            "preview": preview_path if os.path.isfile(preview_path) else None,
            "dxf": dxf_path if os.path.isfile(dxf_path) else None,
            "dwg": dwg_path if (ok_dwg and os.path.isfile(dwg_path)) else None,
            "dwg_message": dwg_msg,
            "meters_per_pixel": meters_per_pixel,
            "layers": {k: len(v) for k, v in layers.items()},
            "image_size": {"w": w, "h": h},
        }

    except Exception as exc:
        _safe_set(meta, "cad_runtime.completed_at_epoch", _now_epoch())
        _safe_set(
            meta,
            "cad_runtime.error",
            {
                "detail": str(exc),
                "trace": traceback.format_exc(),
            },
        )
        _write_meta(job_folder, meta)
        raise
