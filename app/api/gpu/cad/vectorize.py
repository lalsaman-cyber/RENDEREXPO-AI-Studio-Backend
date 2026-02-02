# app/api/gpu/cad/vectorize.py
from __future__ import annotations

import os
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np


Polyline = List[Tuple[float, float]]
LayerVectors = Dict[str, List[Polyline]]


def extract_linework_vectors(
    image_path: str,
    preview_out_path: str,
    semantics_level: str = "architectural",
) -> Dict[str, Any]:
    """
    REAL linework extraction:
    - grayscale + denoise
    - edge detection (Canny)
    - morphological cleanup
    - contour extraction -> polylines
    - polyline simplification for snapping
    - basic semantic layering heuristics

    Returns dict:
      {
        "WALLS": [polyline, ...],
        "WINDOWS": [...],
        ...
        "_image_size": {"w": W, "h": H}
      }
    Coordinates are in IMAGE PIXELS.
    """

    if not os.path.isfile(image_path):
        raise RuntimeError(f"Image not found: {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cv2.imread failed (unsupported image?)")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise + sharpen slightly to help edges
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # Adaptive contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Edge detect
    v = np.median(gray)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, lower, upper, apertureSize=3, L2gradient=True)

    # Clean up edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Convert contours to simplified polylines
    polylines: List[Polyline] = []
    min_len = max(40, int(0.002 * (w + h)))  # ignore tiny junk
    eps = 1.5  # simplification tolerance in pixels

    for cnt in contours:
        if len(cnt) < min_len:
            continue
        pts = cnt.reshape(-1, 2).astype(np.float32)

        # Approx poly
        approx = cv2.approxPolyDP(pts, epsilon=eps, closed=False)
        if approx is None or len(approx) < 2:
            continue
        pl = [(float(x), float(y)) for [x, y] in approx.reshape(-1, 2)]
        if len(pl) >= 2:
            polylines.append(pl)

    # Layer assignment (v1 heuristics)
    layers: LayerVectors = _assign_layers(polylines, w=w, h=h, semantics_level=semantics_level)

    # Write preview
    preview = np.zeros((h, w, 3), dtype=np.uint8)
    for layer_name, pls in layers.items():
        if layer_name.startswith("_"):
            continue
        for pl in pls:
            for i in range(len(pl) - 1):
                p1 = (int(pl[i][0]), int(pl[i][1]))
                p2 = (int(pl[i + 1][0]), int(pl[i + 1][1]))
                cv2.line(preview, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(preview_out_path), exist_ok=True)
    cv2.imwrite(preview_out_path, preview)

    out: Dict[str, Any] = {**layers}
    out["_image_size"] = {"w": int(w), "h": int(h)}
    return out


def _assign_layers(polylines: List[Polyline], w: int, h: int, semantics_level: str) -> LayerVectors:
    """
    Practical v1 layering:
    - Long mostly-straight polylines -> WALLS/STRUCTURE
    - Medium-length rectilinear clusters -> WINDOWS/DOORS
    - Small/noisy -> CONTEXT/FURNITURE
    This is not "perfect semantics", but it is real and useful.
    """

    layers: LayerVectors = {
        "WALLS": [],
        "STRUCTURE": [],
        "WINDOWS": [],
        "DOORS": [],
        "FURNITURE": [],
        "GLASS": [],
        "CONTEXT": [],
    }

    def poly_length(pl: Polyline) -> float:
        s = 0.0
        for i in range(len(pl) - 1):
            dx = pl[i + 1][0] - pl[i][0]
            dy = pl[i + 1][1] - pl[i][1]
            s += (dx * dx + dy * dy) ** 0.5
        return s

    def straightness(pl: Polyline) -> float:
        # 1.0 means perfectly straight
        if len(pl) < 2:
            return 0.0
        x0, y0 = pl[0]
        x1, y1 = pl[-1]
        chord = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        arc = max(poly_length(pl), 1e-6)
        return float(chord / arc)

    # thresholds relative to image size
    diag = (w * w + h * h) ** 0.5
    long_thr = 0.25 * diag
    med_thr = 0.08 * diag

    for pl in polylines:
        L = poly_length(pl)
        S = straightness(pl)

        if L >= long_thr and S >= 0.92:
            # long straight edges: walls/structure
            layers["WALLS"].append(pl)
        elif L >= med_thr and S >= 0.88:
            # medium straight segments: structure/windows
            layers["STRUCTURE"].append(pl)
        elif L >= med_thr * 0.6:
            # mid sized detail: windows/doors candidates
            layers["WINDOWS"].append(pl)
        elif L >= med_thr * 0.35:
            layers["DOORS"].append(pl)
        else:
            layers["CONTEXT"].append(pl)

    return layers
