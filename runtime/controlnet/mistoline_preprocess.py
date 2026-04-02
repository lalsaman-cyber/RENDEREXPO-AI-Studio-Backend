# runtime/controlnet/mistoline_preprocess.py
from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _load_image(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    return img


def _maybe_cv2_canny(gray_np: np.ndarray, low: int, high: int) -> np.ndarray:
    try:
        import cv2  # type: ignore

        edges = cv2.Canny(gray_np, low, high)
        return edges
    except Exception:
        pil = Image.fromarray(gray_np, mode="L").filter(ImageFilter.FIND_EDGES)
        return np.array(pil, dtype=np.uint8)


def _cleanup_image(img: Image.Image, cleanup: Dict[str, Any]) -> Image.Image:
    out = img.convert("L") if _to_bool(cleanup.get("grayscale"), True) else img.convert("RGB")

    if _to_bool(cleanup.get("autocontrast"), True):
        out = ImageOps.autocontrast(out)

    contrast_boost = _to_float(cleanup.get("contrast_boost"), 1.25)
    if abs(contrast_boost - 1.0) > 1e-6:
        out = ImageEnhance.Contrast(out).enhance(contrast_boost)

    median_ksize = _to_int(cleanup.get("median_blur_ksize"), 3)
    if median_ksize >= 3 and median_ksize % 2 == 1:
        out = out.filter(ImageFilter.MedianFilter(size=median_ksize))

    if _to_bool(cleanup.get("invert_to_white_bg_black_lines"), True):
        arr = np.array(out, dtype=np.uint8)
        mean_val = float(arr.mean()) if arr.size else 255.0
        if mean_val < 127.0:
            out = ImageOps.invert(out)

    threshold_value = cleanup.get("threshold_value")
    if threshold_value is not None:
        t = _to_int(threshold_value, 180)
        out = out.point(lambda p: 255 if p > t else 0)

    return out


def build_mistoline_control_image(
    *,
    sketch_path: str,
    output_path: str,
    cleanup: Dict[str, Any] | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Build the normalized sketch control image fed into MistoLine.

    Goal:
    - stable line extraction
    - white background
    - dark / crisp line structure
    - simple, predictable preprocessing for SDXL ControlNet sketch runs
    """
    if not os.path.isfile(sketch_path):
        raise FileNotFoundError(f"Sketch input not found: {sketch_path}")

    cleanup = cleanup or {}

    src = _load_image(sketch_path)
    cleaned = _cleanup_image(src, cleanup)

    gray = cleaned.convert("L")
    gray_np = np.array(gray, dtype=np.uint8)

    low = _to_int(cleanup.get("canny_low_threshold"), 100)
    high = _to_int(cleanup.get("canny_high_threshold"), 200)

    edges = _maybe_cv2_canny(gray_np, low, high)
    edges_img = Image.fromarray(edges, mode="L")

    # Convert to white background with dark lines.
    edges_img = ImageOps.invert(edges_img)
    edges_img = ImageOps.autocontrast(edges_img)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    edges_img.save(output_path)

    meta = {
        "cleanup_applied": {
            "grayscale": _to_bool(cleanup.get("grayscale"), True),
            "autocontrast": _to_bool(cleanup.get("autocontrast"), True),
            "contrast_boost": _to_float(cleanup.get("contrast_boost"), 1.25),
            "median_blur_ksize": _to_int(cleanup.get("median_blur_ksize"), 3),
            "invert_to_white_bg_black_lines": _to_bool(cleanup.get("invert_to_white_bg_black_lines"), True),
            "threshold_value": cleanup.get("threshold_value"),
            "canny_low_threshold": low,
            "canny_high_threshold": high,
        },
    }

    return output_path, meta