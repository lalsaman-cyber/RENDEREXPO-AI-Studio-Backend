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

        return cv2.Canny(gray_np, low, high)
    except Exception:
        pil = Image.fromarray(gray_np, mode="L").filter(ImageFilter.FIND_EDGES)
        return np.array(pil, dtype=np.uint8)


def _dilate_lines(img: Image.Image, iterations: int = 1) -> Image.Image:
    try:
        import cv2  # type: ignore

        arr = np.array(img, dtype=np.uint8)
        inv = 255 - arr
        kernel = np.ones((2, 2), np.uint8)
        dil = cv2.dilate(inv, kernel, iterations=iterations)
        out = 255 - dil
        return Image.fromarray(out, mode="L")
    except Exception:
        out = img
        for _ in range(max(1, iterations)):
            out = out.filter(ImageFilter.MaxFilter(size=3))
        return out


def _cleanup_image(img: Image.Image, cleanup: Dict[str, Any]) -> tuple[Image.Image, int]:
    out = img.convert("L") if _to_bool(cleanup.get("grayscale"), True) else img.convert("RGB")

    if _to_bool(cleanup.get("autocontrast"), True):
        out = ImageOps.autocontrast(out)

    contrast_boost = _to_float(cleanup.get("contrast_boost"), 1.45)
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

    threshold_value = _to_int(cleanup.get("threshold_value"), 165)
    out = out.point(lambda p: 255 if p > threshold_value else 0)

    return out, threshold_value


def build_mistoline_control_image(
    *,
    sketch_path: str,
    output_path: str,
    cleanup: Dict[str, Any] | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Build a stronger normalized sketch control image for MistoLine.

    Goals:
    - preserve architectural structure
    - produce darker / clearer line hierarchy
    - reduce airy / weak edge behavior
    - give SDXL + MistoLine a more decisive structural control image
    """
    if not os.path.isfile(sketch_path):
        raise FileNotFoundError(f"Sketch input not found: {sketch_path}")

    cleanup = cleanup or {}

    src = _load_image(sketch_path)
    cleaned, threshold_value = _cleanup_image(src, cleanup)

    gray = cleaned.convert("L")
    gray_np = np.array(gray, dtype=np.uint8)

    low = _to_int(cleanup.get("canny_low_threshold"), 70)
    high = _to_int(cleanup.get("canny_high_threshold"), 150)

    edges = _maybe_cv2_canny(gray_np, low, high)
    edges_img = Image.fromarray(edges, mode="L")

    # Convert to white background / dark lines.
    edges_img = ImageOps.invert(edges_img)
    edges_img = ImageOps.autocontrast(edges_img)

    if _to_bool(cleanup.get("thicken_lines"), True):
        iterations = _to_int(cleanup.get("thicken_iterations"), 1)
        edges_img = _dilate_lines(edges_img, iterations=iterations)

    final_contrast_boost = _to_float(cleanup.get("final_contrast_boost"), 1.35)
    if abs(final_contrast_boost - 1.0) > 1e-6:
        edges_img = ImageEnhance.Contrast(edges_img).enhance(final_contrast_boost)

    final_threshold_value = _to_int(cleanup.get("final_threshold_value"), 175)
    edges_img = edges_img.point(lambda p: 255 if p > final_threshold_value else 0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    edges_img.save(output_path)

    meta = {
        "cleanup_applied": {
            "grayscale": _to_bool(cleanup.get("grayscale"), True),
            "autocontrast": _to_bool(cleanup.get("autocontrast"), True),
            "contrast_boost": _to_float(cleanup.get("contrast_boost"), 1.45),
            "median_blur_ksize": _to_int(cleanup.get("median_blur_ksize"), 3),
            "invert_to_white_bg_black_lines": _to_bool(cleanup.get("invert_to_white_bg_black_lines"), True),
            "threshold_value": threshold_value,
            "canny_low_threshold": low,
            "canny_high_threshold": high,
            "thicken_lines": _to_bool(cleanup.get("thicken_lines"), True),
            "thicken_iterations": _to_int(cleanup.get("thicken_iterations"), 1),
            "final_contrast_boost": final_contrast_boost,
            "final_threshold_value": final_threshold_value,
        },
    }

    return output_path, meta