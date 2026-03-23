from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    from PIL import Image, ImageOps, ImageFilter, ImageEnhance  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    ImageOps = None  # type: ignore
    ImageFilter = None  # type: ignore
    ImageEnhance = None  # type: ignore


@dataclass
class SketchCleanupConfig:
    enabled: bool = True
    grayscale: bool = True
    autocontrast: bool = True
    contrast_boost: float = 1.2
    median_blur_ksize: int = 3
    adaptive_threshold: bool = False
    threshold_value: int = 185
    morphology_close: bool = False
    close_kernel: int = 2
    thicken_lines: bool = False
    thicken_kernel: int = 2
    invert_to_white_bg_black_lines: bool = True


def _require_deps() -> None:
    if cv2 is None:
        raise RuntimeError("opencv-python-headless is required for sketch cleanup.")
    if np is None:
        raise RuntimeError("numpy is required for sketch cleanup.")
    if Image is None:
        raise RuntimeError("Pillow is required for sketch cleanup.")


def _clamp_uint8(arr: Any) -> Any:
    return np.clip(arr, 0, 255).astype("uint8")


def _pil_to_gray_array(image: Any) -> Any:
    if not isinstance(image, Image.Image):
        raise RuntimeError("sketch cleanup expects a PIL image.")
    gray = image.convert("L")
    return np.array(gray).astype("uint8")


def _gray_array_to_rgb_pil(arr: Any) -> Any:
    arr = _clamp_uint8(arr)
    rgb = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def _maybe_autocontrast(img: Any, enabled: bool) -> Any:
    if not enabled:
        return img
    return ImageOps.autocontrast(img)


def _maybe_contrast_boost(img: Any, factor: float) -> Any:
    if factor is None or abs(float(factor) - 1.0) < 1e-6:
        return img
    return ImageEnhance.Contrast(img).enhance(float(factor))


def _odd_kernel(v: int, fallback: int = 3) -> int:
    try:
        v = int(v)
    except Exception:
        v = fallback
    if v < 1:
        v = fallback
    if v % 2 == 0:
        v += 1
    return v


def _threshold_binary(gray: Any, cfg: SketchCleanupConfig) -> Any:
    if cfg.adaptive_threshold:
        block_size = 11
        c_val = 2
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c_val,
        )
    _, out = cv2.threshold(gray, int(cfg.threshold_value), 255, cv2.THRESH_BINARY)
    return out


def _maybe_close(binary: Any, cfg: SketchCleanupConfig) -> Any:
    if not cfg.morphology_close:
        return binary
    k = max(1, int(cfg.close_kernel))
    kernel = np.ones((k, k), dtype=np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def _maybe_thicken(binary: Any, cfg: SketchCleanupConfig) -> Any:
    if not cfg.thicken_lines:
        return binary
    k = max(1, int(cfg.thicken_kernel))
    kernel = np.ones((k, k), dtype=np.uint8)
    return cv2.dilate(binary, kernel, iterations=1)


def _maybe_invert_to_white_bg(binary: Any, cfg: SketchCleanupConfig) -> Any:
    """
    Target format for architectural line prep:
    - white background
    - dark lines
    """
    if not cfg.invert_to_white_bg_black_lines:
        return binary

    # If the image is mostly dark, invert it.
    white_ratio = float((binary > 127).sum()) / float(binary.size)
    if white_ratio < 0.5:
        return 255 - binary
    return binary


def cleanup_sketch_image(
    image: Any,
    config: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Clean an uploaded architectural sketch before Canny / Depth preprocessing.

    Returns:
        PIL.Image in RGB mode

    Goals:
    - normalize contrast
    - reduce scan noise
    - optionally threshold to strengthen architectural lines
    - optionally thicken weak lines
    - normalize toward white background + dark lines
    """
    _require_deps()

    cfg = SketchCleanupConfig()
    if isinstance(config, dict):
        for k, v in config.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    if not cfg.enabled:
        if not isinstance(image, Image.Image):
            raise RuntimeError("cleanup_sketch_image expects a PIL image when cleanup is disabled.")
        return image.convert("RGB")

    if not isinstance(image, Image.Image):
        raise RuntimeError("cleanup_sketch_image expects a PIL image.")

    img = image.convert("RGB")

    if cfg.grayscale:
        img = img.convert("L")
    else:
        img = ImageOps.grayscale(img)

    img = _maybe_autocontrast(img, cfg.autocontrast)
    img = _maybe_contrast_boost(img, cfg.contrast_boost)

    gray = np.array(img).astype("uint8")

    if cfg.median_blur_ksize and int(cfg.median_blur_ksize) > 1:
        gray = cv2.medianBlur(gray, _odd_kernel(int(cfg.median_blur_ksize), fallback=3))

    # Threshold only if explicitly requested or if the sketch looks faint.
    use_threshold = bool(cfg.adaptive_threshold)
    if not use_threshold:
        std_val = float(gray.std())
        if std_val < 40.0:
            use_threshold = True

    if use_threshold:
        binary = _threshold_binary(gray, cfg)
        binary = _maybe_close(binary, cfg)
        binary = _maybe_thicken(binary, cfg)
        binary = _maybe_invert_to_white_bg(binary, cfg)
        return _gray_array_to_rgb_pil(binary)

    gray = _maybe_invert_to_white_bg(gray, cfg)
    return _gray_array_to_rgb_pil(gray)