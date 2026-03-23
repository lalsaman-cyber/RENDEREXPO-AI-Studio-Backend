from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

from runtime.preprocess.sketch_cleanup import cleanup_sketch_image


@dataclass
class SketchCannyConfig:
    enabled: bool = True
    cleanup_enabled: bool = True
    low_threshold: int = 100
    high_threshold: int = 200
    aperture_size: int = 3
    l2gradient: bool = False
    blur_ksize: int = 3
    invert_output: bool = False
    line_boost: bool = False
    line_boost_kernel: int = 2


def _require_deps() -> None:
    if cv2 is None:
        raise RuntimeError("opencv-python-headless is required for sketch canny preprocessing.")
    if np is None:
        raise RuntimeError("numpy is required for sketch canny preprocessing.")
    if Image is None:
        raise RuntimeError("Pillow is required for sketch canny preprocessing.")


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


def _pil_to_gray_array(image: Any) -> Any:
    if not isinstance(image, Image.Image):
        raise RuntimeError("build_canny_image expects a PIL image.")
    return np.array(image.convert("L")).astype("uint8")


def _gray_array_to_rgb_pil(arr: Any) -> Any:
    arr = np.clip(arr, 0, 255).astype("uint8")
    rgb = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def _resize_if_needed(image: Any, target_size: Optional[Tuple[int, int]]) -> Any:
    if target_size is None:
        return image
    if not isinstance(image, Image.Image):
        raise RuntimeError("_resize_if_needed expects a PIL image.")
    tw, th = int(target_size[0]), int(target_size[1])
    if image.size == (tw, th):
        return image
    return image.resize((tw, th), Image.LANCZOS)


def _maybe_blur(gray: Any, blur_ksize: int) -> Any:
    if int(blur_ksize) <= 1:
        return gray
    return cv2.GaussianBlur(gray, (_odd_kernel(blur_ksize), _odd_kernel(blur_ksize)), 0)


def _maybe_boost_lines(edges: Any, enabled: bool, kernel_size: int) -> Any:
    if not enabled:
        return edges
    k = max(1, int(kernel_size))
    kernel = np.ones((k, k), dtype=np.uint8)
    return cv2.dilate(edges, kernel, iterations=1)


def build_canny_image(
    image: Any,
    config: Optional[Dict[str, Any]] = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> Any:
    """
    Build a clean Canny conditioning image for architectural sketch ControlNet.

    Returns:
        PIL.Image in RGB mode

    Flow:
    - optional cleanup
    - grayscale
    - optional blur
    - canny edge detection
    - optional line boost
    - output as 3-channel RGB
    """
    _require_deps()

    cfg = SketchCannyConfig()
    if isinstance(config, dict):
        for k, v in config.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    if not cfg.enabled:
        if not isinstance(image, Image.Image):
            raise RuntimeError("build_canny_image expects a PIL image when disabled.")
        image = image.convert("RGB")
        image = _resize_if_needed(image, target_size)
        return image

    if not isinstance(image, Image.Image):
        raise RuntimeError("build_canny_image expects a PIL image.")

    work = image.convert("RGB")
    work = _resize_if_needed(work, target_size)

    if cfg.cleanup_enabled:
        work = cleanup_sketch_image(work, config={"enabled": True})

    gray = _pil_to_gray_array(work)
    gray = _maybe_blur(gray, cfg.blur_ksize)

    aperture = _odd_kernel(cfg.aperture_size, fallback=3)
    if aperture not in (3, 5, 7):
        aperture = 3

    edges = cv2.Canny(
        gray,
        threshold1=int(cfg.low_threshold),
        threshold2=int(cfg.high_threshold),
        apertureSize=aperture,
        L2gradient=bool(cfg.l2gradient),
    )

    edges = _maybe_boost_lines(edges, cfg.line_boost, cfg.line_boost_kernel)

    if cfg.invert_output:
        edges = 255 - edges

    return _gray_array_to_rgb_pil(edges)