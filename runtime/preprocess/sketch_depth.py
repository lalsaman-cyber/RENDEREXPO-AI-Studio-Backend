from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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
class SketchDepthConfig:
    enabled: bool = True
    cleanup_enabled: bool = True
    normalize_output: bool = True
    invert_output: bool = False
    blur_radius: float = 0.0


def _require_deps() -> None:
    if np is None:
        raise RuntimeError("numpy is required for sketch depth preprocessing.")
    if Image is None:
        raise RuntimeError("Pillow is required for sketch depth preprocessing.")


def _resize_if_needed(image: Any, target_size: Optional[Tuple[int, int]]) -> Any:
    if target_size is None:
        return image
    if not isinstance(image, Image.Image):
        raise RuntimeError("_resize_if_needed expects a PIL image.")
    tw, th = int(target_size[0]), int(target_size[1])
    if image.size == (tw, th):
        return image
    return image.resize((tw, th), Image.LANCZOS)


def _pil_gray_to_rgb(image: Any) -> Any:
    if not isinstance(image, Image.Image):
        raise RuntimeError("_pil_gray_to_rgb expects a PIL image.")
    gray = image.convert("L")
    rgb = Image.merge("RGB", (gray, gray, gray))
    return rgb


def _normalize_depth_array(arr: Any) -> Any:
    arr = arr.astype("float32")
    dmin = float(arr.min())
    dmax = float(arr.max())

    if dmax - dmin < 1e-8:
        return np.zeros_like(arr, dtype="uint8")

    norm = (arr - dmin) / (dmax - dmin)
    return np.clip(norm * 255.0, 0, 255).astype("uint8")


def build_depth_image(
    image: Any,
    image_processor: Any,
    depth_model: Any,
    device: str = "cuda",
    torch_module: Any = None,
    config: Optional[Dict[str, Any]] = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> Any:
    """
    Build a depth conditioning image for architectural sketch ControlNet.

    Returns:
        PIL.Image in RGB mode

    Required inputs:
    - image: PIL image
    - image_processor: transformers AutoImageProcessor-compatible object
    - depth_model: transformers depth model (e.g. DPTForDepthEstimation)
    - torch_module: imported torch module from runtime

    Flow:
    - optional sketch cleanup
    - resize to target working size
    - run depth estimator
    - normalize depth to 0..255 grayscale
    - return 3-channel RGB PIL image
    """
    _require_deps()

    cfg = SketchDepthConfig()
    if isinstance(config, dict):
        for k, v in config.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    if not cfg.enabled:
        if not isinstance(image, Image.Image):
            raise RuntimeError("build_depth_image expects a PIL image when disabled.")
        image = image.convert("RGB")
        image = _resize_if_needed(image, target_size)
        return image

    if torch_module is None:
        raise RuntimeError("build_depth_image requires torch_module.")
    if image_processor is None:
        raise RuntimeError("build_depth_image requires a valid image_processor.")
    if depth_model is None:
        raise RuntimeError("build_depth_image requires a valid depth_model.")
    if not isinstance(image, Image.Image):
        raise RuntimeError("build_depth_image expects a PIL image.")

    work = image.convert("RGB")
    work = _resize_if_needed(work, target_size)

    if cfg.cleanup_enabled:
        work = cleanup_sketch_image(work, config={"enabled": True})

    inputs = image_processor(images=work, return_tensors="pt")

    try:
        inputs = {
            k: v.to(device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }
    except Exception:
        pass

    depth_model.eval()

    with torch_module.no_grad():
        outputs = depth_model(**inputs)
        predicted_depth = outputs.predicted_depth

    if predicted_depth.ndim == 3:
        predicted_depth = predicted_depth.unsqueeze(1)

    target_h = int(work.size[1])
    target_w = int(work.size[0])

    resized = torch_module.nn.functional.interpolate(
        predicted_depth,
        size=(target_h, target_w),
        mode="bicubic",
        align_corners=False,
    )

    depth = resized.squeeze().detach().float().cpu().numpy()

    if cfg.normalize_output:
        depth_u8 = _normalize_depth_array(depth)
    else:
        depth_u8 = np.clip(depth, 0, 255).astype("uint8")

    if cfg.invert_output:
        depth_u8 = 255 - depth_u8

    depth_img = Image.fromarray(depth_u8, mode="L")

    if cfg.blur_radius and float(cfg.blur_radius) > 0:
        try:
            from PIL import ImageFilter  # type: ignore
            depth_img = depth_img.filter(ImageFilter.GaussianBlur(radius=float(cfg.blur_radius)))
        except Exception:
            pass

    return _pil_gray_to_rgb(depth_img)