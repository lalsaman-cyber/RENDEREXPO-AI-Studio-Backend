from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

try:
    from logs.log_utils import get_logger  # type: ignore
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_logger(name: str):
        return logging.getLogger(name)

from file_utils import generate_output_filename, validate_image_file  # type: ignore

from PIL import Image
import numpy as np

logger = get_logger(__name__)

# Try to import torch. If not available, we will fall back to a dummy behavior.
try:
    import torch  # type: ignore
except ImportError as e:  # pragma: no cover - defensive
    logger.warning(
        "PyTorch not available. MiDaS depth pipeline will use a dummy fallback. Error: %s",
        e,
    )
    torch = None  # type: ignore


# Backend/pipelines/midas_depth.py
# parents[0] -> Backend/pipelines
# parents[1] -> Backend
BACKEND_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = BACKEND_ROOT / "outputs"

# We will lazily load MiDaS via torch.hub from intel-isl/MiDaS.
# This is CPU-only for now and purely for local use.
_MIDAS_MODEL: Optional["torch.nn.Module"] = None  # type: ignore
_MIDAS_TRANSFORMS = None


def _ensure_outputs_dir() -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR


def _load_midas_model(
    model_type: str = "DPT_Large",
) -> Optional[Tuple["torch.nn.Module", Any]]:  # type: ignore
    """
    Lazily load a MiDaS model and its transforms via torch.hub.

    If torch or the hub load fails, returns None and logs a warning.
    """
    global _MIDAS_MODEL, _MIDAS_TRANSFORMS

    if torch is None:
        logger.warning("PyTorch is not available; cannot load MiDaS.")
        return None

    if _MIDAS_MODEL is not None and _MIDAS_TRANSFORMS is not None:
        return _MIDAS_MODEL, _MIDAS_TRANSFORMS

    try:
        logger.info("Loading MiDaS model (%s) via torch.hub...", model_type)
        # Load model
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas.eval()

        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type in ("DPT_Large", "DPT_Hybrid"):
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform

        device = "cuda" if torch.cuda.is_available() else "cpu"
        midas.to(device)

        _MIDAS_MODEL = midas
        _MIDAS_TRANSFORMS = (transform, device)

        logger.info("MiDaS model loaded and moved to device=%s", device)
        return _MIDAS_MODEL, _MIDAS_TRANSFORMS

    except Exception as e:  # pragma: no cover - defensive
        logger.error("Failed to load MiDaS model via torch.hub. Error: %s", e)
        _MIDAS_MODEL = None
        _MIDAS_TRANSFORMS = None
        return None


def _compute_midas_depth(
    image: Image.Image,
    model_type: str = "DPT_Large",
) -> Optional[np.ndarray]:
    """
    Run MiDaS on a PIL image and return a depth map as a numpy array
    normalized to [0, 1].

    If anything fails, returns None.
    """
    if torch is None:
        logger.warning("PyTorch is not available; cannot compute MiDaS depth.")
        return None

    loaded = _load_midas_model(model_type=model_type)
    if loaded is None:
        return None

    midas, (transform, device) = loaded

    try:
        # Apply MiDaS transform
        input_batch = transform(image).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)

            # Upsample to original resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.size[::-1],  # (height, width)
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()

        # Normalize to [0,1]
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > 1e-6:
            depth_norm = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(depth)

        return depth_norm

    except Exception as e:
        logger.error("Error during MiDaS depth computation. Error: %s", e)
        return None


def run_midas_depth_map(
    input_image_path: Union[str, Path],
    model_type: str = "DPT_Large",
    **extra: Any,
) -> Dict[str, Any]:
    """
    High-level entry point to generate a depth map using MiDaS.

    Behavior:
      - Validates that 'input_image_path' exists and is an image.
      - Attempts to compute a REAL MiDaS depth map (DPT_Large by default).
      - Saves a grayscale depth PNG to Backend/outputs/depth_midas_*.png.
      - If MiDaS is unavailable or fails, falls back to a dummy depth image
        (e.g. a grayscale version of the input), and marks this in debug info.

    Returns:
      A dict with:
        - status
        - engine
        - input_image_path
        - output_depth_path
        - model_type
        - params / debug fields
    """
    input_image_path = Path(input_image_path)
    logger.info(
        "MiDaS depth request: input=%s, model_type=%s",
        input_image_path,
        model_type,
    )

    # Validate the input image file first.
    validate_image_file(input_image_path)

    outputs_dir = _ensure_outputs_dir()
    filename = generate_output_filename(prefix="depth_midas", ext="png")
    output_path = outputs_dir / filename

    # Load the input image
    image = Image.open(input_image_path).convert("RGB")

    # Try REAL MiDaS depth
    depth_norm = _compute_midas_depth(image=image, model_type=model_type)

    if depth_norm is None:
        # Dummy fallback: grayscale version of the input image.
        logger.warning(
            "MiDaS depth not available; using dummy grayscale fallback."
        )

        gray = image.convert("L")
        gray.save(output_path)

        return {
            "status": "ok",
            "type": "depth",
            "engine": "MiDaS (dummy grayscale fallback)",
            "input_image_path": str(input_image_path),
            "output_depth_path": str(output_path),
            "model_type": model_type,
            "params": {
                "extra": extra,
            },
            "debug": {
                "reason": "MiDaS unavailable or failed; used grayscale fallback.",
            },
        }

    # REAL depth: map [0,1] to 8-bit grayscale
    try:
        depth_8bit = (depth_norm * 255.0).clip(0, 255).astype(np.uint8)
        depth_img = Image.fromarray(depth_8bit, mode="L")
        depth_img.save(output_path)

        logger.info("Saved REAL MiDaS depth map to %s", output_path)

        return {
            "status": "ok",
            "type": "depth",
            "engine": "MiDaS",
            "input_image_path": str(input_image_path),
            "output_depth_path": str(output_path),
            "model_type": model_type,
            "params": {
                "extra": extra,
            },
            "debug": {
                "model_type": model_type,
            },
        }

    except Exception as e:
        logger.error(
            "Error while saving MiDaS depth map; attempting grayscale fallback. Error: %s",
            e,
        )
        try:
            gray = image.convert("L")
            gray.save(output_path)
            fallback_saved = True
        except Exception as e2:  # pragma: no cover - defensive
            logger.error(
                "Failed to save grayscale fallback depth image. Error: %s",
                e2,
            )
            fallback_saved = False

        return {
            "status": "error",
            "type": "depth",
            "engine": "MiDaS",
            "input_image_path": str(input_image_path),
            "output_depth_path": str(output_path) if fallback_saved else None,
            "model_type": model_type,
            "params": {
                "extra": extra,
            },
            "debug": {
                "error": str(e),
                "fallback_saved": fallback_saved,
            },
        }


__all__ = [
    "run_midas_depth_map",
]
