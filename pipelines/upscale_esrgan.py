from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    from logs.log_utils import get_logger  # type: ignore
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_logger(name: str):
        return logging.getLogger(name)

from file_utils import generate_output_filename, validate_image_file  # type: ignore
from PIL import Image

logger = get_logger(__name__)

# Try to import realesrgan + torch. If not available, we will fall back
# to a high-quality Pillow resize (LANCZOS) so the pipeline still works.
try:
    import torch  # type: ignore
    from realesrgan import RealESRGANer  # type: ignore
except ImportError as e:  # pragma: no cover - defensive
    logger.warning(
        "Real-ESRGAN or torch not available. ESRGAN upscaler will use a "
        "placeholder resize instead of real ESRGAN. Error: %s",
        e,
    )
    torch = None  # type: ignore
    RealESRGANer = None  # type: ignore


# Backend/pipelines/upscale_esrgan.py
# parents[0] -> Backend/pipelines
# parents[1] -> Backend
BACKEND_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = BACKEND_ROOT / "outputs"

# Top-level Models/ folder (sibling of Backend/)
MODELS_ROOT = BACKEND_ROOT.parent / "Models"

# Default model name and expected weight path for Real-ESRGAN.
# NOTE: This is just a convention; the actual .pth file must be placed
# under RENDEREXPO-AI-Studio/Models/ if you want REAL ESRGAN.
DEFAULT_ESRGAN_MODEL_NAME = "RealESRGAN_x2plus"
DEFAULT_ESRGAN_WEIGHTS = MODELS_ROOT / f"{DEFAULT_ESRGAN_MODEL_NAME}.pth"


def _ensure_outputs_dir() -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR


def _get_real_esrgan_upscaler(
    model_path: Path,
    scale: int = 2,
    tile: int = 0,
    tile_pad: int = 10,
) -> Optional["RealESRGANer"]:  # type: ignore
    """
    Try to create a RealESRGANer upscaler if realesrgan + torch are available
    and the model weights exist.

    If anything fails, returns None and logs an explanation.
    """
    if RealESRGANer is None or torch is None:
        logger.warning(
            "Real-ESRGAN library is not available. "
            "Falling back to placeholder resize."
        )
        return None

    if not model_path.is_file():
        logger.warning(
            "Real-ESRGAN weights not found at %s. "
            "Falling back to placeholder resize.",
            model_path,
        )
        return None

    try:
        logger.info(
            "Creating RealESRGANer upscaler: model=%s, scale=%d",
            model_path.name,
            scale,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        upscaler = RealESRGANer(
            scale=scale,
            model_path=str(model_path),
            model=DEFAULT_ESRGAN_MODEL_NAME,
            device=device,
            tile=tile,
            tile_pad=tile_pad,
        )

        logger.info("RealESRGANer created successfully on device=%s", device)
        return upscaler

    except Exception as e:  # pragma: no cover - defensive
        logger.error("Failed to create RealESRGANer. Error: %s", e)
        return None


def run_esrgan_upscale(
    input_image_path: Union[str, Path],
    scale: int = 2,
    model_name: str = DEFAULT_ESRGAN_MODEL_NAME,
    model_weights_path: Optional[Union[str, Path]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """
    High-level upscaling entry point for Real-ESRGAN.

    Behavior:
      - Validates that 'input_image_path' exists and is an image.
      - Attempts to use Real-ESRGAN if available and weights are present.
      - If Real-ESRGAN is unavailable, falls back to a high-quality
        Pillow resize (LANCZOS) so the call still succeeds.

    Returns a structured dict similar to other pipelines, with:
      - status
      - engine
      - input_image_path
      - output_image_path
      - scale
      - model_name
      - params / debug fields
    """
    input_image_path = Path(input_image_path)
    logger.info(
        "ESRGAN upscale request: input=%s, scale=%d, model_name=%s",
        input_image_path,
        scale,
        model_name,
    )

    # Validate the input image file first.
    validate_image_file(input_image_path)

    outputs_dir = _ensure_outputs_dir()
    filename = generate_output_filename(prefix="upscale_esrgan", ext="png")
    output_path = outputs_dir / filename

    # Decide which weights path to use.
    if model_weights_path is not None:
        model_weights = Path(model_weights_path)
    else:
        model_weights = DEFAULT_ESRGAN_WEIGHTS

    # Try to get a REAL ESRGAN upscaler.
    upscaler = _get_real_esrgan_upscaler(
        model_path=model_weights,
        scale=scale,
    )

    # Load the input image once.
    image = Image.open(input_image_path).convert("RGB")

    if upscaler is None:
        # Placeholder path: use Pillow's LANCZOS resize to mimic upscaling.
        logger.warning(
            "Real-ESRGAN not available; using Pillow LANCZOS resize as placeholder."
        )

        new_width = image.width * scale
        new_height = image.height * scale
        upscaled = image.resize((new_width, new_height), Image.LANCZOS)
        upscaled.save(output_path)

        logger.info(
            "Saved placeholder upscaled image (LANCZOS) to %s", output_path
        )

        return {
            "status": "ok",
            "type": "upscale",
            "engine": "Real-ESRGAN (placeholder LANCZOS)",
            "input_image_path": str(input_image_path),
            "output_image_path": str(output_path),
            "scale": scale,
            "model_name": model_name,
            "params": {
                "model_weights_path": str(model_weights),
                "extra": extra,
            },
            "debug": {
                "reason": "Real-ESRGAN not available; used Pillow LANCZOS resize.",
            },
        }

    # REAL ESRGAN path.
    try:
        upscaled, _ = upscaler.enhance(image)
        upscaled.save(output_path)
        logger.info("Saved REAL ESRGAN upscaled image to %s", output_path)

        return {
            "status": "ok",
            "type": "upscale",
            "engine": "Real-ESRGAN",
            "input_image_path": str(input_image_path),
            "output_image_path": str(output_path),
            "scale": scale,
            "model_name": model_name,
            "params": {
                "model_weights_path": str(model_weights),
                "extra": extra,
            },
            "debug": {
                "device": str(upscaler.device) if hasattr(upscaler, "device") else None,  # type: ignore
            },
        }

    except Exception as e:
        logger.error("Error while running REAL ESRGAN upscaling. Error: %s", e)

        # In case of failure, we still try to give the caller something.
        try:
            new_width = image.width * scale
            new_height = image.height * scale
            upscaled = image.resize((new_width, new_height), Image.LANCZOS)
            upscaled.save(output_path)
            fallback_saved = True
        except Exception as e2:  # pragma: no cover - defensive
            logger.error(
                "Failed to save fallback LANCZOS image after ESRGAN error. Error: %s",
                e2,
            )
            fallback_saved = False

        return {
            "status": "error",
            "type": "upscale",
            "engine": "Real-ESRGAN",
            "input_image_path": str(input_image_path),
            "output_image_path": str(output_path) if fallback_saved else None,
            "scale": scale,
            "model_name": model_name,
            "params": {
                "model_weights_path": str(model_weights),
                "extra": extra,
            },
            "debug": {
                "error": str(e),
                "fallback_saved": fallback_saved,
            },
        }


__all__ = [
    "run_esrgan_upscale",
]
