from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    from logs.log_utils import get_logger  # type: ignore
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_logger(name: str):
        return logging.getLogger(name)

from config import app_config  # type: ignore
from file_utils import generate_output_filename, validate_image_file  # type: ignore

from PIL import Image

logger = get_logger(__name__)

# We rely on diffusers + torch, which are already used for SDXL.
try:
    import torch
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
except ImportError as e:  # pragma: no cover - defensive
    logger.error(
        "diffusers/torch not available for ControlNet. "
        "ControlNet depth pipeline will always fall back to dummy. Error: %s",
        e,
    )
    torch = None  # type: ignore
    ControlNetModel = None  # type: ignore
    StableDiffusionXLControlNetPipeline = None  # type: ignore


# Backend/pipelines/controlnet_depth.py
# parents[0] -> Backend/pipelines
# parents[1] -> Backend
BACKEND_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = BACKEND_ROOT / "outputs"

# Hugging Face repo IDs (must match your licensing decisions).
SDXL_BASE_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_CONTROLNET_DEPTH_REPO = "diffusers/controlnet-depth-sdxl-1.0"

# Lazy-loaded global for the ControlNet pipeline
_SDXL_CONTROLNET_DEPTH_PIPELINE: Optional["StableDiffusionXLControlNetPipeline"] = None  # type: ignore


def _ensure_outputs_dir() -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR


def _get_sdxl_controlnet_depth_pipeline() -> Optional["StableDiffusionXLControlNetPipeline"]:  # type: ignore
    """
    Lazily create (and cache) a Stable Diffusion XL + ControlNet Depth pipeline.

    CPU-only, safe for local testing. If anything fails, returns None and logs.
    """
    global _SDXL_CONTROLNET_DEPTH_PIPELINE

    if _SDXL_CONTROLNET_DEPTH_PIPELINE is not None:
        return _SDXL_CONTROLNET_DEPTH_PIPELINE

    if torch is None or ControlNetModel is None or StableDiffusionXLControlNetPipeline is None:
        logger.error("ControlNet dependencies are missing; cannot create pipeline.")
        return None

    try:
        logger.info(
            "Loading SDXL ControlNet Depth pipeline: base=%s, controlnet=%s",
            SDXL_BASE_REPO,
            SDXL_CONTROLNET_DEPTH_REPO,
        )

        # Load ControlNet for depth.
        controlnet = ControlNetModel.from_pretrained(
            SDXL_CONTROLNET_DEPTH_REPO,
            torch_dtype=torch.float32,  # CPU-only
        )

        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            SDXL_BASE_REPO,
            controlnet=controlnet,
            torch_dtype=torch.float32,  # CPU-only
        )

        pipe.to("cpu")
        pipe.set_progress_bar_config(disable=False)

        _SDXL_CONTROLNET_DEPTH_PIPELINE = pipe
        logger.info("SDXL ControlNet Depth pipeline created and cached.")
        return pipe

    except Exception as e:
        logger.error("Failed to create SDXL ControlNet Depth pipeline. Error: %s", e)
        _SDXL_CONTROLNET_DEPTH_PIPELINE = None
        return None


def run_controlnet_depth_txt2img(
    prompt: str,
    control_image_path: Union[str, Path],
    negative_prompt: Optional[str] = None,
    cfg_scale: Optional[float] = None,
    num_inference_steps: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    model_id: str = "sdxl-controlnet-depth",
    **extra: Any,
) -> Dict[str, Any]:
    """
    High-level TXT2IMG entry point for SDXL + ControlNet (depth).

    For now:
      - Expects a 'control_image_path' pointing to an existing image file.
      - If SDXL ControlNet pipeline is available, runs REAL CPU generation.
      - Otherwise, returns a dummy placeholder response.
    """
    control_image_path = Path(control_image_path)
    logger.info(
        "ControlNet Depth TXT2IMG request: prompt=%r, model_id=%s, control_image=%s",
        prompt,
        model_id,
        control_image_path,
    )

    # Validate the control image file.
    validate_image_file(control_image_path)

    # Defaults from config.
    cfg_scale = cfg_scale or app_config.txt2img_default_cfg_scale
    num_inference_steps = (
        num_inference_steps or app_config.txt2img_default_inference_steps
    )
    width = width or app_config.txt2img_default_width
    height = height or app_config.txt2img_default_height

    outputs_dir = _ensure_outputs_dir()
    filename = generate_output_filename(prefix="controlnet_depth", ext="png")
    output_path = outputs_dir / filename

    # Try to get a REAL SDXL ControlNet pipeline.
    pipe = _get_sdxl_controlnet_depth_pipeline()

    if pipe is None:
        # Dummy behavior if we couldn't create the pipeline.
        logger.warning(
            "SDXL ControlNet Depth pipeline unavailable. Returning dummy response. "
            "No real image will be written, reserved path: %s",
            output_path,
        )

        return {
            "status": "ok",
            "type": "txt2img_controlnet_depth",
            "engine": "SDXL-ControlNet-Depth (dummy)",
            "model_id": model_id,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "control_image_path": str(control_image_path),
            "width": width,
            "height": height,
            "cfg_scale": cfg_scale,
            "num_inference_steps": num_inference_steps,
            "output_image_path": str(output_path),
            "params": {
                "dummy": True,
                "extra": extra,
            },
            "debug": {
                "reason": "SDXL ControlNet Depth pipeline unavailable.",
            },
        }

    # REAL ControlNet path.
    try:
        control_image = Image.open(control_image_path).convert("RGB")
        logger.info("Running REAL SDXL ControlNet Depth TXT2IMG on CPU.")

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            # IMPORTANT: diffusers expects 'image' not to be None.
            # For now we reuse the same image for both base and control.
            image=control_image,
            control_image=control_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=cfg_scale,
            width=width,
            height=height,
            **extra,
        )

        image = result.images[0]
        image.save(output_path)
        logger.info("Saved REAL SDXL ControlNet Depth image to %s", output_path)

        return {
            "status": "ok",
            "type": "txt2img_controlnet_depth",
            "engine": "SDXL-ControlNet-Depth",
            "model_id": model_id,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "control_image_path": str(control_image_path),
            "width": width,
            "height": height,
            "cfg_scale": cfg_scale,
            "num_inference_steps": num_inference_steps,
            "output_image_path": str(output_path),
            "params": {
                "extra": extra,
            },
            "debug": {
                "pipeline": "StableDiffusionXLControlNetPipeline",
                "base_repo": SDXL_BASE_REPO,
                "controlnet_repo": SDXL_CONTROLNET_DEPTH_REPO,
            },
        }

    except Exception as e:
        logger.error(
            "Error while running REAL SDXL ControlNet Depth TXT2IMG. Error: %s",
            e,
        )

        return {
            "status": "error",
            "type": "txt2img_controlnet_depth",
            "engine": "SDXL-ControlNet-Depth",
            "model_id": model_id,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "control_image_path": str(control_image_path),
            "width": width,
            "height": height,
            "cfg_scale": cfg_scale,
            "num_inference_steps": num_inference_steps,
            "output_image_path": str(output_path),
            "params": {
                "extra": extra,
            },
            "debug": {
                "error": str(e),
            },
        }


__all__ = [
    "run_controlnet_depth_txt2img",
]
