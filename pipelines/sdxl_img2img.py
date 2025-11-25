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

from .pipeline_manager import get_img2img_pipeline
from .sdxl_pipelines import SDXLImg2ImgPipelineWrapper  # type: ignore

from PIL import Image

logger = get_logger(__name__)

# Backend/pipelines/sdxl_img2img.py
BACKEND_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = BACKEND_ROOT / "outputs"


def _ensure_outputs_dir() -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR


def run_img2img(
    prompt: str,
    init_image_path: Union[str, Path],
    negative_prompt: Optional[str] = None,
    strength: Optional[float] = None,
    cfg_scale: Optional[float] = None,
    num_inference_steps: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    model_id: str = "sdxl-base",
    **extra: Any,
) -> Dict[str, Any]:
    """
    High-level IMG2IMG entry point for FastAPI.

    Behavior:
      - Validates the input image file.
      - Loads REAL SDXL pipeline when available.
      - Saves PIL output image if real, or dummy metadata if not.
    """
    init_image_path = Path(init_image_path)
    logger.info(
        "IMG2IMG request: prompt=%r, model_id=%s, init_image=%s",
        prompt,
        model_id,
        init_image_path,
    )

    validate_image_file(init_image_path)

    # defaults
    strength = strength if strength is not None else app_config.img2img_default_strength
    cfg_scale = cfg_scale or app_config.img2img_default_cfg_scale
    num_inference_steps = num_inference_steps or app_config.img2img_default_inference_steps
    width = width or app_config.img2img_default_width
    height = height or app_config.img2img_default_height

    outputs_dir = _ensure_outputs_dir()
    filename = generate_output_filename(prefix="img2img", ext="png")
    output_path = outputs_dir / filename

    pipeline = get_img2img_pipeline(model_id=model_id)

    init_image_info = {"path": str(init_image_path)}

    # REAL SDXL PIPELINE
    if isinstance(pipeline, SDXLImg2ImgPipelineWrapper):
        logger.info("Using REAL SDXL IMG2IMG pipeline.")
        init_image = Image.open(init_image_path).convert("RGB")

        pipeline_result = pipeline.run(
            prompt=prompt,
            init_image=init_image,
            negative_prompt=negative_prompt,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=cfg_scale,
            width=width,
            height=height,
            **extra,
        )

    # DUMMY PIPELINE
    else:
        logger.info("Using DUMMY IMG2IMG pipeline (no real generation).")
        pipeline_result = pipeline.run(
            prompt=prompt,
            init_image_info=init_image_info,
            negative_prompt=negative_prompt,
            strength=strength,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            **extra,
        )

    # Handle PIL output
    pil_image = pipeline_result.pop("pil_image", None)

    if pil_image is not None:
        try:
            pil_image.save(output_path)
            logger.info("Saved REAL IMG2IMG image to %s", output_path)
        except Exception as e:
            logger.error(
                "Failed to save REAL IMG2IMG image to %s. Error: %s",
                output_path,
                e,
            )
    else:
        logger.info(
            "IMG2IMG dummy mode. No real output generated. Reserved path: %s",
            output_path,
        )

    response: Dict[str, Any] = {
        "status": "ok",
        "type": "img2img",
        "engine": pipeline_result.get("model_name"),
        "model_id": pipeline_result.get("model_id"),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "init_image_path": str(init_image_path),
        "width": width,
        "height": height,
        "strength": strength,
        "cfg_scale": cfg_scale,
        "num_inference_steps": num_inference_steps,
        "output_image_path": str(output_path),
        "params": pipeline_result.get("params", {}),
        "debug": pipeline_result,
    }

    return response


# -------------------------------------------------------------------
# API aliases
# -------------------------------------------------------------------

def generate_sd3_img2img(
    *,
    prompt: str,
    init_image_path: Union[str, Path, None] = None,
    init_image: Union[str, Path, None] = None,
    **kwargs: Any,
) -> Dict[str, Any]:

    resolved_path = init_image_path or init_image or kwargs.pop("image_path", None)
    if resolved_path is None:
        raise ValueError(
            "generate_sd3_img2img requires init image path "
            "(init_image_path, init_image, or image_path)."
        )

    logger.info(
        "generate_sd3_img2img delegating to run_img2img with init=%s",
        resolved_path,
    )

    return run_img2img(prompt=prompt, init_image_path=resolved_path, **kwargs)


def img2img(*, prompt: str, init_image_path: Union[str, Path], **kwargs: Any) -> Dict[str, Any]:
    return run_img2img(prompt=prompt, init_image_path=init_image_path, **kwargs)


def transform(*, prompt: str, init_image_path: Union[str, Path], **kwargs: Any) -> Dict[str, Any]:
    return run_img2img(prompt=prompt, init_image_path=init_image_path, **kwargs)


__all__ = [
    "run_img2img",
    "generate_sd3_img2img",
    "img2img",
    "transform",
]
