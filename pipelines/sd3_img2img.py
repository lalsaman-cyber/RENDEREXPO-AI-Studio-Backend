# pipelines/sd3_img2img.py

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
from .sd35_pipelines import SD35Img2ImgPipelineWrapper  # type: ignore

from PIL import Image

logger = get_logger(__name__)

# -------------------------------------------------------------------
# Output directory (FORCE /app/outputs so it matches Docker volume)
# -------------------------------------------------------------------

BACKEND_ROOT = Path("/app")
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
    model_id: str = "sd3.5-large",
    seed: Optional[int] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """
    High-level IMG2IMG entry point used by FastAPI.

    Behavior:
      - Validates the input image file.
      - Gets a pipeline via get_img2img_pipeline(...).
      - If REAL SD3.5: loads init image as PIL, runs it, saves output PNG under /app/outputs.
      - If Dummy: returns placeholder metadata and copies init image as fallback.
      - Ensures there is ALWAYS a real file at output_image_path.
    """
    init_image_path = Path(init_image_path)
    logger.info(
        "IMG2IMG request received: prompt=%r, model_id=%s, init_image=%s",
        prompt,
        model_id,
        init_image_path,
    )

    # Validate the input image.
    validate_image_file(init_image_path)

    # Fill in defaults from config if values are not provided.
    strength = strength if strength is not None else app_config.img2img_default_strength
    cfg_scale = cfg_scale or app_config.img2img_default_cfg_scale
    num_inference_steps = (
        num_inference_steps or app_config.img2img_default_inference_steps
    )
    width = width or app_config.img2img_default_width
    height = height or app_config.img2img_default_height

    outputs_dir = _ensure_outputs_dir()
    filename = generate_output_filename(prefix="img2img", ext="png")
    output_path = outputs_dir / filename

    # Get (or lazily create) the pipeline instance.
    pipeline = get_img2img_pipeline(model_id=model_id)

    init_image_info = {
        "path": str(init_image_path),
    }

    # -------------------------------------------------------------------
    # Decide how to call .run() based on the actual pipeline type.
    # -------------------------------------------------------------------
    if isinstance(pipeline, SD35Img2ImgPipelineWrapper):
        # REAL SD3.5 path: load the init image into memory as PIL.
        init_image = Image.open(init_image_path).convert("RGB")
        logger.info("Using REAL SD3.5 IMG2IMG pipeline with loaded init image.")

        pipeline_result = pipeline.run(
            prompt=prompt,
            init_image=init_image,
            negative_prompt=negative_prompt,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=cfg_scale,
            width=width,
            height=height,
            seed=seed,
            **extra,
        )
    else:
        # Dummy (or other non-SD3.5) path: keep old behavior (no real image processing).
        logger.info("Using Dummy IMG2IMG pipeline (no real SD3.5 transformation).")
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

    # -------------------------------------------------------------------
    # Try to get a real PIL image from the pipeline.
    # If we don't get one, we fall back to copying the init image.
    # -------------------------------------------------------------------
    pil_image = pipeline_result.pop("pil_image", None)

    if pil_image is not None:
        # Normal path: real SD3.5 output.
        try:
            pil_image.save(output_path)
            logger.info("Saved REAL IMG2IMG image to %s", output_path)
        except Exception as e:  # noqa: BLE001
            logger.error(
                "Failed to save REAL IMG2IMG image to %s. Error: %s",
                output_path,
                e,
            )
            # Fallback: copy init image
            try:
                placeholder_img = Image.open(init_image_path).convert("RGB")
                placeholder_img.save(output_path)
                logger.info(
                    "Wrote placeholder IMG2IMG output by copying init image to %s.",
                    output_path,
                )
            except Exception as e2:  # noqa: BLE001
                logger.error(
                    "Failed to write placeholder IMG2IMG output to %s. Error: %s",
                    output_path,
                    e2,
                )
    else:
        # Fallback: ensure there is ALWAYS a file at output_path.
        try:
            placeholder_img = Image.open(init_image_path).convert("RGB")
            placeholder_img.save(output_path)
            logger.info(
                "No 'pil_image' in pipeline_result; copied init_image to %s "
                "as a placeholder.",
                output_path,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(
                "Failed to write placeholder IMG2IMG output to %s. Error: %s",
                output_path,
                e,
            )

    # Build JSON-serializable response (no PIL objects).
    response: Dict[str, Any] = {
        "status": "ok",
        "type": "img2img",
        "engine": pipeline_result.get("model_name"),
        "model_id": pipeline_result.get("model_id", model_id),
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
        "debug": pipeline_result,  # JSON-safe, no PIL image object now
    }

    return response


# -------------------------------------------------------------------
# Backwards-compatible aliases
# -------------------------------------------------------------------

def generate_sd3_img2img(
    *,
    prompt: str,
    init_image_path: Union[str, Path, None] = None,
    init_image: Union[str, Path, None] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Legacy-compatible entrypoint (for main.py) that funnels into run_img2img.

    We accept multiple possible parameter names for the init image:
      - init_image_path
      - init_image
      - image_path (if passed via kwargs)
    """

    # Try to resolve the init image path from several possible argument names.
    resolved_path = init_image_path or init_image or kwargs.pop("image_path", None)

    if resolved_path is None:
        raise ValueError(
            "generate_sd3_img2img requires an init image path. "
            "Expected one of: init_image_path, init_image, image_path."
        )

    logger.info(
        "generate_sd3_img2img called; delegating to run_img2img with init_image_path=%s",
        resolved_path,
    )

    return run_img2img(prompt=prompt, init_image_path=resolved_path, **kwargs)


def img2img(*, prompt: str, init_image_path: Union[str, Path], **kwargs: Any) -> Dict[str, Any]:
    """
    Convenience/compatibility wrapper for older code that calls:
      img2img(prompt=..., init_image_path=..., ...)
    """
    return run_img2img(prompt=prompt, init_image_path=init_image_path, **kwargs)


def transform(*, prompt: str, init_image_path: Union[str, Path], **kwargs: Any) -> Dict[str, Any]:
    """
    Another compatibility alias that funnels into run_img2img.
    """
    return run_img2img(prompt=prompt, init_image_path=init_image_path, **kwargs)


__all__ = [
    "run_img2img",
    "generate_sd3_img2img",
    "img2img",
    "transform",
]
