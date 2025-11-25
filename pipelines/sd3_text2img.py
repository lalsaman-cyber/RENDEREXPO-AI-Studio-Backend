from pathlib import Path
from typing import Any, Dict, Optional

try:
    from logs.log_utils import get_logger  # type: ignore
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_logger(name: str):
        return logging.getLogger(name)

from config import app_config  # type: ignore
from file_utils import generate_output_filename  # type: ignore

from .pipeline_manager import get_txt2img_pipeline

logger = get_logger(__name__)

# Backend/pipelines/sd3_txt2img.py
# parents[0] -> Backend/pipelines
# parents[1] -> Backend
BACKEND_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = BACKEND_ROOT / "outputs"


def _ensure_outputs_dir() -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR


def run_txt2img(
    prompt: str,
    negative_prompt: Optional[str] = None,
    cfg_scale: Optional[float] = None,
    num_inference_steps: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    model_id: str = "sd3.5-large",
    seed: Optional[int] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """
    High-level TXT2IMG entry point used by FastAPI.

    Behavior:

    - Uses get_txt2img_pipeline(...) from pipeline_manager.py
    - If the underlying pipeline returns a real PIL image (SD3.5),
      we save it to Backend/outputs/ as a PNG.
    - If it's a dummy pipeline (e.g. SDXL in placeholder mode), we
      DO NOT write an image, but still return a reserved output path.

    This keeps the API stable while letting us upgrade the engine.
    """

    logger.info(
        "TXT2IMG request received: prompt=%r, model_id=%s",
        prompt,
        model_id,
    )

    # Fill in defaults from config if values are not provided.
    cfg_scale = cfg_scale or app_config.txt2img_default_cfg_scale
    num_inference_steps = (
        num_inference_steps or app_config.txt2img_default_inference_steps
    )
    width = width or app_config.txt2img_default_width
    height = height or app_config.txt2img_default_height

    outputs_dir = _ensure_outputs_dir()
    filename = generate_output_filename(prefix="txt2img", ext="png")
    output_path = outputs_dir / filename

    # Get (or lazily create) the pipeline instance.
    pipeline = get_txt2img_pipeline(model_id=model_id)

    # Run the pipeline.
    # For SD3.5, this should return a dict with "pil_image".
    # For dummy pipelines, it returns metadata only.
    pipeline_result = pipeline.run(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=cfg_scale,  # SD3.5 wrapper expects "guidance_scale"
        seed=seed,
        **extra,
    )

    real_image = False

    # If the pipeline produced a PIL image (real SD3.5), save it.
    pil_image = pipeline_result.get("pil_image")
    if pil_image is not None:
        try:
            pil_image.save(output_path, format="PNG")
            real_image = True
            logger.info(
                "TXT2IMG: Saved real SD3.5 image to: %s",
                output_path,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Failed to save SD3.5 image to %s: %s", output_path, e)
    else:
        # Dummy pipeline case: no real image, just reserve the path.
        logger.info(
            "TXT2IMG: No PIL image returned by pipeline (likely dummy). "
            "Reserved output path (no image written): %s",
            output_path,
        )

    response: Dict[str, Any] = {
        "status": "ok",
        "type": "txt2img",
        "engine": pipeline_result.get("model_name"),
        "model_id": pipeline_result.get("model_id", model_id),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "cfg_scale": cfg_scale,
        "num_inference_steps": num_inference_steps,
        "output_image_path": str(output_path),
        "real_image": real_image,
        "params": pipeline_result.get("params", {}),
        "debug": {
            # We only include safe, non-huge debug info.
            "device": pipeline_result.get("device"),
            "pipeline_type": pipeline_result.get("type"),
        },
    }

    return response


# -------------------------------------------------------------------
# Backwards-compatible aliases
# -------------------------------------------------------------------

def txt2img(*, prompt: str, **kwargs: Any) -> Dict[str, Any]:
    """
    Convenience/compatibility wrapper.

    If older code calls `txt2img(prompt=..., ...)`, it will still work.
    """
    return run_txt2img(prompt=prompt, **kwargs)


def generate(*, prompt: str, **kwargs: Any) -> Dict[str, Any]:
    """
    Another convenience alias that some earlier stubs might have used.
    All paths go through run_txt2img to keep behavior consistent.
    """
    return run_txt2img(prompt=prompt, **kwargs)


__all__ = [
    "run_txt2img",
    "txt2img",
    "generate",
]
