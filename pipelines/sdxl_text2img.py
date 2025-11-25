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

# Backend/pipelines/sdxl_text2img.py
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
    model_id: str = "sdxl-base",
    **extra: Any,
) -> Dict[str, Any]:
    """
    High-level TXT2IMG entry point used by FastAPI.

    Behavior:
      - Gets a pipeline via get_txt2img_pipeline(...) (REAL SDXL or Dummy).
      - If pipeline returns a PIL image (REAL SDXL), we save it to disk.
      - If not, we behave as a placeholder (no real image).
    """
    logger.info("TXT2IMG request received: prompt=%r, model_id=%s", prompt, model_id)

    # Defaults from config
    cfg_scale = cfg_scale or app_config.txt2img_default_cfg_scale
    num_inference_steps = num_inference_steps or app_config.txt2img_default_inference_steps
    width = width or app_config.txt2img_default_width
    height = height or app_config.txt2img_default_height

    outputs_dir = _ensure_outputs_dir()
    filename = generate_output_filename(prefix="txt2img", ext="png")
    output_path = outputs_dir / filename

    # Fetch pipeline (SDXL or Dummy)
    pipeline = get_txt2img_pipeline(model_id=model_id)

    # Run the pipeline
    pipeline_result = pipeline.run(
        prompt=prompt,
        negative_prompt=negative_prompt,
        cfg_scale=cfg_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        **extra,
    )

    # REAL SDXL returns a PIL image inside the dict
    pil_image = pipeline_result.pop("pil_image", None)

    if pil_image is not None:
        try:
            pil_image.save(output_path)
            logger.info("Saved REAL TXT2IMG image to %s", output_path)
        except Exception as e:
            logger.error(
                "Failed to save REAL TXT2IMG image to %s. Error: %s",
                output_path,
                e,
            )
    else:
        logger.info(
            "TXT2IMG dummy mode. No real image generated. Reserved path: %s",
            output_path,
        )

    response: Dict[str, Any] = {
        "status": "ok",
        "type": "txt2img",
        "engine": pipeline_result.get("model_name"),
        "model_id": pipeline_result.get("model_id"),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "cfg_scale": cfg_scale,
        "num_inference_steps": num_inference_steps,
        "output_image_path": str(output_path),
        "params": pipeline_result.get("params", {}),
        "debug": pipeline_result,  # always JSON-safe now
    }

    return response


# -------------------------------------------------------------------
# Backwards-compatible aliases for main.py and older code
# -------------------------------------------------------------------

def generate_sd3_text2img(*, prompt: str, **kwargs: Any) -> Dict[str, Any]:
    logger.info("generate_sd3_text2img called; delegating to run_txt2img")
    return run_txt2img(prompt=prompt, **kwargs)


def txt2img(*, prompt: str, **kwargs: Any) -> Dict[str, Any]:
    return run_txt2img(prompt=prompt, **kwargs)


def generate(*, prompt: str, **kwargs: Any) -> Dict[str, Any]:
    return run_txt2img(prompt=prompt, **kwargs)


__all__ = [
    "run_txt2img",
    "generate_sd3_text2img",
    "txt2img",
    "generate",
]
