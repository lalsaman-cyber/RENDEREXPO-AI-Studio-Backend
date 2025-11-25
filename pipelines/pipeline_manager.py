from datetime import datetime
from typing import Any, Dict

try:
    from logs.log_utils import get_logger  # type: ignore
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_logger(name: str):
        return logging.getLogger(name)

from models.registry import get_model_info, ModelInfo  # type: ignore

# NEW: real SD3.5 pipeline factories
from .sd35_pipelines import (
    create_sd35_txt2img_pipeline,
    create_sd35_img2img_pipeline,
)

logger = get_logger(__name__)


# -------------------------------------------------------------------
# Base pipeline + dummy implementations
# -------------------------------------------------------------------

class BasePipeline:
    """
    Base class for pipelines.

    For dummy pipelines, this just stores ModelInfo and timestamps.
    For real pipelines (like SD3.5), we use separate wrapper classes
    defined in their own modules, but they follow the same idea:
    they expose a .run(...) method that returns a dict.
    """

    def __init__(self, model_info: ModelInfo):
        self.model_info = model_info
        self.created_at = datetime.utcnow()

    def run(self, **kwargs) -> dict:
        raise NotImplementedError


class DummyTxt2ImgPipeline(BasePipeline):
    """
    Placeholder TXT2IMG pipeline.

    Does NOT perform any real image generation.
    Returns structured metadata useful for FastAPI responses.
    """

    def run(self, prompt: str, **kwargs) -> dict:
        logger.info(
            "Running DummyTxt2ImgPipeline for model_id=%s",
            self.model_info.id,
        )
        return {
            "type": "txt2img",
            "model_id": self.model_info.id,
            "model_name": self.model_info.display_name,
            "prompt": prompt,
            "created_at": self.created_at.isoformat() + "Z",
            "run_at": datetime.utcnow().isoformat() + "Z",
            "params": kwargs,
        }


class DummyImg2ImgPipeline(BasePipeline):
    """
    Placeholder IMG2IMG pipeline.

    Does NOT perform any real image transformation.
    Returns structured metadata useful for FastAPI responses.
    """

    def run(self, prompt: str, init_image_info: dict | None = None, **kwargs) -> dict:
        logger.info(
            "Running DummyImg2ImgPipeline for model_id=%s",
            self.model_info.id,
        )
        return {
            "type": "img2img",
            "model_id": self.model_info.id,
            "model_name": self.model_info.display_name,
            "prompt": prompt,
            "init_image": init_image_info or {},
            "created_at": self.created_at.isoformat() + "Z",
            "run_at": datetime.utcnow().isoformat() + "Z",
            "params": kwargs,
        }


# -------------------------------------------------------------------
# Lazy-loading caches
# -------------------------------------------------------------------
# NOTE:
#   We store "Any" because the cache can hold either:
#     - DummyTxt2ImgPipeline / DummyImg2ImgPipeline
#     - Real SD3.5 wrappers from sd35_pipelines.py
#   All of them expose a .run(...) method returning a dict.
# -------------------------------------------------------------------

_TXT2IMG_PIPELINES: Dict[str, Any] = {}
_IMG2IMG_PIPELINES: Dict[str, Any] = {}


# -------------------------------------------------------------------
# Internal loaders for NON-SD3.5 models (dummy pipelines)
# -------------------------------------------------------------------

def _load_txt2img_pipeline(model_id: str) -> DummyTxt2ImgPipeline:
    """
    Internal helper to create a TXT2IMG pipeline for a given model_id.

    For non-SD3.5 models:
      - fetches ModelInfo from the registry
      - ensures the 'weights' file path exists (no real download)
      - constructs a DummyTxt2ImgPipeline instance
    """
    model_info = get_model_info(model_id)

    # We try to ensure the weights path exists.
    # If 'weights' is not defined, we just log a warning and move on.
    try:
        model_info.ensure_file("weights")
    except KeyError:
        logger.warning(
            "Model '%s' has no 'weights' entry in expected_files; "
            "skipping file ensure step for now.",
            model_id,
        )

    pipeline = DummyTxt2ImgPipeline(model_info=model_info)
    logger.info(
        "Created new DummyTxt2ImgPipeline for model_id=%s",
        model_id,
    )
    return pipeline


def _load_img2img_pipeline(model_id: str) -> DummyImg2ImgPipeline:
    """
    Internal helper to create an IMG2IMG pipeline for a given model_id.

    Same logic as TXT2IMG for now, for non-SD3.5 models.
    """
    model_info = get_model_info(model_id)

    try:
        model_info.ensure_file("weights")
    except KeyError:
        logger.warning(
            "Model '%s' has no 'weights' entry in expected_files; "
            "skipping file ensure step for now.",
            model_id,
        )

    pipeline = DummyImg2ImgPipeline(model_info=model_info)
    logger.info(
        "Created new DummyImg2ImgPipeline for model_id=%s",
        model_id,
    )
    return pipeline


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def get_txt2img_pipeline(model_id: str = "sdxl-base") -> Any:
    """
    Get (or lazily create) the TXT2IMG pipeline for a given model_id.

    Behavior:
      - If model_id starts with "sd3.5", we create a REAL SD3.5 txt2img pipeline
        using create_sd35_txt2img_pipeline(...).
      - Otherwise, we fall back to the DummyTxt2ImgPipeline.

    Later, this will let us:
      - use SD3.5 as the primary engine
      - keep SDXL/dummy pipelines around for testing or fallback
    """
    if model_id not in _TXT2IMG_PIPELINES:
        # SD3.5 branch (REAL pipeline)
        if model_id.startswith("sd3.5"):
            logger.info(
                "No cached REAL SD3.5 TXT2IMG pipeline for model_id=%s; loading...",
                model_id,
            )
            _TXT2IMG_PIPELINES[model_id] = create_sd35_txt2img_pipeline(
                model_id=model_id
            )
        else:
            # Non-SD3.5 models -> dummy
            logger.info(
                "No cached TXT2IMG pipeline for model_id=%s; creating dummy pipeline...",
                model_id,
            )
            _TXT2IMG_PIPELINES[model_id] = _load_txt2img_pipeline(model_id)

    return _TXT2IMG_PIPELINES[model_id]


def get_img2img_pipeline(model_id: str = "sdxl-base") -> Any:
    """
    Get (or lazily create) the IMG2IMG pipeline for a given model_id.

    Behavior:
      - If model_id starts with "sd3.5", we create a REAL SD3.5 img2img pipeline.
      - Otherwise, we fall back to the DummyImg2ImgPipeline.
    """
    if model_id not in _IMG2IMG_PIPELINES:
        if model_id.startswith("sd3.5"):
            logger.info(
                "No cached REAL SD3.5 IMG2IMG pipeline for model_id=%s; loading...",
                model_id,
            )
            _IMG2IMG_PIPELINES[model_id] = create_sd35_img2img_pipeline(
                model_id=model_id
            )
        else:
            logger.info(
                "No cached IMG2IMG pipeline for model_id=%s; creating dummy pipeline...",
                model_id,
            )
            _IMG2IMG_PIPELINES[model_id] = _load_img2img_pipeline(model_id)

    return _IMG2IMG_PIPELINES[model_id]


__all__ = [
    "BasePipeline",
    "DummyTxt2ImgPipeline",
    "DummyImg2ImgPipeline",
    "get_txt2img_pipeline",
    "get_img2img_pipeline",
]
