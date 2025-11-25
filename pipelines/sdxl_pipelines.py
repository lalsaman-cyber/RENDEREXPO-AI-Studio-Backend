# Backend/pipelines/sdxl_pipelines.py

import os
from typing import Any, Dict, Optional

try:
    from logs.log_utils import get_logger  # type: ignore
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_logger(name: str):
        return logging.getLogger(name)

import torch  # type: ignore
from diffusers import (  # type: ignore
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)

logger = get_logger(__name__)

# -------------------------------------------------------------------
# Device / model helpers
# -------------------------------------------------------------------

def _resolve_device() -> torch.device:
    """
    Decide where SDXL runs: CPU now, GPU later.
    Uses env var RENDEREXPO_DEVICE: "cpu" | "cuda" | "mps".
    Default is "cpu" for safety/cost.
    """
    env_val = os.environ.get("RENDEREXPO_DEVICE", "cpu").lower()

    if env_val == "cuda" and torch.cuda.is_available():
        logger.info("Using CUDA device for SDXL pipelines.")
        return torch.device("cuda")
    if env_val == "mps" and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        logger.info("Using MPS device for SDXL pipelines.")
        return torch.device("mps")

    logger.info("Using CPU device for SDXL pipelines.")
    return torch.device("cpu")


# Map our logical model IDs to Hugging Face repos.
_MODEL_ID_MAP: Dict[str, str] = {
    "sdxl-base": "stabilityai/stable-diffusion-xl-base-1.0",
    # Later we can add:
    # "sdxl-refiner": "stabilityai/stable-diffusion-xl-refiner-1.0",
}


def _resolve_repo_id(model_id: str) -> str:
    """
    Translate our model_id (e.g. 'sdxl-base') to a repo id.
    If the key is missing, assume the user passed a repo id directly.
    """
    return _MODEL_ID_MAP.get(model_id, model_id)


# -------------------------------------------------------------------
# SDXL TXT2IMG wrapper
# -------------------------------------------------------------------

class SDXLTxt2ImgPipelineWrapper:
    """
    Thin wrapper around StableDiffusionXLPipeline.

    .run(...) returns a dict that always contains:
      - "model_id"
      - "model_name"
      - "pil_image" (PIL.Image.Image) for REAL runs
      - "params" (dict of arguments actually used)
    so that sd3_text2img.py can:
      - save the image to /app/outputs
      - pass back clean JSON metadata to FastAPI.
    """

    def __init__(self, model_id: str = "sdxl-base"):
        self.model_id = model_id
        self.repo_id = _resolve_repo_id(model_id)
        self.device = _resolve_device()

        dtype = torch.float16 if self.device.type in ("cuda", "mps") else torch.float32

        logger.info(
            "Loading SDXL TXT2IMG pipeline: model_id=%s, repo_id=%s, device=%s, dtype=%s",
            self.model_id,
            self.repo_id,
            self.device,
            dtype,
        )

        self.pipe: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_pretrained(
            self.repo_id,
            torch_dtype=dtype,
            use_safetensors=True,
        )
        self.pipe.to(self.device)

        # A friendly display name for logs / responses.
        self.model_name = f"Stable Diffusion XL (txt2img, {self.model_id})"

    def run(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        cfg_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """
        Execute a real SDXL txt2img generation and return a metadata dict.
        The PIL image is attached as 'pil_image'; callers will save it.
        """
        logger.info(
            "Running REAL SDXL TXT2IMG: model_id=%s, prompt=%r",
            self.model_id,
            prompt,
        )

        # Sensible defaults if caller passed None
        if cfg_scale is None:
            cfg_scale = 7.0
        if num_inference_steps is None:
            num_inference_steps = 20
        if width is None:
            width = 1024
        if height is None:
            height = 768

        with torch.inference_mode():
            out = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=cfg_scale,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
            )

        pil_image = out.images[0]

        return {
            "type": "txt2img",
            "model_id": self.model_id,
            "model_name": self.model_name,
            "pil_image": pil_image,
            "params": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "cfg_scale": cfg_scale,
                "num_inference_steps": num_inference_steps,
                "width": width,
                "height": height,
                **extra,
            },
        }


# -------------------------------------------------------------------
# SDXL IMG2IMG wrapper
# -------------------------------------------------------------------

class SDXLImg2ImgPipelineWrapper:
    """
    Thin wrapper around StableDiffusionXLImg2ImgPipeline.

    .run(...) takes a PIL init_image and returns:
      - "model_id"
      - "model_name"
      - "pil_image"
      - "params"
    """

    def __init__(self, model_id: str = "sdxl-base"):
        self.model_id = model_id
        self.repo_id = _resolve_repo_id(model_id)
        self.device = _resolve_device()

        dtype = torch.float16 if self.device.type in ("cuda", "mps") else torch.float32

        logger.info(
            "Loading SDXL IMG2IMG pipeline: model_id=%s, repo_id=%s, device=%s, dtype=%s",
            self.model_id,
            self.repo_id,
            self.device,
            dtype,
        )

        self.pipe: StableDiffusionXLImg2ImgPipeline = (
            StableDiffusionXLImg2ImgPipeline.from_pretrained(
                self.repo_id,
                torch_dtype=dtype,
                use_safetensors=True,
            )
        )
        self.pipe.to(self.device)

        self.model_name = f"Stable Diffusion XL (img2img, {self.model_id})"

    def run(
        self,
        prompt: str,
        init_image,
        negative_prompt: Optional[str] = None,
        strength: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """
        Execute a real SDXL img2img transformation and return metadata + PIL image.
        """
        logger.info(
            "Running REAL SDXL IMG2IMG: model_id=%s, prompt=%r, strength=%s",
            self.model_id,
            prompt,
            strength,
        )

        if strength is None:
            strength = 0.7
        if guidance_scale is None:
            guidance_scale = 7.0
        if num_inference_steps is None:
            num_inference_steps = 25

        # width/height are optional; SDXL can infer from init_image.
        with torch.inference_mode():
            out = self.pipe(
                prompt=prompt,
                image=init_image,
                negative_prompt=negative_prompt,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
            )

        pil_image = out.images[0]

        return {
            "type": "img2img",
            "model_id": self.model_id,
            "model_name": self.model_name,
            "pil_image": pil_image,
            "params": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "strength": strength,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "width": width,
                "height": height,
                **extra,
            },
        }


# -------------------------------------------------------------------
# Factory functions used by pipeline_manager.py
# -------------------------------------------------------------------

def create_sdxl_txt2img_pipeline(model_id: str = "sdxl-base") -> SDXLTxt2ImgPipelineWrapper:
    """
    Factory for SDXL txt2img wrapper.
    Called by pipeline_manager._load_txt2img_pipeline(...)
    """
    return SDXLTxt2ImgPipelineWrapper(model_id=model_id)


def create_sdxl_img2img_pipeline(model_id: str = "sdxl-base") -> SDXLImg2ImgPipelineWrapper:
    """
    Factory for SDXL img2img wrapper.
    Called by pipeline_manager._load_img2img_pipeline(...)
    """
    return SDXLImg2ImgPipelineWrapper(model_id=model_id)


__all__ = [
    "SDXLTxt2ImgPipelineWrapper",
    "SDXLImg2ImgPipelineWrapper",
    "create_sdxl_txt2img_pipeline",
    "create_sdxl_img2img_pipeline",
]
