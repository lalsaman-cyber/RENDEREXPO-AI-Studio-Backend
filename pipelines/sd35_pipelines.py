from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import os

try:
    from logs.log_utils import get_logger  # type: ignore
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)

# Optional imports – we keep things graceful if deps are missing.
try:
    import torch
    from diffusers import DiffusionPipeline
except ImportError:  # pragma: no cover - graceful degradation
    torch = None  # type: ignore
    DiffusionPipeline = None  # type: ignore
    logger.warning(
        "SD3.5 dependencies (torch/diffusers) are not installed. "
        "SD3.5 pipelines will not be usable until they are."
    )

from models.registry import get_model_info, ModelInfo  # type: ignore


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _ensure_sd35_dependencies() -> None:
    """
    Ensure that torch + diffusers are importable.
    Raise a clear error if not, so FastAPI responses can surface it.
    """
    if torch is None or DiffusionPipeline is None:
        raise RuntimeError(
            "SD3.5 dependencies are not available. "
            "Install with: pip install diffusers transformers accelerate safetensors torch"
        )


def _select_device_and_dtype() -> Tuple[str, "torch.dtype"]:
    """
    Decide whether to use CPU or GPU and which dtype to use.

    - On your local machine (likely no CUDA): CPU + float32
    - On RunPod A6000: CUDA + float16
    """
    assert torch is not None  # for type-checkers

    if torch.cuda.is_available():
        logger.info("CUDA is available – using GPU (float16) for SD3.5.")
        return "cuda", torch.float16
    else:
        logger.info("CUDA is NOT available – using CPU (float32) for SD3.5.")
        return "cpu", torch.float32


def _log_hf_env() -> None:
    """
    Log basic Hugging Face environment info (without printing secrets).
    Helpful for debugging access issues (like 401).
    """
    hf_token_present = bool(os.environ.get("HF_TOKEN"))
    hf_home = os.environ.get("HF_HOME")
    logger.info(
        "HF environment – token_present=%s, HF_HOME=%s",
        hf_token_present,
        hf_home,
    )


# -------------------------------------------------------------------
# SD3.5 TXT2IMG Pipeline Wrapper
# -------------------------------------------------------------------

@dataclass
class SD35Txt2ImgPipelineWrapper:
    """
    Thin wrapper around a diffusers DiffusionPipeline for SD3.5 TXT2IMG.

    NOTE:
      - Uses Hugging Face repo from ModelInfo (registry)
      - CPU/GPU is selected automatically
      - Real generation will be integrated into the FastAPI layer
        via sd3_txt2img.py (which will call this wrapper).
    """

    model_info: ModelInfo
    pipe: Any
    device: str

    @classmethod
    def create(cls, model_id: str = "sd3.5-large") -> "SD35Txt2ImgPipelineWrapper":
        _ensure_sd35_dependencies()
        _log_hf_env()

        model_info = get_model_info(model_id)
        logger.info(
            "Creating SD3.5 TXT2IMG pipeline for model_id=%s (repo=%s)",
            model_id,
            model_info.repo_id,
        )

        device, dtype = _select_device_and_dtype()

        # IMPORTANT:
        # We rely on HF_TOKEN being set in the environment if the repo is gated.
        # DiffusionPipeline.from_pretrained will use that automatically via
        # huggingface_hub auth.
        pipe = DiffusionPipeline.from_pretrained(
            model_info.repo_id,
            torch_dtype=dtype,
            use_safetensors=True,
        )

        pipe = pipe.to(device)

        # Memory-friendly options
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()

        logger.info(
            "SD3.5 TXT2IMG pipeline created on device=%s (dtype=%s)",
            device,
            dtype,
        )

        return cls(model_info=model_info, pipe=pipe, device=device)

    def run(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run SD3.5 TXT2IMG and return a dict with:
          - PIL image
          - metadata useful for logging and API responses
        """
        assert torch is not None  # type: ignore

        logger.info(
            "Running REAL SD3.5 TXT2IMG on device=%s for model_id=%s",
            self.device,
            self.model_info.id,
        )

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **kwargs,
        )

        # Most diffusers pipelines return an object with .images (list of PIL images)
        image = result.images[0]

        return {
            "type": "txt2img",
            "model_id": self.model_info.id,
            "model_name": self.model_info.display_name,
            "pil_image": image,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "device": self.device,
        }


# -------------------------------------------------------------------
# SD3.5 IMG2IMG Pipeline Wrapper
# -------------------------------------------------------------------

@dataclass
class SD35Img2ImgPipelineWrapper:
    """
    Thin wrapper around a diffusers DiffusionPipeline for SD3.5 IMG2IMG.

    We keep this generic, assuming the SD3.5 pipeline supports:
      pipe(
        prompt=...,
        image=init_image,
        negative_prompt=...,
        strength=...,
        num_inference_steps=...,
        guidance_scale=...,
        width=...,
        height=...
      )
    """

    model_info: ModelInfo
    pipe: Any
    device: str

    @classmethod
    def create(cls, model_id: str = "sd3.5-large") -> "SD35Img2ImgPipelineWrapper":
        _ensure_sd35_dependencies()
        _log_hf_env()

        model_info = get_model_info(model_id)
        logger.info(
            "Creating SD3.5 IMG2IMG pipeline for model_id=%s (repo=%s)",
            model_id,
            model_info.repo_id,
        )

        device, dtype = _select_device_and_dtype()

        pipe = DiffusionPipeline.from_pretrained(
            model_info.repo_id,
            torch_dtype=dtype,
            use_safetensors=True,
        )

        pipe = pipe.to(device)

        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()

        logger.info(
            "SD3.5 IMG2IMG pipeline created on device=%s (dtype=%s)",
            device,
            dtype,
        )

        return cls(model_info=model_info, pipe=pipe, device=device)

    def run(
        self,
        prompt: str,
        init_image,
        negative_prompt: Optional[str] = None,
        strength: float = 0.8,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run SD3.5 IMG2IMG and return:
          - PIL image
          - metadata
        """
        assert torch is not None  # type: ignore

        logger.info(
            "Running REAL SD3.5 IMG2IMG on device=%s for model_id=%s",
            self.device,
            self.model_info.id,
        )

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            image=init_image,
            negative_prompt=negative_prompt,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            **kwargs,
        )

        image = result.images[0]

        return {
            "type": "img2img",
            "model_id": self.model_info.id,
            "model_name": self.model_info.display_name,
            "pil_image": image,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "width": width,
            "height": height,
            "device": self.device,
        }


# -------------------------------------------------------------------
# Factories (for Pipeline Manager to use later)
# -------------------------------------------------------------------

def create_sd35_txt2img_pipeline(
    model_id: str = "sd3.5-large",
) -> SD35Txt2ImgPipelineWrapper:
    """
    Factory used by the pipeline manager to get a REAL SD3.5 txt2img pipeline.
    """
    return SD35Txt2ImgPipelineWrapper.create(model_id=model_id)


def create_sd35_img2img_pipeline(
    model_id: str = "sd3.5-large",
) -> SD35Img2ImgPipelineWrapper:
    """
    Factory used by the pipeline manager to get a REAL SD3.5 img2img pipeline.
    """
    return SD35Img2ImgPipelineWrapper.create(model_id=model_id)


__all__ = [
    "SD35Txt2ImgPipelineWrapper",
    "SD35Img2ImgPipelineWrapper",
    "create_sd35_txt2img_pipeline",
    "create_sd35_img2img_pipeline",
]
