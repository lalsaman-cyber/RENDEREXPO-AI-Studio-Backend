# runtime/sd35_runtime.py
#
# SD3.5 RUNTIME (LOCAL SKELETON)
#
# IMPORTANT:
# - This version is for your **local dev** environment only.
# - It does NOT import diffusers or load the real SD3.5 model.
# - We ONLY define the class + data structures so that:
#       * app.gpu_entry:app can start
#       * pipeline_manager can import SD35Runtime
# - Real SD3.5 loading + inference will be implemented later
#   inside the RunPod Docker image with a newer `diffusers`
#   version that supports Stable Diffusion 3 / 3.5.
#
# So: no CUDA, no VRAM use, no model weights loaded here.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class GenerationResult:
    """
    Minimal result object for a generation call.

    For now this is just a placeholder so the rest of the
    codebase has something consistent to work with.
    """
    image_path: Optional[str]
    meta: Dict[str, Any]


class SD35Runtime:
    """
    Skeleton SD3.5 runtime.

    In this local version:
    - `load()` does NOT load any model weights.
    - `generate_text2img()` and `generate_img2img()` will raise,
      because real inference is not available locally yet.

    The real implementation (with diffusers + CUDA) will live in
    the RunPod Docker container, where we can safely install a
    newer `diffusers` that supports SD3 / SD3.5.
    """

    def __init__(self, device: str = "cpu", model_dir: str = "models/sd35-large"):
        self.device = device
        self.model_dir = model_dir
        self.is_loaded: bool = False

    def load(self) -> None:
        """
        LOCAL SKELETON:

        - Do NOT load any actual models.
        - Just mark runtime as 'loaded' so other code paths
          can check the flag if they want to.
        """
        self.is_loaded = True

    def unload(self) -> None:
        """
        LOCAL SKELETON:

        - Simply flip the flag back.
        """
        self.is_loaded = False

    # ------------------------------------------------------------------
    # Text2Img / Img2Img placeholders
    # ------------------------------------------------------------------

    def generate_text2img(self, meta: Dict[str, Any]) -> GenerationResult:
        """
        Placeholder for SD3.5 text-to-image.

        On local dev:
        - This should NOT be called for real inference.
        - If it is, we raise a clear error so you know
          this needs to run on the GPU (RunPod) version.
        """
        raise RuntimeError(
            "SD35Runtime.generate_text2img is not implemented in the local "
            "skeleton runtime. Real SD3.5 inference will run in the RunPod "
            "GPU container with a newer diffusers version."
        )

    def generate_img2img(self, meta: Dict[str, Any]) -> GenerationResult:
        """
        Placeholder for SD3.5 image-to-image.

        Same note as generate_text2img.
        """
        raise RuntimeError(
            "SD35Runtime.generate_img2img is not implemented in the local "
            "skeleton runtime. Real SD3.5 inference will run in the RunPod "
            "GPU container with a newer diffusers version."
        )
