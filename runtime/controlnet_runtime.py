"""
runtime/controlnet_runtime.py

Skeleton ControlNet runtime for RENDEREXPO AI STUDIO.

IMPORTANT:
- This is a placeholder for future GPU usage.
- On your laptop (CPU), we DO NOT load any heavy ControlNet models here.
- The real implementation will live in the GPU environment (RunPod).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional


# Allowed control types we plan to support
ALLOWED_CONTROL_TYPES = [
    "canny",
    "depth",
    "lineart",
    "sketch",
    "floorplan",
    "normal",
]


@dataclass
class ControlNetConfig:
    """
    Simple configuration holder for future ControlNet setup.
    """
    device: str = "cuda"
    dtype: str = "float16"


class ControlNetRuntime:
    """
    Skeleton runtime wrapper for ControlNet.

    For NOW (local CPU):
    - We DO NOT load any real ControlNet weights.
    - We DO NOT run inference.
    - We only exist so the PipelineManager can reference ControlNetRuntime
      without crashing.

    Later (on GPU / RunPod):
    - This class will:
        * load specific ControlNet models
        * apply them to SD3.5 pipelines
        * support sketch, floorplan, canny, lineart, depth, etc.
    """

    def __init__(self, device: str = "cpu", config: Optional[ControlNetConfig] = None) -> None:
        self.device = device
        self.config = config or ControlNetConfig(device=device)
        self.is_loaded: bool = False
        self.models: Dict[str, Any] = {}  # e.g. {"canny": <ControlNetModel>, ...}

    def load(self) -> None:
        """
        Placeholder load method.

        In the GPU version, this would:
        - Load specific ControlNet models from disk or Hugging Face.
        - Move them to the correct device (cuda).
        - Possibly wrap them in a diffusers pipeline.

        On LOCAL (CPU) dev:
        - DO NOT load anything.
        - Just mark as 'loaded' so other code doesn't break.
        """
        # In the future, we might do something like:
        #   from diffusers import ControlNetModel
        #   self.models["canny"] = ControlNetModel.from_pretrained(...)
        #
        # But for now: no-op
        self.is_loaded = True

    def is_ready(self) -> bool:
        """
        Returns True if the runtime is considered 'loaded'.
        For now, this just reflects the flag set in load().
        """
        return self.is_loaded

    def list_supported_controls(self) -> Dict[str, Any]:
        """
        Return the list of supported control types (conceptually).
        """
        return {
            "supported_control_types": ALLOWED_CONTROL_TYPES,
            "device": self.device,
            "loaded": self.is_loaded,
        }

    def apply_control(
        self,
        control_type: str,
        image_path: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        strength: float = 1.0,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Placeholder for applying control to an image.

        In the future:
        - This will run ControlNet-guided SD3.5 inference.
        - It will output a path to the newly rendered image.

        For NOW (local dev):
        - We raise NotImplementedError to avoid accidental use.
        """
        raise NotImplementedError(
            "ControlNetRuntime.apply_control is not implemented yet. "
            "This will be implemented in the GPU / RunPod environment."
        )
