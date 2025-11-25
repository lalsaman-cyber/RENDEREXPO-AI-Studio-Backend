# modules/sd35_text2img/__init__.py
"""
SD3.5 Text-to-Image pipeline skeleton for RENDEREXPO AI STUDIO.

IMPORTANT (Phase 1):
- This file defines the *shape* of the text2img pipeline.
- It does NOT import torch or diffusers.
- It does NOT load any models.
- It is safe to import on CPU-only machines.

Later (Phase 3, on GPU / RunPod):
- We will fill in the real SD3.5 loading and inference code.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class SD35Text2ImgConfig:
    """
    Basic configuration for an SD3.5 text2img run.

    All fields here map very closely to what you already store
    in meta.json for a text2img job.
    """
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 25
    guidance_scale: float = 6.0
    seed: Optional[int] = None
    style_preset: Optional[str] = None
    material_preset: Optional[str] = None
    lighting_preset: Optional[str] = None


class SD35Text2ImgPipeline:
    """
    Skeleton Text-to-Image pipeline for SD3.5.

    Phase 1 (now):
    - This is just a placeholder with the *interface* we will need.
    - No real AI happens here.

    Phase 3 (GPU / RunPod):
    - This class will take a loaded SD3.5 model (from SD35Runtime),
      build a diffusers pipeline, and actually generate images.
    """

    def __init__(self, config: Optional[SD35Text2ImgConfig] = None):
        self.config = config or SD35Text2ImgConfig()

    def run(
        self,
        prompt: str,
        negative_prompt: Optional[str],
        job_folder: str,
        planned_output_image: str,
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run a single text2img job.

        Phase 1:
        - This method does NOT do real image generation.
        - It only returns a dict describing what *would* happen.

        Phase 3:
        - We will:
            * use SD35Runtime to get the loaded SD3.5 model
            * run actual inference on GPU
            * save the real PNG to planned_output_image
            * update and return meta info
        """
        # NOTE: No real image generation here on purpose.
        # This is purely a planning / placeholder method.

        result = {
            "status": "not_implemented",
            "message": (
                "SD35Text2ImgPipeline.run was called, but real "
                "SD3.5 inference is not implemented yet (Phase 1 skeleton)."
            ),
            "job_folder": job_folder,
            "planned_output_image": planned_output_image,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "config": {
                "width": self.config.width,
                "height": self.config.height,
                "num_inference_steps": self.config.num_inference_steps,
                "guidance_scale": self.config.guidance_scale,
                "seed": self.config.seed,
                "style_preset": self.config.style_preset,
                "material_preset": self.config.material_preset,
                "lighting_preset": self.config.lighting_preset,
            },
            "meta_snapshot": meta,
        }

        return result
