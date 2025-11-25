# modules/sd35_img2img/__init__.py
"""
SD3.5 Image-to-Image pipeline skeleton for RENDEREXPO AI STUDIO.

IMPORTANT (Phase 1):
- This defines the *shape* of the img2img pipeline.
- No torch, no diffusers, no model loading.
- Safe on CPU-only laptop.

Later (Phase 3, on GPU / RunPod):
- We will implement the real SD3.5 img2img logic here.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class SD35Img2ImgConfig:
    """
    Basic configuration for an SD3.5 img2img run.
    """
    strength: float = 0.7
    num_inference_steps: int = 25
    guidance_scale: float = 6.0
    seed: Optional[int] = None
    style_preset: Optional[str] = None
    material_preset: Optional[str] = None
    lighting_preset: Optional[str] = None


class SD35Img2ImgPipeline:
    """
    Skeleton Image-to-Image pipeline for SD3.5.

    Phase 1:
    - Placeholder interfaces only.
    - No heavy AI work.

    Phase 3:
    - Will use SD35Runtime (GPU) + ControlNet (optional) +
      real SD3.5 img2img logic.
    """

    def __init__(self, config: Optional[SD35Img2ImgConfig] = None):
        self.config = config or SD35Img2ImgConfig()

    def run(
        self,
        prompt: str,
        negative_prompt: Optional[str],
        input_image_path: str,
        job_folder: str,
        planned_output_image: str,
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run a single img2img job.

        Phase 1:
        - No real image generation.
        - Returns a dict describing the *planned* run.

        Phase 3:
        - Will:
            * load input image
            * run SD3.5 img2img on GPU
            * save output PNG
            * update meta and return it
        """

        result = {
            "status": "not_implemented",
            "message": (
                "SD35Img2ImgPipeline.run was called, but real "
                "SD3.5 img2img inference is not implemented yet (Phase 1 skeleton)."
            ),
            "job_folder": job_folder,
            "input_image": input_image_path,
            "planned_output_image": planned_output_image,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "config": {
                "strength": self.config.strength,
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
