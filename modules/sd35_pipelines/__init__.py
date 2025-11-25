# modules/sd35_pipelines/__init__.py
"""
High-level SD3.5 pipeline manager for RENDEREXPO AI STUDIO.

Phase 1:
- Only provides a clean interface.
- Does NOT load SD3.5.
- Does NOT perform real inference.

Phase 3 (on GPU / RunPod):
- Will:
    * receive jobs from pipeline_manager / SD35Runtime
    * route them to SD35Text2ImgPipeline or SD35Img2ImgPipeline
    * run real SD3.5 inference
    * save images and update meta.json
"""

from typing import Dict, Any

from modules.sd35_text2img import SD35Text2ImgPipeline, SD35Text2ImgConfig
from modules.sd35_img2img import SD35Img2ImgPipeline, SD35Img2ImgConfig


class SD35PipelineManager:
    """
    Simple high-level wrapper that can run either:
    - text2img
    - img2img

    In Phase 3 we will also extend this to:
    - depth / ControlNet
    - product insertion
    - floor plan cameras
    - VR texture baking
    etc.
    """

    def __init__(self):
        # For now, we just create default configs and pipelines.
        # Later we can inject SD35Runtime / GPU models into them.
        self.text2img_pipeline = SD35Text2ImgPipeline(SD35Text2ImgConfig())
        self.img2img_pipeline = SD35Img2ImgPipeline(SD35Img2ImgConfig())

    def run_text2img(self, meta: Dict[str, Any], job_folder: str) -> Dict[str, Any]:
        """
        Run a text2img job using meta information.

        meta is expected to contain:
        - prompt
        - negative_prompt
        - planned_output_image
        and other fields you already store in meta.json.
        """
        prompt = meta.get("prompt", "")
        negative_prompt = meta.get("negative_prompt")
        planned_output = meta.get("planned_output_image", "output.png")

        return self.text2img_pipeline.run(
            prompt=prompt,
            negative_prompt=negative_prompt,
            job_folder=job_folder,
            planned_output_image=planned_output,
            meta=meta,
        )

    def run_img2img(self, meta: Dict[str, Any], job_folder: str) -> Dict[str, Any]:
        """
        Run an img2img job using meta information.

        meta is expected to contain:
        - prompt
        - negative_prompt
        - input_image (path relative to job_folder)
        - planned_output_image
        """
        prompt = meta.get("prompt", "")
        negative_prompt = meta.get("negative_prompt")
        planned_output = meta.get("planned_output_image", "output.png")
        input_image_rel = meta.get("input_image", "input.png")
        input_image_path = f"{job_folder}/{input_image_rel}"

        return self.img2img_pipeline.run(
            prompt=prompt,
            negative_prompt=negative_prompt,
            input_image_path=input_image_path,
            job_folder=job_folder,
            planned_output_image=planned_output,
            meta=meta,
        )
