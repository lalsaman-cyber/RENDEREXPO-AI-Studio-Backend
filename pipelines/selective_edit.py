"""
Backend/pipelines/selective_edit.py

Selective-Edit endpoint stub for RENDEREXPO AI STUDIO.
Currently uses IMG2IMG under the hood (CPU).
Future: real inpainting on GPU.

Behavior:
- Validates base_image_path and mask_image_path.
- Applies presets: style, lighting, camera, mood, furniture_style, material_key.
- Delegates to generate_sd3_img2img(...) with strength/cfg defaults.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from file_utils import validate_image_file  # type: ignore
from pipelines.presets import get_presets       # type: ignore
from pipelines.materials import MATERIAL_LIBRARY  # type: ignore
from pipelines.sd3_img2img import generate_sd3_img2img  # type: ignore

logger = logging.getLogger(__name__)

def _apply_presets_to_prompt(
    base_prompt: str,
    style: Optional[str],
    lighting: Optional[str],
    camera: Optional[str],
    mood: Optional[str],
    furniture_style: Optional[str],
    material_key: Optional[str],
) -> str:
    """
    Append prompt snippets from the presets library based on keys.
    """
    prompt = base_prompt
    presets = get_presets()

    # STYLE
    if style:
        for group in ["architecture_styles", "interior_styles", "landscape_site"]:
            group_dict = presets.get(group, {})
            if style in group_dict:
                snippet = group_dict[style].get("prompt_snippet")
                if snippet:
                    prompt += ", " + snippet
                break

    # LIGHTING
    if lighting:
        lighting_dict = presets.get("lighting", {})
        if lighting in lighting_dict:
            snippet = lighting_dict[lighting].get("prompt_snippet")
            if snippet:
                prompt += ", " + snippet

    # CAMERA
    if camera:
        camera_dict = presets.get("camera", {})
        if camera in camera_dict:
            snippet = camera_dict[camera].get("prompt_snippet")
            if snippet:
                prompt += ", " + snippet

    # MOOD
    if mood:
        mood_dict = presets.get("mood", {})
        if mood in mood_dict:
            snippet = mood_dict[mood].get("prompt_snippet")
            if snippet:
                prompt += ", " + snippet

    # FURNITURE STYLE
    if furniture_style:
        furn_dict = presets.get("furniture_styles", {})
        if furniture_style in furn_dict:
            snippet = furn_dict[furniture_style].get("prompt_snippet")
            if snippet:
                prompt += ", " + snippet

    # MATERIAL
    if material_key:
        mat = MATERIAL_LIBRARY.get(material_key)
        if mat:
            snippet = mat.get("prompt_snippet")
            if snippet:
                prompt += ", " + snippet

    return prompt


def selective_edit(
    base_image_path: str,
    mask_image_path: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
    num_inference_steps: Optional[int] = 20,
    cfg_scale: Optional[float] = 7.0,
    style: Optional[str] = None,
    lighting: Optional[str] = None,
    camera: Optional[str] = None,
    mood: Optional[str] = None,
    furniture_style: Optional[str] = None,
    material_key: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """
    Entry point for selective-edit logic.
    """
    logger.info("Selective-Edit stub: base=%s, mask=%s, prompt=%r",
                base_image_path, mask_image_path, prompt)

    # Validate inputs (just files for now)
    validate_image_file(base_image_path)
    validate_image_file(mask_image_path)

    final_prompt = _apply_presets_to_prompt(
        base_prompt=prompt,
        style=style,
        lighting=lighting,
        camera=camera,
        mood=mood,
        furniture_style=furniture_style,
        material_key=material_key,
    )

    result = generate_sd3_img2img(
        prompt=final_prompt,
        init_image_path=base_image_path,
        negative_prompt=negative_prompt,
        strength=cfg_scale,  # mapping cfg_scale to strength for now
        width=None,
        height=None,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
        model_id="sdxl-base",
        **extra,
    )

    return {
        "status": "ok",
        "message": "Selective-Edit stub completed.",
        "result": result,
    }
