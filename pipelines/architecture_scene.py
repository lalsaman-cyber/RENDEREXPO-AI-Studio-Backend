"""
High-level architecture scene generator.

This module builds a rich, architecture-focused prompt from a clean JSON,
then delegates to the existing SDXL txt2img pipeline.

Used by /v1/architecture/scene in main.py.
"""

from typing import Optional, List, Dict, Any
import logging

from pipelines.sd3_text2img import generate_sd3_text2img

# Optional: try to use style / lighting presets if they exist.
try:
    from pipelines.presets import STYLE_PRESETS, LIGHTING_PRESETS  # type: ignore
except Exception:  # pragma: no cover - safe fallback
    STYLE_PRESETS = {}
    LIGHTING_PRESETS = {}

logger = logging.getLogger(__name__)


def _lookup_style(style_key: Optional[str]) -> str:
    """
    Turn a style key into a nice snippet using STYLE_PRESETS if available.
    """
    if not style_key:
        return ""
    preset = STYLE_PRESETS.get(style_key, {})
    base = preset.get("prompt") or preset.get("label") or style_key
    return f"architecture style: {base}"


def _lookup_lighting(lighting_key: Optional[str]) -> str:
    """
    Turn a lighting key into a nice snippet using LIGHTING_PRESETS if available.
    """
    if not lighting_key:
        return ""
    preset = LIGHTING_PRESETS.get(lighting_key, {})
    base = preset.get("prompt") or preset.get("label") or lighting_key
    return f"lighting setup: {base}"


def build_architecture_prompt(
    description: str,
    project_type: Optional[str] = None,
    room_type: Optional[str] = None,
    style: Optional[str] = None,
    lighting: Optional[str] = None,
    camera: Optional[str] = None,
    mood: Optional[str] = None,
    material_tags: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Build the main prompt + a sensible negative prompt for architecture images.
    """

    parts: List[str] = []

    # Core subject
    if project_type:
        parts.append(f"{project_type}")
    if room_type:
        parts.append(f"{room_type}")
    if description:
        parts.append(description)

    # Style & lighting (using presets if available)
    style_snippet = _lookup_style(style)
    if style_snippet:
        parts.append(style_snippet)

    lighting_snippet = _lookup_lighting(lighting)
    if lighting_snippet:
        parts.append(lighting_snippet)

    # Camera, mood, materials as free-form helpers
    if camera:
        parts.append(f"camera: {camera}")
    if mood:
        parts.append(f"mood: {mood}")
    if material_tags:
        mat_text = ", ".join(material_tags)
        parts.append(f"materials emphasis: {mat_text}")

    # Always push the quality up: this is RENDEREXPO
    parts.append(
        "ultra photorealistic, 4k, physically based lighting, "
        "architectural visualization, clean composition, no text"
    )

    full_prompt = ", ".join([p for p in parts if p])

    # A default, safe negative prompt. Can be overridden by caller.
    negative_prompt = (
        "low quality, blurry, lowres, distortion, cartoon, extra limbs, "
        "overexposed, underexposed, noise, watermark, text, logo, frame"
    )

    return {
        "prompt": full_prompt,
        "negative_prompt": negative_prompt,
    }


def generate_architecture_scene(
    *,
    description: str,
    project_type: Optional[str] = None,
    room_type: Optional[str] = None,
    style: Optional[str] = None,
    lighting: Optional[str] = None,
    camera: Optional[str] = None,
    mood: Optional[str] = None,
    material_tags: Optional[List[str]] = None,
    negative_prompt: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    num_inference_steps: Optional[int] = None,
    cfg_scale: Optional[float] = None,
    model_id: str = "sdxl-base",
) -> Dict[str, Any]:
    """
    High-level wrapper:
      1) Build prompt from architecture parameters.
      2) Call generate_sd3_text2img (SDXL) under the hood.
      3) Return a unified dict for the API.
    """
    logger.info(
        "generate_architecture_scene called: project_type=%s, room_type=%s, "
        "style=%s, lighting=%s, camera=%s, mood=%s, material_tags=%s",
        project_type,
        room_type,
        style,
        lighting,
        camera,
        mood,
        material_tags,
    )

    prompt_bundle = build_architecture_prompt(
        description=description,
        project_type=project_type,
        room_type=room_type,
        style=style,
        lighting=lighting,
        camera=camera,
        mood=mood,
        material_tags=material_tags,
    )

    final_prompt = prompt_bundle["prompt"]
    default_negative = prompt_bundle["negative_prompt"]

    negative_to_use = negative_prompt or default_negative

    # Delegate to existing SDXL txt2img helper.
    txt2img_result = generate_sd3_text2img(
        prompt=final_prompt,
        negative_prompt=negative_to_use,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
        model_id=model_id,
    )

    # generate_sd3_text2img already returns a dict with output path, duration, etc.
    # We wrap it with a slightly more friendly shape.
    return {
        "status": "ok",
        "message": "Architecture scene generated.",
        "prompt": final_prompt,
        "negative_prompt": negative_to_use,
        "engine": "SDXL-base (via RENDEREXPO architecture_scene)",
        "result": txt2img_result,
        "meta": {
            "project_type": project_type,
            "room_type": room_type,
            "style": style,
            "lighting": lighting,
            "camera": camera,
            "mood": mood,
            "material_tags": material_tags or [],
            "model_id": model_id,
        },
    }
