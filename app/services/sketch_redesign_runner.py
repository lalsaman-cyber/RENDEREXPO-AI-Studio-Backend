from __future__ import annotations

"""
app/services/sketch_redesign_runner.py

RENDEREXPO AI STUDIO
MistoLine-based Sketch to Redesign runner

PURPOSE
-------
- Keep Sketch to Render untouched.
- Add a NEW parallel Sketch to Redesign execution helper.
- Reuse the same working MistoLine / ComfyUI service family.
- Use redesign-oriented prompt construction instead of strict render-mode prompting.

IMPORTANT
---------
This file does NOT replace the working Sketch to Render flow.
It is only for the new Sketch to Redesign mode.
"""

from typing import Any, Dict, Optional

from app.config.sketch_runtime import get_anyline_mistoline_config
from app.services.comfy_anyline_mistoline import (
    AnylineMistolineSketchService,
    SketchJobConfig,
)
from app.services.sketch_redesign_prompt import build_sketch_redesign_prompt_package


def _build_redesign_config() -> SketchJobConfig:
    """
    Redesign-only runtime looseners.

    IMPORTANT:
    - We intentionally do NOT modify get_anyline_mistoline_config().
    - Sketch to Render keeps using the locked shared runtime.
    - Sketch to Redesign gets a derived config with looser sketch anchoring.

    Why:
    - render is preservation-first
    - redesign should allow more reinterpretation of facade and structure
    """
    base = get_anyline_mistoline_config()

    return SketchJobConfig(
        comfy_url=base.comfy_url,
        sdxl_checkpoint_name=base.sdxl_checkpoint_name,
        controlnet_name=base.controlnet_name,
        sampler_name=base.sampler_name,
        scheduler=base.scheduler,
        steps=base.steps,
        cfg=base.cfg,
        denoise=0.84,
        control_strength=0.58,
        start_percent=0.0,
        end_percent=0.40,
        output_prefix=base.output_prefix,
        poll_timeout=base.poll_timeout,
    )


def run_anyline_mistoline_sketch_redesign(
    *,
    input_image_path: str,
    output_dir: str,
    style_preset: Optional[str] = None,
    materials_notes: Optional[str] = None,
    atmosphere_notes: Optional[str] = None,
    background_notes: Optional[str] = None,
    mood_notes: Optional[str] = None,
    style_notes: Optional[str] = None,
    aesthetic_notes: Optional[str] = None,
    negative_prompt_override: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run the NEW Sketch to Redesign mode using the same working Anyline + MistoLine service family.

    This mode is intentionally more creative than Sketch to Render:
    - same sketch input family
    - same Comfy/MistoLine backbone
    - redesign-oriented prompt package
    - still starts from the sketch
    - not guaranteed to preserve exact structure

    Returns the underlying service result, plus redesign prompt metadata.
    """
    prompt_package = build_sketch_redesign_prompt_package(
        style_preset=style_preset,
        materials_notes=materials_notes,
        atmosphere_notes=atmosphere_notes,
        background_notes=background_notes,
        mood_notes=mood_notes,
        style_notes=style_notes,
        aesthetic_notes=aesthetic_notes,
    )

    config = _build_redesign_config()
    service = AnylineMistolineSketchService(config=config)

    negative_prompt = (
        negative_prompt_override.strip()
        if isinstance(negative_prompt_override, str) and negative_prompt_override.strip()
        else prompt_package.negative_prompt
    )

    result = service.run(
        input_image_path=input_image_path,
        output_dir=output_dir,
        prompt=prompt_package.prompt,
        negative_prompt=negative_prompt,
        seed=seed,
    )

    if not isinstance(result, dict):
        result = {"status": "unknown", "raw_result": result}

    result["mode"] = "sketch_to_redesign"
    result["product_promise"] = prompt_package.product_promise
    result["warning_text"] = prompt_package.warning_text
    result["style_preset"] = prompt_package.style_preset
    result["prompt"] = prompt_package.prompt
    result["negative_prompt"] = negative_prompt
    result["allowed_client_fields"] = prompt_package.allowed_client_fields

    result["redesign_runtime"] = {
        "steps": config.steps,
        "cfg": config.cfg,
        "denoise": config.denoise,
        "control_strength": config.control_strength,
        "start_percent": config.start_percent,
        "end_percent": config.end_percent,
    }

    return result