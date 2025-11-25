# app/routers/img2img.py

"""
SD3.5 Img2Img router (LOCAL SKELETON)

- No SD3.5 weights are loaded.
- No actual AI inference is run.
- We ONLY:
    * accept an uploaded image
    * validate LoRA + refiner profiles (if provided)
    * create outputs/{date}/{job_id}/
    * save input image
    * write meta.json with full planning data
"""

import os
import uuid
import datetime
from typing import Optional, Dict, Any

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.core.lora_registry import (
    get_lora_profile,
    get_refiner_profile,
)

router = APIRouter(prefix="/api/sd35", tags=["SD3.5 Img2Img"])


def _create_job_folder() -> str:
    """Create a new timestamped job folder."""
    today = datetime.date.today().isoformat()
    job_id = uuid.uuid4().hex
    folder = os.path.join("outputs", today, job_id)
    os.makedirs(folder, exist_ok=True)
    return folder


@router.post("/render-from-image")
async def render_from_image(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form(None),
    strength: float = Form(0.7),
    num_inference_steps: int = Form(25),
    guidance_scale: float = Form(6.0),
    style_preset: Optional[str] = Form(None),
    material_preset: Optional[str] = Form(None),
    lighting_preset: Optional[str] = Form(None),
    seed: Optional[int] = Form(None),
    lora_profile: Optional[str] = Form(
        None, description="Optional RENDEREXPO LoRA profile (e.g. 'interiors')."
    ),
    refiner_profile: Optional[str] = Form(
        None, description="Optional RENDEREXPO refiner profile (e.g. 'ultra_detail')."
    ),
):
    """
    Skeleton Img2Img endpoint (NO SD3.5 inference yet).

    - Saves uploaded image
    - Validates LoRA + refiner profiles (if provided)
    - Saves meta.json
    - Creates job folder
    """

    # Validate basic strength range
    if not (0.0 <= strength <= 1.0):
        raise HTTPException(status_code=400, detail="strength must be between 0.0 and 1.0")

    # 1) Validate LoRA profile (if provided)
    resolved_lora_profile: Optional[Dict[str, Any]] = None
    if lora_profile:
        resolved_lora_profile = get_lora_profile(lora_profile)
        if resolved_lora_profile is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown lora_profile: '{lora_profile}'. "
                       f"Check config/lora_profiles.json.",
            )

    # 2) Validate refiner profile (if provided)
    resolved_refiner_profile: Optional[Dict[str, Any]] = None
    if refiner_profile:
        resolved_refiner_profile = get_refiner_profile(refiner_profile)
        if resolved_refiner_profile is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown refiner_profile: '{refiner_profile}'. "
                       f"Check config/lora_profiles.json.",
            )

    # 3) Create folder
    job_folder = _create_job_folder()
    job_id = os.path.basename(job_folder)

    # 4) Save uploaded image
    input_path = os.path.join(job_folder, "input.png")
    with open(input_path, "wb") as f:
        f.write(await image.read())

    # 5) Prepare meta information
    meta = {
        "job_id": job_id,
        "type": "img2img",
        "model_name": "sd3.5-large",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "strength": strength,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "style_preset": style_preset,
        "material_preset": material_preset,
        "lighting_preset": lighting_preset,
        "seed": seed,
        "input_image": "input.png",
        "planned_output": "output.png",
        "created_at": datetime.datetime.utcnow().isoformat(),
        "mode": "skeleton-no-inference",
        "status": "planned",
        "lora_profile": lora_profile,
        "refiner_profile": refiner_profile,
        "lora_profile_resolved": resolved_lora_profile,
        "refiner_profile_resolved": resolved_refiner_profile,
    }

    # 6) Write meta.json
    meta_path = os.path.join(job_folder, "meta.json")
    import json
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    return {
        "status": "ok",
        "message": "Img2Img job created (skeleton, no AI run yet).",
        "job_folder": job_folder,
        "input_saved_as": input_path,
        "meta_path": meta_path,
        "planned_output_image": os.path.join(job_folder, "output.png"),
    }
