"""
app/routers/sd35.py

SD3.5 endpoint skeletons for RENDEREXPO AI STUDIO.

IMPORTANT:
- These endpoints DO NOT run any AI yet.
- They only:
  * validate inputs
  * create a job folder
  * save a meta.json file
  * (optionally) save the uploaded input image

Later, on GPU, we will plug real SD3.5 inference into these endpoints.
"""

import os
import uuid
import json
from datetime import datetime
from typing import Optional, Literal

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_job_folder(job_type: str) -> str:
    """
    Create a folder like:
        outputs/YYYY-MM-DD/JOBID/

    Returns the full path to that folder.
    """
    today = datetime.utcnow().strftime("%Y-%m-%d")
    job_id = uuid.uuid4().hex  # unique ID

    base_dir = os.path.join("outputs", today, job_id)
    os.makedirs(base_dir, exist_ok=True)

    # Also write a very small marker file so you can see it in Explorer
    marker_path = os.path.join(base_dir, "job_type.txt")
    try:
        with open(marker_path, "w", encoding="utf-8") as f:
            f.write(job_type)
    except Exception:
        # Not critical if this fails
        pass

    return base_dir


def _save_meta(base_dir: str, meta: dict) -> str:
    """
    Save a meta.json file in the job folder.
    """
    meta_path = os.path.join(base_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta_path


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SD35Text2ImgRequest(BaseModel):
    prompt: str = Field(..., description="Main prompt for SD3.5")
    negative_prompt: Optional[str] = Field(
        None, description="What you do NOT want to see"
    )

    width: int = Field(
        1024,
        ge=64,
        le=2048,
        description="Target image width (pixels). No AI is run yet.",
    )
    height: int = Field(
        1024,
        ge=64,
        le=2048,
        description="Target image height (pixels). No AI is run yet.",
    )

    num_inference_steps: int = Field(
        25,
        ge=1,
        le=150,
        description="Planned # of denoising steps (for later GPU inference).",
    )
    guidance_scale: float = Field(
        6.0,
        ge=0.0,
        le=20.0,
        description="How strongly the model follows the prompt (later).",
    )

    # Presets (we will wire these to your RENDEREXPO style system later)
    style_preset: Optional[str] = Field(
        None, description="e.g. 'scandinavian_minimal', 'soft_luxury', etc."
    )
    material_preset: Optional[str] = Field(
        None, description="e.g. 'warm_oak_veneer', 'brushed_brass', etc."
    )
    lighting_preset: Optional[str] = Field(
        None, description="e.g. 'daylight_soft', 'evening_warm', etc."
    )

    seed: Optional[int] = Field(
        None,
        description="Optional seed. If None, a random seed will be used later.",
    )


# ---------------------------------------------------------------------------
# Routes: Text-to-Image
# ---------------------------------------------------------------------------

@router.post("/render", tags=["sd35"])
async def sd35_text2img(request: SD35Text2ImgRequest):
    """
    SD3.5 Text-to-Image endpoint (SKELETON).

    Right now:
    - NO SD3.5 model is loaded.
    - NO GPU is used.
    - We only:
        * create a job folder
        * save a meta.json file with your settings
        * return a fake 'output_path' that we will fill later.
    """
    base_dir = _create_job_folder(job_type="sd35_text2img")

    # In the future, seed will control randomness. For now, if None, we pick one.
    seed = request.seed if request.seed is not None else int(uuid.uuid4().int % 1_000_000_000)

    meta = {
        "job_type": "sd35_text2img",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "width": request.width,
        "height": request.height,
        "num_inference_steps": request.num_inference_steps,
        "guidance_scale": request.guidance_scale,
        "style_preset": request.style_preset,
        "material_preset": request.material_preset,
        "lighting_preset": request.lighting_preset,
        "seed": seed,
        "status": "pending_inference",  # later: "completed", "failed", etc.
        "model_name": "sd3.5-large",
        "note": "This is a skeleton job. No image was generated yet.",
    }

    meta_path = _save_meta(base_dir, meta)

    # We define where the final PNG *will* go later on GPU
    output_png = os.path.join(base_dir, "output.png")

    return {
        "status": "ok",
        "message": "Text2Img job created (skeleton, no AI run yet).",
        "job_folder": base_dir,
        "meta_path": meta_path,
        "planned_output_image": output_png,
    }


# ---------------------------------------------------------------------------
# Routes: Image-to-Image
# ---------------------------------------------------------------------------

@router.post("/render-from-image", tags=["sd35"])
async def sd35_img2img(
    image: UploadFile = File(..., description="Base image / sketch / clay render"),
    prompt: str = Form(..., description="Main prompt for SD3.5"),
    negative_prompt: Optional[str] = Form(
        None, description="What you do NOT want to see"
    ),
    strength: float = Form(
        0.7,
        ge=0.0,
        le=1.0,
        description=(
            "How strongly SD3.5 should override the input image (later, on GPU)."
        ),
    ),
    control_type: Optional[Literal["canny", "depth", "lineart"]] = Form(
        None,
        description="Planned ControlNet type. Not active yet.",
    ),
    control_strength: float = Form(
        1.0,
        ge=0.0,
        le=2.0,
        description="Planned ControlNet strength. Not active yet.",
    ),
):
    """
    SD3.5 Image-to-Image endpoint (SKELETON).

    Right now:
    - We accept an image upload + form fields.
    - We create a job folder.
    - We save the input image into that folder.
    - We write a meta.json file.
    - NO SD3.5 model is loaded.
    - NO GPU is used.
    """
    # Basic check for image filename
    if not image.filename:
        raise HTTPException(status_code=400, detail="Uploaded image has no filename.")

    base_dir = _create_job_folder(job_type="sd35_img2img")

    # Save the uploaded image as input.png in the job folder
    input_path = os.path.join(base_dir, "input.png")
    try:
        with open(input_path, "wb") as f:
            f.write(await image.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded image: {e}")

    seed = int(uuid.uuid4().int % 1_000_000_000)

    meta = {
        "job_type": "sd35_img2img",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "strength": strength,
        "control_type": control_type,
        "control_strength": control_strength,
        "seed": seed,
        "status": "pending_inference",
        "model_name": "sd3.5-large",
        "input_image": input_path,
        "note": "Skeleton job. No img2img inference has been run yet.",
    }

    meta_path = _save_meta(base_dir, meta)
    output_png = os.path.join(base_dir, "output.png")

    return {
        "status": "ok",
        "message": "Img2Img job created (skeleton, no AI run yet).",
        "job_folder": base_dir,
        "input_image": input_path,
        "meta_path": meta_path,
        "planned_output_image": output_png,
    }
