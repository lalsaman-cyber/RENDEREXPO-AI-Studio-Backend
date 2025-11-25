# app/routers/controlnet.py

import os
import uuid
import datetime
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

# This router is for PLANNING ControlNet-based jobs, not running them.
router = APIRouter(
    prefix="/api/controlnet",
    tags=["ControlNet Planning (NO GPU)"],
)


ALLOWED_CONTROL_TYPES = [
    "canny",
    "depth",
    "lineart",
    "sketch",
    "floorplan",
    "normal",
]


def _create_job_folder() -> str:
    """
    Create a new timestamped job folder under outputs/{YYYY-MM-DD}/{job_id}/.
    """
    today = datetime.date.today().isoformat()
    job_id = uuid.uuid4().hex
    folder = os.path.join("outputs", today, job_id)
    os.makedirs(folder, exist_ok=True)
    return folder


@router.post("/plan")
async def plan_controlnet_job(
    image: UploadFile = File(..., description="Sketch, floor plan, edge map, or line-art image."),
    control_type: str = Form(..., description="Type of control: canny, depth, lineart, sketch, floorplan, normal."),
    prompt: str = Form(..., description="Main text prompt for SD3.5 guided by this control."),
    negative_prompt: Optional[str] = Form(
        "",
        description="Optional negative prompt.",
    ),
    control_strength: float = Form(
        1.0,
        description="How strongly ControlNet should influence the result (0.0–2.0 in future).",
    ),
    width: int = Form(
        1024,
        description="Planned output width (pixels).",
    ),
    height: int = Form(
        1024,
        description="Planned output height (pixels).",
    ),
    num_inference_steps: int = Form(
        25,
        description="Planned number of inference steps.",
    ),
    guidance_scale: float = Form(
        6.0,
        description="Planned CFG guidance scale.",
    ),
    seed: Optional[int] = Form(
        None,
        description="Random seed (optional).",
    ),
):
    """
    PLAN a ControlNet job (NO real inference, NO SD3.5 execution).

    What this does:
    - Validates control_type and basic parameters.
    - Creates a job folder under outputs/{YYYY-MM-DD}/{job_id}/.
    - Saves the uploaded control image as input.png.
    - Writes meta.json with all planned settings and:
        * type: "controlnet"
        * mode: "skeleton-no-inference"

    Later, the GPU runtime will:
    - Read meta.json
    - Run SD3.5 + ControlNet
    - Save the actual output.png
    """

    # 1) Validate control_type
    if control_type not in ALLOWED_CONTROL_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid control_type '{control_type}'. "
                   f"Allowed: {', '.join(ALLOWED_CONTROL_TYPES)}",
        )

    # 2) Validate control_strength (soft range for now)
    if not (0.0 <= control_strength <= 2.0):
        raise HTTPException(
            status_code=400,
            detail="control_strength must be between 0.0 and 2.0",
        )

    # 3) Create job folder
    job_folder = _create_job_folder()
    job_id = os.path.basename(job_folder)

    # 4) Save uploaded image as input.png
    input_path = os.path.join(job_folder, "input.png")
    with open(input_path, "wb") as f:
        f.write(await image.read())

    # 5) Build meta data
    meta = {
        "job_id": job_id,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "type": "controlnet",
        "model_name": "sd3.5-large",
        "mode": "skeleton-no-inference",
        "control_type": control_type,
        "control_strength": control_strength,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "input_image": "input.png",
        "planned_output": "output.png",
        "status": "planned",
    }

    # 6) Write meta.json
    meta_path = os.path.join(job_folder, "meta.json")
    import json

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    # 7) Return response
    return {
        "status": "ok",
        "message": "ControlNet job planned (skeleton, no AI run yet).",
        "job_folder": job_folder,
        "input_saved_as": input_path,
        "meta_path": meta_path,
        "planned_output_image": os.path.join(job_folder, "output.png"),
    }
