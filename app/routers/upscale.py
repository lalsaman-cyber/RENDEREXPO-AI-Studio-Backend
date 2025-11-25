"""
app/routers/upscale.py

Skeleton Upscale endpoint (NO ESRGAN, NO AI yet).

What it does:
- Accepts an uploaded image (room, render, etc.)
- Accepts scale (2 or 4) and mode ("realesrgan" for now)
- Creates a job folder under outputs/{YYYY-MM-DD}/{job_id}/
- Saves:
    - input.png
    - meta.json with all planned settings
- Returns basic info about the planned upscale job

What it does NOT do:
- It does NOT run Real-ESRGAN.
- It does NOT require a GPU.
- It is 100% safe to run on your laptop.
"""

import os
import uuid
import datetime
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

router = APIRouter(
    prefix="/api",
    tags=["Upscale (Planning Only)"],
)


def _create_job_folder(base_outputs_dir: str = "outputs") -> str:
    """
    Create outputs/{YYYY-MM-DD}/{job_id}/ and return its path.
    """
    today = datetime.date.today().isoformat()
    job_id = uuid.uuid4().hex
    folder = os.path.join(base_outputs_dir, today, job_id)
    os.makedirs(folder, exist_ok=True)
    return folder


@router.post("/upscale")
async def plan_upscale(
    image: UploadFile = File(...),
    scale: int = Form(2, description="Upscale factor, typically 2 or 4."),
    mode: str = Form(
        "realesrgan",
        description="Upscale mode. For now only 'realesrgan' is planned.",
    ),
    note: Optional[str] = Form(
        None,
        description="Optional note from user (e.g. 'make it extra sharp').",
    ),
):
    """
    Plan an upscale job.

    This is a SKELETON endpoint:
    - Saves the uploaded image as input.png in a job folder.
    - Writes meta.json with planned settings.
    - Does NOT run any actual ESRGAN or SD3.5 yet.
    """

    # Basic validation for scale
    if scale not in (2, 4):
        raise HTTPException(
            status_code=400,
            detail="scale must be 2 or 4 for now."
        )

    # Basic validation for mode (we can add more later)
    allowed_modes = {"realesrgan"}
    if mode not in allowed_modes:
        raise HTTPException(
            status_code=400,
            detail=f"mode must be one of: {', '.join(sorted(allowed_modes))}",
        )

    # 1) Create job folder
    job_folder = _create_job_folder()
    job_id = os.path.basename(job_folder)

    # 2) Save uploaded image as input.png
    input_path = os.path.join(job_folder, "input.png")
    with open(input_path, "wb") as f:
        f.write(await image.read())

    # 3) Build meta.json
    meta = {
        "job_id": job_id,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "type": "upscale",
        "model_name": "realesrgan-planned",  # future: actual ESRGAN model name
        "scale": scale,
        "mode": mode,
        "note": note,
        "input_image": "input.png",
        "planned_output_image": "upscaled.png",
        "status": "planned",
        "mode_runtime": "skeleton-no-inference",
    }

    meta_path = os.path.join(job_folder, "meta.json")

    import json
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    # 4) Return a simple response
    return {
        "status": "ok",
        "message": "Upscale job created (skeleton, no AI run yet).",
        "job_folder": job_folder,
        "input_saved_as": input_path,
        "meta_path": meta_path,
        "planned_output_image": os.path.join(job_folder, "upscaled.png"),
    }
