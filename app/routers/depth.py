# app/routers/depth.py

import os
import uuid
import datetime
from typing import Optional, Dict, Any

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.core.safety import check_prompt_safety

router = APIRouter(
    prefix="/api/depth",
    tags=["Depth Maps (MiDaS skeleton)"],
)


def _create_job_folder(base_outputs_dir: str = "outputs") -> str:
    """
    Create a new timestamped job folder.

    Example: outputs/2025-11-23/<job_id>/
    """
    today = datetime.date.today().isoformat()
    job_id = uuid.uuid4().hex
    folder = os.path.join(base_outputs_dir, today, job_id)
    os.makedirs(folder, exist_ok=True)
    return folder


@router.post("/plan")
async def plan_depth_map(
    image: UploadFile = File(...),
    prompt: Optional[str] = Form(
        None,
        description="Optional description of what this depth will be used for "
                    "(VR, parallax, product insertion, etc.).",
    ),
    negative_prompt: Optional[str] = Form(
        "",
        description="Optional negative prompt (still checked by safety).",
    ),
):
    """
    Plan a depth-map generation job (MiDaS skeleton, NO inference).

    What it does:
    - (Optionally) runs safety check on prompt text
    - Saves the uploaded image as input.png
    - Writes meta.json with:
        * type = "depth-map"
        * model_name = "midas-large"
        * planned_output = "depth.png"
        * status = "planned"

    Later, the GPU runtime will:
    - read this meta.json
    - run MiDaS on the input image
    - write depth.png
    - update meta.json with status = "completed"
    """

    # 1) Optional safety check on text
    if prompt:
        is_safe, reason = check_prompt_safety(
            prompt=prompt,
            negative_prompt=negative_prompt,
        )
        if not is_safe:
            raise HTTPException(
                status_code=400,
                detail=f"Prompt rejected by safety filter: {reason}",
            )

    # 2) Create job folder
    job_folder = _create_job_folder(base_outputs_dir="outputs")
    job_id = os.path.basename(job_folder)

    # 3) Save uploaded image
    input_path = os.path.join(job_folder, "input.png")
    with open(input_path, "wb") as f:
        f.write(await image.read())

    # 4) Prepare meta information
    meta: Dict[str, Any] = {
        "job_id": job_id,
        "type": "depth-map",
        "model_name": "midas-large",
        "created_at": datetime.datetime.utcnow().isoformat(),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "input_image": "input.png",
        "planned_output": "depth.png",
        "status": "planned",
        "mode": "skeleton-no-inference",
    }

    # 5) Write meta.json
    meta_path = os.path.join(job_folder, "meta.json")
    import json

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    # 6) Return response
    return {
        "status": "ok",
        "message": "Depth-map job planned (skeleton, no AI run yet).",
        "job_folder": job_folder,
        "meta_path": meta_path,
        "input_saved_as": input_path,
        "planned_output_image": os.path.join(job_folder, "depth.png"),
    }
