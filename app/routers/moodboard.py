"""
app/routers/moodboard.py

SKELETON MOODBOARD → SPACE PLANNING (NO AI, NO GPU)

What this does:
- Accepts:
    * moodboard_images (1..N) — finishes, furniture, references, etc.
    * prompt (optional) — text description of the desired space
    * floorplan_image (optional) — if the user wants a specific layout
- Creates outputs/{YYYY-MM-DD}/{job_id}/
- Saves uploaded images to disk
- Writes meta.json describing a future moodboard → space generation pipeline

What this DOES NOT do:
- No CLIP tagging
- No material / style embedding
- No SD3.5 generation
- No layout inference

All of that will be implemented later in the GPU runtime.
"""

import os
import uuid
import datetime
from typing import Optional, List

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

router = APIRouter(
    prefix="/api/moodboard",
    tags=["Moodboard (Planning Only)"],
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


@router.post("/plan")
async def plan_moodboard_space(
    moodboard_images: List[UploadFile] = File(
        ...,
        description=(
            "One or more moodboard images (materials, references, furniture, "
            "lighting, etc.)."
        ),
    ),
    prompt: Optional[str] = Form(
        None,
        description=(
            "Optional description of the desired space, e.g. "
            "'soft luxury living room with warm oak and brass accents'."
        ),
    ),
    floorplan_image: Optional[UploadFile] = File(
        None,
        description=(
            "Optional floor plan image, if you want the generated space to "
            "respect a specific layout."
        ),
    ),
):
    """
    Plan a Moodboard → Space generation job (NO AI, NO GPU).

    For now:
    - Require at least one moodboard image.
    - Floorplan is optional.
    - We only:
        * create a job folder
        * save the uploaded images
        * write a meta.json describing what a future GPU pipeline will do
    """

    if not moodboard_images or len(moodboard_images) == 0:
        raise HTTPException(
            status_code=400,
            detail="At least one moodboard image is required.",
        )

    # 1) Create job folder
    job_folder = _create_job_folder()
    job_id = os.path.basename(job_folder)

    saved_files = {
        "moodboard_images": [],
    }

    # 2) Save each moodboard image as moodboard_0.png, moodboard_1.png, ...
    for idx, upload in enumerate(moodboard_images):
        filename = f"moodboard_{idx}.png"
        full_path = os.path.join(job_folder, filename)
        with open(full_path, "wb") as f:
            f.write(await upload.read())
        saved_files["moodboard_images"].append(filename)

    # 3) Save floorplan image if provided
    if floorplan_image is not None:
        floorplan_filename = "floorplan.png"
        floorplan_path = os.path.join(job_folder, floorplan_filename)
        with open(floorplan_path, "wb") as f:
            f.write(await floorplan_image.read())
        saved_files["floorplan_image"] = floorplan_filename

    # 4) Build meta.json describing a FUTURE moodboard pipeline
    meta = {
        "job_id": job_id,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "type": "moodboard_space",
        "mode_runtime": "skeleton-no-inference",
        "status": "planned",
        "files": saved_files,
        "prompt": prompt,
        "planned_outputs": {
            "generated_space": "output.png",
            "palette_json": "palette.json",
            "style_embedding": "style_embedding.json",
        },
        "gpu_planned_pipeline": [
            {
                "stage": "analyze_moodboard",
                "description": (
                    "Extract colors, materials, patterns, and lighting cues from "
                    "moodboard images (future CLIP / feature extractor step)."
                ),
                "inputs": saved_files["moodboard_images"],
                "outputs": ["palette.json", "style_embedding.json"],
            },
            {
                "stage": "combine_with_prompt",
                "description": (
                    "Combine style embedding with text prompt into a final "
                    "conditioning vector for SD3.5 (future step)."
                ),
                "inputs": ["style_embedding.json", "palette.json"],
                "outputs": ["combined_style.json"],
            },
            {
                "stage": "generate_space",
                "description": (
                    "Generate space (interior/exterior) with SD3.5 using "
                    "combined style and optional floorplan (future GPU step)."
                ),
                "inputs": ["combined_style.json"] + (
                    ["floorplan.png"] if "floorplan_image" in saved_files else []
                ),
                "outputs": ["output.png"],
            },
        ],
    }

    meta_path = os.path.join(job_folder, "meta.json")

    import json
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    # 5) Return simple response
    return {
        "status": "ok",
        "message": "Moodboard → space job planned (skeleton, no AI run yet).",
        "job_folder": job_folder,
        "files_saved": saved_files,
        "meta_path": meta_path,
    }
