# app/routers/product.py

"""
Product insertion planning endpoints (skeleton, NO AI yet).

Goal:
- Let users upload:
    * a base room image
    * a product image (couch, lamp, table, etc.)
- Describe where/how to place it with a prompt.
- We create:
    * outputs/{YYYY-MM-DD}/{job_id}/ folder
    * Save room + object images
    * meta.json with planned pipeline steps (no real inference yet)

Later on GPU:
- We'll segment the object (SAM / NIM / etc.)
- Estimate depth / scale / perspective
- Composite into the room
- Optionally re-render the full scene with SD3.5.
"""

import os
import uuid
import datetime
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

router = APIRouter(prefix="/api/product", tags=["Product Insertion"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_job_folder() -> str:
    """
    Create a new timestamped job folder:

        outputs/{YYYY-MM-DD}/{job_id}/
    """
    today = datetime.date.today().isoformat()
    job_id = uuid.uuid4().hex
    folder = os.path.join("outputs", today, job_id)
    os.makedirs(folder, exist_ok=True)
    return folder


def _save_upload(file: UploadFile, folder: str, target_name: str) -> str:
    """
    Save an uploaded file under the given folder with the target name.
    Returns the full path.
    """
    full_path = os.path.join(folder, target_name)
    # NOTE: in real code we may validate content-type etc.
    with open(full_path, "wb") as f:
        f.write(file.file.read())
    return full_path


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/plan-insert")
async def plan_product_insertion(
    room_image: UploadFile = File(..., description="Base room / space image."),
    object_image: UploadFile = File(..., description="Product image (couch, lamp, table, etc.)."),
    placement_prompt: str = Form(
        ...,
        description=(
            "Plain-language instruction like: "
            "'place the couch against the back wall facing the window, "
            "scale it realistically, and match the warm lighting.'"
        ),
    ),
    style_hint: Optional[str] = Form(
        None,
        description="Optional style hint (e.g. 'soft luxury living room', 'Scandinavian minimal').",
    ),
    mode: str = Form(
        "insert_and_rerender",
        description=(
            "insert_only = just composite object; "
            "insert_and_rerender = re-run SD3.5 over scene after insertion."
        ),
    ),
):
    """
    Skeleton endpoint to PLAN a product insertion job.

    What this does NOW (local skeleton):
    - Creates job folder under outputs/{date}/{job_id}/
    - Saves:
        room_image -> room.png
        object_image -> object.png
    - Writes meta.json with:
        * basic job info
        * user prompts
        * a 'planned_actions' list describing what *will* happen on GPU

    What this will do LATER (on GPU / RunPod):
    - Run object segmentation on object.png (SAM / NIM / etc.)
    - Extract object with transparency.
    - Estimate depth / scale / perspective from room.png.
    - Composite object into room.
    - Optionally call SD3.5 img2img to harmonize lighting/materials.
    """
    # Basic validation on mode
    if mode not in ("insert_only", "insert_and_rerender"):
        raise HTTPException(
            status_code=400,
            detail="mode must be 'insert_only' or 'insert_and_rerender'.",
        )

    # 1) Create job folder
    job_folder = _create_job_folder()
    job_id = os.path.basename(job_folder)

    # 2) Save uploads
    room_path = _save_upload(room_image, job_folder, "room.png")
    object_path = _save_upload(object_image, job_folder, "object.png")

    # 3) Build meta payload
    created_at = datetime.datetime.utcnow().isoformat()

    planned_actions: List[Dict[str, Any]] = [
        {
            "stage": "segment_object",
            "input": "object.png",
            "output": "object_mask.png",
            "description": "Segment the product from background with SAM / NIM (future, GPU only).",
        },
        {
            "stage": "extract_object_rgba",
            "inputs": ["object.png", "object_mask.png"],
            "output": "object_rgba.png",
            "description": "Cut out object with transparency (future, GPU only).",
        },
        {
            "stage": "estimate_room_depth",
            "input": "room.png",
            "output": "room_depth.png",
            "description": "Estimate depth of the room for realistic placement (future, MiDaS on GPU).",
        },
        {
            "stage": "compute_placement",
            "inputs": ["room.png", "room_depth.png", "object_rgba.png"],
            "output": "layout_plan.json",
            "description": "Compute approximate position/scale/orientation in 2.5D (future).",
        },
        {
            "stage": "composite_object",
            "inputs": ["room.png", "object_rgba.png", "layout_plan.json"],
            "output": "room_with_object_raw.png",
            "description": "Composite object into room with rough lighting match (future).",
        },
    ]

    if mode == "insert_and_rerender":
        planned_actions.append(
            {
                "stage": "sd35_rerender",
                "inputs": ["room_with_object_raw.png"],
                "output": "room_with_object_final.png",
                "description": (
                    "Run SD3.5 img2img to harmonize materials and lighting (future, GPU only)."
                ),
            }
        )

    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": created_at,
        "type": "product_insertion",
        "base_room_image": "room.png",
        "object_image": "object.png",
        "placement_prompt": placement_prompt,
        "style_hint": style_hint,
        "mode": mode,
        "status": "planned",
        "pipeline": "product_insertion_v1",
        "planned_actions": planned_actions,
        "mode_runtime": "skeleton-no-inference",
    }

    # 4) Write meta.json
    meta_path = os.path.join(job_folder, "meta.json")
    import json

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    # 5) Return skeleton response
    return {
        "status": "ok",
        "message": "Product insertion job planned (skeleton, no AI run yet).",
        "job_folder": job_folder,
        "files_saved": {
            "room_image": os.path.relpath(room_path),
            "object_image": os.path.relpath(object_path),
        },
        "meta_path": os.path.relpath(meta_path),
    }
