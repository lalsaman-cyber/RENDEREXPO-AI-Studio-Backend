"""
app/routers/insert_object.py

SKELETON PRODUCT INSERTION ENDPOINT (NO AI, NO GPU)

What this does:
- Accepts:
    * product_image (required) — couch, lamp, table, etc.
    * scene_image (optional) — existing room / space photo or render
    * floorplan_image (optional) — if user wants floorplan-aware placement later
    * prompt (optional) — text description of style / placement
    * placement_hint (optional) — e.g. "against back wall", "near window"
- Creates outputs/{YYYY-MM-DD}/{job_id}/
- Saves uploaded images to disk
- Writes meta.json describing a future product insertion pipeline

What this DOES NOT do:
- No segmentation (no SAM, no NIM)
- No depth, no perspective estimation
- No SD3.5 generation
- No lighting matching

All of that will be implemented later in the GPU runtime.
"""

import os
import uuid
import datetime
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

router = APIRouter(
    prefix="/api/insert-object",
    tags=["Product Insertion (Planning Only)"],
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
async def plan_insert_object(
    product_image: UploadFile = File(
        ...,
        description="Image of the product to insert (couch, chair, lamp, table, etc.)",
    ),
    scene_image: Optional[UploadFile] = File(
        None,
        description="Optional existing room / scene image where the product will be placed.",
    ),
    floorplan_image: Optional[UploadFile] = File(
        None,
        description="Optional floor plan image if we want floorplan-aware placement later.",
    ),
    prompt: Optional[str] = Form(
        None,
        description="Optional style / design prompt (e.g. 'soft luxury living room with neutral tones').",
    ),
    placement_hint: Optional[str] = Form(
        None,
        description="Optional placement hint (e.g. 'against the back wall under the windows').",
    ),
):
    """
    Plan a product insertion job (NO AI, NO GPU).

    For now:
    - Require at least a product_image.
    - Scene and floorplan are optional.
    - We only:
        * create a job folder
        * save the uploaded images
        * write a meta.json describing what a future GPU pipeline will do
    """

    if product_image is None:
        raise HTTPException(
            status_code=400,
            detail="product_image is required for product insertion planning.",
        )

    # 1) Create job folder
    job_folder = _create_job_folder()
    job_id = os.path.basename(job_folder)

    saved_files = {}

    # 2) Save product image as product.png
    product_path = os.path.join(job_folder, "product.png")
    with open(product_path, "wb") as f:
        f.write(await product_image.read())
    saved_files["product_image"] = "product.png"

    # 3) Save scene image if provided
    if scene_image is not None:
        scene_path = os.path.join(job_folder, "scene.png")
        with open(scene_path, "wb") as f:
            f.write(await scene_image.read())
        saved_files["scene_image"] = "scene.png"

    # 4) Save floorplan image if provided
    if floorplan_image is not None:
        floorplan_path = os.path.join(job_folder, "floorplan.png")
        with open(floorplan_path, "wb") as f:
            f.write(await floorplan_image.read())
        saved_files["floorplan_image"] = "floorplan.png"

    # 5) Build meta.json describing a FUTURE insertion pipeline
    meta = {
        "job_id": job_id,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "type": "insert_object",
        "mode_runtime": "skeleton-no-inference",
        "status": "planned",
        "files": saved_files,
        "prompt": prompt,
        "placement_hint": placement_hint,
        "planned_outputs": {
            "composited_image": "output.png",           # future composite
            "segmentation_mask": "product_mask.png",    # future SAM/NIM mask
            "depth_map_scene": "scene_depth.png",       # future MiDaS depth
        },
        "gpu_planned_pipeline": [
            {
                "stage": "segment_product",
                "description": "Use SAM / NIM to segment the product from product_image (future GPU step).",
                "inputs": [saved_files["product_image"]],
                "outputs": ["product_mask.png", "product_rgba.png"],
            },
            {
                "stage": "estimate_depth_and_pose",
                "description": "Estimate depth / orientation for scene and product for correct placement (future GPU step).",
                "inputs": [saved_files.get("scene_image"), saved_files["product_image"]],
                "outputs": ["scene_depth.png", "product_pose.json"],
            },
            {
                "stage": "compose_product_in_scene",
                "description": "Insert product into scene using SD3.5 img2img over masks (future GPU step).",
                "inputs": ["scene.png", "product_rgba.png", "product_mask.png"],
                "outputs": ["output.png"],
            },
            {
                "stage": "relight_match",
                "description": "Match lighting between product and scene (future SD3.5 refiner step).",
                "inputs": ["output.png"],
                "outputs": ["output.png"],
            },
        ],
    }

    meta_path = os.path.join(job_folder, "meta.json")

    import json
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    # 6) Return simple response
    return {
        "status": "ok",
        "message": "Product insertion job planned (skeleton, no AI run yet).",
        "job_folder": job_folder,
        "files_saved": saved_files,
        "meta_path": meta_path,
    }
