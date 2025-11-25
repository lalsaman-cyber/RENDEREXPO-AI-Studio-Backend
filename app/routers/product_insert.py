# app/routers/product_insert.py

"""
Product insertion skeleton router.

Goal:
- Upload a product image (e.g. couch, lamp, table).
- Optionally upload a room / floorplan image.
- Store prompts and placement hints.
- Save everything into outputs/{date}/{job_id}/meta.json.

IMPORTANT:
- No SD3.5 inference yet.
- No real segmentation / lighting match yet.
- Just job folders + metadata (skeleton).
"""

import os
import uuid
import datetime
import json
from typing import Dict, Any, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

router = APIRouter(
    prefix="/api/product-insert",
    tags=["Product Insertion"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_job_folder(base_outputs_dir: str = "outputs") -> str:
    """
    Create a new outputs/{YYYY-MM-DD}/{job_id}/ folder.
    """
    today = datetime.date.today().isoformat()
    job_id = uuid.uuid4().hex
    folder = os.path.join(base_outputs_dir, today, job_id)
    os.makedirs(folder, exist_ok=True)
    return folder


def _ensure_folder_exists(job_folder: str) -> None:
    if not os.path.isdir(job_folder):
        raise HTTPException(
            status_code=400,
            detail=f"job_folder does not exist: {job_folder}",
        )


def _meta_path(job_folder: str) -> str:
    return os.path.join(job_folder, "meta.json")


def _read_meta(job_folder: str) -> Dict[str, Any]:
    meta_file = _meta_path(job_folder)
    if not os.path.isfile(meta_file):
        return {}
    with open(meta_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_meta(job_folder: str, meta: Dict[str, Any]) -> None:
    meta_file = _meta_path(job_folder)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)


# ---------------------------------------------------------------------------
# 1) Upload product (and optional room) + basic prompts
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_product_and_room(
    product_image: UploadFile = File(..., description="Image of the product (e.g. couch, chair, lamp)."),
    room_image: Optional[UploadFile] = File(
        None,
        description="Optional base room / space image where the product will be inserted.",
    ),
    prompt: str = Form(
        ...,
        description="High-level description of the desired final scene (style, mood, etc.).",
    ),
    placement_prompt: Optional[str] = Form(
        None,
        description="Where and how to place the product in the room (e.g. 'against the back wall under the window').",
    ),
):
    """
    Start a product-insertion job.

    Skeleton behavior:
    - creates job folder
    - saves product image as 'product.png'
    - optionally saves room image as 'room.png'
    - writes meta.json with prompts and basic info
    """
    job_folder = _create_job_folder()
    _ensure_folder_exists(job_folder)

    # Save product image
    product_path = os.path.join(job_folder, "product.png")
    with open(product_path, "wb") as f:
        f.write(await product_image.read())

    room_path: Optional[str] = None
    if room_image is not None:
        room_path = os.path.join(job_folder, "room.png")
        with open(room_path, "wb") as f:
            f.write(await room_image.read())

    # Build meta
    meta = _read_meta(job_folder)
    meta.setdefault("job_id", os.path.basename(job_folder))
    meta["type"] = "product_insert"
    meta["created_at"] = datetime.datetime.utcnow().isoformat()
    meta["mode"] = "skeleton-no-inference"

    meta["product_image"] = "product.png"
    if room_path is not None:
        meta["room_image"] = "room.png"
    else:
        meta["room_image"] = None

    meta["prompt"] = prompt
    meta["placement_prompt"] = placement_prompt
    meta["planned_output"] = "product_insert_result.png"

    _write_meta(job_folder, meta)

    return {
        "status": "ok",
        "message": "Product insertion job created (skeleton, no AI yet).",
        "job_folder": job_folder,
        "product_image": product_path,
        "room_image": room_path,
        "meta_path": _meta_path(job_folder),
    }


# ---------------------------------------------------------------------------
# 2) Plan SD3.5 insertion pass (still skeleton)
# ---------------------------------------------------------------------------

@router.post("/plan")
async def plan_product_insertion(
    job_folder: str = Form(..., description="Job folder returned by /product-insert/upload"),
    width: int = Form(1024),
    height: int = Form(1024),
    num_inference_steps: int = Form(25),
):
    """
    Plan an SD3.5-based insertion of the product into the room.

    Skeleton behavior:
    - validates job_folder
    - adds a 'planned_insertion' block into meta.json
    - NO actual SD3.5 or segmentation yet
    """
    _ensure_folder_exists(job_folder)

    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="width and height must be positive")

    meta = _read_meta(job_folder)
    if meta.get("type") != "product_insert":
        raise HTTPException(
            status_code=400,
            detail="meta.json does not describe a product_insert job.",
        )

    planned_insertion = {
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "planned_output_image": meta.get("planned_output", "product_insert_result.png"),
        "created_at": datetime.datetime.utcnow().isoformat(),
    }

    meta["planned_insertion"] = planned_insertion
    meta["mode"] = "skeleton-no-inference"
    _write_meta(job_folder, meta)

    return {
        "status": "ok",
        "message": "Product insertion planned (skeleton, no AI yet).",
        "job_folder": job_folder,
        "planned_insertion": planned_insertion,
        "meta_path": _meta_path(job_folder),
    }
