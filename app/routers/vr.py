"""
app/routers/vr.py

SKELETON VR RECONSTRUCTION ENDPOINT (NO AI, NO GPU)

What this does:
- Accepts 3 or more images (front/left/right, or any combo)
- Optional prompt (description of the space / scene)
- Optional plan_hint (e.g. "loft living room", "gallery")
- Creates outputs/{YYYY-MM-DD}/{job_id}/
- Saves the uploaded images as view_1.png, view_2.png, ...
- Writes meta.json describing a future VR reconstruction job

What this DOES NOT do:
- No MiDaS depth
- No 3D point cloud
- No WebGL / Three.js generation
- No SD3.5 inference

All of that will be implemented later in the GPU runtime.
"""

import os
import uuid
import datetime
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

router = APIRouter(
    prefix="/api/vr",
    tags=["VR (Planning Only)"],
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


@router.post("/reconstruct/plan")
async def plan_vr_reconstruction(
    images: List[UploadFile] = File(..., description="At least 3 images of the same space."),
    prompt: Optional[str] = Form(
        None,
        description="Optional description / prompt for the VR scene (e.g. 'modern living room, neutral tones').",
    ),
    plan_hint: Optional[str] = Form(
        None,
        description="Optional hint about the plan or usage (e.g. 'loft layout', 'gallery space').",
    ),
):
    """
    Plan a VR reconstruction job from multiple images.

    RULES (for now):
    - Must provide at LEAST 3 images.
    - All logic is SKELETON ONLY (no depth, no 3D, no SD3.5).
    - We simply:
        * create a job folder
        * save the images as view_1.png, view_2.png, ...
        * write meta.json describing what a future GPU pipeline will do
    """

    # Require at least 3 images for a meaningful reconstruction
    if len(images) < 3:
        raise HTTPException(
            status_code=400,
            detail="You must upload at least 3 images for VR reconstruction planning.",
        )

    # 1) Create job folder
    job_folder = _create_job_folder()
    job_id = os.path.basename(job_folder)

    # 2) Save uploaded images as view_1.png, view_2.png, ...
    saved_views = []
    for idx, upload in enumerate(images, start=1):
        view_name = f"view_{idx}.png"
        view_path = os.path.join(job_folder, view_name)
        # Read file bytes and write to disk
        with open(view_path, "wb") as f:
            f.write(await upload.read())

        saved_views.append(view_name)

    # 3) Build meta.json describing a FUTURE VR pipeline
    meta = {
        "job_id": job_id,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "type": "vr_reconstruct",
        "mode_runtime": "skeleton-no-inference",
        "status": "planned",
        "input_views": saved_views,
        "prompt": prompt,
        "plan_hint": plan_hint,
        "planned_outputs": {
            "point_cloud": "point_cloud.ply",   # future artifact
            "mesh": "scene_mesh.glb",           # future artifact
            "textures": "textures/",            # future folder
            "vr_scene_config": "vr_scene.json", # future WebGL config
        },
        "gpu_planned_pipeline": [
            {
                "stage": "depth_estimation",
                "description": "Run MiDaS depth on each input view (future GPU step).",
                "inputs": saved_views,
                "outputs": ["depth_" + name for name in saved_views],
            },
            {
                "stage": "point_cloud_fusion",
                "description": "Fuse depth maps into a unified 3D point cloud (future GPU step).",
                "inputs": ["depth_" + name for name in saved_views],
                "outputs": ["point_cloud.ply"],
            },
            {
                "stage": "mesh_reconstruction",
                "description": "Convert point cloud into a navigable mesh (future GPU step).",
                "inputs": ["point_cloud.ply"],
                "outputs": ["scene_mesh.glb"],
            },
            {
                "stage": "texture_baking",
                "description": "Use SD3.5 + ControlNet to bake photorealistic textures (future GPU step).",
                "inputs": saved_views,
                "outputs": ["textures/"],
            },
            {
                "stage": "vr_scene_setup",
                "description": "Create Three.js / WebXR config for browser VR (future GPU/CPU step).",
                "inputs": ["scene_mesh.glb", "textures/"],
                "outputs": ["vr_scene.json"],
            },
        ],
    }

    meta_path = os.path.join(job_folder, "meta.json")

    import json
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    # 4) Return simple response
    return {
        "status": "ok",
        "message": "VR reconstruction job planned (skeleton, no AI run yet).",
        "job_folder": job_folder,
        "views_saved": saved_views,
        "meta_path": meta_path,
    }
