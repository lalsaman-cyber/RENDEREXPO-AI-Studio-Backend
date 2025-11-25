# app/routers/floorplan.py

import os
import uuid
import datetime
import json
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/floorplan", tags=["Floorplan"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_job_folder(base_outputs_dir: str = "outputs") -> str:
    """
    Create a new outputs/{YYYY-MM-DD}/{job_id}/ folder.
    Used when you upload a new floorplan or plan a 3D pipeline.
    """
    today = datetime.date.today().isoformat()
    job_id = uuid.uuid4().hex
    folder = os.path.join(base_outputs_dir, today, job_id)
    os.makedirs(folder, exist_ok=True)
    return folder


def _ensure_folder_exists(job_folder: str) -> None:
    """Ensure the given job_folder exists, or raise 400."""
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
# 0) Pydantic model for prompt-based floorplan generator
# ---------------------------------------------------------------------------

class FloorplanGenerateRequest(BaseModel):
    """
    Skeleton schema for 'draw floor plan from prompt'.

    Later this will drive a real SD3.5 + layout model.
    Right now it only creates:
    - job folder
    - meta.json with planned outputs
    """
    prompt: str = Field(
        ...,
        description=(
            "High-level description, e.g. "
            "'3-bedroom apartment, open kitchen, corridor, 2 bathrooms, balcony'."
        ),
    )
    width: int = Field(
        1024,
        ge=128,
        le=4096,
        description="Planned width of the floorplan image (pixels).",
    )
    height: int = Field(
        1024,
        ge=128,
        le=4096,
        description="Planned height of the floorplan image (pixels).",
    )
    wall_thickness: float = Field(
        0.2,
        ge=0.05,
        le=1.0,
        description="Wall thickness in meters (conceptual planning value).",
    )
    num_bedrooms: Optional[int] = Field(
        default=None,
        ge=0,
        le=20,
        description="Optional hint for number of bedrooms.",
    )
    num_bathrooms: Optional[int] = Field(
        default=None,
        ge=0,
        le=20,
        description="Optional hint for number of bathrooms.",
    )
    include_kitchen: bool = Field(
        True,
        description="Include a kitchen in the layout.",
    )
    include_living_room: bool = Field(
        True,
        description="Include a living / lounge area.",
    )
    include_corridor: bool = Field(
        True,
        description="Include corridors / circulation space.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Extra constraints, e.g. 'U-shaped kitchen, master suite near balcony'.",
    )


# ---------------------------------------------------------------------------
# 1) Upload floorplan  (basic skeleton)
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_floorplan(image: UploadFile = File(...)):
    """
    Upload a floorplan image (PNG/JPG/etc.).

    Skeleton behavior:
    - creates a new job folder
    - saves the uploaded image as 'floorplan.png'
    - writes/updates meta.json

    This is a low-level endpoint: just upload + basic meta, no pipeline planning.
    """
    job_folder = _create_job_folder()
    _ensure_folder_exists(job_folder)

    # Save floorplan image
    floorplan_path = os.path.join(job_folder, "floorplan.png")
    with open(floorplan_path, "wb") as f:
        f.write(await image.read())

    # Build / update meta
    meta = _read_meta(job_folder)
    meta.setdefault("job_id", os.path.basename(job_folder))
    meta.setdefault("type", "floorplan")
    meta["created_at"] = datetime.datetime.utcnow().isoformat()
    meta["floorplan_image"] = "floorplan.png"
    meta.setdefault("cameras", [])
    meta.setdefault("planned_render", None)
    meta["mode"] = "skeleton-no-inference"

    _write_meta(job_folder, meta)

    return {
        "status": "ok",
        "message": "Floorplan uploaded (skeleton, no AI yet).",
        "job_folder": job_folder,
        "floorplan_image": floorplan_path,
        "meta_path": _meta_path(job_folder),
    }


# ---------------------------------------------------------------------------
# 2) Set camera inside the floorplan
# ---------------------------------------------------------------------------

@router.post("/set-camera")
async def set_camera(
    job_folder: str = Form(..., description="Job folder returned by /floorplan/upload"),
    camera_x: float = Form(..., description="Camera X in floorplan coordinates"),
    camera_y: float = Form(..., description="Camera Y in floorplan coordinates"),
    rotation_deg: float = Form(
        ...,
        description="Camera rotation in degrees (0 = facing right, etc.).",
    ),
):
    """
    Attach a virtual camera to a floorplan.

    Skeleton behavior:
    - checks that job_folder exists
    - appends a camera entry into meta['cameras']
    - no 3D math or SD3.5 yet
    """
    _ensure_folder_exists(job_folder)

    meta = _read_meta(job_folder)
    if "cameras" not in meta or not isinstance(meta["cameras"], list):
        meta["cameras"] = []

    camera_id = f"cam_{len(meta['cameras']) + 1}"

    camera_info = {
        "camera_id": camera_id,
        "x": camera_x,
        "y": camera_y,
        "rotation_deg": rotation_deg,
        "created_at": datetime.datetime.utcnow().isoformat(),
    }

    meta["cameras"].append(camera_info)
    _write_meta(job_folder, meta)

    return {
        "status": "ok",
        "message": "Camera added to floorplan (skeleton, no AI yet).",
        "job_folder": job_folder,
        "camera": camera_info,
        "meta_path": _meta_path(job_folder),
    }


# ---------------------------------------------------------------------------
# 3) Plan SD3.5 render from that floorplan + camera
# ---------------------------------------------------------------------------

@router.post("/plan-render")
async def plan_floorplan_render(
    job_folder: str = Form(..., description="Job folder returned by /floorplan/upload"),
    width: int = Form(1024),
    height: int = Form(1024),
    num_inference_steps: int = Form(25),
):
    """
    Plan a SD3.5 render from the floorplan + cameras.

    Skeleton behavior:
    - reads meta.json
    - sets meta['planned_render'] with basic SD3.5 parameters
    - does NOT run any SD3.5 yet
    """
    _ensure_folder_exists(job_folder)

    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="width and height must be positive")

    meta = _read_meta(job_folder)

    planned_render = {
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "planned_output_image": "floorplan_view.png",
        "created_at": datetime.datetime.utcnow().isoformat(),
    }

    meta["planned_render"] = planned_render
    meta["mode"] = "skeleton-no-inference"
    _write_meta(job_folder, meta)

    return {
        "status": "ok",
        "message": "Floorplan → SD3.5 render planned (skeleton, no AI yet).",
        "job_folder": job_folder,
        "planned_render": planned_render,
        "meta_path": _meta_path(job_folder),
    }


# ---------------------------------------------------------------------------
# 4) One-shot Floorplan → 3D Pipeline Planning
# ---------------------------------------------------------------------------

@router.post("/plan-3d")
async def plan_floorplan_to_3d(
    floorplan: UploadFile = File(..., description="Floorplan image (PNG/JPG/etc.)"),
    prompt: str = Form(
        ...,
        description=(
            "High-level description, e.g. "
            "'3-bedroom apartment, open living area, warm modern style'."
        ),
    ),
):
    """
    One-shot endpoint:

    - Upload a floorplan
    - Immediately plan a full 'floorplan → 3D' pipeline
    - NO SD3.5 or geometry work yet (skeleton-only)
    """
    # 1) Create job folder and save the floorplan
    job_folder = _create_job_folder()
    floorplan_path = os.path.join(job_folder, "floorplan.png")

    with open(floorplan_path, "wb") as f:
        f.write(await floorplan.read())

    # 2) Plan a conceptual pipeline we will later implement on GPU
    created = datetime.datetime.utcnow().isoformat()

    planned_actions: List[Dict[str, Any]] = [
        {
            "stage": "load_floorplan",
            "file": "floorplan.png",
            "description": "Load floorplan image for analysis.",
        },
        {
            "stage": "detect_layout",
            "description": (
                "Detect walls, rooms, doors, windows, and basic layout "
                "(future SD3.5 + vision model step)."
            ),
        },
        {
            "stage": "place_cameras",
            "description": (
                "Automatically place virtual cameras in key rooms for interior views."
            ),
        },
        {
            "stage": "generate_views",
            "description": (
                "Use SD3.5 + ControlNet to generate interior views "
                "from the cameras (future GPU-only step)."
            ),
        },
    ]

    meta: Dict[str, Any] = {
        "job_id": os.path.basename(job_folder),
        "created_at": created,
        "type": "floorplan_to_3d",
        "floorplan": "floorplan.png",
        "prompt": prompt,
        "status": "planned",
        "pipeline": "floorplan_to_3d_v1",
        "planned_actions": planned_actions,
        "mode_runtime": "skeleton-no-inference",
    }

    meta_path = os.path.join(job_folder, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    return {
        "status": "ok",
        "message": "Floorplan → 3D pipeline planned (skeleton, no AI).",
        "job_folder": job_folder,
        "floorplan_saved": os.path.relpath(floorplan_path),
        "meta_path": os.path.relpath(meta_path),
    }


# ---------------------------------------------------------------------------
# 5) Plan camera-based room view from a floorplan
# ---------------------------------------------------------------------------

@router.post("/plan-camera-view")
async def plan_camera_view(
    job_folder: str = Form(..., description="Job folder returned by /floorplan/upload"),
    camera_id: str = Form(..., description="camera_id from /floorplan/set-camera"),
    prompt: str = Form(
        ...,
        description="Prompt for the room view, e.g. 'soft luxury living room'.",
    ),
    width: int = Form(1024),
    height: int = Form(1024),
    steps: int = Form(25),
):
    """
    Plan a single camera-based room view from a floorplan.

    Skeleton behavior:
    - loads meta.json
    - finds the camera by camera_id
    - attaches a 'planned_view' entry to meta under that camera
    - does not run AI yet
    """
    _ensure_folder_exists(job_folder)

    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="width and height must be positive")

    meta = _read_meta(job_folder)
    cameras = meta.get("cameras", [])
    if not isinstance(cameras, list) or not cameras:
        raise HTTPException(
            status_code=400,
            detail="No cameras found in meta.json. Use /floorplan/set-camera first.",
        )

    # Find the requested camera
    target_camera: Optional[Dict[str, Any]] = None
    for cam in cameras:
        if cam.get("camera_id") == camera_id:
            target_camera = cam
            break

    if not target_camera:
        raise HTTPException(
            status_code=404,
            detail=f"camera_id '{camera_id}' not found in this job.",
        )

    planned_view = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "planned_output": f"{camera_id}_view.png",
        "created_at": datetime.datetime.utcnow().isoformat(),
    }

    target_camera["planned_view"] = planned_view
    meta["cameras"] = cameras
    _write_meta(job_folder, meta)

    return {
        "status": "ok",
        "message": "Camera-based room view planned (skeleton).",
        "job_folder": job_folder,
        "camera_id": camera_id,
        "planned_view": planned_view,
        "meta_path": _meta_path(job_folder),
    }


# ---------------------------------------------------------------------------
# 6) NEW – Generate floorplan from prompt (no upload)
# ---------------------------------------------------------------------------

@router.post("/generate-from-prompt")
async def generate_floorplan_from_prompt(request: FloorplanGenerateRequest):
    """
    Prompt-based floorplan generator (skeleton).

    This is the backend for:

        "draw me a 3-bedroom floor with kitchen, corridors, wall thickness 0.2m"

    CURRENT BEHAVIOR (no AI yet):
    - creates outputs/{DATE}/{job_id}/
    - writes meta.json with:
        * type: 'floorplan_generate'
        * your prompt + constraints
        * planned_output_files:
            - 'floorplan_generated.png'
            - 'floorplan_layout.json'
        * planned_actions describing how the pipeline will work later
    """
    # 1) Create job folder
    job_folder = _create_job_folder()

    created = datetime.datetime.utcnow().isoformat()

    planned_output_image = "floorplan_generated.png"
    planned_output_layout = "floorplan_layout.json"

    planned_actions: List[Dict[str, Any]] = [
        {
            "stage": "parse_prompt",
            "description": (
                "Parse natural language prompt into structured layout constraints "
                "(rooms, adjacency, sizes)."
            ),
        },
        {
            "stage": "layout_synthesis",
            "description": (
                "Synthesize a 2D layout grid / graph from constraints "
                "(future SD3.5 + layout model step)."
            ),
        },
        {
            "stage": "geometry_sampling",
            "description": (
                "Convert abstract layout into precise wall segments with "
                f"wall thickness ≈ {request.wall_thickness} m."
            ),
        },
        {
            "stage": "export_raster",
            "target": planned_output_image,
            "description": (
                "Render a clean 2D floorplan image (PNG) from geometry "
                "(future vector → raster step)."
            ),
        },
        {
            "stage": "export_vector",
            "target": planned_output_layout,
            "description": (
                "Export layout as JSON (and later DXF/SVG) for editing and cameras."
            ),
        },
    ]

    meta: Dict[str, Any] = {
        "job_id": os.path.basename(job_folder),
        "created_at": created,
        "type": "floorplan_generate",
        "generator": "floorplan_prompt_v1",
        "prompt": request.prompt,
        "width": request.width,
        "height": request.height,
        "wall_thickness": request.wall_thickness,
        "num_bedrooms": request.num_bedrooms,
        "num_bathrooms": request.num_bathrooms,
        "include_kitchen": request.include_kitchen,
        "include_living_room": request.include_living_room,
        "include_corridor": request.include_corridor,
        "notes": request.notes,
        "status": "planned",
        "planned_output_files": {
            "image": planned_output_image,
            "layout_json": planned_output_layout,
        },
        "planned_actions": planned_actions,
        "mode_runtime": "skeleton-no-inference",
    }

    meta_path = _meta_path(job_folder)
    _write_meta(job_folder, meta)

    return {
        "status": "ok",
        "message": "Prompt-based floorplan generation planned (skeleton, no AI yet).",
        "job_folder": job_folder,
        "meta_path": meta_path,
        "planned_outputs": {
            "image": os.path.join(job_folder, planned_output_image),
            "layout_json": os.path.join(job_folder, planned_output_layout),
        },
    }
