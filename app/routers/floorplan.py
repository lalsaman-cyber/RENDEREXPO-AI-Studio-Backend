# app/routers/floorplan.py

"""
RENDEREXPO AI STUDIO - Floorplan Router (Planning-first, production-real intent)

Purpose:
- Planning-only router for floorplan workflows
- Creates job folders, saves files, writes meta.json
- Embeds locked SD3.5 preset blocks for later REAL GPU execution

IMPORTANT:
- Planner = port 8012
- GPU worker = port 8002
- This router does NOT run inference
- This router does NOT load SD3.5
- Preset logic comes from app.presets_sd35.apply_preset_to_meta(...)
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.presets_sd35 import apply_preset_to_meta

router = APIRouter(prefix="/api/floorplan", tags=["Floorplan"])

Category = Literal["urban", "suburban", "interior", "wide_hero"]
Shot = Literal["wide", "close"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_iso() -> str:
    return datetime.datetime.utcnow().isoformat()


def _today_utc_str() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


def _create_job_folder(base_outputs_dir: str = "outputs", job_type: str = "floorplan") -> str:
    """Create outputs/{YYYY-MM-DD}/{job_id}/ folder (UTC)."""
    today = _today_utc_str()
    job_id = uuid.uuid4().hex
    folder = os.path.join(base_outputs_dir, today, job_id)
    os.makedirs(folder, exist_ok=True)

    try:
        with open(os.path.join(folder, "job_type.txt"), "w", encoding="utf-8") as f:
            f.write(job_type)
    except Exception:
        pass

    return folder


def _ensure_folder_exists(job_folder: str) -> None:
    """Ensure the given job_folder exists, or raise 400."""
    if not job_folder or not os.path.isdir(job_folder):
        raise HTTPException(status_code=400, detail=f"job_folder does not exist: {job_folder}")


def _meta_path(job_folder: str) -> str:
    return os.path.join(job_folder, "meta.json")


def _read_meta(job_folder: str) -> Dict[str, Any]:
    meta_file = _meta_path(job_folder)
    if not os.path.isfile(meta_file):
        return {}
    with open(meta_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _write_meta(job_folder: str, meta: Dict[str, Any]) -> None:
    meta_file = _meta_path(job_folder)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)


def _parse_job_path(job_folder: str) -> Tuple[Optional[str], Optional[str]]:
    parts = os.path.normpath(job_folder).split(os.sep)
    if len(parts) < 3:
        return None, None
    return parts[-2], parts[-1]


def _outputs_public_urls(job_folder: str) -> Dict[str, Optional[str]]:
    """
    Stable URLs assuming FastAPI mounts outputs/ at /outputs.
    """
    date_str, job_id = _parse_job_path(job_folder)
    if not date_str or not job_id:
        return {
            "meta_url": None,
            "floorplan_url": None,
        }

    base = f"/outputs/{date_str}/{job_id}"
    return {
        "meta_url": f"{base}/meta.json",
        "floorplan_url": f"{base}/floorplan.png",
    }


def _ensure_png_jpg_only(upload: UploadFile) -> None:
    ct = (getattr(upload, "content_type", "") or "").lower().strip()
    if ct and ct not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(status_code=400, detail="Only PNG and JPG are supported.")

    name = (upload.filename or "").lower()
    if name and not (name.endswith(".png") or name.endswith(".jpg") or name.endswith(".jpeg")):
        raise HTTPException(status_code=400, detail="Only .png, .jpg, .jpeg are supported.")


async def _save_upload_stream(upload: UploadFile, dst_path: str) -> None:
    """
    Save UploadFile without reading everything into RAM.
    """
    try:
        try:
            upload.file.seek(0)
        except Exception:
            pass

        with open(dst_path, "wb") as out:
            shutil.copyfileobj(upload.file, out)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed saving upload '{upload.filename}': {exc}") from exc


def _build_sd35_planned_meta(
    *,
    prompt: str,
    negative_prompt: Optional[str],
    category: Category,
    shot: Shot,
    upscale_2x: Optional[bool],
    seed: Optional[int] = None,
    job_type: str = "floorplan_sd35_planned",
    input_image: Optional[str] = None,
    strength: Optional[float] = None,
    planned_output_image: str = "output.png",
) -> Dict[str, Any]:
    """
    Build a SD3.5 planned meta dict using locked presets.
    This is planning only and saved into floorplan meta.
    """
    final_seed = seed if seed is not None else int(uuid.uuid4().int % 1_000_000_000)

    meta: Dict[str, Any] = {
        "job_type": job_type,
        "created_at": _utc_iso(),
        "type": "text2img" if input_image is None else "img2img",
        "engine": "sd35_large_pro_v2_1",
        "model_name": "sd35_large_pro_v2_1",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": final_seed,
        "category": category,
        "shot": shot,
        "planned_output_image": planned_output_image,
    }

    if input_image is not None:
        meta["input_image"] = input_image
    if strength is not None:
        meta["strength"] = float(strength)

    apply_preset_to_meta(meta, category=category, shot=shot, upscale_2x=upscale_2x)
    return meta


# ---------------------------------------------------------------------------
# 0) Pydantic model for prompt-based floorplan generator
# ---------------------------------------------------------------------------

class FloorplanGenerateRequest(BaseModel):
    """
    Prompt → top-down, clean 2D rendered floorplan planning.
    """
    prompt: str = Field(
        ...,
        description=(
            "Describe the building/function clearly, e.g. "
            "'single-family house, 3 bedrooms, open kitchen, 2 bathrooms, garage, patio'."
        ),
    )
    width: int = Field(1024, ge=128, le=4096, description="Planned output width (pixels).")
    height: int = Field(1024, ge=128, le=4096, description="Planned output height (pixels).")

    wall_thickness: float = Field(0.2, ge=0.05, le=1.0, description="Wall thickness in meters (conceptual).")
    num_bedrooms: Optional[int] = Field(default=None, ge=0, le=20)
    num_bathrooms: Optional[int] = Field(default=None, ge=0, le=20)
    include_kitchen: bool = Field(True)
    include_living_room: bool = Field(True)
    include_corridor: bool = Field(True)
    notes: Optional[str] = Field(default=None, description="Extra constraints.")

    category: Category = Field("interior", description="Preset category")
    shot: Shot = Field("wide", description="Preset shot")
    upscale_2x: Optional[bool] = Field(
        None,
        description="Optional override: true/false. If omitted, preset default applies.",
    )
    seed: Optional[int] = Field(default=None, description="Optional seed. If omitted, backend generates one.")
    negative_prompt: Optional[str] = Field(default=None, description="Optional negative prompt.")


# ---------------------------------------------------------------------------
# 1) Upload floorplan
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_floorplan(image: UploadFile = File(..., description="Floorplan image (PNG/JPG only)")) -> Dict[str, Any]:
    if not image.filename:
        raise HTTPException(status_code=400, detail="Uploaded image has no filename.")
    _ensure_png_jpg_only(image)

    job_folder = _create_job_folder(job_type="floorplan_upload")
    _ensure_folder_exists(job_folder)

    floorplan_path = os.path.join(job_folder, "floorplan.png")
    await _save_upload_stream(image, floorplan_path)

    meta = _read_meta(job_folder)
    meta.setdefault("job_id", os.path.basename(job_folder))
    meta.setdefault("type", "floorplan")

    meta["created_at"] = _utc_iso()
    meta["engine"] = "sd35_large_pro_v2_1"
    meta["model_name"] = "sd35_large_pro_v2_1"

    meta["inputs"] = {
        "floorplan_image": "floorplan.png",
        "content_type": getattr(image, "content_type", None),
        "original_filename": image.filename,
    }

    meta["floorplan_image"] = "floorplan.png"
    meta.setdefault("cameras", [])
    meta.setdefault("planned_render", None)
    meta["mode_runtime"] = "planned-real"
    meta["status"] = "uploaded"

    _write_meta(job_folder, meta)

    return {
        "status": "ok",
        "message": "Floorplan uploaded.",
        "job_folder": job_folder,
        "meta_path": _meta_path(job_folder),
        "public_urls": _outputs_public_urls(job_folder),
        "saved_files": {
            "floorplan": os.path.join(job_folder, "floorplan.png"),
        },
    }


# ---------------------------------------------------------------------------
# 2) Set camera inside the floorplan
# ---------------------------------------------------------------------------

@router.post("/set-camera")
async def set_camera(
    job_folder: str = Form(..., description="Job folder returned by /floorplan/upload"),
    camera_x: float = Form(..., description="Camera X in floorplan coordinates"),
    camera_y: float = Form(..., description="Camera Y in floorplan coordinates"),
    rotation_deg: float = Form(..., description="Camera rotation in degrees"),
) -> Dict[str, Any]:
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
        "created_at": _utc_iso(),
    }

    meta["cameras"].append(camera_info)
    meta["updated_at"] = _utc_iso()
    _write_meta(job_folder, meta)

    return {
        "status": "ok",
        "message": "Camera added to floorplan.",
        "job_folder": job_folder,
        "camera": camera_info,
        "meta_path": _meta_path(job_folder),
        "public_urls": _outputs_public_urls(job_folder),
    }


# ---------------------------------------------------------------------------
# 3) Plan SD3.5 render from that floorplan + camera(s)
# ---------------------------------------------------------------------------

@router.post("/plan-render")
async def plan_floorplan_render(
    job_folder: str = Form(..., description="Job folder returned by /floorplan/upload"),
    prompt: str = Form(..., description="Prompt for the planned render"),
    negative_prompt: Optional[str] = Form(None),
    category: Category = Form(...),
    shot: Shot = Form(...),
    upscale_2x: Optional[bool] = Form(None),
    seed: Optional[int] = Form(None),
) -> Dict[str, Any]:
    _ensure_folder_exists(job_folder)

    meta = _read_meta(job_folder)

    sd35_meta = _build_sd35_planned_meta(
        prompt=prompt,
        negative_prompt=negative_prompt,
        category=category,
        shot=shot,
        upscale_2x=upscale_2x,
        seed=seed,
        job_type="floorplan_plan_render",
        planned_output_image="floorplan_view.png",
    )

    sd35_meta["source"] = {
        "type": "floorplan",
        "floorplan_image": meta.get("floorplan_image", "floorplan.png"),
        "cameras_count": len(meta.get("cameras", []) or []),
    }

    planned_render = {
        "created_at": _utc_iso(),
        "planned_output_image": "floorplan_view.png",
        "sd35_meta": sd35_meta,
    }

    meta["planned_render"] = planned_render
    meta["mode_runtime"] = "planned-real"
    meta["updated_at"] = _utc_iso()
    meta["status"] = "planned"

    _write_meta(job_folder, meta)

    date_str, job_id = _parse_job_path(job_folder)
    base = f"/outputs/{date_str}/{job_id}" if date_str and job_id else None
    planned_public = {
        "planned_output_image_url": f"{base}/floorplan_view.png" if base else None,
    }

    return {
        "status": "ok",
        "message": "Floorplan render planned using locked presets.",
        "job_folder": job_folder,
        "planned_render": planned_render,
        "meta_path": _meta_path(job_folder),
        "public_urls": _outputs_public_urls(job_folder),
        "planned_public_urls": planned_public,
    }


# ---------------------------------------------------------------------------
# 4) One-shot Floorplan → 3D Pipeline Planning
# ---------------------------------------------------------------------------

@router.post("/plan-3d")
async def plan_floorplan_to_3d(
    floorplan: UploadFile = File(..., description="Floorplan image (PNG/JPG only)"),
    prompt: str = Form(..., description="High-level description"),
    negative_prompt: Optional[str] = Form(None),
    category: Category = Form("interior"),
    shot: Shot = Form("wide"),
    upscale_2x: Optional[bool] = Form(None),
) -> Dict[str, Any]:
    if not floorplan.filename:
        raise HTTPException(status_code=400, detail="Uploaded floorplan has no filename.")
    _ensure_png_jpg_only(floorplan)

    job_folder = _create_job_folder(job_type="floorplan_to_3d")
    floorplan_path = os.path.join(job_folder, "floorplan.png")
    await _save_upload_stream(floorplan, floorplan_path)

    created = _utc_iso()

    sd35_template = _build_sd35_planned_meta(
        prompt=prompt,
        negative_prompt=negative_prompt,
        category=category,
        shot=shot,
        upscale_2x=upscale_2x,
        seed=None,
        job_type="floorplan_to_3d_generate_views_template",
        planned_output_image="output.png",
    )
    sd35_template["source"] = {"type": "floorplan", "floorplan_image": "floorplan.png"}

    planned_actions: List[Dict[str, Any]] = [
        {"stage": "load_floorplan", "file": "floorplan.png", "description": "Load floorplan image for analysis."},
        {"stage": "detect_layout", "description": "Detect walls/rooms/doors/windows (future)."},
        {"stage": "place_cameras", "description": "Auto-place cameras in key rooms (future)."},
        {
            "stage": "generate_views",
            "description": "Use SD3.5 + ControlNet to generate interior views (future).",
            "sd35_meta_template": sd35_template,
        },
    ]

    meta: Dict[str, Any] = {
        "job_id": os.path.basename(job_folder),
        "created_at": created,
        "type": "floorplan_to_3d",
        "engine": "sd35_large_pro_v2_1",
        "model_name": "sd35_large_pro_v2_1",
        "floorplan": "floorplan.png",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "status": "planned",
        "pipeline": "floorplan_to_3d_v1",
        "planned_actions": planned_actions,
        "mode_runtime": "planned-real",
        "inputs": {
            "floorplan_image": "floorplan.png",
            "content_type": getattr(floorplan, "content_type", None),
            "original_filename": floorplan.filename,
        },
    }

    _write_meta(job_folder, meta)

    return {
        "status": "ok",
        "message": "Floorplan to 3D pipeline planned.",
        "job_folder": job_folder,
        "meta_path": _meta_path(job_folder),
        "public_urls": _outputs_public_urls(job_folder),
        "sd35_meta_template": sd35_template,
    }


# ---------------------------------------------------------------------------
# 5) Plan camera-based room view from a floorplan
# ---------------------------------------------------------------------------

@router.post("/plan-camera-view")
async def plan_camera_view(
    job_folder: str = Form(..., description="Job folder returned by /floorplan/upload"),
    camera_id: str = Form(..., description="camera_id from /floorplan/set-camera"),
    prompt: str = Form(..., description="Prompt for the room view"),
    negative_prompt: Optional[str] = Form(None),
    category: Category = Form(...),
    shot: Shot = Form(...),
    upscale_2x: Optional[bool] = Form(None),
    seed: Optional[int] = Form(None),
) -> Dict[str, Any]:
    _ensure_folder_exists(job_folder)

    meta = _read_meta(job_folder)
    cameras = meta.get("cameras", [])
    if not isinstance(cameras, list) or not cameras:
        raise HTTPException(status_code=400, detail="No cameras found. Use /floorplan/set-camera first.")

    target_camera: Optional[Dict[str, Any]] = None
    for cam in cameras:
        if cam.get("camera_id") == camera_id:
            target_camera = cam
            break

    if not target_camera:
        raise HTTPException(status_code=404, detail=f"camera_id '{camera_id}' not found in this job.")

    sd35_meta = _build_sd35_planned_meta(
        prompt=prompt,
        negative_prompt=negative_prompt,
        category=category,
        shot=shot,
        upscale_2x=upscale_2x,
        seed=seed,
        job_type="floorplan_plan_camera_view",
        planned_output_image=f"{camera_id}_view.png",
    )
    sd35_meta["source"] = {
        "type": "floorplan_camera",
        "floorplan_image": meta.get("floorplan_image", "floorplan.png"),
        "camera_id": camera_id,
    }

    planned_view = {
        "created_at": _utc_iso(),
        "planned_output": f"{camera_id}_view.png",
        "sd35_meta": sd35_meta,
    }

    target_camera["planned_view"] = planned_view
    meta["cameras"] = cameras
    meta["mode_runtime"] = "planned-real"
    meta["updated_at"] = _utc_iso()
    meta["status"] = "planned"

    _write_meta(job_folder, meta)

    date_str, job_id = _parse_job_path(job_folder)
    base = f"/outputs/{date_str}/{job_id}" if date_str and job_id else None
    planned_public = {
        "planned_output_image_url": f"{base}/{camera_id}_view.png" if base else None,
    }

    return {
        "status": "ok",
        "message": "Camera-based room view planned using locked presets.",
        "job_folder": job_folder,
        "camera_id": camera_id,
        "planned_view": planned_view,
        "meta_path": _meta_path(job_folder),
        "public_urls": _outputs_public_urls(job_folder),
        "planned_public_urls": planned_public,
    }


# ---------------------------------------------------------------------------
# 6) Generate floorplan from prompt (planning)
# ---------------------------------------------------------------------------

@router.post("/generate-from-prompt")
async def generate_floorplan_from_prompt(request: FloorplanGenerateRequest) -> Dict[str, Any]:
    job_folder = _create_job_folder(job_type="floorplan_generate")
    created = _utc_iso()

    planned_output_image = "floorplan_generated.png"
    planned_output_layout = "floorplan_layout.json"
    planned_output_materials = "floorplan_materials.json"

    render_intent: Dict[str, Any] = {
        "view": "top_down",
        "projection": "orthographic",
        "style": "clean_2d_architectural_plan",
        "requirements": {
            "consistent_wall_thickness": True,
            "doors_and_windows_symbols": True,
            "furnished_for_program": True,
            "rendered_flooring_materials": True,
            "rendered_furniture_materials": True,
            "legible_room_boundaries": True,
            "high_contrast_linework": True,
        },
        "wall_thickness_m": float(request.wall_thickness),
        "program_hints": {
            "num_bedrooms": request.num_bedrooms,
            "num_bathrooms": request.num_bathrooms,
            "include_kitchen": bool(request.include_kitchen),
            "include_living_room": bool(request.include_living_room),
            "include_corridor": bool(request.include_corridor),
        },
        "notes": request.notes,
    }

    sd35_floorplan_meta = _build_sd35_planned_meta(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        category=request.category,
        shot=request.shot,
        upscale_2x=request.upscale_2x,
        seed=request.seed,
        job_type="floorplan_generate_topdown_sd35",
        planned_output_image=planned_output_image,
    )
    sd35_floorplan_meta["render_intent"] = render_intent
    sd35_floorplan_meta["output_constraints"] = {
        "width": int(request.width),
        "height": int(request.height),
        "format": "png",
    }

    planned_actions: List[Dict[str, Any]] = [
        {
            "stage": "parse_program",
            "description": "Parse prompt into a structured building program + constraints (future).",
            "inputs": ["prompt"],
            "outputs": ["program.json"],
        },
        {
            "stage": "layout_synthesis",
            "description": "Synthesize a valid 2D layout graph with consistent wall thickness (future).",
            "inputs": ["program.json"],
            "outputs": [planned_output_layout],
        },
        {
            "stage": "material_assignment",
            "description": "Assign flooring/material/furniture material tags for rendering (future).",
            "inputs": [planned_output_layout],
            "outputs": [planned_output_materials],
        },
        {
            "stage": "sd35_render_topdown_floorplan",
            "description": "Render the top-down 2D plan with SD3.5 using locked presets (future GPU).",
            "inputs": [planned_output_layout, planned_output_materials],
            "outputs": [planned_output_image],
            "sd35_meta": sd35_floorplan_meta,
        },
    ]

    meta: Dict[str, Any] = {
        "job_id": os.path.basename(job_folder),
        "created_at": created,
        "type": "floorplan_generate",
        "engine": "sd35_large_pro_v2_1",
        "model_name": "sd35_large_pro_v2_1",
        "generator": "floorplan_prompt_v2_topdown_materialized",
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "render_intent": render_intent,
        "status": "planned",
        "mode_runtime": "planned-real",
        "planned_output_files": {
            "image": planned_output_image,
            "layout_json": planned_output_layout,
            "materials_json": planned_output_materials,
        },
        "planned_actions": planned_actions,
        "sd35_planned_stage": {
            "enabled": True,
            "sd35_meta": sd35_floorplan_meta,
        },
    }

    _write_meta(job_folder, meta)

    date_str, job_id = _parse_job_path(job_folder)
    base = f"/outputs/{date_str}/{job_id}" if date_str and job_id else None

    return {
        "status": "ok",
        "message": "Top-down floorplan generation planned with locked SD3.5 meta embedded.",
        "job_folder": job_folder,
        "meta_path": _meta_path(job_folder),
        "public_urls": {
            "meta_url": f"{base}/meta.json" if base else None,
            "planned_floorplan_image_url": f"{base}/{planned_output_image}" if base else None,
            "planned_layout_json_url": f"{base}/{planned_output_layout}" if base else None,
            "planned_materials_json_url": f"{base}/{planned_output_materials}" if base else None,
        },
        "sd35_preset_applied": sd35_floorplan_meta.get("preset", {}),
    }