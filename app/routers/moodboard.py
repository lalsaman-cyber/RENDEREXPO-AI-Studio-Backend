# app/routers/moodboard.py
"""
RENDEREXPO AI STUDIO - Moodboard (REAL via GPU dispatch)

Two-way street:

1) Moodboard -> Space
   - User uploads 1..N moodboard images (+ optional prompt, optional floorplan)
   - We create job folder, save files, write meta.json
   - We DISPATCH to GPU worker to generate:
        - output.png (generated space)
        - palette.json, extracted_assets.json (optional / best-effort)
        - meta.json updated

2) Space -> Moodboard
   - User uploads a space image (+ optional prompt)
   - We DISPATCH to GPU worker to produce:
        - moodboard_grid.png
        - palette.json
        - extracted_assets.json
        - meta.json updated

3) Moodboard -> Apply to Render
   - User references a moodboard job + uploads a target image OR references a prior job image
   - We DISPATCH to GPU worker to apply materials/style from moodboard to the target render:
        - output.png
        - meta.json updated

CRITICAL (Doc 18):
- Any SD3.5 stage must use locked preset system:
  steps, CFG, LyCORIS(PRO 2.1) multiplier, GEO multiplier, resolution
- NO denoise anywhere (denoise always 0.0)
- Upscale is OPTIONAL (but YOU said you want best outcome: we default to preset default)
"""

from __future__ import annotations

import os
import uuid
import json
import shutil
import datetime
from typing import Optional, List, Dict, Any, Tuple

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field

from app.presets_sd35 import apply_preset_to_meta

# GPU dispatch (you will paste/align these later in app.clients.gpu_client)
from app.clients.gpu_client import (
    dispatch_sd35_moodboard_to_space,
    dispatch_space_to_moodboard,
    dispatch_sd35_apply_moodboard_to_render,
)

router = APIRouter(prefix="/api/moodboard", tags=["Moodboard (REAL)"])


# ---------------------------------------------------------------------------
# Preset auto-choice (you said: "I dont like choosing, you choose best outcome")
# ---------------------------------------------------------------------------

DEFAULT_CATEGORY = "interior"
DEFAULT_SHOT = "wide"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _today_utc_str() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


def _create_job_folder(base_outputs_dir: str = "outputs") -> str:
    today = _today_utc_str()
    job_id = uuid.uuid4().hex
    folder = os.path.join(base_outputs_dir, today, job_id)
    os.makedirs(folder, exist_ok=True)

    # marker file
    try:
        with open(os.path.join(folder, "job_type.txt"), "w", encoding="utf-8") as f:
            f.write("moodboard")
    except Exception:
        pass

    return folder


def _parse_job_path(job_folder: str) -> Tuple[Optional[str], Optional[str]]:
    parts = os.path.normpath(job_folder).split(os.sep)
    if len(parts) < 3:
        return None, None
    return parts[-2], parts[-1]


def _outputs_public_urls(job_folder: str) -> Dict[str, Optional[str]]:
    date_str, job_id = _parse_job_path(job_folder)
    if not date_str or not job_id:
        return {"image_url": None, "meta_url": None}

    base = f"/outputs/{date_str}/{job_id}"
    return {
        "image_url": f"{base}/output.png",
        "meta_url": f"{base}/meta.json",
        "moodboard_grid_url": f"{base}/moodboard_grid.png",
        "palette_url": f"{base}/palette.json",
        "assets_url": f"{base}/extracted_assets.json",
    }


def _write_meta(job_folder: str, meta: Dict[str, Any]) -> str:
    meta_path = os.path.join(job_folder, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
    return meta_path


async def _save_upload_stream(upload: UploadFile, dst_path: str) -> None:
    try:
        try:
            upload.file.seek(0)
        except Exception:
            pass

        with open(dst_path, "wb") as out:
            shutil.copyfileobj(upload.file, out)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to save upload '{upload.filename}': {exc}") from exc


def _ensure_png_jpg(upload: UploadFile, label: str) -> None:
    ct = (getattr(upload, "content_type", "") or "").lower().strip()
    # allow missing content_type (some clients)
    if ct and ("png" not in ct and "jpeg" not in ct and "jpg" not in ct):
        raise HTTPException(status_code=400, detail=f"{label}: Only PNG and JPG are allowed")


def _find_job_folder_by_job_id(job_id: str) -> Optional[str]:
    outputs_dir = "outputs"
    if not os.path.isdir(outputs_dir):
        return None

    for date_dir in sorted(os.listdir(outputs_dir), reverse=True):
        date_path = os.path.join(outputs_dir, date_dir)
        if not os.path.isdir(date_path):
            continue
        candidate = os.path.join(date_path, job_id)
        if os.path.isdir(candidate):
            return candidate
    return None


def _safe_relpath(rel: str) -> str:
    rel = (rel or "").strip().replace("\\", "/")
    rel = rel.lstrip("/")
    norm = os.path.normpath(rel).replace("\\", "/")
    if not norm or norm == ".":
        return "output.png"
    if norm.startswith("../") or norm == ".." or "/../" in f"/{norm}/":
        raise HTTPException(status_code=400, detail="Invalid source_image_path.")
    return norm


# ---------------------------------------------------------------------------
# 1) Moodboard -> Space (REAL)
# ---------------------------------------------------------------------------

@router.post("/from-moodboard/render")
async def moodboard_to_space_render(
    moodboard_images: List[UploadFile] = File(..., description="1..N moodboard images (PNG/JPG)"),
    prompt: Optional[str] = Form(None),
    floorplan_image: Optional[UploadFile] = File(None),

    # Optional: upscale override (you can still override, but default is preset best)
    upscale_2x: Optional[bool] = Form(None),

    # Optional seed
    seed: Optional[int] = Form(None),
):
    if not moodboard_images:
        raise HTTPException(status_code=400, detail="At least one moodboard image is required.")

    for i, up in enumerate(moodboard_images):
        if not up.filename:
            raise HTTPException(status_code=400, detail=f"moodboard_images[{i}] has no filename.")
        _ensure_png_jpg(up, f"moodboard_images[{i}]")

    if floorplan_image is not None and floorplan_image.filename:
        _ensure_png_jpg(floorplan_image, "floorplan_image")

    job_folder = _create_job_folder()
    job_id = os.path.basename(job_folder)
    created_at = datetime.datetime.utcnow().isoformat()
    final_seed = seed if seed is not None else int(uuid.uuid4().int % 1_000_000_000)

    saved_files: Dict[str, Any] = {"moodboard_images": []}

    # Save moodboard images
    for idx, upload in enumerate(moodboard_images):
        fn = f"moodboard_{idx:03d}.png"
        await _save_upload_stream(upload, os.path.join(job_folder, fn))
        saved_files["moodboard_images"].append(fn)

    # Save optional floorplan
    if floorplan_image is not None and floorplan_image.filename:
        await _save_upload_stream(floorplan_image, os.path.join(job_folder, "floorplan.png"))
        saved_files["floorplan_image"] = "floorplan.png"

    # Locked Doc 18 preset system (auto-best choice)
    category = DEFAULT_CATEGORY
    shot = DEFAULT_SHOT

    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": created_at,
        "type": "moodboard_to_space",
        "status": "planned",
        "mode_runtime": "gpu-dispatch",

        "prompt": prompt,
        "seed": final_seed,

        "files": saved_files,

        # Doc 18 selectors (auto)
        "category": category,
        "shot": shot,

        # Hard lock
        "denoise": 0.0,

        "outputs": {
            "output_image": "output.png",
            "palette_json": "palette.json",
            "extracted_assets": "extracted_assets.json",
        },
    }

    # Apply locked presets (steps/cfg/lycoris/geo/resolution + upscale default/override)
    apply_preset_to_meta(meta, category=category, shot=shot, upscale_2x=upscale_2x)

    meta["denoise"] = 0.0
    if isinstance(meta.get("upscale"), dict):
        meta["upscale"]["denoise"] = 0.0

    meta_path = _write_meta(job_folder, meta)
    public_urls = _outputs_public_urls(job_folder)

    ok, gpu_resp = dispatch_sd35_moodboard_to_space(job_folder=job_folder, meta=meta)
    if not ok:
        try:
            meta["status"] = "gpu_error"
            meta["gpu_error"] = gpu_resp
            _write_meta(job_folder, meta)
        except Exception:
            pass
        return {
            "status": "gpu_error",
            "message": "Job created but GPU worker failed.",
            "job_folder": job_folder,
            "meta_path": meta_path,
            "public_urls": public_urls,
            "gpu_error": gpu_resp,
        }

    try:
        meta["status"] = "dispatched"
        meta["gpu_response"] = gpu_resp
        _write_meta(job_folder, meta)
    except Exception:
        pass

    return {
        "status": "dispatched",
        "message": "Moodboard -> Space dispatched to GPU worker (Doc 18 locked presets, auto-best).",
        "job_folder": job_folder,
        "meta_path": meta_path,
        "public_urls": public_urls,
        "gpu_response": gpu_resp,
        "preset_applied": meta.get("preset", {}),
        "upscale_enabled": bool(meta.get("upscale", {}).get("enabled", False)),
    }


# ---------------------------------------------------------------------------
# 2) Space -> Moodboard (REAL)
# ---------------------------------------------------------------------------

@router.post("/from-space/render")
async def space_to_moodboard_render(
    space_image: UploadFile = File(..., description="Space/room image (PNG/JPG)"),
    prompt: Optional[str] = Form(None),
    num_tiles: int = Form(12, ge=4, le=36),
):
    if not space_image.filename:
        raise HTTPException(status_code=400, detail="space_image has no filename.")
    _ensure_png_jpg(space_image, "space_image")

    job_folder = _create_job_folder()
    job_id = os.path.basename(job_folder)
    created_at = datetime.datetime.utcnow().isoformat()

    await _save_upload_stream(space_image, os.path.join(job_folder, "space.png"))

    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": created_at,
        "type": "space_to_moodboard",
        "status": "planned",
        "mode_runtime": "gpu-dispatch",

        "prompt": prompt,
        "num_tiles": int(num_tiles),

        "files": {"space_image": "space.png"},

        # Global rule consistency
        "denoise": 0.0,

        "outputs": {
            "moodboard_grid": "moodboard_grid.png",
            "palette_json": "palette.json",
            "extracted_assets": "extracted_assets.json",
        },
    }

    meta_path = _write_meta(job_folder, meta)
    public_urls = _outputs_public_urls(job_folder)

    ok, gpu_resp = dispatch_space_to_moodboard(job_folder=job_folder, meta=meta)
    if not ok:
        try:
            meta["status"] = "gpu_error"
            meta["gpu_error"] = gpu_resp
            _write_meta(job_folder, meta)
        except Exception:
            pass
        return {
            "status": "gpu_error",
            "message": "Job created but GPU worker failed.",
            "job_folder": job_folder,
            "meta_path": meta_path,
            "public_urls": public_urls,
            "gpu_error": gpu_resp,
        }

    try:
        meta["status"] = "dispatched"
        meta["gpu_response"] = gpu_resp
        _write_meta(job_folder, meta)
    except Exception:
        pass

    return {
        "status": "dispatched",
        "message": "Space -> Moodboard dispatched to GPU worker.",
        "job_folder": job_folder,
        "meta_path": meta_path,
        "public_urls": public_urls,
        "gpu_response": gpu_resp,
    }


# ---------------------------------------------------------------------------
# 3) Apply Moodboard to Render (REAL)
# ---------------------------------------------------------------------------

class ApplyMoodboardFromJobRequest(BaseModel):
    moodboard_job_id: str = Field(..., description="Job id of a prior moodboard_to_space or space_to_moodboard job")
    target_source_job_id: str = Field(..., description="Job id containing the target image")
    target_image_path: str = Field(default="output.png", description="Relative path inside the target job folder")

    prompt: Optional[str] = Field(default=None, description="Optional override prompt to guide application")
    negative_prompt: Optional[str] = Field(default=None)

    strength: float = Field(default=0.55, ge=0.0, le=1.0, description="0..1 (lower preserves more)")
    upscale_2x: Optional[bool] = Field(default=None)
    seed: Optional[int] = Field(default=None)


@router.post("/apply-to-render/from-job/render")
async def apply_moodboard_to_render_from_job(req: ApplyMoodboardFromJobRequest):
    mb_id = (req.moodboard_job_id or "").strip()
    tgt_id = (req.target_source_job_id or "").strip()
    if not mb_id or not tgt_id:
        raise HTTPException(status_code=400, detail="moodboard_job_id and target_source_job_id are required")

    mb_folder = _find_job_folder_by_job_id(mb_id)
    if not mb_folder:
        raise HTTPException(status_code=404, detail=f"moodboard job folder not found for job_id={mb_id}")

    tgt_folder = _find_job_folder_by_job_id(tgt_id)
    if not tgt_folder:
        raise HTTPException(status_code=404, detail=f"target job folder not found for job_id={tgt_id}")

    safe_rel = _safe_relpath(req.target_image_path)
    src_img = os.path.join(tgt_folder, safe_rel)
    if not os.path.isfile(src_img):
        raise HTTPException(status_code=404, detail=f"target image not found: {safe_rel}")

    job_folder = _create_job_folder()
    job_id = os.path.basename(job_folder)
    created_at = datetime.datetime.utcnow().isoformat()

    # Copy target image into this new job
    dst_img = os.path.join(job_folder, "input.png")
    try:
        shutil.copyfile(src_img, dst_img)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed copying target image: {exc}")

    final_seed = req.seed if req.seed is not None else int(uuid.uuid4().int % 1_000_000_000)

    # Auto-best preset choice
    category = DEFAULT_CATEGORY
    shot = DEFAULT_SHOT

    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": created_at,
        "type": "apply_moodboard_to_render",
        "status": "planned",
        "mode_runtime": "gpu-dispatch",

        "moodboard_job_id": mb_id,
        "moodboard_folder_ref": os.path.basename(mb_folder),  # minimal trace

        "prompt": req.prompt,
        "negative_prompt": req.negative_prompt,
        "seed": final_seed,

        "strength": float(req.strength),

        "files": {
            "input_image": "input.png",
            "moodboard_job_id": mb_id,
        },

        "planned_output_image": "output.png",

        "category": category,
        "shot": shot,

        "denoise": 0.0,
    }

    apply_preset_to_meta(meta, category=category, shot=shot, upscale_2x=req.upscale_2x)
    meta["denoise"] = 0.0
    if isinstance(meta.get("upscale"), dict):
        meta["upscale"]["denoise"] = 0.0

    meta_path = _write_meta(job_folder, meta)
    public_urls = _outputs_public_urls(job_folder)

    ok, gpu_resp = dispatch_sd35_apply_moodboard_to_render(
        job_folder=job_folder,
        meta=meta,
        moodboard_job_folder=mb_folder,
    )
    if not ok:
        try:
            meta["status"] = "gpu_error"
            meta["gpu_error"] = gpu_resp
            _write_meta(job_folder, meta)
        except Exception:
            pass
        return {
            "status": "gpu_error",
            "message": "Job created but GPU worker failed.",
            "job_folder": job_folder,
            "meta_path": meta_path,
            "public_urls": public_urls,
            "gpu_error": gpu_resp,
        }

    try:
        meta["status"] = "dispatched"
        meta["gpu_response"] = gpu_resp
        _write_meta(job_folder, meta)
    except Exception:
        pass

    return {
        "status": "dispatched",
        "message": "Apply-moodboard job dispatched to GPU worker (auto-best, Doc 18 locked).",
        "job_folder": job_folder,
        "meta_path": meta_path,
        "public_urls": public_urls,
        "gpu_response": gpu_resp,
        "preset_applied": meta.get("preset", {}),
    }
