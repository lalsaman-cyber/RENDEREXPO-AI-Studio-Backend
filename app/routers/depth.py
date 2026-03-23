# app/routers/depth.py
"""
RENDEREXPO AI STUDIO - Depth Maps Router (Planning only)

PLANNING ONLY:
- No GPU
- No inference
- No SD3.5 loading
- No diffusion

Purpose:
- creates outputs/YYYY-MM-DD/<job_id>/
- saves input image (input.png)
- writes meta.json with a depth_plan section
- can optionally attach locked SD3.5 preset context for downstream stages

IMPORTANT:
- Planner = port 8012
- GPU worker = port 8002
- This router must NOT introduce diffusion denoise.
- This router may store preset context so downstream SD3.5 stages stay aligned.

IMPORTANT DISTINCTION:
- This router is for standalone planned depth jobs.
- It is NOT the runtime sketch-depth preprocessing path.
- Sketch depth preprocessing now belongs to the GPU/runtime side and is executed
  as part of:
      sketch -> canny + depth -> SD3.5 dual ControlNet -> output
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import uuid
from typing import Any, Dict, Literal, Optional, Tuple

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.presets_sd35 import apply_preset_to_meta

router = APIRouter(prefix="/api/depth", tags=["Depth Maps (Planning Only)"])

Category = Literal["urban", "suburban", "interior", "wide_hero"]
Shot = Literal["wide", "close"]

ALLOWED_BACKENDS = {"midas", "zoedepth", "leres"}
ALLOWED_OUTPUT_FORMATS = {"png", "exr", "npy"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_iso() -> str:
    return datetime.datetime.utcnow().isoformat()


def _today_utc_str() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


def _create_job_folder(job_type: str) -> str:
    today = _today_utc_str()
    job_id = uuid.uuid4().hex
    folder = os.path.join("outputs", today, job_id)
    os.makedirs(folder, exist_ok=True)

    try:
        with open(os.path.join(folder, "job_type.txt"), "w", encoding="utf-8") as f:
            f.write(job_type)
    except Exception:
        pass

    return folder


def _meta_path(job_folder: str) -> str:
    return os.path.join(job_folder, "meta.json")


def _write_meta(job_folder: str, meta: Dict[str, Any]) -> None:
    try:
        with open(_meta_path(job_folder), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed writing meta.json: {exc}") from exc


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
        return {"input_url": None, "meta_url": None}

    base = f"/outputs/{date_str}/{job_id}"
    return {
        "input_url": f"{base}/input.png",
        "meta_url": f"{base}/meta.json",
        "planned_depth_url": f"{base}/depth.png",
        "planned_vis_url": f"{base}/depth_vis.png",
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/plan")
async def plan_depth_map(
    image: UploadFile = File(..., description="Input image to compute depth map from (planning only)."),
    category: Optional[Category] = Form(
        None,
        description="Optional preset category context for downstream SD3.5 stages.",
    ),
    shot: Optional[Shot] = Form(
        None,
        description="Optional preset shot context for downstream SD3.5 stages.",
    ),
    upscale_2x: Optional[bool] = Form(
        None,
        description="Optional: store upscale intent for downstream stages.",
    ),
    depth_backend: str = Form(
        "midas",
        description="Planned depth backend label (planning only). Allowed: midas/zoedepth/leres",
    ),
    output_format: str = Form(
        "png",
        description="Planned output format for depth (planning only). Allowed: png/exr/npy",
    ),
    attach_sd35_context: bool = Form(
        True,
        description="If true, attach locked SD3.5 preset context when category+shot are provided.",
    ),
) -> Dict[str, Any]:
    """
    Plan a depth map job (planning only).

    Creates:
    - outputs/YYYY-MM-DD/<job_id>/
      - input.png
      - meta.json

    Does NOT create:
    - real depth output
    - GPU inference
    - diffusion output
    """
    if not image.filename:
        raise HTTPException(status_code=400, detail="Uploaded image has no filename.")
    _ensure_png_jpg_only(image)

    depth_backend_norm = str(depth_backend).strip().lower()
    output_format_norm = str(output_format).strip().lower()

    if depth_backend_norm not in ALLOWED_BACKENDS:
        raise HTTPException(
            status_code=400,
            detail=f"depth_backend must be one of {sorted(ALLOWED_BACKENDS)}",
        )

    if output_format_norm not in ALLOWED_OUTPUT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"output_format must be one of {sorted(ALLOWED_OUTPUT_FORMATS)}",
        )

    # Avoid half-attached preset context
    if (category is None) ^ (shot is None):
        raise HTTPException(
            status_code=400,
            detail="If providing preset context, you must provide BOTH category and shot.",
        )

    job_folder = _create_job_folder(job_type="depth_plan")
    job_id = os.path.basename(job_folder)

    # Save input image
    input_path = os.path.join(job_folder, "input.png")
    await _save_upload_stream(image, input_path)

    planned_depth_name = f"depth.{output_format_norm}"
    planned_vis_name = "depth_vis.png"

    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": _utc_iso(),
        "type": "depth_plan",
        "job_type": "depth_plan",
        "status": "planned",
        "mode_runtime": "plan-only",
        "engine": "sd35_large_pro_v2_1",
        "model_name": "sd35_large_pro_v2_1",
        "inputs": {
            "image": "input.png",
            "content_type": getattr(image, "content_type", None),
            "original_filename": image.filename,
        },
        "depth_plan": {
            "backend": depth_backend_norm,
            "output_format": output_format_norm,
            "planned_output_depth": planned_depth_name,
            "planned_output_vis": planned_vis_name,
            "note": "Planning only. No depth inference executed here.",
            "planned_at": _utc_iso(),
        },
        "outputs": {
            "depth": planned_depth_name,
            "vis": planned_vis_name,
            "meta": "meta.json",
        },
        "runtime_notes": {
            "gpu_required_later": True,
            "inference_not_implemented_here": True,
            "sketch_runtime_depth_is_separate": True,
            "note": (
                "Standalone depth planning route only. "
                "Sketch-mode depth preprocessing is executed separately inside the GPU sketch ControlNet runtime."
            ),
        },
    }

    # Optional: attach preset context for downstream SD3.5 stages
    preset_context_attached = False
    if attach_sd35_context and category is not None and shot is not None:
        meta["preset_context"] = {
            "category": category,
            "shot": shot,
            "upscale_2x": upscale_2x,
            "note": "Preset context stored for downstream SD3.5 stages. Depth itself does not use steps/CFG.",
        }

        try:
            apply_preset_to_meta(meta, category=category, shot=shot, upscale_2x=upscale_2x)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Preset error: {exc}") from exc

        preset_context_attached = True

    _write_meta(job_folder, meta)

    return {
        "status": "ok",
        "message": "Depth map planned (planning only).",
        "job_folder": job_folder,
        "job_id": job_id,
        "input_saved_as": input_path,
        "meta_path": _meta_path(job_folder),
        "public_urls": _outputs_public_urls(job_folder),
        "planned_outputs": {
            "depth": os.path.join(job_folder, planned_depth_name),
            "vis": os.path.join(job_folder, planned_vis_name),
        },
        "preset_context_attached": preset_context_attached,
        "note": "No depth inference was executed by this route.",
    }