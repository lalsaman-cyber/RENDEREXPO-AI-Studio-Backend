# app/routers/controlnet.py
"""
RENDEREXPO AI STUDIO - ControlNet Planning Router (NO GPU)

CRITICAL (Doc 18):
- ControlNet planning MUST inherit the SAME locked SD3.5 presets:
  steps, CFG, LyCORIS(PRO 2.1), GEO multipliers, resolution
- NO denoise anywhere
- Upscale remains optional and stored for downstream stages
- This router is PLANNING ONLY (no inference, no GPU)

Purpose:
- Accept a conditioning image (sketch, lineart, depth, canny, etc.)
- Store a ControlNet plan + locked preset context in meta.json
- Allow downstream SD3.5 stages to reuse EXACT preset logic
"""

from __future__ import annotations

import os
import uuid
import json
import shutil
import datetime
from typing import Optional, Dict, Any, Literal, Tuple

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.presets_sd35 import apply_preset_to_meta

router = APIRouter(prefix="/api/controlnet", tags=["ControlNet Planning (NO GPU)"])

Category = Literal["urban", "suburban", "interior", "wide_hero"]
Shot = Literal["wide", "close"]
ControlType = Literal["canny", "depth", "lineart", "scribble", "normal"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    with open(_meta_path(job_folder), "w", encoding="utf-8") as f:
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
        return {"control_input_url": None, "meta_url": None}

    base = f"/outputs/{date_str}/{job_id}"
    return {
        "control_input_url": f"{base}/control_input.png",
        "meta_url": f"{base}/meta.json",
    }


def _ensure_png_jpg_only(upload: UploadFile) -> None:
    # best-effort content-type check (not fully trusted)
    ct = (getattr(upload, "content_type", "") or "").lower().strip()
    if ct and ct not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(status_code=400, detail="Only PNG and JPG are supported.")

    # filename extension check
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
# Route
# ---------------------------------------------------------------------------

@router.post("/plan")
async def plan_controlnet_job(
    image: UploadFile = File(..., description="Conditioning image for ControlNet (PNG/JPG only)"),

    control_type: ControlType = Form(
        ...,
        description="Type of ControlNet conditioning (canny, depth, lineart, etc.)",
    ),
    control_strength: float = Form(
        1.0,
        ge=0.0,
        le=2.0,
        description="ControlNet influence strength (planning only).",
    ),

    # LOCKED PRESET SELECTORS (Doc 18)
    category: Category = Form(..., description="urban/suburban/interior/wide_hero"),
    shot: Shot = Form(..., description="wide/close"),

    # Optional per-request upscale override
    upscale_2x: Optional[bool] = Form(
        None,
        description="Optional override: true/false. If omitted, preset default is used.",
    ),
):
    """
    Plan a ControlNet conditioning job (NO GPU).

    Output:
    - outputs/YYYY-MM-DD/<job_id>/
        - control_input.png
        - meta.json (includes locked SD3.5 preset context)
    """
    if not image.filename:
        raise HTTPException(status_code=400, detail="Uploaded image has no filename.")
    _ensure_png_jpg_only(image)

    job_folder = _create_job_folder(job_type="controlnet_plan")
    job_id = os.path.basename(job_folder)

    # Save conditioning image
    input_path = os.path.join(job_folder, "control_input.png")
    await _save_upload_stream(image, input_path)

    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": datetime.datetime.utcnow().isoformat(),

        "type": "controlnet_plan",
        "status": "planned",
        "mode_runtime": "plan-only",

        # Engine identity (NO SDXL)
        "engine": "sd35_large_pro_v2_1",
        "model_name": "sd35_large_pro_v2_1",

        # Preset selectors (Doc 18)
        "category": category,
        "shot": shot,

        # Hard lock
        "denoise": 0.0,

        # Inputs contract
        "inputs": {
            "control_image": "control_input.png",
            "content_type": getattr(image, "content_type", None),
        },

        # ControlNet planning block
        "controlnet": {
            "control_type": control_type,
            "control_strength": float(control_strength),
            "input_image": "control_input.png",
            "note": "Planning only. ControlNet will be applied during SD3.5 inference.",
        },

        # Outputs contract (planning artifact only)
        "outputs": {
            "control_input": "control_input.png",
            "meta": "meta.json",
        },
    }

    # Apply locked Doc 18 preset logic (DO NOT CHANGE multipliers here)
    try:
        apply_preset_to_meta(
            meta,
            category=category,
            shot=shot,
            upscale_2x=upscale_2x,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Preset error: {exc}")

    # Safety re-lock
    meta["denoise"] = 0.0
    if isinstance(meta.get("upscale"), dict):
        meta["upscale"]["denoise"] = 0.0

    _write_meta(job_folder, meta)

    return {
        "status": "ok",
        "message": "ControlNet job planned with locked Doc 18 presets (NO GPU).",
        "job_folder": job_folder,
        "meta_path": _meta_path(job_folder),
        "public_urls": _outputs_public_urls(job_folder),
        "control_input_saved_as": input_path,
        "preset_applied": meta.get("preset", {}),
    }
