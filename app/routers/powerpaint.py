# app/routers/powerpaint.py
"""
RENDEREXPO AI STUDIO - PowerPaint Router

LOCKED SERVICE FAMILY:
    AI Interior Cleanup & Small Decor Enhancement

PUBLIC/PLANNER ENDPOINTS:
    1) POST /api/powerpaint/object-removal/render
       Service name:
           AI Object Removal

    2) POST /api/powerpaint/small-decor/render
       Service name:
           AI Small Decor Enhancement / Micro-Staging

IMPORTANT:
    - Planner-side only.
    - No local inference here.
    - Saves uploads into outputs/{YYYY-MM-DD}/{job_id}/
    - Writes meta.json.
    - Dispatches to GPU worker through app.clients.gpu_client.
    - GPU worker executes app/gpu/powerpaint.py.

EXPLICITLY NOT INCLUDED:
    - Furniture staging
    - Product staging
    - Chair / sofa / bed insertion
    - Reference-guided IP-Adapter workflows

Those belong to future Option A:
    Reference-Guided Furniture / Product Staging
    using IP-Adapter + SDXL 1.0 after separate validation.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import uuid
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.clients.gpu_client import (
    dispatch_powerpaint_object_removal,
    dispatch_powerpaint_small_decor_insert,
)

router = APIRouter(
    prefix="/api/powerpaint",
    tags=["PowerPaint - AI Interior Cleanup & Small Decor Enhancement"],
)


# ---------------------------------------------------------------------------
# Locked PowerPaint service identities
# ---------------------------------------------------------------------------

JOB_TYPE_OBJECT_REMOVAL = "powerpaint_object_removal"
PIPELINE_KEY_OBJECT_REMOVAL = "powerpaint::object_removal"

JOB_TYPE_SMALL_DECOR = "powerpaint_small_decor_insert"
PIPELINE_KEY_SMALL_DECOR = "powerpaint::small_decor_insert"

ENGINE = "PowerPaint-v2-1"
ENGINE_FAMILY = "powerpaint"

ALLOWED_CT = {"image/png", "image/jpeg", "image/jpg"}


# ---------------------------------------------------------------------------
# Defaults based on successful sandbox validation
# ---------------------------------------------------------------------------

DEFAULT_OBJECT_REMOVAL_PROMPT = "empty rug, clean floor"
DEFAULT_OBJECT_REMOVAL_NEGATIVE = "table, bowl, decor, object, furniture, artifact, blurry"

DEFAULT_SMALL_DECOR_PROMPT = (
    "small bronze decorative bowl on a low tray, luxury staged decor, "
    "tabletop scale, realistic contact shadow, subtle reflection"
)
DEFAULT_SMALL_DECOR_NEGATIVE = (
    "chair, sofa, table, large furniture, oversized object, floating object, "
    "wrong scale, blurry, low quality, distorted, warped, heavy reflection"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_iso() -> str:
    return datetime.datetime.utcnow().isoformat()


def _today_utc_str() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


def _create_job_folder(base_outputs_dir: str = "outputs", job_type: str = "powerpaint") -> str:
    """
    Create outputs/{YYYY-MM-DD}/{job_id}/ using repo-relative outputs path.

    This matches the rest of the planner routing style and keeps /outputs/... URLs stable.
    """
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


def _parse_job_path(job_folder: str) -> Tuple[Optional[str], Optional[str]]:
    parts = os.path.normpath(job_folder).split(os.sep)
    if len(parts) < 3:
        return None, None
    return parts[-2], parts[-1]


def _outputs_public_urls(job_folder: str) -> Dict[str, Optional[str]]:
    date_str, job_id = _parse_job_path(job_folder)
    if not date_str or not job_id:
        return {
            "input_url": None,
            "mask_url": None,
            "output_url": None,
            "meta_url": None,
        }

    base = f"/outputs/{date_str}/{job_id}"
    return {
        "input_url": f"{base}/input.png",
        "mask_url": f"{base}/mask.png",
        "output_url": f"{base}/output.png",
        "meta_url": f"{base}/meta.json",
    }


def _ensure_png_jpg(upload: UploadFile, label: str) -> None:
    ct = (getattr(upload, "content_type", "") or "").lower().strip()
    name = (upload.filename or "").lower().strip()

    ok_ct = (not ct) or (ct in ALLOWED_CT)
    ok_ext = (not name) or name.endswith((".png", ".jpg", ".jpeg"))

    if not (ok_ct and ok_ext):
        raise HTTPException(status_code=400, detail=f"{label} must be PNG or JPG")


async def _save_upload_stream(upload: UploadFile, dst_path: str) -> None:
    try:
        try:
            upload.file.seek(0)
        except Exception:
            pass

        with open(dst_path, "wb") as out:
            shutil.copyfileobj(upload.file, out)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed saving upload '{upload.filename}': {exc}") from exc


def _write_meta(job_folder: str, meta: Dict[str, Any]) -> str:
    meta_path = os.path.join(job_folder, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    return meta_path


def _final_seed(seed: Optional[int]) -> int:
    if seed is not None:
        return int(seed)
    return int(uuid.uuid4().int % 1_000_000_000)


def _sanitize_prompt(value: Optional[str], default: str) -> str:
    txt = (value or "").strip()
    return txt if txt else default


def _build_common_response(
    *,
    status: str,
    message: str,
    job_id: str,
    job_folder: str,
    meta_path: str,
    public_urls: Dict[str, Optional[str]],
    gpu_response: Dict[str, Any],
    service_name: str,
    job_type: str,
    pipeline_key: str,
) -> Dict[str, Any]:
    return {
        "status": status,
        "message": message,
        "service_name": service_name,
        "job_type": job_type,
        "pipeline_key": pipeline_key,
        "job_id": job_id,
        "job_folder": job_folder,
        "meta_path": meta_path,
        "public_urls": public_urls,
        "output_image": public_urls.get("output_url"),
        "gpu_response": gpu_response,
    }


# ---------------------------------------------------------------------------
# 1) AI Object Removal
# ---------------------------------------------------------------------------

@router.post("/object-removal/render")
async def powerpaint_object_removal_render(
    image: UploadFile = File(..., description="Base image to clean up / edit. PNG or JPG."),
    mask: UploadFile = File(..., description="Mask image. White area is removed/reconstructed."),
    prompt: Optional[str] = Form(None, description="Optional short cleanup prompt."),
    negative_prompt: Optional[str] = Form(None, description="Optional negative prompt."),
    seed: Optional[int] = Form(None),
    steps: int = Form(30, ge=10, le=60),
    guidance_scale: float = Form(6.5, ge=1.0, le=12.0),
    fitting_degree: float = Form(1.0, ge=0.0, le=1.0),
) -> Dict[str, Any]:
    """
    AI Object Removal.

    Best suited for:
        - removing unwanted decor
        - removing small objects
        - removing clutter
        - cleaning visual distractions
        - reconstructing local floor/rug/wall/background regions

    Not intended for:
        - full furniture replacement
        - large structural changes
        - reference-guided product staging
    """
    if not image.filename:
        raise HTTPException(status_code=400, detail="image has no filename.")
    if not mask.filename:
        raise HTTPException(status_code=400, detail="mask has no filename.")

    _ensure_png_jpg(image, "image")
    _ensure_png_jpg(mask, "mask")

    job_folder = _create_job_folder(job_type=JOB_TYPE_OBJECT_REMOVAL)
    job_id = os.path.basename(job_folder)
    created_at = _utc_iso()
    final_seed = _final_seed(seed)

    await _save_upload_stream(image, os.path.join(job_folder, "input.png"))
    await _save_upload_stream(mask, os.path.join(job_folder, "mask.png"))

    final_prompt = _sanitize_prompt(prompt, DEFAULT_OBJECT_REMOVAL_PROMPT)
    final_negative = _sanitize_prompt(negative_prompt, DEFAULT_OBJECT_REMOVAL_NEGATIVE)

    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": created_at,
        "updated_at": created_at,
        "type": JOB_TYPE_OBJECT_REMOVAL,
        "job_type": JOB_TYPE_OBJECT_REMOVAL,
        "pipeline_key": PIPELINE_KEY_OBJECT_REMOVAL,
        "status": "planned",
        "mode_runtime": "gpu-dispatch",
        "service_family": "AI Interior Cleanup & Small Decor Enhancement",
        "service_name": "AI Object Removal",
        "engine_family": ENGINE_FAMILY,
        "engine": ENGINE,
        "task": "object-removal",
        "prompt": final_prompt,
        "negative_prompt": final_negative,
        "seed": final_seed,
        "steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "fitting_degree": float(fitting_degree),
        "files": {
            "input_image": "input.png",
            "mask_image": "mask.png",
        },
        "outputs": {
            "output_image": "output.png",
        },
        "scope_guardrails": {
            "supports": [
                "object_removal",
                "visual_cleanup",
                "small_clutter_removal",
                "local_background_reconstruction",
            ],
            "excludes": [
                "furniture_staging",
                "product_staging",
                "reference_guided_insertion",
                "large_furniture_replacement",
            ],
        },
        "legal_notice_family": "PowerPaint / PowerPaint-v2-1",
    }

    meta_path = _write_meta(job_folder, meta)
    public_urls = _outputs_public_urls(job_folder)

    ok, gpu_resp = dispatch_powerpaint_object_removal(job_folder=job_folder, meta=meta)

    if not ok:
        try:
            meta["status"] = "gpu_error"
            meta["gpu_error"] = gpu_resp
            meta["updated_at"] = _utc_iso()
            _write_meta(job_folder, meta)
        except Exception:
            pass

        return _build_common_response(
            status="gpu_error",
            message="AI Object Removal job created, but GPU worker dispatch failed.",
            job_id=job_id,
            job_folder=job_folder,
            meta_path=meta_path,
            public_urls=public_urls,
            gpu_response=gpu_resp,
            service_name="AI Object Removal",
            job_type=JOB_TYPE_OBJECT_REMOVAL,
            pipeline_key=PIPELINE_KEY_OBJECT_REMOVAL,
        )

    try:
        meta["status"] = "dispatched"
        meta["gpu_response"] = gpu_resp
        meta["updated_at"] = _utc_iso()
        _write_meta(job_folder, meta)
    except Exception:
        pass

    return _build_common_response(
        status="dispatched",
        message="AI Object Removal dispatched to GPU worker.",
        job_id=job_id,
        job_folder=job_folder,
        meta_path=meta_path,
        public_urls=public_urls,
        gpu_response=gpu_resp,
        service_name="AI Object Removal",
        job_type=JOB_TYPE_OBJECT_REMOVAL,
        pipeline_key=PIPELINE_KEY_OBJECT_REMOVAL,
    )


# ---------------------------------------------------------------------------
# 2) AI Small Decor Enhancement / Micro-Staging
# ---------------------------------------------------------------------------

@router.post("/small-decor/render")
async def powerpaint_small_decor_render(
    image: UploadFile = File(..., description="Base image to enhance. PNG or JPG."),
    mask: UploadFile = File(..., description="Mask image. White area receives small decor."),
    prompt: Optional[str] = Form(None, description="Short prompt for the small decor object."),
    negative_prompt: Optional[str] = Form(None, description="Optional negative prompt."),
    seed: Optional[int] = Form(None),
    steps: int = Form(30, ge=10, le=60),
    guidance_scale: float = Form(6.5, ge=1.0, le=12.0),
    fitting_degree: float = Form(0.55, ge=0.0, le=1.0),
) -> Dict[str, Any]:
    """
    AI Small Decor Enhancement / Micro-Staging.

    Best suited for:
        - small decor bowls
        - trays
        - vases
        - small sculptural accessories
        - subtle styling details
        - micro-staging

    Not intended for:
        - full furniture insertion
        - chairs / sofas / beds
        - exact product placement
        - reference-guided furniture staging
    """
    if not image.filename:
        raise HTTPException(status_code=400, detail="image has no filename.")
    if not mask.filename:
        raise HTTPException(status_code=400, detail="mask has no filename.")

    _ensure_png_jpg(image, "image")
    _ensure_png_jpg(mask, "mask")

    job_folder = _create_job_folder(job_type=JOB_TYPE_SMALL_DECOR)
    job_id = os.path.basename(job_folder)
    created_at = _utc_iso()
    final_seed = _final_seed(seed)

    await _save_upload_stream(image, os.path.join(job_folder, "input.png"))
    await _save_upload_stream(mask, os.path.join(job_folder, "mask.png"))

    final_prompt = _sanitize_prompt(prompt, DEFAULT_SMALL_DECOR_PROMPT)
    final_negative = _sanitize_prompt(negative_prompt, DEFAULT_SMALL_DECOR_NEGATIVE)

    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": created_at,
        "updated_at": created_at,
        "type": JOB_TYPE_SMALL_DECOR,
        "job_type": JOB_TYPE_SMALL_DECOR,
        "pipeline_key": PIPELINE_KEY_SMALL_DECOR,
        "status": "planned",
        "mode_runtime": "gpu-dispatch",
        "service_family": "AI Interior Cleanup & Small Decor Enhancement",
        "service_name": "AI Small Decor Enhancement / Micro-Staging",
        "engine_family": ENGINE_FAMILY,
        "engine": ENGINE,
        "task": "text-guided",
        "prompt": final_prompt,
        "negative_prompt": final_negative,
        "seed": final_seed,
        "steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "fitting_degree": float(fitting_degree),
        "files": {
            "input_image": "input.png",
            "mask_image": "mask.png",
        },
        "outputs": {
            "output_image": "output.png",
        },
        "scope_guardrails": {
            "supports": [
                "small_decor_insert",
                "micro_staging",
                "small_accessory_addition",
                "subtle_interior_styling",
            ],
            "excludes": [
                "furniture_staging",
                "product_staging",
                "chair_insert",
                "sofa_insert",
                "bed_insert",
                "large_table_insert",
                "reference_guided_insertion",
            ],
        },
        "legal_notice_family": "PowerPaint / PowerPaint-v2-1",
    }

    meta_path = _write_meta(job_folder, meta)
    public_urls = _outputs_public_urls(job_folder)

    ok, gpu_resp = dispatch_powerpaint_small_decor_insert(job_folder=job_folder, meta=meta)

    if not ok:
        try:
            meta["status"] = "gpu_error"
            meta["gpu_error"] = gpu_resp
            meta["updated_at"] = _utc_iso()
            _write_meta(job_folder, meta)
        except Exception:
            pass

        return _build_common_response(
            status="gpu_error",
            message="AI Small Decor Enhancement job created, but GPU worker dispatch failed.",
            job_id=job_id,
            job_folder=job_folder,
            meta_path=meta_path,
            public_urls=public_urls,
            gpu_response=gpu_resp,
            service_name="AI Small Decor Enhancement / Micro-Staging",
            job_type=JOB_TYPE_SMALL_DECOR,
            pipeline_key=PIPELINE_KEY_SMALL_DECOR,
        )

    try:
        meta["status"] = "dispatched"
        meta["gpu_response"] = gpu_resp
        meta["updated_at"] = _utc_iso()
        _write_meta(job_folder, meta)
    except Exception:
        pass

    return _build_common_response(
        status="dispatched",
        message="AI Small Decor Enhancement / Micro-Staging dispatched to GPU worker.",
        job_id=job_id,
        job_folder=job_folder,
        meta_path=meta_path,
        public_urls=public_urls,
        gpu_response=gpu_resp,
        service_name="AI Small Decor Enhancement / Micro-Staging",
        job_type=JOB_TYPE_SMALL_DECOR,
        pipeline_key=PIPELINE_KEY_SMALL_DECOR,
    )