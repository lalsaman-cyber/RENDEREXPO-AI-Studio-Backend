# app/routers/img2img.py
"""
RENDEREXPO AI STUDIO - SD3.5 Img2Img Router (REAL via GPU dispatch)

GOAL:
- Client uploads ONE image + prompt
- Create job folder, save input.png, write meta.json
- DISPATCH to GPU worker (HMAC-signed Option A via app.clients.gpu_client)
- GPU worker writes:
    - output.png (primary)
    - meta.json updated

Also supports:
- from-job: re-edit an existing image from a prior job (JSON)
- inpaint: upload image + mask to edit only selected area (multipart)

CRITICAL (Doc 18 lock):
- Presets must be applied for ALL categories/shots:
    * locked steps + CFG
    * LyCORIS PRO 2.1 multiplier + path
    * GEO multiplier + path
    * resolution
    * NO denoise anywhere (denoise=0.0)
- Upscale is OPTIONAL:
    * preset default, OR
    * overridden per request via upscale_2x true/false.

NOTES:
- "strength" is allowed and is NOT diffusion denoise. It controls how much we preserve the input image.
- Diffusion denoise remains hard-locked to 0.0.
"""

from __future__ import annotations

import os
import uuid
import json
import shutil
import datetime
from typing import Optional, Dict, Any, Literal, Tuple

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field

from app.core.lora_registry import get_lora_profile, get_refiner_profile
from app.presets_sd35 import apply_preset_to_meta

# GPU dispatch (same pattern used by text2img)
from app.clients.gpu_client import dispatch_sd35_img2img, dispatch_sd35_inpaint

router = APIRouter(prefix="/api/sd35", tags=["SD3.5 Img2Img (REAL)"])

Category = Literal["urban", "suburban", "interior", "wide_hero"]
Shot = Literal["wide", "close"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALLOWED_CT = {"image/png", "image/jpeg", "image/jpg"}

def _today_utc_str() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


def _create_job_folder(base_outputs_dir: str = "outputs") -> str:
    """
    Create outputs/{YYYY-MM-DD}/{job_id}/ and return its path.
    """
    today = _today_utc_str()
    job_id = uuid.uuid4().hex
    folder = os.path.join(base_outputs_dir, today, job_id)
    os.makedirs(folder, exist_ok=True)

    # marker file
    try:
        with open(os.path.join(folder, "job_type.txt"), "w", encoding="utf-8") as f:
            f.write("sd35_img2img")
    except Exception:
        pass

    return folder


def _parse_job_path(job_folder: str) -> Tuple[Optional[str], Optional[str]]:
    parts = os.path.normpath(job_folder).split(os.sep)
    if len(parts) < 3:
        return None, None
    return parts[-2], parts[-1]


def _outputs_public_urls(job_folder: str, has_mask: bool = False) -> Dict[str, Optional[str]]:
    """
    Stable URLs assuming FastAPI mounts outputs/ at /outputs.
    """
    date_str, job_id = _parse_job_path(job_folder)
    if not date_str or not job_id:
        return {"image_url": None, "meta_url": None, "input_url": None, "mask_url": None}

    base = f"/outputs/{date_str}/{job_id}"
    return {
        "image_url": f"{base}/output.png",
        "meta_url": f"{base}/meta.json",
        "input_url": f"{base}/input.png",
        "mask_url": (f"{base}/mask.png" if has_mask else None),
    }


def _validate_upload_is_png_jpg(upload: UploadFile, label: str) -> None:
    ct = (getattr(upload, "content_type", "") or "").lower().strip()
    # content-type isn't perfect security, but it's a clean UX gate
    if ct and ct not in ALLOWED_CT:
        raise HTTPException(status_code=400, detail=f"{label} must be PNG or JPG")


async def _save_upload_stream(upload: UploadFile, dst_path: str) -> None:
    """Save UploadFile without reading everything into RAM."""
    try:
        try:
            upload.file.seek(0)
        except Exception:
            pass

        with open(dst_path, "wb") as out:
            shutil.copyfileobj(upload.file, out)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to save upload '{upload.filename}': {exc}") from exc


def _write_meta(job_folder: str, meta: Dict[str, Any]) -> str:
    meta_path = os.path.join(job_folder, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
    return meta_path


def _find_job_folder_by_job_id(job_id: str) -> Optional[str]:
    """
    Scan outputs/*/<job_id> and return first match.
    """
    outputs_dir = "outputs"
    if not os.path.isdir(outputs_dir):
        return None

    for date_dir in os.listdir(outputs_dir):
        date_path = os.path.join(outputs_dir, date_dir)
        if not os.path.isdir(date_path):
            continue
        candidate = os.path.join(date_path, job_id)
        if os.path.isdir(candidate):
            return candidate
    return None


def _safe_relpath(rel: str) -> str:
    """
    Prevent path traversal. Return a safe relative path (no ../).
    """
    rel = (rel or "").strip().replace("\\", "/")
    rel = rel.lstrip("/")
    norm = os.path.normpath(rel).replace("\\", "/")
    if not norm or norm == ".":
        return "output.png"
    if norm.startswith("../") or norm == ".." or "/../" in f"/{norm}/":
        raise HTTPException(status_code=400, detail="Invalid source_image_path.")
    return norm


def _validate_strength(strength: float) -> None:
    if not (0.0 <= float(strength) <= 1.0):
        raise HTTPException(status_code=400, detail="strength must be between 0.0 and 1.0")


# ---------------------------------------------------------------------------
# JSON model for from-job re-edit
# ---------------------------------------------------------------------------

class Img2ImgFromJobRequest(BaseModel):
    source_job_id: str = Field(..., description="Job ID (folder name) under outputs/*/<job_id>/")
    source_image_path: str = Field(default="output.png", description="Relative path within the source job folder")

    prompt: str = Field(..., description="Prompt to guide edit/materialization.")
    negative_prompt: Optional[str] = Field(default=None)

    strength: float = Field(default=0.55, ge=0.0, le=1.0, description="How much to change vs preserve input (0..1).")

    category: Category = Field(..., description="Doc 18 preset category.")
    shot: Shot = Field(..., description="Doc 18 preset shot.")

    upscale_2x: Optional[bool] = Field(default=None)
    seed: Optional[int] = Field(default=None)

    style_preset: Optional[str] = None
    material_preset: Optional[str] = None
    lighting_preset: Optional[str] = None

    lora_profile: Optional[str] = None
    refiner_profile: Optional[str] = None


# ---------------------------------------------------------------------------
# 1) REAL Img2Img: upload image + prompt (multipart)
# ---------------------------------------------------------------------------

@router.post("/img2img/render")
async def sd35_img2img_render(
    image: UploadFile = File(..., description="Input image (render, wireframe, B/W, previous output, etc.)"),
    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form(None),

    strength: float = Form(0.55, description="0..1 (lower preserves more)"),

    category: Category = Form(...),
    shot: Shot = Form(...),

    # Optional override (if omitted, preset default is used)
    upscale_2x: Optional[bool] = Form(None),

    seed: Optional[int] = Form(None),

    style_preset: Optional[str] = Form(None),
    material_preset: Optional[str] = Form(None),
    lighting_preset: Optional[str] = Form(None),

    lora_profile: Optional[str] = Form(None),
    refiner_profile: Optional[str] = Form(None),
):
    _validate_strength(float(strength))
    _validate_upload_is_png_jpg(image, "image")

    # Optional profiles stored only (must not override Doc 18 multipliers)
    resolved_lora_profile: Optional[Dict[str, Any]] = None
    if lora_profile:
        resolved_lora_profile = get_lora_profile(lora_profile)
        if resolved_lora_profile is None:
            raise HTTPException(status_code=400, detail=f"Unknown lora_profile: '{lora_profile}'")

    resolved_refiner_profile: Optional[Dict[str, Any]] = None
    if refiner_profile:
        resolved_refiner_profile = get_refiner_profile(refiner_profile)
        if resolved_refiner_profile is None:
            raise HTTPException(status_code=400, detail=f"Unknown refiner_profile: '{refiner_profile}'")

    job_folder = _create_job_folder()
    job_id = os.path.basename(job_folder)

    input_path = os.path.join(job_folder, "input.png")
    await _save_upload_stream(image, input_path)

    final_seed = seed if seed is not None else int(uuid.uuid4().int % 1_000_000_000)

    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": datetime.datetime.utcnow().isoformat(),

        "type": "img2img",
        "model_name": "sd3.5-large-pro-2.1",

        # REAL semantics
        "status": "queued",
        "mode_runtime": "gpu-dispatch",

        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": final_seed,

        "strength": float(strength),

        "inputs": {
            "input_image": "input.png",
            "content_type": getattr(image, "content_type", None),
            "source": "upload",
        },

        "planned_output_image": "output.png",

        "category": category,
        "shot": shot,

        "denoise": 0.0,

        "style_preset": style_preset,
        "material_preset": material_preset,
        "lighting_preset": lighting_preset,

        "optional_profiles": {
            "lora_profile": lora_profile,
            "lora_profile_resolved": resolved_lora_profile,
            "refiner_profile": refiner_profile,
            "refiner_profile_resolved": resolved_refiner_profile,
        },
    }

    try:
        apply_preset_to_meta(meta, category=category, shot=shot, upscale_2x=upscale_2x)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Preset error: {exc}") from exc

    meta["denoise"] = 0.0
    if isinstance(meta.get("upscale"), dict):
        meta["upscale"]["denoise"] = 0.0

    meta_path = _write_meta(job_folder, meta)
    public_urls = _outputs_public_urls(job_folder, has_mask=False)

    ok, gpu_resp = dispatch_sd35_img2img(job_folder=job_folder, meta=meta)

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
        "message": "Img2Img job dispatched to GPU worker (Doc 18 locked presets).",
        "job_folder": job_folder,
        "meta_path": meta_path,
        "public_urls": public_urls,
        "expected_output": os.path.join(job_folder, "output.png"),
        "gpu_response": gpu_resp,
        "preset_applied": meta.get("preset", {}),
        "upscale_enabled": bool(meta.get("upscale", {}).get("enabled", False)),
    }


# ---------------------------------------------------------------------------
# 2) REAL Img2Img: from-job (JSON) re-edit existing platform outputs
# ---------------------------------------------------------------------------

@router.post("/img2img/from-job/render")
async def sd35_img2img_from_job(req: Img2ImgFromJobRequest):
    source_job_id = (req.source_job_id or "").strip()
    if not source_job_id:
        raise HTTPException(status_code=400, detail="source_job_id is required")

    found_folder = _find_job_folder_by_job_id(source_job_id)
    if not found_folder:
        raise HTTPException(status_code=404, detail=f"source job folder not found for job_id={source_job_id}")

    safe_rel = _safe_relpath(req.source_image_path)
    src_img = os.path.join(found_folder, safe_rel)
    if not os.path.isfile(src_img):
        raise HTTPException(status_code=404, detail=f"source image not found: {safe_rel}")

    _validate_strength(float(req.strength))

    resolved_lora_profile: Optional[Dict[str, Any]] = None
    if req.lora_profile:
        resolved_lora_profile = get_lora_profile(req.lora_profile)
        if resolved_lora_profile is None:
            raise HTTPException(status_code=400, detail=f"Unknown lora_profile: '{req.lora_profile}'")

    resolved_refiner_profile: Optional[Dict[str, Any]] = None
    if req.refiner_profile:
        resolved_refiner_profile = get_refiner_profile(req.refiner_profile)
        if resolved_refiner_profile is None:
            raise HTTPException(status_code=400, detail=f"Unknown refiner_profile: '{req.refiner_profile}'")

    job_folder = _create_job_folder()
    job_id = os.path.basename(job_folder)

    dst_img = os.path.join(job_folder, "input.png")
    try:
        shutil.copyfile(src_img, dst_img)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed copying source image: {exc}") from exc

    final_seed = req.seed if req.seed is not None else int(uuid.uuid4().int % 1_000_000_000)

    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": datetime.datetime.utcnow().isoformat(),

        "type": "img2img",
        "model_name": "sd3.5-large-pro-2.1",

        "status": "queued",
        "mode_runtime": "gpu-dispatch",

        "prompt": req.prompt,
        "negative_prompt": req.negative_prompt,
        "seed": final_seed,

        "strength": float(req.strength),

        "inputs": {
            "input_image": "input.png",
            "source": "job_reference",
            "source_job_id": source_job_id,
            "source_image_path": safe_rel,
        },

        "planned_output_image": "output.png",

        "category": req.category,
        "shot": req.shot,

        "denoise": 0.0,

        "style_preset": req.style_preset,
        "material_preset": req.material_preset,
        "lighting_preset": req.lighting_preset,

        "optional_profiles": {
            "lora_profile": req.lora_profile,
            "lora_profile_resolved": resolved_lora_profile,
            "refiner_profile": req.refiner_profile,
            "refiner_profile_resolved": resolved_refiner_profile,
        },
    }

    try:
        apply_preset_to_meta(meta, category=req.category, shot=req.shot, upscale_2x=req.upscale_2x)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Preset error: {exc}") from exc

    meta["denoise"] = 0.0
    if isinstance(meta.get("upscale"), dict):
        meta["upscale"]["denoise"] = 0.0

    meta_path = _write_meta(job_folder, meta)
    public_urls = _outputs_public_urls(job_folder, has_mask=False)

    ok, gpu_resp = dispatch_sd35_img2img(job_folder=job_folder, meta=meta)
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
        "message": "Img2Img (from-job) dispatched to GPU worker (Doc 18 locked presets).",
        "job_folder": job_folder,
        "meta_path": meta_path,
        "public_urls": public_urls,
        "gpu_response": gpu_resp,
        "preset_applied": meta.get("preset", {}),
    }


# ---------------------------------------------------------------------------
# 3) REAL Inpaint: image + mask + prompt (multipart)
# ---------------------------------------------------------------------------

@router.post("/inpaint/render")
async def sd35_inpaint_render(
    image: UploadFile = File(..., description="Input image"),
    mask: UploadFile = File(..., description="Mask image (white = edit, black = keep)"),

    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form(None),

    strength: float = Form(0.55, description="0..1 (lower preserves more)"),

    category: Category = Form(...),
    shot: Shot = Form(...),
    upscale_2x: Optional[bool] = Form(None),

    seed: Optional[int] = Form(None),
):
    _validate_strength(float(strength))
    _validate_upload_is_png_jpg(image, "image")
    _validate_upload_is_png_jpg(mask, "mask")

    job_folder = _create_job_folder()
    job_id = os.path.basename(job_folder)

    input_path = os.path.join(job_folder, "input.png")
    mask_path = os.path.join(job_folder, "mask.png")
    await _save_upload_stream(image, input_path)
    await _save_upload_stream(mask, mask_path)

    final_seed = seed if seed is not None else int(uuid.uuid4().int % 1_000_000_000)

    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": datetime.datetime.utcnow().isoformat(),

        "type": "inpaint",
        "model_name": "sd3.5-large-pro-2.1",

        "status": "queued",
        "mode_runtime": "gpu-dispatch",

        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": final_seed,

        "strength": float(strength),

        "inputs": {
            "input_image": "input.png",
            "mask_image": "mask.png",
            "source": "upload",
        },

        "planned_output_image": "output.png",

        "category": category,
        "shot": shot,

        "denoise": 0.0,
    }

    try:
        apply_preset_to_meta(meta, category=category, shot=shot, upscale_2x=upscale_2x)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Preset error: {exc}") from exc

    meta["denoise"] = 0.0
    if isinstance(meta.get("upscale"), dict):
        meta["upscale"]["denoise"] = 0.0

    meta_path = _write_meta(job_folder, meta)
    public_urls = _outputs_public_urls(job_folder, has_mask=True)

    ok, gpu_resp = dispatch_sd35_inpaint(job_folder=job_folder, meta=meta)
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
        "message": "Inpaint job dispatched to GPU worker (Doc 18 locked presets).",
        "job_folder": job_folder,
        "meta_path": meta_path,
        "public_urls": public_urls,
        "gpu_response": gpu_resp,
        "preset_applied": meta.get("preset", {}),
    }
