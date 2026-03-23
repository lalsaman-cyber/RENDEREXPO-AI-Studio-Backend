# app/routers/img2img.py
"""
RENDEREXPO AI STUDIO - SD3.5 Img2Img Router (Planner)

GOAL:
- Client uploads ONE image + prompt
- Planner creates job folder, saves input.png, writes meta.json
- Planner dispatches to GPU worker through app.clients.gpu_client
- GPU worker performs REAL img2img / inpaint execution and writes:
    - output.png
    - updated meta.json

Also supports:
- from-job: re-edit an existing image from a prior job (JSON)
- inpaint: upload image + mask to edit only selected area (multipart)

LOCKED ARCHITECTURE:
- Planner = port 8012
- GPU worker = port 8002
- Root = /workspace-data/RENDEREXPO-AI-Studio-Backend

IMPORTANT:
- Presets must be applied for all categories/shots:
    * locked steps + CFG
    * LyCORIS PRO 2.1 multiplier + path
    * GEO multiplier + path
    * resolution
    * optional upscale defaults
- Img2img strength MUST be preserved.
- Do NOT globally force denoise/strength to 0.0 here.
- Planner writes meta; GPU runtime executes meta.

RENDEREXPO img2img ratio rule:
- Output should match the INPUT ratio unless explicitly specified otherwise.
- Because presets currently inject default resolution, planner marks whether
  dimensions were explicitly supplied by caller or just inherited from defaults.
- Runtime uses that signal to auto-follow source aspect when appropriate.

CRITICAL SKETCH RULE:
- This router is NOT the primary route for architectural sketch-to-render.
- Sketch mode must use the dedicated sketch planner/router and dispatch to:
    sd35_sketch_controlnet
- Do NOT try to solve sketch structure through plain img2img here.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import uuid
from typing import Any, Dict, Literal, Optional, Tuple

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.clients.gpu_client import dispatch_sd35_img2img, dispatch_sd35_inpaint
from app.core.lora_registry import get_lora_profile, get_refiner_profile
from app.presets_sd35 import apply_preset_to_meta

router = APIRouter(prefix="/api/sd35", tags=["SD3.5 Img2Img (Planner)"])

Category = Literal["urban", "suburban", "interior", "wide_hero"]
Shot = Literal["wide", "close"]

ALLOWED_CT = {"image/png", "image/jpeg", "image/jpg"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _today_utc_str() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


def _create_job_folder(base_outputs_dir: str = "outputs", job_type: str = "sd35_img2img") -> str:
    """
    Create outputs/{YYYY-MM-DD}/{job_id}/ and return its path.
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


def _outputs_public_urls(job_folder: str, has_mask: bool = False) -> Dict[str, Optional[str]]:
    """
    Stable URLs assuming planner mounts outputs/ at /outputs.
    """
    date_str, job_id = _parse_job_path(job_folder)
    if not date_str or not job_id:
        return {
            "image_url": None,
            "meta_url": None,
            "input_url": None,
            "mask_url": None,
        }

    base = f"/outputs/{date_str}/{job_id}"
    return {
        "image_url": f"{base}/output.png",
        "meta_url": f"{base}/meta.json",
        "input_url": f"{base}/input.png",
        "mask_url": f"{base}/mask.png" if has_mask else None,
    }


def _validate_upload_is_png_jpg(upload: UploadFile, label: str) -> None:
    ct = (getattr(upload, "content_type", "") or "").lower().strip()
    if ct and ct not in ALLOWED_CT:
        raise HTTPException(status_code=400, detail=f"{label} must be PNG or JPG")


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


def _resolve_optional_profiles(
    lora_profile: Optional[str],
    refiner_profile: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Optional / legacy metadata only.
    These must not override locked preset multipliers.
    """
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

    return resolved_lora_profile, resolved_refiner_profile


def _get_image_size(image_path: str) -> Tuple[int, int]:
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"PIL is required to inspect input image size: {exc}") from exc

    try:
        with Image.open(image_path) as im:
            return int(im.width), int(im.height)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed reading image size: {exc}") from exc


def _augment_meta_with_input_geometry(
    meta: Dict[str, Any],
    *,
    input_width: int,
    input_height: int,
    explicit_dimensions: bool,
) -> None:
    """
    Record the original input image geometry and whether planner dimensions were
    explicitly requested by caller vs injected by presets/defaults.

    Runtime should use:
    - explicit_dimensions=False  -> follow input ratio unless non-default explicit size exists
    - explicit_dimensions=True   -> respect target width/height even if aspect differs
    """
    meta["input_width"] = int(input_width)
    meta["input_height"] = int(input_height)
    meta["input_aspect_ratio"] = float(input_width) / float(input_height) if input_height else None
    meta["preserve_input_aspect_ratio"] = not bool(explicit_dimensions)
    meta["explicit_dimensions"] = bool(explicit_dimensions)


def _base_meta(
    *,
    job_id: str,
    job_type: str,
    prompt: str,
    negative_prompt: Optional[str],
    seed: int,
    category: str,
    shot: str,
    strength: Optional[float] = None,
    input_image_name: Optional[str] = None,
    mask_image_name: Optional[str] = None,
    style_preset: Optional[str] = None,
    material_preset: Optional[str] = None,
    lighting_preset: Optional[str] = None,
    lora_profile: Optional[str] = None,
    lora_profile_resolved: Optional[Dict[str, Any]] = None,
    refiner_profile: Optional[str] = None,
    refiner_profile_resolved: Optional[Dict[str, Any]] = None,
    source_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "type": job_type,
        "model_name": "sd35_large_pro_v2_1",
        "engine": "sd35_large_pro_v2_1",
        "status": "queued",
        "mode_runtime": "gpu-dispatch",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "planned_output_image": "output.png",
        "category": category,
        "shot": shot,
        "style_preset": style_preset,
        "material_preset": material_preset,
        "lighting_preset": lighting_preset,
        "optional_profiles": {
            "lora_profile": lora_profile,
            "lora_profile_resolved": lora_profile_resolved,
            "refiner_profile": refiner_profile,
            "refiner_profile_resolved": refiner_profile_resolved,
        },
        "pipeline_key": f"sd35::{'sd35_inpaint' if job_type == 'inpaint' else 'sd35_img2img'}",
    }

    if strength is not None:
        meta["strength"] = float(strength)

    inputs: Dict[str, Any] = {}
    if input_image_name is not None:
        inputs["input_image"] = input_image_name
    if mask_image_name is not None:
        inputs["mask_image"] = mask_image_name
    if source_info:
        inputs.update(source_info)
    if inputs:
        meta["inputs"] = inputs

    return meta


# ---------------------------------------------------------------------------
# JSON model for from-job re-edit
# ---------------------------------------------------------------------------

class Img2ImgFromJobRequest(BaseModel):
    source_job_id: str = Field(..., description="Job ID (folder name) under outputs/*/<job_id>/")
    source_image_path: str = Field(default="output.png", description="Relative path within the source job folder")

    prompt: str = Field(..., description="Prompt to guide edit/materialization.")
    negative_prompt: Optional[str] = Field(default=None)

    strength: float = Field(default=0.55, ge=0.0, le=1.0, description="How much to change vs preserve input (0..1).")

    category: Category = Field(..., description="Preset category.")
    shot: Shot = Field(..., description="Preset shot.")

    upscale_2x: Optional[bool] = Field(default=None)
    seed: Optional[int] = Field(default=None)

    style_preset: Optional[str] = None
    material_preset: Optional[str] = None
    lighting_preset: Optional[str] = None

    lora_profile: Optional[str] = None
    refiner_profile: Optional[str] = None

    # Optional explicit override path for future callers.
    # When omitted, planner/runtime should preserve the source aspect ratio.
    width: Optional[int] = Field(default=None, ge=64)
    height: Optional[int] = Field(default=None, ge=64)


# ---------------------------------------------------------------------------
# 1) REAL Img2Img: upload image + prompt (multipart)
# ---------------------------------------------------------------------------

@router.post("/img2img/render")
async def sd35_img2img_render(
    image: UploadFile = File(..., description="Input image (render, wireframe, B/W, previous output, etc.)"),
    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form(None),
    strength: float = Form(0.22, description="0..1 (lower preserves more)"),
    category: Category = Form(...),
    shot: Shot = Form(...),
    upscale_2x: Optional[bool] = Form(None),
    seed: Optional[int] = Form(None),
    style_preset: Optional[str] = Form(None),
    material_preset: Optional[str] = Form(None),
    lighting_preset: Optional[str] = Form(None),
    lora_profile: Optional[str] = Form(None),
    refiner_profile: Optional[str] = Form(None),
    width: Optional[int] = Form(None, description="Optional explicit output width. If omitted, follow input ratio."),
    height: Optional[int] = Form(None, description="Optional explicit output height. If omitted, follow input ratio."),
) -> Dict[str, Any]:
    _validate_strength(float(strength))
    _validate_upload_is_png_jpg(image, "image")

    prompt_clean = (prompt or "").strip()
    if not prompt_clean:
        raise HTTPException(status_code=400, detail="prompt is required")

    explicit_dimensions = (width is not None) or (height is not None)
    if explicit_dimensions and (width is None or height is None):
        raise HTTPException(status_code=400, detail="width and height must both be provided when overriding aspect ratio")

    resolved_lora_profile, resolved_refiner_profile = _resolve_optional_profiles(
        lora_profile=lora_profile,
        refiner_profile=refiner_profile,
    )

    job_folder = _create_job_folder(job_type="sd35_img2img")
    job_id = os.path.basename(job_folder)

    input_path = os.path.join(job_folder, "input.png")
    await _save_upload_stream(image, input_path)
    input_width, input_height = _get_image_size(input_path)

    final_seed = seed if seed is not None else int(uuid.uuid4().int % 1_000_000_000)

    meta = _base_meta(
        job_id=job_id,
        job_type="img2img",
        prompt=prompt_clean,
        negative_prompt=negative_prompt,
        seed=int(final_seed),
        category=category,
        shot=shot,
        strength=float(strength),
        input_image_name="input.png",
        style_preset=style_preset,
        material_preset=material_preset,
        lighting_preset=lighting_preset,
        lora_profile=lora_profile,
        lora_profile_resolved=resolved_lora_profile,
        refiner_profile=refiner_profile,
        refiner_profile_resolved=resolved_refiner_profile,
        source_info={
            "content_type": getattr(image, "content_type", None),
            "source": "upload",
        },
    )

    try:
        apply_preset_to_meta(meta, category=category, shot=shot, upscale_2x=upscale_2x)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Preset error: {exc}") from exc

    # Preserve true img2img strength explicitly after preset application.
    meta["strength"] = float(strength)

    # Explicit size wins over preset defaults.
    if explicit_dimensions:
        meta["width"] = int(width)
        meta["height"] = int(height)

    _augment_meta_with_input_geometry(
        meta,
        input_width=input_width,
        input_height=input_height,
        explicit_dimensions=explicit_dimensions,
    )

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
        "message": "Img2Img job dispatched to GPU worker.",
        "job_folder": job_folder,
        "meta_path": meta_path,
        "public_urls": public_urls,
        "expected_output": os.path.join(job_folder, "output.png"),
        "gpu_response": gpu_resp,
        "preset_applied": meta.get("preset", {}),
        "upscale_enabled": bool(meta.get("upscale", {}).get("enabled", False)),
        "input_geometry": {
            "width": input_width,
            "height": input_height,
            "preserve_input_aspect_ratio": meta.get("preserve_input_aspect_ratio"),
            "explicit_dimensions": meta.get("explicit_dimensions"),
        },
    }


# ---------------------------------------------------------------------------
# 2) REAL Img2Img: from-job (JSON) re-edit existing platform outputs
# ---------------------------------------------------------------------------

@router.post("/img2img/from-job/render")
async def sd35_img2img_from_job(req: Img2ImgFromJobRequest) -> Dict[str, Any]:
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

    explicit_dimensions = (req.width is not None) or (req.height is not None)
    if explicit_dimensions and (req.width is None or req.height is None):
        raise HTTPException(status_code=400, detail="width and height must both be provided when overriding aspect ratio")

    resolved_lora_profile, resolved_refiner_profile = _resolve_optional_profiles(
        lora_profile=req.lora_profile,
        refiner_profile=req.refiner_profile,
    )

    job_folder = _create_job_folder(job_type="sd35_img2img")
    job_id = os.path.basename(job_folder)

    dst_img = os.path.join(job_folder, "input.png")
    try:
        shutil.copyfile(src_img, dst_img)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed copying source image: {exc}") from exc

    input_width, input_height = _get_image_size(dst_img)
    final_seed = req.seed if req.seed is not None else int(uuid.uuid4().int % 1_000_000_000)

    meta = _base_meta(
        job_id=job_id,
        job_type="img2img",
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        seed=int(final_seed),
        category=req.category,
        shot=req.shot,
        strength=float(req.strength),
        input_image_name="input.png",
        style_preset=req.style_preset,
        material_preset=req.material_preset,
        lighting_preset=req.lighting_preset,
        lora_profile=req.lora_profile,
        lora_profile_resolved=resolved_lora_profile,
        refiner_profile=req.refiner_profile,
        refiner_profile_resolved=resolved_refiner_profile,
        source_info={
            "source": "job_reference",
            "source_job_id": source_job_id,
            "source_image_path": safe_rel,
        },
    )

    try:
        apply_preset_to_meta(meta, category=req.category, shot=req.shot, upscale_2x=req.upscale_2x)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Preset error: {exc}") from exc

    meta["strength"] = float(req.strength)

    # Explicit size wins over preset defaults.
    if explicit_dimensions:
        meta["width"] = int(req.width)
        meta["height"] = int(req.height)

    _augment_meta_with_input_geometry(
        meta,
        input_width=input_width,
        input_height=input_height,
        explicit_dimensions=explicit_dimensions,
    )

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
        "message": "Img2Img (from-job) dispatched to GPU worker.",
        "job_folder": job_folder,
        "meta_path": meta_path,
        "public_urls": public_urls,
        "gpu_response": gpu_resp,
        "preset_applied": meta.get("preset", {}),
        "input_geometry": {
            "width": input_width,
            "height": input_height,
            "preserve_input_aspect_ratio": meta.get("preserve_input_aspect_ratio"),
            "explicit_dimensions": meta.get("explicit_dimensions"),
        },
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
    strength: float = Form(0.22, description="0..1 (lower preserves more)"),
    category: Category = Form(...),
    shot: Shot = Form(...),
    upscale_2x: Optional[bool] = Form(None),
    seed: Optional[int] = Form(None),
    width: Optional[int] = Form(None, description="Optional explicit output width. If omitted, follow input ratio."),
    height: Optional[int] = Form(None, description="Optional explicit output height. If omitted, follow input ratio."),
) -> Dict[str, Any]:
    _validate_strength(float(strength))
    _validate_upload_is_png_jpg(image, "image")
    _validate_upload_is_png_jpg(mask, "mask")

    prompt_clean = (prompt or "").strip()
    if not prompt_clean:
        raise HTTPException(status_code=400, detail="prompt is required")

    explicit_dimensions = (width is not None) or (height is not None)
    if explicit_dimensions and (width is None or height is None):
        raise HTTPException(status_code=400, detail="width and height must both be provided when overriding aspect ratio")

    job_folder = _create_job_folder(job_type="sd35_inpaint")
    job_id = os.path.basename(job_folder)

    input_path = os.path.join(job_folder, "input.png")
    mask_path = os.path.join(job_folder, "mask.png")
    await _save_upload_stream(image, input_path)
    await _save_upload_stream(mask, mask_path)
    input_width, input_height = _get_image_size(input_path)

    final_seed = seed if seed is not None else int(uuid.uuid4().int % 1_000_000_000)

    meta = _base_meta(
        job_id=job_id,
        job_type="inpaint",
        prompt=prompt_clean,
        negative_prompt=negative_prompt,
        seed=int(final_seed),
        category=category,
        shot=shot,
        strength=float(strength),
        input_image_name="input.png",
        mask_image_name="mask.png",
        source_info={"source": "upload"},
    )

    try:
        apply_preset_to_meta(meta, category=category, shot=shot, upscale_2x=upscale_2x)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Preset error: {exc}") from exc

    meta["strength"] = float(strength)

    if explicit_dimensions:
        meta["width"] = int(width)
        meta["height"] = int(height)

    _augment_meta_with_input_geometry(
        meta,
        input_width=input_width,
        input_height=input_height,
        explicit_dimensions=explicit_dimensions,
    )

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
        "message": "Inpaint job dispatched to GPU worker.",
        "job_folder": job_folder,
        "meta_path": meta_path,
        "public_urls": public_urls,
        "gpu_response": gpu_resp,
        "preset_applied": meta.get("preset", {}),
        "input_geometry": {
            "width": input_width,
            "height": input_height,
            "preserve_input_aspect_ratio": meta.get("preserve_input_aspect_ratio"),
            "explicit_dimensions": meta.get("explicit_dimensions"),
        },
    }