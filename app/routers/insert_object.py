# app/routers/insert_object.py
"""
RENDEREXPO AI STUDIO - Insert Object (REAL via GPU dispatch)

GOAL (REAL, Wix-ready):
- Client uploads:
    * product_image (required) — couch, lamp, table, etc.
    * scene_image (required) — existing room / scene image
    * floorplan_image (optional)
    * prompt / placement_hint (optional)
- Create outputs/{YYYY-MM-DD}/{job_id}/
- Save uploaded images
- Write meta.json
- DISPATCH to GPU worker immediately (HMAC-signed Option A via app.clients.gpu_client)
- GPU worker writes (best effort, depending on pipeline availability):
    - product_mask.png
    - product_rgba.png
    - composite_raw.png
    - output.png (final)
    - meta.json updated

Doc 18 rule (critical):
- If ANY SD3.5 stage exists (harmonize), we MUST lock presets:
  steps, CFG, LyCORIS(PRO 2.1) multiplier, GEO multiplier, resolution,
  NO denoise anywhere, upscale optional per request OR preset default.

No skeletons. No planning-only mode.
"""

from __future__ import annotations

import os
import uuid
import json
import datetime
from typing import Optional, Dict, Any, List, Literal, Tuple

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.presets_sd35 import apply_preset_to_meta

# GPU dispatch (you will wire this later in app/clients/gpu_client.py)
from app.clients.gpu_client import dispatch_insert_object  # noqa: F401


router = APIRouter(
    prefix="/api/insert-object",
    tags=["Insert Object (REAL)"],
)

Category = Literal["urban", "suburban", "interior", "wide_hero"]
Shot = Literal["wide", "close"]

ALLOWED_CT = {"image/png", "image/jpeg", "image/jpg"}


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
            f.write("insert_object")
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
            "meta_url": None,
            "product_url": None,
            "scene_url": None,
            "floorplan_url": None,
            "output_url": None,
            "composite_raw_url": None,
        }

    base = f"/outputs/{date_str}/{job_id}"
    return {
        "meta_url": f"{base}/meta.json",
        "product_url": f"{base}/product.png",
        "scene_url": f"{base}/scene.png",
        "floorplan_url": f"{base}/floorplan.png",
        "output_url": f"{base}/output.png",
        "composite_raw_url": f"{base}/composite_raw.png",
    }


def _validate_upload_is_png_jpg(upload: UploadFile, label: str) -> None:
    ct = (getattr(upload, "content_type", "") or "").lower().strip()
    if ct and ct not in ALLOWED_CT:
        raise HTTPException(status_code=400, detail=f"{label} must be PNG or JPG")


async def _save_upload(upload: UploadFile, dst_path: str) -> None:
    try:
        try:
            upload.file.seek(0)
        except Exception:
            pass

        with open(dst_path, "wb") as out:
            out.write(await upload.read())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to save upload '{upload.filename}': {exc}") from exc


def _write_meta(job_folder: str, meta: Dict[str, Any]) -> str:
    meta_path = os.path.join(job_folder, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
    return meta_path


def _build_sd35_harmonize_meta(
    job_id: str,
    prompt: str,
    category: Category,
    shot: Shot,
    upscale_2x: Optional[bool],
    seed: Optional[int],
    negative_prompt: Optional[str] = None,
    strength: float = 0.55,
    input_image: str = "composite_raw.png",
    planned_output: str = "output.png",
) -> Dict[str, Any]:
    """
    This is the REAL SD3.5 block the GPU worker will run at the end to harmonize.
    Locked by Doc 18 via apply_preset_to_meta().
    """
    final_seed = seed if seed is not None else int(uuid.uuid4().int % 1_000_000_000)

    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": datetime.datetime.utcnow().isoformat(),

        "type": "img2img",
        "model_name": "sd3.5-large-pro-2.1",

        "prompt": prompt or "",
        "negative_prompt": negative_prompt,
        "seed": final_seed,

        "category": category,
        "shot": shot,

        # strength is allowed; denoise is always 0.0
        "strength": float(strength),
        "denoise": 0.0,

        "input_image": input_image,
        "planned_output_image": planned_output,

        "mode_runtime": "gpu-dispatch",
        "status": "queued",
    }

    apply_preset_to_meta(meta, category=category, shot=shot, upscale_2x=upscale_2x)

    # Safety hard lock
    meta["denoise"] = 0.0
    if isinstance(meta.get("upscale"), dict):
        meta["upscale"]["denoise"] = 0.0

    return meta


# ---------------------------------------------------------------------------
# REAL Route
# ---------------------------------------------------------------------------

@router.post("/render")
async def insert_object_render(
    product_image: UploadFile = File(..., description="Product image (PNG/JPG)"),
    scene_image: UploadFile = File(..., description="Scene image (PNG/JPG)"),
    floorplan_image: Optional[UploadFile] = File(None, description="Optional floorplan (PNG/JPG)"),

    prompt: Optional[str] = Form(None, description="Optional style / harmonization prompt"),
    placement_hint: Optional[str] = Form(None, description="Optional placement hint"),

    # Optional: upscale toggle for SD3.5 final harmonize (if omitted, preset default applies)
    upscale_2x: Optional[bool] = Form(None),

    # Optional: seed for SD3.5 final harmonize
    seed: Optional[int] = Form(None),

    # Optional: strength for final harmonize (controls preserve vs change)
    strength: float = Form(0.55, description="0..1 (lower preserves more)"),

    # You said you don't like choosing: we default category/shot to best for this pipeline
    category: Optional[Category] = Form(None, description="Optional override. Default: interior"),
    shot: Optional[Shot] = Form(None, description="Optional override. Default: wide"),
):
    if not product_image.filename:
        raise HTTPException(status_code=400, detail="product_image has no filename.")
    if not scene_image.filename:
        raise HTTPException(status_code=400, detail="scene_image has no filename.")

    _validate_upload_is_png_jpg(product_image, "product_image")
    _validate_upload_is_png_jpg(scene_image, "scene_image")
    if floorplan_image is not None and floorplan_image.filename:
        _validate_upload_is_png_jpg(floorplan_image, "floorplan_image")

    if not (0.0 <= float(strength) <= 1.0):
        raise HTTPException(status_code=400, detail="strength must be between 0.0 and 1.0")

    # Auto-best defaults (you are the pilot, I choose the knobs)
    final_category: Category = category or "interior"
    final_shot: Shot = shot or "wide"

    # 1) Create job folder
    job_folder = _create_job_folder()
    job_id = os.path.basename(job_folder)

    # 2) Save files
    saved_files: Dict[str, str] = {}

    product_path = os.path.join(job_folder, "product.png")
    await _save_upload(product_image, product_path)
    saved_files["product_image"] = "product.png"

    scene_path = os.path.join(job_folder, "scene.png")
    await _save_upload(scene_image, scene_path)
    saved_files["scene_image"] = "scene.png"

    if floorplan_image is not None and floorplan_image.filename:
        floorplan_path = os.path.join(job_folder, "floorplan.png")
        await _save_upload(floorplan_image, floorplan_path)
        saved_files["floorplan_image"] = "floorplan.png"

    created_at = datetime.datetime.utcnow().isoformat()

    # 3) Define REAL GPU pipeline stages (worker will execute them)
    pipeline: List[Dict[str, Any]] = [
        {
            "stage": "segment_product",
            "inputs": ["product.png"],
            "outputs": ["product_mask.png", "product_rgba.png"],
        },
        {
            "stage": "estimate_depth_and_pose",
            "inputs": ["scene.png", "product.png"],
            "outputs": ["scene_depth.png", "product_pose.json"],
        },
        {
            "stage": "compose_product_in_scene",
            "inputs": ["scene.png", "product_rgba.png", "product_mask.png"],
            "outputs": ["composite_raw.png"],
        },
        {
            "stage": "sd35_harmonize",
            "inputs": ["composite_raw.png"],
            "outputs": ["output.png"],
        },
    ]

    # 4) Build SD3.5 final harmonize meta (Doc 18 locked)
    # If user didn't provide a prompt, we still run harmonize with a safe default.
    sd35_prompt = (prompt or "").strip() or (
        "photorealistic interior scene, natural lighting, cohesive materials, consistent shadows, "
        "high-end design finish, realistic textures"
    )

    sd35_meta = _build_sd35_harmonize_meta(
        job_id=job_id,
        prompt=sd35_prompt,
        category=final_category,
        shot=final_shot,
        upscale_2x=upscale_2x,
        seed=seed,
        negative_prompt=None,
        strength=float(strength),
        input_image="composite_raw.png",
        planned_output="output.png",
    )

    # 5) Write meta.json
    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": created_at,
        "type": "insert_object",

        "status": "queued",
        "mode_runtime": "gpu-dispatch",

        "files": saved_files,

        "prompt": prompt,
        "placement_hint": placement_hint,

        "pipeline_key": "insert_object::render",

        "gpu_pipeline": pipeline,

        # Doc 18 locked SD3.5 stage (final harmonize)
        "sd35": {
            "enabled": True,
            "category": final_category,
            "shot": final_shot,
            "meta": sd35_meta,
        },

        "outputs": {
            "product_mask": "product_mask.png",
            "product_rgba": "product_rgba.png",
            "composite_raw": "composite_raw.png",
            "final_image": "output.png",
            "meta": "meta.json",
        },

        "dispatch": {
            "job_type": "insert_object",
            "dispatched_at": None,
            "gpu_response": None,
            "error": None,
        },
    }

    meta_path = _write_meta(job_folder, meta)
    public_urls = _outputs_public_urls(job_folder)

    # 6) DISPATCH to GPU worker (REAL)
    ok, gpu_resp = dispatch_insert_object(job_folder=job_folder, meta=meta)

    if not ok:
        try:
            meta["status"] = "gpu_error"
            meta["dispatch"]["error"] = gpu_resp
            _write_meta(job_folder, meta)
        except Exception:
            pass

        return {
            "status": "gpu_error",
            "message": "Insert-object job created but GPU worker failed.",
            "job_folder": job_folder,
            "meta_path": meta_path,
            "public_urls": public_urls,
            "gpu_error": gpu_resp,
        }

    try:
        meta["status"] = "dispatched"
        meta["dispatch"]["dispatched_at"] = datetime.datetime.utcnow().isoformat()
        meta["dispatch"]["gpu_response"] = gpu_resp
        _write_meta(job_folder, meta)
    except Exception:
        pass

    return {
        "status": "dispatched",
        "message": "Insert-object job dispatched to GPU worker (REAL).",
        "job_folder": job_folder,
        "meta_path": meta_path,
        "public_urls": public_urls,
        "gpu_response": gpu_resp,
        "sd35_preset_applied": sd35_meta.get("preset", {}),
        "upscale_enabled": bool(sd35_meta.get("upscale", {}).get("enabled", False)),
        "expected_outputs": {
            "output": os.path.join(job_folder, "output.png"),
            "composite_raw": os.path.join(job_folder, "composite_raw.png"),
            "product_mask": os.path.join(job_folder, "product_mask.png"),
        },
    }
