# app/routers/product.py

"""
Product insertion planning endpoints (NO AI yet; planning-first, real file I/O).

What this does NOW (real):
- Accepts:
    * room_image (required)  — the base room / space image
    * object_image (required) — the product image (couch, lamp, table, etc.)
    * placement_prompt (required) — where/how to place it
    * style_hint (optional) — a label to enrich the prompt
    * mode (insert_only | insert_and_rerender)
    * if insert_and_rerender: Doc 18 selectors (category, shot) + optional upscale_2x + seed
- Creates outputs/{YYYY-MM-DD}/{job_id}/
- Saves:
    * room.png
    * object.png
- Writes meta.json describing the future pipeline steps
- If rerender is enabled, also writes a Doc 18 preset-locked SD3.5 "rerender meta block"
  that a future GPU stage can execute.

CRITICAL (Doc 18):
- If mode includes SD3.5 re-render, we MUST store locked preset parameters in meta:
  steps, CFG, LyCORIS(PRO 2.1) multiplier, GEO multiplier, resolution,
  NO denoise anywhere, upscale optional per request.
- Diffusion denoise is hard-locked to 0.0.
- PNG/JPG only for uploads (global rule).

What this does NOT do YET:
- No segmentation (SAM / etc.)
- No depth or pose estimation
- No compositing
- No GPU dispatch
"""

from __future__ import annotations

import os
import uuid
import json
import shutil
import datetime
from typing import Optional, Dict, Any, List, Literal, Tuple

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.presets_sd35 import apply_preset_to_meta

router = APIRouter(prefix="/api/product", tags=["Product Insertion"])

Category = Literal["urban", "suburban", "interior", "wide_hero"]
Shot = Literal["wide", "close"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_iso() -> str:
    return datetime.datetime.utcnow().isoformat()


def _today_utc_str() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


def _create_job_folder(base_outputs_dir: str = "outputs") -> str:
    """
    Create outputs/{YYYY-MM-DD}/{job_id}/ (UTC) and return its path.
    """
    today = _today_utc_str()
    job_id = uuid.uuid4().hex
    folder = os.path.join(base_outputs_dir, today, job_id)
    os.makedirs(folder, exist_ok=True)

    # marker file (best-effort)
    try:
        with open(os.path.join(folder, "job_type.txt"), "w", encoding="utf-8") as f:
            f.write("product_insertion")
    except Exception:
        pass

    return folder


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
            "room_url": None,
            "object_url": None,
            "planned_raw_url": None,
            "planned_final_url": None,
        }

    base = f"/outputs/{date_str}/{job_id}"
    return {
        "meta_url": f"{base}/meta.json",
        "room_url": f"{base}/room.png",
        "object_url": f"{base}/object.png",
        "planned_raw_url": f"{base}/room_with_object_raw.png",
        "planned_final_url": f"{base}/room_with_object_final.png",
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


def _write_meta(job_folder: str, meta: Dict[str, Any]) -> str:
    meta_path = os.path.join(job_folder, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
    return meta_path


def _build_sd35_rerender_meta(
    *,
    prompt: str,
    category: Category,
    shot: Shot,
    upscale_2x: Optional[bool],
    seed: Optional[int],
    negative_prompt: Optional[str] = None,
    planned_output_image: str = "room_with_object_final.png",
    input_image: str = "room_with_object_raw.png",
    strength: float = 0.70,
) -> Dict[str, Any]:
    """
    Build a Doc 18 preset-locked SD3.5 img2img meta block for a future GPU rerender/harmonize stage.
    IMPORTANT:
    - "strength" is allowed and is NOT diffusion denoise.
    - diffusion denoise is hard-locked to 0.0.
    """
    final_seed = seed if seed is not None else int(uuid.uuid4().int % 1_000_000_000)

    meta: Dict[str, Any] = {
        "job_id": None,  # filled by caller
        "created_at": _utc_iso(),

        "type": "img2img",

        # Explicit identity: SD3.5 Large PRO 2.1
        "engine": "sd35_large_pro_v2_1",
        "model_name": "sd35_large_pro_v2_1",

        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": final_seed,

        "category": category,
        "shot": shot,

        # Hard lock: no denoise anywhere
        "denoise": 0.0,

        # Img2Img transform knob (NOT denoise)
        "strength": float(strength),

        # Stage IO
        "input_image": input_image,
        "planned_output_image": planned_output_image,

        "status": "planned",
        "mode_runtime": "planned-gpu-dispatch",
    }

    apply_preset_to_meta(meta, category=category, shot=shot, upscale_2x=upscale_2x)

    # Safety lock again
    meta["denoise"] = 0.0
    if isinstance(meta.get("upscale"), dict):
        meta["upscale"]["denoise"] = 0.0

    return meta


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/plan-insert")
async def plan_product_insertion(
    room_image: UploadFile = File(..., description="Base room / space image (PNG/JPG only)."),
    object_image: UploadFile = File(..., description="Product image (PNG/JPG only)."),
    placement_prompt: str = Form(
        ...,
        description=(
            "Instruction like: 'place the couch against the back wall, "
            "scale realistically, match warm lighting.'"
        ),
    ),
    style_hint: Optional[str] = Form(
        None,
        description="Optional style hint label (e.g. 'soft luxury living room').",
    ),
    mode: str = Form(
        "insert_and_rerender",
        description="insert_only OR insert_and_rerender",
    ),

    # --- Doc 18 preset selectors for rerender stage (only used if mode includes rerender)
    category: Optional[Category] = Form(
        None,
        description="Doc 18 preset category for rerender stage (required if insert_and_rerender).",
    ),
    shot: Optional[Shot] = Form(
        None,
        description="Doc 18 preset shot for rerender stage (required if insert_and_rerender).",
    ),
    upscale_2x: Optional[bool] = Form(
        None,
        description="Optional override for upscale during rerender stage (true/false).",
    ),
    seed: Optional[int] = Form(
        None,
        description="Optional seed for rerender stage. If omitted, generated.",
    ),
):
    """
    Planning-first product insertion job (real file saves + meta.json; no AI/GPU yet).

    Creates:
      outputs/{date}/{job_id}/
        room.png
        object.png
        meta.json
    """
    # global rule: PNG/JPG only
    if not room_image.filename or not object_image.filename:
        raise HTTPException(status_code=400, detail="room_image and object_image must have filenames.")
    _ensure_png_jpg_only(room_image)
    _ensure_png_jpg_only(object_image)

    if mode not in ("insert_only", "insert_and_rerender"):
        raise HTTPException(
            status_code=400,
            detail="mode must be 'insert_only' or 'insert_and_rerender'.",
        )

    if mode == "insert_and_rerender":
        if category is None or shot is None:
            raise HTTPException(
                status_code=400,
                detail="category and shot are required when mode='insert_and_rerender'.",
            )

    # 1) Create job folder + job_id
    job_folder = _create_job_folder()
    job_id = os.path.basename(job_folder)

    # 2) Save uploads (normalized names)
    room_rel = "room.png"
    object_rel = "object.png"
    await _save_upload_stream(room_image, os.path.join(job_folder, room_rel))
    await _save_upload_stream(object_image, os.path.join(job_folder, object_rel))

    # 3) Build planned pipeline actions (conceptual; future GPU stages)
    created_at = _utc_iso()

    planned_actions: List[Dict[str, Any]] = [
        {
            "stage": "segment_object",
            "inputs": ["object.png"],
            "outputs": ["object_mask.png"],
            "description": "Segment the product from background (future GPU).",
        },
        {
            "stage": "extract_object_rgba",
            "inputs": ["object.png", "object_mask.png"],
            "outputs": ["object_rgba.png"],
            "description": "Cut out object with transparency (future GPU).",
        },
        {
            "stage": "estimate_room_depth",
            "inputs": ["room.png"],
            "outputs": ["room_depth.png"],
            "description": "Estimate room depth for realistic placement (future GPU).",
        },
        {
            "stage": "compute_placement",
            "inputs": ["room.png", "room_depth.png", "object_rgba.png"],
            "outputs": ["layout_plan.json"],
            "description": "Compute position/scale/orientation in 2.5D (future).",
        },
        {
            "stage": "composite_object",
            "inputs": ["room.png", "object_rgba.png", "layout_plan.json"],
            "outputs": ["room_with_object_raw.png"],
            "description": "Composite object into room (future).",
        },
    ]

    # 4) Optional: add SD3.5 rerender/harmonize stage plan (Doc 18 locked)
    sd35_rerender_meta: Optional[Dict[str, Any]] = None

    if mode == "insert_and_rerender":
        planned_actions.append(
            {
                "stage": "sd35_rerender",
                "inputs": ["room_with_object_raw.png"],
                "outputs": ["room_with_object_final.png"],
                "description": "Run SD3.5 img2img to harmonize lighting/materials using Doc 18 locked presets (future GPU).",
            }
        )

        # combine style_hint + placement_prompt into the SD3.5 prompt
        combined_prompt = (style_hint + ", " if style_hint else "") + placement_prompt

        sd35_rerender_meta = _build_sd35_rerender_meta(
            prompt=combined_prompt,
            category=category,  # type: ignore[arg-type]
            shot=shot,          # type: ignore[arg-type]
            upscale_2x=upscale_2x,
            seed=seed,
            negative_prompt=None,
            input_image="room_with_object_raw.png",
            planned_output_image="room_with_object_final.png",
            strength=0.70,
        )
        sd35_rerender_meta["job_id"] = job_id

    # 5) Write meta.json
    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": created_at,
        "type": "product_insertion",

        # Explicit identity: SD3.5 Large PRO 2.1 (even though we do not run it here)
        "engine": "sd35_large_pro_v2_1",
        "model_name": "sd35_large_pro_v2_1",

        "status": "planned",
        "mode_runtime": "planning-only",
        "pipeline": "product_insertion_v1",

        "files": {
            "room_image": room_rel,
            "object_image": object_rel,
        },

        "placement_prompt": placement_prompt,
        "style_hint": style_hint,
        "mode": mode,

        # Global hard-lock
        "denoise": 0.0,

        "planned_outputs": {
            "mask": "object_mask.png",
            "rgba": "object_rgba.png",
            "room_depth": "room_depth.png",
            "layout_plan": "layout_plan.json",
            "composite_raw": "room_with_object_raw.png",
            "final": "room_with_object_final.png",
        },

        "planned_actions": planned_actions,
        "sd35_rerender": {"enabled": bool(sd35_rerender_meta is not None), "meta": sd35_rerender_meta},
    }

    # ensure sd35_rerender meta uses denoise=0.0 even if something upstream changes
    if isinstance(meta.get("sd35_rerender"), dict) and isinstance(meta["sd35_rerender"].get("meta"), dict):
        meta["sd35_rerender"]["meta"]["denoise"] = 0.0
        if isinstance(meta["sd35_rerender"]["meta"].get("upscale"), dict):
            meta["sd35_rerender"]["meta"]["upscale"]["denoise"] = 0.0

    meta_path = _write_meta(job_folder, meta)
    public_urls = _outputs_public_urls(job_folder)

    # 6) Return
    return {
        "status": "ok",
        "message": "Product insertion job planned (planning-only; files saved; no AI/GPU executed).",
        "job_folder": job_folder,
        "meta_path": meta_path,
        "public_urls": public_urls,
        "files_saved": {
            "room_image": os.path.join(job_folder, room_rel),
            "object_image": os.path.join(job_folder, object_rel),
        },
        "sd35_rerender_enabled": bool(sd35_rerender_meta is not None),
        "sd35_preset_applied": (sd35_rerender_meta or {}).get("preset", {}) if sd35_rerender_meta else None,
    }
