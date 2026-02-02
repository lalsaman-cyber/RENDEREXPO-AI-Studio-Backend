# app/routers/product_insert.py
"""
app/routers/product_insert.py

Product insertion skeleton router (planning only).

Goal:
- Upload a product image (couch, lamp, table, etc.)
- Optionally upload a room image (recommended)
- Store prompts and placement hints
- Plan a future SD3.5-based "harmonize/rerender" stage using Doc 18 locked presets

CRITICAL (Doc 18):
- Any planned SD3.5 work MUST use locked preset parameters:
  steps, CFG, LyCORIS(PRO 2.1) multiplier, GEO multiplier, resolution,
  NO denoise anywhere, upscale optional per-request.

NOW (real behavior):
- Creates outputs/{YYYY-MM-DD}/{job_id}/
- Saves product.png (+ room.png if provided)
- Writes meta.json with:
    * selectors (category/shot/upscale_2x/seed)
    * computed locked preset block ("locked_generation_controls")
- /plan writes/updates a "planned_product_insertion" pipeline with an optional SD3.5 harmonize stage.
"""

from __future__ import annotations

import os
import uuid
import json
import shutil
import datetime
from typing import Dict, Any, Optional, Literal, Tuple

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.presets_sd35 import apply_preset_to_meta

router = APIRouter(
    prefix="/api/product-insert",
    tags=["Product Insertion"],
)

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
    """Create outputs/{YYYY-MM-DD}/{job_id}/ (UTC)."""
    today = _today_utc_str()
    job_id = uuid.uuid4().hex
    folder = os.path.join(base_outputs_dir, today, job_id)
    os.makedirs(folder, exist_ok=True)
    return folder


def _ensure_folder_exists(job_folder: str) -> None:
    if not job_folder or not os.path.isdir(job_folder):
        raise HTTPException(status_code=400, detail=f"job_folder does not exist: {job_folder}")


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
            "product_url": None,
            "room_url": None,
        }

    base = f"/outputs/{date_str}/{job_id}"
    return {
        "meta_url": f"{base}/meta.json",
        "product_url": f"{base}/product.png",
        "room_url": f"{base}/room.png",
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


def _compute_locked_controls_probe(
    *,
    prompt: str,
    category: Category,
    shot: Shot,
    upscale_2x: Optional[bool],
    seed: int,
    planned_output_image: str,
) -> Dict[str, Any]:
    """
    Build a "probe" dict and apply preset to expose locked knobs for UI/debugging.
    """
    probe: Dict[str, Any] = {
        "type": "img2img",
        # Explicit identity: SD3.5 Large PRO 2.1
        "engine": "sd35_large_pro_v2_1",
        "model_name": "sd35_large_pro_v2_1",
        "prompt": prompt,
        "seed": seed,
        "category": category,
        "shot": shot,
        "denoise": 0.0,
        "status": "planned",
        "mode": "plan-only",
        "planned_output_image": planned_output_image,
    }

    apply_preset_to_meta(probe, category=category, shot=shot, upscale_2x=upscale_2x)

    # Safety lock again
    probe["denoise"] = 0.0
    if isinstance(probe.get("upscale"), dict):
        probe["upscale"]["denoise"] = 0.0

    return probe


# ---------------------------------------------------------------------------
# 1) Upload product (and optional room) + prompts + preset selectors
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_product_and_room(
    product_image: UploadFile = File(..., description="Image of the product (PNG/JPG only)."),
    room_image: Optional[UploadFile] = File(
        None,
        description="Optional base room / space image (PNG/JPG only) where the product will be inserted.",
    ),
    prompt: str = Form(..., description="High-level description of desired final scene (style, mood, etc.)."),
    placement_prompt: Optional[str] = Form(
        None,
        description="Where/how to place product (e.g. 'against back wall under window').",
    ),

    # Doc 18 preset selectors (used later when we call SD3.5 to harmonize/rerender)
    category: Category = Form(..., description="Doc 18 preset category"),
    shot: Shot = Form(..., description="Doc 18 preset shot"),
    upscale_2x: Optional[bool] = Form(
        None,
        description="Optional upscale override. If omitted, preset default applies.",
    ),
    seed: Optional[int] = Form(
        None,
        description="Optional seed. If omitted, backend generates one.",
    ),
):
    """
    Start a product insertion job (planning only).

    Creates:
      outputs/{date}/{job_id}/
        product.png
        room.png (optional)
        meta.json
    """
    if not product_image.filename:
        raise HTTPException(status_code=400, detail="Uploaded product_image has no filename.")

    _ensure_png_jpg_only(product_image)
    if room_image is not None:
        if not room_image.filename:
            raise HTTPException(status_code=400, detail="Uploaded room_image has no filename.")
        _ensure_png_jpg_only(room_image)

    job_folder = _create_job_folder()
    _ensure_folder_exists(job_folder)

    final_seed = seed if seed is not None else int(uuid.uuid4().int % 1_000_000_000)

    # Save files with normalized names
    await _save_upload_stream(product_image, os.path.join(job_folder, "product.png"))
    if room_image is not None:
        await _save_upload_stream(room_image, os.path.join(job_folder, "room.png"))

    meta: Dict[str, Any] = _read_meta(job_folder)
    meta.setdefault("job_id", os.path.basename(job_folder))

    meta.update(
        {
            "type": "product_insert",
            "created_at": _utc_iso(),
            "mode_runtime": "planning-only",
            "status": "planned",

            # Explicit identity: SD3.5 Large PRO 2.1 (even though we do not run it here)
            "engine": "sd35_large_pro_v2_1",
            "model_name": "sd35_large_pro_v2_1",

            "product_image": "product.png",
            "room_image": "room.png" if room_image is not None else None,

            "prompt": prompt,
            "placement_prompt": placement_prompt,

            # Store Doc 18 selectors for later SD3.5 harmonization stage
            "category": category,
            "shot": shot,
            "upscale_2x": upscale_2x,
            "seed": final_seed,

            # Hard lock: NO denoise anywhere (global rule)
            "denoise": 0.0,

            # Planned outputs (future)
            "planned_output": "product_insert_result.png",
        }
    )

    # Precompute locked preset knobs for UI visibility/debug
    probe = _compute_locked_controls_probe(
        prompt=prompt,
        category=category,
        shot=shot,
        upscale_2x=upscale_2x,
        seed=final_seed,
        planned_output_image="product_insert_result.png",
    )

    meta["preset"] = probe.get("preset", {})
    meta["locked_generation_controls"] = {
        "width": probe.get("width"),
        "height": probe.get("height"),
        "num_inference_steps": probe.get("num_inference_steps"),
        "guidance_scale": probe.get("guidance_scale"),
        "lora_config": probe.get("lora_config"),
        "geo_config": probe.get("geo_config"),
        "upscale": probe.get("upscale"),
        "denoise": 0.0,
    }

    _write_meta(job_folder, meta)

    return {
        "status": "ok",
        "message": "Product insertion job created (planning-only, preset-aware, no AI yet).",
        "job_folder": job_folder,
        "public_urls": _outputs_public_urls(job_folder),
        "files": {
            "product_image": os.path.join(job_folder, "product.png"),
            "room_image": os.path.join(job_folder, "room.png") if room_image is not None else None,
        },
        "meta_path": _meta_path(job_folder),
        "preset_applied": meta.get("preset", {}),
    }


# ---------------------------------------------------------------------------
# 2) Plan SD3.5 insertion/harmonization pass (still planning-only, preset-locked)
# ---------------------------------------------------------------------------

@router.post("/plan")
async def plan_product_insertion(
    job_folder: str = Form(..., description="Job folder returned by /product-insert/upload"),

    # Optional override of selectors (otherwise use job meta)
    category: Optional[Category] = Form(None),
    shot: Optional[Shot] = Form(None),
    upscale_2x: Optional[bool] = Form(
        None,
        description="Optional override. If omitted, uses job meta override; else preset default.",
    ),
    seed: Optional[int] = Form(
        None,
        description="Optional override. If omitted, uses job meta seed; else auto.",
    ),

    # Planning knobs for insertion pipeline (not SD3.5 core knobs)
    mode: str = Form(
        "insert_and_harmonize",
        description="insert_only (no SD3.5) | insert_and_harmonize (plans SD3.5 stage).",
    ),
):
    """
    Plan the future pipeline for product insertion.

    IMPORTANT:
    - We do NOT accept width/height/steps/CFG here. Those are Doc 18 locked by presets.
    - If mode includes SD3.5 harmonization/rerender, we attach a planned_sd35_meta block
      with apply_preset_to_meta() applied.
    """
    _ensure_folder_exists(job_folder)

    meta = _read_meta(job_folder)
    if meta.get("type") != "product_insert":
        raise HTTPException(status_code=400, detail="meta.json does not describe a product_insert job.")

    if mode not in ("insert_only", "insert_and_harmonize"):
        raise HTTPException(status_code=400, detail="mode must be 'insert_only' or 'insert_and_harmonize'.")

    # Resolve selectors (override > stored meta)
    use_category: Optional[Category] = category or meta.get("category")
    use_shot: Optional[Shot] = shot or meta.get("shot")

    if use_category is None or use_shot is None:
        raise HTTPException(
            status_code=400,
            detail="Missing category/shot. Provide in /upload or override in /plan.",
        )

    # Resolve seed
    base_seed = seed if seed is not None else meta.get("seed")
    final_seed = base_seed if base_seed is not None else int(uuid.uuid4().int % 1_000_000_000)

    # Resolve upscale override:
    # - if caller provides upscale_2x -> use it
    # - else if meta stored upscale_2x override -> use it
    # - else preset default applies (pass None)
    if upscale_2x is None:
        upscale_override = meta.get("upscale_2x", None)
    else:
        upscale_override = upscale_2x

    planned_pipeline: Dict[str, Any] = {
        "created_at": _utc_iso(),
        "mode": mode,
        "planned_outputs": {
            "mask": "product_mask.png",
            "rgba": "product_rgba.png",
            "depth_room": "room_depth.png",
            "composite_raw": "product_insert_raw.png",
            "final": meta.get("planned_output", "product_insert_result.png"),
        },
        "planned_actions": [
            {
                "stage": "segment_product",
                "description": "Segment product from background (future GPU: SAM/NIM/etc).",
                "inputs": ["product.png"],
                "outputs": ["product_mask.png", "product_rgba.png"],
            },
            {
                "stage": "estimate_room_depth",
                "description": "Estimate depth of room for realistic placement (future GPU: MiDaS).",
                "inputs": ["room.png"] if meta.get("room_image") else [],
                "outputs": ["room_depth.png"],
            },
            {
                "stage": "compute_placement",
                "description": "Compute plausible position/scale/orientation based on placement_prompt (future).",
                "inputs": ["room.png", "room_depth.png", "product_rgba.png"],
                "outputs": ["placement.json"],
            },
            {
                "stage": "composite",
                "description": "Composite product into room (future).",
                "inputs": ["room.png", "product_rgba.png", "placement.json"],
                "outputs": ["product_insert_raw.png"],
            },
        ],
    }

    planned_sd35_meta: Optional[Dict[str, Any]] = None

    if mode == "insert_and_harmonize":
        # Plan a future SD3.5 img2img harmonization pass using Doc 18 locked presets
        planned_sd35_meta = {
            "type": "img2img",

            # Explicit identity: SD3.5 Large PRO 2.1
            "engine": "sd35_large_pro_v2_1",
            "model_name": "sd35_large_pro_v2_1",

            "prompt": meta.get("prompt"),
            "negative_prompt": None,
            "seed": final_seed,

            # input for future stage
            "input_image": "product_insert_raw.png",
            "strength": 0.70,  # planning default; UI can override later

            # locked selectors
            "category": use_category,
            "shot": use_shot,

            # hard lock: no denoise anywhere
            "denoise": 0.0,

            "status": "planned",
            "mode": "plan-only",
            "planned_output_image": meta.get("planned_output", "product_insert_result.png"),
        }

        apply_preset_to_meta(planned_sd35_meta, category=use_category, shot=use_shot, upscale_2x=upscale_override)
        planned_sd35_meta["denoise"] = 0.0
        if isinstance(planned_sd35_meta.get("upscale"), dict):
            planned_sd35_meta["upscale"]["denoise"] = 0.0

        planned_pipeline["planned_actions"].append(
            {
                "stage": "sd35_harmonize",
                "description": "Run SD3.5 img2img to harmonize lighting/materials using Doc 18 locked presets (future GPU).",
                "inputs": ["product_insert_raw.png"],
                "outputs": [meta.get("planned_output", "product_insert_result.png")],
                "sd35_meta": planned_sd35_meta,
            }
        )

    # Persist plan
    meta["planned_product_insertion"] = planned_pipeline
    meta["category"] = use_category
    meta["shot"] = use_shot
    meta["seed"] = final_seed
    meta["upscale_2x"] = upscale_override
    meta["denoise"] = 0.0
    meta["mode_runtime"] = "planning-only"
    meta["status"] = "planned"
    meta["updated_at"] = _utc_iso()

    _write_meta(job_folder, meta)

    return {
        "status": "ok",
        "message": "Product insertion planned (planning-only). SD3.5 preset-locked meta included when requested.",
        "job_folder": job_folder,
        "public_urls": _outputs_public_urls(job_folder),
        "planned_pipeline": planned_pipeline,
        "sd35_meta_included": planned_sd35_meta is not None,
        "sd35_preset_applied": (planned_sd35_meta or {}).get("preset", {}) if planned_sd35_meta else None,
        "meta_path": _meta_path(job_folder),
    }
