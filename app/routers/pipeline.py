# app/routers/pipeline.py
"""
Unified pipeline planner for RENDEREXPO AI STUDIO (planning-first + optional execution).

CRITICAL (Doc 18):
- Any stage that uses SD3.5 MUST store the locked preset system:
  steps, CFG, LyCORIS(PRO 2.1) multiplier, GEO multiplier, resolution,
  NO denoise anywhere, upscale optional per request.

This router provides:
1) /plan  -> CPU-only planning; writes meta["pipeline_plan"] and injects resolved_sd35_meta blocks.
2) /run   -> Executes supported stages NOW (REAL). Currently:
           - vr      -> builds a real Three.js viewer + zip package under job_folder/vr/
           - upscale -> deterministic upscaler (NO diffusion), writes output image + updates meta

Notes:
- We DO NOT run SD3.5 inference here (GPU runtime stays in GPU dispatch flows).
- /run is incremental: we add real execution stage-by-stage.
"""

from __future__ import annotations

import os
import json
import datetime
from pathlib import Path
from typing import List, Dict, Any, Literal, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from PIL import Image  # deterministic upscale

from app.presets_sd35 import apply_preset_to_meta

# REAL stage builders (implemented now)
from app.services.vr_builder import build_vr_package


router = APIRouter(
    prefix="/api/pipeline",
    tags=["Pipeline"],
)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

StageType = Literal[
    "text2img",
    "img2img",
    "depth",
    "controlnet",
    "upscale",
    "vr",
    "moodboard",
    "floorplan",
    "product",
    "product_insert",
    "sketch",
]

Category = Literal["urban", "suburban", "interior", "wide_hero"]
Shot = Literal["wide", "close"]

# stages that will use SD3.5 later
SD35_STAGE_TYPES = {"text2img", "img2img", "controlnet"}

# Engine identity (per your locked standard)
SD35_ENGINE_NAME = "sd35_large_pro_v2_1"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PipelineStage(BaseModel):
    """
    One step in a planned pipeline.

    params is free-form, but for SD3.5 stages we support these keys:

    Required for SD3.5 planning:
      - category: urban/suburban/interior/wide_hero
      - shot: wide/close

    Optional:
      - upscale_2x: true/false (if omitted, preset default)
      - prompt / negative_prompt
      - seed
      - strength (img2img/controlnet only; NOT denoise)
      - control_type / control_strength / conditioning_image (controlnet stage)
      - input_image / planned_output_image

    For UPSCALE (deterministic, real execution supported in /run):
      - enabled: bool (default true)
      - factor: 2 or 4 (default 2)
      - method: "lanczos" (default)
      - input_image: filename in job_folder (default auto)
      - output_image: filename (default output_upscaled_2x.png)
      - denoise: MUST be 0.0 (rejected otherwise)
    """
    stage_type: StageType = Field(..., description="Type of stage in the pipeline.")
    params: Dict[str, Any] = Field(default_factory=dict, description="Free-form parameter dict for this stage.")


class PipelinePlanRequest(BaseModel):
    job_folder: str = Field(..., description="Path to an existing job folder under outputs/{date}/{job_id}/")
    stages: List[PipelineStage] = Field(..., description="List of pipeline stages to attach to this job.")


class PipelinePlanResponse(BaseModel):
    status: str
    message: str
    job_folder: str
    pipeline: Dict[str, Any]
    meta_path: str


class PipelineRunResponse(BaseModel):
    status: str
    message: str
    job_folder: str
    results: Dict[str, Any]
    meta_path: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_job_folder(job_folder: str) -> None:
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


def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat()


def _safe_bool(v: Any) -> bool:
    return bool(v) is True


def _safe_basename(name: str) -> str:
    safe = os.path.basename((name or "").strip())
    if not safe:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    return safe


def _ensure_png_jpg_name(name: str) -> str:
    safe = _safe_basename(name)
    low = safe.lower()
    if not (low.endswith(".png") or low.endswith(".jpg") or low.endswith(".jpeg")):
        raise HTTPException(status_code=400, detail="Only PNG/JPG filenames are supported.")
    return safe


def _require_sd35_selectors(params: Dict[str, Any], idx: int, stage_type: str) -> Tuple[Category, Shot, Optional[bool]]:
    category = params.get("category")
    shot = params.get("shot")
    upscale_2x = params.get("upscale_2x", None)

    if category is None or shot is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Stage {idx} ({stage_type}) requires 'category' and 'shot' in params "
                f"to apply Doc 18 locked presets."
            ),
        )

    if category not in ("urban", "suburban", "interior", "wide_hero"):
        raise HTTPException(status_code=400, detail=f"Stage {idx} invalid category: {category}")

    if shot not in ("wide", "close"):
        raise HTTPException(status_code=400, detail=f"Stage {idx} invalid shot: {shot}")

    if upscale_2x is not None:
        upscale_2x = bool(upscale_2x)

    return category, shot, upscale_2x


def _build_sd35_meta_for_stage(
    stage_type: str,
    params: Dict[str, Any],
    category: Category,
    shot: Shot,
    upscale_2x: Optional[bool],
) -> Dict[str, Any]:
    """
    Create a meta-like dict that the GPU runtime can execute later.
    Enforces Doc 18 locks via apply_preset_to_meta().
    """
    prompt = params.get("prompt")
    negative_prompt = params.get("negative_prompt")
    seed = params.get("seed")

    # Img2img/controlnet transform strength (NOT denoise)
    strength = params.get("strength", 0.70)

    base: Dict[str, Any] = {
        "created_at": _now_iso(),

        # Explicit identity
        "engine": SD35_ENGINE_NAME,
        "model_name": SD35_ENGINE_NAME,

        "category": category,
        "shot": shot,
        "seed": seed,

        # Hard lock: no denoise anywhere
        "denoise": 0.0,

        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "status": "planned",
        "mode": "planned-local",
    }

    if stage_type == "text2img":
        base["type"] = "text2img"
        base["planned_output_image"] = params.get("planned_output_image", "output.png")

    elif stage_type in ("img2img", "controlnet"):
        base["type"] = "img2img"
        base["strength"] = float(strength)
        base["input_image"] = params.get("input_image", "input.png")
        base["planned_output_image"] = params.get("planned_output_image", "output.png")

        if stage_type == "controlnet":
            base["controlnet"] = {
                "control_type": params.get("control_type"),
                "control_strength": float(params.get("control_strength", 1.0)),
                "conditioning_image": params.get("conditioning_image"),
            }

    else:
        base["type"] = stage_type

    # Apply Doc 18 preset locks (width/height/steps/CFG/LyCORIS/GEO + optional upscale block)
    apply_preset_to_meta(base, category=category, shot=shot, upscale_2x=upscale_2x)

    # Safety: enforce denoise hard-lock again
    base["denoise"] = 0.0
    if isinstance(base.get("upscale"), dict):
        base["upscale"]["denoise"] = 0.0

    return base


def _detect_overrides(params: Dict[str, Any]) -> Dict[str, Any]:
    locked_keys = ["width", "height", "steps", "num_inference_steps", "guidance_scale", "cfg", "denoise"]
    overrides = {}
    for k in locked_keys:
        if k in params and params.get(k) is not None:
            overrides[k] = params.get(k)
    return overrides


def _infer_date_and_job_id(job_folder: str) -> Tuple[str, str]:
    """
    job_folder is expected like: outputs/YYYY-MM-DD/<job_id>
    Returns (date_str, job_id)
    """
    p = Path(job_folder)
    job_id = p.name
    date_str = p.parent.name
    return date_str, job_id


def _public_outputs_base(date_str: str, job_id: str) -> str:
    return f"/outputs/{date_str}/{job_id}"


def _open_image(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Input image not found: {os.path.basename(path)}")
    try:
        img = Image.open(path)
        img.load()
        return img
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed opening image: {exc}")


def _save_image(img: Image.Image, out_path: str) -> None:
    try:
        ext = os.path.splitext(out_path)[1].lower()
        if ext == ".png":
            img.save(out_path, format="PNG", optimize=True)
        else:
            rgb = img.convert("RGB")
            rgb.save(out_path, format="JPEG", quality=95, optimize=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed saving image: {exc}")


def _run_deterministic_upscale(
    *,
    job_folder: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    REAL upscale execution (no GPU):
    - deterministic resize only (Lanczos)
    - NO diffusion, denoise must be 0.0
    """
    enabled = True if "enabled" not in params else bool(params.get("enabled"))
    factor = int(params.get("factor", 2))
    method = str(params.get("method", "lanczos")).lower().strip()
    denoise = float(params.get("denoise", 0.0))

    if denoise != 0.0:
        raise HTTPException(status_code=400, detail="Upscale denoise must be 0.0 (NO diffusion).")
    if method != "lanczos":
        raise HTTPException(status_code=400, detail="Upscale method must be 'lanczos' (deterministic).")
    if factor not in (2, 4):
        raise HTTPException(status_code=400, detail="Upscale factor must be 2 or 4.")

    if enabled is False:
        return {
            "enabled": False,
            "ran_at": _now_iso(),
            "denoise": 0.0,
            "note": "Upscale disabled by request.",
        }

    # Resolve input image
    if params.get("input_image"):
        in_name = _ensure_png_jpg_name(str(params["input_image"]))
    else:
        candidates = ["output.png", "output.jpg", "output.jpeg", "input.png", "input.jpg", "input.jpeg"]
        found = None
        for c in candidates:
            if os.path.isfile(os.path.join(job_folder, c)):
                found = c
                break
        in_name = found or "output.png"

    # Resolve output image
    if params.get("output_image"):
        out_name = _ensure_png_jpg_name(str(params["output_image"]))
    else:
        ext = os.path.splitext(in_name)[1].lower()
        if ext in (".jpg", ".jpeg"):
            out_name = f"output_upscaled_{factor}x.jpg"
        else:
            out_name = f"output_upscaled_{factor}x.png"

    in_path = os.path.join(job_folder, in_name)
    out_path = os.path.join(job_folder, out_name)

    img = _open_image(in_path)
    w, h = img.size
    new_w, new_h = w * factor, h * factor

    try:
        up = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    except Exception:
        up = img.resize((new_w, new_h), resample=Image.LANCZOS)  # pillow backward compat

    _save_image(up, out_path)

    return {
        "enabled": True,
        "factor": factor,
        "method": "lanczos",
        "denoise": 0.0,
        "input_image": in_name,
        "output_image": out_name,
        "input_size": {"width": w, "height": h},
        "output_size": {"width": new_w, "height": new_h},
        "ran_at": _now_iso(),
        "mode": "executed-local",
        "note": "Deterministic upscale executed (no diffusion).",
    }


# ---------------------------------------------------------------------------
# Route: PLAN
# ---------------------------------------------------------------------------

@router.post("/plan", response_model=PipelinePlanResponse)
async def plan_pipeline(request: PipelinePlanRequest):
    """
    Planning-only.
    - writes meta["pipeline_plan"]
    - injects resolved_sd35_meta for SD3.5 stages (Doc 18 locked)
    - injects resolved_upscale_plan for upscale stages (denoise=0.0)
    """
    job_folder = request.job_folder
    _ensure_job_folder(job_folder)

    meta = _read_meta(job_folder)
    meta.setdefault("job_id", os.path.basename(job_folder))

    resolved_stages: List[Dict[str, Any]] = []

    for idx, stage in enumerate(request.stages, start=1):
        stage_dict = stage.model_dump()
        stage_type = stage_dict.get("stage_type")
        params = stage_dict.get("params") or {}

        stage_entry: Dict[str, Any] = {
            "stage_index": idx,
            "stage_type": stage_type,
            "params": params,
            "created_at": _now_iso(),
        }

        overrides_applied = _detect_overrides(params)
        if overrides_applied:
            stage_entry["overrides_applied"] = {
                "note": "These keys are preset-locked by Doc 18 and will be ignored/overridden.",
                "attempted": overrides_applied,
            }

        if stage_type in SD35_STAGE_TYPES:
            category, shot, upscale_2x = _require_sd35_selectors(params, idx, stage_type)

            resolved_sd35_meta = _build_sd35_meta_for_stage(
                stage_type=stage_type,
                params=params,
                category=category,
                shot=shot,
                upscale_2x=upscale_2x,
            )

            stage_entry["resolved_sd35_meta"] = resolved_sd35_meta
            stage_entry["doc18_locked"] = True

        elif stage_type in ("product", "product_insert"):
            # Planning helper: only inject SD35 meta if caller asks for rerender/harmonize.
            rerender_flag = bool(params.get("rerender_with_sd35", False) or params.get("harmonize_with_sd35", False))
            if rerender_flag:
                category, shot, upscale_2x = _require_sd35_selectors(params, idx, stage_type)
                resolved_sd35_meta = _build_sd35_meta_for_stage(
                    stage_type="img2img",
                    params={
                        "prompt": params.get("prompt"),
                        "negative_prompt": params.get("negative_prompt"),
                        "seed": params.get("seed"),
                        "strength": params.get("strength", 0.70),
                        "input_image": params.get("input_image", "product_insert_raw.png"),
                        "planned_output_image": params.get("planned_output_image", "product_insert_result.png"),
                        **params,
                    },
                    category=category,
                    shot=shot,
                    upscale_2x=upscale_2x,
                )
                stage_entry["resolved_sd35_meta"] = resolved_sd35_meta
                stage_entry["doc18_locked"] = True
            else:
                stage_entry["doc18_locked"] = False

        elif stage_type == "upscale":
            # Deterministic upscale plan (NO diffusion)
            enabled = True if "enabled" not in params else bool(params.get("enabled"))
            factor = int(params.get("factor", 2))
            method = str(params.get("method", "lanczos")).lower().strip()
            denoise = float(params.get("denoise", 0.0))

            if denoise != 0.0:
                raise HTTPException(status_code=400, detail=f"Stage {idx} (upscale) denoise must be 0.0.")
            if method != "lanczos":
                raise HTTPException(status_code=400, detail=f"Stage {idx} (upscale) method must be 'lanczos'.")
            if factor not in (2, 4):
                raise HTTPException(status_code=400, detail=f"Stage {idx} (upscale) factor must be 2 or 4.")

            # choose defaults (plan-only, file may not exist yet)
            if params.get("input_image"):
                in_name = _ensure_png_jpg_name(str(params["input_image"]))
            else:
                # planned default: output.png
                in_name = "output.png"

            if params.get("output_image"):
                out_name = _ensure_png_jpg_name(str(params["output_image"]))
            else:
                out_name = f"output_upscaled_{factor}x.png"

            stage_entry["resolved_upscale_plan"] = {
                "enabled": bool(enabled),
                "factor": factor,
                "method": "lanczos",
                "denoise": 0.0,
                "input_image": in_name,
                "output_image": out_name,
                "planned_at": _now_iso(),
                "mode": "plan-only",
                "note": "Deterministic upscale only. NO diffusion. denoise=0.0 hard-locked.",
            }
            stage_entry["doc18_locked"] = True  # denoise lock applies

        else:
            stage_entry["doc18_locked"] = False

        resolved_stages.append(stage_entry)

    pipeline_plan = {
        "created_at": _now_iso(),
        "planner": "pipeline_planner_v3_doc18_locked",
        "stages": resolved_stages,
    }

    meta["pipeline_plan"] = pipeline_plan
    meta["pipeline_mode"] = "planned"
    meta["last_updated"] = _now_iso()

    _write_meta(job_folder, meta)

    return PipelinePlanResponse(
        status="ok",
        message="Pipeline planned. SD3.5 stages include Doc 18 locked resolved meta blocks. Upscale stages include deterministic denoise=0 plans.",
        job_folder=job_folder,
        pipeline=pipeline_plan,
        meta_path=_meta_path(job_folder),
    )


# ---------------------------------------------------------------------------
# Route: RUN (REAL execution for supported stages)
# ---------------------------------------------------------------------------

@router.post("/run", response_model=PipelineRunResponse)
async def run_pipeline(request: PipelinePlanRequest):
    """
    Execute supported pipeline stages now (REAL).

    Supported:
    - vr: builds a navigable Three.js viewer + zip (no GPU).
    - upscale: deterministic resize (Pillow Lanczos), writes upscaled file + meta updates.

    We DO NOT execute SD3.5 here. SD3.5 stages are still dispatched via GPU worker routes.
    """
    job_folder = request.job_folder
    _ensure_job_folder(job_folder)

    meta = _read_meta(job_folder)
    meta.setdefault("job_id", os.path.basename(job_folder))

    results: Dict[str, Any] = {}
    date_str, job_id = _infer_date_and_job_id(job_folder)
    public_base = _public_outputs_base(date_str, job_id)

    for idx, stage in enumerate(request.stages, start=1):
        stage_type = stage.stage_type
        params = stage.params or {}

        if stage_type == "vr":
            # VR expects images already present in job folder as view_*.png
            view_files = sorted([p.name for p in Path(job_folder).glob("view_*.png")])
            if len(view_files) < 3:
                iv = meta.get("input_views")
                if isinstance(iv, list):
                    view_files = [str(x) for x in iv if isinstance(x, str)]
                    view_files = [x for x in view_files if (Path(job_folder) / x).is_file()]

            if len(view_files) < 3:
                raise HTTPException(
                    status_code=400,
                    detail=f"Stage {idx} (vr) requires 3+ images in job_folder named view_*.png.",
                )

            prompt = params.get("prompt") if isinstance(params, dict) else None
            plan_hint = params.get("plan_hint") if isinstance(params, dict) else None

            try:
                built = build_vr_package(
                    job_folder=job_folder,
                    image_files=view_files,
                    prompt=prompt,
                    plan_hint=plan_hint,
                )
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=500, detail=f"VR build failed: {exc}")

            viewer_url = f"{public_base}/{built['viewer_rel']}"
            download_url = f"/api/vr/download/{date_str}/{job_id}"

            results.setdefault("vr", {})
            results["vr"] = {
                "viewer_url": viewer_url,
                "download_url": download_url,
                "zip_path": built.get("zip_path"),
            }

            meta.setdefault("outputs", {})
            meta["outputs"]["vr_viewer"] = built["viewer_rel"]
            meta["outputs"]["vr_zip"] = "vr_package.zip"
            meta["last_updated"] = _now_iso()

        elif stage_type == "upscale":
            upscale_result = _run_deterministic_upscale(job_folder=job_folder, params=params)

            # Update meta
            meta.setdefault("outputs", {})
            meta["upscale"] = upscale_result
            if upscale_result.get("enabled") is True:
                out_name = str(upscale_result.get("output_image"))
                meta["outputs"]["upscaled_image"] = out_name

                results.setdefault("upscale", {})
                results["upscale"] = {
                    "output_image": out_name,
                    "output_url": f"{public_base}/{out_name}",
                    "input_image": upscale_result.get("input_image"),
                    "factor": upscale_result.get("factor"),
                    "method": upscale_result.get("method"),
                }
            else:
                results.setdefault("upscale", {})
                results["upscale"] = {"enabled": False}

            meta["last_updated"] = _now_iso()

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Stage '{stage_type}' is not implemented in /run yet.",
            )

    _write_meta(job_folder, meta)

    return PipelineRunResponse(
        status="ok",
        message="Pipeline executed for supported stages (vr, upscale).",
        job_folder=job_folder,
        results=results,
        meta_path=_meta_path(job_folder),
    )
