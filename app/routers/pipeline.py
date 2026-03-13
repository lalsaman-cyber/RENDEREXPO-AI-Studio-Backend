# app/routers/pipeline.py
"""
Unified pipeline planner for RENDEREXPO AI STUDIO (planner-first + selective execution).

IMPORTANT:
- Planner = port 8012
- GPU worker = port 8002

What this router does:
1) /plan
   - CPU-only planning
   - writes meta["pipeline_plan"]
   - injects resolved SD3.5 meta blocks using locked presets

2) /run
   - executes only supported stages now
   - currently:
       - vr      -> REAL GPU dispatch via planner -> GPU worker
       - upscale -> deterministic upscale only (Pillow Lanczos)

What this router does NOT do:
- no SD3.5 inference here
- no local VR builder import
- SD3.5 execution remains in dedicated planner -> GPU worker flows
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

from app.presets_sd35 import apply_preset_to_meta

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
VRMode = Literal["gaussian_splat", "nerf", "mesh"]

SD35_STAGE_TYPES = {"text2img", "img2img", "controlnet"}
SD35_ENGINE_NAME = "sd35_large_pro_v2_1"

# ---------------------------------------------------------------------------
# HMAC / GPU dispatch
# ---------------------------------------------------------------------------

HMAC_SECRET_ENV = "RENDEREXPO_HMAC_SECRET"
SIG_HEADER = "X-RENDEREXPO-SIGNATURE"
TS_HEADER = "X-RENDEREXPO-TIMESTAMP"
NONCE_HEADER = "X-RENDEREXPO-NONCE"

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PipelineStage(BaseModel):
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


def _repo_root() -> str:
    return "/workspace-data/RENDEREXPO-AI-Studio-Backend"


def _abs_repo_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(_repo_root(), p))


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
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _write_meta(job_folder: str, meta: Dict[str, Any]) -> None:
    meta_file = _meta_path(job_folder)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)


def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat()


def _now_epoch() -> int:
    return int(time.time())


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
                "to apply locked presets."
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
    """
    prompt = params.get("prompt")
    negative_prompt = params.get("negative_prompt")
    seed = params.get("seed")
    strength = params.get("strength", 0.22)

    base: Dict[str, Any] = {
        "created_at": _now_iso(),
        "engine": SD35_ENGINE_NAME,
        "model_name": SD35_ENGINE_NAME,
        "category": category,
        "shot": shot,
        "seed": seed,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "status": "planned",
        "mode": "planned-local",
        "denoise": 0.0,
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

    apply_preset_to_meta(base, category=category, shot=shot, upscale_2x=upscale_2x)

    base["denoise"] = 0.0
    if stage_type in ("img2img", "controlnet"):
        base["strength"] = float(strength)
    if isinstance(base.get("upscale"), dict):
        base["upscale"]["denoise"] = 0.0

    return base


def _detect_overrides(params: Dict[str, Any]) -> Dict[str, Any]:
    locked_keys = ["width", "height", "steps", "num_inference_steps", "guidance_scale", "cfg", "denoise"]
    overrides: Dict[str, Any] = {}
    for k in locked_keys:
        if k in params and params.get(k) is not None:
            overrides[k] = params.get(k)
    return overrides


def _infer_date_and_job_id(job_folder: str) -> Tuple[str, str]:
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
        raise HTTPException(status_code=400, detail=f"Failed opening image: {exc}") from exc


def _save_image(img: Image.Image, out_path: str) -> None:
    try:
        ext = os.path.splitext(out_path)[1].lower()
        if ext == ".png":
            img.save(out_path, format="PNG", optimize=True)
        else:
            rgb = img.convert("RGB")
            rgb.save(out_path, format="JPEG", quality=95, optimize=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed saving image: {exc}") from exc


def _run_deterministic_upscale(*, job_folder: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    REAL upscale execution (no GPU):
    - deterministic resize only
    - no diffusion
    """
    enabled = True if "enabled" not in params else bool(params.get("enabled"))
    factor = int(params.get("factor", 2))
    method = str(params.get("method", "lanczos")).lower().strip()
    denoise = float(params.get("denoise", 0.0))

    if denoise != 0.0:
        raise HTTPException(status_code=400, detail="Upscale denoise must be 0.0 (NO diffusion).")
    if method != "lanczos":
        raise HTTPException(status_code=400, detail="Upscale method must be 'lanczos'.")
    if factor not in (2, 4):
        raise HTTPException(status_code=400, detail="Upscale factor must be 2 or 4.")

    if enabled is False:
        return {
            "enabled": False,
            "ran_at": _now_iso(),
            "denoise": 0.0,
            "note": "Upscale disabled by request.",
        }

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

    if params.get("output_image"):
        out_name = _ensure_png_jpg_name(str(params["output_image"]))
    else:
        ext = os.path.splitext(in_name)[1].lower()
        out_name = f"output_upscaled_{factor}x.jpg" if ext in (".jpg", ".jpeg") else f"output_upscaled_{factor}x.png"

    in_path = os.path.join(job_folder, in_name)
    out_path = os.path.join(job_folder, out_name)

    img = _open_image(in_path)
    w, h = img.size
    new_w, new_h = w * factor, h * factor

    try:
        up = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    except Exception:
        up = img.resize((new_w, new_h), resample=Image.LANCZOS)

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


def _gpu_dispatch_url() -> str:
    """
    GPU dispatch endpoint for pipeline-triggered VR stages.
    """
    return os.getenv("PIPELINE_GPU_DISPATCH_URL", "http://127.0.0.1:8002/api/gpu/dispatch").strip()


def _compute_signature(secret: str, timestamp: str, nonce: str, body: bytes) -> str:
    prefix = f"{timestamp}\n{nonce}\n".encode("utf-8")
    msg = prefix + (body or b"")
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def _dispatch_vr_to_gpu(job_folder: str, meta: Dict[str, Any], vr_mode: VRMode) -> Dict[str, Any]:
    url = _gpu_dispatch_url()

    secret = (os.getenv(HMAC_SECRET_ENV) or "").strip()
    if not secret or len(secret) < 32:
        raise HTTPException(
            status_code=500,
            detail=f"Missing/weak {HMAC_SECRET_ENV}. Pipeline VR dispatch requires HMAC.",
        )

    payload = {
        "job_type": "vr_reconstruct",
        "job_folder": _abs_repo_path(job_folder),
        "meta": meta,
        "pipeline_key": f"vr::{vr_mode}",
        "vr_mode": vr_mode,
    }

    body_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ts = str(_now_epoch())
    nonce = uuid.uuid4().hex
    sig = _compute_signature(secret=secret, timestamp=ts, nonce=nonce, body=body_bytes)

    headers = {
        "Content-Type": "application/json",
        SIG_HEADER: sig,
        TS_HEADER: ts,
        NONCE_HEADER: nonce,
    }

    try:
        r = requests.post(url, data=body_bytes, headers=headers, timeout=(10, 1800))
        if not (200 <= r.status_code < 300):
            raise HTTPException(status_code=502, detail=f"GPU dispatch HTTP {r.status_code}: {r.text[:2000]}")
        return r.json() if r.content else {"status": "ok"}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"GPU dispatch failed: {exc}") from exc


def _normalize_vr_mode(params: Dict[str, Any], meta: Dict[str, Any]) -> VRMode:
    vr_mode = params.get("vr_mode") or meta.get("vr_mode") or "gaussian_splat"
    if vr_mode not in ("gaussian_splat", "nerf", "mesh"):
        raise HTTPException(status_code=400, detail="VR stage vr_mode must be gaussian_splat | nerf | mesh")
    return vr_mode  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Route: PLAN
# ---------------------------------------------------------------------------


@router.post("/plan", response_model=PipelinePlanResponse)
async def plan_pipeline(request: PipelinePlanRequest) -> PipelinePlanResponse:
    """
    Planning-only.
    - writes meta["pipeline_plan"]
    - injects resolved SD3.5 meta for SD3.5 stages
    - injects deterministic upscale plan for upscale stages
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
                "note": "These keys are preset-locked and will be ignored or overridden.",
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
            stage_entry["preset_locked"] = True

        elif stage_type in ("product", "product_insert"):
            rerender_flag = bool(params.get("rerender_with_sd35", False) or params.get("harmonize_with_sd35", False))
            if rerender_flag:
                category, shot, upscale_2x = _require_sd35_selectors(params, idx, stage_type)
                resolved_sd35_meta = _build_sd35_meta_for_stage(
                    stage_type="img2img",
                    params={
                        "prompt": params.get("prompt"),
                        "negative_prompt": params.get("negative_prompt"),
                        "seed": params.get("seed"),
                        "strength": params.get("strength", 0.22),
                        "input_image": params.get("input_image", "product_insert_raw.png"),
                        "planned_output_image": params.get("planned_output_image", "product_insert_result.png"),
                        **params,
                    },
                    category=category,
                    shot=shot,
                    upscale_2x=upscale_2x,
                )
                stage_entry["resolved_sd35_meta"] = resolved_sd35_meta
                stage_entry["preset_locked"] = True
            else:
                stage_entry["preset_locked"] = False

        elif stage_type == "upscale":
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

            in_name = _ensure_png_jpg_name(str(params["input_image"])) if params.get("input_image") else "output.png"
            out_name = _ensure_png_jpg_name(str(params["output_image"])) if params.get("output_image") else f"output_upscaled_{factor}x.png"

            stage_entry["resolved_upscale_plan"] = {
                "enabled": bool(enabled),
                "factor": factor,
                "method": "lanczos",
                "denoise": 0.0,
                "input_image": in_name,
                "output_image": out_name,
                "planned_at": _now_iso(),
                "mode": "plan-only",
                "note": "Deterministic upscale only. No diffusion.",
            }
            stage_entry["preset_locked"] = True

        elif stage_type == "vr":
            vr_mode = params.get("vr_mode", meta.get("vr_mode", "gaussian_splat"))
            if vr_mode not in ("gaussian_splat", "nerf", "mesh"):
                raise HTTPException(status_code=400, detail=f"Stage {idx} (vr) invalid vr_mode: {vr_mode}")

            stage_entry["resolved_vr_plan"] = {
                "vr_mode": vr_mode,
                "pipeline_key": f"vr::{vr_mode}",
                "planned_at": _now_iso(),
                "mode": "gpu-dispatch",
                "note": "VR execution is dispatched to the GPU worker during /run.",
            }
            stage_entry["preset_locked"] = False

        else:
            stage_entry["preset_locked"] = False

        resolved_stages.append(stage_entry)

    pipeline_plan = {
        "created_at": _now_iso(),
        "planner": "pipeline_planner_locked",
        "stages": resolved_stages,
    }

    meta["pipeline_plan"] = pipeline_plan
    meta["pipeline_mode"] = "planned"
    meta["last_updated"] = _now_iso()

    _write_meta(job_folder, meta)

    return PipelinePlanResponse(
        status="ok",
        message="Pipeline planned. SD3.5 stages include locked resolved meta blocks. Upscale stages include deterministic plans.",
        job_folder=job_folder,
        pipeline=pipeline_plan,
        meta_path=_meta_path(job_folder),
    )


# ---------------------------------------------------------------------------
# Route: RUN
# ---------------------------------------------------------------------------


@router.post("/run", response_model=PipelineRunResponse)
async def run_pipeline(request: PipelinePlanRequest) -> PipelineRunResponse:
    """
    Execute supported stages now.

    Supported:
    - vr      -> REAL GPU dispatch
    - upscale -> deterministic local resize

    SD3.5 execution is not done here.
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
            view_files = sorted([p.name for p in Path(job_folder).glob("view_*.png")])

            if len(view_files) < 3:
                view_files = sorted([p.name for p in Path(job_folder).glob("view_*.jpg")])

            if len(view_files) < 3:
                view_files = sorted([p.name for p in Path(job_folder).glob("view_*.jpeg")])

            if len(view_files) < 3:
                iv = meta.get("input_views")
                if isinstance(iv, list):
                    extracted: List[str] = []
                    for item in iv:
                        if isinstance(item, str):
                            extracted.append(item)
                        elif isinstance(item, dict) and isinstance(item.get("file"), str):
                            extracted.append(item["file"])
                    view_files = [x for x in extracted if (Path(job_folder) / x).is_file()]

            if len(view_files) < 3:
                raise HTTPException(
                    status_code=400,
                    detail=f"Stage {idx} (vr) requires 3+ images in job_folder named view_*.png or view_*.jpg.",
                )

            vr_mode = _normalize_vr_mode(params, meta)
            prompt = params.get("prompt") if isinstance(params, dict) else None
            plan_hint = params.get("plan_hint") if isinstance(params, dict) else None

            meta["type"] = "vr_reconstruct"
            meta["vr_mode"] = vr_mode
            if prompt is not None:
                meta["prompt"] = prompt
            if plan_hint is not None:
                meta["plan_hint"] = plan_hint

            meta.setdefault("dispatch", {})
            meta["pipeline_key"] = f"vr::{vr_mode}"
            meta["last_updated"] = _now_iso()
            _write_meta(job_folder, meta)

            gpu_resp = _dispatch_vr_to_gpu(job_folder=job_folder, meta=meta, vr_mode=vr_mode)

            results["vr"] = {
                "status": "dispatched",
                "vr_mode": vr_mode,
                "viewer_url": f"{public_base}/viewer/index.html",
                "preview_video_url": f"{public_base}/preview.mp4",
                "gpu_response": gpu_resp,
            }

        elif stage_type == "upscale":
            upscale_result = _run_deterministic_upscale(job_folder=job_folder, params=params)

            meta.setdefault("outputs", {})
            meta["upscale"] = upscale_result

            if upscale_result.get("enabled") is True:
                out_name = str(upscale_result.get("output_image"))
                meta["outputs"]["upscaled_image"] = out_name
                results["upscale"] = {
                    "output_image": out_name,
                    "output_url": f"{public_base}/{out_name}",
                    "input_image": upscale_result.get("input_image"),
                    "factor": upscale_result.get("factor"),
                    "method": upscale_result.get("method"),
                }
            else:
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
        message="Pipeline executed for supported stages (vr dispatch, upscale local).",
        job_folder=job_folder,
        results=results,
        meta_path=_meta_path(job_folder),
    )