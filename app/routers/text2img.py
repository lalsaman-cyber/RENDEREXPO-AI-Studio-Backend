# app/routers/text2img.py
"""
RENDEREXPO AI STUDIO - SD3.5 Text2Img Router (Planner)

Planner responsibilities:
- Validate request shape
- Create outputs/{date}/{job_id}/
- Build and write meta.json
- Apply centralized locked presets via app.presets_sd35.apply_preset_to_meta(...)
- Dispatch to GPU worker through app.clients.gpu_client.dispatch_sd35_text2img

IMPORTANT:
- This file is planner-side only.
- It must NOT load SD3.5 directly.
- Planner = port 8012
- GPU worker = port 8002
- Root = /workspace-data/RENDEREXPO-AI-Studio-Backend

Locked behavior:
- Presets decide width / height / steps / cfg / PRO / GEO / upscale defaults
- This router stores optional legacy profile labels only
- It does not let legacy profile configs override locked preset multipliers
"""

from __future__ import annotations

import datetime
import json
import os
import uuid
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.clients.gpu_client import dispatch_sd35_text2img
from app.presets_sd35 import apply_preset_to_meta

router = APIRouter(prefix="/api/sd35", tags=["SD3.5 Text2Img"])

# ---------------------------------------------------------------------------
# Config paths for LoRA & Refiner profiles (optional / legacy metadata only)
# ---------------------------------------------------------------------------

CONFIG_DIR = "config"
LORA_PROFILES_PATH = os.path.join(CONFIG_DIR, "lora_profiles.json")
REFINER_PROFILES_PATH = os.path.join(CONFIG_DIR, "refiner_profiles.json")

LORA_PROFILES: Dict[str, Any] = {}
REFINER_PROFILES: Dict[str, Any] = {}

Category = Literal["urban", "suburban", "interior", "wide_hero"]
Shot = Literal["wide", "close"]


def _load_json_file(path: str) -> Dict[str, Any]:
    """
    Safely load a JSON file.
    If the file does not exist or is invalid, return an empty dict.
    """
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# Load profiles at import time
LORA_PROFILES = _load_json_file(LORA_PROFILES_PATH)
REFINER_PROFILES = _load_json_file(REFINER_PROFILES_PATH)


def _ensure_job_folder(base_outputs_dir: str = "outputs") -> str:
    """
    Create outputs/{YYYY-MM-DD}/{job_id}/ and return its path.
    """
    today_str = datetime.date.today().isoformat()
    job_id = uuid.uuid4().hex
    job_folder = os.path.join(base_outputs_dir, today_str, job_id)
    os.makedirs(job_folder, exist_ok=True)

    try:
        with open(os.path.join(job_folder, "job_type.txt"), "w", encoding="utf-8") as f:
            f.write("sd35_text2img")
    except Exception:
        pass

    return job_folder


def _validate_lora_profile(name: Optional[str]) -> Optional[Dict[str, Any]]:
    if not name:
        return None
    if name not in LORA_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown lora_profile: '{name}'. Check {LORA_PROFILES_PATH}.",
        )
    return LORA_PROFILES[name]


def _validate_refiner_profile(name: Optional[str]) -> Optional[Dict[str, Any]]:
    if not name:
        return None
    if name not in REFINER_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown refiner_profile: '{name}'. Check {REFINER_PROFILES_PATH}.",
        )
    return REFINER_PROFILES[name]


def _build_detail_pass(
    mode: Literal["off", "standard", "strong"],
    custom: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Produces meta["detail_pass"] that the GPU runtime understands.

    NOTE:
    - This is a post-process clarity pass concept, not diffusion denoise.
    """
    presets = {
        "off": {"enabled": False},
        "standard": {"enabled": True, "amount": 1.0, "radius": 1.15, "threshold": 3},
        "strong": {"enabled": True, "amount": 1.6, "radius": 1.25, "threshold": 2},
    }
    dp = presets.get(mode, presets["standard"]).copy()
    if custom and isinstance(custom, dict):
        dp.update(custom)
        if "enabled" not in dp:
            dp["enabled"] = True
    return dp


def _output_public_url(job_folder: str) -> Optional[str]:
    """
    Convert outputs/YYYY-MM-DD/JOBID to /outputs/YYYY-MM-DD/JOBID/output.png
    """
    parts = os.path.normpath(job_folder).split(os.sep)
    if len(parts) < 3:
        return None
    date_str = parts[-2]
    job_id = parts[-1]
    return f"/outputs/{date_str}/{job_id}/output.png"


# ---------------------------------------------------------------------------
# Pydantic request model
# ---------------------------------------------------------------------------

class SD35Text2ImgRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Main text prompt for SD3.5.")
    negative_prompt: Optional[str] = Field(default=None)

    # Locked preset selectors
    category: Category = Field(..., description="Preset category.")
    shot: Shot = Field(..., description="Preset shot (wide/close).")

    # Optional per-request upscale override
    upscale_2x: Optional[bool] = Field(
        default=None,
        description="Optional per-request upscale (true/false). If omitted, preset default applies.",
    )

    # Optional seed
    seed: Optional[int] = Field(default=None)

    # Compatibility inputs (accepted but ignored; presets override them)
    width: Optional[int] = Field(default=None, description="Ignored (preset-locked).")
    height: Optional[int] = Field(default=None, description="Ignored (preset-locked).")
    num_inference_steps: Optional[int] = Field(default=None, description="Ignored (preset-locked).")
    guidance_scale: Optional[float] = Field(default=None, description="Ignored (preset-locked).")

    # Labels only (stored, not used to change locked preset knobs)
    style_preset: Optional[str] = None
    material_preset: Optional[str] = None
    lighting_preset: Optional[str] = None

    # Optional / legacy profiles (stored only)
    lora_profile: Optional[str] = Field(
        default=None,
        description="Optional legacy profile name (stored only; locked preset multipliers remain primary).",
    )
    refiner_profile: Optional[str] = Field(
        default=None,
        description="Optional legacy profile name (stored only; locked preset multipliers remain primary).",
    )

    render_mode: Literal["precise", "balanced", "creative"] = Field(
        default="balanced",
        description="Planner metadata only; GPU runtime may interpret it later.",
    )

    detail_mode: Literal["off", "standard", "strong"] = Field(
        default="standard",
        description="Post-process clarity boost metadata for GPU runtime.",
    )

    detail_pass: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional detail pass override dict: {enabled, amount, radius, threshold}.",
    )


@router.post("/render")
async def sd35_render(request: SD35Text2ImgRequest) -> Dict[str, Any]:
    """
    SD3.5 Text2Img endpoint (planner job creation + GPU dispatch).

    Flow:
    - Validate optional profiles
    - Create job folder
    - Build meta.json with centralized preset system
    - Dispatch to GPU worker
    """
    prompt_clean = (request.prompt or "").strip()
    if not prompt_clean:
        raise HTTPException(status_code=400, detail="prompt is required")

    # Validate optional profiles (stored only)
    lora_cfg = _validate_lora_profile(request.lora_profile)
    refiner_cfg = _validate_refiner_profile(request.refiner_profile)

    # Create job folder
    job_folder = _ensure_job_folder(base_outputs_dir="outputs")
    job_id = os.path.basename(job_folder)
    meta_path = os.path.join(job_folder, "meta.json")
    planned_output_image_abs = os.path.join(job_folder, "output.png")
    public_output_url = _output_public_url(job_folder)

    # Seed
    seed = request.seed if request.seed is not None else int(uuid.uuid4().int % 1_000_000_000)

    # Build base meta
    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "type": "text2img",
        "model_name": "sd35_large_pro_v2_1",
        "engine": "sd35_large_pro_v2_1",
        "prompt": prompt_clean,
        "negative_prompt": request.negative_prompt,
        "seed": seed,
        "category": request.category,
        "shot": request.shot,
        "upscale_request": request.upscale_2x,
        "style_preset": request.style_preset,
        "material_preset": request.material_preset,
        "lighting_preset": request.lighting_preset,
        "planned_output_image": "output.png",
        "status": "planned",
        "render_mode": request.render_mode,
        "detail_pass": _build_detail_pass(request.detail_mode, request.detail_pass),
        "optional_profiles": {
            "lora_profile": request.lora_profile,
            "lora_profile_config": lora_cfg,
            "refiner_profile": request.refiner_profile,
            "refiner_profile_config": refiner_cfg,
        },
        "mode": "planner-dispatch",
        "pipeline_key": "sd35::sd35_text2img",
    }

    # Apply centralized locked preset system
    try:
        apply_preset_to_meta(
            meta,
            category=request.category,
            shot=request.shot,
            upscale_2x=request.upscale_2x,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Preset error: {exc}") from exc

    # Write meta.json
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to write meta.json: {exc}") from exc

    # Dispatch to GPU worker
    ok, gpu_resp = dispatch_sd35_text2img(job_folder=job_folder, meta=meta)

    if not ok:
        return {
            "status": "gpu_error",
            "message": "Job planned but GPU worker failed.",
            "job_folder": job_folder,
            "meta_path": meta_path,
            "planned_output_image": planned_output_image_abs,
            "public_output_url": public_output_url,
            "gpu_error": gpu_resp,
            "preset_applied": meta.get("preset", {}),
        }

    return {
        "status": "dispatched",
        "message": "Text2Img job dispatched to GPU worker.",
        "job_folder": job_folder,
        "meta_path": meta_path,
        "output_image": planned_output_image_abs,
        "public_output_url": public_output_url,
        "gpu_response": gpu_resp,
        "preset_applied": meta.get("preset", {}),
    }


# ---------------------------------------------------------------------------
# Debug endpoints
# ---------------------------------------------------------------------------

@router.get("/config/lora-profiles")
async def list_lora_profiles() -> Dict[str, Any]:
    return {
        "status": "ok",
        "source": LORA_PROFILES_PATH,
        "profiles": list(LORA_PROFILES.keys()),
    }


@router.get("/config/refiner-profiles")
async def list_refiner_profiles() -> Dict[str, Any]:
    return {
        "status": "ok",
        "source": REFINER_PROFILES_PATH,
        "profiles": list(REFINER_PROFILES.keys()),
    }