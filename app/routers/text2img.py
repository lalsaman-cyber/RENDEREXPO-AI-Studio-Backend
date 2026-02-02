# app/routers/text2img.py
"""
RENDEREXPO AI STUDIO - SD3.5 Text2Img Router

CRITICAL (Doc 18):
- This endpoint MUST write meta using the locked preset system:
  steps, CFG, LyCORIS(PRO 2.1) multiplier, GEO multiplier, resolution
- NO denoise anywhere (denoise always 0.0)
- Upscale is OPTIONAL and must be explicitly requested per job (or preset default)

This router:
- Validates optional LoRA/refiner profiles (legacy/extra metadata only)
- Creates outputs/{date}/{job_id}/
- Writes meta.json
- Dispatches to GPU worker via app.clients.gpu_client.dispatch_sd35_text2img
"""

from __future__ import annotations

import os
import uuid
import json
import datetime
from typing import Optional, Dict, Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.clients.gpu_client import dispatch_sd35_text2img
from app.presets_sd35 import apply_preset_to_meta

router = APIRouter(prefix="/api/sd35", tags=["SD3.5 Text2Img"])

# ---------------------------------------------------------------------------
# Config paths for LoRA & Refiner profiles (optional / legacy)
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

    # marker file
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
    Produces meta["detail_pass"] that your GPU runtime understands.

    NOTE:
    - This is separate from diffusion denoise.
    - Diffusion denoise remains HARD-LOCKED to 0.0 everywhere.
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


# ---------------------------------------------------------------------------
# Pydantic request model
# ---------------------------------------------------------------------------

class SD35Text2ImgRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Main text prompt for SD3.5.")
    negative_prompt: Optional[str] = Field(default=None)

    # Doc 18 preset selectors (LOCKED SYSTEM)
    category: Category = Field(..., description="Doc 18 preset category.")
    shot: Shot = Field(..., description="Doc 18 preset shot (wide/close).")

    # Upscale is OPTIONAL per job
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

    # Optional / legacy profiles (DO NOT overwrite Doc 18 multipliers)
    lora_profile: Optional[str] = Field(
        default=None,
        description="Optional legacy profile name (stored only; Doc 18 multipliers remain primary).",
    )
    refiner_profile: Optional[str] = Field(
        default=None,
        description="Optional legacy profile name (stored only; Doc 18 multipliers remain primary).",
    )

    # Precision modes (stored; GPU runtime may interpret)
    render_mode: Literal["precise", "balanced", "creative"] = Field(
        default="balanced",
        description="Controls how strictly architecture is preserved vs creative variation.",
    )

    # Detail pass control (post-process clarity boost)
    detail_mode: Literal["off", "standard", "strong"] = Field(
        default="standard",
        description="Post-process clarity boost (GPU runtime).",
    )

    detail_pass: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional override dict for detail pass: {enabled, amount, radius, threshold}.",
    )


@router.post("/render")
async def sd35_render(request: SD35Text2ImgRequest):
    """
    SD3.5 Text2Img endpoint (job creation + GPU dispatch).

    Flow:
    - Validate optional profiles
    - Create job folder
    - Build meta.json with Doc 18 locked preset system
    - HARD LOCK: denoise = 0.0 always (and upscale.denoise = 0.0 if present)
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

    # Seed
    seed = request.seed if request.seed is not None else int(uuid.uuid4().int % 1_000_000_000)

    # Build base meta (preset-locked will fill width/height/steps/cfg + multipliers)
    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": datetime.datetime.utcnow().isoformat(),

        "type": "text2img",

        # Always use PRO 2.1 (your locked rule)
        "model_name": "sd3.5-large-pro-2.1",

        "prompt": prompt_clean,
        "negative_prompt": request.negative_prompt,
        "seed": seed,

        # Doc 18 selectors (these drive the preset system)
        "category": request.category,
        "shot": request.shot,

        # Traceability for decision-making (apply_preset_to_meta decides final upscale)
        "upscale_request": request.upscale_2x,

        # HARD LOCK: no denoise anywhere
        "denoise": 0.0,

        # Labels only
        "style_preset": request.style_preset,
        "material_preset": request.material_preset,
        "lighting_preset": request.lighting_preset,

        # Output
        "planned_output_image": "output.png",
        "status": "planned",

        # Modes
        "render_mode": request.render_mode,
        "detail_pass": _build_detail_pass(request.detail_mode, request.detail_pass),

        # Optional/legacy profiles are stored but must NOT override Doc 18 preset multipliers
        "optional_profiles": {
            "lora_profile": request.lora_profile,
            "lora_profile_config": lora_cfg,
            "refiner_profile": request.refiner_profile,
            "refiner_profile_config": refiner_cfg,
        },

        # Runtime decides skeleton vs real
        "mode": "skeleton-or-real",
    }

    # Apply Doc 18 locked preset system (this injects steps/cfg/resolution + lycoris/geo configs)
    try:
        apply_preset_to_meta(
            meta,
            category=request.category,
            shot=request.shot,
            upscale_2x=request.upscale_2x,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Preset error: {exc}") from exc

    # Safety hard-lock again
    meta["denoise"] = 0.0
    if isinstance(meta.get("upscale"), dict):
        meta["upscale"]["denoise"] = 0.0

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
            "gpu_error": gpu_resp,
            "preset_applied": meta.get("preset", {}),
        }

    return {
        "status": "dispatched",
        "message": "Text2Img job dispatched to GPU worker (Doc 18 locked presets).",
        "job_folder": job_folder,
        "meta_path": meta_path,
        "output_image": planned_output_image_abs,
        "gpu_response": gpu_resp,
        "preset_applied": meta.get("preset", {}),
    }


# ---------------------------------------------------------------------------
# Debug endpoints
# ---------------------------------------------------------------------------

@router.get("/config/lora-profiles")
async def list_lora_profiles():
    return {
        "status": "ok",
        "source": LORA_PROFILES_PATH,
        "profiles": list(LORA_PROFILES.keys()),
    }


@router.get("/config/refiner-profiles")
async def list_refiner_profiles():
    return {
        "status": "ok",
        "source": REFINER_PROFILES_PATH,
        "profiles": list(REFINER_PROFILES.keys()),
    }
