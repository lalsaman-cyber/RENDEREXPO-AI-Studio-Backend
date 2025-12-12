# app/routers/text2img.py

import os
import uuid
import json
import datetime
from typing import Optional, Dict, Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.clients.gpu_client import dispatch_sd35_text2img

router = APIRouter(prefix="/api/sd35", tags=["SD3.5 Text2Img"])

# ---------------------------------------------------------------------------
# Config paths for LoRA & Refiner profiles
# ---------------------------------------------------------------------------

CONFIG_DIR = "config"
LORA_PROFILES_PATH = os.path.join(CONFIG_DIR, "lora_profiles.json")
REFINER_PROFILES_PATH = os.path.join(CONFIG_DIR, "refiner_profiles.json")

LORA_PROFILES: Dict[str, Any] = {}
REFINER_PROFILES: Dict[str, Any] = {}


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
    Create an outputs/{YYYY-MM-DD}/{job_id}/ folder and return its path.
    """
    today_str = datetime.date.today().isoformat()
    job_id = uuid.uuid4().hex
    job_folder = os.path.join(base_outputs_dir, today_str, job_id)
    os.makedirs(job_folder, exist_ok=True)
    return job_folder


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

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

    - off      -> disabled
    - standard -> safe clarity boost
    - strong   -> stronger clarity boost (still controlled)

    If custom dict is provided, it overrides presets.
    Expected keys (your runtime supports):
      enabled: bool
      amount: float
      radius: float
      threshold: int
    """
    presets = {
        "off": {"enabled": False},
        "standard": {"enabled": True, "amount": 1.0, "radius": 1.15, "threshold": 3},
        "strong": {"enabled": True, "amount": 1.6, "radius": 1.25, "threshold": 2},
    }
    dp = presets.get(mode, presets["standard"]).copy()
    if custom and isinstance(custom, dict):
        dp.update(custom)
        # If they pass any params but forget enabled, assume enabled
        if "enabled" not in dp:
            dp["enabled"] = True
    return dp


# ---------------------------------------------------------------------------
# Pydantic request model
# ---------------------------------------------------------------------------

class SD35Text2ImgRequest(BaseModel):
    prompt: str = Field(..., description="Main text prompt for SD3.5.")
    negative_prompt: Optional[str] = Field(
        default=None,
        description="Optional negative prompt to avoid bad artifacts.",
    )

    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)

    num_inference_steps: int = Field(default=25, ge=1, le=100)
    guidance_scale: float = Field(default=6.0, ge=0.0, le=20.0)

    style_preset: Optional[str] = None
    material_preset: Optional[str] = None
    lighting_preset: Optional[str] = None

    seed: Optional[int] = Field(default=None)

    # Existing profiles
    lora_profile: Optional[str] = Field(
        default=None,
        description="Name of LoRA profile defined in config/lora_profiles.json.",
    )
    refiner_profile: Optional[str] = Field(
        default=None,
        description="Name of refiner profile defined in config/refiner_profiles.json.",
    )

    # NEW: precision modes (stored into meta, GPU will interpret)
    render_mode: Literal["precise", "balanced", "creative"] = Field(
        default="balanced",
        description="Controls how strictly architecture is preserved vs creative variation.",
    )

    # NEW: detail pass control (groundwork for sharper materials/details)
    detail_mode: Literal["off", "standard", "strong"] = Field(
        default="standard",
        description="Post-process clarity boost (implemented on GPU runtime).",
    )

    # Optional manual override for detail pass params
    detail_pass: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional override dict for detail pass: {enabled, amount, radius, threshold}.",
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/render")
async def sd35_render(request: SD35Text2ImgRequest):
    """
    SD3.5 Text2Img planner endpoint.

    - Validates LoRA + refiner profiles (if any).
    - Creates job folder outputs/{date}/{job_id}/
    - Writes meta.json
    - Dispatches to GPU worker (port 8011) via /api/gpu/dispatch
    """
    # 1) Validate profiles
    lora_cfg = _validate_lora_profile(request.lora_profile)
    refiner_cfg = _validate_refiner_profile(request.refiner_profile)

    # 2) Create job folder
    job_folder = _ensure_job_folder(base_outputs_dir="outputs")
    job_id = os.path.basename(job_folder)
    meta_path = os.path.join(job_folder, "meta.json")
    planned_output_image = os.path.join(job_folder, "output.png")

    # 3) Build meta
    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "type": "text2img",
        "model_name": "sd3.5-large",

        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,

        "width": request.width,
        "height": request.height,
        "num_inference_steps": request.num_inference_steps,
        "guidance_scale": request.guidance_scale,

        "style_preset": request.style_preset,
        "material_preset": request.material_preset,
        "lighting_preset": request.lighting_preset,

        "seed": request.seed,

        "planned_output_image": "output.png",
        "status": "planned",

        # NEW: modes
        "render_mode": request.render_mode,
        "detail_pass": _build_detail_pass(request.detail_mode, request.detail_pass),

        # Profiles
        "lora_profile": request.lora_profile,
        "lora_config": lora_cfg,
        "refiner_profile": request.refiner_profile,
        "refiner_config": refiner_cfg,

        # Runtime decides skeleton vs real
        "mode": "skeleton-or-real",
    }

    # 4) Write meta.json
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 5) Dispatch to GPU worker
    ok, gpu_resp = dispatch_sd35_text2img(job_folder=job_folder, meta=meta)

    if not ok:
        return {
            "status": "gpu_error",
            "message": "Job planned but GPU worker failed.",
            "job_folder": job_folder,
            "meta_path": meta_path,
            "planned_output_image": planned_output_image,
            "gpu_error": gpu_resp,
        }

    return {
        "status": "dispatched",
        "message": "Text2Img job dispatched to GPU worker.",
        "job_folder": job_folder,
        "meta_path": meta_path,
        "output_image": planned_output_image,
        "gpu_response": gpu_resp,
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
