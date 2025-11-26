# app/routers/text2img.py

import os
import uuid
import json
import datetime
from typing import Optional, Dict, Any

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
        # treat it as "no profiles defined yet"
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        # If there's a JSON error, also treat as empty.
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
# Pydantic model
# ---------------------------------------------------------------------------

class SD35Text2ImgRequest(BaseModel):
    """
    Schema for SD3.5 text-to-image.
    """
    prompt: str = Field(..., description="Main text prompt for SD3.5.")
    negative_prompt: Optional[str] = Field(
        default=None,
        description="Optional negative prompt to avoid bad artifacts.",
    )
    width: int = Field(
        default=1024,
        ge=64,
        le=2048,
        description="Target width of the output image (pixels).",
    )
    height: int = Field(
        default=1024,
        ge=64,
        le=2048,
        description="Target height of the output image (pixels).",
    )
    num_inference_steps: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Number of diffusion steps.",
    )
    guidance_scale: float = Field(
        default=6.0,
        ge=0.0,
        le=20.0,
        description="CFG guidance scale.",
    )
    style_preset: Optional[str] = Field(
        default=None,
        description="Style preset ID (e.g. 'soft_luxury').",
    )
    material_preset: Optional[str] = Field(
        default=None,
        description="Material preset ID (e.g. 'warm_oak_veneer').",
    )
    lighting_preset: Optional[str] = Field(
        default=None,
        description="Lighting preset ID (e.g. 'daylight_soft').",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed. If null, will be random in real GPU runtime.",
    )
    lora_profile: Optional[str] = Field(
        default=None,
        description="Name of LoRA profile defined in config/lora_profiles.json.",
    )
    refiner_profile: Optional[str] = Field(
        default=None,
        description="Name of refiner profile defined in config/refiner_profiles.json.",
    )


# ---------------------------------------------------------------------------
# Helpers for validation
# ---------------------------------------------------------------------------

def _validate_lora_profile(name: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Ensure lora_profile (if provided) exists in LORA_PROFILES.
    Returns the profile dict or None.
    """
    if not name:
        return None
    if name not in LORA_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown lora_profile: '{name}'. Check {LORA_PROFILES_PATH}.",
        )
    return LORA_PROFILES[name]


def _validate_refiner_profile(name: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Ensure refiner_profile (if provided) exists in REFINER_PROFILES.
    Returns the profile dict or None.
    """
    if not name:
        return None
    if name not in REFINER_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown refiner_profile: '{name}'. Check {REFINER_PROFILES_PATH}.",
        )
    return REFINER_PROFILES[name]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/render")
async def sd35_render(request: SD35Text2ImgRequest):
    """
    SD3.5 Text2Img endpoint for RENDEREXPO AI STUDIO.

    Flow:
    - Validate LoRA + refiner profiles (if any).
    - Create job folder under outputs/{date}/{job_id}/.
    - Write meta.json with planned settings.
    - Dispatch the job to GPU worker (port 8001) via /api/gpu/dispatch.
    - Return:
        * status="dispatched" and GPU response, or
        * status="gpu_error" with error details.
    """
    # 1) Validate LoRA + refiner if provided
    lora_cfg = _validate_lora_profile(request.lora_profile)
    refiner_cfg = _validate_refiner_profile(request.refiner_profile)

    # 2) Create job folder
    job_folder = _ensure_job_folder(base_outputs_dir="outputs")
    job_id = os.path.basename(job_folder)
    meta_path = os.path.join(job_folder, "meta.json")
    planned_output_image = os.path.join(job_folder, "output.png")

    # 3) Build meta data
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
        "mode": "skeleton-or-real",  # actual mode decided on GPU
        "lora_profile": request.lora_profile,
        "lora_config": lora_cfg,
        "refiner_profile": request.refiner_profile,
        "refiner_config": refiner_cfg,
    }

    # 4) Write meta.json
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 5) Dispatch to GPU worker (port 8001)
    ok, gpu_resp = dispatch_sd35_text2img(job_folder=job_folder, meta=meta)

    if not ok:
        # GPU worker failed or is unreachable; job remains "planned"
        return {
            "status": "gpu_error",
            "message": "Job planned but GPU worker failed.",
            "job_folder": job_folder,
            "meta_path": meta_path,
            "planned_output_image": planned_output_image,
            "gpu_error": gpu_resp,
        }

    # If GPU accepted the job, it may update meta.json and create output.png
    return {
        "status": "dispatched",
        "message": "Text2Img job dispatched to GPU worker.",
        "job_folder": job_folder,
        "meta_path": meta_path,
        "output_image": planned_output_image,
        "gpu_response": gpu_resp,
    }


# ---------------------------------------------------------------------------
# DEBUG endpoints (optional but helpful)
# ---------------------------------------------------------------------------

@router.get("/config/lora-profiles")
async def list_lora_profiles():
    """
    Small helper to see which LoRA profile keys are currently loaded.
    """
    return {
        "status": "ok",
        "source": LORA_PROFILES_PATH,
        "profiles": list(LORA_PROFILES.keys()),
    }


@router.get("/config/refiner-profiles")
async def list_refiner_profiles():
    """
    Small helper to see which refiner profile keys are currently loaded.
    """
    return {
        "status": "ok",
        "source": REFINER_PROFILES_PATH,
        "profiles": list(REFINER_PROFILES.keys()),
    }
