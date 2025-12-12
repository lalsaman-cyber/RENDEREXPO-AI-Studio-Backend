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
# Pydantic model
# ---------------------------------------------------------------------------

ControlMode = Literal["precise", "balanced", "creative"]


class SD35Text2ImgRequest(BaseModel):
    """
    Schema for SD3.5 text-to-image.
    """
    prompt: str = Field(..., description="Main text prompt for SD3.5.")
    negative_prompt: Optional[str] = Field(
        default=None,
        description="Optional negative prompt to avoid bad artifacts.",
    )

    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)

    num_inference_steps: int = Field(default=25, ge=1, le=100)
    guidance_scale: float = Field(default=6.0, ge=0.0, le=20.0)

    style_preset: Optional[str] = Field(default=None)
    material_preset: Optional[str] = Field(default=None)
    lighting_preset: Optional[str] = Field(default=None)

    seed: Optional[int] = Field(default=None)

    lora_profile: Optional[str] = Field(
        default=None,
        description="Name of LoRA profile defined in config/lora_profiles.json.",
    )
    refiner_profile: Optional[str] = Field(
        default=None,
        description="Name of refiner profile defined in config/refiner_profiles.json.",
    )

    # -----------------------------------------------------------------------
    # NEW: Precision modes + detail pass switches (stored in meta for GPU side)
    # -----------------------------------------------------------------------

    control_mode: ControlMode = Field(
        default="balanced",
        description="Controls how strictly we preserve structure: precise | balanced | creative.",
    )

    detail_pass: bool = Field(
        default=False,
        description="If true, run a second enhancement pass after base generation (GPU real mode only).",
    )

    detail_strength: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="How aggressive the detail pass is (0 = off, higher = more change).",
    )

    upscale_factor: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Upscale multiplier for the detail pass (e.g. 2 = 2x).",
    )


# ---------------------------------------------------------------------------
# Helpers for validation
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
    - Dispatch the job to GPU worker (port 8011) via /api/gpu/dispatch.
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

        # NEW: precision modes + detail pass (GPU uses these)
        "control_mode": request.control_mode,          # precise | balanced | creative
        "detail_pass": request.detail_pass,            # bool
        "detail_strength": request.detail_strength,    # 0..1
        "upscale_factor": request.upscale_factor,      # 1..4
    }

    # 4) Write meta.json
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 5) Dispatch to GPU worker (port 8011)
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
# DEBUG endpoints
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
