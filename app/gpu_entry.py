# app/gpu_entry.py
"""
GPU Entry FastAPI app for RENDEREXPO AI STUDIO.

This app is meant to run on a GPU environment (e.g., RunPod), NOT your laptop.

Responsibilities:
- Receive GPU dispatch requests (/api/gpu/dispatch) from the local app.
- Update job meta.json files.
- In "skeleton" mode:
    * Only updates statuses and (optionally) creates dummy PNGs.
- In "real" mode (SD35_RUNTIME_MODE=real):
    * Loads SD3.5 Large via SD35Runtime.
    * Actually runs txt2img for text2img jobs.
    * Saves real output.png in the job folder.

IMPORTANT:
- Default is skeleton (no heavy model load).
- Enable real SD3.5 only on GPU by setting:
    SD35_RUNTIME_MODE=real
"""

import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from runtime.sd35_runtime import SD35Runtime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Environment flags
# ---------------------------------------------------------------------------

def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean-like env var."""
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    return raw in ("1", "true", "yes", "on")


# If this is True AND SD35_RUNTIME_MODE=real, we will attempt real SD3.5 txt2img.
ENABLE_REAL_SD35 = _env_flag("RUN_REAL_SD35", False)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RENDEREXPO AI STUDIO - GPU Runtime API",
    description=(
        "GPU-side API that receives dispatches from the local dev server "
        "(port 8000) and either:\n"
        "- updates job meta (skeleton), or\n"
        "- runs real SD3.5 txt2img when enabled."
    ),
    version="0.1.0",
)

sd35_runtime: Optional[SD35Runtime] = None  # set on startup if needed


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class GPUDispatchPayload(BaseModel):
    job_folder: str = Field(..., description="Path to the job folder under outputs/...")
    meta: Dict[str, Any] = Field(
        ..., description="The meta.json contents sent by the local API."
    )


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    global sd35_runtime

    # Decide if we're in real SD3.5 mode or skeleton
    runtime_mode = os.getenv("SD35_RUNTIME_MODE", "skeleton").lower()
    logger.info(
        "GPU entry startup: SD35_RUNTIME_MODE=%s, RUN_REAL_SD35=%s",
        runtime_mode,
        ENABLE_REAL_SD35,
    )

    if runtime_mode == "real" and ENABLE_REAL_SD35:
        # Initialize real SD35 runtime
        logger.info("Initializing SD35Runtime in REAL mode (GPU).")
        sd35_runtime = SD35Runtime(mode="real", device="cuda")
        sd35_runtime.load()
    else:
        logger.info(
            "Running in SKELETON mode (no SD3.5 load). "
            "Set SD35_RUNTIME_MODE=real and RUN_REAL_SD35=1 on GPU to enable real generation."
        )
        sd35_runtime = None


@app.on_event("shutdown")
async def on_shutdown():
    global sd35_runtime
    if sd35_runtime is not None:
        sd35_runtime.unload()
        sd35_runtime = None
        logger.info("SD35Runtime unloaded on shutdown.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _meta_path(job_folder: str) -> str:
    return os.path.join(job_folder, "meta.json")


def _ensure_job_folder(job_folder: str) -> None:
    if not os.path.isdir(job_folder):
        raise HTTPException(
            status_code=400,
            detail=f"job_folder does not exist: {job_folder}",
        )


def _read_meta(job_folder: str) -> Dict[str, Any]:
    meta_file = _meta_path(job_folder)
    if not os.path.isfile(meta_file):
        raise HTTPException(
            status_code=400,
            detail=f"meta.json not found in job_folder: {meta_file}",
        )
    with open(meta_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_meta(job_folder: str, meta: Dict[str, Any]) -> None:
    meta_file = _meta_path(job_folder)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)


def _create_dummy_png(job_folder: str, filename: str = "output.png") -> str:
    """
    Create a simple 512x512 gray PNG for skeleton testing.
    """
    try:
        from PIL import Image
    except Exception as e:  # noqa: BLE001
        logger.warning("PIL not available for dummy PNG: %s", e)
        return filename

    os.makedirs(job_folder, exist_ok=True)
    img = Image.new("RGB", (512, 512), (64, 64, 64))
    out_path = os.path.join(job_folder, filename)
    img.save(out_path, format="PNG")
    return out_path


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "message": "GPU Runtime for RENDEREXPO AI STUDIO.",
        "mode": os.getenv("SD35_RUNTIME_MODE", "skeleton"),
        "run_real_sd35": ENABLE_REAL_SD35,
    }


@app.post("/api/gpu/dispatch")
async def gpu_dispatch(payload: GPUDispatchPayload):
    """
    Called by the local API (port 8000):

    - Ensures job_folder exists
    - Reads meta.json (and/or uses provided meta)
    - If REAL SD3.5 is enabled and job type is 'text2img':
        * runs txt2img
        * writes real output.png
        * updates meta status to 'completed'
    - Else:
        * skeleton behavior: update status to 'dispatched_skeleton'
    """
    global sd35_runtime

    job_folder = payload.job_folder
    incoming_meta = payload.meta

    _ensure_job_folder(job_folder)

    # Prefer the meta from disk, but merge with incoming.
    try:
        meta = _read_meta(job_folder)
    except HTTPException:
        meta = {}

    meta.update(incoming_meta)
    meta.setdefault("job_id", os.path.basename(job_folder))
    meta.setdefault("type", "text2img")
    meta.setdefault("model_name", "sd3.5-large")
    meta.setdefault("mode", "skeleton-no-inference")

    job_type = meta.get("type", "text2img")
    now_iso = datetime.utcnow().isoformat()

    # Decide if we will attempt real SD3.5 or skeleton
    runtime_mode = os.getenv("SD35_RUNTIME_MODE", "skeleton").lower()
    real_mode = (
        runtime_mode == "real"
        and ENABLE_REAL_SD35
        and sd35_runtime is not None
        and sd35_runtime.is_real
    )

    if real_mode and job_type == "text2img":
        # ---------------------------------------------------------------
        # REAL SD3.5 txt2img path
        # ---------------------------------------------------------------
        logger.info("GPU dispatch: running REAL SD3.5 txt2img for job %s", meta["job_id"])

        # Extract generation params
        prompt = meta.get("prompt", "")
        negative_prompt = meta.get("negative_prompt")
        width = int(meta.get("width", 1024))
        height = int(meta.get("height", 1024))
        steps = int(meta.get("num_inference_steps", 25))
        guidance_scale = float(meta.get("guidance_scale", 6.0))
        seed_raw = meta.get("seed", None)
        seed: Optional[int]
        if seed_raw is None:
            seed = None
        else:
            try:
                seed = int(seed_raw)
            except Exception:  # noqa: BLE001
                seed = None

        planned_output_rel = meta.get("planned_output_image", "output.png")
        output_filename = os.path.basename(planned_output_rel)
        output_path = os.path.join(job_folder, output_filename)

        # Run generation
        try:
            image = sd35_runtime.generate_text2img(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
            os.makedirs(job_folder, exist_ok=True)
            image.save(output_path, format="PNG")
        except Exception as e:  # noqa: BLE001
            logger.exception("Error during SD3.5 txt2img generation: %s", e)
            # Mark meta as failed
            meta["status"] = "failed"
            meta["mode"] = "real-sd35-error"
            meta["error"] = str(e)
            meta["failed_at"] = now_iso
            _write_meta(job_folder, meta)
            raise HTTPException(
                status_code=500,
                detail=f"SD3.5 txt2img generation failed: {e}",
            )

        # Update meta for completed job
        meta["status"] = "completed"
        meta["mode"] = "real-sd35"
        meta["dispatched_at"] = meta.get("dispatched_at", now_iso)
        meta["completed_at"] = now_iso
        meta["output_image"] = output_filename

        _write_meta(job_folder, meta)

        return {
            "status": "ok",
            "message": "Real SD3.5 txt2img completed on GPU.",
            "job_folder": job_folder,
            "output_image": output_path,
            "meta": meta,
        }

    else:
        # ---------------------------------------------------------------
        # SKELETON path (no real generation)
        # ---------------------------------------------------------------
        logger.info(
            "GPU dispatch in SKELETON mode for job %s (type=%s).",
            meta["job_id"],
            job_type,
        )

        meta["status"] = "dispatched_skeleton"
        meta["mode"] = "skeleton-no-inference"
        meta["dispatched_at"] = now_iso

        _write_meta(job_folder, meta)

        return {
            "status": "ok",
            "message": "GPU dispatch received and meta status updated (skeleton, no inference).",
            "job_folder": job_folder,
            "meta": meta,
        }


@app.post("/api/gpu/complete-simulated")
async def gpu_complete_simulated(payload: GPUDispatchPayload):
    """
    Old helper endpoint for your earlier tests.

    Still available if you ever want to fake-complete jobs and create dummy PNGs
    without touching real SD3.5.
    """
    job_folder = payload.job_folder
    _ensure_job_folder(job_folder)

    try:
        meta = _read_meta(job_folder)
    except HTTPException:
        meta = {}

    now_iso = datetime.utcnow().isoformat()

    # Create a dummy output if none exists
    planned_output_rel = meta.get("planned_output_image", "output.png")
    output_filename = os.path.basename(planned_output_rel)
    output_path = os.path.join(job_folder, output_filename)
    if not os.path.isfile(output_path):
        _create_dummy_png(job_folder, output_filename)

    meta["status"] = "completed_skeleton"
    meta["mode"] = "skeleton-no-inference"
    meta["completed_at"] = now_iso
    meta["output_image"] = output_filename

    _write_meta(job_folder, meta)

    return {
        "status": "ok",
        "message": "GPU completion simulated; meta status set to 'completed_skeleton'.",
        "job_folder": job_folder,
        "meta": meta,
    }
