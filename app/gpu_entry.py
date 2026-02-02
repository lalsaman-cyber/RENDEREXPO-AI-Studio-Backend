# app/gpu_entry.py
"""
GPU Entry FastAPI app for RENDEREXPO AI STUDIO.

Runs on GPU (RunPod).

Responsibilities:
- Receive GPU dispatch requests (/api/gpu/dispatch) from the local API (8002).
- Read/merge meta.json in job folder.
- Execute REAL SD3.5 generation via runtime.sd35_runtime.SD35Runtime when enabled.
- Write output.png + update meta.json status.

Important:
- This file NEVER decides presets. Presets are locked in local planning (routers) and saved into meta.json.
- This file simply executes what meta.json says (steps/CFG/multipliers/no-denoise/upscale optional).
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


ENABLE_REAL_SD35 = _env_flag("RUN_REAL_SD35", False)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RENDEREXPO AI STUDIO - GPU Runtime API",
    description=(
        "GPU-side API that receives dispatches from the local dev server "
        "(port 8002) and executes REAL SD3.5 jobs when enabled."
    ),
    version="0.2.0",
)

sd35_runtime: Optional[SD35Runtime] = None  # set on startup


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class GPUDispatchPayload(BaseModel):
    job_folder: str = Field(..., description="Absolute path to job folder under outputs/... on the pod")
    meta: Dict[str, Any] = Field(..., description="The meta.json contents sent by the local API.")


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    global sd35_runtime

    runtime_mode = os.getenv("SD35_RUNTIME_MODE", "skeleton").lower()
    logger.info(
        "GPU entry startup: SD35_RUNTIME_MODE=%s, RUN_REAL_SD35=%s",
        runtime_mode,
        ENABLE_REAL_SD35,
    )

    if runtime_mode == "real" and ENABLE_REAL_SD35:
        logger.info("Initializing SD35Runtime in REAL mode (GPU).")
        sd35_runtime = SD35Runtime(mode="real", device="cuda")
        sd35_runtime.load()

        if sd35_runtime.mode != "real" or sd35_runtime.pipe is None:
            logger.warning(
                "SD35Runtime failed to stay in real mode (likely missing model or VRAM). "
                "Falling back to skeleton behavior."
            )
            sd35_runtime = None
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


def _touch_status(meta: Dict[str, Any], status: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta["status"] = status
    meta["updated_at"] = datetime.utcnow().isoformat()
    if extra:
        meta.update(extra)
    return meta


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "message": "GPU Runtime for RENDEREXPO AI STUDIO.",
        "mode": os.getenv("SD35_RUNTIME_MODE", "skeleton"),
        "run_real_sd35": ENABLE_REAL_SD35,
        "real_runtime_loaded": sd35_runtime is not None,
    }


@app.post("/api/gpu/dispatch")
async def gpu_dispatch(payload: GPUDispatchPayload):
    """
    Called by the local API (8002):

    - Ensures job_folder exists
    - Reads meta.json and merges payload.meta (payload wins)
    - If REAL SD3.5 runtime is loaded:
        - If meta["type"] == "text2img": run generate_text2img()
        - If meta["type"] == "img2img": run generate_img2img() (if available)
        - Else: error
    - Writes updated meta.json and returns it
    """
    global sd35_runtime

    job_folder = payload.job_folder
    _ensure_job_folder(job_folder)

    # Start from meta on disk, merge with payload.meta (payload wins)
    try:
        meta = _read_meta(job_folder)
    except HTTPException:
        meta = {}

    meta.update(payload.meta or {})
    meta["job_folder"] = job_folder
    meta["dispatched_at"] = datetime.utcnow().isoformat()

    # Always enforce “no denoise anywhere” at dispatch time too (belt + suspenders)
    meta["denoise"] = 0.0
    if isinstance(meta.get("upscale"), dict):
        meta["upscale"]["denoise"] = 0.0

    job_type = (meta.get("type") or "").strip().lower()

    # If runtime not loaded -> fail clearly (local API should treat this as gpu_error)
    if sd35_runtime is None:
        meta = _touch_status(
            meta,
            "failed",
            {
                "error": "runtime_not_loaded",
                "error_detail": "SD35 runtime is not loaded on GPU. Check SD35_RUNTIME_MODE/RUN_REAL_SD35 and logs.",
            },
        )
        _write_meta(job_folder, meta)
        raise HTTPException(status_code=500, detail=meta.get("error_detail"))

    # REAL PATH
    try:
        meta = _touch_status(meta, "running")
        _write_meta(job_folder, meta)

        if job_type == "text2img":
            updated_meta = sd35_runtime.generate_text2img(job_folder, meta)
        elif job_type == "img2img":
            # This will work if you later wire the GPU runtime for img2img.
            # If not supported by your diffusers version, it raises a clear RuntimeError.
            updated_meta = sd35_runtime.generate_img2img(job_folder, meta)
        else:
            raise RuntimeError(f"Unsupported job type for GPU runtime: '{job_type}'")

        _write_meta(job_folder, updated_meta)

        return {
            "status": "ok",
            "message": "GPU dispatch completed in REAL SD3.5 mode.",
            "job_folder": job_folder,
            "meta": updated_meta,
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("GPU dispatch failed: %s", exc)
        meta = _touch_status(
            meta,
            "failed",
            {
                "error": "gpu_dispatch_failed",
                "error_detail": str(exc),
                "failed_at": datetime.utcnow().isoformat(),
            },
        )
        _write_meta(job_folder, meta)
        raise HTTPException(status_code=500, detail=str(exc))
