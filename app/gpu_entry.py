# app/gpu_entry.py
"""
GPU Entry FastAPI app for RENDEREXPO AI STUDIO.

This app is meant to run on a GPU environment (e.g., RunPod), NOT your laptop.

Responsibilities:
- Receive GPU dispatch requests (/api/gpu/dispatch) from the local app (8000).
- Update job meta.json files.
- In "skeleton" mode:
    * Only updates statuses and creates dummy PNGs.
- In "real" mode (SD35_RUNTIME_MODE=real and RUN_REAL_SD35=1):
    * Loads SD3.5 Large via SD35Runtime.
    * Actually runs txt2img for text2img jobs.
    * Saves real output.png in the job folder.
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

        if sd35_runtime.mode != "real":
            logger.warning(
                "SD35Runtime failed to stay in real mode (likely missing model or imports). "
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


def _create_dummy_png(job_folder: str, filename: str = "output.png") -> str:
    """
    Create a simple 512x512 gray PNG for skeleton testing.
    Requires Pillow, but if it's missing we fail silently.
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
        "real_runtime_loaded": sd35_runtime is not None,
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
          and create a dummy output.png
    """
    global sd35_runtime

    job_folder = payload.job_folder
    _ensure_job_folder(job_folder)

    # Start from the meta on disk, merge with payload.meta (payload wins)
    try:
        meta = _read_meta(job_folder)
    except HTTPException:
        meta = {}

    meta.update(payload.meta or {})
    meta["job_folder"] = job_folder
    meta["dispatched_at"] = datetime.utcnow().isoformat()

    # REAL SD3.5 PATH
    if sd35_runtime is not None and meta.get("type") == "text2img":
        try:
            updated_meta = sd35_runtime.generate_text2img(job_folder, meta)
            _write_meta(job_folder, updated_meta)
            return {
                "status": "ok",
                "message": "GPU dispatch completed in REAL SD3.5 mode.",
                "job_folder": job_folder,
                "meta": updated_meta,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("SD35Runtime.generate_text2img() failed: %s", exc)
            # Fall through to skeleton behavior below.

    # SKELETON PATH
    _create_dummy_png(job_folder, filename="output.png")
    meta["status"] = "dispatched_skeleton"
    meta["mode"] = "skeleton-no-inference"
    _write_meta(job_folder, meta)

    return {
        "status": "ok",
        "message": "GPU dispatch received and meta status updated (skeleton, no inference).",
        "job_folder": job_folder,
        "meta": meta,
    }
