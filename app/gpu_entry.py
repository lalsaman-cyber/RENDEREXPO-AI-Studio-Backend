"""
GPU Entry FastAPI app for RENDEREXPO AI STUDIO.

Runs on GPU worker (port 8002).

Responsibilities:
- Own the shared SD35Runtime instance for the GPU worker.
- Boot LIGHT by default.
- Lazily load SD3.5 only when needed.
- Expose health / root endpoints.
- Mount the real GPU dispatch router from app.api.gpu.dispatch.

Important:
- This file NEVER decides presets.
- Presets, strengths, steps, CFG, img2img strength, upscale behavior, etc.
  are locked by the planner/routers and saved into meta.json.
- This file executes what meta.json says by exposing the dispatcher routes.
- This file must boot LIGHT and must NOT eagerly load SD3.5 on startup.

RENDEREXPO sketch rule:
- Sketch mode is no longer plain img2img.
- Dedicated sketch jobs are dispatched as:
    job_type = "sd35_sketch_controlnet"
    pipeline_key = "sd35::sd35_sketch_controlnet"

RENDEREXPO img2img ratio note:
- Planner may include:
    * preserve_input_aspect_ratio
    * explicit_dimensions
    * input_width / input_height
    * preset_resolution / resolution_policy
- GPU entry must pass those fields through untouched.
- Runtime decides how to honor them.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI

from app.api.gpu.dispatch import router as gpu_dispatch_router
from app.gpu.sd35 import set_runtime as sd35_set_runtime
from runtime.sd35_runtime import SD35Runtime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw if raw else default


REQUESTED_RUNTIME_MODE = _env_str("SD35_RUNTIME_MODE", "lazy").lower()
ENABLE_REAL_SD35 = _env_flag("RUN_REAL_SD35", False)
PRELOAD_SD35_ON_STARTUP = _env_flag("PRELOAD_SD35_ON_STARTUP", False)
GPU_DEVICE = _env_str("SD35_DEVICE", "cuda")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RENDEREXPO AI STUDIO - GPU Runtime API",
    description=(
        "GPU-side API that receives dispatches from the planner "
        "(port 8012) and executes REAL SD3.5 jobs when enabled."
    ),
    version="0.4.0",
)

# Mount the real dispatcher router.
app.include_router(gpu_dispatch_router)

sd35_runtime: Optional[SD35Runtime] = None


# ---------------------------------------------------------------------------
# Runtime control
# ---------------------------------------------------------------------------

def _runtime_enabled() -> bool:
    """
    Real runtime is allowed only if:
      - RUN_REAL_SD35 is truthy
      - requested mode is lazy or real
    """
    return ENABLE_REAL_SD35 and REQUESTED_RUNTIME_MODE in {"lazy", "real"}


def _runtime_loaded() -> bool:
    return (
        sd35_runtime is not None
        and sd35_runtime.mode == "real"
        and sd35_runtime.is_loaded
    )


def _ensure_runtime_loaded_sync() -> SD35Runtime:
    """
    Lazily initialize and load SD35Runtime exactly once.

    IMPORTANT:
    - Dispatch thread workers call into app.gpu.sd35._get_runtime().
    - To avoid duplicate ownership, we inject the single shared runtime
      into app.gpu.sd35 via set_runtime(...).
    """
    global sd35_runtime

    if not _runtime_enabled():
        raise RuntimeError(
            "SD35 runtime is disabled. "
            "Enable RUN_REAL_SD35=1 and set SD35_RUNTIME_MODE to lazy or real."
        )

    if _runtime_loaded():
        return sd35_runtime  # type: ignore[return-value]

    logger.info("Lazy-loading SD35Runtime now (first real GPU job or preload).")
    runtime = SD35Runtime(mode="real", device=GPU_DEVICE)
    runtime.load()

    if not runtime.is_loaded:
        raise RuntimeError(
            "SD35Runtime failed to load in real mode. "
            "Check model path, runtime settings, and GPU memory."
        )

    sd35_runtime = runtime

    # Inject the shared runtime into the sd35 execution wrapper so
    # dispatch jobs reuse one owner instead of creating a second runtime.
    sd35_set_runtime(runtime)

    logger.info("SD35Runtime load complete and injected into app.gpu.sd35.")
    return runtime


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup() -> None:
    """
    Boot LIGHT.

    We do NOT eagerly load SD3.5 on startup unless PRELOAD_SD35_ON_STARTUP=1.
    This avoids occupying VRAM/RAM just because the GPU worker process started.
    """
    global sd35_runtime
    sd35_runtime = None

    logger.info(
        "GPU entry startup: SD35_RUNTIME_MODE=%s, RUN_REAL_SD35=%s, PRELOAD_SD35_ON_STARTUP=%s, DEVICE=%s",
        REQUESTED_RUNTIME_MODE,
        ENABLE_REAL_SD35,
        PRELOAD_SD35_ON_STARTUP,
        GPU_DEVICE,
    )

    if REQUESTED_RUNTIME_MODE == "real" and ENABLE_REAL_SD35 and PRELOAD_SD35_ON_STARTUP:
        logger.info("PRELOAD_SD35_ON_STARTUP enabled. Attempting eager SD3.5 load.")
        try:
            _ensure_runtime_loaded_sync()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Eager preload failed. GPU worker will remain booted but unloaded: %s", exc)
    else:
        logger.info("GPU worker booted LIGHT. SD3.5 will load lazily on first real job if enabled.")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    global sd35_runtime
    if sd35_runtime is not None:
        try:
            sd35_runtime.unload()
            logger.info("SD35Runtime unloaded on shutdown.")
        finally:
            sd35_runtime = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "message": "GPU Runtime for RENDEREXPO AI STUDIO.",
        "requested_mode": REQUESTED_RUNTIME_MODE,
        "run_real_sd35": ENABLE_REAL_SD35,
        "preload_on_startup": PRELOAD_SD35_ON_STARTUP,
        "real_runtime_loaded": _runtime_loaded(),
        "device": GPU_DEVICE,
        "dispatch_route": "/api/gpu/dispatch",
        "supported_job_types": [
            "sd35_txt2img",
            "sd35_text2img",
            "sd35_img2img",
            "sd35_sketch_controlnet",
            "upscale_2x",
            "video_from_image",
            "video_between_frames",
            "cad_from_image",
            "mesh_from_image",
        ],
    }


@app.get("/api/gpu/health")
async def gpu_health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "gpu_worker",
        "requested_mode": REQUESTED_RUNTIME_MODE,
        "run_real_sd35": ENABLE_REAL_SD35,
        "preload_on_startup": PRELOAD_SD35_ON_STARTUP,
        "real_runtime_loaded": _runtime_loaded(),
        "device": GPU_DEVICE,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "dispatch_route": "/api/gpu/dispatch",
    }


@app.post("/api/gpu/runtime/load")
async def gpu_runtime_load() -> Dict[str, Any]:
    """
    Optional manual load endpoint for debugging.
    """
    runtime = _ensure_runtime_loaded_sync()
    return {
        "status": "ok",
        "message": "SD35 runtime loaded.",
        "real_runtime_loaded": runtime.is_loaded,
        "device": GPU_DEVICE,
    }