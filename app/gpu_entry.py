"""
GPU Entry FastAPI app for RENDEREXPO AI STUDIO.

Runs on GPU worker (port 8002).

Responsibilities:
- Receive GPU dispatch requests (/api/gpu/dispatch) from the planner (8012).
- Read/merge meta.json in the job folder.
- Execute REAL SD3.5 generation via runtime.sd35_runtime.SD35Runtime only when needed.
- Write output.png / output artifacts + update meta.json status.

Important:
- This file NEVER decides presets.
- Presets, strengths, steps, CFG, img2img strength, upscale behavior, etc.
  are locked by the planner/routers and saved into meta.json.
- This file executes what meta.json says.
- This file must boot LIGHT and must NOT eagerly load SD3.5 on startup.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from runtime.sd35_runtime import SD35Runtime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean-like env var."""
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
    version="0.3.0",
)

sd35_runtime: Optional[SD35Runtime] = None
runtime_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class GPUDispatchPayload(BaseModel):
    job_folder: str = Field(..., description="Absolute path to the job folder under outputs/... on the pod.")
    meta: Dict[str, Any] = Field(..., description="The merged planner meta.json contents.")


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

    logger.info(
        "GPU entry startup: SD35_RUNTIME_MODE=%s, RUN_REAL_SD35=%s, PRELOAD_SD35_ON_STARTUP=%s, DEVICE=%s",
        REQUESTED_RUNTIME_MODE,
        ENABLE_REAL_SD35,
        PRELOAD_SD35_ON_STARTUP,
        GPU_DEVICE,
    )

    sd35_runtime = None

    if REQUESTED_RUNTIME_MODE == "real" and ENABLE_REAL_SD35 and PRELOAD_SD35_ON_STARTUP:
        logger.info("PRELOAD_SD35_ON_STARTUP enabled. Attempting eager SD3.5 load.")
        try:
            await _ensure_runtime_loaded()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Eager preload failed. GPU worker will remain booted but unloaded: %s", exc)
    else:
        logger.info(
            "GPU worker booted LIGHT. SD3.5 will load lazily on first real job if enabled."
        )


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
# Runtime control
# ---------------------------------------------------------------------------

def _runtime_enabled() -> bool:
    """
    Real runtime is allowed only if:
      - RUN_REAL_SD35 is truthy
      - requested mode is lazy or real
    """
    return ENABLE_REAL_SD35 and REQUESTED_RUNTIME_MODE in {"lazy", "real"}


async def _ensure_runtime_loaded() -> SD35Runtime:
    """
    Lazily initialize and load SD35Runtime exactly once.
    Protected by an asyncio lock so concurrent requests do not double-load.
    """
    global sd35_runtime

    if not _runtime_enabled():
        raise RuntimeError(
            "SD35 runtime is disabled. "
            "Enable RUN_REAL_SD35=1 and set SD35_RUNTIME_MODE to lazy or real."
        )

    if sd35_runtime is not None and sd35_runtime.mode == "real" and sd35_runtime.pipe is not None:
        return sd35_runtime

    async with runtime_lock:
        # Check again inside lock
        if sd35_runtime is not None and sd35_runtime.mode == "real" and sd35_runtime.pipe is not None:
            return sd35_runtime

        logger.info("Lazy-loading SD35Runtime now (first real GPU job).")
        runtime = SD35Runtime(mode="real", device=GPU_DEVICE)
        runtime.load()

        if runtime.mode != "real" or runtime.pipe is None:
            raise RuntimeError(
                "SD35Runtime failed to load in real mode. "
                "Check model path, runtime settings, and GPU memory."
            )

        sd35_runtime = runtime
        logger.info("SD35Runtime lazy-load complete.")
        return sd35_runtime


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


def _read_meta_file(meta_file: str) -> Dict[str, Any]:
    if not os.path.isfile(meta_file):
        return {}
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    """
    Atomic write to avoid corrupt meta.json if process crashes mid-write.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix="meta_",
        suffix=".json",
        dir=os.path.dirname(path),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _read_meta(job_folder: str) -> Dict[str, Any]:
    meta_file = _meta_path(job_folder)
    if not os.path.isfile(meta_file):
        raise HTTPException(
            status_code=400,
            detail=f"meta.json not found in job_folder: {meta_file}",
        )
    return _read_meta_file(meta_file)


def _write_meta(job_folder: str, meta: Dict[str, Any]) -> None:
    _atomic_write_json(_meta_path(job_folder), meta)


def _merge_meta_preserve_planner(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge meta safely:
      - Start with existing disk meta (planner-written)
      - Overlay incoming payload meta (payload wins)
      - Preserve planner fields GPU does not know about
    """
    if not isinstance(existing, dict):
        existing = {}
    if not isinstance(incoming, dict):
        incoming = {}

    merged = dict(existing)
    merged.update(incoming)
    return merged


def _touch_status(
    meta: Dict[str, Any],
    status: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta["status"] = status
    meta["updated_at"] = datetime.utcnow().isoformat()
    if extra:
        meta.update(extra)
    return meta


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root() -> Dict[str, Any]:
    loaded = sd35_runtime is not None and sd35_runtime.mode == "real" and sd35_runtime.pipe is not None
    return {
        "message": "GPU Runtime for RENDEREXPO AI STUDIO.",
        "requested_mode": REQUESTED_RUNTIME_MODE,
        "run_real_sd35": ENABLE_REAL_SD35,
        "preload_on_startup": PRELOAD_SD35_ON_STARTUP,
        "real_runtime_loaded": loaded,
        "device": GPU_DEVICE,
    }


@app.get("/api/gpu/health")
async def gpu_health() -> Dict[str, Any]:
    loaded = sd35_runtime is not None and sd35_runtime.mode == "real" and sd35_runtime.pipe is not None
    return {
        "status": "ok",
        "service": "gpu_worker",
        "requested_mode": REQUESTED_RUNTIME_MODE,
        "run_real_sd35": ENABLE_REAL_SD35,
        "preload_on_startup": PRELOAD_SD35_ON_STARTUP,
        "real_runtime_loaded": loaded,
        "device": GPU_DEVICE,
        "timestamp_utc": datetime.utcnow().isoformat(),
    }


@app.post("/api/gpu/dispatch")
async def gpu_dispatch(payload: GPUDispatchPayload) -> Dict[str, Any]:
    """
    Called by the planner (8012):

    - Ensures job_folder exists
    - Reads meta.json from disk, merges payload.meta (payload wins)
    - Lazy-loads SD3.5 runtime if enabled and needed
    - Runs the job based on meta["type"]
    - Writes updated meta.json and returns it
    """
    job_folder = payload.job_folder
    _ensure_job_folder(job_folder)

    meta_file = _meta_path(job_folder)

    # 1) Load existing planner-written meta if present
    existing_meta = _read_meta_file(meta_file)

    # 2) Merge payload meta on top
    meta = _merge_meta_preserve_planner(existing_meta, payload.meta or {})

    # 3) Stamp runtime fields
    meta["job_folder"] = job_folder
    meta["dispatched_at"] = datetime.utcnow().isoformat()

    job_type = (meta.get("type") or "").strip().lower()

    try:
        runtime = await _ensure_runtime_loaded()
    except Exception as exc:  # noqa: BLE001
        logger.exception("GPU runtime unavailable: %s", exc)
        meta = _touch_status(
            meta,
            "failed",
            {
                "error": "runtime_not_loaded",
                "error_detail": str(exc),
                "failed_at": datetime.utcnow().isoformat(),
            },
        )
        _atomic_write_json(meta_file, meta)
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        meta = _touch_status(meta, "running")
        _atomic_write_json(meta_file, meta)

        if job_type == "text2img":
            updated_meta = runtime.generate_text2img(job_folder, meta)
        elif job_type == "img2img":
            if not hasattr(runtime, "generate_img2img"):
                raise RuntimeError("SD35Runtime does not implement generate_img2img().")
            updated_meta = runtime.generate_img2img(job_folder, meta)
        else:
            raise RuntimeError(f"Unsupported job type for GPU runtime: '{job_type}'")

        final_existing = _read_meta_file(meta_file)
        final_meta = _merge_meta_preserve_planner(
            final_existing,
            updated_meta if isinstance(updated_meta, dict) else {},
        )
        _atomic_write_json(meta_file, final_meta)

        return {
            "status": "ok",
            "message": "GPU dispatch completed in REAL SD3.5 mode.",
            "job_folder": job_folder,
            "meta": final_meta,
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
        _atomic_write_json(meta_file, meta)
        raise HTTPException(status_code=500, detail=str(exc))