# app/main.py
"""
LOCAL-ONLY FastAPI app for RENDEREXPO AI STUDIO.

IMPORTANT:
- This API is INTERNAL (for Wix backend proxy + your own testing).
- It should NOT be exposed directly to end users.
- It does NOT load SD3.5 directly (planner/orchestrator role).
- It MUST enforce HMAC auth on ALL endpoints except /api/health.

LOCKED SERVICE MAP:
- Planner API  -> 8012
- GPU worker   -> 8002

HMAC SECURITY (LOCKED):
- Sign RAW request body bytes
- Headers:
    X-RENDEREXPO-SIGNATURE: hex digest (HMAC-SHA256)
    X-RENDEREXPO-TIMESTAMP: unix epoch seconds (int)
    X-RENDEREXPO-NONCE: random string (unique per request)
- Message to sign (bytes):
    f"{timestamp}\\n{nonce}\\n".encode("utf-8") + raw_body_bytes
- Timestamp window: ±30 seconds
- Nonce replay protection: in-memory cache (short TTL)
- Errors: structured minimal codes (invalid_signature, timestamp_expired, nonce_reused, missing_auth)

SECRET:
- Environment variable required:
    RENDEREXPO_HMAC_SECRET

IMPORTANT:
- This is the PLANNER service.
- It must NOT expose the GPU worker's /api/gpu/dispatch route.
- That route belongs to the separate GPU worker service.

SKETCH PIPELINE NOTE:
- Sketch routing is planner-side only.
- Real sketch execution happens on the GPU worker through dedicated dispatch.
- Sketch to Render uses the working MistoLine path:
    job_type     = "sdxl_mistoline_sketch"
    pipeline_key = "sdxl::mistoline_sketch"
- Sketch to Redesign uses the NEW parallel MistoLine path:
    job_type     = "sdxl_mistoline_sketch_redesign"
    pipeline_key = "sdxl::mistoline_sketch_redesign"
- DO NOT route sketch through the removed SD3.5 redesign path.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.routers import (
    cad,
    controlnet,
    depth,
    floorplan,
    img2img,
    insert_object,
    jobs,
    mesh_from_image,
    moodboard,
    pipeline,
    plan,
    product,
    product_insert,
    sketch,
    sketch_redesign,
    text2img,
    upscale,
    video_between_frames,
    video_from_image,
    vr,
)

# ---------------------------------------------------------------------------
# HMAC Auth Config (LOCKED)
# ---------------------------------------------------------------------------

HMAC_SECRET_ENV = "RENDEREXPO_HMAC_SECRET"

SIG_HEADER = "X-RENDEREXPO-SIGNATURE"
TS_HEADER = "X-RENDEREXPO-TIMESTAMP"
NONCE_HEADER = "X-RENDEREXPO-NONCE"

TS_WINDOW_SECONDS = 30
NONCE_TTL_SECONDS = 90

OPEN_PATHS = {"/api/health"}
OPEN_PREFIXES = ("/outputs/",)

# In-memory nonce replay cache (nonce -> expires_at_epoch)
_NONCE_CACHE: Dict[str, int] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_epoch() -> int:
    return int(time.time())


def _purge_expired_nonces(nonce_cache: Dict[str, int]) -> None:
    """
    Purge expired nonces from the in-memory cache.
    nonce_cache maps nonce -> expires_at_epoch.
    """
    now = _now_epoch()
    expired = [n for n, exp in nonce_cache.items() if exp <= now]
    for n in expired:
        nonce_cache.pop(n, None)


def _compute_signature(secret: str, timestamp: str, nonce: str, body: bytes) -> str:
    """
    Compute hex HMAC-SHA256 over:
        f"{timestamp}\\n{nonce}\\n".encode("utf-8") + body
    """
    prefix = f"{timestamp}\n{nonce}\n".encode("utf-8")
    msg = prefix + (body or b"")
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def _constant_time_equal(a: str, b: str) -> bool:
    return hmac.compare_digest(a or "", b or "")


def _ensure_outputs_dir() -> None:
    """
    Ensure outputs/ exists before StaticFiles mounts it.
    """
    os.makedirs("outputs", exist_ok=True)


def _read_sd35_model_dir_from_config() -> str:
    """
    Very simple parser for config/model_paths.yaml to find sd35_large_dir.
    We avoid adding extra dependencies (like PyYAML).

    Expected line:
        sd35_large_dir: "models/sd35-large"
    """
    config_path = os.path.join("config", "model_paths.yaml")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    sd35_dir: Optional[str] = None

    with open(config_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("sd35_large_dir"):
                parts = line.split(":", 1)
                if len(parts) != 2:
                    continue

                value = parts[1].strip()

                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                if value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                sd35_dir = value
                break

    if not sd35_dir:
        raise KeyError("Could not find 'sd35_large_dir' in config/model_paths.yaml.")

    return sd35_dir


def _list_directory_contents(path: str, max_items: int = 200) -> Dict[str, Any]:
    """
    Return a simple listing of the given directory (one level deep).
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory not found: {path}")

    items: List[Dict[str, Any]] = []

    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        item_type = "dir" if os.path.isdir(full_path) else "file"
        items.append({"name": name, "type": item_type})

        if len(items) >= max_items:
            break

    return {
        "path": path,
        "items": items,
        "count": len(items),
        "truncated": len(items) >= max_items,
    }


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

_ensure_outputs_dir()

app = FastAPI(
    title="RENDEREXPO AI STUDIO - Internal API (Planner)",
    description=(
        "Internal API for RENDEREXPO AI STUDIO. "
        "This is NOT client-facing. Wix UI will be client-facing. "
        "All endpoints (except /api/health) require HMAC authentication."
    ),
    version="0.4.0",
)

# Serve outputs so /outputs/... URLs work
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


@app.on_event("startup")
async def _startup_check_secret() -> None:
    """
    Fail fast if the HMAC secret is missing.
    """
    secret = os.getenv(HMAC_SECRET_ENV, "").strip()
    if len(secret) < 32:
        raise RuntimeError(
            f"Missing or too-short {HMAC_SECRET_ENV}. "
            "Set a strong secret (64+ chars recommended) before running the API."
        )


@app.middleware("http")
async def hmac_auth_middleware(request: Request, call_next):
    """
    Enforce HMAC auth on all paths except OPEN_PATHS and OPEN_PREFIXES.
    """
    path = request.url.path

    if path in OPEN_PATHS or any(path.startswith(prefix) for prefix in OPEN_PREFIXES):
        return await call_next(request)

    provided_sig = request.headers.get(SIG_HEADER, "")
    ts = request.headers.get(TS_HEADER, "")
    nonce = request.headers.get(NONCE_HEADER, "")

    if not provided_sig or not ts or not nonce:
        return JSONResponse(
            status_code=401,
            content={
                "error": "missing_auth",
                "detail": f"Required headers: {SIG_HEADER}, {TS_HEADER}, {NONCE_HEADER}",
            },
        )

    try:
        ts_int = int(ts)
    except Exception:
        return JSONResponse(
            status_code=401,
            content={
                "error": "timestamp_invalid",
                "detail": "Timestamp must be unix epoch seconds (int).",
            },
        )

    now = _now_epoch()
    if abs(now - ts_int) > TS_WINDOW_SECONDS:
        return JSONResponse(
            status_code=401,
            content={
                "error": "timestamp_expired",
                "detail": f"Timestamp outside ±{TS_WINDOW_SECONDS}s window.",
            },
        )

    _purge_expired_nonces(_NONCE_CACHE)
    if nonce in _NONCE_CACHE:
        return JSONResponse(
            status_code=401,
            content={
                "error": "nonce_reused",
                "detail": "Nonce has already been used within the allowed window.",
            },
        )

    body = await request.body()

    # Preserve body for downstream handlers
    request._body = body  # type: ignore[attr-defined]

    secret = os.getenv(HMAC_SECRET_ENV, "").strip()
    expected_sig = _compute_signature(secret=secret, timestamp=ts, nonce=nonce, body=body)

    if not _constant_time_equal(provided_sig.lower(), expected_sig.lower()):
        return JSONResponse(
            status_code=401,
            content={
                "error": "invalid_signature",
                "detail": "Signature verification failed.",
            },
        )

    _NONCE_CACHE[nonce] = now + NONCE_TTL_SECONDS

    return await call_next(request)


# ---------------------------------------------------------------------------
# Attach routers
# ---------------------------------------------------------------------------

# Planner / orchestration routes
app.include_router(plan.router)
app.include_router(text2img.router)
app.include_router(img2img.router)
app.include_router(jobs.router)
app.include_router(depth.router)
app.include_router(controlnet.router)
app.include_router(upscale.router)
app.include_router(vr.router)
app.include_router(moodboard.router)
app.include_router(product.router)
app.include_router(product_insert.router)
app.include_router(insert_object.router)
app.include_router(floorplan.router)
app.include_router(sketch.router)
app.include_router(sketch_redesign.router)
app.include_router(pipeline.router)

# Real generation routes that dispatch to GPU worker
app.include_router(cad.router)
app.include_router(mesh_from_image.router)
app.include_router(video_between_frames.router)
app.include_router(video_from_image.router)

# IMPORTANT:
# Do NOT include the GPU worker router here.
# The GPU worker service hosts /api/gpu/dispatch separately.


# ---------------------------------------------------------------------------
# Basic routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """
    Simple welcome endpoint.
    NOTE: This endpoint is protected by HMAC.
    """
    return {
        "message": (
            "RENDEREXPO AI STUDIO - PLANNER API (HMAC protected). "
            "Use /api/health for unauthenticated health."
        )
    }


@app.get("/api/health")
async def health():
    """Basic health check (the only open endpoint)."""
    return {
        "status": "ok",
        "mode": "planner",
        "details": "FastAPI is running. HMAC auth enabled on all endpoints except /api/health.",
        "timestamp_epoch": _now_epoch(),
        "ts_window_seconds": TS_WINDOW_SECONDS,
    }


@app.get("/api/sd35/files")
async def sd35_files():
    """
    List the contents of the SD3.5 model directory, based on config/model_paths.yaml.
    NOTE: This endpoint is protected by HMAC.
    """
    try:
        sd35_dir = _read_sd35_model_dir_from_config()
    except (FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        listing = _list_directory_contents(sd35_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "ok",
        "sd35_large_dir": sd35_dir,
        "contents": listing,
    }