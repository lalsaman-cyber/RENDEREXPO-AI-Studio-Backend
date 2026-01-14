# app/main.py
"""
LOCAL-ONLY FastAPI app for RENDEREXPO AI STUDIO.

IMPORTANT (UPDATED):
- This API is INTERNAL (for Wix backend proxy + your own testing).
- It should NOT be exposed directly to end users.
- It does NOT load SD3.5 directly (planner/orchestrator role).
- It MUST enforce HMAC auth on ALL endpoints except /api/health.

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
"""

import os
import time
import hmac
import hashlib
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.routers import (
    plan,
    text2img,
    img2img,
    jobs,
    depth,
    controlnet,
    upscale,
    vr,
    moodboard,
    product,
    floorplan,
    sketch,  # <-- NEW
)

# ---------------------------------------------------------------------------
# HMAC Auth Config (LOCKED)
# ---------------------------------------------------------------------------

HMAC_SECRET_ENV = "RENDEREXPO_HMAC_SECRET"

SIG_HEADER = "X-RENDEREXPO-SIGNATURE"
TS_HEADER = "X-RENDEREXPO-TIMESTAMP"
NONCE_HEADER = "X-RENDEREXPO-NONCE"

# Timestamp acceptance window (±30s locked)
TS_WINDOW_SECONDS = 30

# Nonce cache TTL (a little longer than TS window)
NONCE_TTL_SECONDS = 90

# Only health is open (everything else requires HMAC)
OPEN_PATHS = {"/api/health"}


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
    digest = hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()
    return digest


def _constant_time_equal(a: str, b: str) -> bool:
    # hmac.compare_digest is constant-time comparison
    return hmac.compare_digest(a or "", b or "")


app = FastAPI(
    title="RENDEREXPO AI STUDIO - Internal API (Planner)",
    description=(
        "Internal API for RENDEREXPO AI STUDIO. "
        "This is NOT client-facing. Wix UI will be client-facing. "
        "All endpoints (except /api/health) require HMAC authentication."
    ),
    version="0.2.0",
)

# In-memory nonce replay cache (nonce -> expires_at_epoch)
_NONCE_CACHE: Dict[str, int] = {}


@app.on_event("startup")
async def _startup_check_secret() -> None:
    """
    Fail fast if the HMAC secret is missing.
    """
    secret = os.getenv(HMAC_SECRET_ENV, "")
    if not secret or len(secret.strip()) < 32:
        # Require at least 32 chars; you will use 64+ in production.
        raise RuntimeError(
            f"Missing or too-short {HMAC_SECRET_ENV}. "
            f"Set a strong secret (64+ chars) before running the API."
        )


@app.middleware("http")
async def hmac_auth_middleware(request: Request, call_next):
    """
    Enforce HMAC auth on all paths except OPEN_PATHS.
    """
    path = request.url.path

    # Allow health endpoint unauthenticated
    if path in OPEN_PATHS:
        return await call_next(request)

    # Read headers
    provided_sig = request.headers.get(SIG_HEADER, "")
    ts = request.headers.get(TS_HEADER, "")
    nonce = request.headers.get(NONCE_HEADER, "")

    # Missing auth headers
    if not provided_sig or not ts or not nonce:
        return JSONResponse(
            status_code=401,
            content={
                "error": "missing_auth",
                "detail": f"Required headers: {SIG_HEADER}, {TS_HEADER}, {NONCE_HEADER}",
            },
        )

    # Validate timestamp format
    try:
        ts_int = int(ts)
    except Exception:
        return JSONResponse(
            status_code=401,
            content={"error": "timestamp_invalid", "detail": "Timestamp must be unix epoch seconds (int)."},
        )

    # Validate timestamp window ±30s
    now = _now_epoch()
    if abs(now - ts_int) > TS_WINDOW_SECONDS:
        return JSONResponse(
            status_code=401,
            content={
                "error": "timestamp_expired",
                "detail": f"Timestamp outside ±{TS_WINDOW_SECONDS}s window.",
            },
        )

    # Purge old nonces and enforce replay protection
    _purge_expired_nonces(_NONCE_CACHE)
    if nonce in _NONCE_CACHE:
        return JSONResponse(
            status_code=401,
            content={"error": "nonce_reused", "detail": "Nonce has already been used within the allowed window."},
        )

    # Read raw body bytes exactly as sent
    body = await request.body()

    # Compute expected signature
    secret = os.getenv(HMAC_SECRET_ENV, "")
    expected_sig = _compute_signature(secret=secret, timestamp=ts, nonce=nonce, body=body)

    if not _constant_time_equal(provided_sig.lower(), expected_sig.lower()):
        return JSONResponse(
            status_code=401,
            content={"error": "invalid_signature", "detail": "Signature verification failed."},
        )

    # Mark nonce as used (store with TTL)
    _NONCE_CACHE[nonce] = now + NONCE_TTL_SECONDS

    return await call_next(request)


# Attach routers
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
app.include_router(floorplan.router)
app.include_router(sketch.router)  # <-- NEW


# ---------------------------------------------------------------------------
# Helpers: read config/model_paths.yaml WITHOUT extra libraries
# ---------------------------------------------------------------------------

def _read_sd35_model_dir_from_config() -> str:
    """
    Very simple parser for config/model_paths.yaml to find sd35_large_dir.

    We avoid adding extra dependencies (like PyYAML) for this small task.

    Expected line in config/model_paths.yaml:

        sd35_large_dir: "models/sd35-large"

    We:
    - look for a line starting with 'sd35_large_dir'
    - split on ':'
    - strip quotes and spaces
    """
    config_path = os.path.join("config", "model_paths.yaml")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    sd35_dir: Optional[str] = None

    with open(config_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            if line.startswith("sd35_large_dir"):
                # Example: sd35_large_dir: "models/sd35-large"
                parts = line.split(":", 1)
                if len(parts) != 2:
                    continue
                value = parts[1].strip()

                # Remove optional quotes
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
    Return a simple listing of the given directory.

    - If the directory does not exist, raise an error.
    - Only goes ONE level deep (top-level files & folders).
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory not found: {path}")

    items: List[Dict[str, Any]] = []

    # List only the immediate contents (no deep recursion)
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
# Basic routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """
    Simple welcome endpoint.

    NOTE: This endpoint is protected by HMAC (per locked policy).
    """
    return {"message": "RENDEREXPO AI STUDIO - Internal API (HMAC protected). Use /api/health for unauthenticated health."}


@app.get("/api/health")
async def health():
    """Basic health check (the only open endpoint)."""
    return {
        "status": "ok",
        "mode": "internal",
        "details": "FastAPI is running. HMAC auth enabled on all endpoints except /api/health.",
        "timestamp_epoch": _now_epoch(),
        "ts_window_seconds": TS_WINDOW_SECONDS,
    }


@app.get("/api/sd35/files")
async def sd35_files():
    """
    List the contents of the SD3.5 model directory, based on config/model_paths.yaml.

    This helps verify:
    - The config file exists
    - The model path is set
    - The SD3.5 files are actually present

    NOTE: This endpoint is protected by HMAC.
    """
    try:
        sd35_dir = _read_sd35_model_dir_from_config()
    except (FileNotFoundError, KeyError) as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        listing = _list_directory_contents(sd35_dir)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "ok",
        "sd35_large_dir": sd35_dir,
        "contents": listing,
    }
