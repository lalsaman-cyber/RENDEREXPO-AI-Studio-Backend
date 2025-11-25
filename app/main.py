# app/main.py
"""
LOCAL-ONLY FastAPI app for RENDEREXPO AI STUDIO.

IMPORTANT:
- This file is meant to run on your LAPTOP (CPU) only.
- It does NOT load SD3.5.
- It does NOT run any heavy AI.
- It is just for:
    * health checks
    * verifying SD3.5 files exist on disk
    * creating skeleton SD3.5 jobs (no inference yet)
"""

import os
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException

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

app = FastAPI(
    title="RENDEREXPO AI STUDIO - Local Dev API",
    description=(
        "Local-only API for health checks, SD3.5 file verification, "
        "and skeleton job creation. No heavy model loading or inference "
        "happens here."
    ),
    version="0.1.0",
)


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
        raise KeyError(
            "Could not find 'sd35_large_dir' in config/model_paths.yaml."
        )

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
    """Simple welcome endpoint."""
    return {
        "message": "RENDEREXPO AI STUDIO - Local Dev API (no SD3.5 inference here)."
    }


@app.get("/api/health")
async def health():
    """Basic health check for local development."""
    return {
        "status": "ok",
        "mode": "local-dev",
        "details": "FastAPI is running. No GPU, no SD3.5 loading on this app.",
    }


@app.get("/api/sd35/files")
async def sd35_files():
    """
    List the contents of the SD3.5 model directory, based on config/model_paths.yaml.

    This helps verify:
    - The config file exists
    - The model path is set
    - The SD3.5 files are actually present
    """
    try:
        sd35_dir = _read_sd35_model_dir_from_config()
    except (FileNotFoundError, KeyError) as e:
        # Config issues: return 500 so you know something is wrong in setup
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
