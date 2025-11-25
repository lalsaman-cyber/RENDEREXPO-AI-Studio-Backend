# runpod_server.py

"""
GPU RUNTIME (SKELETON)

This file is the ENTRYPOINT for RunPod GPU execution.
Right now it does NOT:
- load SD3.5
- load ControlNet
- run any inference
- use GPU

Later (Phase 3), this becomes the real GPU app.

Right now it only:
- loads SD35Runtime(device="cpu") safely
- loads PipelineManager
- exposes a simple API to test dispatching
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from runtime.sd35_runtime import SD35Runtime
from runtime.pipeline_manager import PipelineManager


# -------------------------------------------------------------------
# FastAPI APP (GPU runtime)
# -------------------------------------------------------------------

app = FastAPI(
    title="RENDEREXPO AI STUDIO - GPU Runtime (SKELETON)",
    description=(
        "This is the placeholder API for GPU execution. "
        "No GPU, no SD3.5 loading yet — safe for local use."
    ),
    version="0.1.0-gpu-skeleton"
)


# -------------------------------------------------------------------
# TEMPORARY: SD35Runtime using CPU only
# (We switch to device='cuda' later)
# -------------------------------------------------------------------

runtime = SD35Runtime(device="cpu")
pipeline_manager = PipelineManager(base_outputs_dir="outputs")


# -------------------------------------------------------------------
# REQUEST MODEL
# -------------------------------------------------------------------

class GPUJobRequest(BaseModel):
    date_str: str = Field(..., description="Folder date, e.g. 2025-11-23")
    job_id: str = Field(..., description="The job UUID folder name")


# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "message": "RENDEREXPO GPU Runtime Skeleton (no inference yet)."
    }


@app.post("/api/gpu/dispatch")
async def gpu_dispatch(req: GPUJobRequest):
    """
    Skeleton dispatcher.

    WHAT HAPPENS NOW:
    - Loads meta.json from outputs/{date}/{job_id}
    - Returns it with a small message
    - No SD3.5, no GPU, no inference

    WHAT WILL HAPPEN LATER:
    - runtime.load()
    - choose pipeline (text2img/img2img/etc)
    - run inference on GPU
    - write output.png for real
    """
    try:
        result = pipeline_manager.dispatch_job(
            date_str=req.date_str,
            job_id=req.job_id,
            sd35_runtime=runtime
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "status": "ok",
        "message": "GPU dispatcher skeleton ran successfully.",
        "job": result
    }
