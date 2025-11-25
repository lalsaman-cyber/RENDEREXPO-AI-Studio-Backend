# app/routers/pipeline.py

"""
Unified pipeline planner for RENDEREXPO AI STUDIO (skeleton).

Goal:
- Attach a high-level "pipeline plan" to an existing job folder.
- This pipeline can include stages like:
    * text2img
    * img2img
    * depth
    * controlnet
    * upscale
    * vr
    * moodboard
    * floorplan
    * product_insert

IMPORTANT:
- This is still CPU-only and skeleton-only.
- No SD3.5 inference.
- We only read/write meta.json and describe what *would* happen on GPU later.
"""

import os
import json
import datetime
from typing import List, Dict, Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(
    prefix="/api/pipeline",
    tags=["Pipeline Planning"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

StageType = Literal[
    "text2img",
    "img2img",
    "depth",
    "controlnet",
    "upscale",
    "vr",
    "moodboard",
    "floorplan",
    "product_insert",
]


class PipelineStage(BaseModel):
    """
    One step in a planned pipeline.

    Example:
    {
        "stage_type": "text2img",
        "params": {
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 25
        }
    }
    """
    stage_type: StageType = Field(..., description="Type of stage in the pipeline.")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form parameter dict for this stage.",
    )


class PipelinePlanRequest(BaseModel):
    """
    Request body for planning a pipeline on an existing job folder.
    """
    job_folder: str = Field(
        ...,
        description="Path to an existing job folder under outputs/{date}/{job_id}/",
    )
    stages: List[PipelineStage] = Field(
        ...,
        description="List of pipeline stages to attach to this job.",
    )


class PipelinePlanResponse(BaseModel):
    status: str
    message: str
    job_folder: str
    pipeline: Dict[str, Any]
    meta_path: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_job_folder(job_folder: str) -> None:
    if not os.path.isdir(job_folder):
        raise HTTPException(
            status_code=400,
            detail=f"job_folder does not exist: {job_folder}",
        )


def _meta_path(job_folder: str) -> str:
    return os.path.join(job_folder, "meta.json")


def _read_meta(job_folder: str) -> Dict[str, Any]:
    meta_file = _meta_path(job_folder)
    if not os.path.isfile(meta_file):
        # If there is no meta yet, start fresh
        return {}
    with open(meta_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_meta(job_folder: str, meta: Dict[str, Any]) -> None:
    meta_file = _meta_path(job_folder)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)


# ---------------------------------------------------------------------------
# Route: plan a pipeline
# ---------------------------------------------------------------------------

@router.post("/plan", response_model=PipelinePlanResponse)
async def plan_pipeline(request: PipelinePlanRequest):
    """
    Attach a high-level "pipeline plan" to `meta.json` in the given job folder.

    This does NOT run any AI. It only:
    - validates the job folder
    - reads existing meta.json (if any)
    - stores a list of stages under meta["pipeline_plan"]
    - marks `meta["pipeline_mode"] = "skeleton-no-inference"`

    Later, the GPU runtime will:
    - read meta["pipeline_plan"]
    - execute each stage in order using SD3.5 + ControlNet + ESRGAN + etc.
    """
    job_folder = request.job_folder
    _ensure_job_folder(job_folder)

    # Read or start meta
    meta = _read_meta(job_folder)

    if "job_id" not in meta:
        # Try to infer job_id from folder name
        meta["job_id"] = os.path.basename(job_folder)

    # Prepare the pipeline structure
    pipeline_plan = {
        "created_at": datetime.datetime.utcnow().isoformat(),
        "stages": [stage.model_dump() for stage in request.stages],
    }

    # Attach to meta
    meta["pipeline_plan"] = pipeline_plan
    meta["pipeline_mode"] = "skeleton-no-inference"

    # For convenience, track last_updated
    meta["last_updated"] = datetime.datetime.utcnow().isoformat()

    _write_meta(job_folder, meta)

    return PipelinePlanResponse(
        status="ok",
        message="Pipeline planned (skeleton, no AI yet).",
        job_folder=job_folder,
        pipeline=pipeline_plan,
        meta_path=_meta_path(job_folder),
    )
