# app/routers/plan.py

"""
RENDEREXPO AI STUDIO - Planning Router (NO GPU)

CRITICAL (Doc 18):
- Planning endpoints MUST report the SAME locked preset parameters that real jobs use:
  steps, CFG, LyCORIS(PRO 2.1) multiplier, GEO multiplier, resolution,
  NO denoise anywhere, upscale optional per-request.

This router does NOT:
- create job folders
- write meta.json
- call GPU worker

It ONLY:
- validates inputs
- builds a "meta-like plan" using apply_preset_to_meta()
- returns exactly what WOULD be written into meta.json by real endpoints
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, Literal, Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.presets_sd35 import apply_preset_to_meta

router = APIRouter(
    prefix="/api/plan",
    tags=["SD3.5 Planning (NO GPU)"],
)

Category = Literal["urban", "suburban", "interior", "wide_hero"]
Shot = Literal["wide", "close"]


# ============================
# Existing Request Schema
# ============================

class SD35PlanRequest(BaseModel):
    prompt: str = Field(..., description="User prompt for SD3.5 rendering")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt text")

    # LOCKED PRESET SELECTORS (Doc 18)
    category: Category = Field(..., description="urban/suburban/interior/wide_hero")
    shot: Shot = Field(..., description="wide/close")

    # OPTIONAL per-request upscale override
    upscale_2x: Optional[bool] = Field(
        None,
        description="Optional override: true/false. If omitted, preset default is used.",
    )

    # Optional seed (if None, backend generates one)
    seed: Optional[int] = Field(None, description="If None, backend will generate one")

    # Compatibility knobs (accepted but ignored; presets are locked)
    width: Optional[int] = Field(None, description="Ignored (preset-locked).")
    height: Optional[int] = Field(None, description="Ignored (preset-locked).")
    steps: Optional[int] = Field(None, description="Ignored (preset-locked).")
    guidance_scale: Optional[float] = Field(None, description="Ignored (preset-locked).")


class SD35PlanResponse(BaseModel):
    job_id: str
    created_at: str
    model: str
    plan_type: str
    settings: Dict[str, Any]
    message: str


# ============================
# NEW: Floorplan Planning
# ============================

class FloorplanPlanRequest(BaseModel):
    """
    Plan a top-down, clean, legit 2D floorplan render from text.

    You do NOT choose category/shot; we pick the best defaults:
      category=interior, shot=wide, render_mode=precise

    This endpoint is STILL planning-only (NO GPU).
    """
    description: str = Field(
        ...,
        description=(
            "What to build, e.g. "
            "'Single-family house, 3 bedrooms, open kitchen, 2 bathrooms, laundry, "
            "1-car garage, rear patio. Modern style. Furnished.'"
        ),
    )
    negative_prompt: Optional[str] = Field(None, description="Optional negative prompt")

    # Optional: keep override hook for upscale; default None uses preset default
    upscale_2x: Optional[bool] = Field(
        None,
        description="Optional override: true/false. If omitted, preset default is used.",
    )

    # Optional seed
    seed: Optional[int] = Field(None, description="If None, backend will generate one")

    # Optional constraints (stored as planning hints)
    wall_thickness_m: float = Field(
        0.20,
        ge=0.05,
        le=1.00,
        description="Target wall thickness in meters (planning hint, not a locked knob).",
    )
    include_dimensions: bool = Field(
        True,
        description="If true, plan to include dimension lines and room labels (planning hint).",
    )
    furnished: bool = Field(
        True,
        description="If true, include furniture appropriate to room functions (planning hint).",
    )
    style: Optional[str] = Field(
        "clean architectural plan",
        description="Stylistic direction (planning hint).",
    )


class FloorplanPlanResponse(BaseModel):
    job_id: str
    created_at: str
    model: str
    plan_type: str
    settings: Dict[str, Any]
    message: str


# ============================
# ROUTE: /api/plan/sd35
# ============================

@router.post("/sd35", response_model=SD35PlanResponse)
def plan_sd35_render(request: SD35PlanRequest):
    job_id = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    seed = request.seed if request.seed is not None else int(uuid.uuid4().int % 1_000_000_000)

    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": now,

        "type": "text2img",
        "model_name": "sd3.5-large",

        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "seed": seed,

        "category": request.category,
        "shot": request.shot,

        "denoise": 0.0,

        "status": "planned",
        "mode": "plan-only",
        "planned_output_image": "output.png",
    }

    try:
        apply_preset_to_meta(
            meta,
            category=request.category,
            shot=request.shot,
            upscale_2x=request.upscale_2x,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Preset error: {exc}")

    meta["denoise"] = 0.0
    if isinstance(meta.get("upscale"), dict):
        meta["upscale"]["denoise"] = 0.0

    return SD35PlanResponse(
        job_id=job_id,
        created_at=now,
        model="sd3.5-large",
        plan_type="sd35_text2img_plan",
        settings=meta,
        message="DRY RUN plan generated using locked Doc 18 presets (NO GPU, no job folder created).",
    )


# ============================
# ROUTE: /api/plan/floorplan
# ============================

@router.post("/floorplan", response_model=FloorplanPlanResponse)
def plan_floorplan_render(request: FloorplanPlanRequest):
    """
    Planning-only: Floorplan text -> top-down 2D rendered plan.

    We pick best defaults (you don't choose):
      - category = interior
      - shot = wide
      - render_mode = precise

    Returns locked Doc 18 preset settings + floorplan-specific intent hints.
    """
    job_id = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    seed = request.seed if request.seed is not None else int(uuid.uuid4().int % 1_000_000_000)

    # Best default choices for floorplans
    category: Category = "interior"
    shot: Shot = "wide"

    # Build a strong floorplan-specific prompt wrapper.
    # This is NOT changing locked knobs; it's just making the intent unambiguous.
    floorplan_prompt = (
        "TOP-DOWN ORTHOGRAPHIC 2D FLOOR PLAN, clean architectural drafting, "
        "consistent wall thickness, accurate room layout, readable room labels, "
        "doors and windows drawn with standard plan symbols, "
        "furnished appropriately for each room function, "
        "rendered flooring materials and furniture materials, "
        "crisp linework, high legibility, "
        f"wall thickness approximately {request.wall_thickness_m}m, "
        f"{'include dimension lines' if request.include_dimensions else 'no dimension lines'}, "
        f"{'furnished' if request.furnished else 'unfurnished'}, "
        f"style: {request.style or 'clean architectural plan'}, "
        "NO perspective, NO 3D camera, NO isometric view, NO exterior elevation, "
        "deliver a single clean plan sheet image. "
        "CLIENT BRIEF: "
        f"{request.description}"
    )

    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": now,

        # This is still a SD3.5 generation plan (text2img in execution)
        "type": "text2img",
        "model_name": "sd3.5-large-pro-2.1",

        "task": "floorplan_text_to_2d_plan",
        "view": "top_down_orthographic",
        "render_mode": "precise",

        "prompt": floorplan_prompt,
        "negative_prompt": request.negative_prompt,
        "seed": seed,

        # We choose these; user doesn't.
        "category": category,
        "shot": shot,

        # Hard lock
        "denoise": 0.0,

        "status": "planned",
        "mode": "plan-only",
        "planned_output_image": "floorplan.png",

        # Store human constraints (not locked knobs)
        "floorplan_constraints": {
            "wall_thickness_m": request.wall_thickness_m,
            "include_dimensions": request.include_dimensions,
            "furnished": request.furnished,
            "style": request.style,
        },
    }

    try:
        apply_preset_to_meta(meta, category=category, shot=shot, upscale_2x=request.upscale_2x)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Preset error: {exc}")

    meta["denoise"] = 0.0
    if isinstance(meta.get("upscale"), dict):
        meta["upscale"]["denoise"] = 0.0

    return FloorplanPlanResponse(
        job_id=job_id,
        created_at=now,
        model="sd3.5-large-pro-2.1",
        plan_type="sd35_floorplan_text_to_2d_plan",
        settings=meta,
        message="DRY RUN floorplan plan generated using locked Doc 18 presets (NO GPU).",
    )
