# app/routers/plan.py

"""
RENDEREXPO AI STUDIO - Planning Router (Planner only)

Purpose:
- No GPU
- No job folder creation
- No meta.json writing
- No worker dispatch

This router only:
- validates inputs
- builds a meta-like dry-run plan using apply_preset_to_meta(...)
- returns exactly what would be written by real endpoints

IMPORTANT:
- Planner = port 8012
- GPU worker = port 8002
- Locked preset logic lives in app.presets_sd35
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.presets_sd35 import apply_preset_to_meta

router = APIRouter(
    prefix="/api/plan",
    tags=["Planning"],
)

Category = Literal["urban", "suburban", "interior", "wide_hero"]
Shot = Literal["wide", "close"]


# ============================
# Request / Response Models
# ============================

class SD35PlanRequest(BaseModel):
    prompt: str = Field(..., description="User prompt for SD3.5 rendering")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt text")
    category: Category = Field(..., description="urban/suburban/interior/wide_hero")
    shot: Shot = Field(..., description="wide/close")
    upscale_2x: Optional[bool] = Field(
        None,
        description="Optional override: true/false. If omitted, preset default is used.",
    )
    seed: Optional[int] = Field(None, description="If None, backend will generate one")

    # accepted for compatibility, ignored by preset logic
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


class FloorplanPlanRequest(BaseModel):
    """
    Plan a top-down, clean, legit 2D floorplan render from text.

    Best defaults are chosen automatically:
      category=interior, shot=wide, render_mode=precise
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
    upscale_2x: Optional[bool] = Field(
        None,
        description="Optional override: true/false. If omitted, preset default is used.",
    )
    seed: Optional[int] = Field(None, description="If None, backend will generate one")
    wall_thickness_m: float = Field(
        0.20,
        ge=0.05,
        le=1.00,
        description="Target wall thickness in meters (planning hint).",
    )
    include_dimensions: bool = Field(
        True,
        description="If true, include dimension lines and room labels (planning hint).",
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
# Route: /api/plan/sd35
# ============================

@router.post("/sd35", response_model=SD35PlanResponse)
def plan_sd35_render(request: SD35PlanRequest) -> SD35PlanResponse:
    job_id = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    seed = request.seed if request.seed is not None else int(uuid.uuid4().int % 1_000_000_000)

    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": now,
        "type": "text2img",
        "model_name": "sd35_large_pro_v2_1",
        "engine": "sd35_large_pro_v2_1",
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "seed": seed,
        "category": request.category,
        "shot": request.shot,
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
        raise HTTPException(status_code=400, detail=f"Preset error: {exc}") from exc

    return SD35PlanResponse(
        job_id=job_id,
        created_at=now,
        model="sd35_large_pro_v2_1",
        plan_type="sd35_text2img_plan",
        settings=meta,
        message="Dry-run plan generated using locked presets.",
    )


# ============================
# Route: /api/plan/floorplan
# ============================

@router.post("/floorplan", response_model=FloorplanPlanResponse)
def plan_floorplan_render(request: FloorplanPlanRequest) -> FloorplanPlanResponse:
    """
    Planning-only: Floorplan text -> top-down 2D rendered plan.

    Best defaults:
      - category = interior
      - shot = wide
      - render_mode = precise
    """
    job_id = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    seed = request.seed if request.seed is not None else int(uuid.uuid4().int % 1_000_000_000)

    category: Category = "interior"
    shot: Shot = "wide"

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
        "type": "text2img",
        "model_name": "sd35_large_pro_v2_1",
        "engine": "sd35_large_pro_v2_1",
        "task": "floorplan_text_to_2d_plan",
        "view": "top_down_orthographic",
        "render_mode": "precise",
        "prompt": floorplan_prompt,
        "negative_prompt": request.negative_prompt,
        "seed": seed,
        "category": category,
        "shot": shot,
        "status": "planned",
        "mode": "plan-only",
        "planned_output_image": "floorplan.png",
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
        raise HTTPException(status_code=400, detail=f"Preset error: {exc}") from exc

    return FloorplanPlanResponse(
        job_id=job_id,
        created_at=now,
        model="sd35_large_pro_v2_1",
        plan_type="sd35_floorplan_text_to_2d_plan",
        settings=meta,
        message="Dry-run floorplan plan generated using locked presets.",
    )