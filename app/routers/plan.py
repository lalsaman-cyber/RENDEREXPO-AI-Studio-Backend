# Backend/app/routers/plan.py

from fastapi import APIRouter
from pydantic import BaseModel, Field
import uuid
from datetime import datetime

router = APIRouter(
    prefix="/api/plan",
    tags=["SD3.5 Planning (NO GPU)"]
)

# ============================
# Request Schema
# ============================

class SD35PlanRequest(BaseModel):
    prompt: str = Field(..., description="User prompt for SD3.5 rendering")
    negative_prompt: str | None = Field(None, description="Negative prompt text")
    width: int = Field(1024, description="Requested width")
    height: int = Field(1024, description="Requested height")
    steps: int = Field(25, description="Planned inference steps")
    guidance_scale: float = Field(5.5, description="Planned CFG scale")
    seed: int | None = Field(None, description="If None, backend will generate one")


# ============================
# Response Schema
# ============================

class SD35PlanResponse(BaseModel):
    job_id: str
    created_at: str
    model: str
    would_use_pipeline: str
    would_call_runtime: str
    settings: dict
    message: str


# ============================
# ROUTE: /api/plan/sd35
# ============================

@router.post("/sd35", response_model=SD35PlanResponse)
def plan_sd35_render(request: SD35PlanRequest):
    """
    This endpoint does NOT run SD3.5.
    It only reports what WOULD happen once GPU inference is enabled.
    """

    job_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    # Use a fake seed if none provided
    seed = request.seed if request.seed is not None else int(uuid.uuid4().int % 10_000_000)

    return SD35PlanResponse(
        job_id=job_id,
        created_at=now,
        model="sd3.5-large",
        would_use_pipeline="modules.sd35_text2img (future)",
        would_call_runtime="runtime.sd35_runtime.SD35Runtime.generate_text2img() (future)",
        settings={
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "guidance_scale": request.guidance_scale,
            "seed": seed
        },
        message="This is a DRY RUN. No GPU, no SD3.5, no inference."
    )
