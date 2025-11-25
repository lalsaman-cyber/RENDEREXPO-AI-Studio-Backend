# app/routers/__init__.py

"""
Central export for all FastAPI routers used in RENDEREXPO AI STUDIO.

Each router file defines:
- an APIRouter instance
- one or more endpoints (skeleton or real)

We import them here so app.main can do:

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
        sketch,
    )

and then:

    app.include_router(text2img.router)
    ...
"""

from . import (
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

__all__ = [
    "plan",
    "text2img",
    "img2img",
    "jobs",
    "depth",
    "controlnet",
    "upscale",
    "vr",
    "moodboard",
    "product",
    "floorplan",
    "sketch",  # <-- NEW
]
