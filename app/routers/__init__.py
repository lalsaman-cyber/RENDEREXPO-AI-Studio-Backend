# app/routers/__init__.py
"""
Central export for all FastAPI routers used in RENDEREXPO AI STUDIO (PLANNER).

Each router file defines:
- an APIRouter instance
- one or more endpoints

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
        video_between_frames,
        video_from_image,
        cad,
        mesh_from_image,
    )

IMPORTANT:
- This module is for PLANNER routers only.
- Do NOT add the GPU worker dispatch router here (/api/gpu/dispatch).
  That route belongs to the separate GPU worker service (port 8012).
"""

from __future__ import annotations

from . import (
    cad,
    controlnet,
    depth,
    floorplan,
    img2img,
    jobs,
    mesh_from_image,
    moodboard,
    plan,
    product,
    sketch,
    text2img,
    upscale,
    video_between_frames,
    video_from_image,
    vr,
)

__all__ = [
    "cad",
    "controlnet",
    "depth",
    "floorplan",
    "img2img",
    "jobs",
    "mesh_from_image",
    "moodboard",
    "plan",
    "product",
    "sketch",
    "text2img",
    "upscale",
    "video_between_frames",
    "video_from_image",
    "vr",
]
