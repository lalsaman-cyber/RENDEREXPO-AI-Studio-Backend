# app/routers/__init__.py
"""
Central export for all FastAPI routers used in RENDEREXPO AI STUDIO (PLANNER).

Each router file defines:
- an APIRouter instance
- one or more endpoints

We import them here so app.main can do package-level router imports cleanly.

IMPORTANT:
- This module is for PLANNER routers only.
- Do NOT add the GPU worker dispatch router here (/api/gpu/dispatch).
  That route belongs to the separate GPU worker service.
"""

from __future__ import annotations

from . import (
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
    sd35,
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
    "insert_object",
    "jobs",
    "mesh_from_image",
    "moodboard",
    "pipeline",
    "plan",
    "product",
    "product_insert",
    "sd35",
    "sketch",
    "text2img",
    "upscale",
    "video_between_frames",
    "video_from_image",
    "vr",
]