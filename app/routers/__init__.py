# app/routers/__init__.py
"""
Central export for planner-side FastAPI routers used in RENDEREXPO AI STUDIO.

IMPORTANT:
- Planner = port 8012
- GPU worker = port 8002
- This module is for planner routers only.
- Do NOT add the GPU worker dispatch router here.
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