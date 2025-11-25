# runtime/pipeline_manager.py
"""
Pipeline manager for SD3.5 jobs (skeleton version).

IMPORTANT:
- This file does NOT load SD3.5.
- It does NOT run real inference.
- It ONLY:
    * reads/writes meta.json
    * plans text2img actions
    * can simulate a "render" by writing a dummy PNG
"""

import os
import json
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from typing import Dict, Any

from PIL import Image


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    PLANNED = "planned"
    DISPATCHED = "dispatched_skeleton"
    COMPLETED = "completed_skeleton"


@dataclass
class SD35Job:
    job_folder: str
    job_type: str
    meta: Dict[str, Any]


# ---------------------------------------------------------------------------
# Meta helpers
# ---------------------------------------------------------------------------

def _meta_path(job_folder: str) -> str:
    """Return the path to meta.json inside a job folder."""
    return os.path.join(job_folder, "meta.json")


def load_job_meta(job_folder: str) -> Dict[str, Any]:
    """
    Load meta.json from a job folder.

    Raises FileNotFoundError if meta.json is missing.
    """
    path = _meta_path(job_folder)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"meta.json not found in job folder: {job_folder}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_job_meta(job_folder: str, meta: Dict[str, Any]) -> None:
    """
    Save meta.json to a job folder.
    """
    os.makedirs(job_folder, exist_ok=True)
    path = _meta_path(job_folder)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Planning helpers (no real AI yet)
# ---------------------------------------------------------------------------

def make_text2img_plan_for_job(job_folder: str) -> Dict[str, Any]:
    """
    Given a job_folder for a text2img job, update meta.json with a
    "planned_actions" list describing what a future REAL GPU run
    would do.

    This does NOT run any AI. It just writes the plan.
    """
    meta = load_job_meta(job_folder)

    # Basic safety: ensure job type is text2img
    job_type = meta.get("type", "text2img")
    if job_type != "text2img":
        # For now we only support planning for text2img in this skeleton
        raise ValueError(f"make_text2img_plan_for_job only supports text2img, got: {job_type}")

    prompt = meta.get("prompt", "")
    negative_prompt = meta.get("negative_prompt", "")
    width = meta.get("width", 1024)
    height = meta.get("height", 1024)
    steps = meta.get("num_inference_steps", 25)
    guidance_scale = meta.get("guidance_scale", 6.0)
    style_preset = meta.get("style_preset")
    material_preset = meta.get("material_preset")
    lighting_preset = meta.get("lighting_preset")
    seed = meta.get("seed", None)

    out_rel = meta.get("planned_output_image", "output.png")

    planned_actions = [
        {
            "stage": "load_model",
            "model_name": "sd3.5-large",
            "device": "cuda",
            "description": "Load SD3.5 base model (future, real GPU only).",
        },
        {
            "stage": "text2img",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "style_preset": style_preset,
            "material_preset": material_preset,
            "lighting_preset": lighting_preset,
            "seed": seed,
            "out_path": out_rel,
            "description": "Run SD3.5 text2img (future, real GPU only).",
        },
        {
            "stage": "save_output",
            "target": out_rel,
            "description": "Save generated image (future, real GPU only).",
        },
    ]

    meta["planned_actions"] = planned_actions
    meta["gpu_planning"] = "done_skeleton"
    # We leave meta["status"] as-is ("planned") for now

    save_job_meta(job_folder, meta)
    return meta


# ---------------------------------------------------------------------------
# Simulated "render" helpers (still no real AI)
# ---------------------------------------------------------------------------

def simulate_text2img_render(job_folder: str) -> Dict[str, Any]:
    """
    Simulate a text2img GPU render by:

    - Reading meta.json
    - Creating a dummy output.png (solid color)
    - Updating meta["status"] to "completed_skeleton"
    - Setting meta["output_image"] and meta["completed_at"]
    - Saving meta.json

    This is still SKELETON-ONLY. No SD3.5, no torch, no GPU.
    """
    meta = load_job_meta(job_folder)

    job_type = meta.get("type", "text2img")
    if job_type != "text2img":
        # For now, we only simulate text2img
        raise ValueError(f"simulate_text2img_render only supports text2img, got: {job_type}")

    # Determine output path
    planned_rel = meta.get("planned_output_image", "output.png")
    out_name = os.path.basename(planned_rel) or "output.png"
    out_path = os.path.join(job_folder, out_name)

    # Make sure folder exists
    os.makedirs(job_folder, exist_ok=True)

    # Get width/height with defaults
    width = int(meta.get("width", 1024))
    height = int(meta.get("height", 1024))

    # Create a very simple dummy image (dark gray)
    img = Image.new("RGB", (width, height), (32, 32, 32))
    img.save(out_path)

    # Update meta
    meta["status"] = JobStatus.COMPLETED.value
    meta["output_image"] = out_name
    meta["completed_at"] = datetime.utcnow().isoformat()
    # Keep mode field to remind this is not real inference
    meta["mode"] = meta.get("mode", "skeleton-no-inference")

    save_job_meta(job_folder, meta)
    return meta
