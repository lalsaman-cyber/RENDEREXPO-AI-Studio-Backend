# app/routers/sketch.py

"""
Sketch → Architecture (Realtime) — SKELETON ROUTER

This does NOT run SD3.5 or any realtime AI yet.

It only:
- creates a "sketch session" job folder
- saves uploaded sketch frames (PNG)
- writes meta.json with planned pipeline steps

Later, the GPU runtime will:
- watch these sessions
- take latest frame + prompt
- run SD3.5 + ControlNet
- stream back previews to the UI (Wix canvas)
"""

import os
import uuid
import datetime
import json
from typing import Dict, Any, List

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

router = APIRouter(prefix="/api/sketch", tags=["Sketch Realtime"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_job_folder(base_outputs_dir: str = "outputs") -> str:
    """
    Create a new outputs/{YYYY-MM-DD}/{job_id}/ folder
    for a sketch session.
    """
    today = datetime.date.today().isoformat()
    job_id = uuid.uuid4().hex
    folder = os.path.join(base_outputs_dir, today, job_id)
    os.makedirs(folder, exist_ok=True)
    return folder


def _ensure_folder_exists(job_folder: str) -> None:
    """Ensure the given job_folder exists, or raise 400."""
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
        return {}
    with open(meta_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_meta(job_folder: str, meta: Dict[str, Any]) -> None:
    meta_file = _meta_path(job_folder)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)


# ---------------------------------------------------------------------------
# 1) Start a realtime sketch session
# ---------------------------------------------------------------------------

@router.post("/start-session")
async def start_sketch_session(
    prompt: str = Form(
        ...,
        description=(
            "Design intent, e.g. "
            "'modern living room, soft luxury, neutral tones' "
            "or 'urban block massing, mid-rise, green roofs'."
        ),
    ),
    width: int = Form(
        1024,
        description="Target viewport width for the rendered previews (pixels).",
    ),
    height: int = Form(
        1024,
        description="Target viewport height for the rendered previews (pixels).",
    ),
):
    """
    Create a new SKETCH session.

    SKELETON behavior:
    - creates job folder
    - writes meta.json with:
        * session info
        * prompt
        * target resolution
        * planned pipeline stages

    No realtime AI yet.
    """
    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="width and height must be positive")

    job_folder = _create_job_folder()
    created = datetime.datetime.utcnow().isoformat()

    planned_actions: List[Dict[str, Any]] = [
        {
            "stage": "init_session",
            "description": "Initialize realtime sketch session.",
        },
        {
            "stage": "receive_strokes",
            "description": (
                "Receive strokes / sketch frames from the UI and "
                "store them in this job folder."
            ),
        },
        {
            "stage": "build_control_signal",
            "description": (
                "Convert latest sketch frame into canny/lineart/depth "
                "maps for ControlNet (future GPU step)."
            ),
        },
        {
            "stage": "sd35_render_live",
            "description": (
                "Use SD3.5 + ControlNet to render live previews from "
                "the sketch + prompt (future GPU step)."
            ),
        },
    ]

    meta: Dict[str, Any] = {
        "job_id": os.path.basename(job_folder),
        "created_at": created,
        "type": "sketch_realtime_session",
        "prompt": prompt,
        "width": width,
        "height": height,
        "status": "session_started",
        "pipeline": "sketch_realtime_v1",
        "planned_actions": planned_actions,
        "frames": [],  # will append here as frames arrive
        "mode_runtime": "skeleton-no-inference",
    }

    meta_file = _meta_path(job_folder)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    return {
        "status": "ok",
        "message": "Realtime sketch session started (skeleton, no AI yet).",
        "job_folder": job_folder,
        "meta_path": meta_file,
    }


# ---------------------------------------------------------------------------
# 2) Upload a sketch frame (snapshot of the canvas)
# ---------------------------------------------------------------------------

@router.post("/upload-frame")
async def upload_sketch_frame(
    job_folder: str = Form(..., description="Session job folder from /start-session"),
    frame_index: int = Form(
        ...,
        description=(
            "Frame index (0, 1, 2, ...). "
            "The UI can just increment this for each snapshot."
        ),
    ),
    sketch_image: UploadFile = File(
        ...,
        description="Current canvas as PNG/JPG. The UI can send a snapshot every few strokes.",
    ),
):
    """
    Upload a single snapshot of the sketch canvas.

    SKELETON behavior:
    - checks job_folder
    - saves the image as sketch_frame_{frame_index}.png
    - appends metadata into meta['frames']
    - NO SD3.5 yet
    """
    _ensure_folder_exists(job_folder)

    # Save frame image
    frame_filename = f"sketch_frame_{frame_index}.png"
    frame_path = os.path.join(job_folder, frame_filename)
    with open(frame_path, "wb") as f:
        f.write(await sketch_image.read())

    # Update meta
    meta = _read_meta(job_folder)
    if "frames" not in meta or not isinstance(meta["frames"], list):
        meta["frames"] = []

    frame_info = {
        "index": frame_index,
        "file": frame_filename,
        "uploaded_at": datetime.datetime.utcnow().isoformat(),
    }
    meta["frames"].append(frame_info)
    meta["last_frame_index"] = frame_index
    meta["status"] = "frames_receiving"

    _write_meta(job_folder, meta)

    return {
        "status": "ok",
        "message": "Sketch frame uploaded (skeleton, no AI yet).",
        "job_folder": job_folder,
        "frame": frame_info,
        "meta_path": _meta_path(job_folder),
    }
