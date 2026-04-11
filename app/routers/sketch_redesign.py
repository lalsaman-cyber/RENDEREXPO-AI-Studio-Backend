# app/routers/sketch_redesign.py
"""
RENDEREXPO AI STUDIO - Sketch to Redesign Router (Planner -> GPU worker)

PURPOSE
-------
- Keep the working Sketch to Render route untouched.
- Add a NEW parallel Sketch to Redesign planner route.
- Use the same planner/session/upload/output style as sketch.py.
- Dispatch redesign jobs to a NEW MistoLine-based redesign pipeline.
- Do NOT revive the removed SD3.5 sketch redesign path.

LOCKED PRODUCT RULE
-------------------
Sketch to Render:
- preservation-first
- existing working MistoLine route
- untouched by this file

Sketch to Redesign:
- reinterpretation-first
- same sketch family / same backend system
- different route identity
- different prompt-building behavior
- more creative, less strict than Sketch to Render

LOCKED PIPELINE KEY
-------------------
    sdxl::mistoline_sketch_redesign

LOCKED JOB TYPE
---------------
    sdxl_mistoline_sketch_redesign
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.services.sketch_redesign_prompt import (
    SKETCH_REDESIGN_PRODUCT_PROMISE,
    SKETCH_REDESIGN_WARNING_TEXT,
    build_sketch_redesign_prompt_package,
)

router = APIRouter(prefix="/api/sketch-redesign", tags=["Sketch Redesign"])


# ---------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return default
    value = str(value).strip()
    return value if value else default


OUTPUTS_ROOT = _env("RENDEREXPO_OUTPUTS_ROOT", "outputs") or "outputs"
OUTPUTS_MOUNT = _env("RENDEREXPO_OUTPUTS_MOUNT", "/outputs") or "/outputs"
GPU_BASE_URL = _env("GPU_BASE_URL", "http://127.0.0.1:8002") or "http://127.0.0.1:8002"
GPU_TIMEOUT_SECONDS = int(_env("GPU_TIMEOUT_SECONDS", "600") or "600")


# ---------------------------------------------------------------------
# Locked route identity
# ---------------------------------------------------------------------

SKETCH_REDESIGN_JOB_TYPE = "sdxl_mistoline_sketch_redesign"
SKETCH_REDESIGN_PIPELINE_KEY = "sdxl::mistoline_sketch_redesign"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _utc_iso() -> str:
    return datetime.datetime.utcnow().isoformat()


def _today_utc_str() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, data: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _clean(value: Optional[str]) -> str:
    return " ".join(str(value or "").strip().split())


def _ensure_png_jpg_only(upload: UploadFile) -> None:
    ct = (getattr(upload, "content_type", "") or "").lower().strip()
    if ct and ct not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail="Only PNG and JPG are supported for sketch redesign uploads.",
        )


def _save_upload(upload: UploadFile, target_path: str) -> None:
    _ensure_dir(os.path.dirname(target_path))
    try:
        with open(target_path, "wb") as f:
            shutil.copyfileobj(upload.file, f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc
    finally:
        try:
            upload.file.close()
        except Exception:
            pass


def _session_root(session_id: str) -> str:
    return os.path.join(OUTPUTS_ROOT, _today_utc_str(), "sessions", session_id)


def _session_meta_path(session_id: str) -> str:
    return os.path.join(_session_root(session_id), "session_meta.json")


def _frames_dir(session_id: str) -> str:
    return os.path.join(_session_root(session_id), "frames")


def _create_session_folder(session_id: str) -> str:
    root = _session_root(session_id)
    _ensure_dir(root)
    _ensure_dir(_frames_dir(session_id))
    return root


def _create_job_folder() -> str:
    job_id = uuid.uuid4().hex
    root = os.path.join(OUTPUTS_ROOT, _today_utc_str(), job_id)
    _ensure_dir(root)
    return root


def _job_public_urls(job_folder: str) -> Dict[str, str]:
    abs_job = os.path.abspath(job_folder)
    abs_root = os.path.abspath(OUTPUTS_ROOT)

    try:
        rel = os.path.relpath(abs_job, abs_root).replace("\\", "/")
    except Exception:
        rel = os.path.basename(abs_job)

    if rel.startswith("../"):
        rel = os.path.basename(abs_job)

    mount_root = OUTPUTS_MOUNT.rstrip("/")

    return {
        "job_folder": f"{mount_root}/{rel}",
        "meta_json": f"{mount_root}/{rel}/meta.json",
        "sketch_png": f"{mount_root}/{rel}/sketch.png",
        "output_png": f"{mount_root}/{rel}/output.png",
        "final_up2x_png": f"{mount_root}/{rel}/final_up2x.png",
    }


def _dispatch_to_gpu(*, job_type: str, job_folder: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    body = {
        "job_type": job_type,
        "job_folder": os.path.abspath(job_folder),
        "meta": meta,
        "pipeline_key": meta.get("pipeline_key"),
        "vr_mode": None,
    }
    data = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(
        url=f"{GPU_BASE_URL.rstrip('/')}/api/gpu/dispatch",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=GPU_TIMEOUT_SECONDS) as resp:
            raw = resp.read().decode("utf-8")
            payload = json.loads(raw) if raw else {}
            return payload if isinstance(payload, dict) else {"status": "accepted"}
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            detail = str(exc)
        raise HTTPException(
            status_code=502,
            detail=f"GPU dispatch failed ({exc.code}): {detail}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"GPU dispatch failed: {exc}") from exc


def _latest_frame_path(session_id: str) -> Optional[str]:
    frames_root = _frames_dir(session_id)
    if not os.path.isdir(frames_root):
        return None

    candidates = []
    for name in os.listdir(frames_root):
        full = os.path.join(frames_root, name)
        if os.path.isfile(full):
            candidates.append(full)

    if not candidates:
        return None

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _build_session_meta(session_id: str) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "created_at": _utc_iso(),
        "updated_at": _utc_iso(),
        "mode": "sketch_to_redesign",
        "product_promise": SKETCH_REDESIGN_PRODUCT_PROMISE,
        "warning_text": SKETCH_REDESIGN_WARNING_TEXT,
        "frames_dir": _frames_dir(session_id),
        "latest_frame": None,
    }


# ---------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------

class CreateSketchRedesignSessionResponse(BaseModel):
    status: str = "ok"
    session_id: str
    session_root: str
    session_meta_json: str
    frames_dir: str
    mode: str
    product_promise: str
    warning_text: str


class UploadSketchRedesignFrameResponse(BaseModel):
    status: str = "ok"
    session_id: str
    saved_filename: str
    saved_path: str
    session_meta_json: str


class SketchRedesignGenerateResponse(BaseModel):
    status: str
    session_id: str
    job_folder: str
    meta_json: str
    sketch_png: str
    output_png: str
    final_up2x_png: str
    pipeline_key: str
    job_type: str
    message: str


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@router.post("/session/create", response_model=CreateSketchRedesignSessionResponse)
def create_sketch_redesign_session() -> CreateSketchRedesignSessionResponse:
    session_id = uuid.uuid4().hex
    root = _create_session_folder(session_id)
    meta = _build_session_meta(session_id)
    _write_json(_session_meta_path(session_id), meta)

    urls = _job_public_urls(root)

    return CreateSketchRedesignSessionResponse(
        session_id=session_id,
        session_root=urls["job_folder"],
        session_meta_json=urls["meta_json"],
        frames_dir=f"{urls['job_folder']}/frames",
        mode="sketch_to_redesign",
        product_promise=SKETCH_REDESIGN_PRODUCT_PROMISE,
        warning_text=SKETCH_REDESIGN_WARNING_TEXT,
    )


@router.post("/session/{session_id}/upload", response_model=UploadSketchRedesignFrameResponse)
def upload_sketch_redesign_frame(
    session_id: str,
    file: UploadFile = File(...),
) -> UploadSketchRedesignFrameResponse:
    _ensure_png_jpg_only(file)

    session_root = _session_root(session_id)
    if not os.path.isdir(session_root):
        raise HTTPException(status_code=404, detail="Sketch redesign session not found.")

    ext = os.path.splitext(file.filename or "")[1].lower().strip()
    if ext not in {".png", ".jpg", ".jpeg"}:
        ext = ".png"

    saved_filename = f"frame_{uuid.uuid4().hex}{ext}"
    saved_path = os.path.join(_frames_dir(session_id), saved_filename)
    _save_upload(file, saved_path)

    meta_path = _session_meta_path(session_id)
    meta = _read_json(meta_path)
    meta["updated_at"] = _utc_iso()
    meta["latest_frame"] = saved_path
    meta.setdefault("uploaded_frames", [])
    if isinstance(meta["uploaded_frames"], list):
        meta["uploaded_frames"].append(saved_path)
    _write_json(meta_path, meta)

    urls = _job_public_urls(session_root)

    return UploadSketchRedesignFrameResponse(
        session_id=session_id,
        saved_filename=saved_filename,
        saved_path=f"{urls['job_folder']}/frames/{saved_filename}",
        session_meta_json=urls["meta_json"],
    )


@router.post("/session/{session_id}/generate", response_model=SketchRedesignGenerateResponse)
def generate_sketch_redesign(
    session_id: str,
    style_preset: Optional[str] = Form(default=None),
    materials_notes: Optional[str] = Form(default=None),
    atmosphere_notes: Optional[str] = Form(default=None),
    background_notes: Optional[str] = Form(default=None),
    mood_notes: Optional[str] = Form(default=None),
    style_notes: Optional[str] = Form(default=None),
    aesthetic_notes: Optional[str] = Form(default=None),
    seed: Optional[int] = Form(default=None),
) -> SketchRedesignGenerateResponse:
    session_root = _session_root(session_id)
    if not os.path.isdir(session_root):
        raise HTTPException(status_code=404, detail="Sketch redesign session not found.")

    source_sketch = _latest_frame_path(session_id)
    if not source_sketch:
        raise HTTPException(status_code=400, detail="No uploaded sketch frame found for this redesign session.")

    prompt_package = build_sketch_redesign_prompt_package(
        style_preset=style_preset,
        materials_notes=materials_notes,
        atmosphere_notes=atmosphere_notes,
        background_notes=background_notes,
        mood_notes=mood_notes,
        style_notes=style_notes,
        aesthetic_notes=aesthetic_notes,
    )

    job_folder = _create_job_folder()
    sketch_png_path = os.path.join(job_folder, "sketch.png")
    shutil.copy2(source_sketch, sketch_png_path)

    job_meta: Dict[str, Any] = {
        "status": "queued",
        "created_at": _utc_iso(),
        "updated_at": _utc_iso(),
        "mode": "sketch_to_redesign",
        "product_promise": SKETCH_REDESIGN_PRODUCT_PROMISE,
        "warning_text": SKETCH_REDESIGN_WARNING_TEXT,
        "session_id": session_id,
        "job_type": SKETCH_REDESIGN_JOB_TYPE,
        "pipeline_key": SKETCH_REDESIGN_PIPELINE_KEY,
        "engine_family": "sdxl",
        "engine": "sdxl_base_1_0",
        "control_model": "TheMistoAI/MistoLine",
        "source_sketch_path": source_sketch,
        "input_image_path": sketch_png_path,
        "prompt": prompt_package.prompt,
        "negative_prompt": prompt_package.negative_prompt,
        "seed": seed,
        "style_preset": prompt_package.style_preset,
        "allowed_client_fields": prompt_package.allowed_client_fields,
        "outputs": {
            "sketch_png": "sketch.png",
            "output_png": "output.png",
            "final_up2x_png": "final_up2x.png",
        },
        "redesign": {
            "mode": "reinterpretation_first",
            "strict_structure_preservation": False,
        },
    }

    _write_json(os.path.join(job_folder, "meta.json"), job_meta)

    dispatch_payload = _dispatch_to_gpu(
        job_type=SKETCH_REDESIGN_JOB_TYPE,
        job_folder=job_folder,
        meta=job_meta,
    )

    public_urls = _job_public_urls(job_folder)
    status = str(dispatch_payload.get("status", "accepted"))
    message = str(dispatch_payload.get("message", "Sketch redesign job accepted."))

    return SketchRedesignGenerateResponse(
        status=status,
        session_id=session_id,
        job_folder=public_urls["job_folder"],
        meta_json=public_urls["meta_json"],
        sketch_png=public_urls["sketch_png"],
        output_png=public_urls["output_png"],
        final_up2x_png=public_urls["final_up2x_png"],
        pipeline_key=SKETCH_REDESIGN_PIPELINE_KEY,
        job_type=SKETCH_REDESIGN_JOB_TYPE,
        message=message,
    )