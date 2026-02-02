# app/routers/video_between_frames.py
"""
RENDEREXPO AI STUDIO - Video Between Frames (REAL via GPU dispatch)

GOAL:
- Client uploads TWO images (PNG/JPG):
    - first_frame (start)
    - last_frame  (end)
- Create a job folder, save both images, write meta.json
- DISPATCH to GPU dispatcher so this is REAL

Output:
- video_between.mp4 (primary)
- frames/ (optional)
- viewer/index.html (optional)
- meta.json updated by GPU worker

Dispatch:
- job_type: "video_between_frames"
- GPU endpoint: VIDEO_GPU_DISPATCH_URL

SECURITY (Option A):
- GPU dispatch is HMAC-signed using RENDEREXPO_HMAC_SECRET
- Headers:
    X-RENDEREXPO-SIGNATURE
    X-RENDEREXPO-TIMESTAMP
    X-RENDEREXPO-NONCE
- Signed bytes:
    f"{timestamp}\\n{nonce}\\n".encode("utf-8") + raw_body_bytes
"""

from __future__ import annotations

import os
import uuid
import json
import time
import hmac
import hashlib
import shutil
import datetime
from typing import Optional, Any, Dict, Tuple

import requests
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

router = APIRouter(prefix="/api/video", tags=["Video (REAL)"])

HMAC_SECRET_ENV = "RENDEREXPO_HMAC_SECRET"
SIG_HEADER = "X-RENDEREXPO-SIGNATURE"
TS_HEADER = "X-RENDEREXPO-TIMESTAMP"
NONCE_HEADER = "X-RENDEREXPO-NONCE"


def _now_epoch() -> int:
    return int(time.time())


def _gpu_dispatch_url() -> str:
    return os.getenv("VIDEO_GPU_DISPATCH_URL", "http://127.0.0.1:8012/api/gpu/dispatch").strip()


def _abs(p: str) -> str:
    return os.path.abspath(p)


def _today_utc_str() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


def _create_job_folder(base_outputs_dir: str = "outputs") -> str:
    today = _today_utc_str()
    job_id = uuid.uuid4().hex
    folder = os.path.join(base_outputs_dir, today, job_id)
    os.makedirs(folder, exist_ok=True)
    return folder


def _parse_job_path(job_folder: str) -> Tuple[Optional[str], Optional[str]]:
    parts = os.path.normpath(job_folder).split(os.sep)
    if len(parts) < 3:
        return None, None
    return parts[-2], parts[-1]


def _outputs_public_urls(job_folder: str) -> Dict[str, Optional[str]]:
    date_str, job_id = _parse_job_path(job_folder)
    if not date_str or not job_id:
        return {
            "video_url": None,
            "meta_url": None,
            "viewer_url": None,
            "first_url": None,
            "last_url": None,
        }

    base = f"/outputs/{date_str}/{job_id}"
    return {
        "video_url": f"{base}/video_between.mp4",
        "meta_url": f"{base}/meta.json",
        "viewer_url": f"{base}/viewer/index.html",
        "first_url": f"{base}/first.png",
        "last_url": f"{base}/last.png",
    }


def _ensure_image_type(upload: UploadFile) -> None:
    ct = (getattr(upload, "content_type", "") or "").lower().strip()
    if ct and ct not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(status_code=400, detail="Only PNG and JPG are supported.")
    name = (upload.filename or "").lower().strip()
    if name and not (name.endswith(".png") or name.endswith(".jpg") or name.endswith(".jpeg")):
        raise HTTPException(status_code=400, detail="Only .png, .jpg, .jpeg are supported.")


async def _save_upload_stream(upload: UploadFile, dst_path: str) -> None:
    try:
        try:
            upload.file.seek(0)
        except Exception:
            pass
        with open(dst_path, "wb") as out:
            shutil.copyfileobj(upload.file, out)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to save upload '{upload.filename}': {exc}") from exc


def _compute_signature(secret: str, timestamp: str, nonce: str, body: bytes) -> str:
    prefix = f"{timestamp}\n{nonce}\n".encode("utf-8")
    msg = prefix + (body or b"")
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def _dispatch_to_gpu(job_folder_rel: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    url = _gpu_dispatch_url()

    payload = {
        "job_type": "video_between_frames",
        "job_folder": _abs(job_folder_rel),
        "meta": meta,
        "pipeline_key": "video::between_frames",
    }

    body_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    secret = (os.getenv(HMAC_SECRET_ENV) or "").strip()
    if not secret or len(secret) < 32:
        raise RuntimeError(
            f"Missing/weak {HMAC_SECRET_ENV}. "
            "Set the same strong secret on BOTH planner and GPU worker for Option A."
        )

    ts = str(_now_epoch())
    nonce = uuid.uuid4().hex
    sig = _compute_signature(secret=secret, timestamp=ts, nonce=nonce, body=body_bytes)

    headers = {
        "Content-Type": "application/json",
        SIG_HEADER: sig,
        TS_HEADER: ts,
        NONCE_HEADER: nonce,
    }

    try:
        r = requests.post(url, data=body_bytes, headers=headers, timeout=(10, 60))
        if not (200 <= r.status_code < 300):
            raise RuntimeError(f"GPU dispatch HTTP {r.status_code}: {r.text[:2000]}")
        return r.json() if r.content else {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"GPU dispatch failed: {exc}") from exc


@router.post("/between-frames")
async def video_between_frames(
    first_frame: UploadFile = File(..., description="Start image (PNG/JPG)"),
    last_frame: UploadFile = File(..., description="End image (PNG/JPG)"),

    prompt: Optional[str] = Form(None, description="Optional prompt to guide transition style"),
    negative_prompt: Optional[str] = Form(None, description="Optional negative prompt"),

    duration_seconds: float = Form(4.0, description="Target duration (seconds). 2.0..10.0"),
    fps: int = Form(24, description="FPS. 12..60"),
    motion_strength: float = Form(0.8, description="0..1 general motion amount"),
    seed: Optional[int] = Form(None, description="Optional seed"),
):
    if not first_frame.filename or not last_frame.filename:
        raise HTTPException(status_code=400, detail="Both uploaded images must have filenames.")
    _ensure_image_type(first_frame)
    _ensure_image_type(last_frame)

    if duration_seconds < 2.0 or duration_seconds > 10.0:
        raise HTTPException(status_code=400, detail="duration_seconds must be between 2.0 and 10.0")
    if fps < 12 or fps > 60:
        raise HTTPException(status_code=400, detail="fps must be between 12 and 60")
    if motion_strength < 0.0 or motion_strength > 1.0:
        raise HTTPException(status_code=400, detail="motion_strength must be between 0.0 and 1.0")

    # 1) Create job folder
    job_folder = _create_job_folder()
    job_id = os.path.basename(job_folder)
    public_urls = _outputs_public_urls(job_folder)

    # 2) Save frames
    first_path = os.path.join(job_folder, "first.png")
    last_path = os.path.join(job_folder, "last.png")
    await _save_upload_stream(first_frame, first_path)
    await _save_upload_stream(last_frame, last_path)

    # 3) Meta
    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "type": "video_between_frames",

        "status": "queued",
        "mode_runtime": "gpu-dispatch",

        "pipeline_key": "video::between_frames",

        "inputs": {
            "first_frame": "first.png",
            "last_frame": "last.png",
            "first_content_type": getattr(first_frame, "content_type", None),
            "last_content_type": getattr(last_frame, "content_type", None),
            "source": "upload",
        },

        "guidance": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
        },

        "video_runtime": {
            "duration_seconds": float(duration_seconds),
            "fps": int(fps),
            "motion_strength": float(motion_strength),
            "seed": seed,
        },

        "outputs": {
            "video_between": "video_between.mp4",
            "frames_dir": "frames/",
            "viewer_dir": "viewer/",
        },

        "dispatch": {
            "job_type": "video_between_frames",
            "target": _gpu_dispatch_url(),
            "dispatched_at": None,
            "gpu_response": None,
            "error": None,
        },
    }

    meta_path = os.path.join(job_folder, "meta.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to write meta.json: {exc}") from exc

    # 4) Dispatch
    try:
        gpu_resp = _dispatch_to_gpu(job_folder, meta)

        try:
            meta["status"] = "dispatched"
            meta["dispatch"]["dispatched_at"] = datetime.datetime.utcnow().isoformat()
            meta["dispatch"]["gpu_response"] = gpu_resp
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4)
        except Exception:
            pass

        return {
            "status": "dispatched",
            "job_id": job_id,
            "job_folder": job_folder,
            "meta_path": meta_path,
            "public_urls": public_urls,
            "gpu_response": gpu_resp,
        }

    except Exception as exc:  # noqa: BLE001
        try:
            meta["status"] = "gpu_error"
            meta["dispatch"]["error"] = {"detail": str(exc)}
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4)
        except Exception:
            pass

        return {
            "status": "gpu_error",
            "job_id": job_id,
            "job_folder": job_folder,
            "meta_path": meta_path,
            "public_urls": public_urls,
            "gpu_error": {
                "detail": str(exc),
                "dispatch_url": _gpu_dispatch_url(),
                "job_folder_sent": _abs(job_folder),
            },
        }
