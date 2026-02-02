# app/routers/mesh_from_image.py
"""
RENDEREXPO AI STUDIO - Mesh From Image (REAL via GPU dispatch)

Planner-side router (PC-first / POD-once):
- Creates job folder under outputs/YYYY-MM-DD/<job_id>/
- Writes image.png + meta.json
- Dispatches to GPU worker via HMAC-signed request (Option A)

Why add /from-image-base64?
- Wix/Velo multipart uploads are annoying and inconsistent.
- JSON base64 is reliable, easy to sign, easy to send from Wix backend.

Endpoints:
1) POST /api/mesh/from-image         (multipart UploadFile)  -> good for local testing
2) POST /api/mesh/from-image-base64  (JSON base64)           -> best for Wix

GPU worker writes:
- mesh.obj (required)
- mesh.glb (optional)
- preview.png (optional)
- depth.png (optional)
- meta.json updated by GPU worker
"""

from __future__ import annotations

import os
import uuid
import json
import time
import hmac
import base64
import hashlib
import shutil
import datetime
from typing import Optional, Any, Dict, Tuple

import requests
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/mesh", tags=["Mesh (REAL)"])

# ---------------------------------------------------------------------------
# HMAC constants (must match app/main.py)
# ---------------------------------------------------------------------------

HMAC_SECRET_ENV = "RENDEREXPO_HMAC_SECRET"

SIG_HEADER = "X-RENDEREXPO-SIGNATURE"
TS_HEADER = "X-RENDEREXPO-TIMESTAMP"
NONCE_HEADER = "X-RENDEREXPO-NONCE"

# Safety limits (Planner side)
MAX_IMAGE_BYTES = int(os.getenv("MESH_MAX_IMAGE_BYTES", str(20 * 1024 * 1024)))  # 20MB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_epoch() -> int:
    return int(time.time())


def _mesh_dispatch_url() -> str:
    """
    GPU dispatch endpoint (GPU worker service).
    Recommended:
      MESH_GPU_DISPATCH_URL=http://127.0.0.1:8012/api/gpu/dispatch
    """
    return os.getenv("MESH_GPU_DISPATCH_URL", "http://127.0.0.1:8012/api/gpu/dispatch").strip()


def _abs(p: str) -> str:
    return os.path.abspath(p)


def _today_utc_str() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


def _create_job_folder(base_outputs_dir: str = "outputs") -> str:
    """Create outputs/{YYYY-MM-DD}/{job_id}/ and return its relative path."""
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
    """
    Stable URLs assuming FastAPI mounts outputs/ at /outputs.
    """
    date_str, job_id = _parse_job_path(job_folder)
    if not date_str or not job_id:
        return {
            "obj_url": None,
            "glb_url": None,
            "preview_url": None,
            "depth_url": None,
            "meta_url": None,
        }

    base = f"/outputs/{date_str}/{job_id}"
    return {
        "obj_url": f"{base}/mesh.obj",
        "glb_url": f"{base}/mesh.glb",
        "preview_url": f"{base}/preview.png",
        "depth_url": f"{base}/depth.png",
        "meta_url": f"{base}/meta.json",
    }


async def _save_upload_stream(upload: UploadFile, dst_path: str) -> None:
    """Save UploadFile without reading everything into RAM."""
    try:
        try:
            upload.file.seek(0)
        except Exception:
            pass

        written = 0
        with open(dst_path, "wb") as out:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_IMAGE_BYTES:
                    raise HTTPException(status_code=413, detail=f"Image too large (>{MAX_IMAGE_BYTES} bytes)")
                out.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to save upload '{upload.filename}': {exc}") from exc


def _compute_signature(secret: str, timestamp: str, nonce: str, body: bytes) -> str:
    prefix = f"{timestamp}\n{nonce}\n".encode("utf-8")
    msg = prefix + (body or b"")
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def _dispatch_to_gpu(job_folder_rel: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatch to GPU handler (HMAC signed, Option A).

    We send ABSOLUTE job_folder so the worker can access the folder (same filesystem).
    IMPORTANT: we sign the exact raw bytes we send.
    """
    url = _mesh_dispatch_url()

    payload = {
        "job_type": "mesh_from_image",
        "job_folder": _abs(job_folder_rel),
        "meta": meta,
        # keep routing explicit/consistent with dispatcher
        "pipeline_key": "mesh::from_image",
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


def _decode_base64_image(data_b64: str) -> Tuple[bytes, str]:
    """
    Accepts:
      - raw base64
      - data URL: data:image/png;base64,....
    Returns: (bytes, inferred_mime)
    """
    if not data_b64 or not isinstance(data_b64, str):
        raise HTTPException(status_code=400, detail="image_base64 is required")

    s = data_b64.strip()

    inferred_mime = "application/octet-stream"
    if s.startswith("data:"):
        # data:image/png;base64,AAAA
        try:
            header, b64 = s.split(",", 1)
            inferred_mime = header.split(";")[0].replace("data:", "").strip() or inferred_mime
            s = b64.strip()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid data URL format for image_base64")

    try:
        raw = base64.b64decode(s, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 data for image_base64")

    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"Image too large (>{MAX_IMAGE_BYTES} bytes)")

    # very light sniffing (optional)
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        inferred_mime = "image/png"
    elif raw[:3] == b"\xff\xd8\xff":
        inferred_mime = "image/jpeg"

    return raw, inferred_mime


def _validate_knobs(detail_level: str, target_faces: int, max_depth_m: float) -> str:
    detail_clean = (detail_level or "").strip().lower() or "medium"
    if detail_clean not in ("low", "medium", "high"):
        raise HTTPException(status_code=400, detail="detail_level must be low|medium|high")
    if target_faces < 2000 or target_faces > 2_000_000:
        raise HTTPException(status_code=400, detail="target_faces must be 2000..2000000")
    if max_depth_m < 2.0 or max_depth_m > 200.0:
        raise HTTPException(status_code=400, detail="max_depth_m must be 2..200")
    return detail_clean


def _write_meta(job_folder: str, meta: Dict[str, Any]) -> str:
    meta_path = os.path.join(job_folder, "meta.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to write meta.json: {exc}") from exc
    return meta_path


# ---------------------------------------------------------------------------
# JSON request model (best for Wix)
# ---------------------------------------------------------------------------

class MeshFromImageBase64Request(BaseModel):
    image_base64: str = Field(..., description="Base64 image, raw or data URL.")
    detail_level: str = Field(default="medium", description="low|medium|high")
    target_faces: int = Field(default=250000, description="Target face count (approx)")
    max_depth_m: float = Field(default=40.0, description="Clamp depth (meters)")
    seed: Optional[int] = Field(default=None, description="Optional seed")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/from-image")
async def mesh_from_image(
    image: UploadFile = File(..., description="Single input image"),
    detail_level: str = Form("medium", description="low|medium|high"),
    target_faces: int = Form(250000, description="Target face count (approx)"),
    max_depth_m: float = Form(40.0, description="Clamp depth (meters)"),
    seed: Optional[int] = Form(None, description="Optional seed"),
):
    """
    Multipart upload (good for local testing).
    Wix should prefer /from-image-base64 (JSON).
    """
    detail_clean = _validate_knobs(detail_level, target_faces, max_depth_m)

    job_folder = _create_job_folder()
    job_id = os.path.basename(job_folder)

    img_path = os.path.join(job_folder, "image.png")
    await _save_upload_stream(image, img_path)

    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "type": "mesh_from_image",
        "status": "queued",
        "mode_runtime": "gpu-dispatch",
        "inputs": {
            "image": "image.png",
            "content_type": getattr(image, "content_type", None),
            "source": "multipart_upload",
        },
        "mesh_runtime": {
            "detail_level": detail_clean,
            "target_faces": int(target_faces),
            "max_depth_m": float(max_depth_m),
            "seed": seed,
        },
        "outputs": {
            "obj": "mesh.obj",
            "glb": "mesh.glb",
            "preview": "preview.png",
            "depth": "depth.png",
            "meta": "meta.json",
        },
        "pipeline_key": "mesh::from_image",
        "dispatch": {
            "job_type": "mesh_from_image",
            "target": _mesh_dispatch_url(),
            "dispatched_at": None,
            "gpu_response": None,
            "error": None,
        },
    }

    meta_path = _write_meta(job_folder, meta)
    public_urls = _outputs_public_urls(job_folder)

    try:
        gpu_resp = _dispatch_to_gpu(job_folder, meta)

        try:
            meta["status"] = "dispatched"
            meta["dispatch"]["dispatched_at"] = datetime.datetime.utcnow().isoformat()
            meta["dispatch"]["gpu_response"] = gpu_resp
            _write_meta(job_folder, meta)
        except Exception:
            pass

        return {
            "status": "dispatched",
            "message": "Mesh-from-image job dispatched to GPU.",
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
            _write_meta(job_folder, meta)
        except Exception:
            pass

        return {
            "status": "gpu_error",
            "message": "Job created but GPU dispatch failed.",
            "job_id": job_id,
            "job_folder": job_folder,
            "meta_path": meta_path,
            "public_urls": public_urls,
            "gpu_error": {
                "detail": str(exc),
                "dispatch_url": _mesh_dispatch_url(),
                "job_folder_sent": _abs(job_folder),
            },
        }


@router.post("/from-image-base64")
async def mesh_from_image_base64(req: MeshFromImageBase64Request):
    """
    JSON base64 (best for Wix).
    """
    detail_clean = _validate_knobs(req.detail_level, req.target_faces, req.max_depth_m)

    raw, inferred_mime = _decode_base64_image(req.image_base64)

    job_folder = _create_job_folder()
    job_id = os.path.basename(job_folder)

    img_path = os.path.join(job_folder, "image.png")
    try:
        with open(img_path, "wb") as f:
            f.write(raw)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed writing image.png: {exc}")

    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "type": "mesh_from_image",
        "status": "queued",
        "mode_runtime": "gpu-dispatch",
        "inputs": {
            "image": "image.png",
            "content_type": inferred_mime,
            "source": "base64_json",
        },
        "mesh_runtime": {
            "detail_level": detail_clean,
            "target_faces": int(req.target_faces),
            "max_depth_m": float(req.max_depth_m),
            "seed": req.seed,
        },
        "outputs": {
            "obj": "mesh.obj",
            "glb": "mesh.glb",
            "preview": "preview.png",
            "depth": "depth.png",
            "meta": "meta.json",
        },
        "pipeline_key": "mesh::from_image",
        "dispatch": {
            "job_type": "mesh_from_image",
            "target": _mesh_dispatch_url(),
            "dispatched_at": None,
            "gpu_response": None,
            "error": None,
        },
    }

    meta_path = _write_meta(job_folder, meta)
    public_urls = _outputs_public_urls(job_folder)

    try:
        gpu_resp = _dispatch_to_gpu(job_folder, meta)

        try:
            meta["status"] = "dispatched"
            meta["dispatch"]["dispatched_at"] = datetime.datetime.utcnow().isoformat()
            meta["dispatch"]["gpu_response"] = gpu_resp
            _write_meta(job_folder, meta)
        except Exception:
            pass

        return {
            "status": "dispatched",
            "message": "Mesh-from-image job dispatched to GPU.",
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
            _write_meta(job_folder, meta)
        except Exception:
            pass

        return {
            "status": "gpu_error",
            "message": "Job created but GPU dispatch failed.",
            "job_id": job_id,
            "job_folder": job_folder,
            "meta_path": meta_path,
            "public_urls": public_urls,
            "gpu_error": {
                "detail": str(exc),
                "dispatch_url": _mesh_dispatch_url(),
                "job_folder_sent": _abs(job_folder),
            },
        }
