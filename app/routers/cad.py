# app/routers/cad.py
"""
RENDEREXPO AI STUDIO - CAD From Image (REAL via GPU dispatch)

GOAL (Wix-ready):
- Planner creates job folder, writes input.png, writes meta.json
- Planner DISPATCHES to GPU worker (HMAC-signed Option A)
- GPU worker writes:
    - output.dxf (required)
    - output.dwg (best-effort)
    - lines_preview.png (recommended)
    - meta.json updated

Endpoints:
1) POST /api/cad/from-image
   - multipart upload (BEST default for Wix: normal upload)
2) POST /api/cad/from-image-base64
   - JSON upload (image as text) (kept as backup if Wix JSON-only)
3) POST /api/cad/from-job
   - JSON: reference an image from a prior job
4) POST /api/cad/from-job-base64
   - Alias name for Wix consistency (same behavior as /from-job)

Security:
- Planner is HMAC protected globally (middleware in app/main.py)
- Dispatch to GPU worker is HMAC signed (Option A) using RENDEREXPO_HMAC_SECRET

Notes:
- Planner runs on PC (port 8002)
- GPU worker runs on POD (port 8012) reachable via CAD_GPU_DISPATCH_URL (or VIDEO_GPU_DISPATCH_URL fallback)
- job_folder is sent as an ABSOLUTE path; GPU worker must be able to read/write it
  (shared volume / same filesystem view).

UPDATE IN THIS VERSION:
- Adds strict input validation (units, semantic level, numeric ranges).
- Prevents path traversal on from-job via _safe_relpath().
- Adds lightweight job_type.txt marker.
- Keeps HMAC signing EXACT raw bytes (stable json separators).
- Keeps BOTH multipart and base64 routes (since you said keep base64 as backup).
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
import base64
import binascii
from typing import Optional, Any, Dict, Tuple, Literal

import requests
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/cad", tags=["CAD (REAL)"])

# ---------------------------------------------------------------------------
# HMAC constants (must match app/main.py)
# ---------------------------------------------------------------------------

HMAC_SECRET_ENV = "RENDEREXPO_HMAC_SECRET"
SIG_HEADER = "X-RENDEREXPO-SIGNATURE"
TS_HEADER = "X-RENDEREXPO-TIMESTAMP"
NONCE_HEADER = "X-RENDEREXPO-NONCE"

# ---------------------------------------------------------------------------
# Types / Models
# ---------------------------------------------------------------------------

ScaleMode = Literal["two_point", "door_height_fallback"]
ScaleUnit = Literal["m", "cm", "mm", "ft"]
SemanticLevel = Literal["basic", "architectural"]


class Point(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0, description="Normalized 0..1")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalized 0..1")


class CadFromJobRequest(BaseModel):
    source_job_id: str = Field(..., description="Job ID (folder name) under outputs/*/<job_id>/")
    source_image_path: str = Field(default="output.png", description="Relative path within the source job folder")

    scale_mode: ScaleMode = Field(default="two_point")
    scale_reference_distance: Optional[float] = Field(default=None, gt=0.0)
    scale_reference_unit: ScaleUnit = Field(default="m")

    scale_point_1: Optional[Point] = None
    scale_point_2: Optional[Point] = None

    fallback_door_height: float = Field(default=2.10, gt=0.5, lt=10.0)

    semantic_level: SemanticLevel = Field(default="architectural")
    layer_profile: str = Field(default="standard")


class CadFromImageBase64Request(BaseModel):
    """
    JSON upload option (backup if Wix is JSON-only):
    - Send image_base64 as:
        - pure base64 OR
        - data URL like "data:image/png;base64,...."
    - PNG/JPG only (locked by user)
    """
    image_base64: str = Field(..., description="Base64 or data URL (PNG/JPG only)")

    scale_mode: ScaleMode = Field(default="two_point")
    scale_reference_distance: Optional[float] = Field(
        default=None, description="Real distance between points, in unit", gt=0.0
    )
    scale_reference_unit: ScaleUnit = Field(default="m")

    scale_point_1: Optional[Point] = None
    scale_point_2: Optional[Point] = None

    fallback_door_height: float = Field(default=2.10, gt=0.5, lt=10.0)

    semantic_level: SemanticLevel = Field(default="architectural")
    layer_profile: str = Field(default="standard")


class CadFromJobBase64Request(BaseModel):
    """
    Wix-friendly naming alias of from-job (still JSON).
    """
    source_job_id: str = Field(..., description="Job ID (folder name) under outputs/*/<job_id>/")
    source_image_path: str = Field(default="output.png", description="Relative path within the source job folder")

    scale_mode: ScaleMode = Field(default="two_point")
    scale_reference_distance: Optional[float] = Field(default=None, gt=0.0)
    scale_reference_unit: ScaleUnit = Field(default="m")

    scale_point_1: Optional[Point] = None
    scale_point_2: Optional[Point] = None

    fallback_door_height: float = Field(default=2.10, gt=0.5, lt=10.0)

    semantic_level: SemanticLevel = Field(default="architectural")
    layer_profile: str = Field(default="standard")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_epoch() -> int:
    return int(time.time())


def _gpu_dispatch_url() -> str:
    """
    CAD dispatch endpoint.
    Uses same GPU dispatch service as VR/Video.

    Priority:
      CAD_GPU_DISPATCH_URL
      VIDEO_GPU_DISPATCH_URL
      default local gpu worker
    """
    return os.getenv(
        "CAD_GPU_DISPATCH_URL",
        os.getenv("VIDEO_GPU_DISPATCH_URL", "http://127.0.0.1:8012/api/gpu/dispatch"),
    ).strip()


def _abs(p: str) -> str:
    return os.path.abspath(p)


def _today_utc_str() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


def _create_job_folder(base_outputs_dir: str = "outputs") -> str:
    today = _today_utc_str()
    job_id = uuid.uuid4().hex
    folder = os.path.join(base_outputs_dir, today, job_id)
    os.makedirs(folder, exist_ok=True)

    # marker file
    try:
        with open(os.path.join(folder, "job_type.txt"), "w", encoding="utf-8") as f:
            f.write("cad_from_image")
    except Exception:
        pass

    return folder


def _parse_job_path(job_folder: str) -> Tuple[Optional[str], Optional[str]]:
    parts = os.path.normpath(job_folder).split(os.sep)
    if len(parts) < 3:
        return None, None
    return parts[-2], parts[-1]


def _outputs_public_urls(job_folder: str) -> Dict[str, Optional[str]]:
    date_str, job_id = _parse_job_path(job_folder)
    if not date_str or not job_id:
        return {"dxf_url": None, "dwg_url": None, "preview_url": None, "meta_url": None}

    base = f"/outputs/{date_str}/{job_id}"
    return {
        "dxf_url": f"{base}/output.dxf",
        "dwg_url": f"{base}/output.dwg",
        "preview_url": f"{base}/lines_preview.png",
        "meta_url": f"{base}/meta.json",
    }


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
    """
    Dispatch to GPU handler (HMAC signed, Option A).
    IMPORTANT: we sign exact raw bytes we send.
    """
    url = _gpu_dispatch_url()
    payload = {
        "job_type": "cad_from_image",
        "job_folder": _abs(job_folder_rel),
        "meta": meta,
        "pipeline_key": "cad::from_image",
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


def _validate_scale_two_point(distance: Optional[float], p1: Optional[Point], p2: Optional[Point]) -> None:
    if distance is None or distance <= 0:
        raise HTTPException(
            status_code=400,
            detail="scale_reference_distance must be provided and > 0 when scale_mode=two_point",
        )
    if p1 is None or p2 is None:
        raise HTTPException(
            status_code=400,
            detail="scale_point_1 and scale_point_2 are required when scale_mode=two_point",
        )


def _decode_base64_image_png_jpg_only(image_base64: str) -> Tuple[bytes, str]:
    """
    Returns: (image_bytes, ext) where ext in {"png","jpg"}
    Enforces PNG/JPG only based on magic bytes, NOT on claimed MIME.
    """
    s = (image_base64 or "").strip()
    if not s:
        raise HTTPException(status_code=400, detail="image_base64 is required")

    # Accept data URL
    if s.startswith("data:"):
        try:
            _, b64 = s.split(",", 1)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid data URL format for image_base64")
        s = b64.strip()

    # Fix spaces (common with form encoders)
    s = s.replace(" ", "+")

    try:
        raw = base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Invalid base64 image payload")

    # PNG magic
    if len(raw) >= 8 and raw[:8] == b"\x89PNG\r\n\x1a\n":
        return raw, "png"

    # JPEG magic
    if len(raw) >= 3 and raw[:3] == b"\xff\xd8\xff":
        return raw, "jpg"

    raise HTTPException(status_code=400, detail="Only PNG and JPG are allowed")


def _find_job_folder_by_job_id(job_id: str) -> Optional[str]:
    """
    Scan outputs/*/<job_id> and return first match.
    """
    outputs_dir = "outputs"
    if not os.path.isdir(outputs_dir):
        return None

    for date_dir in os.listdir(outputs_dir):
        date_path = os.path.join(outputs_dir, date_dir)
        if not os.path.isdir(date_path):
            continue
        candidate = os.path.join(date_path, job_id)
        if os.path.isdir(candidate):
            return candidate
    return None


def _write_meta(job_folder: str, meta: Dict[str, Any]) -> None:
    meta_path = os.path.join(job_folder, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)


def _safe_relpath(rel: str) -> str:
    """
    Prevent path traversal. Return a safe relative path (no ../).
    """
    rel = (rel or "").strip().replace("\\", "/")
    rel = rel.lstrip("/")
    norm = os.path.normpath(rel).replace("\\", "/")
    if not norm or norm == ".":
        return "output.png"
    if norm.startswith("../") or norm == ".." or "/../" in f"/{norm}/":
        raise HTTPException(status_code=400, detail="Invalid source_image_path.")
    return norm


def _normalize_scale_unit(unit: str) -> ScaleUnit:
    u = (unit or "m").strip().lower()
    if u not in ("m", "cm", "mm", "ft"):
        raise HTTPException(status_code=400, detail="scale_reference_unit must be m|cm|mm|ft")
    return u  # type: ignore[return-value]


def _normalize_semantic_level(level: str) -> SemanticLevel:
    s = (level or "architectural").strip().lower()
    if s not in ("basic", "architectural"):
        raise HTTPException(status_code=400, detail="semantic_level must be basic|architectural")
    return s  # type: ignore[return-value]


def _ensure_png_jpg_upload(image: UploadFile) -> None:
    """
    Best-effort enforcement: PNG/JPG only.
    (We trust GPU worker can do deeper validation if desired.)
    """
    ct = (getattr(image, "content_type", "") or "").lower().strip()
    if ct and ("png" not in ct and "jpeg" not in ct and "jpg" not in ct):
        raise HTTPException(status_code=400, detail="Only PNG and JPG are allowed")


# ---------------------------------------------------------------------------
# ROUTE 1: multipart upload (BEST default for Wix)
# ---------------------------------------------------------------------------

@router.post("/from-image")
async def cad_from_image(
    image: UploadFile = File(..., description="PNG/JPG render/image to extract CAD linework from"),

    # Scaling
    scale_mode: str = Form("two_point", description="two_point | door_height_fallback"),
    scale_reference_distance: Optional[float] = Form(None, description="Real distance between points (in unit)"),
    scale_reference_unit: str = Form("m", description="m|cm|mm|ft"),

    scale_point_1_x: Optional[float] = Form(None, description="0..1 normalized"),
    scale_point_1_y: Optional[float] = Form(None, description="0..1 normalized"),
    scale_point_2_x: Optional[float] = Form(None, description="0..1 normalized"),
    scale_point_2_y: Optional[float] = Form(None, description="0..1 normalized"),

    fallback_door_height: float = Form(2.10, description="Door height used if fallback triggered"),

    semantic_level: str = Form("architectural", description="basic|architectural"),
    layer_profile: str = Form("standard", description="layer profile name"),
):
    _ensure_png_jpg_upload(image)

    scale_mode_clean = (scale_mode or "").strip() or "two_point"
    if scale_mode_clean not in ("two_point", "door_height_fallback"):
        raise HTTPException(status_code=400, detail="scale_mode must be 'two_point' or 'door_height_fallback'")

    unit_clean = _normalize_scale_unit(scale_reference_unit)
    semantic_clean = _normalize_semantic_level(semantic_level)

    if float(fallback_door_height) <= 0.5 or float(fallback_door_height) >= 10.0:
        raise HTTPException(status_code=400, detail="fallback_door_height must be between 0.5 and 10.0")

    p1: Optional[Point] = None
    p2: Optional[Point] = None
    if scale_mode_clean == "two_point":
        pts = [scale_point_1_x, scale_point_1_y, scale_point_2_x, scale_point_2_y]
        if any(v is None for v in pts):
            raise HTTPException(status_code=400, detail="All scale_point_* must be provided when scale_mode=two_point")
        for v in pts:
            if v is None or v < 0.0 or v > 1.0:
                raise HTTPException(status_code=400, detail="Scale points must be normalized 0..1")
        p1 = Point(x=float(scale_point_1_x), y=float(scale_point_1_y))
        p2 = Point(x=float(scale_point_2_x), y=float(scale_point_2_y))
        _validate_scale_two_point(scale_reference_distance, p1, p2)

    # 1) Create job folder
    job_folder = _create_job_folder()
    job_id = os.path.basename(job_folder)

    # 2) Save input image (GPU expects input.png)
    img_path = os.path.join(job_folder, "input.png")
    await _save_upload_stream(image, img_path)

    # 3) Meta for GPU
    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "type": "cad_from_image",
        "status": "queued",
        "mode_runtime": "gpu-dispatch",

        "inputs": {
            "image": "input.png",
            "content_type": getattr(image, "content_type", None),
            "filename": getattr(image, "filename", None),
            "source": "upload",
        },

        "scaling": {
            "mode": scale_mode_clean,
            "distance": float(scale_reference_distance) if scale_reference_distance is not None else None,
            "unit": unit_clean,
            "points": (
                [{"x": p1.x, "y": p1.y}, {"x": p2.x, "y": p2.y}]
                if scale_mode_clean == "two_point" and p1 and p2 else None
            ),
            "fallback_door_height": float(fallback_door_height),
            "no_dimensions": True,
        },

        "semantics": {
            "level": semantic_clean,
            "layer_profile": layer_profile,
        },

        "outputs": {
            "preview": "lines_preview.png",
            "dxf": "output.dxf",
            "dwg": "output.dwg",
            "meta": "meta.json",
        },

        "pipeline_key": "cad::from_image",

        "dispatch": {
            "job_type": "cad_from_image",
            "target": _gpu_dispatch_url(),
            "dispatched_at": None,
            "gpu_response": None,
            "error": None,
        },
    }

    meta_path = os.path.join(job_folder, "meta.json")
    try:
        _write_meta(job_folder, meta)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to write meta.json: {exc}") from exc

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
            "message": "CAD-from-image job dispatched to GPU.",
            "job_folder": job_folder,
            "meta_path": meta_path,
            "public_urls": public_urls,
            "expected_outputs": {
                "dxf": os.path.join(job_folder, "output.dxf"),
                "dwg": os.path.join(job_folder, "output.dwg"),
                "preview": os.path.join(job_folder, "lines_preview.png"),
            },
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
            "job_folder": job_folder,
            "meta_path": meta_path,
            "public_urls": public_urls,
            "gpu_error": {"detail": str(exc), "dispatch_url": _gpu_dispatch_url()},
        }


# ---------------------------------------------------------------------------
# ROUTE 2: JSON upload (backup if Wix is JSON-only)
# ---------------------------------------------------------------------------

@router.post("/from-image-base64")
async def cad_from_image_base64(req: CadFromImageBase64Request):
    scale_mode_clean = (req.scale_mode or "two_point").strip()
    if scale_mode_clean not in ("two_point", "door_height_fallback"):
        raise HTTPException(status_code=400, detail="scale_mode must be 'two_point' or 'door_height_fallback'")

    if scale_mode_clean == "two_point":
        _validate_scale_two_point(req.scale_reference_distance, req.scale_point_1, req.scale_point_2)

    # validate unit + semantic
    _ = _normalize_scale_unit(req.scale_reference_unit)
    _ = _normalize_semantic_level(req.semantic_level)

    img_bytes, ext = _decode_base64_image_png_jpg_only(req.image_base64)

    job_folder = _create_job_folder()
    job_id = os.path.basename(job_folder)

    img_path = os.path.join(job_folder, "input.png")
    try:
        with open(img_path, "wb") as f:
            f.write(img_bytes)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed writing input.png: {exc}") from exc

    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "type": "cad_from_image",
        "status": "queued",
        "mode_runtime": "gpu-dispatch",

        "inputs": {
            "image": "input.png",
            "content_type": f"image/{'jpeg' if ext == 'jpg' else 'png'}",
            "source": "base64",
        },

        "scaling": {
            "mode": scale_mode_clean,
            "distance": float(req.scale_reference_distance) if req.scale_reference_distance is not None else None,
            "unit": _normalize_scale_unit(req.scale_reference_unit),
            "points": (
                [{"x": req.scale_point_1.x, "y": req.scale_point_1.y},
                 {"x": req.scale_point_2.x, "y": req.scale_point_2.y}]
                if scale_mode_clean == "two_point" and req.scale_point_1 and req.scale_point_2 else None
            ),
            "fallback_door_height": float(req.fallback_door_height),
            "no_dimensions": True,
        },

        "semantics": {
            "level": _normalize_semantic_level(req.semantic_level),
            "layer_profile": req.layer_profile,
        },

        "outputs": {
            "preview": "lines_preview.png",
            "dxf": "output.dxf",
            "dwg": "output.dwg",
            "meta": "meta.json",
        },

        "pipeline_key": "cad::from_image",

        "dispatch": {
            "job_type": "cad_from_image",
            "target": _gpu_dispatch_url(),
            "dispatched_at": None,
            "gpu_response": None,
            "error": None,
        },
    }

    meta_path = os.path.join(job_folder, "meta.json")
    try:
        _write_meta(job_folder, meta)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to write meta.json: {exc}") from exc

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
            "message": "CAD-from-image (JSON) job dispatched to GPU.",
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
            "job_folder": job_folder,
            "meta_path": meta_path,
            "public_urls": public_urls,
            "gpu_error": {"detail": str(exc), "dispatch_url": _gpu_dispatch_url()},
        }


# ---------------------------------------------------------------------------
# ROUTE 3: from-job (JSON)
# ---------------------------------------------------------------------------

@router.post("/from-job")
async def cad_from_job(req: CadFromJobRequest):
    source_job_id = (req.source_job_id or "").strip()
    if not source_job_id:
        raise HTTPException(status_code=400, detail="source_job_id is required")

    found_folder = _find_job_folder_by_job_id(source_job_id)
    if not found_folder:
        raise HTTPException(status_code=404, detail=f"source job folder not found for job_id={source_job_id}")

    safe_rel = _safe_relpath(req.source_image_path)
    src_img = os.path.join(found_folder, safe_rel)
    if not os.path.isfile(src_img):
        raise HTTPException(status_code=404, detail=f"source image not found: {safe_rel}")

    if req.scale_mode not in ("two_point", "door_height_fallback"):
        raise HTTPException(status_code=400, detail="scale_mode must be 'two_point' or 'door_height_fallback'")
    if req.scale_mode == "two_point":
        _validate_scale_two_point(req.scale_reference_distance, req.scale_point_1, req.scale_point_2)

    # Create CAD job
    job_folder = _create_job_folder()
    job_id = os.path.basename(job_folder)

    # Copy source image into job folder as input.png
    dst_img = os.path.join(job_folder, "input.png")
    try:
        shutil.copyfile(src_img, dst_img)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed copying source image: {exc}")

    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "type": "cad_from_image",
        "status": "queued",
        "mode_runtime": "gpu-dispatch",

        "inputs": {
            "image": "input.png",
            "source": "job_reference",
            "source_job_id": source_job_id,
            "source_image_path": safe_rel,
        },

        "scaling": {
            "mode": req.scale_mode,
            "distance": float(req.scale_reference_distance) if req.scale_reference_distance is not None else None,
            "unit": _normalize_scale_unit(req.scale_reference_unit),
            "points": (
                [{"x": req.scale_point_1.x, "y": req.scale_point_1.y},
                 {"x": req.scale_point_2.x, "y": req.scale_point_2.y}]
                if req.scale_mode == "two_point" and req.scale_point_1 and req.scale_point_2 else None
            ),
            "fallback_door_height": float(req.fallback_door_height),
            "no_dimensions": True,
        },

        "semantics": {
            "level": _normalize_semantic_level(req.semantic_level),
            "layer_profile": req.layer_profile,
        },

        "outputs": {
            "preview": "lines_preview.png",
            "dxf": "output.dxf",
            "dwg": "output.dwg",
            "meta": "meta.json",
        },

        "pipeline_key": "cad::from_image",

        "dispatch": {
            "job_type": "cad_from_image",
            "target": _gpu_dispatch_url(),
            "dispatched_at": None,
            "gpu_response": None,
            "error": None,
        },
    }

    meta_path = os.path.join(job_folder, "meta.json")
    try:
        _write_meta(job_folder, meta)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write meta.json: {exc}")

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
            "message": "CAD-from-job job dispatched to GPU.",
            "job_folder": job_folder,
            "meta_path": meta_path,
            "public_urls": public_urls,
            "gpu_response": gpu_resp,
        }

    except Exception as exc:
        try:
            meta["status"] = "gpu_error"
            meta["dispatch"]["error"] = {"detail": str(exc)}
            _write_meta(job_folder, meta)
        except Exception:
            pass

        return {
            "status": "gpu_error",
            "message": "Job created but GPU dispatch failed.",
            "job_folder": job_folder,
            "meta_path": meta_path,
            "public_urls": public_urls,
            "gpu_error": {"detail": str(exc), "dispatch_url": _gpu_dispatch_url()},
        }


# ---------------------------------------------------------------------------
# ROUTE 4: Wix naming alias (same behavior as /from-job)
# ---------------------------------------------------------------------------

@router.post("/from-job-base64")
async def cad_from_job_base64(req: CadFromJobBase64Request):
    r = CadFromJobRequest(
        source_job_id=req.source_job_id,
        source_image_path=req.source_image_path,
        scale_mode=req.scale_mode,
        scale_reference_distance=req.scale_reference_distance,
        scale_reference_unit=req.scale_reference_unit,
        scale_point_1=req.scale_point_1,
        scale_point_2=req.scale_point_2,
        fallback_door_height=req.fallback_door_height,
        semantic_level=req.semantic_level,
        layer_profile=req.layer_profile,
    )
    return await cad_from_job(r)
