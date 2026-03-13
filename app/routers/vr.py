# app/routers/vr.py
"""
RENDEREXPO AI STUDIO - VR Reconstruction Router (Planner)

GOAL:
- User uploads 3+ images of the same space.
- Planner creates a job folder, saves images, writes meta.json.
- Planner dispatches to GPU worker so this is REAL, not skeleton.

3 VR modes -> MUST route to 3 DIFFERENT REAL pipelines on GPU:
- gaussian_splat
- nerf
- mesh

CRITICAL:
- If ANY VR pipeline step uses SD3.5 (texture fill, relight, harmonize, inpaint),
  it MUST use the SAME locked preset system.
- This router writes an sd35_preset block in meta using apply_preset_to_meta().
- Planner = port 8012
- GPU worker = port 8002
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import os
import shutil
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple

import requests
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.presets_sd35 import apply_preset_to_meta

router = APIRouter(prefix="/api/vr", tags=["VR (Planner)"])

Category = Literal["urban", "suburban", "interior", "wide_hero"]
Shot = Literal["wide", "close"]
VRMode = Literal["gaussian_splat", "nerf", "mesh"]

# ---------------------------------------------------------------------------
# HMAC (must match app/main.py / GPU worker)
# ---------------------------------------------------------------------------

HMAC_SECRET_ENV = "RENDEREXPO_HMAC_SECRET"
SIG_HEADER = "X-RENDEREXPO-SIGNATURE"
TS_HEADER = "X-RENDEREXPO-TIMESTAMP"
NONCE_HEADER = "X-RENDEREXPO-NONCE"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_epoch() -> int:
    return int(time.time())


def _gpu_dispatch_url() -> str:
    """
    GPU dispatch endpoint.

    Recommended:
      VR_GPU_DISPATCH_URL=http://127.0.0.1:8002/api/gpu/dispatch
    """
    return os.getenv("VR_GPU_DISPATCH_URL", "http://127.0.0.1:8002/api/gpu/dispatch").strip()


def _repo_root() -> str:
    return "/workspace-data/RENDEREXPO-AI-Studio-Backend"


def _abs(p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(_repo_root(), p))


def _today_utc_str() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


def _create_job_folder(base_outputs_dir: str = "outputs") -> str:
    """Create outputs/{YYYY-MM-DD}/{job_id}/ and return its relative path."""
    today = _today_utc_str()
    job_id = uuid.uuid4().hex
    folder = os.path.join(base_outputs_dir, today, job_id)
    os.makedirs(folder, exist_ok=True)

    try:
        with open(os.path.join(folder, "job_type.txt"), "w", encoding="utf-8") as f:
            f.write("vr_reconstruct")
    except Exception:
        pass

    return folder


def _safe_filename(name: str) -> str:
    """Minimal filename sanitizer for uploads (stored only in meta)."""
    name = (name or "").strip().replace("\\", "_").replace("/", "_")
    if not name:
        return "upload"
    return name[:200]


def _parse_job_path(job_folder: str) -> Tuple[Optional[str], Optional[str]]:
    parts = os.path.normpath(job_folder).split(os.sep)
    if len(parts) < 3:
        return None, None
    date_str, job_id = parts[-2], parts[-1]
    if not date_str or not job_id:
        return None, None
    return date_str, job_id


def _outputs_public_urls(job_folder: str, preview_video: bool) -> Dict[str, Optional[str]]:
    date_str, job_id = _parse_job_path(job_folder)
    if not date_str or not job_id:
        return {"viewer_url": None, "preview_video_url": None}

    base = f"/outputs/{date_str}/{job_id}"
    return {
        "viewer_url": f"{base}/viewer/index.html",
        "preview_video_url": f"{base}/preview.mp4" if preview_video else None,
    }


def _compute_signature(secret: str, timestamp: str, nonce: str, body: bytes) -> str:
    prefix = f"{timestamp}\n{nonce}\n".encode("utf-8")
    msg = prefix + (body or b"")
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def _dispatch_to_gpu(job_folder_rel: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    HMAC-signed dispatch to GPU worker (Option A).
    IMPORTANT: sign the exact raw bytes we send.

    GPU dispatch contract requires:
    - job_type
    - job_folder
    - meta
    - optional pipeline_key / vr_mode
    """
    url = _gpu_dispatch_url()
    vr_mode = meta.get("vr_mode")
    pipeline_key = meta.get("pipeline_key") or (f"vr::{vr_mode}" if vr_mode else None)

    payload = {
        "job_type": "vr_reconstruct",
        "job_folder": _abs(job_folder_rel),
        "meta": meta,
        "pipeline_key": pipeline_key,
        "vr_mode": vr_mode,
    }

    body_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    secret = (os.getenv(HMAC_SECRET_ENV) or "").strip()
    if not secret or len(secret) < 32:
        raise RuntimeError(
            f"Missing/weak {HMAC_SECRET_ENV}. "
            "Set the same strong secret on BOTH planner and GPU worker."
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
        r = requests.post(url, data=body_bytes, headers=headers, timeout=(10, 1800))
        if not (200 <= r.status_code < 300):
            raise RuntimeError(f"GPU dispatch HTTP {r.status_code}: {r.text[:2000]}")
        return r.json() if r.content else {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"GPU dispatch failed: {exc}") from exc


async def _save_upload_stream(upload: UploadFile, dst_path: str) -> None:
    """Save UploadFile without reading everything into RAM."""
    try:
        try:
            upload.file.seek(0)
        except Exception:
            pass

        with open(dst_path, "wb") as out:
            shutil.copyfileobj(upload.file, out)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to save upload '{upload.filename}': {exc}") from exc


def _validate_vr_mode(vr_mode: str) -> VRMode:
    if vr_mode not in ("gaussian_splat", "nerf", "mesh"):
        raise HTTPException(status_code=400, detail="vr_mode must be gaussian_splat | nerf | mesh")
    return vr_mode  # type: ignore[return-value]


def _detect_ext(upload: UploadFile) -> str:
    """
    Wix uploads PNG/JPG. We keep that extension on disk.
    """
    ct = (getattr(upload, "content_type", "") or "").lower().strip()
    name = (upload.filename or "").lower().strip()

    if ct in ("image/jpeg", "image/jpg") or name.endswith((".jpg", ".jpeg")):
        return ".jpg"
    if ct == "image/png" or name.endswith(".png"):
        return ".png"

    raise HTTPException(status_code=400, detail="Only PNG and JPG are supported for VR images.")


def _validate_upload_is_png_jpg(upload: UploadFile) -> None:
    ct = (getattr(upload, "content_type", "") or "").lower().strip()
    name = (upload.filename or "").lower().strip()

    ok_ct = (not ct) or (ct in ("image/png", "image/jpeg", "image/jpg"))
    ok_ext = (not name) or name.endswith((".png", ".jpg", ".jpeg"))
    if not (ok_ct and ok_ext):
        raise HTTPException(status_code=400, detail="Only PNG and JPG are supported for VR images.")


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

async def _build_vr_job(
    images: List[UploadFile],
    prompt: Optional[str],
    plan_hint: Optional[str],
    vr_mode: VRMode,
    category: Optional[Category],
    shot: Optional[Shot],
    upscale_2x: Optional[bool],
    seed: Optional[int],
    max_resolution: int,
    preview_video: bool,
) -> Dict[str, Any]:
    if not images or len(images) < 3:
        raise HTTPException(status_code=400, detail="You must upload at least 3 images.")

    if max_resolution < 512 or max_resolution > 4096:
        raise HTTPException(status_code=400, detail="max_resolution must be between 512 and 4096.")

    for u in images:
        _validate_upload_is_png_jpg(u)

    use_category: Category = category or "interior"
    use_shot: Shot = shot or "wide"
    use_seed: int = seed if seed is not None else int(uuid.uuid4().int % 1_000_000_000)

    # 1) Create job folder
    job_folder = _create_job_folder()
    job_id = os.path.basename(job_folder)

    # 2) Save images (preserve png/jpg extension)
    saved_views: List[Dict[str, Any]] = []
    for idx, upload in enumerate(images, start=1):
        ext = _detect_ext(upload)
        view_name = f"view_{idx:03d}{ext}"
        view_path = os.path.join(job_folder, view_name)

        await _save_upload_stream(upload, view_path)

        saved_views.append(
            {
                "file": view_name,
                "original_filename": _safe_filename(upload.filename or view_name),
                "content_type": getattr(upload, "content_type", None),
            }
        )

    # 3) SD3.5 preset block for any downstream SD3.5-assisted stage
    sd35_preset: Dict[str, Any] = {
        "type": "sd35_preset_block",
        "model_name": "sd35_large_pro_v2_1",
        "engine": "sd35_large_pro_v2_1",
        "category": use_category,
        "shot": use_shot,
        "seed": use_seed,
    }

    apply_preset_to_meta(sd35_preset, category=use_category, shot=use_shot, upscale_2x=upscale_2x)

    # 4) Outputs contract (mode-specific)
    outputs: Dict[str, Any] = {
        "viewer_dir": "viewer/",
        "preview_video": "preview.mp4" if preview_video else None,
        "artifact": (
            "scene.splat" if vr_mode == "gaussian_splat"
            else "nerf_model/" if vr_mode == "nerf"
            else "scene_mesh.glb"
        ),
        "gaussian_splat": "scene.splat" if vr_mode == "gaussian_splat" else None,
        "nerf_dir": "nerf_model/" if vr_mode == "nerf" else None,
        "mesh_glb": "scene_mesh.glb" if vr_mode == "mesh" else None,
    }

    # 5) Meta
    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "type": "vr_reconstruct",
        "status": "queued",
        "mode_runtime": "gpu-dispatch",
        "vr_mode": vr_mode,
        "prompt": prompt,
        "plan_hint": plan_hint,
        "input_views": saved_views,
        "vr_runtime": {
            "max_resolution": int(max_resolution),
            "preview_video": bool(preview_video),
        },
        "sd35_preset": sd35_preset,
        "outputs": outputs,
        "pipeline_key": f"vr::{vr_mode}",
        "dispatch": {
            "target": _gpu_dispatch_url(),
            "job_type": "vr_reconstruct",
            "pipeline_key": f"vr::{vr_mode}",
            "vr_mode": vr_mode,
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

    return {
        "job_folder": job_folder,
        "meta_path": meta_path,
        "meta": meta,
        "views_saved": [v["file"] for v in saved_views],
        "sd35_category_used": use_category,
        "sd35_shot_used": use_shot,
        "sd35_preset_applied": sd35_preset.get("preset", {}),
        "sd35_upscale_enabled": bool(sd35_preset.get("upscale", {}).get("enabled", False)),
    }


# ---------------------------------------------------------------------------
# REAL Route (GPU dispatch)
# ---------------------------------------------------------------------------

@router.post("/reconstruct")
async def reconstruct_vr(
    images: List[UploadFile] = File(..., description="At least 3 images of the same space."),
    vr_mode: str = Form("gaussian_splat", description="gaussian_splat | nerf | mesh"),
    prompt: Optional[str] = Form(None),
    plan_hint: Optional[str] = Form(None),
    category: Optional[Category] = Form(None),
    shot: Optional[Shot] = Form(None),
    upscale_2x: Optional[bool] = Form(None),
    seed: Optional[int] = Form(None),
    max_resolution: int = Form(1600),
    preview_video: bool = Form(True),
) -> Dict[str, Any]:
    vr_mode_typed: VRMode = _validate_vr_mode(vr_mode)

    built = await _build_vr_job(
        images=images,
        prompt=prompt,
        plan_hint=plan_hint,
        vr_mode=vr_mode_typed,
        category=category,
        shot=shot,
        upscale_2x=upscale_2x,
        seed=seed,
        max_resolution=max_resolution,
        preview_video=preview_video,
    )

    job_folder = built["job_folder"]
    meta = built["meta"]
    public_urls = _outputs_public_urls(job_folder, preview_video)

    try:
        gpu_resp = _dispatch_to_gpu(job_folder, meta)

        try:
            meta["status"] = "dispatched"
            meta["dispatch"]["dispatched_at"] = datetime.datetime.utcnow().isoformat()
            meta["dispatch"]["gpu_response"] = gpu_resp
            with open(os.path.join(job_folder, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4)
        except Exception:
            pass

        return {
            "status": "dispatched",
            "message": "VR reconstruction dispatched to GPU.",
            "job_folder": job_folder,
            "meta_path": built["meta_path"],
            "views_saved": built["views_saved"],
            "vr_mode": vr_mode_typed,
            "public_urls": public_urls,
            "expected_outputs": {
                "viewer_dir": os.path.join(job_folder, "viewer"),
                "preview_video": os.path.join(job_folder, "preview.mp4") if preview_video else None,
                "artifact": os.path.join(job_folder, meta["outputs"]["artifact"]),
            },
            "sd35_preset_applied": built["sd35_preset_applied"],
            "sd35_upscale_enabled": built["sd35_upscale_enabled"],
            "sd35_category_used": built["sd35_category_used"],
            "sd35_shot_used": built["sd35_shot_used"],
            "gpu_response": gpu_resp,
        }

    except Exception as exc:  # noqa: BLE001
        try:
            meta["status"] = "gpu_error"
            meta["dispatch"]["error"] = {"detail": str(exc)}
            with open(os.path.join(job_folder, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4)
        except Exception:
            pass

        return {
            "status": "gpu_error",
            "message": "Job created but GPU dispatch failed.",
            "job_folder": job_folder,
            "meta_path": built["meta_path"],
            "views_saved": built["views_saved"],
            "vr_mode": vr_mode_typed,
            "public_urls": public_urls,
            "gpu_error": {
                "detail": str(exc),
                "dispatch_url": _gpu_dispatch_url(),
                "job_folder_sent": _abs(job_folder),
            },
        }


# ---------------------------------------------------------------------------
# Legacy plan route (NO DISPATCH) — disabled by default
# ---------------------------------------------------------------------------

@router.post("/reconstruct/plan")
async def plan_vr_reconstruction(
    images: List[UploadFile] = File(...),
    prompt: Optional[str] = Form(None),
    plan_hint: Optional[str] = Form(None),
    category: Optional[Category] = Form(None),
    shot: Optional[Shot] = Form(None),
    upscale_2x: Optional[bool] = Form(None),
    seed: Optional[int] = Form(None),
    max_resolution: int = Form(1600),
    preview_video: bool = Form(True),
    vr_mode: str = Form("gaussian_splat"),
) -> Dict[str, Any]:
    if (os.getenv("ENABLE_VR_PLAN_ROUTE") or "").strip().lower() not in ("1", "true", "yes"):
        raise HTTPException(status_code=404, detail="Not found.")

    vr_mode_typed: VRMode = _validate_vr_mode(vr_mode)

    built = await _build_vr_job(
        images=images,
        prompt=prompt,
        plan_hint=plan_hint,
        vr_mode=vr_mode_typed,
        category=category,
        shot=shot,
        upscale_2x=upscale_2x,
        seed=seed,
        max_resolution=max_resolution,
        preview_video=preview_video,
    )

    public_urls = _outputs_public_urls(built["job_folder"], preview_video)

    return {
        "status": "planned",
        "message": "VR reconstruction planned (no GPU dispatch). Use /api/vr/reconstruct for REAL runs.",
        "job_folder": built["job_folder"],
        "views_saved": built["views_saved"],
        "meta_path": built["meta_path"],
        "public_urls": public_urls,
        "vr_mode": vr_mode_typed,
    }