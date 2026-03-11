"""
SD3.5 endpoints for RENDEREXPO AI STUDIO (Planner).

CRITICAL:
- Planner creates job folder + meta.json with LOCKED presets (apply_preset_to_meta)
- Planner DISPATCHES to GPU worker (/api/gpu/dispatch) using HMAC Option A
- GPU worker performs REAL SD3.5 execution and writes outputs into the job folder

Service split (LOCKED):
- Planner = port 8012
- GPU worker = port 8002

Wix-friendly:
- /api/sd35/render-form        (prompt-only via multipart/form-data)
- /api/sd35/render-from-image  (prompt+image via multipart/form-data)

JSON-friendly:
- /api/sd35/render             (application/json)

Env:
- GPU_WORKER_URL default http://127.0.0.1:8002/api/gpu/dispatch
- RENDEREXPO_HMAC_SECRET required (must match GPU worker)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Literal, Optional, Tuple

import requests
from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile

from app.presets_sd35 import apply_preset_to_meta

router = APIRouter(prefix="/api/sd35", tags=["sd35"])

Category = Literal["urban", "suburban", "interior", "wide_hero"]
Shot = Literal["wide", "close"]

# HMAC (must match app/main.py and GPU worker auth)
HMAC_SECRET_ENV = "RENDEREXPO_HMAC_SECRET"
SIG_HEADER = "X-RENDEREXPO-SIGNATURE"
TS_HEADER = "X-RENDEREXPO-TIMESTAMP"
NONCE_HEADER = "X-RENDEREXPO-NONCE"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_epoch() -> int:
    return int(time.time())


def _repo_root() -> str:
    return "/workspace-data/RENDEREXPO-AI-Studio-Backend"


def _abs(rel_path: str) -> str:
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.abspath(os.path.join(_repo_root(), rel_path))


def _gpu_worker_url() -> str:
    return os.getenv("GPU_WORKER_URL", "http://127.0.0.1:8002/api/gpu/dispatch").strip()


def _today_utc_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def _create_job_folder(job_type: str) -> str:
    """
    Create outputs/YYYY-MM-DD/JOBID/
    Returns RELATIVE path rooted at the planner repo.
    """
    today = _today_utc_str()
    job_id = uuid.uuid4().hex
    base_dir = os.path.join("outputs", today, job_id)
    os.makedirs(base_dir, exist_ok=True)

    try:
        with open(os.path.join(base_dir, "job_type.txt"), "w", encoding="utf-8") as f:
            f.write(job_type)
    except Exception:
        pass

    return base_dir


def _parse_job_path(job_folder: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Expect: outputs/YYYY-MM-DD/JOBID
    """
    parts = os.path.normpath(job_folder).split(os.sep)
    if len(parts) < 3:
        return None, None
    return parts[-2], parts[-1]


def _outputs_public_urls(job_folder: str) -> Dict[str, Optional[str]]:
    """
    Stable URLs assuming planner FastAPI mounts outputs/ at /outputs.
    """
    date_str, job_id = _parse_job_path(job_folder)
    if not date_str or not job_id:
        return {
            "output_image_url": None,
            "meta_url": None,
            "input_url": None,
        }
    base = f"/outputs/{date_str}/{job_id}"
    return {
        "output_image_url": f"{base}/output.png",
        "meta_url": f"{base}/meta.json",
        "input_url": f"{base}/input.png",
    }


def _save_meta(job_folder_rel: str, meta: Dict[str, Any]) -> str:
    meta_path = os.path.join(job_folder_rel, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
    return meta_path


def _save_upload_stream(upload: UploadFile, dst_path: str) -> None:
    """
    Save UploadFile without reading everything into RAM.
    """
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


def _dispatch_to_gpu(job_type: str, job_folder_rel: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    HMAC-signed dispatch to GPU worker (Option A).
    Signs the exact raw bytes sent.
    """
    url = _gpu_worker_url()

    payload = {
        "job_folder": _abs(job_folder_rel),
        "meta": meta,
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


def _ensure_image_type(upload: UploadFile) -> None:
    """
    Wix uploads PNG/JPG.
    We accept content_type if present; otherwise rely on filename extension.
    """
    ct = (getattr(upload, "content_type", "") or "").lower().strip()
    if ct and ct not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(status_code=400, detail="Only PNG and JPG are supported.")

    name = (upload.filename or "").lower().strip()
    if name and not (name.endswith(".png") or name.endswith(".jpg") or name.endswith(".jpeg")):
        raise HTTPException(status_code=400, detail="Only .png, .jpg, .jpeg are supported.")


def _apply_override_to_meta(meta: Dict[str, Any], payload: Dict[str, Any]) -> None:
    """
    STRICT override handling:
    - Accept override object: payload["override"] = {"lycoris_multiplier": x, "geo_multiplier": y}
    - Or accept top-level keys: payload["lycoris_multiplier"], payload["geo_multiplier"]
    - Applies even if value == 0.0
    - Records raw + resolved values for audit.
    """
    override = payload.get("override")
    ly = None
    ge = None

    if isinstance(override, dict):
        ly = override.get("lycoris_multiplier")
        ge = override.get("geo_multiplier")

    if ly is None:
        ly = payload.get("lycoris_multiplier")
    if ge is None:
        ge = payload.get("geo_multiplier")

    meta["override_received_raw"] = override
    meta["override_resolved"] = {
        "lycoris_multiplier": ly,
        "geo_multiplier": ge,
    }

    applied_any = False

    if ly is not None:
        ly_f = float(ly)
        if isinstance(meta.get("lora_config"), dict):
            meta["lora_config"]["strength"] = ly_f
            meta["lora_config"]["scale"] = ly_f
            applied_any = True
        if isinstance(meta.get("preset"), dict):
            meta["preset"]["lycoris_multiplier"] = ly_f

    if ge is not None:
        ge_f = float(ge)
        if isinstance(meta.get("geo_config"), dict):
            meta["geo_config"]["strength"] = ge_f
            meta["geo_config"]["scale"] = ge_f
            applied_any = True
        if isinstance(meta.get("preset"), dict):
            meta["preset"]["geo_multiplier"] = ge_f

    meta["override_applied"] = {
        "lycoris_multiplier": float(ly) if ly is not None else None,
        "geo_multiplier": float(ge) if ge is not None else None,
        "applied": bool(applied_any),
    }


def _validate_category_shot(category: str, shot: str) -> None:
    if category not in ("urban", "suburban", "interior", "wide_hero"):
        raise HTTPException(status_code=400, detail="Invalid category")
    if shot not in ("wide", "close"):
        raise HTTPException(status_code=400, detail="Invalid shot")


def _base_meta(
    *,
    job_id: str,
    job_type: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    category: str,
    shot: str,
    input_image: Optional[str] = None,
    strength: Optional[float] = None,
    style_preset: Optional[str] = None,
    material_preset: Optional[str] = None,
    lighting_preset: Optional[str] = None,
) -> Dict[str, Any]:
    pipeline_key = f"sd35::{job_type}"

    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": datetime.utcnow().isoformat(),
        "type": "img2img" if job_type == "sd35_img2img" else "text2img",
        "model_name": "sd35_large_pro_v2_1",
        "engine": "sd35_large_pro_v2_1",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": int(seed),
        "style_preset": style_preset,
        "material_preset": material_preset,
        "lighting_preset": lighting_preset,
        "category": category,
        "shot": shot,
        "status": "queued",
        "mode_runtime": "gpu-dispatch",
        "pipeline_key": pipeline_key,
        "outputs": {
            "image": "output.png",
            "meta": "meta.json",
        },
        "dispatch": {
            "job_type": job_type,
            "target": _gpu_worker_url(),
            "dispatched_at": None,
            "gpu_response": None,
            "error": None,
        },
    }

    if input_image is not None:
        meta["input_image"] = input_image
        meta["outputs"]["input"] = input_image

    if strength is not None:
        meta["strength"] = float(strength)

    return meta


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/render")
async def sd35_render_json(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Text2Img (application/json):
    - Create job folder
    - Write meta.json WITH locked presets
    - Optional override for LyCORIS/GEO multipliers (A/B testing)
    - Dispatch to GPU worker -> REAL SD3.5
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON payload must be an object/dict")

    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")

    negative_prompt = str(payload.get("negative_prompt") or "").strip()

    category = payload.get("category")
    shot = payload.get("shot")
    _validate_category_shot(category, shot)

    upscale_2x = payload.get("upscale_2x")
    seed = payload.get("seed")

    style_preset = payload.get("style_preset")
    material_preset = payload.get("material_preset")
    lighting_preset = payload.get("lighting_preset")

    job_folder = _create_job_folder(job_type="sd35_text2img")
    job_id = os.path.basename(job_folder)
    public_urls = _outputs_public_urls(job_folder)

    final_seed = seed if seed is not None else int(uuid.uuid4().int % 1_000_000_000)

    meta = _base_meta(
        job_id=job_id,
        job_type="sd35_text2img",
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=int(final_seed),
        category=category,
        shot=shot,
        style_preset=style_preset,
        material_preset=material_preset,
        lighting_preset=lighting_preset,
    )

    try:
        apply_preset_to_meta(meta, category=category, shot=shot, upscale_2x=upscale_2x)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Preset error: {exc}")

    _apply_override_to_meta(meta, payload)

    meta_path = _save_meta(job_folder, meta)

    try:
        gpu_resp = _dispatch_to_gpu("sd35_text2img", job_folder, meta)

        try:
            meta["status"] = "dispatched"
            meta["dispatch"]["dispatched_at"] = datetime.utcnow().isoformat()
            meta["dispatch"]["gpu_response"] = gpu_resp
            _save_meta(job_folder, meta)
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
            _save_meta(job_folder, meta)
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
                "url": _gpu_worker_url(),
                "job_folder_sent": _abs(job_folder),
            },
        }


@router.post("/render-form")
async def sd35_render_form(
    prompt: str = Form(..., description="Main prompt for SD3.5"),
    negative_prompt: Optional[str] = Form(None),
    category: Category = Form(...),
    shot: Shot = Form(...),
    upscale_2x: Optional[bool] = Form(None),
    seed: Optional[int] = Form(None),
    style_preset: Optional[str] = Form(None),
    material_preset: Optional[str] = Form(None),
    lighting_preset: Optional[str] = Form(None),
    lycoris_multiplier: Optional[float] = Form(None),
    geo_multiplier: Optional[float] = Form(None),
) -> Dict[str, Any]:
    """
    Text2Img (multipart/form-data).
    """
    prompt_clean = (prompt or "").strip()
    if not prompt_clean:
        raise HTTPException(status_code=400, detail="Missing prompt")

    _validate_category_shot(category, shot)

    job_folder = _create_job_folder(job_type="sd35_text2img")
    job_id = os.path.basename(job_folder)
    public_urls = _outputs_public_urls(job_folder)

    final_seed = seed if seed is not None else int(uuid.uuid4().int % 1_000_000_000)

    meta = _base_meta(
        job_id=job_id,
        job_type="sd35_text2img",
        prompt=prompt_clean,
        negative_prompt=(negative_prompt or "").strip(),
        seed=int(final_seed),
        category=category,
        shot=shot,
        style_preset=style_preset,
        material_preset=material_preset,
        lighting_preset=lighting_preset,
    )

    try:
        apply_preset_to_meta(meta, category=category, shot=shot, upscale_2x=upscale_2x)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Preset error: {exc}")

    payload_for_override = {
        "override": None,
        "lycoris_multiplier": lycoris_multiplier,
        "geo_multiplier": geo_multiplier,
    }
    _apply_override_to_meta(meta, payload_for_override)

    meta_path = _save_meta(job_folder, meta)

    try:
        gpu_resp = _dispatch_to_gpu("sd35_text2img", job_folder, meta)

        try:
            meta["status"] = "dispatched"
            meta["dispatch"]["dispatched_at"] = datetime.utcnow().isoformat()
            meta["dispatch"]["gpu_response"] = gpu_resp
            _save_meta(job_folder, meta)
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
            _save_meta(job_folder, meta)
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
                "url": _gpu_worker_url(),
                "job_folder_sent": _abs(job_folder),
            },
        }


@router.post("/render-from-image")
async def sd35_render_from_image(
    image: UploadFile = File(..., description="Base image / sketch / clay render (PNG/JPG)"),
    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form(None),
    strength: float = Form(0.22, ge=0.0, le=1.0),
    category: Category = Form(...),
    shot: Shot = Form(...),
    upscale_2x: Optional[bool] = Form(None),
    seed: Optional[int] = Form(None),
    lycoris_multiplier: Optional[float] = Form(None),
    geo_multiplier: Optional[float] = Form(None),
) -> Dict[str, Any]:
    """
    Img2Img (multipart/form-data):
    - Save input.png
    - Write meta.json WITH locked presets + strength
    - Optional override for LyCORIS/GEO multipliers
    - Dispatch to GPU worker -> REAL SD3.5 Img2Img
    """
    if not image.filename:
        raise HTTPException(status_code=400, detail="Uploaded image has no filename.")
    _ensure_image_type(image)

    prompt_clean = (prompt or "").strip()
    if not prompt_clean:
        raise HTTPException(status_code=400, detail="Missing prompt")

    _validate_category_shot(category, shot)

    job_folder = _create_job_folder(job_type="sd35_img2img")
    job_id = os.path.basename(job_folder)
    public_urls = _outputs_public_urls(job_folder)

    input_path = os.path.join(job_folder, "input.png")
    _save_upload_stream(image, input_path)

    final_seed = seed if seed is not None else int(uuid.uuid4().int % 1_000_000_000)

    meta = _base_meta(
        job_id=job_id,
        job_type="sd35_img2img",
        prompt=prompt_clean,
        negative_prompt=(negative_prompt or "").strip(),
        seed=int(final_seed),
        category=category,
        shot=shot,
        input_image="input.png",
        strength=float(strength),
    )

    try:
        apply_preset_to_meta(meta, category=category, shot=shot, upscale_2x=upscale_2x)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Preset error: {exc}")

    # Preserve img2img strength; preset layer must not zero it out.
    meta["strength"] = float(strength)

    payload_for_override = {
        "override": None,
        "lycoris_multiplier": lycoris_multiplier,
        "geo_multiplier": geo_multiplier,
    }
    _apply_override_to_meta(meta, payload_for_override)

    meta_path = _save_meta(job_folder, meta)

    try:
        gpu_resp = _dispatch_to_gpu("sd35_img2img", job_folder, meta)

        try:
            meta["status"] = "dispatched"
            meta["dispatch"]["dispatched_at"] = datetime.utcnow().isoformat()
            meta["dispatch"]["gpu_response"] = gpu_resp
            _save_meta(job_folder, meta)
        except Exception:
            pass

        return {
            "status": "dispatched",
            "job_id": job_id,
            "job_folder": job_folder,
            "input_saved_as": input_path,
            "meta_path": meta_path,
            "public_urls": public_urls,
            "gpu_response": gpu_resp,
        }
    except Exception as exc:  # noqa: BLE001
        try:
            meta["status"] = "gpu_error"
            meta["dispatch"]["error"] = {"detail": str(exc)}
            _save_meta(job_folder, meta)
        except Exception:
            pass

        return {
            "status": "gpu_error",
            "job_id": job_id,
            "job_folder": job_folder,
            "input_saved_as": input_path,
            "meta_path": meta_path,
            "public_urls": public_urls,
            "gpu_error": {
                "detail": str(exc),
                "url": _gpu_worker_url(),
                "job_folder_sent": _abs(job_folder),
            },
        }