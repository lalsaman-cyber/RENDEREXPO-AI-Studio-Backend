# app/routers/sketch.py
"""
app/routers/sketch.py

RENDEREXPO AI STUDIO - Sketch Realtime (REAL execution, session storage)

WHAT THIS DOES (your Wix “two screens” flow):
- Client draws on Wix canvas + types a prompt.
- Wix sends frames (PNG/JPG) + prompt to this API.
- This router:
  1) Stores frames in a session folder (for history + audit)
  2) Creates a real SD3.5 img2img job folder (outputs/YYYY-MM-DD/<job_id>/)
  3) Writes a real meta.json with Doc 18 locked presets via apply_preset_to_meta()
  4) Dispatches to your GPU runtime endpoint (HTTP) and returns output URLs

NO SKELETON:
- If GPU dispatch is not configured, we fail fast with HTTP 503 (not “plan-only”).

CRITICAL (Doc 18):
- Any SD3.5 stage MUST store locked preset parameters:
  steps, CFG, LyCORIS(PRO 2.1) multiplier, GEO multiplier, resolution,
  NO denoise anywhere, upscale optional per-request.

REQUIRED RUNTIME CONFIG (env vars):
- RENDEREXPO_GPU_DISPATCH_URL
    Example: "http://127.0.0.1:8000/api/sd35/img2img"
    (This must be your real GPU/runtime endpoint that executes an img2img job.)
- Optional:
  - RENDEREXPO_GPU_TIMEOUT_SECONDS (default 90)
  - RENDEREXPO_GPU_POLL_SECONDS (default 0.5)  # if your runtime is async and returns a status_url
  - RENDEREXPO_OUTPUTS_MOUNT (default "/outputs")  # where outputs/ is served publicly

RUNTIME CONTRACT (expected from GPU endpoint):
Option A (SYNC):
- Returns JSON including either:
    {"status":"ok","output_image":"output.png"} OR {"status":"ok","output_path":".../output.png"}

Option B (ASYNC + polling):
- Returns JSON including:
    {"status":"queued","status_url":"http://.../api/jobs/<id>/status"}
- Status URL returns:
    {"status":"running"} or {"status":"ok","output_image":"output.png"} or {"status":"error","detail":"..."}

If your runtime differs, adjust _dispatch_to_gpu() mapping at the bottom.
"""

from __future__ import annotations

import os
import io
import json
import uuid
import shutil
import time
import datetime
import urllib.request
import urllib.error
from typing import Optional, Any, Dict, Literal, Tuple

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field

from app.presets_sd35 import apply_preset_to_meta

router = APIRouter(prefix="/api/sketch", tags=["Sketch Realtime"])


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Category = Literal["urban", "suburban", "interior", "wide_hero"]
Shot = Literal["wide", "close"]


# ---------------------------------------------------------------------------
# Env / Config
# ---------------------------------------------------------------------------

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return v


GPU_DISPATCH_URL = _env("RENDEREXPO_GPU_DISPATCH_URL")  # REQUIRED for realtime
OUTPUTS_MOUNT = _env("RENDEREXPO_OUTPUTS_MOUNT", "/outputs")  # public mount path
GPU_TIMEOUT_SECONDS = float(_env("RENDEREXPO_GPU_TIMEOUT_SECONDS", "90") or "90")
GPU_POLL_SECONDS = float(_env("RENDEREXPO_GPU_POLL_SECONDS", "0.5") or "0.5")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_iso() -> str:
    return datetime.datetime.utcnow().isoformat()


def _today_utc_str() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_png_jpg_only(upload: UploadFile) -> None:
    ct = (getattr(upload, "content_type", "") or "").lower().strip()
    if ct and ct not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(status_code=400, detail="Only PNG and JPG are supported.")
    name = (upload.filename or "").lower()
    if name and not (name.endswith(".png") or name.endswith(".jpg") or name.endswith(".jpeg")):
        raise HTTPException(status_code=400, detail="Only .png, .jpg, .jpeg are supported.")


def _session_root(session_id: str) -> str:
    # outputs/YYYY-MM-DD/sessions/<session_id>/
    return os.path.join("outputs", _today_utc_str(), "sessions", session_id)


def _session_meta_path(session_id: str) -> str:
    return os.path.join(_session_root(session_id), "session_meta.json")


def _frames_dir(session_id: str) -> str:
    return os.path.join(_session_root(session_id), "frames")


def _create_session_folder(session_id: str) -> str:
    root = _session_root(session_id)
    _ensure_dir(root)
    _ensure_dir(_frames_dir(session_id))
    return root


def _create_job_folder(job_type: str) -> str:
    """
    Create outputs/YYYY-MM-DD/<job_id>/ and write a small marker.
    """
    today = _today_utc_str()
    job_id = uuid.uuid4().hex
    folder = os.path.join("outputs", today, job_id)
    _ensure_dir(folder)
    try:
        with open(os.path.join(folder, "job_type.txt"), "w", encoding="utf-8") as f:
            f.write(job_type)
    except Exception:
        pass
    return folder


def _parse_job_path(job_folder: str) -> Tuple[Optional[str], Optional[str]]:
    parts = os.path.normpath(job_folder).split(os.sep)
    if len(parts) < 3:
        return None, None
    return parts[-2], parts[-1]


def _job_public_urls(job_folder: str) -> Dict[str, Optional[str]]:
    date_str, job_id = _parse_job_path(job_folder)
    if not date_str or not job_id:
        return {"meta_url": None, "input_url": None, "output_url": None}
    base = f"{OUTPUTS_MOUNT}/{date_str}/{job_id}"
    return {
        "meta_url": f"{base}/meta.json",
        "input_url": f"{base}/input.png",
        "output_url": f"{base}/output.png",
    }


async def _save_upload_stream(upload: UploadFile, dst_path: str) -> None:
    """
    Stream-save UploadFile without reading whole file into RAM.
    """
    try:
        try:
            upload.file.seek(0)
        except Exception:
            pass
        with open(dst_path, "wb") as out:
            shutil.copyfileobj(upload.file, out)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed saving upload '{upload.filename}': {exc}") from exc


def _require_gpu_dispatch() -> str:
    if not GPU_DISPATCH_URL:
        raise HTTPException(
            status_code=503,
            detail=(
                "Sketch realtime requires GPU dispatch. "
                "Set env var RENDEREXPO_GPU_DISPATCH_URL to your real img2img runtime endpoint."
            ),
        )
    return GPU_DISPATCH_URL


def _http_post_json(url: str, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        raise HTTPException(status_code=502, detail=f"GPU dispatch HTTPError: {e.code}: {detail}") from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"GPU dispatch failed: {e}") from e


def _http_get_json(url: str, timeout_s: float) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        raise HTTPException(status_code=502, detail=f"GPU status HTTPError: {e.code}: {detail}") from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"GPU status check failed: {e}") from e


def _dispatch_to_gpu(job_folder: str, timeout_s: float) -> Dict[str, Any]:
    """
    Dispatch the SD3.5 img2img job to your GPU/runtime endpoint.

    EXPECTED:
    - The runtime reads job_folder/meta.json and runs the job, writing output.png into the same folder.

    We send:
      {"job_folder": "...", "meta_path": ".../meta.json"}

    If runtime is SYNC: returns {"status":"ok", ...}
    If runtime is ASYNC: returns {"status":"queued","status_url":"..."} and we poll until ok/error.
    """
    url = _require_gpu_dispatch()
    meta_path = os.path.join(job_folder, "meta.json")

    start = time.time()
    first = _http_post_json(url, {"job_folder": job_folder, "meta_path": meta_path}, timeout_s=timeout_s)

    status = (first.get("status") or "").lower().strip()

    # Sync success
    if status in ("ok", "success", "completed"):
        return {"dispatch_mode": "sync", "runtime_response": first}

    # Async polling
    status_url = first.get("status_url")
    if status_url:
        while True:
            if (time.time() - start) > timeout_s:
                raise HTTPException(status_code=504, detail="GPU job timed out while polling status_url.")
            s = _http_get_json(status_url, timeout_s=timeout_s)
            st = (s.get("status") or "").lower().strip()
            if st in ("ok", "success", "completed"):
                return {"dispatch_mode": "async", "runtime_response": s, "status_url": status_url}
            if st in ("error", "failed"):
                raise HTTPException(status_code=500, detail=f"GPU job failed: {s.get('detail') or s}")
            time.sleep(GPU_POLL_SECONDS)

    # Unknown response
    raise HTTPException(status_code=502, detail=f"GPU dispatch returned unexpected payload: {first}")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class StartSketchSessionRequest(BaseModel):
    """
    Creates a sketch session that locks the preset selectors.
    These are reused unless overridden per frame upload.
    """
    category: Category = Field(..., description="Preset category: urban/suburban/interior/wide_hero")
    shot: Shot = Field(..., description="Preset shot: wide/close")
    upscale_2x: Optional[bool] = Field(None, description="Optional override. If omitted, preset default applies.")
    seed: Optional[int] = Field(None, description="Optional seed. If omitted, we generate one.")
    notes: Optional[str] = Field(None, description="Optional notes for this session.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/start-session")
async def start_sketch_session(request: StartSketchSessionRequest):
    """
    Start a sketch session and store locked preset selectors.
    """
    session_id = uuid.uuid4().hex
    root = _create_session_folder(session_id)

    seed = request.seed if request.seed is not None else int(uuid.uuid4().int % 1_000_000_000)

    session_meta: Dict[str, Any] = {
        "session_id": session_id,
        "created_at": _utc_iso(),
        "engine": "sd35_large_pro_v2_1",
        "model_name": "sd35_large_pro_v2_1",

        "category": request.category,
        "shot": request.shot,
        "seed": seed,
        "notes": request.notes,
        "upscale_2x": request.upscale_2x,  # optional session-level override

        # Hard lock: NO denoise anywhere
        "denoise": 0.0,

        "last_frame_index": -1,
        "last_frame_filename": None,
        "status": "active",
    }

    # Store preset-applied knobs for visibility
    preset_probe: Dict[str, Any] = {
        "type": "img2img",
        "engine": "sd35_large_pro_v2_1",
        "model_name": "sd35_large_pro_v2_1",
        "category": request.category,
        "shot": request.shot,
        "seed": seed,
        "denoise": 0.0,
        "prompt": "probe",
        "planned_output_image": "output.png",
        "status": "planned",
        "mode": "probe",
    }
    apply_preset_to_meta(preset_probe, category=request.category, shot=request.shot, upscale_2x=request.upscale_2x)
    preset_probe["denoise"] = 0.0
    if isinstance(preset_probe.get("upscale"), dict):
        preset_probe["upscale"]["denoise"] = 0.0

    session_meta["preset"] = preset_probe.get("preset", {})
    session_meta["locked_generation_controls"] = {
        "width": preset_probe.get("width"),
        "height": preset_probe.get("height"),
        "num_inference_steps": preset_probe.get("num_inference_steps"),
        "guidance_scale": preset_probe.get("guidance_scale"),
        "lora_config": preset_probe.get("lora_config"),
        "geo_config": preset_probe.get("geo_config"),
        "upscale": preset_probe.get("upscale"),
        "denoise": 0.0,
    }

    _write_json(_session_meta_path(session_id), session_meta)

    return {
        "status": "ok",
        "message": "Sketch session created. Doc 18 presets are locked for this session.",
        "session_id": session_id,
        "session_folder": root,
        "session_meta_path": _session_meta_path(session_id),
        "preset_applied": session_meta.get("preset", {}),
        "locked_generation_controls": session_meta.get("locked_generation_controls", {}),
    }


@router.post("/upload-frame")
async def upload_sketch_frame(
    # Frame upload
    image: UploadFile = File(..., description="Sketch frame image (PNG/JPG only)"),

    # Required session pointer
    session_id: str = Form(..., description="Session ID from /start-session"),

    # Required for REALTIME generation
    prompt: str = Form(..., description="Prompt used to generate the right-side image in realtime."),
    negative_prompt: Optional[str] = Form(None, description="Optional negative prompt"),

    # Img2img transform strength (NOT denoise)
    strength: float = Form(0.70, ge=0.0, le=1.0, description="Img2img transform strength (NOT denoise)."),

    # Optional per-frame overrides (otherwise session defaults apply)
    category: Optional[Category] = Form(None, description="Optional override category"),
    shot: Optional[Shot] = Form(None, description="Optional override shot"),
    upscale_2x: Optional[bool] = Form(None, description="Optional override upscale"),
    seed: Optional[int] = Form(None, description="Optional override seed"),
):
    """
    REALTIME:
    - Stores the sketch frame in the session
    - Creates a real SD3.5 img2img job folder + meta.json
    - Dispatches to GPU runtime and returns output URL

    This is the backend for your Wix two-panel UI:
      Left panel: sketch canvas -> frame upload
      Right panel: generated output.png -> returned URL
    """
    if not image.filename:
        raise HTTPException(status_code=400, detail="Uploaded image has no filename.")
    _ensure_png_jpg_only(image)

    if not prompt or not str(prompt).strip():
        raise HTTPException(status_code=400, detail="prompt is required for realtime generation.")

    meta_path = _session_meta_path(session_id)
    if not os.path.isfile(meta_path):
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")

    session_meta = _read_json(meta_path)

    # Resolve selectors
    use_category: Category = category or session_meta["category"]
    use_shot: Shot = shot or session_meta["shot"]
    use_seed = seed if seed is not None else session_meta.get("seed")
    if use_seed is None:
        use_seed = int(uuid.uuid4().int % 1_000_000_000)

    # Resolve upscale override (frame override > session override > preset default)
    if upscale_2x is None:
        upscale_override = session_meta.get("upscale_2x", None)
    else:
        upscale_override = upscale_2x

    # 1) Save frame into session history
    frames_dir = _frames_dir(session_id)
    _ensure_dir(frames_dir)

    next_idx = int(session_meta.get("last_frame_index", -1)) + 1
    frame_name = f"frame_{next_idx:05d}.png"
    frame_path = os.path.join(frames_dir, frame_name)

    await _save_upload_stream(image, frame_path)

    session_meta["last_frame_index"] = next_idx
    session_meta["last_frame_filename"] = frame_name
    session_meta["updated_at"] = _utc_iso()
    _write_json(meta_path, session_meta)

    frame_meta: Dict[str, Any] = {
        "session_id": session_id,
        "frame_index": next_idx,
        "frame_filename": frame_name,
        "saved_at": _utc_iso(),
        "category": use_category,
        "shot": use_shot,
        "seed": use_seed,
        "denoise": 0.0,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "strength": float(strength),
    }
    frame_meta_path = os.path.join(frames_dir, f"frame_{next_idx:05d}.json")
    _write_json(frame_meta_path, frame_meta)

    # 2) Create REAL job folder for GPU runtime
    job_folder = _create_job_folder(job_type="sketch_img2img_realtime")
    job_id = os.path.basename(job_folder)

    # Stage input as input.png
    job_input = os.path.join(job_folder, "input.png")
    try:
        with open(frame_path, "rb") as src, open(job_input, "wb") as dst:
            shutil.copyfileobj(src, dst)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to stage input.png for job: {exc}") from exc

    # 3) Write REAL meta.json for the runtime to execute
    job_meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": _utc_iso(),

        "type": "img2img",
        "engine": "sd35_large_pro_v2_1",
        "model_name": "sd35_large_pro_v2_1",

        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": use_seed,

        "strength": float(strength),
        "input_image": "input.png",
        "planned_output_image": "output.png",

        # Doc 18 selectors
        "category": use_category,
        "shot": use_shot,

        # Hard lock: NO denoise anywhere
        "denoise": 0.0,

        # Link back to session/frame
        "source": {
            "feature": "sketch",
            "session_id": session_id,
            "frame_index": next_idx,
            "frame_filename": frame_name,
        },

        "status": "queued",
        "mode": "realtime",
    }

    apply_preset_to_meta(job_meta, category=use_category, shot=use_shot, upscale_2x=upscale_override)
    job_meta["denoise"] = 0.0
    if isinstance(job_meta.get("upscale"), dict):
        job_meta["upscale"]["denoise"] = 0.0

    job_meta_path = os.path.join(job_folder, "meta.json")
    _write_json(job_meta_path, job_meta)

    # 4) Dispatch to GPU runtime (REAL) and wait for completion
    dispatch_result = _dispatch_to_gpu(job_folder=job_folder, timeout_s=GPU_TIMEOUT_SECONDS)

    # 5) Return stable public URLs for Wix UI
    public = _job_public_urls(job_folder)

    return {
        "status": "ok",
        "message": "Frame stored and SD3.5 img2img executed in realtime.",
        "session_id": session_id,
        "frame_index": next_idx,
        "frame_path": frame_path,
        "frame_meta_path": frame_meta_path,

        "job_id": job_id,
        "job_folder": job_folder,
        "job_meta_path": job_meta_path,

        "public_urls": public,
        "dispatch": dispatch_result,
        "category_used": use_category,
        "shot_used": use_shot,
        "preset_applied": job_meta.get("preset", {}),
        "upscale_enabled": bool(isinstance(job_meta.get("upscale"), dict) and job_meta["upscale"].get("enabled", False)),
    }
