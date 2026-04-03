# app/routers/sketch.py
"""
RENDEREXPO AI STUDIO - Sketch Router (Planner -> GPU worker)

SKETCH-ONLY UPDATED PURPOSE:
- Keep the same planner/session/upload/output flow.
- Route sketch generation only to the dedicated SDXL + MistoLine engine path.
- Do NOT route sketch through SD3.5.
- Do NOT disturb text2img or any other working generation family.

LOCKED SKETCH PIPELINE:
    uploaded sketch -> stronger preprocess -> MistoLine control image -> SDXL base -> output.png

PLANNER RULE:
- This file NEVER loads models.
- This file ONLY plans the job, stores files/meta, and dispatches to the GPU worker.

LOCKED PIPELINE KEY:
    sdxl::mistoline_sketch
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, Literal, Optional, cast

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/sketch", tags=["Sketch Realtime"])

Category = Literal["urban", "suburban", "interior", "wide_hero"]
Shot = Literal["wide", "close"]


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None or not str(v).strip():
        return default
    return str(v).strip()


OUTPUTS_ROOT = _env("RENDEREXPO_OUTPUTS_ROOT", "outputs") or "outputs"
OUTPUTS_MOUNT = _env("RENDEREXPO_OUTPUTS_MOUNT", "/outputs") or "/outputs"
GPU_BASE_URL = _env("GPU_BASE_URL", "http://127.0.0.1:8002") or "http://127.0.0.1:8002"
GPU_TIMEOUT_SECONDS = int(_env("GPU_TIMEOUT_SECONDS", "600") or "600")


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
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _clean_prompt(value: Optional[str]) -> str:
    return (value or "").strip()


def _default_negative_prompt() -> str:
    return (
        "bloom, glow, haze, fog, dreamy softness, blurry edges, washed out contrast, "
        "low contrast, soft focus, painterly, illustration, cartoon, warped geometry, "
        "distorted windows, melted facade details, extra balconies, extra windows, "
        "deformed massing, oversmoothed surfaces, noisy image, fake siding, ghosting"
    )


def _normalize_category(value: Any) -> Category:
    s = str(value or "").strip().lower()
    if s in {"urban", "suburban", "interior", "wide_hero"}:
        return cast(Category, s)
    return "urban"


def _normalize_shot(value: Any) -> Shot:
    s = str(value or "").strip().lower()
    if s in {"wide", "close"}:
        return cast(Shot, s)
    return "wide"


def _ensure_png_jpg_only(upload: UploadFile) -> None:
    ct = (getattr(upload, "content_type", "") or "").lower().strip()
    if ct and ct not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(status_code=400, detail="Only PNG and JPG are supported for sketch uploads.")


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
        "control_png": f"{mount_root}/{rel}/mistoline_control.png",
        "output_png": f"{mount_root}/{rel}/output.png",
        "final_up2x_png": f"{mount_root}/{rel}/final_up2x.png",
    }


def _dispatch_to_gpu(*, job_type: str, job_folder: str, meta: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
    payload = {
        "job_type": job_type,
        "job_folder": os.path.abspath(job_folder),
        "meta": meta,
        "pipeline_key": meta.get("pipeline_key"),
    }

    req = urllib.request.Request(
        url=f"{GPU_BASE_URL}/api/gpu/dispatch",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=GPU_TIMEOUT_SECONDS) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            parsed = json.loads(body) if body.strip() else {}
            return True, parsed if isinstance(parsed, dict) else {"raw": body}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body) if body.strip() else {}
        except Exception:
            parsed = {"raw": body}
        return False, {
            "status_code": exc.code,
            "error": "gpu_http_error",
            "response": parsed,
        }
    except Exception as exc:
        return False, {
            "error": "gpu_dispatch_failed",
            "detail": str(exc),
        }


def _cleanup_defaults() -> Dict[str, Any]:
    return {
        "enabled": True,
        "grayscale": True,
        "autocontrast": True,
        "contrast_boost": 1.45,
        "median_blur_ksize": 3,
        "adaptive_threshold": False,
        "threshold_value": 165,
        "thicken_lines": True,
        "thicken_iterations": 1,
        "invert_to_white_bg_black_lines": True,
        "canny_low_threshold": 70,
        "canny_high_threshold": 150,
        "final_contrast_boost": 1.35,
        "final_threshold_value": 175,
    }


def _apply_sketch_preset(meta: Dict[str, Any], *, category: Category, shot: Shot, upscale_2x: Optional[bool]) -> None:
    """
    Local locked preset mapping for sketch mode.

    IMPORTANT:
    - This prevents sketch sessions from drifting into the wrong preset family.
    - If the user asks for urban, the sketch session stays urban.
    - wide_hero is only used when explicitly requested.
    """
    width = 1024
    height = 1024

    if shot == "wide":
        steps = 46
        cfg = 5.6
    else:
        steps = 48
        cfg = 6.0

    profile_name_map: Dict[tuple[Category, Shot], str] = {
        ("urban", "wide"): "r1_urban_wide",
        ("urban", "close"): "r1_urban_close",
        ("suburban", "wide"): "r1_suburban_wide",
        ("suburban", "close"): "r1_suburban_close",
        ("interior", "wide"): "r1_interior_wide",
        ("interior", "close"): "r1_interior_close",
        ("wide_hero", "wide"): "r1_wide_hero",
        ("wide_hero", "close"): "r1_wide_hero_close",
    }

    profile_name = profile_name_map[(category, shot)]
    upscale_enabled = bool(upscale_2x) if isinstance(upscale_2x, bool) else False

    meta["width"] = width
    meta["height"] = height
    meta["resolution_policy"] = "preset_default"
    meta["preset_resolution"] = {
        "width": width,
        "height": height,
        "source": "preset_default",
    }
    meta["explicit_dimensions"] = False
    meta["preserve_input_aspect_ratio"] = True
    meta["num_inference_steps"] = steps
    meta["guidance_scale"] = cfg

    # Keep these locked to current approved sketch baseline unless changed later.
    meta["lora_config"] = {
        "path": "models/lycoris/RENDEREXPO_PRO21.safetensors",
        "strength": 0.05,
        "scale": 0.05,
        "label": "LYCORIS_PRO21",
    }
    meta["geo_config"] = {
        "path": "models/geo/RENDEREXPO_GEO.safetensors",
        "strength": 0.01,
        "scale": 0.01,
        "label": "GEO",
    }
    meta["upscale"] = {
        "enabled": upscale_enabled,
        "factor": 2,
        "method": "lanczos",
        "denoise": 0.0,
    }
    meta["preset"] = {
        "profile_name": profile_name,
        "category": category,
        "shot": shot,
        "steps": steps,
        "cfg": cfg,
        "lycoris_multiplier": 0.05,
        "geo_multiplier": 0.01,
        "upscale_default": False,
        "upscale_enabled": upscale_enabled,
        "width": width,
        "height": height,
    }


def _build_sketch_meta(
    *,
    job_id: str,
    prompt: str,
    negative_prompt: Optional[str],
    category: Category,
    shot: Shot,
    seed: int,
    upscale_2x: Optional[bool],
    sketch_filename: str,
    session_id: str,
    frame_index: int,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": _utc_iso(),
        "type": "controlnet",
        "job_type": "sdxl_mistoline_sketch",
        "engine_family": "sdxl",
        "engine": "sdxl_base_1_0",
        "model_name": "sdxl_base_1_0",
        "control_model": "TheMistoAI/MistoLine",
        "pipeline_key": "sdxl::mistoline_sketch",
        "status": "queued",
        "mode_runtime": "gpu-dispatch",
        "task_family": "sketch_to_render",
        "category": category,
        "shot": shot,
        "seed": int(seed),
        "prompt": _clean_prompt(prompt),
        "negative_prompt": _clean_prompt(negative_prompt or "") or _default_negative_prompt(),
        "inputs": {
            "sketch_image": sketch_filename,
            "content_type": "image/png",
            "source": "sketch_session_upload",
            "session_id": session_id,
            "frame_index": frame_index,
        },
        "outputs": {
            "meta": "meta.json",
            "sketch_image": sketch_filename,
            "control_image": "mistoline_control.png",
            "final_image": "output.png",
        },
        "mistoline": {
            "enabled": True,
            "mode": "sketch_control",
            "base_input": sketch_filename,
            "save_preprocessed_images": True,
            "control_image": "mistoline_control.png",
            "cleanup": _cleanup_defaults(),
        },
        "render_intent": {
            "mode": "materialize_structure",
            "preserve_camera_and_massing": True,
            "preserve_building_geometry": True,
            "allow_realistic_materialization": True,
            "allow_site_completion": True,
            "allow_daylight_rendering": True,
            "prompt_role": "materials_lighting_realism_only",
        },
        "optional_polish": {
            "enabled": False,
            "type": "none",
            "note": "Optional light polish/upscale may run after sketch output exists.",
        },
    }

    _apply_sketch_preset(meta, category=category, shot=shot, upscale_2x=upscale_2x)

    meta["denoise"] = 0.0
    meta.pop("strength", None)

    return meta


class StartSketchSessionRequest(BaseModel):
    category: Category = Field(..., description="Preset category: urban/suburban/interior/wide_hero")
    shot: Shot = Field(..., description="Preset shot: wide/close")
    upscale_2x: Optional[bool] = Field(None, description="Optional override. If omitted, default stays disabled.")
    seed: Optional[int] = Field(None, description="Optional seed. If omitted, a seed is generated.")
    notes: Optional[str] = Field(None, description="Optional notes for this session.")


@router.post("/start-session")
async def start_sketch_session(request: StartSketchSessionRequest) -> Dict[str, Any]:
    session_id = uuid.uuid4().hex
    root = _create_session_folder(session_id)

    category = _normalize_category(request.category)
    shot = _normalize_shot(request.shot)
    seed = request.seed if request.seed is not None else int(uuid.uuid4().int % 1_000_000_000)

    preset_probe: Dict[str, Any] = {
        "type": "controlnet",
        "job_type": "sdxl_mistoline_sketch",
        "engine_family": "sdxl",
        "engine": "sdxl_base_1_0",
        "model_name": "sdxl_base_1_0",
        "control_model": "TheMistoAI/MistoLine",
        "category": category,
        "shot": shot,
        "seed": seed,
        "prompt": "probe",
        "planned_output_image": "output.png",
        "status": "planned",
        "mode_runtime": "probe",
        "pipeline_key": "sdxl::mistoline_sketch",
    }
    _apply_sketch_preset(
        preset_probe,
        category=category,
        shot=shot,
        upscale_2x=request.upscale_2x,
    )
    preset_probe["denoise"] = 0.0
    preset_probe.pop("strength", None)

    session_meta: Dict[str, Any] = {
        "session_id": session_id,
        "created_at": _utc_iso(),
        "engine_family": "sdxl",
        "engine": "sdxl_base_1_0",
        "model_name": "sdxl_base_1_0",
        "control_model": "TheMistoAI/MistoLine",
        "route_mode": "sdxl_mistoline_sketch",
        "category": category,
        "shot": shot,
        "seed": seed,
        "notes": request.notes,
        "upscale_2x": request.upscale_2x,
        "last_frame_index": -1,
        "last_frame_filename": None,
        "status": "active",
        "preset": preset_probe.get("preset", {}),
        "locked_generation_controls": {
            "width": preset_probe.get("width"),
            "height": preset_probe.get("height"),
            "num_inference_steps": preset_probe.get("num_inference_steps"),
            "guidance_scale": preset_probe.get("guidance_scale"),
            "upscale": preset_probe.get("upscale"),
            "pipeline_key": "sdxl::mistoline_sketch",
            "category": category,
            "shot": shot,
        },
    }

    _write_json(_session_meta_path(session_id), session_meta)

    return {
        "status": "ok",
        "message": "Sketch session created. SDXL + MistoLine sketch routing is locked for this session.",
        "session_id": session_id,
        "session_folder": root,
        "session_meta_path": _session_meta_path(session_id),
        "preset_applied": session_meta.get("preset", {}),
        "locked_generation_controls": session_meta.get("locked_generation_controls", {}),
    }


@router.post("/upload-frame")
async def upload_sketch_frame(
    image: UploadFile = File(..., description="Sketch frame image (PNG/JPG only)"),
    session_id: str = Form(..., description="Session ID from /start-session"),
    prompt: str = Form(..., description="Prompt for architectural materialization and rendering realism."),
    negative_prompt: Optional[str] = Form(None, description="Optional negative prompt"),
) -> Dict[str, Any]:
    _ensure_png_jpg_only(image)

    session_meta_file = _session_meta_path(session_id)
    if not os.path.isfile(session_meta_file):
        raise HTTPException(status_code=404, detail="Sketch session not found.")

    session_meta = _read_json(session_meta_file)
    if not session_meta:
        raise HTTPException(status_code=500, detail="Sketch session metadata is invalid.")

    next_idx = int(session_meta.get("last_frame_index", -1)) + 1
    frame_name = f"frame_{next_idx:05d}.png"
    frame_path = os.path.join(_frames_dir(session_id), frame_name)
    _save_upload(image, frame_path)

    frame_meta = {
        "session_id": session_id,
        "frame_index": next_idx,
        "frame_filename": frame_name,
        "frame_path": frame_path,
        "created_at": _utc_iso(),
        "prompt": _clean_prompt(prompt),
        "negative_prompt": _clean_prompt(negative_prompt or ""),
    }
    frame_meta_path = os.path.join(_frames_dir(session_id), f"frame_{next_idx:05d}.json")
    _write_json(frame_meta_path, frame_meta)

    job_folder = _create_job_folder()
    job_id = os.path.basename(job_folder)

    sketch_path = os.path.join(job_folder, "sketch.png")
    shutil.copyfile(frame_path, sketch_path)

    category = _normalize_category(session_meta.get("category"))
    shot = _normalize_shot(session_meta.get("shot"))
    seed = int(session_meta.get("seed") or 0)
    upscale_2x = session_meta.get("upscale_2x")

    job_meta = _build_sketch_meta(
        job_id=job_id,
        prompt=prompt,
        negative_prompt=negative_prompt,
        category=category,
        shot=shot,
        seed=seed,
        upscale_2x=upscale_2x if isinstance(upscale_2x, bool) else None,
        sketch_filename="sketch.png",
        session_id=session_id,
        frame_index=next_idx,
    )

    job_meta["date"] = _today_utc_str()

    meta_path_out = os.path.join(job_folder, "meta.json")
    _write_json(meta_path_out, job_meta)

    session_meta["last_frame_index"] = next_idx
    session_meta["last_frame_filename"] = frame_name
    session_meta["last_job_id"] = job_id
    session_meta["last_job_folder"] = job_folder
    _write_json(session_meta_file, session_meta)

    ok, gpu_resp = _dispatch_to_gpu(
        job_type="sdxl_mistoline_sketch",
        job_folder=job_folder,
        meta=job_meta,
    )

    if not ok:
        try:
            job_meta["status"] = "gpu_error"
            job_meta["gpu_error"] = gpu_resp
            _write_json(meta_path_out, job_meta)
        except Exception:
            pass

        return {
            "status": "gpu_error",
            "message": "Sketch frame stored but GPU worker failed.",
            "session_id": session_id,
            "frame_index": next_idx,
            "frame_filename": frame_name,
            "frame_meta_path": frame_meta_path,
            "job_folder": job_folder,
            "job_id": job_id,
            "meta_path": meta_path_out,
            "public_urls": _job_public_urls(job_folder),
            "gpu_error": gpu_resp,
            "pipeline_key": "sdxl::mistoline_sketch",
        }

    try:
        job_meta["status"] = "dispatched"
        job_meta["gpu_response"] = gpu_resp
        _write_json(meta_path_out, job_meta)
    except Exception:
        pass

    return {
        "status": "dispatched",
        "message": "Sketch frame stored and dispatched as a dedicated SDXL + MistoLine sketch job.",
        "session_id": session_id,
        "frame_index": next_idx,
        "frame_filename": frame_name,
        "frame_meta_path": frame_meta_path,
        "job_folder": job_folder,
        "job_id": job_id,
        "meta_path": meta_path_out,
        "public_urls": _job_public_urls(job_folder),
        "gpu_response": gpu_resp,
        "pipeline_key": "sdxl::mistoline_sketch",
    }