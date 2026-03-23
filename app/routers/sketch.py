# app/routers/sketch.py
"""
RENDEREXPO AI STUDIO - Sketch Router (Planner -> GPU worker)

FINAL PURPOSE:
- Sketch mode is NO LONGER treated as plain img2img.
- Sketch mode plans and dispatches a dedicated SD3.5 Large dual-ControlNet job:
    uploaded sketch -> cleanup -> canny + depth -> SD3.5 Large -> optional light polish/upscale

WHAT THIS ROUTER DOES:
1) Creates/maintains a sketch session
2) Stores uploaded sketch frames for history/audit
3) Creates a real outputs/YYYY-MM-DD/<job_id>/ folder
4) Saves the uploaded sketch as the source image for preprocessing
5) Writes a real meta.json describing a dedicated sketch-control pipeline
6) Dispatches to the GPU worker through app.clients.gpu_client

IMPORTANT:
- This is the PLANNER router only.
- No SD3.5 model is loaded here.
- No preprocessing is executed here.
- The GPU worker must implement the runtime for:
    job_type = "sd35_sketch_controlnet"
    pipeline_key = "sd35::sd35_sketch_controlnet"

LOCKED SERVICE SPLIT:
- Planner = port 8012
- GPU worker = port 8002

DESIGN LOCK:
- Do NOT route sketch mode through plain img2img as the main method.
- Do NOT use prompt-only as the main structural method.
- The structural route is:
    sketch.png -> cleanup -> canny.png + depth.png -> SD3.5 dual ControlNet
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import uuid
from typing import Any, Dict, Literal, Optional, Tuple

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.clients.gpu_client import dispatch_sd35_sketch_controlnet
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


OUTPUTS_MOUNT = _env("RENDEREXPO_OUTPUTS_MOUNT", "/outputs")


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
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _ensure_png_jpg_only(upload: UploadFile) -> None:
    ct = (getattr(upload, "content_type", "") or "").lower().strip()
    if ct and ct not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(status_code=400, detail="Only PNG and JPG are supported.")
    name = (upload.filename or "").lower()
    if name and not (name.endswith(".png") or name.endswith(".jpg") or name.endswith(".jpeg")):
        raise HTTPException(status_code=400, detail="Only .png, .jpg, .jpeg are supported.")


def _session_root(session_id: str) -> str:
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
        return {
            "meta_url": None,
            "sketch_url": None,
            "canny_url": None,
            "depth_url": None,
            "output_url": None,
            "final_up2x_url": None,
        }

    base = f"{OUTPUTS_MOUNT}/{date_str}/{job_id}"
    return {
        "meta_url": f"{base}/meta.json",
        "sketch_url": f"{base}/sketch.png",
        "canny_url": f"{base}/canny.png",
        "depth_url": f"{base}/depth.png",
        "output_url": f"{base}/output.png",
        "final_up2x_url": f"{base}/final_up2x.png",
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
        raise HTTPException(status_code=500, detail=f"Failed saving upload '{upload.filename}': {exc}") from exc


def _clean_prompt(prompt: str) -> str:
    return " ".join((prompt or "").strip().split())


def _default_negative_prompt() -> str:
    return (
        "sketch, line drawing, line art, blueprint, technical drawing, monochrome, unfinished architecture, "
        "white model, clay render, flat shading, cartoon, anime, blurry, distorted windows, warped geometry, "
        "fantasy building, changed camera angle, changed massing, cropped composition, text, watermark, logo"
    )


def _cleanup_defaults() -> Dict[str, Any]:
    return {
        "enabled": True,
        "grayscale": True,
        "autocontrast": True,
        "contrast_boost": 1.2,
        "median_blur_ksize": 3,
        "adaptive_threshold": False,
        "threshold_value": 185,
        "morphology_close": False,
        "close_kernel": 2,
        "thicken_lines": False,
        "thicken_kernel": 2,
        "invert_to_white_bg_black_lines": True,
    }


def _build_sketch_controlnet_meta(
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
        "job_type": "sd35_sketch_controlnet",
        "engine": "sd35_large_pro_v2_1",
        "model_name": "sd35_large_pro_v2_1",
        "pipeline_key": "sd35::sd35_sketch_controlnet",
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
            "canny_image": "canny.png",
            "depth_image": "depth.png",
            "final_image": "output.png",
        },
        "controlnet": {
            "enabled": True,
            "mode": "multi",
            "base_input": sketch_filename,
            "save_preprocessed_images": True,
            "controls": [
                {
                    "control_type": "canny",
                    "input_image": "canny.png",
                    "conditioning_scale": 1.0,
                    "preprocessor": {
                        "name": "opencv_canny",
                        "source_image": sketch_filename,
                        "low_threshold": 100,
                        "high_threshold": 200,
                        "invert_if_dark_background": False,
                        "line_boost": False,
                        "line_boost_kernel": 2,
                        "cleanup": _cleanup_defaults(),
                    },
                },
                {
                    "control_type": "depth",
                    "input_image": "depth.png",
                    "conditioning_scale": 0.85,
                    "preprocessor": {
                        "name": "midas_depth",
                        "source_image": sketch_filename,
                        "normalize_to_png": True,
                        "invert_output": False,
                        "blur_radius": 0.0,
                        "cleanup": _cleanup_defaults(),
                    },
                },
            ],
            "note": (
                "Sketch mode is routed through SD3.5 dual ControlNet. "
                "Do not fall back to plain img2img as the primary generation method."
            ),
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
            "note": "Optional light polish/upscale may run after ControlNet output exists.",
        },
    }

    apply_preset_to_meta(meta, category=category, shot=shot, upscale_2x=upscale_2x)

    # Safety relock: sketch mode must not degrade into img2img denoise logic.
    meta["denoise"] = 0.0
    meta.pop("strength", None)

    if isinstance(meta.get("upscale"), dict):
        meta["upscale"]["denoise"] = 0.0

    return meta


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class StartSketchSessionRequest(BaseModel):
    category: Category = Field(..., description="Preset category: urban/suburban/interior/wide_hero")
    shot: Shot = Field(..., description="Preset shot: wide/close")
    upscale_2x: Optional[bool] = Field(None, description="Optional override. If omitted, preset default applies.")
    seed: Optional[int] = Field(None, description="Optional seed. If omitted, we generate one.")
    notes: Optional[str] = Field(None, description="Optional notes for this session.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/start-session")
async def start_sketch_session(request: StartSketchSessionRequest) -> Dict[str, Any]:
    session_id = uuid.uuid4().hex
    root = _create_session_folder(session_id)

    seed = request.seed if request.seed is not None else int(uuid.uuid4().int % 1_000_000_000)

    preset_probe: Dict[str, Any] = {
        "type": "controlnet",
        "job_type": "sd35_sketch_controlnet",
        "engine": "sd35_large_pro_v2_1",
        "model_name": "sd35_large_pro_v2_1",
        "category": request.category,
        "shot": request.shot,
        "seed": seed,
        "prompt": "probe",
        "planned_output_image": "output.png",
        "status": "planned",
        "mode_runtime": "probe",
        "pipeline_key": "sd35::sd35_sketch_controlnet",
    }
    apply_preset_to_meta(
        preset_probe,
        category=request.category,
        shot=request.shot,
        upscale_2x=request.upscale_2x,
    )
    preset_probe["denoise"] = 0.0
    preset_probe.pop("strength", None)
    if isinstance(preset_probe.get("upscale"), dict):
        preset_probe["upscale"]["denoise"] = 0.0

    session_meta: Dict[str, Any] = {
        "session_id": session_id,
        "created_at": _utc_iso(),
        "engine": "sd35_large_pro_v2_1",
        "model_name": "sd35_large_pro_v2_1",
        "route_mode": "sd35_sketch_controlnet",
        "category": request.category,
        "shot": request.shot,
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
            "lora_config": preset_probe.get("lora_config"),
            "geo_config": preset_probe.get("geo_config"),
            "upscale": preset_probe.get("upscale"),
            "pipeline_key": "sd35::sd35_sketch_controlnet",
        },
    }

    _write_json(_session_meta_path(session_id), session_meta)

    return {
        "status": "ok",
        "message": "Sketch session created. Dual-ControlNet sketch routing is locked for this session.",
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
    category: Optional[Category] = Form(None, description="Optional override category"),
    shot: Optional[Shot] = Form(None, description="Optional override shot"),
    upscale_2x: Optional[bool] = Form(None, description="Optional override upscale"),
    seed: Optional[int] = Form(None, description="Optional override seed"),
) -> Dict[str, Any]:
    if not image.filename:
        raise HTTPException(status_code=400, detail="Uploaded image has no filename.")
    _ensure_png_jpg_only(image)

    prompt_clean = _clean_prompt(prompt)
    if not prompt_clean:
        raise HTTPException(status_code=400, detail="prompt is required for realtime generation.")

    meta_path = _session_meta_path(session_id)
    if not os.path.isfile(meta_path):
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")

    session_meta = _read_json(meta_path)

    use_category: Category = category or session_meta["category"]
    use_shot: Shot = shot or session_meta["shot"]
    use_seed = seed if seed is not None else session_meta.get("seed")
    if use_seed is None:
        use_seed = int(uuid.uuid4().int % 1_000_000_000)

    upscale_override = session_meta.get("upscale_2x", None) if upscale_2x is None else upscale_2x

    # 1) Save frame into session history
    frames_dir = _frames_dir(session_id)
    _ensure_dir(frames_dir)

    next_idx = int(session_meta.get("last_frame_index", -1)) + 1
    frame_name = f"frame_{next_idx:05d}.png"
    frame_path = os.path.join(frames_dir, frame_name)

    await _save_upload_stream(image, frame_path)

    frame_meta: Dict[str, Any] = {
        "session_id": session_id,
        "frame_index": next_idx,
        "frame_filename": frame_name,
        "saved_at": _utc_iso(),
        "category": use_category,
        "shot": use_shot,
        "seed": use_seed,
        "prompt": prompt_clean,
        "negative_prompt": negative_prompt,
        "route_mode": "sd35_sketch_controlnet",
    }
    frame_meta_path = os.path.join(frames_dir, f"frame_{next_idx:05d}.json")
    _write_json(frame_meta_path, frame_meta)

    session_meta["last_frame_index"] = next_idx
    session_meta["last_frame_filename"] = frame_name
    session_meta["updated_at"] = _utc_iso()
    _write_json(meta_path, session_meta)

    # 2) Create real job folder for this frame
    job_folder = _create_job_folder(job_type="sd35_sketch_controlnet")
    job_id = os.path.basename(job_folder)

    sketch_path = os.path.join(job_folder, "sketch.png")
    shutil.copyfile(frame_path, sketch_path)

    # Compatibility duplicate if any legacy tooling expects input.png to exist.
    input_path = os.path.join(job_folder, "input.png")
    try:
        shutil.copyfile(frame_path, input_path)
    except Exception:
        pass

    # 3) Build dedicated sketch-control meta
    job_meta = _build_sketch_controlnet_meta(
        job_id=job_id,
        prompt=prompt_clean,
        negative_prompt=negative_prompt,
        category=use_category,
        shot=use_shot,
        seed=int(use_seed),
        upscale_2x=upscale_override,
        sketch_filename="sketch.png",
        session_id=session_id,
        frame_index=next_idx,
    )

    job_meta["session"] = {
        "session_id": session_id,
        "frame_index": next_idx,
        "frame_filename": frame_name,
        "frame_meta_filename": os.path.basename(frame_meta_path),
    }

    job_meta["planner_artifacts"] = {
        "session_frame_source": os.path.relpath(frame_path).replace("\\", "/"),
        "copied_into_job_folder": ["sketch.png", "input.png"],
    }

    job_meta["public_urls"] = _job_public_urls(job_folder)

    meta_path_out = os.path.join(job_folder, "meta.json")
    _write_json(meta_path_out, job_meta)

    # 4) Dispatch to GPU worker through shared planner client
    ok, gpu_resp = dispatch_sd35_sketch_controlnet(job_folder=job_folder, meta=job_meta)

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
            "pipeline_key": "sd35::sd35_sketch_controlnet",
            "controlnet_mode": "canny+depth",
        }

    try:
        job_meta["status"] = "dispatched"
        job_meta["gpu_response"] = gpu_resp
        _write_json(meta_path_out, job_meta)
    except Exception:
        pass

    # 5) Return
    return {
        "status": "dispatched",
        "message": "Sketch frame stored and dispatched as a dedicated SD3.5 dual-ControlNet sketch job.",
        "session_id": session_id,
        "frame_index": next_idx,
        "frame_filename": frame_name,
        "frame_meta_path": frame_meta_path,
        "job_folder": job_folder,
        "job_id": job_id,
        "meta_path": meta_path_out,
        "public_urls": _job_public_urls(job_folder),
        "gpu_response": gpu_resp,
        "pipeline_key": "sd35::sd35_sketch_controlnet",
        "controlnet_mode": "canny+depth",
    }