# app/api/gpu/dispatch.py
"""
RENDEREXPO AI STUDIO - GPU Dispatch Handler (REAL, Production)

FINAL PURPOSE:
- Accept dispatch requests from the planner service.
- Route each job to the correct REAL GPU/local implementation that actually exists in this repo.
- Run jobs in background threads.
- Update job_folder/meta.json with REAL results or REAL errors.
- Never mark a job completed unless the expected result artifact exists when applicable.

IMPORTANT:
- job_folder must be an ABSOLUTE path on the GPU worker filesystem.
- meta.json inside job_folder is the disk source of truth.

LOCKED SKETCH RULE:
- Dedicated sketch route:
    job_type = "sd35_sketch_controlnet"
    pipeline_key = "sd35::sd35_sketch_controlnet"

That route is NOT plain img2img.
It must run:
    sketch.png -> cleanup -> canny.png + depth.png -> SD3.5 Large dual ControlNet -> output.png
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import traceback
import uuid
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# REAL runners that exist in this repo
from app.gpu.cad.from_image import run_cad_from_image
from app.gpu.mesh.from_image import run_mesh_from_image
from app.gpu.sd35 import (
    run_sd35_img2img,
    run_sd35_sketch_controlnet,
    run_sd35_txt2img,
)
from app.gpu.upscale import run_upscale_2x
from app.gpu.video.between_frames import run_video_between_frames
from app.gpu.video.from_image import run_video_from_image

router = APIRouter(prefix="/api/gpu", tags=["GPU Dispatch"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

JobType = Literal[
    "sd35_txt2img",
    "sd35_text2img",
    "sd35_img2img",
    "sd35_sketch_controlnet",
    "upscale_2x",
    "vr_reconstruct",
    "video_from_image",
    "video_between_frames",
    "cad_from_image",
    "mesh_from_image",
]


class GPUDispatchRequest(BaseModel):
    job_type: JobType = Field(..., description="Type of job to run on GPU worker.")
    job_folder: str = Field(..., description="ABSOLUTE path to job folder on the worker filesystem.")
    meta: Dict[str, Any] = Field(..., description="Meta payload written by planner.")

    pipeline_key: Optional[str] = Field(default=None, description="Optional explicit routing key.")
    vr_mode: Optional[str] = Field(default=None, description="VR mode if used by a future VR implementation.")


class GPUDispatchResponse(BaseModel):
    status: str
    message: str
    job_type: str
    job_folder: str
    run_id: str
    accepted_at_epoch: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_JOB_LOCKS: Dict[str, threading.Lock] = {}
_JOB_LOCKS_GUARD = threading.Lock()


def _get_job_lock(job_folder: str) -> threading.Lock:
    with _JOB_LOCKS_GUARD:
        if job_folder not in _JOB_LOCKS:
            _JOB_LOCKS[job_folder] = threading.Lock()
        return _JOB_LOCKS[job_folder]


def _now_epoch() -> int:
    return int(time.time())


def _worker_name() -> str:
    return (os.getenv("GPU_WORKER_NAME") or "gpu-worker").strip()


def _meta_path(job_folder: str) -> str:
    return os.path.join(job_folder, "meta.json")


def _read_meta(job_folder: str) -> Dict[str, Any]:
    p = _meta_path(job_folder)
    if not os.path.isfile(p):
        return {}

    last_exc: Optional[Exception] = None
    for _ in range(10):
        try:
            with open(p, "r", encoding="utf-8") as f:
                raw = f.read()

            if not raw.strip():
                time.sleep(0.05)
                continue

            return json.loads(raw)
        except json.JSONDecodeError as exc:
            last_exc = exc
            time.sleep(0.05)
        except Exception as exc:
            last_exc = exc
            time.sleep(0.05)

    raise RuntimeError(f"Failed reading valid meta.json after retries: {p} :: {last_exc}")


def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="meta_", suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _write_meta(job_folder: str, meta: Dict[str, Any]) -> None:
    _atomic_write_json(_meta_path(job_folder), meta)


def _ensure_job_folder(job_folder: str) -> None:
    if not job_folder or not os.path.isabs(job_folder):
        raise HTTPException(status_code=400, detail="job_folder must be an ABSOLUTE path on the GPU worker.")
    if not os.path.isdir(job_folder):
        raise HTTPException(status_code=400, detail=f"job_folder does not exist: {job_folder}")


def _safe_set(meta: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cur: Dict[str, Any] = meta
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]  # type: ignore[assignment]
    cur[keys[-1]] = value


def _route_key(job_type: str, vr_mode: Optional[str], pipeline_key: Optional[str]) -> str:
    if pipeline_key:
        return pipeline_key.strip().lower()

    if job_type == "vr_reconstruct":
        return f"vr::{(vr_mode or '').strip().lower()}"

    if job_type == "video_from_image":
        return "video::from_image"

    if job_type == "video_between_frames":
        return "video::between_frames"

    if job_type == "cad_from_image":
        return "cad::from_image"

    if job_type == "mesh_from_image":
        return "mesh::from_image"

    if job_type in ("sd35_txt2img", "sd35_text2img"):
        return "sd35::text2img"

    if job_type == "sd35_img2img":
        return "sd35::img2img"

    if job_type == "sd35_sketch_controlnet":
        return "sd35::sd35_sketch_controlnet"

    if job_type == "upscale_2x":
        return "upscale::2x"

    return f"job::{job_type}"


def _ensure_meta_exists(job_folder: str) -> None:
    if not os.path.isfile(_meta_path(job_folder)):
        _write_meta(job_folder, {"status": "queued", "job_folder": job_folder})


def _enforce_locked_multipliers(meta: Dict[str, Any]) -> None:
    default_ly = 0.05
    default_ge = 0.01

    lora_cfg = meta.get("lora_config") if isinstance(meta.get("lora_config"), dict) else None
    geo_cfg = meta.get("geo_config") if isinstance(meta.get("geo_config"), dict) else None
    preset = meta.get("preset") if isinstance(meta.get("preset"), dict) else None

    ly_val = None
    ge_val = None

    if isinstance(lora_cfg, dict):
        ly_val = lora_cfg.get("strength", lora_cfg.get("scale"))
    if ly_val is None and isinstance(preset, dict):
        ly_val = preset.get("lycoris_multiplier")

    if isinstance(geo_cfg, dict):
        ge_val = geo_cfg.get("strength", geo_cfg.get("scale"))
    if ge_val is None and isinstance(preset, dict):
        ge_val = preset.get("geo_multiplier")

    try:
        ly = float(default_ly if ly_val is None else ly_val)
    except Exception:
        ly = float(default_ly)

    try:
        ge = float(default_ge if ge_val is None else ge_val)
    except Exception:
        ge = float(default_ge)

    if not (0.0 <= ly <= 0.20):
        raise RuntimeError(f"Invalid LyCORIS multiplier: {ly}. Allowed: [0.0, 0.20].")
    if not (0.0 <= ge <= 0.05):
        raise RuntimeError(f"Invalid GEO multiplier: {ge}. Allowed: [0.0, 0.05].")

    if isinstance(lora_cfg, dict):
        lora_cfg["strength"] = ly
        lora_cfg["scale"] = ly
        meta["lora_config"] = lora_cfg

    if isinstance(geo_cfg, dict):
        geo_cfg["strength"] = ge
        geo_cfg["scale"] = ge
        meta["geo_config"] = geo_cfg

    if isinstance(preset, dict):
        preset["lycoris_multiplier"] = ly
        preset["geo_multiplier"] = ge
        meta["preset"] = preset


def _require_file(job_folder: str, *names: str, err: str) -> str:
    for n in names:
        p = os.path.join(job_folder, n)
        if os.path.isfile(p):
            return p
    raise HTTPException(status_code=400, detail=err)


def _assert_artifact_exists(path: Any, description: str) -> None:
    if not path or not isinstance(path, str):
        raise RuntimeError(f"{description} did not return a valid path.")
    if not os.path.isfile(path):
        raise RuntimeError(f"{description} path does not exist on disk: {path}")
    if os.path.getsize(path) <= 0:
        raise RuntimeError(f"{description} output file is empty: {path}")


# ---------------------------------------------------------------------------
# Core execution thread
# ---------------------------------------------------------------------------

def _run_job_in_thread(run_id: str, req: Dict[str, Any]) -> None:
    job_type: str = req["job_type"]
    job_folder: str = req["job_folder"]

    lock = _get_job_lock(job_folder)
    with lock:
        meta = _read_meta(job_folder)
        route = _route_key(
            job_type=job_type,
            vr_mode=req.get("vr_mode") or meta.get("vr_mode"),
            pipeline_key=req.get("pipeline_key") or meta.get("pipeline_key"),
        )

        meta["status"] = "running"
        meta.setdefault("dispatch", {})
        _safe_set(meta, "dispatch.run_id", run_id)
        _safe_set(meta, "dispatch.worker", _worker_name())
        _safe_set(meta, "dispatch.route", route)
        _safe_set(meta, "dispatch.started_at_epoch", _now_epoch())
        _safe_set(meta, "dispatch.job_type", job_type)
        _write_meta(job_folder, meta)

    try:
        with lock:
            meta = _read_meta(job_folder)

        if job_type in (
            "sd35_txt2img",
            "sd35_text2img",
            "sd35_img2img",
            "sd35_sketch_controlnet",
            "upscale_2x",
        ):
            with lock:
                _enforce_locked_multipliers(meta)
                _write_meta(job_folder, meta)

        if job_type in ("sd35_txt2img", "sd35_text2img"):
            prompt = meta.get("prompt")
            if not prompt or not isinstance(prompt, str):
                raise RuntimeError("sd35_text2img requires meta.prompt (string).")

            result_path = run_sd35_txt2img(
                job={"date": meta.get("date"), "job_id": meta.get("job_id")},
                payload={**meta, "job_folder": job_folder},
            )
            _assert_artifact_exists(result_path, "SD35 text2img")
            result = {"output_png": result_path}

        elif job_type == "sd35_img2img":
            input_path = None
            for n in ("ref_input.png", "input.png", "image.png", "output.png"):
                p = os.path.join(job_folder, n)
                if os.path.isfile(p):
                    input_path = p
                    break
            if not input_path:
                raise RuntimeError(
                    "sd35_img2img requires one of: ref_input.png, input.png, image.png, output.png inside job_folder."
                )

            prompt = meta.get("prompt")
            if not prompt or not isinstance(prompt, str):
                raise RuntimeError("sd35_img2img requires meta.prompt (string).")

            payload = {**meta, "job_folder": job_folder, "input_image": input_path}
            result_path = run_sd35_img2img(
                job={"date": meta.get("date"), "job_id": meta.get("job_id")},
                payload=payload,
            )
            _assert_artifact_exists(result_path, "SD35 img2img")
            result = {"refine_png": result_path, "input_image": input_path}

        elif job_type == "sd35_sketch_controlnet":
            prompt = meta.get("prompt")
            if not prompt or not isinstance(prompt, str):
                raise RuntimeError("sd35_sketch_controlnet requires meta.prompt (string).")

            sketch_input = None
            for n in ("sketch.png", "input.png", "image.png"):
                p = os.path.join(job_folder, n)
                if os.path.isfile(p):
                    sketch_input = p
                    break
            if not sketch_input:
                raise RuntimeError(
                    "sd35_sketch_controlnet requires one of: sketch.png, input.png, image.png inside job_folder."
                )

            payload = {**meta, "job_folder": job_folder, "input_image": sketch_input}
            result_obj = run_sd35_sketch_controlnet(
                job={"date": meta.get("date"), "job_id": meta.get("job_id")},
                payload=payload,
            )

            if not isinstance(result_obj, dict):
                raise RuntimeError("SD35 sketch controlnet runner must return a dict.")

            canny_path = result_obj.get("canny_png")
            depth_path = result_obj.get("depth_png")
            output_path = result_obj.get("output_png")
            use_depth_control = bool(result_obj.get("use_depth_control", meta.get("use_depth_control", False)))

            _assert_artifact_exists(canny_path, "Sketch ControlNet Canny preprocess")
            if use_depth_control:
                _assert_artifact_exists(depth_path, "Sketch ControlNet Depth preprocess")
            _assert_artifact_exists(output_path, "Sketch ControlNet final output")

            result = {
                "input_image": sketch_input,
                "canny_png": canny_path,
                "output_png": output_path,
                "use_depth_control": use_depth_control,
            }
            if use_depth_control and depth_path:
                result["depth_png"] = depth_path

            optional_upscaled = result_obj.get("final_up2x_png")
            if optional_upscaled:
                _assert_artifact_exists(optional_upscaled, "Sketch ControlNet optional upscale")
                result["final_up2x_png"] = optional_upscaled

        elif job_type == "upscale_2x":
            inp = None
            for n in ("refine.png", "output.png", "image.png", "input.png"):
                p = os.path.join(job_folder, n)
                if os.path.isfile(p):
                    inp = p
                    break
            if not inp:
                raise RuntimeError(
                    "upscale_2x requires one of: refine.png, output.png, image.png, input.png in job_folder."
                )

            payload = {**meta, "job_folder": job_folder, "input_image": inp}
            result_path = run_upscale_2x(
                job={"date": meta.get("date"), "job_id": meta.get("job_id")},
                payload=payload,
            )
            _assert_artifact_exists(result_path, "Upscale 2x")
            result = {"final_up2x_png": result_path, "input_image": inp}

        elif job_type == "video_from_image":
            result = run_video_from_image(job_folder, meta)

        elif job_type == "video_between_frames":
            result = run_video_between_frames(job_folder, meta)

        elif job_type == "cad_from_image":
            result = run_cad_from_image(job_folder, meta)

        elif job_type == "mesh_from_image":
            result = run_mesh_from_image(job_folder, meta)

        elif job_type == "vr_reconstruct":
            raise RuntimeError(
                "vr_reconstruct is not wired yet in this repo. "
                "A real VR GPU implementation must be added before enabling this route."
            )

        else:
            raise RuntimeError(f"Unsupported job_type: {job_type}")

        with lock:
            meta2 = _read_meta(job_folder) or meta

            # Preserve runtime-returned metadata for jobs that return a full meta dict.
            if job_type == "sd35_sketch_controlnet" and isinstance(result_obj, dict):
                merged_meta = dict(meta2)
                merged_meta.update(result_obj)
                meta2 = merged_meta

            meta2["status"] = "completed"
            meta2.setdefault("dispatch", {})
            _safe_set(meta2, "dispatch.completed_at_epoch", _now_epoch())
            _safe_set(meta2, "dispatch.result", result)
            _write_meta(job_folder, meta2)

    except Exception as exc:
        with lock:
            meta2 = _read_meta(job_folder)
            meta2["status"] = "error"
            meta2.setdefault("dispatch", {})
            _safe_set(meta2, "dispatch.completed_at_epoch", _now_epoch())
            _safe_set(meta2, "dispatch.error", {"detail": str(exc), "trace": traceback.format_exc()})
            _write_meta(job_folder, meta2)


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/dispatch", response_model=GPUDispatchResponse)
async def dispatch(request: GPUDispatchRequest):
    _ensure_job_folder(request.job_folder)
    _ensure_meta_exists(request.job_folder)

    lock = _get_job_lock(request.job_folder)
    with lock:
        disk_meta = _read_meta(request.job_folder)
        if not isinstance(disk_meta, dict):
            disk_meta = {}

        merged_meta = {**disk_meta, **(request.meta or {})}
        merged_meta["job_folder"] = request.job_folder

        merged_meta.setdefault("job_id", os.path.basename(request.job_folder.rstrip("/")))
        parent = os.path.basename(os.path.dirname(request.job_folder.rstrip("/")))
        if isinstance(parent, str) and len(parent) == 10 and parent[4] == "-" and parent[7] == "-":
            merged_meta.setdefault("date", parent)

        _write_meta(request.job_folder, merged_meta)

    if request.job_type == "video_between_frames":
        first_a = os.path.join(request.job_folder, "first.png")
        last_a = os.path.join(request.job_folder, "last.png")
        first_b = os.path.join(request.job_folder, "frame_start.png")
        last_b = os.path.join(request.job_folder, "frame_end.png")
        if not ((os.path.isfile(first_a) and os.path.isfile(last_a)) or (os.path.isfile(first_b) and os.path.isfile(last_b))):
            raise HTTPException(
                status_code=400,
                detail="video_between_frames requires first.png+last.png or frame_start.png+frame_end.png in job_folder.",
            )

    if request.job_type == "video_from_image":
        img = os.path.join(request.job_folder, "image.png")
        if not os.path.isfile(img):
            raise HTTPException(status_code=400, detail="video_from_image requires image.png in job_folder.")

    if request.job_type == "cad_from_image":
        _require_file(
            request.job_folder,
            "input.png",
            "image.png",
            err="cad_from_image requires input.png or image.png in job_folder.",
        )

    if request.job_type == "mesh_from_image":
        img = os.path.join(request.job_folder, "image.png")
        if not os.path.isfile(img):
            raise HTTPException(status_code=400, detail="mesh_from_image requires image.png in job_folder.")

    if request.job_type == "vr_reconstruct":
        raise HTTPException(
            status_code=400,
            detail=(
                "vr_reconstruct is currently not available on this worker because "
                "app.gpu.vr.* does not exist in this repository yet."
            ),
        )

    if request.job_type in ("sd35_txt2img", "sd35_text2img"):
        if not merged_meta.get("prompt"):
            raise HTTPException(status_code=400, detail="sd35_text2img requires meta.prompt (string).")
        try:
            _enforce_locked_multipliers(merged_meta)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    if request.job_type == "sd35_img2img":
        if not merged_meta.get("prompt"):
            raise HTTPException(status_code=400, detail="sd35_img2img requires meta.prompt (string).")
        has_input = any(
            os.path.isfile(os.path.join(request.job_folder, n))
            for n in ("ref_input.png", "input.png", "image.png", "output.png")
        )
        if not has_input:
            raise HTTPException(
                status_code=400,
                detail="sd35_img2img requires one of: ref_input.png, input.png, image.png, output.png in job_folder.",
            )
        try:
            _enforce_locked_multipliers(merged_meta)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    if request.job_type == "sd35_sketch_controlnet":
        if not merged_meta.get("prompt"):
            raise HTTPException(status_code=400, detail="sd35_sketch_controlnet requires meta.prompt (string).")

        has_input = any(
            os.path.isfile(os.path.join(request.job_folder, n))
            for n in ("sketch.png", "input.png", "image.png")
        )
        if not has_input:
            raise HTTPException(
                status_code=400,
                detail="sd35_sketch_controlnet requires one of: sketch.png, input.png, image.png in job_folder.",
            )

        controlnet_cfg = merged_meta.get("controlnet")
        if not isinstance(controlnet_cfg, dict) or not controlnet_cfg.get("enabled"):
            raise HTTPException(
                status_code=400,
                detail="sd35_sketch_controlnet requires meta.controlnet.enabled = true.",
            )

        controls = controlnet_cfg.get("controls")
        if not isinstance(controls, list) or len(controls) < 1:
            raise HTTPException(
                status_code=400,
                detail="sd35_sketch_controlnet requires at least one control: canny.",
            )

        control_types = {
            str(c.get("control_type", "")).strip().lower()
            for c in controls
            if isinstance(c, dict)
        }
        if "canny" not in control_types:
            raise HTTPException(
                status_code=400,
                detail="sd35_sketch_controlnet requires a canny control in meta.controlnet.controls.",
            )

        use_depth_control = bool(merged_meta.get("use_depth_control", False))
        merged_meta["use_depth_control"] = use_depth_control
        if use_depth_control and "depth" not in control_types:
            raise HTTPException(
                status_code=400,
                detail="meta.use_depth_control=true requires a depth control in meta.controlnet.controls.",
            )

        try:
            _enforce_locked_multipliers(merged_meta)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    if request.job_type == "upscale_2x":
        has_input = any(
            os.path.isfile(os.path.join(request.job_folder, n))
            for n in ("refine.png", "output.png", "image.png", "input.png")
        )
        if not has_input:
            raise HTTPException(
                status_code=400,
                detail="upscale_2x requires one of: refine.png, output.png, image.png, input.png in job_folder.",
            )

    run_id = f"{_now_epoch()}_{uuid.uuid4().hex[:12]}"

    with lock:
        meta = _read_meta(request.job_folder)
        meta["status"] = "queued"
        meta.setdefault("dispatch", {})
        _safe_set(meta, "dispatch.run_id", run_id)
        _safe_set(meta, "dispatch.accepted_at_epoch", _now_epoch())
        _safe_set(meta, "dispatch.job_type", request.job_type)
        _safe_set(
            meta,
            "dispatch.route",
            _route_key(
                request.job_type,
                request.vr_mode or meta.get("vr_mode"),
                request.pipeline_key or meta.get("pipeline_key"),
            ),
        )
        _write_meta(request.job_folder, meta)

    t = threading.Thread(
        target=_run_job_in_thread,
        args=(run_id, request.model_dump()),
        daemon=True,
    )
    t.start()

    return GPUDispatchResponse(
        status="accepted",
        message="Job accepted by GPU dispatcher.",
        job_type=request.job_type,
        job_folder=request.job_folder,
        run_id=run_id,
        accepted_at_epoch=_now_epoch(),
    )