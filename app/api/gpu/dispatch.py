# app/api/gpu/dispatch.py
"""
RENDEREXPO AI STUDIO - GPU Dispatch Handler (REAL, Production)

PURPOSE
-------
- Keep existing GPU dispatch behavior for all existing working jobs.
- Keep the working Sketch to Render path untouched:
      job_type     = "sdxl_mistoline_sketch"
      pipeline_key = "sdxl::mistoline_sketch"
- Keep the working Sketch to Redesign path untouched:
      job_type     = "sdxl_mistoline_sketch_redesign"
      pipeline_key = "sdxl::mistoline_sketch_redesign"
- Keep the isolated Moodboard lane:
      job_type     = "space_to_moodboard"
      job_type     = "sd35_moodboard_to_space"
      job_type     = "sd35_apply_moodboard_to_render"
- Add the isolated PowerPaint B+C lane:
      job_type     = "powerpaint_object_removal"
      pipeline_key = "powerpaint::object_removal"

      job_type     = "powerpaint_small_decor_insert"
      pipeline_key = "powerpaint::small_decor_insert"
- Do NOT disturb text2img, img2img, upscale, sketch, redesign, video, CAD, mesh, or VR.

IMPORTANT SAFETY RULE
---------------------
Moodboard is a separate lane.
PowerPaint B+C is a separate lane.
Existing branches are preserved.

LOCKED POWERPAINT SERVICE FAMILY
--------------------------------
AI Interior Cleanup & Small Decor Enhancement

Included:
- AI Object Removal
- AI Small Decor Enhancement / Micro-Staging

Not included:
- furniture staging
- product staging
- reference-guided IP-Adapter workflows
- Option A

STARTUP SAFETY
--------------
Heavy service imports are intentionally lazy.
Do NOT import CAD, mesh, video, SD3.5, SDXL, Comfy, PowerPaint, or other heavy runners
at module startup. Import them only inside the branch that needs them.

This prevents GPU worker port 8002 from hanging during startup when an unrelated
heavy module import stalls.
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

router = APIRouter(prefix="/api/gpu", tags=["GPU Dispatch"])


JobType = Literal[
    "sd35_txt2img",
    "sd35_text2img",
    "sd35_img2img",
    "sdxl_mistoline_sketch",
    "sdxl_mistoline_sketch_redesign",
    "space_to_moodboard",
    "moodboard_to_space",
    "sd35_moodboard_to_space",
    "apply_moodboard_to_render",
    "sd35_apply_moodboard_to_render",
    "powerpaint_object_removal",
    "powerpaint_small_decor_insert",
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
    if job_type == "sdxl_mistoline_sketch":
        return "sdxl::mistoline_sketch"
    if job_type == "sdxl_mistoline_sketch_redesign":
        return "sdxl::mistoline_sketch_redesign"
    if job_type == "space_to_moodboard":
        return "moodboard::space_to_moodboard"
    if job_type in ("moodboard_to_space", "sd35_moodboard_to_space"):
        return "sd35::moodboard_to_space"
    if job_type in ("apply_moodboard_to_render", "sd35_apply_moodboard_to_render"):
        return "sd35::apply_moodboard_to_render"
    if job_type == "powerpaint_object_removal":
        return "powerpaint::object_removal"
    if job_type == "powerpaint_small_decor_insert":
        return "powerpaint::small_decor_insert"
    if job_type == "upscale_2x":
        return "upscale::2x"
    return f"job::{job_type}"


def _ensure_meta_exists(job_folder: str) -> None:
    if not os.path.isfile(_meta_path(job_folder)):
        _write_meta(job_folder, {"status": "queued", "job_folder": job_folder})


def _merge_request_meta_into_job(req: GPUDispatchRequest) -> None:
    lock = _get_job_lock(req.job_folder)
    with lock:
        meta = _read_meta(req.job_folder)
        meta.update(req.meta or {})
        meta["job_folder"] = req.job_folder

        if req.pipeline_key:
            meta["pipeline_key"] = req.pipeline_key
        if req.vr_mode:
            meta["vr_mode"] = req.vr_mode

        meta.setdefault("status", "queued")
        _write_meta(req.job_folder, meta)


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


def _assert_artifact_exists(path: Any, description: str) -> None:
    if not path or not isinstance(path, str):
        raise RuntimeError(f"{description} did not return a valid path.")
    if not os.path.isfile(path):
        raise RuntimeError(f"{description} path does not exist on disk: {path}")
    if os.path.getsize(path) <= 0:
        raise RuntimeError(f"{description} output file is empty: {path}")


def _resolve_input_image_for_sketch_job(job_folder: str, meta: Dict[str, Any]) -> str:
    meta_input_image = meta.get("input_image")
    if isinstance(meta_input_image, str) and meta_input_image.strip():
        candidate = meta_input_image.strip()
        if os.path.isfile(candidate):
            return candidate

    for n in ("sketch.png", "input.png", "image.png"):
        p = os.path.join(job_folder, n)
        if os.path.isfile(p):
            return p

    raise RuntimeError(
        "Sketch job requires meta.input_image or one of: sketch.png, input.png, image.png inside job_folder."
    )


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
            "sd35_moodboard_to_space",
            "moodboard_to_space",
            "sd35_apply_moodboard_to_render",
            "apply_moodboard_to_render",
            "upscale_2x",
        ):
            with lock:
                _enforce_locked_multipliers(meta)
                _write_meta(job_folder, meta)

        if job_type in ("sd35_txt2img", "sd35_text2img"):
            prompt = meta.get("prompt")
            if not prompt or not isinstance(prompt, str):
                raise RuntimeError("sd35_text2img requires meta.prompt (string).")

            from app.gpu.sd35 import run_sd35_txt2img

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

            from app.gpu.sd35 import run_sd35_img2img

            payload = {**meta, "job_folder": job_folder, "input_image": input_path}
            result_path = run_sd35_img2img(
                job={"date": meta.get("date"), "job_id": meta.get("job_id")},
                payload=payload,
            )
            _assert_artifact_exists(result_path, "SD35 img2img")
            result = {"refine_png": result_path, "input_image": input_path}

        elif job_type == "sdxl_mistoline_sketch":
            prompt = meta.get("prompt")
            if not prompt or not isinstance(prompt, str):
                raise RuntimeError("sdxl_mistoline_sketch requires meta.prompt (string).")

            from app.gpu.sdxl_mistoline import run_sdxl_mistoline_sketch

            sketch_input = _resolve_input_image_for_sketch_job(job_folder, meta)

            payload = {**meta, "job_folder": job_folder, "input_image": sketch_input}
            result_obj = run_sdxl_mistoline_sketch(
                job={"date": meta.get("date"), "job_id": meta.get("job_id")},
                payload=payload,
            )

            if not isinstance(result_obj, dict):
                raise RuntimeError("SDXL MistoLine sketch runner must return a dict.")

            output_path = result_obj.get("output_png")
            _assert_artifact_exists(output_path, "SDXL MistoLine final output")

            result = {
                "input_image": sketch_input,
                "output_png": output_path,
                "engine_family": "sdxl",
                "engine": "sdxl_base_1_0",
                "control_model": "TheMistoAI/MistoLine",
                "pipeline_key": "sdxl::mistoline_sketch",
            }

            optional_control = result_obj.get("control_png")
            if optional_control:
                _assert_artifact_exists(optional_control, "MistoLine control image")
                result["control_png"] = optional_control

            optional_upscaled = result_obj.get("final_up2x_png")
            if optional_upscaled:
                _assert_artifact_exists(optional_upscaled, "MistoLine optional upscale")
                result["final_up2x_png"] = optional_upscaled

        elif job_type == "sdxl_mistoline_sketch_redesign":
            from app.services.sketch_redesign_runner import run_anyline_mistoline_sketch_redesign

            sketch_input = _resolve_input_image_for_sketch_job(job_folder, meta)

            result_obj = run_anyline_mistoline_sketch_redesign(
                input_image_path=sketch_input,
                output_dir=job_folder,
                style_preset=meta.get("style_preset"),
                materials_notes=meta.get("materials_notes"),
                atmosphere_notes=meta.get("atmosphere_notes"),
                background_notes=meta.get("background_notes"),
                mood_notes=meta.get("mood_notes"),
                style_notes=meta.get("style_notes"),
                aesthetic_notes=meta.get("aesthetic_notes"),
                negative_prompt_override=meta.get("negative_prompt_override"),
                seed=meta.get("seed"),
            )

            if not isinstance(result_obj, dict):
                raise RuntimeError("Sketch redesign runner must return a dict.")

            output_path = result_obj.get("output_png")
            _assert_artifact_exists(output_path, "SDXL MistoLine redesign output")

            result = {
                "input_image": sketch_input,
                "output_png": output_path,
                "engine_family": "sdxl",
                "engine": "sdxl_base_1_0",
                "control_model": "TheMistoAI/MistoLine",
                "pipeline_key": "sdxl::mistoline_sketch_redesign",
                "mode": "sketch_to_redesign",
                "product_promise": result_obj.get("product_promise"),
                "warning_text": result_obj.get("warning_text"),
                "style_preset": result_obj.get("style_preset"),
                "prompt": result_obj.get("prompt"),
                "negative_prompt": result_obj.get("negative_prompt"),
                "allowed_client_fields": result_obj.get("allowed_client_fields"),
            }

            optional_control = result_obj.get("control_png")
            if optional_control:
                _assert_artifact_exists(optional_control, "MistoLine redesign control image")
                result["control_png"] = optional_control

            optional_upscaled = result_obj.get("final_up2x_png")
            if optional_upscaled:
                _assert_artifact_exists(optional_upscaled, "MistoLine redesign optional upscale")
                result["final_up2x_png"] = optional_upscaled

        elif job_type == "space_to_moodboard":
            from app.gpu.moodboard import run_space_to_moodboard

            result_obj = run_space_to_moodboard(
                job={"date": meta.get("date"), "job_id": meta.get("job_id")},
                payload={**meta, "job_folder": job_folder},
            )

            if not isinstance(result_obj, dict):
                raise RuntimeError("Space to Moodboard runner must return a dict.")

            _assert_artifact_exists(result_obj.get("moodboard_grid_png"), "Space to Moodboard grid")
            _assert_artifact_exists(result_obj.get("palette_json"), "Space to Moodboard palette")
            _assert_artifact_exists(result_obj.get("extracted_assets_json"), "Space to Moodboard assets")

            result = {
                "moodboard_grid_png": result_obj.get("moodboard_grid_png"),
                "palette_json": result_obj.get("palette_json"),
                "extracted_assets_json": result_obj.get("extracted_assets_json"),
                "mode": "space_to_moodboard",
                "engine_family": "analysis",
                "pipeline_key": "moodboard::space_to_moodboard",
            }

        elif job_type in ("moodboard_to_space", "sd35_moodboard_to_space"):
            from app.gpu.moodboard import run_sd35_moodboard_to_space

            result_obj = run_sd35_moodboard_to_space(
                job={"date": meta.get("date"), "job_id": meta.get("job_id")},
                payload={**meta, "job_folder": job_folder},
            )

            if not isinstance(result_obj, dict):
                raise RuntimeError("SD35 Moodboard to Space runner must return a dict.")

            _assert_artifact_exists(result_obj.get("output_png"), "SD35 Moodboard to Space output")
            _assert_artifact_exists(result_obj.get("moodboard_grid_png"), "SD35 Moodboard to Space grid")
            _assert_artifact_exists(result_obj.get("palette_json"), "SD35 Moodboard to Space palette")
            _assert_artifact_exists(result_obj.get("extracted_assets_json"), "SD35 Moodboard to Space assets")

            result = {
                "output_png": result_obj.get("output_png"),
                "moodboard_grid_png": result_obj.get("moodboard_grid_png"),
                "palette_json": result_obj.get("palette_json"),
                "extracted_assets_json": result_obj.get("extracted_assets_json"),
                "mode": "sd35_moodboard_to_space",
                "engine_family": "sd35",
                "pipeline_key": "sd35::moodboard_to_space",
                "conditioning_mode": result_obj.get("conditioning_mode"),
            }

        elif job_type in ("apply_moodboard_to_render", "sd35_apply_moodboard_to_render"):
            from app.gpu.moodboard import run_sd35_apply_moodboard_to_render

            result_obj = run_sd35_apply_moodboard_to_render(
                job={"date": meta.get("date"), "job_id": meta.get("job_id")},
                payload={**meta, "job_folder": job_folder},
            )

            if not isinstance(result_obj, dict):
                raise RuntimeError("SD35 Apply Moodboard to Render runner must return a dict.")

            _assert_artifact_exists(result_obj.get("output_png"), "SD35 Apply Moodboard to Render output")

            result = {
                "output_png": result_obj.get("output_png"),
                "input_image": result_obj.get("input_image"),
                "moodboard_folder": result_obj.get("moodboard_folder"),
                "mode": "sd35_apply_moodboard_to_render",
                "engine_family": "sd35",
                "pipeline_key": "sd35::apply_moodboard_to_render",
                "conditioning_mode": result_obj.get("conditioning_mode"),
            }

        elif job_type == "powerpaint_object_removal":
            from app.gpu.powerpaint import run_powerpaint_object_removal

            result_obj = run_powerpaint_object_removal(
                job={"date": meta.get("date"), "job_id": meta.get("job_id")},
                payload={**meta, "job_folder": job_folder},
            )

            if not isinstance(result_obj, dict):
                raise RuntimeError("PowerPaint Object Removal runner must return a dict.")

            _assert_artifact_exists(result_obj.get("output_png"), "PowerPaint Object Removal output")

            result = {
                "output_png": result_obj.get("output_png"),
                "input_image": result_obj.get("input_image"),
                "mask_image": result_obj.get("mask_image"),
                "mode": "powerpaint_object_removal",
                "service_family": "AI Interior Cleanup & Small Decor Enhancement",
                "service_name": "AI Object Removal",
                "engine_family": "powerpaint",
                "engine": "PowerPaint-v2-1",
                "pipeline_key": "powerpaint::object_removal",
                "task": result_obj.get("task"),
                "seed": result_obj.get("seed"),
                "steps": result_obj.get("steps"),
                "guidance_scale": result_obj.get("guidance_scale"),
                "fitting_degree": result_obj.get("fitting_degree"),
            }

        elif job_type == "powerpaint_small_decor_insert":
            from app.gpu.powerpaint import run_powerpaint_small_decor_insert

            result_obj = run_powerpaint_small_decor_insert(
                job={"date": meta.get("date"), "job_id": meta.get("job_id")},
                payload={**meta, "job_folder": job_folder},
            )

            if not isinstance(result_obj, dict):
                raise RuntimeError("PowerPaint Small Decor runner must return a dict.")

            _assert_artifact_exists(result_obj.get("output_png"), "PowerPaint Small Decor output")

            result = {
                "output_png": result_obj.get("output_png"),
                "input_image": result_obj.get("input_image"),
                "mask_image": result_obj.get("mask_image"),
                "mode": "powerpaint_small_decor_insert",
                "service_family": "AI Interior Cleanup & Small Decor Enhancement",
                "service_name": "AI Small Decor Enhancement / Micro-Staging",
                "engine_family": "powerpaint",
                "engine": "PowerPaint-v2-1",
                "pipeline_key": "powerpaint::small_decor_insert",
                "task": result_obj.get("task"),
                "seed": result_obj.get("seed"),
                "steps": result_obj.get("steps"),
                "guidance_scale": result_obj.get("guidance_scale"),
                "fitting_degree": result_obj.get("fitting_degree"),
            }

        elif job_type == "upscale_2x":
            inp = None
            for n in ("refine.png", "output.png", "image.png", "input.png"):
                p = os.path.join(job_folder, n)
                if os.path.isfile(p):
                    inp = p
                    break
            if not inp:
                raise RuntimeError("upscale_2x requires an input image inside job_folder.")

            from app.gpu.upscale import run_upscale_2x

            result_path = run_upscale_2x(
                job={"date": meta.get("date"), "job_id": meta.get("job_id")},
                payload={**meta, "job_folder": job_folder, "input_image": inp},
            )
            _assert_artifact_exists(result_path, "Upscale 2x")
            result = {"upscaled_png": result_path, "input_image": inp}

        elif job_type == "cad_from_image":
            from app.gpu.cad.from_image import run_cad_from_image

            result_path = run_cad_from_image(
                job={"date": meta.get("date"), "job_id": meta.get("job_id")},
                payload={**meta, "job_folder": job_folder},
            )
            _assert_artifact_exists(result_path, "CAD from image")
            result = {"cad_output": result_path}

        elif job_type == "mesh_from_image":
            from app.gpu.mesh.from_image import run_mesh_from_image

            result_path = run_mesh_from_image(
                job={"date": meta.get("date"), "job_id": meta.get("job_id")},
                payload={**meta, "job_folder": job_folder},
            )
            _assert_artifact_exists(result_path, "Mesh from image")
            result = {"mesh_output": result_path}

        elif job_type == "video_from_image":
            from app.gpu.video.from_image import run_video_from_image

            result_path = run_video_from_image(
                job={"date": meta.get("date"), "job_id": meta.get("job_id")},
                payload={**meta, "job_folder": job_folder},
            )
            _assert_artifact_exists(result_path, "Video from image")
            result = {"video_output": result_path}

        elif job_type == "video_between_frames":
            from app.gpu.video.between_frames import run_video_between_frames

            result_path = run_video_between_frames(
                job={"date": meta.get("date"), "job_id": meta.get("job_id")},
                payload={**meta, "job_folder": job_folder},
            )
            _assert_artifact_exists(result_path, "Video between frames")
            result = {"video_output": result_path}

        elif job_type == "vr_reconstruct":
            raise RuntimeError("vr_reconstruct is not implemented in this dispatcher.")

        else:
            raise RuntimeError(f"Unsupported job_type: {job_type}")

        with lock:
            meta = _read_meta(job_folder)
            meta["status"] = "completed"
            meta["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            meta["result"] = result
            _safe_set(meta, "dispatch.finished_at_epoch", _now_epoch())
            _write_meta(job_folder, meta)

    except Exception as exc:
        err_trace = traceback.format_exc()
        with lock:
            meta = _read_meta(job_folder)
            meta["status"] = "failed"
            meta["error"] = str(exc)
            meta["traceback"] = err_trace
            _safe_set(meta, "dispatch.finished_at_epoch", _now_epoch())
            _write_meta(job_folder, meta)


@router.post("/dispatch", response_model=GPUDispatchResponse)
async def dispatch(req: GPUDispatchRequest) -> GPUDispatchResponse:
    _ensure_job_folder(req.job_folder)
    _ensure_meta_exists(req.job_folder)
    _merge_request_meta_into_job(req)

    run_id = f"{_now_epoch()}_{uuid.uuid4().hex[:12]}"

    t = threading.Thread(
        target=_run_job_in_thread,
        args=(run_id, req.model_dump()),
        daemon=True,
    )
    t.start()

    return GPUDispatchResponse(
        status="accepted",
        message="GPU dispatch accepted.",
        job_type=req.job_type,
        job_folder=req.job_folder,
        run_id=run_id,
        accepted_at_epoch=_now_epoch(),
    )