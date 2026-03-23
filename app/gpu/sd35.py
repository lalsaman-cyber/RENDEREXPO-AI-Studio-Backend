from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from runtime.sd35_runtime import SD35Runtime

# -------------------------------------------------------------------
# Locked profiles (Doc 19+)
# -------------------------------------------------------------------
# These are compatibility defaults only.
# Planner/routers remain the real source of truth and should write
# steps / cfg / width / height / upscale into meta/payload explicitly.
#
# IMPORTANT FOR IMG2IMG / SKETCH:
# - These width/height values are DEFAULT compatibility fallbacks only.
# - They must NOT silently override planner/runtime aspect-preservation logic.
PROFILES = {
    "r1_wide_hero": {
        "cfg": 5.6,
        "steps": 46,
        "width": 1024,
        "height": 1024,
        "upscale": {
            "enabled": True,
            "factor": 2,
            "method": "lanczos",
        },
    },
    "r1_close_detail": {
        "cfg": 6.0,
        "steps": 48,
        "width": 1024,
        "height": 1024,
        "upscale": {
            "enabled": False,
            "factor": 2,
            "method": "lanczos",
        },
    },
    "luxury_interior_heavy_detail": {
        "cfg": 6.0,
        "steps": 60,
        "width": 1024,
        "height": 1024,
        "upscale": {
            "enabled": True,
            "factor": 2,
            "method": "lanczos",
        },
    },
}

# -------------------------------------------------------------------
# Shared runtime ownership
# -------------------------------------------------------------------
# IMPORTANT:
# - We do NOT load a separate diffusers pipeline here.
# - We use SD35Runtime as the single real execution engine.
# - This avoids duplicate model ownership / duplicate VRAM occupancy.
# - gpu_entry.py may inject its runtime via set_runtime(runtime).
# - If no runtime is injected, we lazily create one here as a compatibility fallback.
# -------------------------------------------------------------------

_INJECTED_RUNTIME: Optional[SD35Runtime] = None
_FALLBACK_RUNTIME: Optional[SD35Runtime] = None


def set_runtime(runtime: SD35Runtime) -> None:
    """
    Allow the GPU worker entrypoint to inject the already-owned runtime.
    This is the preferred path for commercial stability.
    """
    global _INJECTED_RUNTIME
    _INJECTED_RUNTIME = runtime


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw if raw else default


def _runtime_enabled() -> bool:
    requested_mode = _env_str("SD35_RUNTIME_MODE", "lazy").lower()
    run_real = _env_flag("RUN_REAL_SD35", False)
    return run_real and requested_mode in {"lazy", "real"}


def _get_runtime() -> SD35Runtime:
    """
    Return the injected runtime if available; otherwise lazy-create
    a compatibility fallback runtime.

    This function must not eagerly preload on import.
    """
    global _FALLBACK_RUNTIME

    if _INJECTED_RUNTIME is not None:
        return _INJECTED_RUNTIME

    if not _runtime_enabled():
        raise RuntimeError(
            "SD35 runtime is disabled. "
            "Enable RUN_REAL_SD35=1 and set SD35_RUNTIME_MODE to lazy or real."
        )

    if _FALLBACK_RUNTIME is None:
        device = _env_str("SD35_DEVICE", "cuda")
        _FALLBACK_RUNTIME = SD35Runtime(mode="real", device=device)

    if not _FALLBACK_RUNTIME.is_loaded:
        _FALLBACK_RUNTIME.load()

    if not _FALLBACK_RUNTIME.is_loaded:
        raise RuntimeError("Fallback SD35Runtime failed to load in real mode.")

    return _FALLBACK_RUNTIME


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _preset(profile: str) -> Dict[str, Any]:
    p = PROFILES.get(profile)
    if not p:
        raise ValueError(f"Unknown profile '{profile}'. Allowed: {list(PROFILES.keys())}")
    return dict(p)


def _job_folder_from_payload(payload: Dict[str, Any]) -> str:
    """
    payload.job_folder must be an ABSOLUTE path under the planner/GPU
    contract. We do not accept relative paths here.
    """
    job_folder = payload.get("job_folder")
    if not job_folder or not isinstance(job_folder, str) or not os.path.isabs(job_folder):
        raise RuntimeError("payload.job_folder must be an ABSOLUTE path (provided by planner/dispatch).")
    if not os.path.isdir(job_folder):
        raise RuntimeError(f"job_folder does not exist on GPU worker: {job_folder}")
    return job_folder


def _merge_profile_defaults(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge profile defaults into the execution meta only when fields
    are missing. Planner values always win.

    IMPORTANT:
    - Planner remains source of truth.
    - These are compatibility fallbacks only.
    - Img2img/sketch aspect-ratio preservation flags from planner must survive unchanged.
    """
    merged = dict(payload)
    profile = str(payload.get("profile") or "r1_wide_hero").strip()
    p = _preset(profile)

    merged.setdefault("guidance_scale", p["cfg"])
    merged.setdefault("num_inference_steps", p["steps"])
    merged.setdefault("width", p["width"])
    merged.setdefault("height", p["height"])

    if "upscale" not in merged and isinstance(p.get("upscale"), dict):
        merged["upscale"] = dict(p["upscale"])

    merged.setdefault(
        "preset_resolution",
        {
            "width": int(p["width"]),
            "height": int(p["height"]),
            "source": "compat_profile_default",
        },
    )
    merged.setdefault("resolution_policy", "compat_profile_default")

    return merged


def _normalize_input_image(job_folder: str, payload: Dict[str, Any]) -> str:
    """
    Ensure img2img/sketch input_image resolves correctly.
    """
    inp = payload.get("input_image")
    if not inp:
        raise ValueError("Missing 'input_image' in payload.")

    inp_path = Path(str(inp))
    if not inp_path.is_absolute():
        inp_path = Path(job_folder) / inp_path

    if not inp_path.exists():
        raise FileNotFoundError(f"input_image not found: {inp_path}")

    return str(inp_path)


def _read_image_size(path: str) -> Tuple[int, int]:
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"PIL is required to inspect input size: {exc}") from exc

    try:
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception as exc:
        raise RuntimeError(f"Failed reading input size from {path}: {exc}") from exc


def _build_runtime_meta_for_text2img(job_folder: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    meta = _merge_profile_defaults(payload)

    prompt = str(meta.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Missing 'prompt' for sd35_txt2img")

    meta["type"] = "text2img"
    meta["job_folder"] = job_folder
    meta["prompt"] = prompt

    negative = str(meta.get("negative_prompt") or "").strip()
    if negative:
        meta["negative_prompt"] = negative

    return meta


def _build_runtime_meta_for_img2img(job_folder: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build runtime meta for img2img while preserving planner intent.

    CRITICAL:
    - If planner says preserve_input_aspect_ratio=True and explicit_dimensions=False,
      runtime should auto-follow source aspect ratio.
    - If planner supplied explicit width/height, runtime should respect them.
    - Compatibility layer should not silently convert img2img back to forced square.
    """
    meta = _merge_profile_defaults(payload)

    prompt = str(meta.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Missing 'prompt' for sd35_img2img")

    input_path = _normalize_input_image(job_folder, meta)
    input_w, input_h = _read_image_size(input_path)

    meta["type"] = "img2img"
    meta["job_folder"] = job_folder
    meta["prompt"] = prompt
    meta["input_image"] = input_path

    negative = str(meta.get("negative_prompt") or "").strip()
    if negative:
        meta["negative_prompt"] = negative

    # Support legacy callers that pass denoise instead of strength.
    if "strength" not in meta and "denoise" in meta:
        try:
            meta["strength"] = float(meta["denoise"])
        except Exception:
            pass

    meta.setdefault("input_width", int(input_w))
    meta.setdefault("input_height", int(input_h))
    meta.setdefault(
        "input_aspect_ratio",
        float(input_w) / float(input_h) if input_h else None,
    )

    meta.setdefault("explicit_dimensions", False)
    meta.setdefault("preserve_input_aspect_ratio", not bool(meta.get("explicit_dimensions", False)))

    return meta


def _build_runtime_meta_for_sketch_controlnet(job_folder: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build runtime meta for the dedicated sketch route.

    DESIGN LOCK:
    - This is NOT plain img2img.
    - Structure comes from dual ControlNet conditioning:
        sketch -> cleanup -> canny + depth -> SD3.5 Large
    - Prompt should mainly carry materials / lighting / realism.
    """
    meta = _merge_profile_defaults(payload)

    prompt = str(meta.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Missing 'prompt' for sd35_sketch_controlnet")

    input_path = _normalize_input_image(job_folder, meta)
    input_w, input_h = _read_image_size(input_path)

    meta["type"] = "controlnet"
    meta["job_folder"] = job_folder
    meta["prompt"] = prompt
    meta["input_image"] = input_path
    meta["sketch_image"] = input_path

    negative = str(meta.get("negative_prompt") or "").strip()
    if negative:
        meta["negative_prompt"] = negative

    meta.setdefault("input_width", int(input_w))
    meta.setdefault("input_height", int(input_h))
    meta.setdefault(
        "input_aspect_ratio",
        float(input_w) / float(input_h) if input_h else None,
    )

    meta.setdefault("explicit_dimensions", False)
    meta.setdefault("preserve_input_aspect_ratio", not bool(meta.get("explicit_dimensions", False)))

    controlnet_cfg = meta.get("controlnet")
    if not isinstance(controlnet_cfg, dict):
        raise ValueError("sd35_sketch_controlnet requires a valid 'controlnet' config object.")

    if not bool(controlnet_cfg.get("enabled")):
        raise ValueError("sd35_sketch_controlnet requires controlnet.enabled = true.")

    controls = controlnet_cfg.get("controls")
    if not isinstance(controls, list) or len(controls) < 2:
        raise ValueError("sd35_sketch_controlnet requires at least two controls: canny and depth.")

    seen = {
        str(c.get("control_type", "")).strip().lower()
        for c in controls
        if isinstance(c, dict)
    }
    if "canny" not in seen or "depth" not in seen:
        raise ValueError("sd35_sketch_controlnet requires both canny and depth controls.")

    meta.setdefault("planned_output_image", "output.png")

    outputs = meta.get("outputs")
    if not isinstance(outputs, dict):
        outputs = {}
        meta["outputs"] = outputs

    outputs.setdefault("sketch_image", "sketch.png")
    outputs.setdefault("canny_image", "canny.png")
    outputs.setdefault("depth_image", "depth.png")
    outputs.setdefault("final_image", "output.png")

    optional_polish = meta.get("optional_polish")
    if not isinstance(optional_polish, dict):
        optional_polish = {"enabled": False, "type": "none"}
        meta["optional_polish"] = optional_polish

    return meta


# -------------------------------------------------------------------
# Public execution functions
# -------------------------------------------------------------------

def run_sd35_txt2img(job: Any, payload: Dict[str, Any]) -> str:
    """
    DISPATCH CONTRACT:
      - payload.job_folder (ABSOLUTE) is the target directory
      - returns a STRING path to a REAL PNG inside job_folder

    NOTE:
      We do NOT write meta.json here. Dispatch/worker orchestration owns meta writing.
    """
    job_folder = _job_folder_from_payload(payload)
    runtime = _get_runtime()
    meta = _build_runtime_meta_for_text2img(job_folder, payload)

    result_meta = runtime.generate_text2img(job_folder, meta)
    out_name = str(result_meta.get("output_image") or "output.png")
    return os.path.join(job_folder, out_name)


def run_sd35_img2img(job: Any, payload: Dict[str, Any]) -> str:
    """
    DISPATCH CONTRACT:
      - payload.job_folder (ABSOLUTE) is the target directory
      - payload.input_image must exist
      - returns a STRING path to a REAL PNG inside job_folder
    """
    job_folder = _job_folder_from_payload(payload)
    runtime = _get_runtime()
    meta = _build_runtime_meta_for_img2img(job_folder, payload)

    result_meta = runtime.generate_img2img(job_folder, meta)
    out_name = str(result_meta.get("output_image") or "output.png")
    return os.path.join(job_folder, out_name)


def run_sd35_sketch_controlnet(job: Any, payload: Dict[str, Any]) -> Dict[str, str]:
    """
    DISPATCH CONTRACT:
      - payload.job_folder (ABSOLUTE) is the target directory
      - payload.input_image / sketch.png must exist
      - returns a DICT with real artifact paths:
            {
                "canny_png": ".../canny.png",
                "depth_png": ".../depth.png",
                "output_png": ".../output.png",
                optional "final_up2x_png": ".../final_up2x.png"
            }

    IMPORTANT:
      - This expects SD35Runtime to implement:
            generate_sketch_controlnet(job_folder, meta) -> dict
      - That runtime method must:
            1) create sketch cleanup artifacts as needed
            2) create canny.png
            3) create depth.png
            4) run SD3.5 Large with dual ControlNet
            5) save output.png
    """
    job_folder = _job_folder_from_payload(payload)
    runtime = _get_runtime()
    meta = _build_runtime_meta_for_sketch_controlnet(job_folder, payload)

    result_meta = runtime.generate_sketch_controlnet(job_folder, meta)

    canny_name = str(result_meta.get("canny_image") or "canny.png")
    depth_name = str(result_meta.get("depth_image") or "depth.png")
    output_name = str(result_meta.get("output_image") or "output.png")

    result: Dict[str, str] = {
        "canny_png": os.path.join(job_folder, canny_name),
        "depth_png": os.path.join(job_folder, depth_name),
        "output_png": os.path.join(job_folder, output_name),
    }

    up2x_name = result_meta.get("final_up2x_image")
    if isinstance(up2x_name, str) and up2x_name.strip():
        result["final_up2x_png"] = os.path.join(job_folder, up2x_name)

    return result