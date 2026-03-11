from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from runtime.sd35_runtime import SD35Runtime

# -------------------------------------------------------------------
# Locked profiles (Doc 19+)
# -------------------------------------------------------------------
# These are compatibility defaults only.
# Planner/routers remain the real source of truth and should write
# steps / cfg / width / height / upscale into meta/payload explicitly.
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

    return merged


def _normalize_input_image(job_folder: str, payload: Dict[str, Any]) -> str:
    """
    Ensure img2img input_image resolves correctly.
    """
    inp = payload.get("input_image")
    if not inp:
        raise ValueError("Missing 'input_image' for sd35_img2img")

    inp_path = Path(str(inp))
    if not inp_path.is_absolute():
        inp_path = Path(job_folder) / inp_path

    if not inp_path.exists():
        raise FileNotFoundError(f"input_image not found: {inp_path}")

    return str(inp_path)


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
    meta = _merge_profile_defaults(payload)

    prompt = str(meta.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Missing 'prompt' for sd35_img2img")

    input_path = _normalize_input_image(job_folder, meta)

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