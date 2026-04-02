# app/gpu/sdxl_mistoline.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from runtime.sdxl_mistoline_runtime import SDXLMistoLineRuntime

_INJECTED_RUNTIME: Optional[SDXLMistoLineRuntime] = None
_FALLBACK_RUNTIME: Optional[SDXLMistoLineRuntime] = None


def set_runtime(runtime: SDXLMistoLineRuntime) -> None:
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
    requested_mode = _env_str("SDXL_MISTOLINE_RUNTIME_MODE", "lazy").lower()
    run_real = _env_flag("RUN_REAL_SDXL_MISTOLINE", True)
    return run_real and requested_mode in {"lazy", "real"}


def _try_release_sd35_runtime() -> None:
    """
    Best-effort VRAM protection:
    if SD35 is loaded and sketch now needs SDXL+MistoLine,
    unload SD35 first so both families do not co-reside by default.
    """
    try:
        import app.gpu.sd35 as sd35_module

        for attr in ("_INJECTED_RUNTIME", "_FALLBACK_RUNTIME"):
            runtime = getattr(sd35_module, attr, None)
            if runtime is not None and getattr(runtime, "is_loaded", False):
                try:
                    runtime.unload()
                except Exception:
                    pass
                setattr(sd35_module, attr, None)
    except Exception:
        pass


def _get_runtime() -> SDXLMistoLineRuntime:
    global _FALLBACK_RUNTIME

    if _INJECTED_RUNTIME is not None:
        if not _INJECTED_RUNTIME.is_loaded:
            _try_release_sd35_runtime()
            _INJECTED_RUNTIME.load()
        return _INJECTED_RUNTIME

    if not _runtime_enabled():
        raise RuntimeError(
            "SDXL MistoLine runtime is disabled. "
            "Enable RUN_REAL_SDXL_MISTOLINE=1 and set SDXL_MISTOLINE_RUNTIME_MODE to lazy or real."
        )

    if _FALLBACK_RUNTIME is None:
        device = _env_str("SDXL_MISTOLINE_DEVICE", "cuda")
        _FALLBACK_RUNTIME = SDXLMistoLineRuntime(mode="real", device=device)

    if not _FALLBACK_RUNTIME.is_loaded:
        _try_release_sd35_runtime()
        _FALLBACK_RUNTIME.load()

    if not _FALLBACK_RUNTIME.is_loaded:
        raise RuntimeError("Fallback SDXL MistoLine runtime failed to load in real mode.")

    return _FALLBACK_RUNTIME


def _job_folder_from_payload(payload: Dict[str, Any]) -> str:
    job_folder = str(payload.get("job_folder") or "").strip()
    if not job_folder:
        raise RuntimeError("payload.job_folder is required.")
    return job_folder


def _build_runtime_meta_for_sketch(job_folder: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    meta = dict(payload)
    meta["job_folder"] = job_folder
    meta["pipeline_key"] = "sdxl::mistoline_sketch"
    meta["job_type"] = "sdxl_mistoline_sketch"
    meta["engine_family"] = "sdxl"
    meta["engine"] = "sdxl_base_1_0"
    meta["control_model"] = "TheMistoAI/MistoLine"
    meta["planned_output_image"] = "output.png"
    return meta


def run_sdxl_mistoline_sketch(job: Any, payload: Dict[str, Any]) -> Dict[str, str]:
    """
    GPU dispatch contract:
    - payload.job_folder is the absolute target directory
    - payload.input_image / sketch.png must exist
    - returns real artifact paths:
        {
            "control_png": ".../mistoline_control.png",
            "output_png": ".../output.png",
            optional "final_up2x_png": ".../final_up2x.png"
        }
    """
    job_folder = _job_folder_from_payload(payload)
    runtime = _get_runtime()
    meta = _build_runtime_meta_for_sketch(job_folder, payload)
    return runtime.generate_mistoline_sketch(job_folder, meta)