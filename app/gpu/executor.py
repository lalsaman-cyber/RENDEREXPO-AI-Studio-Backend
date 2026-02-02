from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .job_io import JobRef, init_meta, set_status, update_meta, utc_datestr
from .sd35 import run_sd35_txt2img, run_sd35_img2img
from .upscale import run_upscale_2x


@dataclass(frozen=True)
class ExecResult:
    ok: bool
    job: JobRef
    artifact: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def _parse_job(payload: Dict[str, Any]) -> JobRef:
    job_id = payload.get("job_id")
    if not job_id:
        raise ValueError("dispatch payload missing 'job_id'")
    date = payload.get("date") or utc_datestr()
    return JobRef(date=str(date), job_id=str(job_id))


def _defaults_locked(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforce your locked multipliers unless explicitly set internally.
    Clients do NOT control these; planner sets profile + prompt only.
    """
    out = dict(payload)

    # locked defaults: PRO 2.1 + GEO
    out.setdefault("pro_weight", 0.05)
    out.setdefault("geo_weight", 0.010)

    # The ckpt paths should be set in env on POD (or planner passes them).
    # We do not guess file paths.
    # Recommended env vars:
    #   RENDEREXPO_PRO_CKPT, RENDEREXPO_GEO_CKPT
    out.setdefault("pro_ckpt", None)
    out.setdefault("geo_ckpt", None)

    return out


def execute_dispatch(payload: Dict[str, Any]) -> ExecResult:
    """
    payload must contain at least:
      - job_id
      - task: 'sd35_txt2img' | 'sd35_img2img' | 'upscale_2x'
      - profile: 'r1_wide_hero' | 'r1_close_detail' | 'luxury_interior_heavy_detail'
      - prompt (for sd35 tasks)
    """
    job = _parse_job(payload)
    payload = _defaults_locked(payload)

    init_meta(job, payload)
    set_status(job, "running")

    try:
        task = str(payload.get("task") or "").strip()
        if not task:
            raise ValueError("dispatch payload missing 'task'")

        if task == "sd35_txt2img":
            artifact = run_sd35_txt2img(job, payload)
        elif task == "sd35_img2img":
            artifact = run_sd35_img2img(job, payload)
        elif task == "upscale_2x":
            artifact = run_upscale_2x(job, payload)
        else:
            raise ValueError(f"unknown task '{task}'")

        meta = update_meta(job, status="done", artifact=artifact)
        return ExecResult(ok=True, job=job, artifact=artifact, meta=meta)

    except Exception as e:
        tb = traceback.format_exc(limit=50)
        meta = update_meta(job, status="failed", error=str(e), traceback=tb)
        return ExecResult(ok=False, job=job, error=str(e), meta=meta)
