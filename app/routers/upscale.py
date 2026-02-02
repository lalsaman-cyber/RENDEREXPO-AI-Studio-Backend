# app/routers/upscale.py

"""
RENDEREXPO AI STUDIO - Upscale (REAL execution + optional planning)

What it does:
- Upscale an existing image inside a job folder (usually output.png) by 2x or 4x
- Deterministic resize ONLY (Pillow Lanczos). NO diffusion, NO denoise.
- Writes meta.json updates:
    meta["upscale_plan"] (optional)
    meta["upscale"] (last run result)
    meta["outputs"]["upscaled_image"] (public-friendly pointer)

Inputs supported:
- job_folder
- OR date_str + job_id
- OR by_id (scan outputs/*/<job_id>)  [Wix-friendly]

Security:
- Prevent path traversal (only basename)
- Only PNG/JPG allowed
"""

from __future__ import annotations

import os
import json
import datetime
from typing import Optional, Dict, Any, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from PIL import Image  # pillow


router = APIRouter(prefix="/api/upscale", tags=["Upscale"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat()


def _meta_path(job_folder: str) -> str:
    return os.path.join(job_folder, "meta.json")


def _read_meta(job_folder: str) -> Dict[str, Any]:
    p = _meta_path(job_folder)
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed reading meta.json: {exc}")


def _write_meta(job_folder: str, meta: Dict[str, Any]) -> None:
    p = _meta_path(job_folder)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed writing meta.json: {exc}")


def _date_dir(date_str: str) -> str:
    return os.path.join("outputs", date_str)


def _find_job_folder_by_id(job_id: str) -> Optional[Tuple[str, str]]:
    """
    Find outputs/<date_str>/<job_id> by scanning outputs/*/<job_id>.
    Returns (date_str, job_folder) or None.
    """
    outputs_dir = "outputs"
    if not os.path.isdir(outputs_dir):
        return None

    # date folders are "YYYY-MM-DD" so lexical sort works
    for date_str in sorted(os.listdir(outputs_dir), reverse=True):
        date_path = os.path.join(outputs_dir, date_str)
        if not os.path.isdir(date_path):
            continue
        candidate = os.path.join(date_path, job_id)
        if os.path.isdir(candidate):
            return date_str, candidate
    return None


def _resolve_job_folder(
    job_folder: Optional[str],
    date_str: Optional[str],
    job_id: Optional[str],
    by_id: Optional[str],
) -> Tuple[str, str, str]:
    """
    Returns (date_str, job_id, job_folder)

    Priority:
    1) job_folder (parse date/job_id if possible)
    2) date_str + job_id
    3) by_id scan
    """
    if job_folder:
        jf = os.path.abspath(job_folder)
        if not os.path.isdir(jf):
            raise HTTPException(status_code=404, detail=f"Job folder not found: {jf}")

        # best-effort parse: outputs/YYYY-MM-DD/<job_id>
        parts = os.path.normpath(jf).split(os.sep)
        if len(parts) >= 3 and parts[-3] == "outputs":
            return parts[-2], parts[-1], jf

        return "", os.path.basename(jf), jf

    if date_str and job_id:
        jf = os.path.abspath(os.path.join("outputs", date_str, job_id))
        if not os.path.isdir(jf):
            raise HTTPException(status_code=404, detail=f"Job folder not found: {jf}")
        return date_str, job_id, jf

    if by_id:
        found = _find_job_folder_by_id(by_id.strip())
        if not found:
            raise HTTPException(status_code=404, detail="Job folder not found by id.")
        d, jf = found
        return d, os.path.basename(jf), os.path.abspath(jf)

    raise HTTPException(status_code=400, detail="Provide job_folder OR (date_str + job_id) OR by_id.")


def _safe_image_name(name: str) -> str:
    safe = os.path.basename((name or "").strip())
    if not safe:
        raise HTTPException(status_code=400, detail="Invalid image name.")
    lower = safe.lower()
    if not (lower.endswith(".png") or lower.endswith(".jpg") or lower.endswith(".jpeg")):
        raise HTTPException(status_code=400, detail="Only PNG/JPG images are supported.")
    return safe


def _open_image(full_path: str) -> Image.Image:
    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail=f"Input image not found: {os.path.basename(full_path)}")
    try:
        img = Image.open(full_path)
        img.load()
        return img
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed opening image: {exc}")


def _save_image(img: Image.Image, out_path: str) -> None:
    try:
        ext = os.path.splitext(out_path)[1].lower()
        if ext == ".png":
            img.save(out_path, format="PNG", optimize=True)
        else:
            # JPEG
            rgb = img.convert("RGB")
            rgb.save(out_path, format="JPEG", quality=95, optimize=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed saving image: {exc}")


def _public_urls(date_str: str, job_id: str, filename: str) -> Dict[str, Optional[str]]:
    if not date_str or not job_id:
        return {"image_url": None, "meta_url": None}
    base = f"/outputs/{date_str}/{job_id}"
    return {"image_url": f"{base}/{filename}", "meta_url": f"{base}/meta.json"}


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class UpscalePlanRequest(BaseModel):
    job_folder: Optional[str] = Field(None, description="Path to outputs/YYYY-MM-DD/<job_id>/")
    date_str: Optional[str] = Field(None, description="YYYY-MM-DD (if job_folder not provided)")
    job_id: Optional[str] = Field(None, description="Job id (if job_folder not provided)")
    by_id: Optional[str] = Field(None, description="Wix-friendly: resolve by job_id only (scan outputs/*/<job_id>)")

    enabled: bool = Field(True, description="If false, writes an explicit disabled plan")
    factor: int = Field(2, ge=2, le=4, description="Upscale factor (2 or 4)")
    method: str = Field("lanczos", description="Deterministic upscale method (no diffusion). Only 'lanczos' supported now.")

    denoise: float = Field(0.0, description="MUST stay 0.0. Any non-zero value will be rejected.")

    input_image: Optional[str] = Field(None, description="Image filename in job folder (default auto-detect)")
    output_image: Optional[str] = Field(None, description="Output filename (default: output_upscaled_2x.png)")


class UpscaleRunRequest(UpscalePlanRequest):
    """
    Same fields as plan, but executes immediately and writes the output image.
    """
    pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/plan")
def plan_upscale(req: UpscalePlanRequest):
    date_str, job_id, job_folder = _resolve_job_folder(req.job_folder, req.date_str, req.job_id, req.by_id)

    if float(req.denoise) != 0.0:
        raise HTTPException(status_code=400, detail="denoise must be 0.0 (NO diffusion upscale).")

    if req.method.lower().strip() != "lanczos":
        raise HTTPException(status_code=400, detail="Only method='lanczos' is supported for deterministic upscale.")

    meta = _read_meta(job_folder)

    # auto input choice
    if req.input_image:
        in_name = _safe_image_name(req.input_image)
    else:
        if os.path.isfile(os.path.join(job_folder, "output.png")):
            in_name = "output.png"
        elif os.path.isfile(os.path.join(job_folder, "output.jpg")):
            in_name = "output.jpg"
        elif os.path.isfile(os.path.join(job_folder, "output.jpeg")):
            in_name = "output.jpeg"
        elif os.path.isfile(os.path.join(job_folder, "input.png")):
            in_name = "input.png"
        else:
            in_name = "output.png"  # planned default

    # default output
    if req.output_image:
        out_name = _safe_image_name(req.output_image)
    else:
        out_name = f"output_upscaled_{int(req.factor)}x.png"

    plan = {
        "enabled": bool(req.enabled),
        "factor": int(req.factor),
        "method": "lanczos",
        "denoise": 0.0,
        "input_image": in_name,
        "output_image": out_name,
        "planned_at": _now_iso(),
        "mode": "plan-only",
        "note": "Deterministic upscale only. NO diffusion. denoise=0.0 hard-locked.",
    }

    meta.setdefault("job_id", os.path.basename(job_folder))
    meta.setdefault("type", meta.get("type", "unknown"))
    meta["upscale_plan"] = plan
    meta["last_updated"] = _now_iso()

    _write_meta(job_folder, meta)

    return {
        "status": "ok",
        "message": "Upscale planned. (No diffusion. denoise=0.0 locked.)",
        "job_folder": job_folder,
        "meta_path": _meta_path(job_folder),
        "upscale_plan": plan,
        "public_urls": _public_urls(date_str, job_id, out_name),
    }


@router.post("/run")
def run_upscale(req: UpscaleRunRequest):
    """
    REAL upscale execution (no GPU required):
    - reads input image from job folder
    - upscales deterministically using Pillow Lanczos
    - writes output image
    - updates meta.json with results
    """
    date_str, job_id, job_folder = _resolve_job_folder(req.job_folder, req.date_str, req.job_id, req.by_id)

    if float(req.denoise) != 0.0:
        raise HTTPException(status_code=400, detail="denoise must be 0.0 (NO diffusion upscale).")

    if req.method.lower().strip() != "lanczos":
        raise HTTPException(status_code=400, detail="Only method='lanczos' is supported for deterministic upscale.")

    if req.enabled is False:
        # still persist a "disabled" run record
        meta = _read_meta(job_folder)
        meta.setdefault("job_id", os.path.basename(job_folder))
        meta.setdefault("type", meta.get("type", "unknown"))
        meta["upscale"] = {
            "enabled": False,
            "ran_at": _now_iso(),
            "denoise": 0.0,
            "note": "Upscale disabled by request.",
        }
        meta["last_updated"] = _now_iso()
        _write_meta(job_folder, meta)

        return {
            "status": "ok",
            "message": "Upscale disabled (no action taken).",
            "job_folder": job_folder,
            "meta_path": _meta_path(job_folder),
            "upscale": meta["upscale"],
        }

    # input
    if req.input_image:
        in_name = _safe_image_name(req.input_image)
    else:
        # best available
        candidates = ["output.png", "output.jpg", "output.jpeg", "input.png", "input.jpg", "input.jpeg"]
        found = None
        for c in candidates:
            if os.path.isfile(os.path.join(job_folder, c)):
                found = c
                break
        in_name = found or "output.png"

    # output
    if req.output_image:
        out_name = _safe_image_name(req.output_image)
    else:
        # keep same extension as input if it was jpg/jpeg, otherwise png
        ext = os.path.splitext(in_name)[1].lower()
        if ext in (".jpg", ".jpeg"):
            out_name = f"output_upscaled_{int(req.factor)}x.jpg"
        else:
            out_name = f"output_upscaled_{int(req.factor)}x.png"

    in_path = os.path.join(job_folder, in_name)
    out_path = os.path.join(job_folder, out_name)

    img = _open_image(in_path)

    w, h = img.size
    factor = int(req.factor)
    new_w = w * factor
    new_h = h * factor

    try:
        up = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    except Exception:
        # backward compat for older pillow
        up = img.resize((new_w, new_h), resample=Image.LANCZOS)

    _save_image(up, out_path)

    # update meta
    meta = _read_meta(job_folder)
    meta.setdefault("job_id", os.path.basename(job_folder))
    meta.setdefault("type", meta.get("type", "unknown"))

    result = {
        "enabled": True,
        "factor": factor,
        "method": "lanczos",
        "denoise": 0.0,
        "input_image": in_name,
        "output_image": out_name,
        "input_size": {"width": w, "height": h},
        "output_size": {"width": new_w, "height": new_h},
        "ran_at": _now_iso(),
        "mode": "executed-local",
        "note": "Deterministic upscale executed (no diffusion).",
    }

    meta["upscale"] = result
    meta.setdefault("outputs", {})
    meta["outputs"]["upscaled_image"] = out_name
    meta["last_updated"] = _now_iso()

    _write_meta(job_folder, meta)

    return {
        "status": "ok",
        "message": "Upscale executed successfully (deterministic, no diffusion).",
        "job_folder": job_folder,
        "meta_path": _meta_path(job_folder),
        "upscale": result,
        "public_urls": _public_urls(date_str, job_id, out_name),
    }


# Optional backward compatibility:
# Old code used POST "" as the plan endpoint. Keep it alive.
@router.post("")
def plan_upscale_compat(req: UpscalePlanRequest):
    return plan_upscale(req)
