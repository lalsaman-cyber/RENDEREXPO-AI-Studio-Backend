# app/routers/jobs.py

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api/jobs", tags=["Jobs"])

# Strict date folder format: YYYY-MM-DD
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _outputs_dir() -> str:
    """
    Planner-side outputs root.
    Must remain repo-relative so it maps to:
      /workspace-data/RENDEREXPO-AI-Studio-Backend/outputs
    when the service runs from the correct root.
    """
    return "outputs"


def _validate_date_str(date_str: str) -> str:
    ds = (date_str or "").strip()
    if not DATE_RE.match(ds):
        raise HTTPException(status_code=400, detail="date_str must be YYYY-MM-DD")
    return ds


def _date_dir(date_str: str) -> str:
    ds = _validate_date_str(date_str)
    return os.path.join(_outputs_dir(), ds)


def _safe_load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _job_meta_path(job_folder: str) -> str:
    return os.path.join(job_folder, "meta.json")


def _resolve_job_folder(date_str: str, job_id: str) -> str:
    """
    Resolve a job folder from (date_str, job_id).

    Convention:
      outputs/{date_str}/{job_id}/

    Security:
    - date_str is validated YYYY-MM-DD
    - job_id is basename-only (prevents traversal)
    """
    ds = _validate_date_str(date_str)
    jid = os.path.basename((job_id or "").strip())
    if not jid:
        raise HTTPException(status_code=400, detail="job_id is required")

    base_dir = os.path.join(_outputs_dir(), ds)
    canonical = os.path.join(base_dir, jid)

    if os.path.isdir(canonical):
        return canonical

    if not os.path.isdir(base_dir):
        raise HTTPException(status_code=404, detail="Job folder not found.")

    for name in os.listdir(base_dir):
        if name == jid:
            p = os.path.join(base_dir, name)
            if os.path.isdir(p):
                return p

    raise HTTPException(status_code=404, detail="Job folder not found.")


def _find_job_folder_by_id(job_id: str) -> Optional[Tuple[str, str]]:
    """
    Find outputs/<date_str>/<job_id> by scanning outputs/*/<job_id>.

    Returns:
      (date_str, job_folder) or None
    """
    jid = os.path.basename((job_id or "").strip())
    if not jid:
        return None

    outputs_dir = _outputs_dir()
    if not os.path.isdir(outputs_dir):
        return None

    for date_str in sorted(os.listdir(outputs_dir), reverse=True):
        if not DATE_RE.match(date_str):
            continue
        date_path = os.path.join(outputs_dir, date_str)
        if not os.path.isdir(date_path):
            continue
        candidate = os.path.join(date_path, jid)
        if os.path.isdir(candidate):
            return date_str, candidate

    return None


def _list_job_files(job_folder: str, max_files: int) -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    for root, _, filenames in os.walk(job_folder):
        for fn in filenames:
            rel = os.path.relpath(os.path.join(root, fn), job_folder).replace("\\", "/")
            full = os.path.join(root, fn)
            try:
                st = os.stat(full)
                files.append({"path": rel, "bytes": st.st_size})
            except Exception:
                files.append({"path": rel, "bytes": None})

            if len(files) >= max_files:
                break
        if len(files) >= max_files:
            break

    return sorted(files, key=lambda x: x["path"])


def _serve_image_from_job(job_folder: str, name: str):
    """
    Serve PNG/JPG ONLY (locked).
    Prevent path traversal via basename.
    """
    safe_name = os.path.basename((name or "").strip() or "output.png")
    image_path = os.path.join(job_folder, safe_name)

    if not os.path.isfile(image_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {safe_name}")

    lower = safe_name.lower()
    if lower.endswith(".png"):
        media = "image/png"
    elif lower.endswith(".jpg") or lower.endswith(".jpeg"):
        media = "image/jpeg"
    else:
        raise HTTPException(status_code=400, detail="Only PNG and JPG are allowed")

    return FileResponse(image_path, media_type=media)


def _public_job_base(date_str: str, job_id: str) -> str:
    # assumes planner mounts outputs/ at /outputs
    return f"/outputs/{date_str}/{job_id}"


def _job_links(date_str: str, job_id: str) -> Dict[str, str]:
    base = _public_job_base(date_str, job_id)
    return {
        "meta_url": f"{base}/meta.json",
        "output_url": f"{base}/output.png",
    }


# ---------------------------------------------------------------------------
# Routes (Wix-friendly job-id-based) - put FIRST to avoid collisions
# ---------------------------------------------------------------------------

@router.get("/by-id/{job_id}")
async def get_job_by_id(job_id: str) -> Dict[str, Any]:
    """
    Wix-friendly: resolve by job_id only (no date needed).
    Returns meta + resolved date_str.
    """
    jid = os.path.basename((job_id or "").strip())
    if not jid:
        raise HTTPException(status_code=400, detail="job_id is required")

    found = _find_job_folder_by_id(jid)
    if not found:
        raise HTTPException(status_code=404, detail="Job folder not found.")

    date_str, job_folder = found
    meta_path = _job_meta_path(job_folder)
    if not os.path.isfile(meta_path):
        raise HTTPException(status_code=404, detail="meta.json not found for this job.")

    meta = _safe_load_json(meta_path)
    return {
        "date": date_str,
        "job_id": jid,
        "meta": meta,
        "links": _job_links(date_str, jid),
    }


@router.get("/by-id/{job_id}/files")
async def list_job_files_by_id(
    job_id: str,
    max_files: int = Query(200, ge=1, le=5000, description="Max files returned"),
) -> Dict[str, Any]:
    """
    Wix-friendly: list files by job_id only.
    """
    jid = os.path.basename((job_id or "").strip())
    if not jid:
        raise HTTPException(status_code=400, detail="job_id is required")

    found = _find_job_folder_by_id(jid)
    if not found:
        raise HTTPException(status_code=404, detail="Job folder not found.")

    date_str, job_folder = found
    files_sorted = _list_job_files(job_folder, max_files=max_files)

    return {
        "date": date_str,
        "job_id": jid,
        "file_count": len(files_sorted),
        "files": files_sorted,
        "links": _job_links(date_str, jid),
    }


@router.get("/by-id/{job_id}/image")
async def get_job_image_by_id(
    job_id: str,
    name: str = Query("output.png", description="File name inside the job folder (PNG/JPG only)."),
):
    """
    Wix-friendly: serve PNG/JPG only by job_id.
    """
    jid = os.path.basename((job_id or "").strip())
    if not jid:
        raise HTTPException(status_code=400, detail="job_id is required")

    found = _find_job_folder_by_id(jid)
    if not found:
        raise HTTPException(status_code=404, detail="Job folder not found.")

    _, job_folder = found
    return _serve_image_from_job(job_folder, name=name)


# ---------------------------------------------------------------------------
# Routes (existing date-based)
# ---------------------------------------------------------------------------

@router.get("/{date_str}")
async def list_jobs_for_date(
    date_str: str = Path(..., description="YYYY-MM-DD"),
) -> Dict[str, Any]:
    """
    List job IDs for a given date (YYYY-MM-DD).
    Returns minimal meta preview if meta.json exists.
    """
    ds = _validate_date_str(date_str)
    base_dir = _date_dir(ds)
    if not os.path.isdir(base_dir):
        return {"date": ds, "jobs": []}

    jobs: List[Dict[str, Any]] = []
    for job_id in sorted(os.listdir(base_dir)):
        job_folder = os.path.join(base_dir, job_id)
        if not os.path.isdir(job_folder):
            continue

        meta_path = _job_meta_path(job_folder)
        meta: Dict[str, Any] = _safe_load_json(meta_path) if os.path.isfile(meta_path) else {}

        meta_preview = {
            "type": meta.get("type") or meta.get("job_type"),
            "model_name": meta.get("model_name"),
            "created_at": meta.get("created_at"),
            "status": meta.get("status"),
            "category": meta.get("category"),
            "shot": meta.get("shot"),
            "pipeline_key": meta.get("pipeline_key"),
        }

        jobs.append(
            {
                "job_id": job_id,
                "meta_preview": meta_preview,
                "links": _job_links(ds, job_id),
            }
        )

    return {"date": ds, "jobs": jobs}


@router.get("/{date_str}/{job_id}")
async def get_job(
    date_str: str = Path(..., description="YYYY-MM-DD"),
    job_id: str = Path(..., description="Job folder name"),
) -> Dict[str, Any]:
    """
    Return meta.json for a specified job (date-based).
    """
    ds = _validate_date_str(date_str)
    jid = os.path.basename((job_id or "").strip())
    if not jid:
        raise HTTPException(status_code=400, detail="job_id is required")

    job_folder = _resolve_job_folder(ds, jid)
    meta_path = _job_meta_path(job_folder)
    if not os.path.isfile(meta_path):
        raise HTTPException(status_code=404, detail="meta.json not found for this job.")

    meta = _safe_load_json(meta_path)
    return {"date": ds, "job_id": jid, "meta": meta, "links": _job_links(ds, jid)}


@router.get("/{date_str}/{job_id}/files")
async def list_job_files(
    date_str: str,
    job_id: str,
    max_files: int = Query(200, ge=1, le=5000, description="Max files returned"),
) -> Dict[str, Any]:
    """
    List files inside a job folder (date-based).
    """
    ds = _validate_date_str(date_str)
    jid = os.path.basename((job_id or "").strip())
    if not jid:
        raise HTTPException(status_code=400, detail="job_id is required")

    job_folder = _resolve_job_folder(ds, jid)
    files_sorted = _list_job_files(job_folder, max_files=max_files)

    return {
        "date": ds,
        "job_id": jid,
        "file_count": len(files_sorted),
        "files": files_sorted,
        "links": _job_links(ds, jid),
    }


@router.get("/{date_str}/{job_id}/image")
async def get_job_image(
    date_str: str,
    job_id: str,
    name: str = Query("output.png", description="File name inside the job folder (PNG/JPG only)."),
):
    """
    Serve PNG/JPG only (date-based).
    """
    ds = _validate_date_str(date_str)
    jid = os.path.basename((job_id or "").strip())
    if not jid:
        raise HTTPException(status_code=400, detail="job_id is required")

    job_folder = _resolve_job_folder(ds, jid)
    return _serve_image_from_job(job_folder, name=name)