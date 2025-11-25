# app/routers/jobs.py

import os
import json
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException

router = APIRouter(
    prefix="/api/jobs",
    tags=["Jobs"],
)


def _build_job_folder(base_outputs_dir: str, date_str: str, job_id: str) -> str:
    """
    Helper to build the path to a specific job folder.
    Example: outputs/2025-11-23/<job_id>/
    """
    return os.path.join(base_outputs_dir, date_str, job_id)


def _load_meta(meta_path: str) -> Dict[str, Any]:
    """
    Load a meta.json file, raising HTTPException if something is wrong.
    """
    if not os.path.isfile(meta_path):
        raise HTTPException(status_code=404, detail=f"meta.json not found at {meta_path}")

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Failed to parse JSON in {meta_path}")

    return meta


@router.get("/{date_str}/{job_id}")
async def get_job(date_str: str, job_id: str):
    """
    Get full meta.json for a single job.

    Example:
    GET /api/jobs/2025-11-23/abcd1234...
    """
    base_outputs_dir = "outputs"
    job_folder = _build_job_folder(base_outputs_dir, date_str, job_id)

    if not os.path.isdir(job_folder):
        raise HTTPException(status_code=404, detail=f"Job folder not found: {job_folder}")

    meta_path = os.path.join(job_folder, "meta.json")
    meta = _load_meta(meta_path)

    return {
        "status": "ok",
        "job_folder": job_folder,
        "meta": meta,
    }


@router.get("/{date_str}")
async def list_jobs_for_date(date_str: str):
    """
    List all jobs for a given date (YYYY-MM-DD).

    Example:
    GET /api/jobs/2025-11-23

    Returns a summary list for each job, based on its meta.json.
    If no jobs exist for that date, returns an empty list with status=ok.
    """
    base_outputs_dir = "outputs"
    date_folder = os.path.join(base_outputs_dir, date_str)

    # If the folder doesn't exist, just return empty jobs list (no 404).
    if not os.path.isdir(date_folder):
        return {
            "status": "ok",
            "date": date_str,
            "jobs": [],
        }

    jobs: List[Dict[str, Any]] = []

    # Each subfolder in date_folder is assumed to be a job_id
    for name in os.listdir(date_folder):
        job_folder = os.path.join(date_folder, name)
        if not os.path.isdir(job_folder):
            continue

        meta_path = os.path.join(job_folder, "meta.json")
        if not os.path.isfile(meta_path):
            # Skip folders without meta.json
            continue

        try:
            meta = _load_meta(meta_path)
        except HTTPException:
            # If one job is broken, skip it instead of killing the whole list.
            continue

        # Build a *summary* for listing (no need to send every detail)
        summary: Dict[str, Any] = {
            "job_id": meta.get("job_id", name),
            "type": meta.get("type", "unknown"),
            "model_name": meta.get("model_name", None),
            "created_at": meta.get("created_at", None),
            "status": meta.get("status", "unknown"),
            "prompt": meta.get("prompt", None),
            "planned_output_image": meta.get("planned_output_image", None),
            "output_image": meta.get("output_image", None),
        }
        jobs.append(summary)

    return {
        "status": "ok",
        "date": date_str,
        "jobs": jobs,
    }
