# app/routers/jobs.py

import os
import json
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api/jobs", tags=["Jobs"])


def _date_dir(date_str: str) -> str:
    return os.path.join("outputs", date_str)


@router.get("/{date_str}")
async def list_jobs_for_date(date_str: str) -> Dict[str, Any]:
    """
    List job IDs for a given date (YYYY-MM-DD).
    """
    base_dir = _date_dir(date_str)
    if not os.path.isdir(base_dir):
        return {"date": date_str, "jobs": []}

    jobs: List[Dict[str, Any]] = []
    for job_id in sorted(os.listdir(base_dir)):
        job_folder = os.path.join(base_dir, job_id)
        if not os.path.isdir(job_folder):
            continue
        meta_path = os.path.join(job_folder, "meta.json")
        meta = {}
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
        jobs.append(
            {
                "job_id": job_id,
                "job_folder": job_folder,
                "meta": meta,
            }
        )

    return {
        "date": date_str,
        "jobs": jobs,
    }


@router.get("/{date_str}/{job_id}")
async def get_job(date_str: str, job_id: str) -> Dict[str, Any]:
    """
    Return the meta.json for the specified job.
    """
    base_dir = _date_dir(date_str)
    job_folder = os.path.join(base_dir, job_id)

    if not os.path.isdir(job_folder):
        raise HTTPException(status_code=404, detail="Job folder not found.")

    meta_path = os.path.join(job_folder, "meta.json")
    if not os.path.isfile(meta_path):
        raise HTTPException(status_code=404, detail="meta.json not found for this job.")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return {
        "date": date_str,
        "job_id": job_id,
        "job_folder": job_folder,
        "meta": meta,
    }


@router.get("/{date_str}/{job_id}/image")
async def get_job_image(date_str: str, job_id: str):
    """
    Return the rendered image (output.png) for a given job, if it exists.

    Example:
        GET /api/jobs/2025-11-26/f61a24fa257a4c32be3714bc0d736f2f/image
    """
    base_dir = _date_dir(date_str)
    job_folder = os.path.join(base_dir, job_id)
    image_path = os.path.join(job_folder, "output.png")

    if not os.path.isdir(job_folder):
        raise HTTPException(status_code=404, detail="Job folder not found.")

    if not os.path.isfile(image_path):
        raise HTTPException(status_code=404, detail="Image not found for this job.")

    return FileResponse(image_path, media_type="image/png")
