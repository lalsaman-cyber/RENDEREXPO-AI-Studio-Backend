import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import requests


# Local GPU worker URL inside the pod
# Can be overridden with:
#   export GPU_BASE_URL="http://127.0.0.1:8012"
GPU_BASE_URL = os.getenv("GPU_BASE_URL", "http://127.0.0.1:8011")


class GPUClientError(Exception):
    """Raised when the GPU worker fails or is unreachable."""
    pass


def _normalize_job_folder(job_folder: str) -> str:
    """
    Ensure job_folder is an absolute path so the GPU runtime
    can always resolve it correctly.

    Examples:
      "outputs/2026-01-18/abcd" ->
        "/workspace-data/RENDEREXPO-AI-Studio-Backend/outputs/2026-01-18/abcd"

      "/workspace-data/.../outputs/..." ->
        unchanged
    """
    p = Path(job_folder)

    if p.is_absolute():
        return str(p)

    # repo root = app/clients -> app -> repo
    repo_root = Path(__file__).resolve().parents[2]
    return str((repo_root / p).resolve())


def dispatch_sd35_text2img(job_folder: str, meta: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Dispatch an SD3.5 text2img job to the GPU worker.

    Sends POST /api/gpu/dispatch with:
      - job_folder (ABSOLUTE PATH)
      - meta (meta.json contents)

    Returns:
        (ok, data)
    """
    job_folder_abs = _normalize_job_folder(job_folder)

    url = f"{GPU_BASE_URL}/api/gpu/dispatch"
    payload = {
        "job_folder": job_folder_abs,
        "meta": meta,
    }

    try:
        resp = requests.post(url, json=payload, timeout=600)
    except requests.RequestException as exc:
        return False, {
            "error": "gpu_request_failed",
            "detail": str(exc),
            "url": url,
            "job_folder_sent": job_folder_abs,
        }

    if resp.status_code != 200:
        return False, {
            "error": "gpu_status_not_200",
            "status_code": resp.status_code,
            "text": resp.text[:1000],
            "url": url,
            "job_folder_sent": job_folder_abs,
        }

    try:
        data = resp.json()
    except json.JSONDecodeError:
        return False, {
            "error": "gpu_invalid_json",
            "raw_text": resp.text[:1000],
            "job_folder_sent": job_folder_abs,
        }

    return True, data
