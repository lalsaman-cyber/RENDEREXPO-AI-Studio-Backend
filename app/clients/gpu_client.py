import hashlib
import hmac
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple

import requests


# -------------------------------------------------------------------
# GPU worker base URL (inside the pod)
# -------------------------------------------------------------------
# Canonical mapping:
#   Planner = 8012
#   GPU worker = 8002
#
# Override with:
#   export GPU_BASE_URL="http://127.0.0.1:8002"
GPU_BASE_URL = os.getenv("GPU_BASE_URL", "http://127.0.0.1:8002").rstrip("/")

HMAC_SECRET_ENV = "RENDEREXPO_HMAC_SECRET"
SIG_HEADER = "X-RENDEREXPO-SIGNATURE"
TS_HEADER = "X-RENDEREXPO-TIMESTAMP"
NONCE_HEADER = "X-RENDEREXPO-NONCE"


class GPUClientError(Exception):
    """Raised when the GPU worker fails or is unreachable."""
    pass


def _get_hmac_secret() -> str:
    secret = os.getenv(HMAC_SECRET_ENV, "").strip()
    if len(secret) < 32:
        raise GPUClientError(
            f"Missing or too-short {HMAC_SECRET_ENV}. "
            "Planner cannot dispatch to GPU worker without HMAC."
        )
    return secret


def _compute_signature(secret: str, timestamp: str, nonce: str, body: bytes) -> str:
    """
    Compute hex HMAC-SHA256 over:
        f"{timestamp}\\n{nonce}\\n".encode("utf-8") + body
    """
    prefix = f"{timestamp}\n{nonce}\n".encode("utf-8")
    msg = prefix + (body or b"")
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def _build_hmac_headers(body: bytes) -> Dict[str, str]:
    secret = _get_hmac_secret()
    timestamp = str(int(time.time()))
    nonce = uuid.uuid4().hex
    signature = _compute_signature(secret=secret, timestamp=timestamp, nonce=nonce, body=body)

    return {
        SIG_HEADER: signature,
        TS_HEADER: timestamp,
        NONCE_HEADER: nonce,
        "Content-Type": "application/json",
    }


def _normalize_job_folder(job_folder: str) -> str:
    """
    Ensure job_folder is an absolute path so the GPU runtime
    can always resolve it correctly.

    IMPORTANT:
    - All live work must resolve relative paths from the volume-backed repo:
      /workspace-data/RENDEREXPO-AI-Studio-Backend
    """
    p = Path(job_folder)
    if p.is_absolute():
        return str(p)

    repo_root = Path("/workspace-data/RENDEREXPO-AI-Studio-Backend")
    return str((repo_root / p).resolve())


def _dispatch(job_folder: str, meta: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Shared dispatcher to GPU worker.
    Contract: POST /api/gpu/dispatch with { job_folder, meta }

    HMAC is required because the GPU worker endpoint is protected.
    """
    job_folder_abs = _normalize_job_folder(job_folder)
    url = f"{GPU_BASE_URL}/api/gpu/dispatch"
    payload = {"job_folder": job_folder_abs, "meta": meta}

    try:
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        headers = _build_hmac_headers(body)

        resp = requests.post(
            url,
            data=body,
            headers=headers,
            timeout=600,
        )
    except requests.RequestException as exc:
        return False, {
            "error": "gpu_request_failed",
            "detail": str(exc),
            "url": url,
            "job_folder_sent": job_folder_abs,
        }
    except GPUClientError as exc:
        return False, {
            "error": "gpu_hmac_unavailable",
            "detail": str(exc),
            "url": url,
            "job_folder_sent": job_folder_abs,
        }
    except Exception as exc:  # noqa: BLE001
        return False, {
            "error": "gpu_dispatch_build_failed",
            "detail": str(exc),
            "url": url,
            "job_folder_sent": job_folder_abs,
        }

    if resp.status_code != 200:
        return False, {
            "error": "gpu_status_not_200",
            "status_code": resp.status_code,
            "text": resp.text[:2000],
            "url": url,
            "job_folder_sent": job_folder_abs,
        }

    try:
        data = resp.json()
    except json.JSONDecodeError:
        return False, {
            "error": "gpu_invalid_json",
            "raw_text": resp.text[:2000],
            "job_folder_sent": job_folder_abs,
        }

    return True, data


# -------------------------------------------------------------------
# SD3.5 Dispatchers (routers import these names)
# All of them share the same dispatch contract.
# -------------------------------------------------------------------

def dispatch_sd35_text2img(job_folder: str, meta: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    return _dispatch(job_folder, meta)


def dispatch_sd35_img2img(job_folder: str, meta: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    return _dispatch(job_folder, meta)


def dispatch_sd35_inpaint(job_folder: str, meta: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    return _dispatch(job_folder, meta)


# Moodboard / Space

def dispatch_sd35_moodboard_to_space(job_folder: str, meta: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    return _dispatch(job_folder, meta)


def dispatch_space_to_moodboard(job_folder: str, meta: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    return _dispatch(job_folder, meta)


def dispatch_sd35_apply_moodboard_to_render(job_folder: str, meta: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    return _dispatch(job_folder, meta)


# Safety aliases (older router versions)
def dispatch_sd35_apply_space_to_render(job_folder: str, meta: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    return _dispatch(job_folder, meta)


def dispatch_sd35_space_to_render(job_folder: str, meta: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    return _dispatch(job_folder, meta)