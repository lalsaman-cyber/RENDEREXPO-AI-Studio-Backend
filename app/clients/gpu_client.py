# app/clients/gpu_client.py
"""
RENDEREXPO AI STUDIO - Planner -> GPU Worker client

Purpose:
- Planner (8012) dispatches prepared jobs to the GPU worker (8002)
- This file does NOT modify generation semantics
- It forwards meta exactly as planner prepared it

IMPORTANT:
- Img2img aspect-ratio behavior must NOT be destroyed here
- If planner/runtime include:
    * preserve_input_aspect_ratio
    * explicit_dimensions
    * input_width / input_height
    * preset_resolution / resolution_policy
  those fields must pass through unchanged
- This client signs and forwards payloads only

NEW SKETCH RULE:
- Sketch is no longer plain img2img as the primary architecture.
- Dedicated sketch jobs must dispatch with:
    job_type = "sd35_sketch_controlnet"
    pipeline_key = "sd35::sd35_sketch_controlnet"
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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

DEFAULT_TIMEOUT_SECONDS = int(os.getenv("GPU_CLIENT_TIMEOUT_SECONDS", "600"))


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


def _resolve_job_type(meta: Dict[str, Any]) -> str:
    """
    Resolve GPU dispatch job_type from planner metadata.

    Priority:
    1) explicit meta.job_type
    2) explicit meta.type
    3) inferred pipeline key fallback

    IMPORTANT:
    - This function does NOT rewrite prompt/settings.
    - It only decides the dispatch label for the GPU worker router.
    """
    raw_job_type = meta.get("job_type")
    if isinstance(raw_job_type, str) and raw_job_type.strip():
        return raw_job_type.strip()

    raw_type = meta.get("type")
    if isinstance(raw_type, str) and raw_type.strip():
        t = raw_type.strip().lower()

        # Planner-side types sometimes differ from GPU dispatch labels
        if t == "text2img":
            return "sd35_text2img"
        if t == "img2img":
            return "sd35_img2img"
        if t == "inpaint":
            # Current GPU path shares img2img-style execution contract
            return "sd35_img2img"
        if t == "controlnet":
            pipeline_key = str(meta.get("pipeline_key") or "").strip().lower()
            if pipeline_key == "sd35::sd35_sketch_controlnet":
                return "sd35_sketch_controlnet"

    pipeline_key = str(meta.get("pipeline_key") or "").strip().lower()
    if pipeline_key == "sd35::text2img":
        return "sd35_text2img"
    if pipeline_key == "sd35::img2img":
        return "sd35_img2img"
    if pipeline_key == "sd35::sd35_sketch_controlnet":
        return "sd35_sketch_controlnet"
    if pipeline_key == "upscale::2x":
        return "upscale_2x"
    if pipeline_key == "video::from_image":
        return "video_from_image"
    if pipeline_key == "video::between_frames":
        return "video_between_frames"
    if pipeline_key == "cad::from_image":
        return "cad_from_image"
    if pipeline_key == "mesh::from_image":
        return "mesh_from_image"

    raise GPUClientError(
        "Unable to resolve GPU dispatch job_type from planner meta. "
        "Expected meta.job_type, meta.type, or known pipeline_key."
    )


def _resolve_pipeline_key(meta: Dict[str, Any]) -> Optional[str]:
    raw = meta.get("pipeline_key")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _dispatch(job_folder: str, meta: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Shared dispatcher to GPU worker.
    Contract: POST /api/gpu/dispatch with:
        {
          "job_type": ...,
          "job_folder": ...,
          "meta": ...,
          "pipeline_key": ...
        }

    HMAC is required because the GPU worker endpoint is protected.

    CRITICAL:
    - meta must be forwarded as-is
    - do NOT strip or rewrite img2img/sketch geometry/aspect keys here
    """
    job_folder_abs = _normalize_job_folder(job_folder)
    url = f"{GPU_BASE_URL}/api/gpu/dispatch"

    try:
        job_type = _resolve_job_type(meta)
        pipeline_key = _resolve_pipeline_key(meta)
    except GPUClientError as exc:
        return False, {
            "error": "gpu_dispatch_type_resolution_failed",
            "detail": str(exc),
            "url": url,
            "job_folder_sent": job_folder_abs,
        }

    # Forward metadata exactly as planner/runtime prepared it.
    payload: Dict[str, Any] = {
        "job_type": job_type,
        "job_folder": job_folder_abs,
        "meta": meta,
    }
    if pipeline_key:
        payload["pipeline_key"] = pipeline_key

    try:
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        headers = _build_hmac_headers(body)

        resp = requests.post(
            url,
            data=body,
            headers=headers,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        return False, {
            "error": "gpu_request_failed",
            "detail": str(exc),
            "url": url,
            "job_folder_sent": job_folder_abs,
            "job_type_sent": job_type,
            "pipeline_key_sent": pipeline_key,
        }
    except GPUClientError as exc:
        return False, {
            "error": "gpu_hmac_unavailable",
            "detail": str(exc),
            "url": url,
            "job_folder_sent": job_folder_abs,
            "job_type_sent": job_type,
            "pipeline_key_sent": pipeline_key,
        }
    except Exception as exc:  # noqa: BLE001
        return False, {
            "error": "gpu_dispatch_build_failed",
            "detail": str(exc),
            "url": url,
            "job_folder_sent": job_folder_abs,
            "job_type_sent": job_type,
            "pipeline_key_sent": pipeline_key,
        }

    if resp.status_code != 200:
        return False, {
            "error": "gpu_status_not_200",
            "status_code": resp.status_code,
            "text": resp.text[:2000],
            "url": url,
            "job_folder_sent": job_folder_abs,
            "job_type_sent": job_type,
            "pipeline_key_sent": pipeline_key,
        }

    try:
        data = resp.json()
    except json.JSONDecodeError:
        return False, {
            "error": "gpu_invalid_json",
            "raw_text": resp.text[:2000],
            "job_folder_sent": job_folder_abs,
            "job_type_sent": job_type,
            "pipeline_key_sent": pipeline_key,
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


def dispatch_sd35_sketch_controlnet(job_folder: str, meta: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    return _dispatch(job_folder, meta)


# -------------------------------------------------------------------
# Moodboard / Space
# -------------------------------------------------------------------

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


# -------------------------------------------------------------------
# Object / product insertion
# -------------------------------------------------------------------

def dispatch_insert_object(job_folder: str, meta: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    return _dispatch(job_folder, meta)


def dispatch_product_insert(job_folder: str, meta: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    return _dispatch(job_folder, meta)


# -------------------------------------------------------------------
# VR / video / CAD / mesh aliases for future-safe router imports
# -------------------------------------------------------------------

def dispatch_vr_reconstruct(job_folder: str, meta: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    return _dispatch(job_folder, meta)


def dispatch_video_from_image(job_folder: str, meta: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    return _dispatch(job_folder, meta)


def dispatch_video_between_frames(job_folder: str, meta: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    return _dispatch(job_folder, meta)


def dispatch_cad_from_image(job_folder: str, meta: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    return _dispatch(job_folder, meta)


def dispatch_mesh_from_image(job_folder: str, meta: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    return _dispatch(job_folder, meta)