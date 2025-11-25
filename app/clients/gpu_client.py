import json
from typing import Any, Dict, Tuple

import requests

# Local GPU worker URL inside the pod
GPU_BASE_URL = "http://127.0.0.1:8001"


class GPUClientError(Exception):
    """Raised when the GPU worker (8001) fails or is unreachable."""
    pass


def dispatch_sd35_text2img(job_folder: str, meta: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Dispatch an SD3.5 text2img job to the GPU worker (port 8001).

    The GPU API expects:
      {
        "job_folder": ".../outputs/YYYY-MM-DD/<job-id>",
        "meta": { ... full meta.json content ... }
      }

    In skeleton mode, the GPU may just update meta + create a dummy PNG.
    In real mode, it will run SD3.5 and write a real output.png.
    """

    url = f"{GPU_BASE_URL}/api/gpu/dispatch"
    payload = {
        "job_folder": job_folder,
        "meta": meta,
    }

    try:
        resp = requests.post(url, json=payload, timeout=600)
    except requests.RequestException as exc:
        return False, {
            "error": "gpu_request_failed",
            "detail": str(exc),
        }

    if resp.status_code != 200:
        return False, {
            "error": "gpu_status_not_200",
            "status_code": resp.status_code,
            "text": resp.text[:1000],
        }

    try:
        data = resp.json()
    except json.JSONDecodeError:
        return False, {
            "error": "gpu_invalid_json",
            "raw_text": resp.text[:1000],
        }

    return True, data
