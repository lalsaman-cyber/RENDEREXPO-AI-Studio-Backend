import json
from typing import Any, Dict, Tuple

import requests

# Local GPU worker URL inside the pod
GPU_BASE_URL = "http://127.0.0.1:8001"


class GPUClientError(Exception):
    """Raised when the GPU worker (8001) fails or is unreachable."""
    pass


def run_sd35_text2img(payload: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Call the SD3.5/SDXL text2img endpoint on GPU worker (8001).

    payload should include:
      - prompt
      - negative_prompt
      - width, height
      - num_inference_steps
      - guidance_scale
      - seed
      - output_path  (where GPU should save output.png)
      - job_folder   (optional)
    """

    url = f"{GPU_BASE_URL}/api/sd35/render"

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
