from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict


def _job_folder_from_payload(payload: Dict[str, Any]) -> str:
    job_folder = payload.get("job_folder")
    if not job_folder or not isinstance(job_folder, str) or not os.path.isabs(job_folder):
        raise RuntimeError("payload.job_folder must be an ABSOLUTE path (provided by planner/dispatch).")
    if not os.path.isdir(job_folder):
        raise RuntimeError(f"job_folder does not exist on GPU worker: {job_folder}")
    return job_folder


def run_upscale_2x(job: Any, payload: Dict[str, Any]) -> str:
    """
    DISPATCH-CONTRACT:
      - payload.job_folder (ABSOLUTE) is the target directory
      - payload.input_image must exist (dispatch already selects one)
      - returns a STRING path to a REAL PNG inside job_folder

    NOTE:
      We do NOT write meta.json here. Dispatch owns meta writing.
    """
    job_folder = _job_folder_from_payload(payload)

    inp = payload.get("input_image")
    if not inp:
        raise ValueError("Missing 'input_image' for upscale_2x")

    inp_path = Path(str(inp))
    if not inp_path.exists():
        raise FileNotFoundError(f"input_image not found for upscale: {inp_path}")

    out_path = Path(job_folder) / "final_up2x.png"

    script = os.getenv("RENDEREXPO_UPSCALE_SCRIPT", "scripts/upscale_realesrgan_cuda.py")
    script_path = Path(script)
    if not script_path.exists():
        raise FileNotFoundError(
            f"Upscale script not found: {script_path}. "
            f"Set RENDEREXPO_UPSCALE_SCRIPT to the correct absolute path on POD."
        )

    env = os.environ.copy()
    env["IMG_IN"] = str(inp_path)
    env["OUT"] = str(out_path)
    env.setdefault("SCALE", "2")
    env.setdefault("TILE", "256")

    subprocess.run(
        ["python3", str(script_path)],
        check=True,
        env=env,
        cwd=str(Path.cwd()),
    )

    if not out_path.exists() or out_path.stat().st_size < 1024:
        raise RuntimeError(f"Upscale did not produce a valid output file: {out_path}")

    return str(out_path)
