# app/gpu/sdxl_mistoline.py
from __future__ import annotations

from typing import Any, Dict

from app.services.sketch_runner import run_anyline_mistoline_sketch


def run_sdxl_mistoline_sketch(job: Any, payload: Dict[str, Any]) -> Dict[str, str]:
    """
    Compatibility wrapper:
    keep the old dispatch entrypoint name so dispatch.py does not need to change yet.

    Expected payload:
    - job_folder: absolute output folder
    - prompt: positive prompt
    - negative_prompt: optional negative prompt
    - input_image: absolute path to uploaded sketch image
      fallback: <job_folder>/sketch.png

    Returns:
        {
            "output_png": ".../output.png"
        }
    """
    job_folder = str(payload.get("job_folder") or "").strip()
    if not job_folder:
        raise RuntimeError("payload.job_folder is required.")

    input_image_path = str(payload.get("input_image") or "").strip()
    if not input_image_path:
        input_image_path = f"{job_folder}/sketch.png"

    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise RuntimeError("payload.prompt is required.")

    negative_prompt_raw = payload.get("negative_prompt")
    negative_prompt = str(negative_prompt_raw).strip() if negative_prompt_raw else None

    seed_raw = payload.get("seed")
    seed = int(seed_raw) if seed_raw is not None and str(seed_raw).strip() != "" else None

    result = run_anyline_mistoline_sketch(
        input_image_path=input_image_path,
        output_dir=job_folder,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
    )

    outputs = result.get("outputs") or []
    if not outputs:
        raise RuntimeError("ComfyUI sketch run returned no outputs.")

    first_output = outputs[0]

    return {
        "output_png": first_output,
    }