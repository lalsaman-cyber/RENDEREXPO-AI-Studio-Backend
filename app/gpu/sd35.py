from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image


# --------------------------
# Locked presets (Doc 19+)
# --------------------------
PROFILES = {
    "r1_wide_hero": {
        "cfg": 5.6,
        "steps": 46,
        "width": 1024,
        "height": 1024,
        "upscale": True,
        "upscale_denoise": 0.25,
    },
    "r1_close_detail": {
        "cfg": 6.0,
        "steps": 48,
        "width": 1024,
        "height": 1024,
        "upscale": False,
        "upscale_denoise": 0.25,
    },
    "luxury_interior_heavy_detail": {
        "cfg": 6.0,
        "steps": 60,
        "width": 1024,
        "height": 1024,
        "upscale": True,
        "upscale_denoise": 0.0,
    },
}


def _device_and_dtype() -> Tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16  # locked default
    return torch.device("cpu"), torch.float32


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip()


def _preset(profile: str) -> Dict[str, Any]:
    p = PROFILES.get(profile)
    if not p:
        raise ValueError(f"Unknown profile '{profile}'. Allowed: {list(PROFILES.keys())}")
    return dict(p)


def _job_folder_from_payload(payload: Dict[str, Any]) -> str:
    job_folder = payload.get("job_folder")
    if not job_folder or not isinstance(job_folder, str) or not os.path.isabs(job_folder):
        raise RuntimeError("payload.job_folder must be an ABSOLUTE path (provided by planner/dispatch).")
    if not os.path.isdir(job_folder):
        raise RuntimeError(f"job_folder does not exist on GPU worker: {job_folder}")
    return job_folder


def _enforce_locked_multipliers(payload: Dict[str, Any]) -> Tuple[float, float, Optional[str], Optional[str]]:
    """
    Respect PRO 2.1 + locked multipliers:
      PRO default 0.05
      GEO default 0.010
    Allow only safe bands to prevent sabotage / instability.
    """
    pro_w = float(payload.get("pro_weight", 0.05))
    geo_w = float(payload.get("geo_weight", 0.010))

    if not (0.0 < pro_w <= 0.20):
        raise RuntimeError(f"Invalid PRO multiplier: {pro_w}. Allowed: (0, 0.20].")
    if not (0.0 <= geo_w <= 0.05):
        raise RuntimeError(f"Invalid GEO multiplier: {geo_w}. Allowed: [0, 0.05].")

    pro_ckpt = payload.get("pro_ckpt") or _env("RENDEREXPO_PRO_CKPT")
    geo_ckpt = payload.get("geo_ckpt") or _env("RENDEREXPO_GEO_CKPT")

    return pro_w, geo_w, pro_ckpt, geo_ckpt


@lru_cache(maxsize=1)
def _load_pipe_txt2img():
    """
    Loads once per GPU worker (8012). Critical for performance.
    """
    model_id = _env("RENDEREXPO_SD35_MODEL_ID")
    if not model_id:
        raise RuntimeError("Missing env RENDEREXPO_SD35_MODEL_ID (SD3.5 model id/path)")

    try:
        from diffusers import StableDiffusion3Pipeline
    except Exception as e:
        raise RuntimeError(f"diffusers StableDiffusion3Pipeline import failed: {e}")

    device, dtype = _device_and_dtype()
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype, variant=None).to(device)

    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    return pipe


def _apply_adapters(pipe, payload: Dict[str, Any]) -> None:
    """
    Apply PRO + GEO adapters (LyCORIS/LoRA) respecting locked multipliers.
    If adapter paths are missing, we still produce REAL SD3.5 outputs.
    """
    pro_w, geo_w, pro_ckpt, geo_ckpt = _enforce_locked_multipliers(payload)

    adapters = []
    weights = []

    if pro_ckpt:
        adapters.append(("pro", pro_ckpt))
        weights.append(pro_w)
    if geo_ckpt and geo_w > 0:
        adapters.append(("geo", geo_ckpt))
        weights.append(geo_w)

    if not adapters:
        return

    for name, ckpt in adapters:
        try:
            pipe.load_lora_weights(ckpt, adapter_name=name)
        except Exception as e:
            raise RuntimeError(f"Failed to load adapter '{name}' from '{ckpt}': {e}")

    try:
        pipe.set_adapters([n for (n, _) in adapters], adapter_weights=weights)
    except Exception:
        # Older diffusers fallback
        for (name, _), w in zip(adapters, weights):
            try:
                pipe.set_adapters([name], adapter_weights=[w])
            except Exception:
                pass


def run_sd35_txt2img(job: Any, payload: Dict[str, Any]) -> str:
    """
    DISPATCH-CONTRACT:
      - payload.job_folder (ABSOLUTE) is the target directory
      - returns a STRING path to a REAL PNG inside job_folder

    NOTE:
      We do NOT write meta.json here. Dispatch owns meta writing.
    """
    job_folder = _job_folder_from_payload(payload)

    profile = str(payload.get("profile") or "r1_wide_hero").strip()
    p = _preset(profile)

    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Missing 'prompt' for sd35_txt2img")

    negative = str(payload.get("negative_prompt") or "").strip() or None

    width = int(payload.get("width") or p["width"])
    height = int(payload.get("height") or p["height"])
    steps = int(payload.get("steps") or p["steps"])
    cfg = float(payload.get("cfg") or p["cfg"])

    seed_raw = payload.get("seed")
    seed = int(seed_raw) if seed_raw not in (None, "", 0) else None

    pipe = _load_pipe_txt2img()
    _apply_adapters(pipe, payload)

    device, _dtype = _device_and_dtype()
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    out = pipe(
        prompt=prompt,
        negative_prompt=negative,
        width=width,
        height=height,
        guidance_scale=cfg,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]

    out_path = os.path.join(job_folder, "output.png")
    out.save(out_path)

    return out_path


def run_sd35_img2img(job: Any, payload: Dict[str, Any]) -> str:
    """
    DISPATCH-CONTRACT:
      - payload.job_folder (ABSOLUTE) is the target directory
      - payload.input_image must exist
      - returns a STRING path to a REAL PNG inside job_folder
    """
    job_folder = _job_folder_from_payload(payload)

    inp = payload.get("input_image")
    if not inp:
        raise ValueError("Missing 'input_image' for sd35_img2img")

    inp_path = Path(str(inp))
    if not inp_path.exists():
        raise FileNotFoundError(f"input_image not found: {inp_path}")

    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Missing 'prompt' for sd35_img2img")

    negative = str(payload.get("negative_prompt") or "").strip() or None

    # Safe defaults (Doc behavior)
    denoise = float(payload.get("denoise", 0.14))
    steps = int(payload.get("steps") or 26)
    cfg = float(payload.get("cfg") or 6.0)

    seed_raw = payload.get("seed")
    seed = int(seed_raw) if seed_raw not in (None, "", 0) else None

    try:
        from diffusers import StableDiffusion3Img2ImgPipeline
    except Exception as e:
        raise RuntimeError(f"diffusers StableDiffusion3Img2ImgPipeline import failed: {e}")

    model_id = _env("RENDEREXPO_SD35_MODEL_ID")
    if not model_id:
        raise RuntimeError("Missing env RENDEREXPO_SD35_MODEL_ID (SD3.5 model id/path)")

    device, dtype = _device_and_dtype()
    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    _apply_adapters(pipe, payload)

    image = Image.open(inp_path).convert("RGB")
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    out = pipe(
        prompt=prompt,
        negative_prompt=negative,
        image=image,
        strength=denoise,
        guidance_scale=cfg,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]

    out_path = os.path.join(job_folder, "refine.png")
    out.save(out_path)

    return out_path
