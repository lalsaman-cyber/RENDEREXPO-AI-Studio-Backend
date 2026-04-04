from __future__ import annotations

import os

from app.services.comfy_anyline_mistoline import SketchJobConfig


def get_anyline_mistoline_config() -> SketchJobConfig:
    return SketchJobConfig(
        comfy_url=os.getenv("COMFYUI_URL", "http://127.0.0.1:8188"),
        sdxl_checkpoint_name=os.getenv(
            "COMFYUI_SDXL_CHECKPOINT",
            "sd_xl_base_1.0.safetensors",
        ),
        controlnet_name=os.getenv(
            "COMFYUI_MISTOLINE_CONTROLNET",
            "mistoLine_rank256.safetensors",
        ),
        sampler_name=os.getenv("COMFYUI_SAMPLER", "dpmpp_2m_sde"),
        scheduler=os.getenv("COMFYUI_SCHEDULER", "karras"),
        steps=int(os.getenv("COMFYUI_STEPS", "30")),
        cfg=float(os.getenv("COMFYUI_CFG", "7.0")),
        denoise=float(os.getenv("COMFYUI_DENOISE", "0.93")),
        control_strength=float(os.getenv("COMFYUI_CONTROL_STRENGTH", "1.0")),
        start_percent=float(os.getenv("COMFYUI_START_PERCENT", "0.0")),
        end_percent=float(os.getenv("COMFYUI_END_PERCENT", "0.9")),
        output_prefix=os.getenv("COMFYUI_OUTPUT_PREFIX", "renderexpo_sketch"),
        poll_timeout=int(os.getenv("COMFYUI_POLL_TIMEOUT", "900")),
    )