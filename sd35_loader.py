"""
SD3.5 Large loader for RENDEREXPO AI STUDIO.

- Uses the *local* download at ./models/sd35-large
- Reads config/model_paths.yaml for paths & enable flags
- Enforces GPU-only use (no CPU fallback)
- Does NOT run automatically; only when called from the API on a GPU machine.

IMPORTANT:
    - Do NOT call load_sd35_pipeline() on your local laptop CPU.
    - This is intended to run on a GPU environment (e.g., RunPod) only.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Literal

# Cached pipeline (so we don't reload every time)
_sd35_pipeline = None

# Type alias for clarity
DeviceType = Literal["cuda", "cpu", "mps"]


def _project_root() -> Path:
    """
    Returns the Backend root folder, assuming this file lives in:
        Backend/app/core/sd35_loader.py
    """
    return Path(__file__).resolve().parents[2]


def _get_model_config() -> dict:
    """
    Load config/model_paths.yaml and return the sd35_large section.
    Lazy-imports yaml so the file can exist even before dependencies are installed.
    """
    import importlib

    root = _project_root()
    cfg_path = root / "config" / "model_paths.yaml"

    if not cfg_path.exists():
        raise FileNotFoundError(
            f"model_paths.yaml not found at: {cfg_path}\n"
            "Make sure you created it with the sd35_large entry."
        )

    try:
        yaml = importlib.import_module("yaml")
    except ImportError as e:
        raise RuntimeError(
            "PyYAML is not installed. Install it with:\n\n"
            "    pip install pyyaml\n"
        ) from e

    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    sd35_cfg = data.get("sd35_large")
    if not sd35_cfg:
        raise KeyError(
            "No 'sd35_large' section found in model_paths.yaml.\n"
            "Please ensure it contains a block like:\n\n"
            "sd35_large:\n"
            "  name: \"Stable Diffusion 3.5 Large\"\n"
            "  path: \"./models/sd35-large\"\n"
            "  type: \"stabilityai/sd3.5-large\"\n"
            "  requires_hf_token: true\n"
            "  enabled: true\n"
        )

    if not sd35_cfg.get("enabled", False):
        raise RuntimeError(
            "sd35_large is marked as disabled in model_paths.yaml. "
            "Set enabled: true to use it."
        )

    return sd35_cfg


def get_sd35_model_path() -> Path:
    """
    Resolve the local path for SD3.5 Large from model_paths.yaml.
    """
    cfg = _get_model_config()
    path_str = cfg.get("path", "./models/sd35-large")
    model_path = (_project_root() / path_str).resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            f"SD3.5 model directory does not exist at: {model_path}\n"
            "Make sure download_sd35.py completed successfully and that\n"
            "model_paths.yaml points to the correct folder."
        )

    return model_path


def _ensure_gpu_only(device: DeviceType) -> None:
    """
    Enforce the policy: SD3.5 must NOT run on CPU.

    - If device != 'cuda'  -> raise
    - If CUDA is not available -> raise

    This encodes your preference: no slow CPU tests, GPU only.
    """
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "PyTorch is not installed. Install it in your GPU environment with e.g.:\n\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cu121\n\n"
            "(Adjust the CUDA version URL as needed for your GPU image.)"
        ) from e

    if device != "cuda":
        raise RuntimeError(
            f"SD3.5 is configured for GPU-only use. Received device='{device}'.\n"
            "Refusing to run on CPU or non-CUDA devices."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this environment.\n"
            "Do NOT run Stable Diffusion 3.5 on CPU. "
            "Start a GPU pod (e.g., on RunPod) and try again."
        )


def load_sd35_pipeline(
    device: DeviceType = "cuda",
    dtype: Literal["bfloat16", "float16"] = "bfloat16",
):
    """
    Lazily load the Stable Diffusion 3.5 Large pipeline on a GPU.

    - Uses local weights from ./models/sd35-large
    - Does NOT reach out to Hugging Face (offline first)
    - Requires a CUDA-capable environment
    - Caches the pipeline in memory

    NOTE:
        This function should only be called in your GPU runtime (RunPod, etc.).
    """
    global _sd35_pipeline
    if _sd35_pipeline is not None:
        return _sd35_pipeline

    # Enforce GPU-only policy
    _ensure_gpu_only(device=device)

    # Lazy imports for heavy libs
    try:
        import torch
        from diffusers import StableDiffusion3Pipeline
    except ImportError as e:
        raise RuntimeError(
            "Missing required libraries for SD3.5.\n"
            "Install in your GPU environment:\n\n"
            "    pip install diffusers transformers accelerate safetensors\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
            "\n(Adjust CUDA wheel URL as needed.)"
        ) from e

    model_path = get_sd35_model_path()

    # Choose dtype
    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # We already have local weights; no need for use_auth_token here.
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        use_auth_token=False,
    )

    pipe = pipe.to(device)

    # Optional memory optimizations (only if available)
    if hasattr(pipe, "enable_model_cpu_offload"):
        # We usually won't use this in production RunPod config, but it's safe.
        pipe.enable_model_cpu_offload()
    elif hasattr(pipe, "enable_sequential_cpu_offload"):
        pipe.enable_sequential_cpu_offload()

    _sd35_pipeline = pipe
    return _sd35_pipeline


def unload_sd35_pipeline() -> None:
    """
    Optional helper to free VRAM if you ever need to unload SD3.5.
    """
    global _sd35_pipeline
    if _sd35_pipeline is None:
        return

    try:
        import torch
    except ImportError:
        torch = None

    _sd35_pipeline = None

    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
