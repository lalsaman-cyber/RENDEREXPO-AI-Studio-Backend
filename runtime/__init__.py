"""
runtime package for RENDEREXPO AI STUDIO.

This folder contains the "brains" of the system:
- sd35_runtime: Stable Diffusion 3.5 + RENDEREXPO-ULTRA runtime
- (later) controlnet_runtime, midas_runtime, esrgan_runtime, etc.

Important:
- Safe to import on your laptop.
- Heavy GPU stuff will only run when we call `.load()` on a GPU machine.
"""

from .sd35_runtime import SD35Runtime, GenerationResult  # re-export for convenience
