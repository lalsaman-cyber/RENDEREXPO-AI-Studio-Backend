# runtime/sd35_runtime.py
"""
SD3.5 Runtime for RENDEREXPO AI STUDIO.

This module encapsulates the "real" SD3.5 text2img pipeline that runs on GPU.

Design:
- In skeleton mode:
    * SD35Runtime.load() does nothing.
    * SD35Runtime.generate_text2img() MUST NOT be called.
- In real mode:
    * SD35Runtime.load() loads the SD3.5 Large model from disk.
    * SD35Runtime.generate_text2img() runs the model and writes output.png.

Safety:
- If imports fail or the model directory is missing, the runtime logs an error,
  stays in skeleton mode, and generate_text2img() will raise, so callers can
  fall back to skeleton behavior.
"""

from __future__ import annotations

import os
import logging
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SD35Runtime:
    """
    Runtime wrapper around the SD3.5 Large text2img pipeline.

    Attributes:
        mode: "skeleton" or "real".
        device: "cuda" (GPU) or "cpu" (for testing).
        model_path: Filesystem location of the SD3.5 model weights.
        pipe: The diffusers pipeline instance when in real mode.
    """

    def __init__(self, mode: str = "skeleton", device: str = "cuda") -> None:
        self.mode = mode
        self.device = device
        self.model_path = os.getenv("SD35_MODEL_PATH", "/workspace/models/sd35-large")

        self.pipe: Optional[Any] = None
        self._torch: Optional[Any] = None

        logger.info(
            "SD35Runtime initialized with mode=%s, device=%s, model_path=%s",
            self.mode,
            self.device,
            self.model_path,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load the SD3.5 model into GPU memory (if mode == 'real').

        If anything fails, self.mode is set to 'skeleton' and self.pipe is None.
        """
        if self.mode != "real":
            logger.info("SD35Runtime.load() called in skeleton mode. No model will be loaded.")
            return

        # Import here so skeleton mode does not require diffusers/torch.
        try:
            import torch  # type: ignore
            from diffusers import StableDiffusion3Pipeline  # type: ignore
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to import torch/diffusers for SD3.5 runtime: %s", exc)
            self.mode = "skeleton"
            self._torch = None
            self.pipe = None
            return

        self._torch = torch

        if not os.path.isdir(self.model_path):
            logger.error(
                "SD35 model path does not exist: %s. "
                "SD35Runtime will remain in skeleton mode.",
                self.model_path,
            )
            self.mode = "skeleton"
            self.pipe = None
            return

        try:
            logger.info("Loading SD3.5 model from %s ...", self.model_path)
            pipe = StableDiffusion3Pipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
            ).to(self.device)

            # Try some memory-friendly options if available
            try:
                pipe.enable_model_cpu_offload()
                logger.info("Enabled model CPU offload for SD3.5 pipeline.")
            except Exception:
                logger.info("Model CPU offload not available; continuing without it.")

            self.pipe = pipe
            logger.info("SD35Runtime successfully loaded SD3.5 model.")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load SD3.5 model: %s", exc)
            self.mode = "skeleton"
            self.pipe = None

    def unload(self) -> None:
        """
        Release model and GPU memory.
        """
        logger.info("Unloading SD35Runtime ...")
        if self.pipe is not None:
            self.pipe = None

        if self._torch is not None:
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                # If CUDA isn't available or something else goes wrong, ignore.
                pass

        logger.info("SD35Runtime unloaded.")

    # ------------------------------------------------------------------
    # Text2Img
    # ------------------------------------------------------------------

    def generate_text2img(self, job_folder: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a real SD3.5 text2img job based on the metadata.

        Args:
            job_folder: Path to outputs/.../<job_id>/
            meta: The current meta dict (prompt, width, height, etc.)

        Returns:
            The updated meta dict (status, completed_at, etc.)

        Raises:
            RuntimeError: If called when runtime is not in real mode or model not loaded.
        """
        if self.mode != "real" or self.pipe is None or self._torch is None:
            raise RuntimeError(
                "SD35Runtime.generate_text2img() called but runtime is not in real mode "
                "or model is not loaded."
            )

        torch = self._torch

        prompt = meta.get("prompt") or ""
        negative_prompt = meta.get("negative_prompt") or None

        width = int(meta.get("width", 1024))
        height = int(meta.get("height", 1024))
        num_steps = int(meta.get("num_inference_steps", 25))
        guidance_scale = float(meta.get("guidance_scale", 7.0))

        seed = meta.get("seed")
        generator = None
        if seed is not None:
            try:
                generator = torch.Generator(device=self.device).manual_seed(int(seed))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to set SD3.5 generator seed %s: %s", seed, exc)
                generator = None

        logger.info(
            "Running SD3.5 text2img: prompt='%s', width=%d, height=%d, steps=%d, scale=%.2f, seed=%s",
            prompt[:80],
            width,
            height,
            num_steps,
            guidance_scale,
            seed,
        )

        generate_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
        }
        if generator is not None:
            generate_kwargs["generator"] = generator

        images = self.pipe(**generate_kwargs).images
        if not images:
            raise RuntimeError("SD3.5 pipeline returned no images.")

        image = images[0]

        os.makedirs(job_folder, exist_ok=True)
        out_path = os.path.join(job_folder, "output.png")
        image.save(out_path)

        meta["status"] = "completed"
        meta["completed_at"] = datetime.utcnow().isoformat()
        meta["mode"] = "real-sd35"
        # Keep 'planned_output_image' as 'output.png', but we can also record actual path:
        meta["output_image"] = "output.png"

        logger.info("SD3.5 text2img completed. Saved to %s", out_path)
        return meta
