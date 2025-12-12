# runtime/sd35_runtime.py
"""
SD3.5 Runtime for RENDEREXPO AI STUDIO.

REAL GPU responsibilities:
- Load SD3.5 Large via diffusers
- Apply LoRA (from meta["lora_config"]) if provided
- Generate text2img
- Optional "detail pass" post-process (to push clarity/sharp micro-contrast)

Important:
- This runtime is the ONLY place where "real quality" is created on GPU.
- Your local API only "plans" and dispatches jobs.

This file intentionally:
- stays safe in skeleton mode
- never hard-crashes if optional detail pass tooling isn't installed
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    ok: bool
    meta: Dict[str, Any]
    error: Optional[str] = None


class SD35Runtime:
    def __init__(self, mode: str = "skeleton", device: str = "cuda") -> None:
        self.mode = mode
        self.device = device

        # IMPORTANT: your repo uses /workspace-data/models/sd35-large on RunPod.
        # Keep env override too.
        self.model_path = os.getenv("SD35_MODEL_PATH", "/workspace-data/models/sd35-large")

        self.pipe: Optional[Any] = None
        self._torch: Optional[Any] = None

        # Track last applied LoRA so we can unload/avoid stacking accidentally
        self._active_lora_name: Optional[str] = None

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
        if self.mode != "real":
            logger.info("SD35Runtime.load() called in skeleton mode. No model will be loaded.")
            return

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
            logger.error("SD3.5 model path does not exist: %s (staying skeleton)", self.model_path)
            self.mode = "skeleton"
            self.pipe = None
            return

        try:
            logger.info("Loading SD3.5 model from %s ...", self.model_path)

            # SD3.5 typically likes fp16 on GPU
            pipe = StableDiffusion3Pipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
            ).to(self.device)

            # Prefer xformers if available (won't crash if missing)
            try:
                pipe.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention.")
            except Exception:
                pass

            # NOTE: cpu offload can reduce VRAM but can also slow. Keep it optional.
            # If you want it ON, set SD35_ENABLE_CPU_OFFLOAD=1
            if os.getenv("SD35_ENABLE_CPU_OFFLOAD", "0").strip().lower() in ("1", "true", "yes", "on"):
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
        logger.info("Unloading SD35Runtime ...")
        self._active_lora_name = None
        self.pipe = None

        if self._torch is not None:
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                pass

        logger.info("SD35Runtime unloaded.")

    # ------------------------------------------------------------------
    # LoRA handling
    # ------------------------------------------------------------------

    def _apply_lora_if_any(self, meta: Dict[str, Any]) -> None:
        """
        Applies LoRA specified in meta["lora_config"].

        Expected lora_config example (from your routers):
            {
              "path": "models/lora/exterior_v1.safetensors",
              "scale": 0.8,
              "strength": 0.8,
              ...
            }

        Notes:
        - We try multiple diffusers APIs because versions differ.
        - If loading fails, we log + continue (base model still runs).
        """
        if self.pipe is None:
            return

        lora_cfg = meta.get("lora_config") or {}
        lora_path = lora_cfg.get("path")
        if not lora_path:
            return

        # Resolve relative paths from repo root
        if not os.path.isabs(lora_path):
            lora_path = os.path.join("/workspace-data/RENDEREXPO-AI-Studio-Backend", lora_path)

        if not os.path.isfile(lora_path):
            logger.warning("LoRA file not found: %s (running without LoRA)", lora_path)
            return

        # strength/scale: allow either key
        scale = lora_cfg.get("scale", None)
        strength = lora_cfg.get("strength", None)
        adapter_scale = strength if strength is not None else scale
        if adapter_scale is None:
            adapter_scale = 0.8

        try:
            adapter_scale = float(adapter_scale)
        except Exception:
            adapter_scale = 0.8

        adapter_name = f"renderexpo_{os.path.basename(lora_path).replace('.', '_')}"

        # Best-effort cleanup so we don't stack old adapters
        try:
            if hasattr(self.pipe, "unload_lora_weights"):
                self.pipe.unload_lora_weights()
        except Exception:
            pass

        try:
            # diffusers newer multi-adapter API
            if hasattr(self.pipe, "set_adapters"):
                self.pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
                try:
                    self.pipe.set_adapters([adapter_name], adapter_weights=[adapter_scale])
                except Exception:
                    try:
                        self.pipe.set_adapters(adapter_name, adapter_weights=adapter_scale)
                    except Exception:
                        pass
            else:
                # older/simpler API
                self.pipe.load_lora_weights(lora_path)
                if hasattr(self.pipe, "fuse_lora"):
                    try:
                        self.pipe.fuse_lora(lora_scale=adapter_scale)
                    except Exception:
                        pass
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load/apply LoRA (%s): %s", lora_path, exc)
            return

        self._active_lora_name = adapter_name
        meta["lora_applied"] = True
        meta["lora_path_resolved"] = lora_path
        meta["lora_scale_applied"] = adapter_scale
        logger.info("Applied LoRA: %s (scale=%.3f)", lora_path, adapter_scale)

    # ------------------------------------------------------------------
    # Detail Pass (post-process)
    # ------------------------------------------------------------------

    def _detail_pass_if_enabled(self, image: Any, meta: Dict[str, Any]) -> Any:
        """
        Optional, safe "detail pass".
        - Uses Pillow if available
        - Controlled micro-contrast + unsharp mask (avoids insane halos)

        meta format:
            meta["detail_pass"] = {
              "enabled": true,
              "amount": 1.0,   # 0.0 - 2.0
              "radius": 1.2,   # 0.5 - 2.5
              "threshold": 3   # 0 - 10
            }
        """
        dp = meta.get("detail_pass") or {}
        if not dp or not dp.get("enabled"):
            return image

        try:
            from PIL import Image, ImageFilter, ImageEnhance  # type: ignore
        except Exception:
            logger.warning("Detail pass requested but PIL is not available. Skipping.")
            meta["detail_pass_applied"] = False
            meta["detail_pass_reason"] = "PIL_missing"
            return image

        if not isinstance(image, Image.Image):
            try:
                image = Image.fromarray(image)
            except Exception:
                meta["detail_pass_applied"] = False
                meta["detail_pass_reason"] = "not_a_PIL_image"
                return image

        amount = float(dp.get("amount", 1.0))
        radius = float(dp.get("radius", 1.15))
        threshold = int(dp.get("threshold", 3))

        # Mild local contrast boost
        try:
            image = ImageEnhance.Contrast(image).enhance(1.05 + 0.10 * min(max(amount, 0.0), 2.0))
        except Exception:
            pass

        # Controlled sharpening
        try:
            percent = int(120 + 120 * min(max(amount, 0.0), 2.0))  # 120..360
            image = image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
        except Exception:
            pass

        meta["detail_pass_applied"] = True
        meta["detail_pass_settings"] = {"amount": amount, "radius": radius, "threshold": threshold}
        return image

    # ------------------------------------------------------------------
    # Text2Img
    # ------------------------------------------------------------------

    def generate_text2img(self, job_folder: str, meta: Dict[str, Any]) -> Dict[str, Any]:
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
        num_steps = int(meta.get("num_inference_steps", 30))
        guidance_scale = float(meta.get("guidance_scale", 6.0))

        # BIG FIX: apply LoRA before generation
        self._apply_lora_if_any(meta)

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
            prompt[:120],
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

        # Optional clarity push
        image = self._detail_pass_if_enabled(image, meta)

        os.makedirs(job_folder, exist_ok=True)
        out_path = os.path.join(job_folder, "output.png")
        image.save(out_path)

        meta["status"] = "completed"
        meta["completed_at"] = datetime.utcnow().isoformat()
        meta["mode"] = "real-sd35"
        meta["output_image"] = "output.png"

        logger.info("SD3.5 text2img completed. Saved to %s", out_path)
        return meta
