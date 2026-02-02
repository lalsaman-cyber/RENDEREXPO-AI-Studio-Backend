# runtime/sd35_runtime.py
"""
SD3.5 Runtime for RENDEREXPO AI STUDIO (GPU)

This is the ONLY place where "real quality" is created on GPU.

What this runtime MUST enforce (Doc 18):
- Presets are the source of truth for: steps, CFG, multipliers, resolution.
- NO DENOISE anywhere (denoise = 0.0).
- Upscale is OPTIONAL per request (meta["upscale"]["enabled"]).
- Presets (LyCORIS PRO 2.1 + GEO) must be applied consistently anywhere SD3.5 is used:
  text2img, img2img, future sketch/floorplan/controlnet pipelines, etc.

Notes:
- In your local routers, you already write meta with:
  - lora_config (LyCORIS PRO 2.1) and geo_config (GEO) + multipliers
  - num_inference_steps, guidance_scale, width, height
  - upscale { enabled, factor, denoise: 0.0, method: "lanczos" }
- This runtime reads ONLY meta and executes it.

Safety:
- If optional dependencies are missing (PIL/xformers), we do NOT hard-crash.
- If LyCORIS/GEO files are missing, we log and continue with base model.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

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

        # IMPORTANT: on your pod you are using /workspace-data/models/sd35
        self.model_path = os.getenv("SD35_MODEL_PATH", "/workspace-data/models/sd35")

        self.pipe: Optional[Any] = None
        self._torch: Optional[Any] = None

        # Track last adapters we applied so we can avoid stacking accidentally.
        # We keep this simple: always unload before applying new adapters.
        self._active_adapters: Tuple[str, ...] = ()

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
        Loads StableDiffusion3Pipeline on GPU.

        VRAM:
        - SD3.5 Large can be heavy. You already validated CPU offload helps.
        - Set SD35_ENABLE_CPU_OFFLOAD=1 to enable offload.
        """
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

            pipe = StableDiffusion3Pipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,  # diffusers warns but OK; keep stable on your pod
            )

            # Prefer xformers if available (safe)
            try:
                pipe.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention.")
            except Exception:
                pass

            # Optional CPU offload (big VRAM saver)
            if os.getenv("SD35_ENABLE_CPU_OFFLOAD", "0").strip().lower() in ("1", "true", "yes", "on"):
                try:
                    pipe.enable_model_cpu_offload()
                    logger.info("Enabled model CPU offload for SD3.5 pipeline.")
                except Exception:
                    logger.info("Model CPU offload not available; continuing without it.")
            else:
                # Standard .to(cuda) if offload not enabled
                pipe = pipe.to(self.device)

            self.pipe = pipe
            logger.info("SD35Runtime successfully loaded SD3.5 model.")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load SD3.5 model: %s", exc)
            self.mode = "skeleton"
            self.pipe = None

    def unload(self) -> None:
        logger.info("Unloading SD35Runtime ...")
        self._active_adapters = ()
        self.pipe = None

        if self._torch is not None:
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                pass

        logger.info("SD35Runtime unloaded.")

    # ------------------------------------------------------------------
    # Adapter (LyCORIS + GEO) handling
    # ------------------------------------------------------------------

    def _safe_unload_loras(self) -> None:
        if self.pipe is None:
            return
        try:
            if hasattr(self.pipe, "unload_lora_weights"):
                self.pipe.unload_lora_weights()
        except Exception:
            pass
        self._active_adapters = ()

    def _resolve_weight_path(self, p: str) -> str:
        """
        Allow relative paths from repo root used on pod:
        /workspace-data/RENDEREXPO-AI-Studio-Backend/<relative>
        """
        if os.path.isabs(p):
            return p
        return os.path.join("/workspace-data/RENDEREXPO-AI-Studio-Backend", p)

    def _extract_adapter_cfg(self, meta: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        We expect:
          meta["lora_config"] -> LyCORIS PRO 2.1
          meta["geo_config"]  -> GEO

        Both are optional but for Doc 18 presets they SHOULD be present.
        """
        lora_cfg = meta.get("lora_config") if isinstance(meta.get("lora_config"), dict) else None
        geo_cfg = meta.get("geo_config") if isinstance(meta.get("geo_config"), dict) else None
        return lora_cfg, geo_cfg

    def _apply_locked_adapters(self, meta: Dict[str, Any]) -> None:
        """
        Apply LyCORIS (PRO 2.1) + GEO adapters using diffusers LoRA APIs.

        We treat GEO as a second adapter using the same LoRA loading mechanism.
        This keeps the system consistent and makes it easy to reuse everywhere.

        If your GEO implementation later becomes ControlNet/other, you can swap
        this function while keeping meta contract stable.
        """
        if self.pipe is None:
            return

        lora_cfg, geo_cfg = self._extract_adapter_cfg(meta)

        # If nothing to apply: unload and return
        if not lora_cfg and not geo_cfg:
            self._safe_unload_loras()
            meta["adapters_applied"] = False
            meta["adapters_reason"] = "no_lora_or_geo_config"
            return

        # Resolve + validate paths and weights
        adapters: list[Tuple[str, str, float]] = []  # (name, path, weight)

        def _pull(cfg: Dict[str, Any], default_name: str) -> Optional[Tuple[str, str, float]]:
            path = cfg.get("path")
            if not path:
                return None
            path = self._resolve_weight_path(str(path))
            if not os.path.isfile(path):
                logger.warning("Adapter file not found: %s (skipping %s)", path, default_name)
                return None

            w = cfg.get("strength", cfg.get("scale", 0.0))
            try:
                w_f = float(w)
            except Exception:
                w_f = 0.0

            # Doc 18 uses small multipliers; allow 0..2 but clamp to safety
            if w_f < 0.0:
                w_f = 0.0
            if w_f > 2.0:
                w_f = 2.0

            # Build stable name
            label = cfg.get("label") or default_name
            name = f"renderexpo_{label}_{os.path.basename(path).replace('.', '_')}"
            return (name, path, w_f)

        l = _pull(lora_cfg, "LYCORIS_PRO21") if lora_cfg else None
        g = _pull(geo_cfg, "GEO") if geo_cfg else None

        if l:
            adapters.append(l)
        if g:
            adapters.append(g)

        # If after validation nothing remains, unload and return
        if not adapters:
            self._safe_unload_loras()
            meta["adapters_applied"] = False
            meta["adapters_reason"] = "files_missing_or_zero_weight"
            return

        # Always unload first to prevent stacking
        self._safe_unload_loras()

        # Apply
        try:
            # Newer diffusers supports multi-adapter w/ set_adapters
            if hasattr(self.pipe, "set_adapters"):
                for (name, path, _w) in adapters:
                    # adapter_name supported in newer versions
                    try:
                        self.pipe.load_lora_weights(path, adapter_name=name)
                    except TypeError:
                        # older signature
                        self.pipe.load_lora_weights(path)

                names = [a[0] for a in adapters]
                weights = [a[2] for a in adapters]

                try:
                    self.pipe.set_adapters(names, adapter_weights=weights)
                except Exception:
                    # Some variants accept singular name; try fallback
                    if len(names) == 1:
                        try:
                            self.pipe.set_adapters(names[0], adapter_weights=weights[0])
                        except Exception:
                            pass

            else:
                # Old API: load one then fuse; we do best-effort sequentially.
                # Note: this is less ideal for true multi-adapter.
                for (name, path, w) in adapters:
                    self.pipe.load_lora_weights(path)
                    if hasattr(self.pipe, "fuse_lora"):
                        try:
                            self.pipe.fuse_lora(lora_scale=w)
                        except Exception:
                            pass

            self._active_adapters = tuple([a[0] for a in adapters])

            meta["adapters_applied"] = True
            meta["adapters"] = [
                {"name": a[0], "path": a[1], "weight": a[2]} for a in adapters
            ]

            logger.info(
                "Applied adapters: %s",
                ", ".join([f"{a[0]}(w={a[2]:.4f})" for a in adapters]),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to apply adapters: %s", exc)
            meta["adapters_applied"] = False
            meta["adapters_reason"] = f"apply_failed: {exc}"

    # ------------------------------------------------------------------
    # Detail Pass (post-process)
    # ------------------------------------------------------------------

    def _detail_pass_if_enabled(self, image: Any, meta: Dict[str, Any]) -> Any:
        """
        Optional micro-contrast + controlled unsharp.
        Safe: if PIL missing, skip.
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
    # Optional deterministic upscale (NO denoise)
    # ------------------------------------------------------------------

    def _upscale_if_enabled(self, image: Any, meta: Dict[str, Any]) -> Any:
        """
        Deterministic upscale (PIL resize), no diffusion, no denoise.
        meta["upscale"] expected:
          { "enabled": bool, "factor": 2, "denoise": 0.0, "method": "lanczos" }
        """
        up = meta.get("upscale") or {}
        if not isinstance(up, dict) or not up.get("enabled"):
            return image

        # Enforce NO denoise
        up["denoise"] = 0.0

        factor = int(up.get("factor", 2))
        if factor < 2:
            factor = 2
        if factor > 4:
            factor = 4  # safety

        method = str(up.get("method", "lanczos")).lower()

        try:
            from PIL import Image  # type: ignore
        except Exception:
            meta["upscale_applied"] = False
            meta["upscale_reason"] = "PIL_missing"
            return image

        if not isinstance(image, Image.Image):
            try:
                image = Image.fromarray(image)
            except Exception:
                meta["upscale_applied"] = False
                meta["upscale_reason"] = "not_a_PIL_image"
                return image

        resample = Image.LANCZOS if method == "lanczos" else Image.BICUBIC

        w, h = image.size
        new_w, new_h = w * factor, h * factor
        try:
            image = image.resize((new_w, new_h), resample=resample)
            meta["upscale_applied"] = True
            meta["upscale_settings"] = {"factor": factor, "method": method, "denoise": 0.0}
        except Exception as exc:  # noqa: BLE001
            meta["upscale_applied"] = False
            meta["upscale_reason"] = f"resize_failed: {exc}"

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

        # Hard lock: NO denoise anywhere
        meta["denoise"] = 0.0
        if isinstance(meta.get("upscale"), dict):
            meta["upscale"]["denoise"] = 0.0

        torch = self._torch

        prompt = meta.get("prompt") or ""
        negative_prompt = meta.get("negative_prompt") or None

        width = int(meta.get("width", 1024))
        height = int(meta.get("height", 1024))
        num_steps = int(meta.get("num_inference_steps", 46))
        guidance_scale = float(meta.get("guidance_scale", 5.6))

        # Apply locked LyCORIS + GEO adapters from meta (Doc 18)
        self._apply_locked_adapters(meta)

        seed = meta.get("seed")
        generator = None
        if seed is not None:
            try:
                generator = torch.Generator(device=self.device).manual_seed(int(seed))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to set generator seed %s: %s", seed, exc)
                generator = None

        logger.info(
            "SD3.5 text2img: w=%d h=%d steps=%d cfg=%.3f seed=%s prompt='%s'",
            width,
            height,
            num_steps,
            guidance_scale,
            seed,
            prompt[:140],
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

        # Optional post detail pass (safe)
        image = self._detail_pass_if_enabled(image, meta)

        # Save base output first (always 1024x1024 or preset size)
        os.makedirs(job_folder, exist_ok=True)
        base_out = os.path.join(job_folder, "output_base.png")
        try:
            image.save(base_out)
        except Exception:
            # If base save fails, still try final output
            pass

        # Optional deterministic upscale (no denoise)
        image = self._upscale_if_enabled(image, meta)

        # Final output expected by system
        out_path = os.path.join(job_folder, "output.png")
        image.save(out_path)

        meta["status"] = "completed"
        meta["completed_at"] = datetime.utcnow().isoformat()
        meta["mode"] = "real-sd35"
        meta["output_image"] = "output.png"
        meta["output_base_image"] = "output_base.png"

        logger.info("SD3.5 text2img completed. Saved to %s", out_path)
        return meta

    # ------------------------------------------------------------------
    # Img2Img (best-effort, for when you wire it on GPU)
    # ------------------------------------------------------------------

    def generate_img2img(self, job_folder: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Real SD3.5 img2img execution.

        This will work ONLY if diffusers exposes StableDiffusion3Img2ImgPipeline
        in your installed version. If not available, we raise a clear error.
        """
        if self.mode != "real" or self._torch is None:
            raise RuntimeError("SD35Runtime.generate_img2img() called but runtime is not in real mode.")

        # Hard lock: NO denoise anywhere
        meta["denoise"] = 0.0
        if isinstance(meta.get("upscale"), dict):
            meta["upscale"]["denoise"] = 0.0

        try:
            from PIL import Image  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"PIL is required for img2img input loading: {exc}") from exc

        input_rel = meta.get("input_image", "input.png")
        input_path = input_rel if os.path.isabs(str(input_rel)) else os.path.join(job_folder, str(input_rel))
        if not os.path.isfile(input_path):
            raise RuntimeError(f"img2img input image not found: {input_path}")

        strength = float(meta.get("strength", 0.70))
        if strength < 0.0:
            strength = 0.0
        if strength > 1.0:
            strength = 1.0

        prompt = meta.get("prompt") or ""
        negative_prompt = meta.get("negative_prompt") or None
        width = int(meta.get("width", 1024))
        height = int(meta.get("height", 1024))
        num_steps = int(meta.get("num_inference_steps", 46))
        guidance_scale = float(meta.get("guidance_scale", 5.6))

        # Apply locked adapters (LyCORIS + GEO)
        self._apply_locked_adapters(meta)

        # Try to build an Img2Img pipeline from the loaded components if possible.
        # If not available in this diffusers version, fail clearly.
        try:
            from diffusers import StableDiffusion3Img2ImgPipeline  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "StableDiffusion3Img2ImgPipeline is not available in your diffusers build. "
                "Upgrade diffusers or keep img2img skeleton for now."
            ) from exc

        # Construct img2img pipe from same pretrained (keeps consistent weights)
        import torch  # type: ignore
        img_pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
        )
        # Offload settings
        if os.getenv("SD35_ENABLE_CPU_OFFLOAD", "0").strip().lower() in ("1", "true", "yes", "on"):
            try:
                img_pipe.enable_model_cpu_offload()
            except Exception:
                img_pipe = img_pipe.to(self.device)
        else:
            img_pipe = img_pipe.to(self.device)

        # Apply xformers if available
        try:
            img_pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        # Apply adapters onto img_pipe too (same meta contract)
        self.pipe, old_pipe = img_pipe, self.pipe
        try:
            self._apply_locked_adapters(meta)
        finally:
            # restore self.pipe back to text2img pipe so runtime stays consistent
            self.pipe = old_pipe

        seed = meta.get("seed")
        generator = None
        if seed is not None:
            try:
                generator = torch.Generator(device=self.device).manual_seed(int(seed))
            except Exception:
                generator = None

        logger.info(
            "SD3.5 img2img: strength=%.3f w=%d h=%d steps=%d cfg=%.3f seed=%s prompt='%s'",
            strength,
            width,
            height,
            num_steps,
            guidance_scale,
            seed,
            prompt[:140],
        )

        init_image = Image.open(input_path).convert("RGB")
        if init_image.size != (width, height):
            # keep deterministic resize for now (no denoise)
            init_image = init_image.resize((width, height), Image.LANCZOS)

        kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": init_image,
            "strength": strength,
            "num_inference_steps": num_steps,
            "guidance_scale": guidance_scale,
        }
        if generator is not None:
            kwargs["generator"] = generator

        images = img_pipe(**kwargs).images
        if not images:
            raise RuntimeError("SD3.5 img2img pipeline returned no images.")

        image = images[0]
        image = self._detail_pass_if_enabled(image, meta)

        os.makedirs(job_folder, exist_ok=True)
        base_out = os.path.join(job_folder, "output_base.png")
        try:
            image.save(base_out)
        except Exception:
            pass

        image = self._upscale_if_enabled(image, meta)

        out_path = os.path.join(job_folder, "output.png")
        image.save(out_path)

        meta["status"] = "completed"
        meta["completed_at"] = datetime.utcnow().isoformat()
        meta["mode"] = "real-sd35"
        meta["output_image"] = "output.png"
        meta["output_base_image"] = "output_base.png"

        logger.info("SD3.5 img2img completed. Saved to %s", out_path)
        return meta
