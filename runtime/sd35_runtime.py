"""
SD3.5 Runtime for RENDEREXPO AI STUDIO (GPU)

This is the ONLY place where "real quality" is created on GPU.

What this runtime MUST enforce:
- Presets are the source of truth for: steps, CFG, multipliers, resolution.
- NO hidden parameter invention inside runtime.
- Upscale is OPTIONAL per request (meta["upscale"]["enabled"]).
- Presets (LyCORIS PRO 2.1 + GEO) must be applied consistently anywhere SD3.5 is used:
  text2img, img2img, future sketch/floorplan/controlnet pipelines, etc.

Important architectural rule:
- This runtime must support LIGHT startup.
- The GPU worker may boot without preloading the heavy model.
- The model loads only when the GPU worker explicitly asks it to load.

Safety:
- If optional dependencies are missing (PIL/xformers), we do NOT hard-crash on import.
- If adapter files are missing and were explicitly requested, we HARD-FAIL.
- We never silently pretend adapters were applied when they were not.
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    ok: bool
    meta: Dict[str, Any]
    error: Optional[str] = None


def _is_lycoris_checkpoint(path: str) -> bool:
    """
    Detect LyCORIS-style safetensors (kohya/lycoris outputs).
    Cheap heuristic: keys contain lycoris_ prefix.
    """
    try:
        from safetensors import safe_open  # type: ignore

        with safe_open(path, framework="pt", device="cpu") as f:
            for i, k in enumerate(f.keys()):
                lk = k.lower()
                if lk.startswith("lycoris_"):
                    return True
                if i > 120:
                    break
    except Exception:
        pass
    return False


class SD35Runtime:
    def __init__(self, mode: str = "skeleton", device: str = "cuda") -> None:
        self.mode = mode
        self.device = device

        # IMPORTANT: on your pod you are using /workspace-data/models/sd35-large
        # If env var is set, it wins.
        self.model_path = os.getenv("SD35_MODEL_PATH", "/workspace-data/models/sd35-large")

        self.pipe: Optional[Any] = None
        self._torch: Optional[Any] = None
        self._loaded: bool = False

        # Track last adapters we applied so we can avoid stacking accidentally.
        self._active_adapters: Tuple[str, ...] = ()

        # LyCORIS caching
        self._lyco_weights_cache: Dict[str, Dict[str, Any]] = {}
        self._active_lyco_networks: List[Any] = []
        self._active_lyco_paths: Tuple[str, ...] = ()

        logger.info(
            "SD35Runtime initialized with mode=%s, device=%s, model_path=%s",
            self.mode,
            self.device,
            self.model_path,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self.pipe is not None and self._torch is not None and self.mode == "real"

    def load(self) -> None:
        """
        Load StableDiffusion3Pipeline on GPU.

        This method is explicit and should only be called when the GPU worker
        decides it needs the real model. It must not be assumed to happen at boot.

        Set SD35_ENABLE_CPU_OFFLOAD=1 to enable offload.
        """
        if self.mode != "real":
            logger.info("SD35Runtime.load() called in non-real mode. No model will be loaded.")
            self._loaded = False
            return

        if self.is_loaded:
            logger.info("SD35Runtime.load() called but runtime is already loaded.")
            return

        try:
            import torch  # type: ignore
            from diffusers import StableDiffusion3Pipeline  # type: ignore
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to import torch/diffusers for SD3.5 runtime: %s", exc)
            self.mode = "skeleton"
            self._torch = None
            self.pipe = None
            self._loaded = False
            return

        self._torch = torch

        if not os.path.isdir(self.model_path):
            logger.error("SD3.5 model path does not exist: %s (staying unloaded)", self.model_path)
            self.mode = "skeleton"
            self.pipe = None
            self._loaded = False
            return

        try:
            logger.info("Loading SD3.5 model from %s ...", self.model_path)

            pipe = StableDiffusion3Pipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
            )

            # Prefer xformers if available
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
                    pipe = pipe.to(self.device)
            else:
                pipe = pipe.to(self.device)

            self.pipe = pipe
            self._loaded = True
            logger.info("SD35Runtime successfully loaded SD3.5 model.")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load SD3.5 model: %s", exc)
            self.mode = "skeleton"
            self.pipe = None
            self._loaded = False

    def unload(self) -> None:
        logger.info("Unloading SD35Runtime ...")

        try:
            self._safe_unload_adapters()
        except Exception:
            pass

        self._active_adapters = ()
        self.pipe = None
        self._loaded = False

        if self._torch is not None:
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                pass

        logger.info("SD35Runtime unloaded.")

    # ------------------------------------------------------------------
    # Adapter (LyCORIS + GEO) handling
    # ------------------------------------------------------------------

    def _safe_unload_adapters(self) -> None:
        """
        Restore LyCORIS networks (prevents stacking across jobs),
        and unload any diffusers LoRA adapters if they were ever used.
        """
        if self.pipe is None:
            return

        if self._active_lyco_networks:
            te_list = self._get_text_encoders(self.pipe)
            unet_like = self._get_unet_like(self.pipe)
            for net in self._active_lyco_networks:
                try:
                    net.restore(te_list, unet_like)
                except TypeError:
                    try:
                        net.restore(unet_like)
                    except Exception:
                        pass
                except Exception:
                    pass

        self._active_lyco_networks = []
        self._active_lyco_paths = ()

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
        """
        lora_cfg = meta.get("lora_config") if isinstance(meta.get("lora_config"), dict) else None
        geo_cfg = meta.get("geo_config") if isinstance(meta.get("geo_config"), dict) else None
        return lora_cfg, geo_cfg

    def _get_text_encoders(self, pipe: Any) -> List[Any]:
        te: List[Any] = []
        if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
            te.append(pipe.text_encoder)
        if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
            te.append(pipe.text_encoder_2)
        if hasattr(pipe, "text_encoder_3") and pipe.text_encoder_3 is not None:
            te.append(pipe.text_encoder_3)
        return te

    def _get_unet_like(self, pipe: Any) -> Any:
        if hasattr(pipe, "transformer") and pipe.transformer is not None:
            return pipe.transformer
        raise RuntimeError("SD3 pipeline missing pipe.transformer (cannot apply LyCORIS).")

    def _load_lycoris_weights_sd(self, path: str) -> Dict[str, Any]:
        if path in self._lyco_weights_cache:
            return self._lyco_weights_cache[path]

        ext = os.path.splitext(path)[1].lower()
        if ext == ".safetensors":
            from safetensors.torch import load_file  # type: ignore

            sd = load_file(path)
        else:
            import torch  # type: ignore

            sd = torch.load(path, map_location="cpu")

        self._lyco_weights_cache[path] = sd
        return sd

    def _unwrap_lyco_net(self, net_obj: Any) -> Any:
        if not isinstance(net_obj, tuple):
            return net_obj
        for item in net_obj:
            if hasattr(item, "apply_to"):
                return item
        raise TypeError("create_network_from_weights returned tuple without a network object")

    def _apply_to_modules(self, net: Any, te_list: List[Any], unet_like: Any) -> None:
        apply_te = bool(te_list)
        apply_unet = unet_like is not None

        try:
            net.apply_to(
                te_list,
                unet_like,
                apply_text_encoder=apply_te,
                apply_unet=apply_unet,
            )
            return
        except TypeError:
            pass

        try:
            net.apply_to(te_list, unet_like)
            return
        except TypeError:
            pass

        te0 = te_list[0] if te_list else None
        try:
            net.apply_to(
                te0,
                unet_like,
                apply_text_encoder=(te0 is not None),
                apply_unet=apply_unet,
            )
            return
        except TypeError:
            pass

        net.apply_to(te0, unet_like)

    def _apply_locked_adapters(self, meta: Dict[str, Any]) -> None:
        """
        Apply LyCORIS (PRO 2.1) + GEO adapters using LyCORIS kohya runtime.

        IMPORTANT:
        - Your .safetensors are LyCORIS-format outputs.
        - We DO NOT use diffusers.load_lora_weights() for these.
        - SD3 uses pipe.transformer (not pipe.unet).
        - Override multipliers MUST be honored.
        """
        if self.pipe is None:
            raise RuntimeError("Cannot apply adapters because SD35Runtime is not loaded.")

        lora_cfg, geo_cfg = self._extract_adapter_cfg(meta)

        # Nothing configured -> ensure clean state
        if not lora_cfg and not geo_cfg:
            self._safe_unload_adapters()
            meta["adapters_applied"] = False
            meta["adapters_reason"] = "no_lora_or_geo_config"
            meta["adapters"] = []
            return

        adapters: List[Tuple[str, str, float, str]] = []

        def _pull(cfg: Dict[str, Any], logical: str, default_label: str) -> Optional[Tuple[str, str, float, str]]:
            raw_path = cfg.get("path")
            if not raw_path:
                raise RuntimeError(f"{logical} config exists but path is missing.")

            path = self._resolve_weight_path(str(raw_path))
            if not os.path.isfile(path):
                raise RuntimeError(f"{logical} adapter file not found: {path}")

            w = cfg.get("strength", cfg.get("scale", 0.0))
            try:
                w_f = float(w)
            except Exception as exc:
                raise RuntimeError(f"{logical} adapter weight is invalid: {w}") from exc

            if w_f < 0.0:
                w_f = 0.0
            if w_f > 2.0:
                w_f = 2.0

            label = str(cfg.get("label") or default_label)

            # If explicitly requested config exists but weight is zero, treat as disabled.
            if w_f <= 0.0:
                return None

            return (logical, path, w_f, label)

        l = _pull(lora_cfg, logical="LYCORIS", default_label="LYCORIS_PRO21") if lora_cfg else None
        g = _pull(geo_cfg, logical="GEO", default_label="GEO") if geo_cfg else None

        if l:
            adapters.append(l)
        if g:
            adapters.append(g)

        if not adapters:
            # Explicit configs present but both disabled -> clean state.
            self._safe_unload_adapters()
            meta["adapters_applied"] = False
            meta["adapters_reason"] = "disabled_or_missing"
            meta["adapters"] = []
            return

        self._safe_unload_adapters()

        try:
            import lycoris.kohya as lk  # type: ignore

            te_list = self._get_text_encoders(self.pipe)
            unet_like = self._get_unet_like(self.pipe)
            vae = getattr(self.pipe, "vae", None)

            applied_info: List[Dict[str, Any]] = []
            active_nets: List[Any] = []
            active_paths: List[str] = []

            for (logical, path, weight, label) in adapters:
                if not _is_lycoris_checkpoint(path):
                    raise ValueError(f"Adapter is not detected as LyCORIS: {os.path.basename(path)}")

                weights_sd = self._load_lycoris_weights_sd(path)

                net_obj = lk.create_network_from_weights(
                    multiplier=weight,
                    file=path,
                    vae=vae,
                    text_encoder=te_list,
                    unet=unet_like,
                    weights_sd=weights_sd,
                    for_inference=True,
                )

                net = self._unwrap_lyco_net(net_obj)
                self._apply_to_modules(net, te_list, unet_like)

                try:
                    net.apply()
                except Exception:
                    pass

                try:
                    net.set_multiplier(weight)
                except Exception:
                    pass

                active_nets.append(net)
                active_paths.append(path)

                name = f"renderexpo_{logical}_{label}_{os.path.basename(path).replace('.', '_')}"
                applied_info.append(
                    {
                        "name": name,
                        "path": path,
                        "weight": weight,
                        "logical": logical,
                        "label": label,
                    }
                )

                logger.info("Applied LyCORIS adapter: %s (weight=%.6f)", path, weight)

            self._active_lyco_networks = active_nets
            self._active_lyco_paths = tuple(active_paths)
            self._active_adapters = tuple([a["name"] for a in applied_info])

            meta["adapters_applied"] = True
            meta["adapters_reason"] = "applied_lycoris"
            meta["adapters"] = applied_info

            logger.info(
                "Applied LyCORIS adapters: %s",
                ", ".join([f"{a['name']}(w={a['weight']:.6f})" for a in applied_info]),
            )

        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to apply LyCORIS adapters: %s", exc)
            meta["adapters_applied"] = False
            meta["adapters_reason"] = f"apply_failed_lycoris: {exc}"
            raise

    # ------------------------------------------------------------------
    # Detail Pass (post-process)
    # ------------------------------------------------------------------

    def _detail_pass_if_enabled(self, image: Any, meta: Dict[str, Any]) -> Any:
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

        try:
            image = ImageEnhance.Contrast(image).enhance(1.05 + 0.10 * min(max(amount, 0.0), 2.0))
        except Exception:
            pass

        try:
            percent = int(120 + 120 * min(max(amount, 0.0), 2.0))
            image = image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
        except Exception:
            pass

        meta["detail_pass_applied"] = True
        meta["detail_pass_settings"] = {"amount": amount, "radius": radius, "threshold": threshold}
        return image

    # ------------------------------------------------------------------
    # Optional deterministic upscale
    # ------------------------------------------------------------------

    def _upscale_if_enabled(self, image: Any, meta: Dict[str, Any]) -> Any:
        up = meta.get("upscale") or {}
        if not isinstance(up, dict) or not up.get("enabled"):
            return image

        factor = int(up.get("factor", 2))
        if factor < 2:
            factor = 2
        if factor > 4:
            factor = 4

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
            meta["upscale_settings"] = {"factor": factor, "method": method}
        except Exception as exc:  # noqa: BLE001
            meta["upscale_applied"] = False
            meta["upscale_reason"] = f"resize_failed: {exc}"

        return image

    # ------------------------------------------------------------------
    # Text2Img
    # ------------------------------------------------------------------

    def generate_text2img(self, job_folder: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError(
                "SD35Runtime.generate_text2img() called but runtime is not loaded in real mode."
            )

        torch = self._torch
        if torch is None:
            raise RuntimeError("Torch runtime is unavailable.")

        prompt = meta.get("prompt") or ""
        negative_prompt = meta.get("negative_prompt") or None

        width = int(meta.get("width", 1024))
        height = int(meta.get("height", 1024))
        num_steps = int(meta.get("num_inference_steps", 46))
        guidance_scale = float(meta.get("guidance_scale", 5.6))

        # Apply locked LyCORIS + GEO adapters from meta
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

        images = self.pipe(**generate_kwargs).images  # type: ignore[misc]
        if not images:
            raise RuntimeError("SD3.5 pipeline returned no images.")

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

        logger.info("SD3.5 text2img completed. Saved to %s", out_path)
        return meta

    # ------------------------------------------------------------------
    # Img2Img
    # ------------------------------------------------------------------

    def generate_img2img(self, job_folder: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Real SD3.5 img2img execution.
        """
        if not self.is_loaded:
            raise RuntimeError("SD35Runtime.generate_img2img() called but runtime is not loaded in real mode.")

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

        # Apply locked adapters to the main pipe context
        self._apply_locked_adapters(meta)

        try:
            from diffusers import StableDiffusion3Img2ImgPipeline  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "StableDiffusion3Img2ImgPipeline is not available in your diffusers build. "
                "Upgrade diffusers or keep img2img skeleton for now."
            ) from exc

        import torch  # type: ignore

        img_pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
        )

        if os.getenv("SD35_ENABLE_CPU_OFFLOAD", "0").strip().lower() in ("1", "true", "yes", "on"):
            try:
                img_pipe.enable_model_cpu_offload()
            except Exception:
                img_pipe = img_pipe.to(self.device)
        else:
            img_pipe = img_pipe.to(self.device)

        try:
            img_pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        old_pipe = self.pipe
        try:
            self.pipe = img_pipe
            self._apply_locked_adapters(meta)

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

        finally:
            try:
                self._safe_unload_adapters()
            except Exception:
                pass
            self.pipe = old_pipe