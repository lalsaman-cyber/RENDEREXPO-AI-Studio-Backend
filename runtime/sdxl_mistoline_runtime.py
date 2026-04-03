# runtime/sdxl_mistoline_runtime.py
from __future__ import annotations

import gc
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

from PIL import Image

from runtime.controlnet.mistoline_preprocess import build_mistoline_control_image

logger = logging.getLogger(__name__)


def _read_model_paths() -> Dict[str, str]:
    path = os.path.join("config", "model_paths.yaml")
    if not os.path.isfile(path):
        return {}

    result: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            result[key] = value
    return result


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _upscale_png(src_path: str, dst_path: str, factor: int = 2) -> str:
    img = Image.open(src_path).convert("RGB")
    w, h = img.size
    up = img.resize((max(1, w * factor), max(1, h * factor)), Image.LANCZOS)
    up.save(dst_path)
    return dst_path


class SDXLMistoLineRuntime:
    """
    Separate sketch-only runtime for:
    - SDXL base
    - MistoLine ControlNet
    - madebyollin SDXL VAE fix

    This implementation is intentionally aligned with the published
    MistoLine Diffusers route and recommended parameter direction:
    - SDXL base 1.0
    - MistoLine with variant='fp16'
    - madebyollin/sdxl-vae-fp16-fix
    - recommended defaults:
        steps=30
        cfg=7.0
        control strength=1.0
        control guidance end=0.9
    """

    def __init__(self, mode: str = "real", device: str = "cuda") -> None:
        self.mode = mode
        self.device = device
        self._pipe = None
        self._controlnet = None
        self._vae = None

        paths = _read_model_paths()
        self.sdxl_base_path = os.getenv(
            "SDXL_BASE_PATH",
            paths.get("sdxl_base_dir", "stabilityai/stable-diffusion-xl-base-1.0"),
        )
        self.mistoline_path = os.getenv(
            "MISTOLINE_CONTROLNET_PATH",
            paths.get("mistoline_controlnet_dir", "TheMistoAI/MistoLine"),
        )
        self.sdxl_vae_path = os.getenv(
            "SDXL_VAE_PATH",
            "madebyollin/sdxl-vae-fp16-fix",
        )

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def _configure_scheduler(self) -> None:
        if self._pipe is None:
            return

        try:
            from diffusers import DPMSolverMultistepScheduler

            # Closest Diffusers analog to the recommended DPM++ 2M SDE Karras path.
            self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self._pipe.scheduler.config,
                algorithm_type="sde-dpmsolver++",
                use_karras_sigmas=True,
            )
            logger.info(
                "Configured scheduler: DPMSolverMultistepScheduler("
                "algorithm_type='sde-dpmsolver++', use_karras_sigmas=True)"
            )
        except Exception as exc:
            logger.warning("Failed to set DPM++ SDE Karras-like scheduler. Keeping default. detail=%s", exc)

    def load(self) -> None:
        if self.is_loaded:
            return

        logger.info(
            "Loading SDXL + MistoLine runtime. base=%s controlnet=%s vae=%s device=%s",
            self.sdxl_base_path,
            self.mistoline_path,
            self.sdxl_vae_path,
            self.device,
        )

        import torch
        from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetPipeline

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32

        # Published MistoLine Diffusers route:
        # ControlNetModel.from_pretrained(..., variant="fp16")
        self._controlnet = ControlNetModel.from_pretrained(
            self.mistoline_path,
            torch_dtype=dtype,
            variant="fp16",
        )

        self._vae = AutoencoderKL.from_pretrained(
            self.sdxl_vae_path,
            torch_dtype=dtype,
        )

        self._pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.sdxl_base_path,
            controlnet=self._controlnet,
            vae=self._vae,
            torch_dtype=dtype,
        )

        self._configure_scheduler()

        if self.device.startswith("cuda"):
            # Keep this VRAM-friendly unless you later decide to trade memory for speed.
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe = self._pipe.to("cpu")

        try:
            self._pipe.enable_vae_slicing()
        except Exception:
            pass

        try:
            self._pipe.enable_attention_slicing()
        except Exception:
            pass

        logger.info("SDXL + MistoLine runtime loaded.")

    def unload(self) -> None:
        if self._pipe is not None:
            try:
                del self._pipe
            except Exception:
                pass
            self._pipe = None

        if self._controlnet is not None:
            try:
                del self._controlnet
            except Exception:
                pass
            self._controlnet = None

        if self._vae is not None:
            try:
                del self._vae
            except Exception:
                pass
            self._vae = None

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        gc.collect()
        logger.info("SDXL + MistoLine runtime unloaded.")

    def _resolve_generation_settings(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use MistoLine's published settings by default unless explicitly overridden.

        MistoLine page:
        - steps: 30
        - CFG: 7.0
        - controlnet_strength: 1.0
        - end_percent: 0.9

        We map these into the current Diffusers runtime.
        """
        user_cfg = meta.get("mistoline_generation")
        if not isinstance(user_cfg, dict):
            user_cfg = {}

        width = _to_int(meta.get("width"), 1024)
        height = _to_int(meta.get("height"), 1024)

        settings = {
            "width": width,
            "height": height,
            "num_inference_steps": _to_int(user_cfg.get("num_inference_steps"), 30),
            "guidance_scale": _to_float(user_cfg.get("guidance_scale"), 7.0),
            "controlnet_conditioning_scale": _to_float(user_cfg.get("controlnet_conditioning_scale"), 1.0),
            "control_guidance_start": _to_float(user_cfg.get("control_guidance_start"), 0.0),
            "control_guidance_end": _to_float(user_cfg.get("control_guidance_end"), 0.9),
        }
        return settings

    def generate_mistoline_sketch(self, job_folder: str, meta: Dict[str, Any]) -> Dict[str, str]:
        if not self.is_loaded:
            self.load()

        sketch_path = os.path.join(job_folder, str(meta.get("inputs", {}).get("sketch_image", "sketch.png")))
        if not os.path.isfile(sketch_path):
            fallback = os.path.join(job_folder, "sketch.png")
            if os.path.isfile(fallback):
                sketch_path = fallback
            else:
                raise FileNotFoundError(f"Sketch input not found: {sketch_path}")

        mistoline_cfg = meta.get("mistoline") if isinstance(meta.get("mistoline"), dict) else {}
        cleanup_cfg = mistoline_cfg.get("cleanup") if isinstance(mistoline_cfg.get("cleanup"), dict) else {}

        control_png = os.path.join(job_folder, str(mistoline_cfg.get("control_image") or "mistoline_control.png"))
        control_png, preprocess_meta = build_mistoline_control_image(
            sketch_path=sketch_path,
            output_path=control_png,
            cleanup=cleanup_cfg,
        )

        prompt = str(meta.get("prompt") or "").strip()
        if not prompt:
            raise RuntimeError("MistoLine sketch generation requires meta.prompt.")

        negative_prompt = str(meta.get("negative_prompt") or "").strip()
        seed = _to_int(meta.get("seed"), 0)

        settings = self._resolve_generation_settings(meta)
        width = settings["width"]
        height = settings["height"]
        steps = settings["num_inference_steps"]
        guidance = settings["guidance_scale"]
        control_strength = settings["controlnet_conditioning_scale"]
        control_guidance_start = settings["control_guidance_start"]
        control_guidance_end = settings["control_guidance_end"]

        import torch

        generator = None
        if seed > 0:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        control_image = Image.open(control_png).convert("RGB")

        logger.info(
            "Running MistoLine sketch generation: steps=%s cfg=%s control_strength=%s "
            "control_guidance_start=%s control_guidance_end=%s size=%sx%s seed=%s",
            steps,
            guidance,
            control_strength,
            control_guidance_start,
            control_guidance_end,
            width,
            height,
            seed,
        )

        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=control_image,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            controlnet_conditioning_scale=control_strength,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            generator=generator,
        )

        image = result.images[0]
        output_png = os.path.join(job_folder, "output.png")
        image.save(output_png)

        out: Dict[str, str] = {
            "control_png": control_png,
            "output_png": output_png,
        }

        upscale_cfg = meta.get("upscale") if isinstance(meta.get("upscale"), dict) else {}
        if bool(upscale_cfg.get("enabled", False)):
            factor = _to_int(upscale_cfg.get("factor"), 2)
            if factor >= 2:
                final_up2x_png = os.path.join(job_folder, "final_up2x.png")
                _upscale_png(output_png, final_up2x_png, factor=factor)
                out["final_up2x_png"] = final_up2x_png

        meta["engine_family"] = "sdxl"
        meta["engine"] = "sdxl_base_1_0"
        meta["control_model"] = "TheMistoAI/MistoLine"
        meta["pipeline_key"] = "sdxl::mistoline_sketch"
        meta["preprocess"] = preprocess_meta
        meta["runtime"] = {
            "family": "sdxl",
            "runtime": "SDXLMistoLineRuntime",
            "base_model": self.sdxl_base_path,
            "controlnet_model": self.mistoline_path,
            "vae_model": self.sdxl_vae_path,
            "generated_at": datetime.utcnow().isoformat(),
            "scheduler": "DPMSolverMultistepScheduler",
            "scheduler_algorithm_type": "sde-dpmsolver++",
            "use_karras_sigmas": True,
        }
        meta["mistoline_generation_applied"] = {
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "controlnet_conditioning_scale": control_strength,
            "control_guidance_start": control_guidance_start,
            "control_guidance_end": control_guidance_end,
            "width": width,
            "height": height,
            "source": "runtime_defaults_aligned_to_mistoline_page",
        }

        meta_path = os.path.join(job_folder, "meta.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4, ensure_ascii=False)
        except Exception:
            pass

        return out