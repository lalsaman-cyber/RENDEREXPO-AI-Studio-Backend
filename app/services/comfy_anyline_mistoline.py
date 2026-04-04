from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from app.services.comfy_client import ComfyUIClient, ComfyUIError


DEFAULT_NEGATIVE_PROMPT = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, "
    "cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, "
    "username, blurry, artist name, trademark, title, multiple view, mutated hands and fingers, "
    "poorly drawn face, mutation, deformed, ugly, bad proportions, malformed limbs, extra arms, "
    "extra legs, fused fingers, too many fingers, long neck, odd eyes, uneven eyes, unnatural face, "
    "crooked mouth, bad teeth, bad perspective, black and white, oversaturated, undersaturated, "
    "bad shadow, draft, grainy, pixelated, blurry background, ugly background, simple background, realistic"
)


@dataclass
class SketchJobConfig:
    comfy_url: str
    sdxl_checkpoint_name: str
    controlnet_name: str = "mistoLine_rank256.safetensors"
    sampler_name: str = "dpmpp_2m_sde"
    scheduler: str = "karras"
    steps: int = 30
    cfg: float = 7.0
    denoise: float = 0.93
    control_strength: float = 1.0
    start_percent: float = 0.0
    end_percent: float = 0.9
    output_prefix: str = "renderexpo_sketch"
    poll_timeout: int = 900


class AnylineMistolineSketchService:
    def __init__(self, config: SketchJobConfig) -> None:
        self.config = config
        self.client = ComfyUIClient(
            base_url=config.comfy_url,
            poll_timeout=config.poll_timeout,
        )

    @staticmethod
    def _round_dimension(value: int) -> int:
        return max(64, int(round(value / 64.0) * 64))

    @classmethod
    def _read_image_size(cls, image_path: str) -> Tuple[int, int]:
        with Image.open(image_path) as img:
            width, height = img.size
        return cls._round_dimension(width), cls._round_dimension(height)

    @staticmethod
    def _build_uploaded_image_value(uploaded_filename: str, uploaded_subfolder: str) -> str:
        uploaded_subfolder = (uploaded_subfolder or "").strip().strip("/\\")
        if uploaded_subfolder:
            return f"{uploaded_subfolder}/{uploaded_filename}"
        return uploaded_filename

    def _build_prompt_workflow(
        self,
        uploaded_filename: str,
        uploaded_subfolder: str,
        width: int,
        height: int,
        prompt: str,
        negative_prompt: Optional[str],
        seed: Optional[int],
    ) -> Dict[str, Any]:
        seed_value = seed if seed is not None else random.randint(1, 2**31 - 1)
        neg = negative_prompt or DEFAULT_NEGATIVE_PROMPT
        uploaded_image_value = self._build_uploaded_image_value(
            uploaded_filename=uploaded_filename,
            uploaded_subfolder=uploaded_subfolder,
        )

        return {
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": self.config.sdxl_checkpoint_name,
                },
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["4", 1],
                    "text": prompt,
                },
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["4", 1],
                    "text": neg,
                },
            },
            "10": {
                "class_type": "LoadImage",
                "inputs": {
                    "image": uploaded_image_value,
                    "upload": "image",
                },
            },
            "38": {
                "class_type": "AnyLinePreprocessor",
                "inputs": {
                    "image": ["10", 0],
                },
            },
            "13": {
                "class_type": "ControlNetLoader",
                "inputs": {
                    "control_net_name": self.config.controlnet_name,
                },
            },
            "14": {
                "class_type": "ControlNetApplyAdvanced",
                "inputs": {
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "control_net": ["13", 0],
                    "image": ["38", 0],
                    "strength": self.config.control_strength,
                    "start_percent": self.config.start_percent,
                    "end_percent": self.config.end_percent,
                },
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1,
                },
            },
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed_value,
                    "steps": self.config.steps,
                    "cfg": self.config.cfg,
                    "sampler_name": self.config.sampler_name,
                    "scheduler": self.config.scheduler,
                    "denoise": self.config.denoise,
                    "model": ["4", 0],
                    "positive": ["14", 0],
                    "negative": ["14", 1],
                    "latent_image": ["5", 0],
                },
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2],
                },
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": self.config.output_prefix,
                    "images": ["8", 0],
                },
            },
        }

    @staticmethod
    def _extract_output_images(history_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        outputs = history_item.get("outputs", {})
        images: List[Dict[str, Any]] = []

        for node_output in outputs.values():
            node_images = node_output.get("images", [])
            for image_info in node_images:
                if "filename" in image_info:
                    images.append(image_info)

        if not images:
            raise ComfyUIError(f"No output images found in history item: {history_item}")

        return images

    def run(
        self,
        input_image_path: str,
        output_dir: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        self.client.check_server()

        width, height = self._read_image_size(input_image_path)

        upload_info = self.client.upload_image(
            image_path=input_image_path,
            subfolder="renderexpo_inputs",
        )

        uploaded_filename = upload_info["name"]
        uploaded_subfolder = upload_info.get("subfolder", "")

        workflow = self._build_prompt_workflow(
            uploaded_filename=uploaded_filename,
            uploaded_subfolder=uploaded_subfolder,
            width=width,
            height=height,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
        )

        prompt_id = self.client.queue_prompt(workflow)
        history_item = self.client.wait_for_completion(prompt_id)
        image_infos = self._extract_output_images(history_item)

        os.makedirs(output_dir, exist_ok=True)
        saved_paths: List[str] = []

        for idx, image_info in enumerate(image_infos, start=1):
            filename = image_info["filename"]
            subfolder = image_info.get("subfolder", "")
            folder_type = image_info.get("type", "output")

            ext = os.path.splitext(filename)[1] or ".png"
            local_path = os.path.join(output_dir, f"sketch_result_{idx}{ext}")

            self.client.download_output(
                filename=filename,
                subfolder=subfolder,
                folder_type=folder_type,
                destination_path=local_path,
            )
            saved_paths.append(local_path)

        return {
            "engine": "comfy_anyline_mistoline_rank256",
            "prompt_id": prompt_id,
            "width": width,
            "height": height,
            "checkpoint": self.config.sdxl_checkpoint_name,
            "controlnet": self.config.controlnet_name,
            "outputs": saved_paths,
        }