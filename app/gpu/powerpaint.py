# app/gpu/powerpaint.py
"""
RENDEREXPO AI STUDIO - PowerPaint GPU Runner

LOCKED SERVICE FAMILY:
    AI Interior Cleanup & Small Decor Enhancement

SUPPORTED JOB TYPES:
    1) powerpaint_object_removal
       - Service name: AI Object Removal
       - Purpose: remove masked objects / clutter / visual distractions

    2) powerpaint_small_decor_insert
       - Service name: AI Small Decor Enhancement / Micro-Staging
       - Purpose: insert small-scale decor only, such as bowls, trays, vases,
         sculptural accessories, and subtle styling objects

EXPLICITLY NOT SUPPORTED HERE:
    - furniture staging
    - product staging
    - chair / sofa / bed insertion
    - reference-guided furniture insertion
    - IP-Adapter workflows

Those belong to future Option A:
    Reference-Guided Furniture / Product Staging
    using IP-Adapter + SDXL 1.0 after separate validation.

IMPORTANT SAFETY:
    - This module is lazy-loaded only by GPU dispatch.
    - It must not affect SD3.5, MistoLine, moodboard, sketch, img2img,
      text2img, video, CAD, mesh, VR, or existing services.
    - It writes only inside the provided job_folder.
    - It expects planner to save input.png and mask.png.
    - It returns output.png and metadata for dispatch to store in meta.json.
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_model
from transformers import CLIPTextModel
from diffusers import UniPCMultistepScheduler

# ---------------------------------------------------------------------------
# Paths / Environment
# ---------------------------------------------------------------------------

# Keep defaults aligned with the tested sandbox.
DEFAULT_POWERPAINT_REPO = "/workspace-data/RENDEREXPO-powerpaint-sandbox/PowerPaint"
DEFAULT_POWERPAINT_CHECKPOINT = "/workspace-data/models/powerpaint/PowerPaint-v2-1"

POWERPAINT_REPO = Path(os.getenv("RENDEREXPO_POWERPAINT_REPO", DEFAULT_POWERPAINT_REPO)).resolve()
CHECKPOINT_DIR = Path(os.getenv("RENDEREXPO_POWERPAINT_MODEL_DIR", DEFAULT_POWERPAINT_CHECKPOINT)).resolve()

BASE_MODEL_PATH = CHECKPOINT_DIR / "realisticVisionV60B1_v51VAE"
BRUSHNET_DIR = CHECKPOINT_DIR / "PowerPaint_Brushnet"

# Make PowerPaint repo importable without touching global app imports.
if str(POWERPAINT_REPO) not in sys.path:
    sys.path.insert(0, str(POWERPAINT_REPO))

# PowerPaint imports must happen after sys.path injection.
from powerpaint.models.BrushNet_CA import BrushNetModel  # noqa: E402
from powerpaint.models.unet_2d_condition import UNet2DConditionModel  # noqa: E402
from powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA import (  # noqa: E402
    StableDiffusionPowerPaintBrushNetPipeline,
)
from powerpaint.utils.utils import TokenizerWrapper, add_tokens  # noqa: E402


# ---------------------------------------------------------------------------
# Locked service constants
# ---------------------------------------------------------------------------

JOB_TYPE_OBJECT_REMOVAL = "powerpaint_object_removal"
JOB_TYPE_SMALL_DECOR = "powerpaint_small_decor_insert"

VALID_JOB_TYPES = {
    JOB_TYPE_OBJECT_REMOVAL,
    JOB_TYPE_SMALL_DECOR,
}

DEFAULT_STEPS = 30

# These defaults come from the successful sandbox tests.
OBJECT_REMOVAL_DEFAULT_PROMPT = "empty rug, clean floor"
OBJECT_REMOVAL_DEFAULT_NEGATIVE = "table, bowl, decor, object, furniture, artifact, blurry"

SMALL_DECOR_DEFAULT_PROMPT = (
    "small bronze decorative bowl on a low tray, luxury staged decor, "
    "tabletop scale, realistic contact shadow, subtle reflection"
)
SMALL_DECOR_DEFAULT_NEGATIVE = (
    "chair, sofa, table, large furniture, oversized object, floating object, "
    "wrong scale, blurry, low quality, distorted, warped, heavy reflection"
)


# ---------------------------------------------------------------------------
# Runtime singleton
# ---------------------------------------------------------------------------

_RUNTIME: Optional["PowerPaintRuntime"] = None


def _utc_epoch() -> int:
    return int(time.time())


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"{label} is empty: {path}")


def _ensure_dir(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if not path.is_dir():
        raise FileNotError(f"{label} is not a directory: {path}")


def _load_image(path: Path, label: str) -> Image.Image:
    _ensure_file(path, label)
    return Image.open(path).convert("RGB")


def _add_task_tokens(prompt: str, negative_prompt: str, task: str) -> Tuple[str, str, str, str]:
    """
    Same task-token behavior as the successful sandbox helper.

    object-removal:
        P_ctxt

    shape-guided:
        P_shape

    text-guided object insertion:
        P_obj
    """
    prompt = (prompt or "").strip()
    negative_prompt = (negative_prompt or "").strip()

    if task in ("object-removal", "image-outpainting"):
        prompt_a = prompt + " P_ctxt"
        prompt_b = prompt + " P_ctxt"
        negative_a = negative_prompt
        negative_b = negative_prompt
    elif task == "shape-guided":
        prompt_a = prompt + " P_shape"
        prompt_b = prompt + " P_shape"
        negative_a = negative_prompt
        negative_b = negative_prompt
    else:
        prompt_a = prompt + " P_obj"
        prompt_b = prompt + " P_obj"
        negative_a = negative_prompt
        negative_b = negative_prompt

    return prompt_a, prompt_b, negative_a, negative_b


class PowerPaintRuntime:
    """
    Loaded once per GPU worker process.

    This mirrors the proven standalone sandbox loader but keeps it isolated
    inside app/gpu/powerpaint.py.
    """

    def __init__(self, weight_dtype: torch.dtype = torch.float16) -> None:
        print("[PowerPaint] Loading PowerPaint v2-1 runtime...")
        print("[PowerPaint] repo:", POWERPAINT_REPO)
        print("[PowerPaint] checkpoint:", CHECKPOINT_DIR)
        print("[PowerPaint] base_model:", BASE_MODEL_PATH)
        print("[PowerPaint] brushnet:", BRUSHNET_DIR)

        _ensure_dir(POWERPAINT_REPO, "PowerPaint repo")
        _ensure_dir(CHECKPOINT_DIR, "PowerPaint checkpoint directory")
        _ensure_dir(BASE_MODEL_PATH, "PowerPaint base model")
        _ensure_dir(BRUSHNET_DIR, "PowerPaint BrushNet directory")
        _ensure_file(BRUSHNET_DIR / "diffusion_pytorch_model.safetensors", "PowerPaint BrushNet safetensors")
        _ensure_file(BRUSHNET_DIR / "pytorch_model.bin", "PowerPaint BrushNet text encoder weights")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. PowerPaint GPU runner requires CUDA.")

        unet = UNet2DConditionModel.from_pretrained(
            str(BASE_MODEL_PATH),
            subfolder="unet",
            torch_dtype=weight_dtype,
            local_files_only=True,
        )

        text_encoder_brushnet = CLIPTextModel.from_pretrained(
            str(BASE_MODEL_PATH),
            subfolder="text_encoder",
            torch_dtype=weight_dtype,
            local_files_only=True,
        )

        brushnet = BrushNetModel.from_unet(unet)

        self.pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
            str(BASE_MODEL_PATH),
            brushnet=brushnet,
            text_encoder_brushnet=text_encoder_brushnet,
            torch_dtype=weight_dtype,
            low_cpu_mem_usage=False,
            safety_checker=None,
            local_files_only=True,
        )

        self.pipe.unet = UNet2DConditionModel.from_pretrained(
            str(BASE_MODEL_PATH),
            subfolder="unet",
            torch_dtype=weight_dtype,
            local_files_only=True,
        )

        self.pipe.tokenizer = TokenizerWrapper(
            from_pretrained=str(BASE_MODEL_PATH),
            subfolder="tokenizer",
            revision=None,
            torch_type=weight_dtype,
            local_files_only=True,
        )

        add_tokens(
            tokenizer=self.pipe.tokenizer,
            text_encoder=self.pipe.text_encoder_brushnet,
            placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
            initialize_tokens=["a", "a", "a"],
            num_vectors_per_token=10,
        )

        load_model(
            self.pipe.brushnet,
            str(BRUSHNET_DIR / "diffusion_pytorch_model.safetensors"),
        )

        self.pipe.text_encoder_brushnet.load_state_dict(
            torch.load(str(BRUSHNET_DIR / "pytorch_model.bin"), map_location="cpu"),
            strict=False,
        )

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        # Keep same behavior as sandbox tests.
        self.pipe.enable_model_cpu_offload()
        self.pipe = self.pipe.to("cuda")

        print("[PowerPaint] PowerPaint v2-1 runtime loaded.")

    def run(
        self,
        *,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str,
        seed: int,
        steps: int,
        guidance_scale: float,
        fitting_degree: float,
        task: str,
    ) -> Image.Image:
        image = image.convert("RGB")
        mask = mask.convert("RGB")

        # Match the successful sandbox sizing behavior.
        size1, size2 = image.size
        if size1 < size2:
            image = image.resize((640, int(size2 / size1 * 640)))
        else:
            image = image.resize((int(size1 / size2 * 640), 640))

        img_np = np.array(image)
        w_runtime = int(np.shape(img_np)[0] - np.shape(img_np)[0] % 8)
        h_runtime = int(np.shape(img_np)[1] - np.shape(img_np)[1] % 8)

        image = image.resize((h_runtime, w_runtime))
        mask = mask.resize((h_runtime, w_runtime))

        _set_seed(seed)

        task = (task or "text-guided").strip()

        # Match official / sandbox behavior for removal and outpainting cues.
        if task == "object-removal":
            prompt = (prompt.strip() + " empty scene blur").strip()
        elif task == "image-outpainting":
            prompt = (prompt.strip() + " empty scene").strip()

        prompt_a, prompt_b, negative_a, negative_b = _add_task_tokens(prompt, negative_prompt, task)

        # Same v2 BrushNet masking behavior as the working sandbox.
        np_inpimg = np.array(image)
        np_inmask = np.array(mask) / 255.0
        np_inpimg = np_inpimg * (1 - np_inmask)
        masked_image = Image.fromarray(np_inpimg.astype(np.uint8)).convert("RGB")

        print("[PowerPaint] task:", task)
        print("[PowerPaint] promptA:", prompt_a)
        print("[PowerPaint] promptB:", prompt_b)
        print("[PowerPaint] negativeA:", negative_a)
        print("[PowerPaint] negativeB:", negative_b)
        print("[PowerPaint] size:", h_runtime, w_runtime)
        print("[PowerPaint] steps:", steps)
        print("[PowerPaint] guidance_scale:", guidance_scale)
        print("[PowerPaint] fitting_degree:", fitting_degree)
        print("[PowerPaint] seed:", seed)

        result = self.pipe(
            promptA=prompt_a,
            promptB=prompt_b,
            promptU=prompt,
            tradoff=fitting_degree,
            tradoff_nag=fitting_degree,
            image=masked_image,
            mask=mask,
            num_inference_steps=steps,
            generator=torch.Generator("cuda").manual_seed(seed),
            brushnet_conditioning_scale=1.0,
            negative_promptA=negative_a,
            negative_promptB=negative_b,
            negative_promptU=negative_prompt,
            guidance_scale=guidance_scale,
            width=h_runtime,
            height=w_runtime,
        ).images[0]

        return result.convert("RGB")


def _get_runtime() -> PowerPaintRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = PowerPaintRuntime()
    return _RUNTIME


def _resolve_seed(payload: Dict[str, Any]) -> int:
    raw = payload.get("seed")
    if raw is None:
        return int(random.randint(1, 2_147_483_000))
    try:
        return int(raw)
    except Exception:
        return int(random.randint(1, 2_147_483_000))


def _resolve_common_paths(job_folder: str) -> Dict[str, Path]:
    jf = Path(job_folder).resolve()
    if not jf.exists() or not jf.is_dir():
        raise FileNotFoundError(f"job_folder does not exist: {jf}")

    input_path = jf / "input.png"
    mask_path = jf / "mask.png"
    output_path = jf / "output.png"

    _ensure_file(input_path, "PowerPaint input.png")
    _ensure_file(mask_path, "PowerPaint mask.png")

    return {
        "job_folder": jf,
        "input": input_path,
        "mask": mask_path,
        "output": output_path,
    }


def _save_result(result: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    _ensure_file(output_path, "PowerPaint output.png")


def run_powerpaint_object_removal(
    *,
    job: Dict[str, Any],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    GPU runner for:
        job_type = powerpaint_object_removal

    Required files in job_folder:
        input.png
        mask.png

    Output:
        output.png
    """
    job_folder = str(payload.get("job_folder") or "").strip()
    paths = _resolve_common_paths(job_folder)

    prompt = str(payload.get("prompt") or OBJECT_REMOVAL_DEFAULT_PROMPT).strip()
    negative_prompt = str(payload.get("negative_prompt") or OBJECT_REMOVAL_DEFAULT_NEGATIVE).strip()

    seed = _resolve_seed(payload)
    steps = int(payload.get("steps") or DEFAULT_STEPS)
    guidance_scale = float(payload.get("guidance_scale") or 6.5)
    fitting_degree = float(payload.get("fitting_degree") or 1.0)

    image = _load_image(paths["input"], "PowerPaint object-removal input")
    mask = _load_image(paths["mask"], "PowerPaint object-removal mask")

    runtime = _get_runtime()

    started_at = _utc_epoch()

    result = runtime.run(
        image=image,
        mask=mask,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        steps=steps,
        guidance_scale=guidance_scale,
        fitting_degree=fitting_degree,
        task="object-removal",
    )

    _save_result(result, paths["output"])

    return {
        "output_png": str(paths["output"]),
        "input_image": str(paths["input"]),
        "mask_image": str(paths["mask"]),
        "mode": "powerpaint_object_removal",
        "service_name": "AI Object Removal",
        "engine": "PowerPaint-v2-1",
        "engine_family": "powerpaint",
        "task": "object-removal",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "fitting_degree": fitting_degree,
        "started_at_epoch": started_at,
        "finished_at_epoch": _utc_epoch(),
    }


def run_powerpaint_small_decor_insert(
    *,
    job: Dict[str, Any],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    GPU runner for:
        job_type = powerpaint_small_decor_insert

    Required files in job_folder:
        input.png
        mask.png

    Output:
        output.png
    """
    job_folder = str(payload.get("job_folder") or "").strip()
    paths = _resolve_common_paths(job_folder)

    prompt = str(payload.get("prompt") or SMALL_DECOR_DEFAULT_PROMPT).strip()
    negative_prompt = str(payload.get("negative_prompt") or SMALL_DECOR_DEFAULT_NEGATIVE).strip()

    seed = _resolve_seed(payload)
    steps = int(payload.get("steps") or DEFAULT_STEPS)
    guidance_scale = float(payload.get("guidance_scale") or 6.5)
    fitting_degree = float(payload.get("fitting_degree") or 0.55)

    image = _load_image(paths["input"], "PowerPaint small-decor input")
    mask = _load_image(paths["mask"], "PowerPaint small-decor mask")

    runtime = _get_runtime()

    started_at = _utc_epoch()

    result = runtime.run(
        image=image,
        mask=mask,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        steps=steps,
        guidance_scale=guidance_scale,
        fitting_degree=fitting_degree,
        task="text-guided",
    )

    _save_result(result, paths["output"])

    return {
        "output_png": str(paths["output"]),
        "input_image": str(paths["input"]),
        "mask_image": str(paths["mask"]),
        "mode": "powerpaint_small_decor_insert",
        "service_name": "AI Small Decor Enhancement / Micro-Staging",
        "engine": "PowerPaint-v2-1",
        "engine_family": "powerpaint",
        "task": "text-guided",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "fitting_degree": fitting_degree,
        "started_at_epoch": started_at,
        "finished_at_epoch": _utc_epoch(),
    }