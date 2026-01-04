#!/usr/bin/env python3
"""
RENDEREXPO AI Studio — SD3.5 Pass2 Img2Img Refiner (General + Masked Region)

Key features:
- Preserves input resolution (e.g., 2048 stays 2048).
- Optional region-focused refinement via:
  - MASK_MODE=none|bottom|top|left|right|center|box|file
  - MASK_PCT (used by bottom/top/left/right/center), default 0.30
  - MASK_BOX="x0,y0,x1,y1" (pixels, for MASK_MODE=box)
  - MASK_IN="path/to/mask.png" (white=refine, black=keep), for MASK_MODE=file
  - MASK_FEATHER (blur radius, default 24) for seamless blending
- Applies up to 2 LoRA/LyCORIS adapters:
  - LORA1_CKPT + LORA1_WEIGHT
  - LORA2_CKPT + LORA2_WEIGHT
- Uses:
  - BASE_SD35 (preferred) or BASE (fallback)
  - IMG_IN, OUT, PROMPT, NEG, STEPS, CFG_SCALE, DENOISE, SEED

This script does NOT require an inpaint pipeline.
It refines a full img2img output, then blends ONLY the masked region back into the original.
This makes "fix only cars/trees/text areas" possible without destroying the tower.
"""

import os
import sys
import math
import time
from typing import Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageFilter

# Diffusers import: SD3.5 image-to-image pipeline
try:
    from diffusers import StableDiffusion3Img2ImgPipeline
except Exception as e:
    raise RuntimeError(
        "Failed to import StableDiffusion3Img2ImgPipeline from diffusers. "
        "Check your diffusers version/environment.\n"
        f"Import error: {e}"
    )


def _env(name: str, default: Optional[str] = None) -> str:
    v = os.environ.get(name, default)
    if v is None or str(v).strip() == "":
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, None)
    if v is None or str(v).strip() == "":
        return float(default)
    return float(v)


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, None)
    if v is None or str(v).strip() == "":
        return int(default)
    return int(v)


def _pick_base() -> str:
    # Prefer BASE_SD35; fall back to BASE for compatibility with older scripts
    base = os.environ.get("BASE_SD35", "").strip()
    if not base:
        base = os.environ.get("BASE", "").strip()
    if not base:
        raise RuntimeError("BASE_SD35 (preferred) or BASE must be set to the SD3.5 model folder.")
    if not os.path.isdir(base):
        raise RuntimeError(f"Base model folder not found: {base}")
    return base


def _parse_box(s: str) -> Tuple[int, int, int, int]:
    # "x0,y0,x1,y1"
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise RuntimeError('MASK_BOX must be "x0,y0,x1,y1" (4 comma-separated ints).')
    return tuple(int(p) for p in parts)  # type: ignore


def build_mask(
    w: int,
    h: int,
    mode: str,
    pct: float,
    feather: int,
    box: Optional[Tuple[int, int, int, int]],
    mask_in: Optional[str],
) -> Image.Image:
    """
    Returns L-mode mask: white=refine area, black=keep original.
    """
    mode = (mode or "none").strip().lower()

    if mode == "none":
        return Image.new("L", (w, h), 255)  # full refine

    if mode == "file":
        if not mask_in:
            raise RuntimeError("MASK_MODE=file requires MASK_IN to be set.")
        if not os.path.isfile(mask_in):
            raise RuntimeError(f"MASK_IN not found: {mask_in}")
        m = Image.open(mask_in).convert("L")
        if m.size != (w, h):
            m = m.resize((w, h), Image.BILINEAR)
        if feather > 0:
            m = m.filter(ImageFilter.GaussianBlur(radius=feather))
        return m

    # presets and box mode build a fresh mask
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)

    pct = max(0.01, min(0.95, float(pct)))

    if mode in ("bottom", "top"):
        band = int(h * pct)
        if mode == "bottom":
            d.rectangle([0, h - band, w, h], fill=255)
        else:
            d.rectangle([0, 0, w, band], fill=255)

    elif mode in ("left", "right"):
        band = int(w * pct)
        if mode == "left":
            d.rectangle([0, 0, band, h], fill=255)
        else:
            d.rectangle([w - band, 0, w, h], fill=255)

    elif mode == "center":
        # center box covering pct of width/height
        cw = int(w * pct)
        ch = int(h * pct)
        x0 = (w - cw) // 2
        y0 = (h - ch) // 2
        d.rectangle([x0, y0, x0 + cw, y0 + ch], fill=255)

    elif mode == "box":
        if box is None:
            raise RuntimeError("MASK_MODE=box requires MASK_BOX.")
        x0, y0, x1, y1 = box
        # clamp
        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        x1 = max(1, min(w, x1))
        y1 = max(1, min(h, y1))
        if x1 <= x0 or y1 <= y0:
            raise RuntimeError("MASK_BOX invalid after clamping.")
        d.rectangle([x0, y0, x1, y1], fill=255)

    else:
        raise RuntimeError(f"Unknown MASK_MODE: {mode}")

    if feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=feather))
    return m


def apply_lora(pipe, ckpt: str, weight: float, adapter_name: str):
    """
    Load LoRA/LyCORIS weights via diffusers, then activate with weight.
    This matches the pattern that works in diffusers for many LoRA formats.
    """
    if not ckpt or str(ckpt).strip() == "":
        return

    if not os.path.exists(ckpt):
        raise RuntimeError(f"LoRA checkpoint not found: {ckpt}")

    # load
    pipe.load_lora_weights(ckpt, adapter_name=adapter_name)

    # set weight
    # Diffusers expects a list of adapter names and weights
    pipe.set_adapters([adapter_name], adapter_weights=[float(weight)])


def main():
    base = _pick_base()

    img_in = _env("IMG_IN")
    out = _env("OUT")
    prompt = _env("PROMPT")
    neg = os.environ.get("NEG", "").strip()

    steps = _env_int("STEPS", 26)
    cfg = _env_float("CFG_SCALE", 6.2)
    denoise = _env_float("DENOISE", 0.22)
    seed = _env_int("SEED", 0)

    # Mask controls
    mask_mode = os.environ.get("MASK_MODE", "none").strip().lower()
    mask_pct = _env_float("MASK_PCT", 0.30)
    mask_feather = _env_int("MASK_FEATHER", 24)
    mask_box = os.environ.get("MASK_BOX", "").strip()
    mask_in = os.environ.get("MASK_IN", "").strip()

    box = _parse_box(mask_box) if mask_mode == "box" else None
    mask_in = mask_in if mask_mode == "file" else None

    # Optional LoRA/LyCORIS adapters
    lora1_ckpt = os.environ.get("LORA1_CKPT", "").strip()
    lora1_w = _env_float("LORA1_WEIGHT", 0.0)
    lora2_ckpt = os.environ.get("LORA2_CKPT", "").strip()
    lora2_w = _env_float("LORA2_WEIGHT", 0.0)

    if not os.path.isfile(img_in):
        raise RuntimeError(f"IMG_IN not found: {img_in}")

    # Load image and preserve resolution
    init = Image.open(img_in).convert("RGB")
    w, h = init.size

    # Device + dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"[Pass2] Base: {base}")
    print(f"[Pass2] IMG_IN: {img_in} ({w}x{h})")
    print(f"[Pass2] OUT: {out}")
    print(f"[Pass2] steps={steps} cfg={cfg} denoise={denoise} seed={seed}")
    print(f"[Pass2] MASK_MODE={mask_mode} MASK_PCT={mask_pct} MASK_FEATHER={mask_feather} MASK_BOX={mask_box} MASK_IN={mask_in or ''}")

    t0 = time.time()
    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(base, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    # Load/activate LoRAs (optional)
    if lora1_ckpt and lora1_w > 0:
        print(f"[Pass2] Loading LoRA1: {lora1_ckpt} weight={lora1_w}")
        apply_lora(pipe, lora1_ckpt, lora1_w, adapter_name="lora1")

    if lora2_ckpt and lora2_w > 0:
        print(f"[Pass2] Loading LoRA2: {lora2_ckpt} weight={lora2_w}")
        apply_lora(pipe, lora2_ckpt, lora2_w, adapter_name="lora2")

    g = torch.Generator(device=device)
    if seed != 0:
        g.manual_seed(int(seed))

    # Run full img2img at the SAME resolution as IMG_IN
    # strength in diffusers is "denoise" here
    result = pipe(
        prompt=prompt,
        negative_prompt=neg if neg else None,
        image=init,
        strength=float(denoise),
        guidance_scale=float(cfg),
        num_inference_steps=int(steps),
        generator=g,
        height=int(h),
        width=int(w),
    )

    refined = result.images[0].convert("RGB")

    # Safety: if model returns a different size for any reason, fix it.
    if refined.size != init.size:
        refined = refined.resize(init.size, Image.LANCZOS)

    # If masking is enabled (anything except "none" which means full refine),
    # blend refined only into the masked region, keeping original elsewhere.
    # NOTE: For MASK_MODE=none we return refined as-is.
    if mask_mode != "none":
        mask = build_mask(
            w=w,
            h=h,
            mode=mask_mode,
            pct=mask_pct,
            feather=mask_feather,
            box=box,
            mask_in=mask_in,
        )
        # composite: where mask is white -> refined, black -> original
        final = Image.composite(refined, init, mask)
    else:
        final = refined

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    final.save(out)
    print(f"[Pass2] Saved: {out}")
    print(f"[Pass2] Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Pass2] Interrupted.")
        sys.exit(130)
