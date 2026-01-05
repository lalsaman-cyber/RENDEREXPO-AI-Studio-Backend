#!/usr/bin/env python3
import os
import argparse
from typing import Optional

import torch
from PIL import Image, ImageFilter, ImageOps

# Keep the same import you already use
from diffusers import StableDiffusion3Img2ImgPipeline


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if (v is not None and str(v).strip() != "") else default


def load_image_rgb(path: str) -> Image.Image:
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def load_mask_l(path: str, size: tuple[int, int], invert: bool, feather: int) -> Image.Image:
    m = Image.open(path)
    if m.mode != "L":
        m = m.convert("L")
    if m.size != size:
        m = m.resize(size, Image.Resampling.LANCZOS)
    if invert:
        m = ImageOps.invert(m)
    if feather and feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=feather))
    return m


def main():
    ap = argparse.ArgumentParser(
        description="SD3.5 Pass2 img2img refine (optionally masked) at native resolution."
    )

    ap.add_argument("--base", default=_env("BASE_SD35", _env("BASE", "/workspace-data/models/sd35-large")),
                    help="Path to SD3.5 model folder.")
    ap.add_argument("--in", dest="img_in", default=_env("IMG_IN", _env("IMG")),
                    help="Input image path.")
    ap.add_argument("--out", dest="out", default=_env("OUT"),
                    help="Output image path.")
    ap.add_argument("--prompt", default=_env("PROMPT", "ultra-photorealistic architectural photo, crisp, clean, realistic"),
                    help="Positive prompt.")
    ap.add_argument("--neg", default=_env("NEG", "smudge, blur, painterly, warped, bent lines, melted, artifacts, noise"),
                    help="Negative prompt.")
    ap.add_argument("--steps", type=int, default=int(_env("STEPS", "26")), help="Steps.")
    ap.add_argument("--cfg", type=float, default=float(_env("CFG_SCALE", "6.0")), help="CFG scale.")
    ap.add_argument("--denoise", type=float, default=float(_env("DENOISE", "0.20")), help="Strength/denoise (0-1).")
    ap.add_argument("--seed", type=int, default=int(_env("SEED", "0")), help="Seed (0 means random).")

    # Masked refine (generic, not skyscraper-specific)
    ap.add_argument("--mask", default=_env("MASK", None), help="Optional mask path (white=apply refine).")
    ap.add_argument("--invert-mask", action="store_true", help="Invert mask (black=apply refine).")
    ap.add_argument("--feather", type=int, default=int(_env("FEATHER", "24")),
                    help="Mask feather blur radius in pixels (0 disables).")

    # Precision controls
    # IMPORTANT DEFAULT: BF16 on CUDA to prevent FP32 OOM.
    prec = ap.add_mutually_exclusive_group()
    prec.add_argument("--fp16", action="store_true", help="Force FP16 on GPU.")
    prec.add_argument("--fp32", action="store_true", help="Force FP32 (NOT recommended on GPU for SD3.5 Large).")
    prec.add_argument("--bf16", action="store_true", help="Force BF16 on GPU (DEFAULT behavior).")

    args = ap.parse_args()

    if not args.img_in or not os.path.exists(args.img_in):
        raise SystemExit(f"Input not found: {args.img_in}")
    if not args.out:
        raise SystemExit("Missing --out (or OUT env var).")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # DEFAULT DTYPE POLICY:
    # - CUDA: BF16 by default (safe)
    # - CPU: FP32
    if device == "cuda":
        if args.fp32:
            dtype = torch.float32
        elif args.fp16:
            dtype = torch.float16
        else:
            # default (also if --bf16 passed)
            dtype = torch.bfloat16
    else:
        dtype = torch.float32

    img = load_image_rgb(args.img_in)
    w, h = img.size

    # Seed / generator (use the correct device)
    if args.seed == 0:
        gen = None
    else:
        gen = torch.Generator(device=device).manual_seed(args.seed)

    print(f"[Pass2] base={args.base}")
    print(f"[Pass2] in={args.img_in} size={w}x{h}")
    print(f"[Pass2] steps={args.steps} cfg={args.cfg} denoise={args.denoise} seed={args.seed}")
    print(f"[Pass2] masked={bool(args.mask)} feather={args.feather} invert={args.invert_mask}")
    print(f"[Pass2] device={device} dtype={dtype}")

    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
        args.base,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)

    # IMPORTANT: do NOT resize. Keep native resolution.
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.neg,
        image=img,
        strength=args.denoise,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        generator=gen,
    ).images[0].convert("RGB")

    # If mask provided, composite: keep original outside mask
    if args.mask:
        m = load_mask_l(args.mask, size=img.size, invert=args.invert_mask, feather=args.feather)
        final = Image.composite(result, img, m)
    else:
        final = result

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    final.save(args.out)
    print(f"[Pass2] Saved: {args.out}")


if __name__ == "__main__":
    main()
