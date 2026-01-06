cat > scripts/sd35_r1b_material_inject.py <<'PY'
#!/usr/bin/env python3
import os
import argparse
from typing import Optional

import torch
from PIL import Image

from diffusers import StableDiffusion3Img2ImgPipeline


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if (v is not None and str(v).strip() != "") else default


def load_image_rgb(path: str) -> Image.Image:
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def main():
    ap = argparse.ArgumentParser(
        description="SD3.5 R1B material inject (img2img) - keep composition, inject micro-detail."
    )

    ap.add_argument("--base", default=_env("BASE_SD35", _env("BASE", "/workspace-data/models/sd35-large")),
                    help="Path to SD3.5 model folder.")
    ap.add_argument("--in", dest="img_in", default=_env("IMG_IN"),
                    help="Input image path.")
    ap.add_argument("--out", dest="out", default=_env("OUT"),
                    help="Output image path.")
    ap.add_argument("--prompt", default=_env("PROMPT", ""),
                    help="Positive prompt.")
    ap.add_argument("--neg", default=_env("NEG", ""),
                    help="Negative prompt.")
    ap.add_argument("--steps", type=int, default=int(_env("STEPS", "28")), help="Steps.")
    ap.add_argument("--cfg", type=float, default=float(_env("CFG_SCALE", _env("CFG", "5.4"))), help="CFG scale.")
    ap.add_argument("--denoise", type=float, default=float(_env("DENOISE", "0.14")), help="Strength/denoise (0-1).")
    ap.add_argument("--seed", type=int, default=int(_env("SEED", "0")), help="Seed (0=random).")

    # Dtype control
    prec = ap.add_mutually_exclusive_group()
    prec.add_argument("--fp16", action="store_true", help="Force FP16 on GPU.")
    prec.add_argument("--fp32", action="store_true", help="Force FP32 (NOT recommended on GPU).")
    prec.add_argument("--bf16", action="store_true", help="Force BF16 on GPU (DEFAULT).")

    args = ap.parse_args()

    if not args.img_in or not os.path.exists(args.img_in):
        raise SystemExit(f"Input not found: {args.img_in}")
    if not args.out:
        raise SystemExit("Missing --out (or OUT env var).")
    if not args.prompt.strip():
        raise SystemExit("PROMPT is empty. R1B must have an explicit material-only prompt.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # DEFAULT DTYPE POLICY:
    # - CUDA: BF16 (safe)
    # - CPU: FP32
    if device == "cuda":
        if args.fp32:
            dtype = torch.float32
        elif args.fp16:
            dtype = torch.float16
        else:
            dtype = torch.bfloat16
    else:
        dtype = torch.float32

    img = load_image_rgb(args.img_in)
    w, h = img.size

    if args.seed == 0:
        gen = None
    else:
        gen = torch.Generator(device=device).manual_seed(args.seed)

    print(f"[R1B] base={args.base}")
    print(f"[R1B] in={args.img_in} size={w}x{h}")
    print(f"[R1B] steps={args.steps} cfg={args.cfg} denoise={args.denoise} seed={args.seed}")
    print(f"[R1B] device={device} dtype={dtype}")
    print(f"[R1B] out={args.out}")

    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(args.base, torch_dtype=dtype).to(device)

    # IMPORTANT: Do NOT resize. Keep native resolution.
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.neg,
        image=img,
        strength=args.denoise,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        generator=gen,
    ).images[0].convert("RGB")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    result.save(args.out)
    print(f"[R1B] Saved: {args.out}")


if __name__ == "__main__":
    main()
PY

chmod +x scripts/sd35_r1b_material_inject.py
python3 scripts/sd35_r1b_material_inject.py --help | head
