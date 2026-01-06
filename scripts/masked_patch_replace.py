import os
import argparse
import subprocess
from PIL import Image, ImageFilter


def bbox_from_mask(mask: Image.Image):
    return mask.getbbox()  # (x0,y0,x1,y1) or None


def expand_box(box, pad: int, W: int, H: int):
    x0, y0, x1, y1 = box
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W, x1 + pad)
    y1 = min(H, y1 + pad)
    return (x0, y0, x1, y1)


def feather(m: Image.Image, r: int):
    r = max(0, int(r))
    return m.filter(ImageFilter.GaussianBlur(radius=r)) if r > 0 else m


def run_inject(in_path: str, out_path: str, args):
    cmd = [
        "python3",
        args.inject_script,
        "--base",
        args.base,
        "--in",
        in_path,
        "--out",
        out_path,
        "--prompt",
        args.prompt,
        "--neg",
        args.neg,
        "--steps",
        str(args.steps),
        "--cfg",
        str(args.cfg),
        "--denoise",
        str(args.denoise),
        "--seed",
        str(args.seed),
    ]
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser(
        description="Masked patch replace (pseudo-inpaint): enhance only masked region and blend back."
    )
    ap.add_argument("--in", dest="img_in", required=True)
    ap.add_argument("--mask", required=True, help="Mask PNG (white=edit, black=keep). Same size as image.")
    ap.add_argument("--out", required=True)

    ap.add_argument("--prompt", required=True)
    ap.add_argument("--neg", default="")

    ap.add_argument("--steps", type=int, default=44)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--denoise", type=float, default=0.26)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--pad", type=int, default=128, help="Padding around mask bbox.")
    ap.add_argument("--feather", type=int, default=96, help="Feather blur for mask edges.")

    ap.add_argument(
        "--base",
        default=os.getenv("BASE_SD35", os.getenv("BASE", "/workspace-data/models/sd35-large")),
    )
    ap.add_argument("--inject_script", default="scripts/sd35_r1b_material_inject.py")

    args = ap.parse_args()

    im = Image.open(args.img_in).convert("RGB")
    mask = Image.open(args.mask).convert("L")

    if im.size != mask.size:
        raise SystemExit("Mask size must match image size.")

    W, H = im.size
    bb = bbox_from_mask(mask)
    if bb is None:
        raise SystemExit("Mask is empty (no white pixels).")

    box = expand_box(bb, args.pad, W, H)
    x0, y0, x1, y1 = box

    patch_in = im.crop(box)
    patch_mask = mask.crop(box)

    tmp = "/tmp/masked_patch"
    os.makedirs(tmp, exist_ok=True)
    tin = os.path.join(tmp, "patch_in.png")
    tout = os.path.join(tmp, "patch_out.png")
    patch_in.save(tin)

    run_inject(tin, tout, args)
    patch_out = Image.open(tout).convert("RGB")

    alpha = feather(patch_mask, args.feather)

    base_region = im.crop(box)
    merged = Image.composite(patch_out, base_region, alpha)

    out = im.copy()
    out.paste(merged, (x0, y0))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.save(args.out)
    print(f"[masked-patch] Saved: {args.out}")


if __name__ == "__main__":
    main()
