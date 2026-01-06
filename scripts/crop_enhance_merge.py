import os
import argparse
import subprocess
from PIL import Image, ImageFilter


def feather_mask(w: int, h: int, feather: int) -> Image.Image:
    feather = max(0, int(feather))
    m = Image.new("L", (w, h), 0)
    if feather * 2 < w and feather * 2 < h:
        m.paste(255, (feather, feather, w - feather, h - feather))
    else:
        m.paste(255, (0, 0, w, h))
    if feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=feather * 0.8))
    return m


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
    ap = argparse.ArgumentParser(description="Crop -> enhance -> merge back with feather.")
    ap.add_argument("--in", dest="img_in", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--prompt", required=True)
    ap.add_argument("--neg", default="")

    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--denoise", type=float, default=0.24)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--feather", type=int, default=120)
    ap.add_argument("--auto", choices=["5", "6"], default="5", help="Auto crop layout count (5/6).")
    ap.add_argument("--pad", type=int, default=120, help="Padding around auto crops.")

    ap.add_argument(
        "--base",
        default=os.getenv("BASE_SD35", os.getenv("BASE", "/workspace-data/models/sd35-large")),
    )
    ap.add_argument("--inject_script", default="scripts/sd35_r1b_material_inject.py")

    args = ap.parse_args()

    im = Image.open(args.img_in).convert("RGB")
    W, H = im.size
    out = im.copy()

    p = int(args.pad)

    # Overlapping crop set designed for wide scenes
    if args.auto == "5":
        crops = [
            (p, p, W - p, H - p),                    # center
            (0, p, int(W * 0.60), H - p),            # left
            (int(W * 0.40), p, W, H - p),            # right
            (p, 0, W - p, int(H * 0.55)),            # top
            (p, int(H * 0.45), W - p, H),            # bottom
        ]
    else:  # "6"
        crops = [
            (0, 0, int(W * 0.60), int(H * 0.60)),
            (int(W * 0.40), 0, W, int(H * 0.60)),
            (0, int(H * 0.40), int(W * 0.60), H),
            (int(W * 0.40), int(H * 0.40), W, H),
            (p, p, W - p, H - p),
            (p, int(H * 0.55), W - p, H),
        ]

    tmp = "/tmp/crop_pass"
    os.makedirs(tmp, exist_ok=True)

    for i, (x0, y0, x1, y1) in enumerate(crops):
        x0 = max(0, min(W - 2, x0))
        y0 = max(0, min(H - 2, y0))
        x1 = max(x0 + 2, min(W, x1))
        y1 = max(y0 + 2, min(H, y1))

        crop = out.crop((x0, y0, x1, y1))
        tin = os.path.join(tmp, f"crop_in_{i}.png")
        tout = os.path.join(tmp, f"crop_out_{i}.png")
        crop.save(tin)

        run_inject(tin, tout, args)

        enhanced = Image.open(tout).convert("RGB")
        mw, mh = enhanced.size
        m = feather_mask(mw, mh, min(args.feather, mw // 8, mh // 8))

        region = out.crop((x0, y0, x1, y1))
        merged = Image.composite(enhanced, region, m)
        out.paste(merged, (x0, y0))

        print(f"[crop] {i} box=({x0},{y0},{x1},{y1}) done")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.save(args.out)
    print(f"[crop-pass] Saved: {args.out}")


if __name__ == "__main__":
    main()
