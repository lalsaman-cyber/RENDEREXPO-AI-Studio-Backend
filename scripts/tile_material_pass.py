import os
import argparse
import subprocess
from PIL import Image, ImageFilter


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def feather_mask(w: int, h: int, feather: int) -> Image.Image:
    """
    White center with feathered edges, used to blend tile seams.
    """
    feather = max(0, int(feather))
    m = Image.new("L", (w, h), 0)
    if feather * 2 < w and feather * 2 < h:
        m.paste(255, (feather, feather, w - feather, h - feather))
    else:
        # if feather too large, just make full white
        m.paste(255, (0, 0, w, h))

    if feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=feather * 0.8))
    return m


def main():
    ap = argparse.ArgumentParser(
        description="Tiled material pass: split into overlapping tiles, run sd35_r1b_material_inject on each, blend back."
    )
    ap.add_argument("--in", dest="img_in", required=True, help="Input image path (ideally 2x-upscaled).")
    ap.add_argument("--out", required=True, help="Output image path.")
    ap.add_argument("--tile", type=int, default=1024, help="Tile size (px). Typical 768/1024/1280.")
    ap.add_argument("--overlap", type=int, default=192, help="Overlap (px). Typical 128-256.")
    ap.add_argument("--feather", type=int, default=72, help="Feather (px) for blending tile edges.")

    ap.add_argument("--prompt", required=True, help="Material-only prompt.")
    ap.add_argument("--neg", default="", help="Negative prompt.")

    ap.add_argument("--steps", type=int, default=38)
    ap.add_argument("--cfg", type=float, default=5.8)
    ap.add_argument("--denoise", type=float, default=0.22)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument(
        "--base",
        default=os.getenv("BASE_SD35", os.getenv("BASE", "/workspace-data/models/sd35-large")),
        help="Path to SD3.5 model folder.",
    )
    ap.add_argument(
        "--inject_script",
        default="scripts/sd35_r1b_material_inject.py",
        help="Material inject script.",
    )

    args = ap.parse_args()

    if not os.path.exists(args.img_in):
        raise SystemExit(f"Input not found: {args.img_in}")

    tile = int(args.tile)
    ov = int(args.overlap)
    step = tile - ov
    if step <= 0:
        raise SystemExit("Tile must be > overlap.")

    im = Image.open(args.img_in).convert("RGB")
    W, H = im.size

    tmpdir = os.path.join("/tmp", "tile_pass")
    os.makedirs(tmpdir, exist_ok=True)

    base_out = Image.new("RGB", (W, H))
    mask_tile = feather_mask(tile, tile, args.feather)

    xs = list(range(0, W, step))
    ys = list(range(0, H, step))

    # ensure last tile covers edge
    if xs and xs[-1] + tile < W:
        xs.append(W - tile)
    if ys and ys[-1] + tile < H:
        ys.append(H - tile)

    def run_inject(tile_path_in: str, tile_path_out: str):
        cmd = [
            "python3",
            args.inject_script,
            "--base",
            args.base,
            "--in",
            tile_path_in,
            "--out",
            tile_path_out,
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

    for y in ys:
        for x in xs:
            x0 = clamp(x, 0, W - tile)
            y0 = clamp(y, 0, H - tile)

            crop = im.crop((x0, y0, x0 + tile, y0 + tile))
            tin = os.path.join(tmpdir, f"tile_in_{x0}_{y0}.png")
            tout = os.path.join(tmpdir, f"tile_out_{x0}_{y0}.png")
            crop.save(tin)

            run_inject(tin, tout)

            out_tile = Image.open(tout).convert("RGB")

            region = base_out.crop((x0, y0, x0 + tile, y0 + tile))
            region = Image.composite(out_tile, region, mask_tile)
            base_out.paste(region, (x0, y0))

            print(f"[tile] ({x0},{y0}) done")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    base_out.save(args.out)
    print(f"[tile-pass] Saved: {args.out}")


if __name__ == "__main__":
    main()
