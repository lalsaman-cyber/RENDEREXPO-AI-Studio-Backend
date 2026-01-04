import os
import torch
import cv2
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


def main() -> None:
    img_in = os.environ["IMG_IN"]
    out = os.environ["OUT"]

    # We want 2x output, but we'll use the stable x4plus model then downscale to 2x cleanly.
    target_scale = int(os.environ.get("SCALE", "2"))
    tile = int(os.environ.get("TILE", "256"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Backbone for RealESRGAN_x4plus
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )

    # Auto-download weights if missing (cached after first run)
    upsampler = RealESRGANer(
        scale=4,
        model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=(device == "cuda"),
        device=device,
    )

    img = cv2.imread(img_in, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_in}")

    # Upscale to 4x then downscale to 2x if requested
    out_img, _ = upsampler.enhance(img, outscale=4)

    if target_scale == 2:
        h, w = out_img.shape[:2]
        out_img = cv2.resize(out_img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    elif target_scale != 4:
        raise ValueError("SCALE must be 2 or 4")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    ok = cv2.imwrite(out, out_img)
    if not ok:
        raise RuntimeError(f"Failed to write output: {out}")

    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
