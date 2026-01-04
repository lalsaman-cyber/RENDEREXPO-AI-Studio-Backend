import os
import time
import torch
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    # Required env vars
    base = os.environ["BASE_SD35"]          # path to SD3.5 base model folder
    img_in = os.environ["IMG_IN"]           # input image (Pass 1 output)
    out = os.environ["OUT"]                 # output file path
    prompt = os.environ["PROMPT"]           # <= 77 tokens
    neg = os.environ.get("NEG", "")

    # Pass 2 controls (safe defaults)
    steps = int(os.environ.get("STEPS", "26"))
    cfg = float(os.environ.get("CFG_SCALE", "6.2"))
    denoise = float(os.environ.get("DENOISE", "0.24"))  # img2img strength
    seed = int(os.environ.get("SEED", "12345"))

    # Device / dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Load image
    img = Image.open(img_in).convert("RGB")

    # Load pipeline
    log(f"Loading SD3.5 img2img from: {base}")
    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(base, torch_dtype=dtype).to(device)

    # Run
    g = torch.Generator(device=device).manual_seed(seed)
    log(f"Running Pass2 img2img: steps={steps} cfg={cfg} denoise={denoise} seed={seed}")
    result = pipe(
        prompt=prompt,
        negative_prompt=neg,
        image=img,
        strength=denoise,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=g,
    ).images[0]

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    result.save(out)
    log(f"Saved: {out}")


if __name__ == "__main__":
    main()
