import os
import time
from pathlib import Path

import torch
import safetensors.torch as sft
from PIL import Image

from diffusers import StableDiffusion3Pipeline
from lycoris import create_lycoris_from_weights


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    """
    Convert torch tensor to PIL.
    Accepts BCHW / CHW / HWC, values in [-1,1] or [0,1].
    """
    if isinstance(img, (list, tuple)):
        img = img[0]
    if img is None:
        raise RuntimeError("tensor_to_pil got None")
    if not isinstance(img, torch.Tensor):
        raise RuntimeError(f"tensor_to_pil expected torch.Tensor, got {type(img)}")

    t = img.detach().float().cpu()

    if t.ndim == 4:
        t = t[0]
    if t.ndim == 3 and t.shape[0] in (1, 3, 4):
        t = t.permute(1, 2, 0)
    if t.ndim != 3:
        raise RuntimeError(f"Unexpected tensor shape: {tuple(t.shape)}")

    tmin = float(t.min())
    tmax = float(t.max())
    if tmin < -0.1 and tmax > 0.1:
        t = (t * 0.5) + 0.5  # [-1,1] -> [0,1]

    t = t.clamp(0.0, 1.0)
    arr = (t * 255.0).round().to(torch.uint8).numpy()

    if arr.shape[2] == 1:
        return Image.fromarray(arr[:, :, 0], mode="L")
    if arr.shape[2] == 4:
        return Image.fromarray(arr, mode="RGBA")
    return Image.fromarray(arr, mode="RGB")


def stats(name: str, x: torch.Tensor):
    if x is None:
        log(f"{name}: None")
        return
    with torch.no_grad():
        xn = x.detach()
        fin = torch.isfinite(xn).all().item()
        any_nan = torch.isnan(xn).any().item()
        any_inf = torch.isinf(xn).any().item()
        mn = float(xn.min().item())
   log(
            f"{name}: finite={fin} nan={any_nan} inf={any_inf} "
            f"min={mn:.6g} max={mx:.6g} dtype={xn.dtype} device={xn.device}"
        )


def _clean_path_list(s: str) -> list[str]:
    # Accept commas, semicolons, whitespace
    if not s:
        return []
    raw = s.replace(";", ",").replace("\n", ",").replace("\t", ",")
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def apply_lycoris_to_sd35_transformer(
    pipe: StableDiffusion3Pipeline,
    lyco_path: str,
    multiplier: float,
    device: str,
    dtype: torch.dtype,
):
    """
    Applies LyCORIS weights to SD3.5 pipeline transformer.
    Returns lyco object (supports restore()).
    """
    log(f"Loading LyCORIS safetensors: {lyco_path}")
    weights_sd = sft.load_file(lyco_path)

    log("Creating LyCORIS network from weights...")
    lyco, _ = create_lycoris_from_weights(
        multiplier=multiplier,
        file=None,
        module=pipe.transformer,  # SD3.5 uses transformer as the denoiser core
        weights_sd=weights_sd,
    )

    log("Applying LyCORIS to transformer...")
    lyco.apply_to()
    lyco.to(device=device, dtype=dtype)
    try:
        lyco.eval()
    except Exception:
        pass
    return lyco


def main():
   # -----------------------
    # ENV
    # -----------------------
    base = os.environ.get("BASE", "").strip()
    if not base:
        raise RuntimeError("BASE is not set. Example: export BASE=/workspace-data/models/sd35-large")

    prompt = os.environ.get("PROMPT", "").strip()
    if not prompt:
        raise RuntimeError("PROMPT is empty. Export PROMPT='...'")

    negative = os.environ.get("NEG", "").strip()
    steps = int(os.environ.get("STEPS", "32"))
    cfg = float(os.environ.get("CFG", "5.0"))
    seed = int(os.environ.get("SEED", "777"))
    out = os.environ.get("OUT", "outputs/out.png")

    # LyCORIS #1 (PRO)
    mult = float(os.environ.get("MULT", "0.0"))
    ckpt = os.environ.get("CKPT", "").strip()

  # LyCORIS #2 (GEO or any secondary)
    mult2 = float(os.environ.get("MULT2", "0.0"))
    ckpt2 = os.environ.get("CKPT2", "").strip()

    # IMPORTANT PATCH:
    # If someone passes CKPT="path1,path2" and CKPT2 is empty, split automatically.
    if ckpt and ("," in ckpt or ";" in ckpt or "\n" in ckpt) and not ckpt2:
        parts = _clean_path_list(ckpt)
        if len(parts) >= 2:
            ckpt = parts[0]
            ckpt2 = parts[1]
            log(f"Detected CKPT list -> CKPT={ckpt} | CKPT2={ckpt2}")
        elif len(parts) == 1:
            ckpt = parts[0]

    # -----------------------
    # DEVICE + DTYPE
    # -----------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

 # BF16 is often more stable; you can force FP16 by export FORCE_FP16=1
    force_fp16 = os.environ.get("FORCE_FP16", "0").strip() == "1"
    if device == "cuda":
        dtype = torch.float16 if force_fp16 else torch.bfloat16
    else:
        dtype = torch.float32

    log(f"Device={device} dtype={dtype} (FORCE_FP16={int(force_fp16)})")

    # -----------------------
    # LOAD PIPELINE
    # -----------------------
    log("Loading SD3.5 pipeline... (slowest step)")
    t0 = time.time()
    pipe = StableDiffusion3Pipeline.from_pretrained(base, torch_dtype=dtype)
    log(f"Pipeline loaded in {time.time() - t0:.1f}s")

    log("Moving pipeline to device...")
    t1 = time.time()
    pipe = pipe.to(device)
    log(f"Moved to {device} in {time.time() - t1:.1f}s")

 # VAE memory helpers (safe if available)
    try:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    except Exception:
        pass

    pipe.vae.to(device=device, dtype=dtype)

    # -----------------------
    # OPTIONAL: APPLY LyCORIS #1
    # -----------------------
    lyco1 = None
    log("LyCORIS #1 step (optional)...")
    t2 = time.time()
    if mult > 0.0:
        if not ckpt:
            raise RuntimeError("MULT>0 but CKPT is empty. Export CKPT=/path/to/pytorch_lycoris_*.safetensors")
        if not Path(ckpt).exists():
            raise RuntimeError(f"CKPT path does not exist: {ckpt}")

        log(f"Loading LyCORIS #1 CKPT={ckpt} MULT={mult}")
        lyco1 = apply_lycoris_to_sd35_transformer(pipe, ckpt, multiplier=mult, device=device, dtype=dtype)
    log(f"LyCORIS #1 step done in {time.time() - t2:.1f}s")

  # -----------------------
    # OPTIONAL: APPLY LyCORIS #2
    # -----------------------
    lyco2 = None
    log("LyCORIS #2 step (optional)...")
    t2b = time.time()
    if mult2 > 0.0:
        if not ckpt2:
            raise RuntimeError("MULT2>0 but CKPT2 is empty. Export CKPT2=/path/to/pytorch_lycoris_*.safetensors")
        if not Path(ckpt2).exists():
            raise RuntimeError(f"CKPT2 path does not exist: {ckpt2}")

        log(f"Loading LyCORIS #2 CKPT2={ckpt2} MULT2={mult2}")
        lyco2 = apply_lycoris_to_sd35_transformer(pipe, ckpt2, multiplier=mult2, device=device, dtype=dtype)
    log(f"LyCORIS #2 step done in {time.time() - t2b:.1f}s")

 # -----------------------
    # GENERATE LATENTS ONLY
    # -----------------------
    gen = torch.Generator(device=device).manual_seed(seed)

    log("Generating (latents only)...")
    t3 = time.time()
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=gen,
            output_type="latent",
        )
    log(f"Inference done in {time.time() - t3:.1f}s")

    latents = getattr(result, "images", None)
    if latents is None:
        raise RuntimeError("Expected latents in result.images but got None")

    stats("latents", latents)

  # -----------------------
    # MANUAL DECODE
    # -----------------------
    log("Manual VAE decode...")
    t4 = time.time()

    scaling = getattr(pipe.vae.config, "scaling_factor", 1.0)
    with torch.inference_mode():
        decoded = pipe.vae.decode(latents / scaling, return_dict=False)[0]

    stats("decoded", decoded)

    image = tensor_to_pil(decoded)
    log(f"Decode done in {time.time() - t4:.1f}s")

    # -----------------------
    # SAVE
    # -----------------------
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(out_path))
    log(f"Saved: {out_path}")

# -----------------------
    # RESTORE LyCORIS (reverse order)
    # -----------------------
    try:
        if lyco2 is not None and hasattr(lyco2, "restore"):
            lyco2.restore()
            log("LyCORIS #2 restored (unapplied).")
        if lyco1 is not None and hasattr(lyco1, "restore"):
            lyco1.restore()
            log("LyCORIS #1 restored (unapplied).")
        if (lyco1 is None or not hasattr(lyco1, "restore")) and (lyco2 is None or not hasattr(lyco2, "restore")):
            log("LyCORIS restore skipped (no restore() available).")
    except Exception as e:
        log(f"LyCORIS restore skipped: {e}")

if __name__ == "__main__":
    main()
