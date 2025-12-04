#!/usr/bin/env python
"""
RENDEREXPO – SD3.5 LoRA v2 quick test

- Loads base SD3.5-Large from a local path
- Loads LoRA weights from outputs/lora/sd35_buildings_v2
- Generates a single test image
"""

import os
import torch
from diffusers import StableDiffusion3Pipeline


def main():
    # -------- CONFIG --------
    base_model_path = "/workspace-data/models/sd35-large"
    lora_dir = "outputs/lora/sd35_buildings_v2"
    out_path = "sd35_lora_v2_test.png"

    prompt = (
        "renderexpo building lora style, ultra detailed architectural visualization, "
        "photorealistic, golden hour lighting, 4k, high detail, sharp focus"
    )

    negative_prompt = (
        "low quality, blurry, distorted, deformed, cartoon, sketch, painting, watermark, text"
    )

    num_inference_steps = 28
    guidance_scale = 5.0
    seed = 42
    # ------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load base SD3.5 pipeline
    print("Loading SD3.5-Large base pipeline...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
    ).to(device)

    # Load LoRA weights
    print(f"Loading LoRA weights from: {lora_dir}")
    pipe.load_lora_weights(lora_dir)

    # Generator for reproducibility
    generator = torch.Generator(device).manual_seed(seed)

    print("Generating image with SD3.5 + LoRA...")
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    image.save(out_path)
    print(f"Done. Image saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
