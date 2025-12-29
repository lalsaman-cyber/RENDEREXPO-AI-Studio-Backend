#!/usr/bin/env python
# RENDEREXPO SD3.5-Large LyCORIS trainer (GEOMETRY+MATERIALS adapter, caption-aware, rank 16, saves+samples every 500)
#
# Goal:
# - Complement PRO 2.1 (style) with a second adapter that forces:
#   sharp geometry, rigid lines, crisp material boundaries, depth clarity, no smudginess
# - Dataset is paired: 1.jpg + 1.txt, etc (342 pairs verified)
# - A6000 training, FP16 transformer, VAE float32 for stability
# - Saves checkpoint safetensors every --save_steps (default 500)
# - ALSO saves reference sample images every --sample_steps (default 500)
#
# Notes:
# - This script trains ONLY LyCORIS params attached to SD3 transformer.
# - For sampling we temporarily build a StableDiffusion3Pipeline and generate 2-3 fixed prompts.
# - This is slower but extremely reliable and produces “proof images” every 500 steps even if you sleep.

import argparse
import os
import random
import time
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import crop
from PIL import Image
from PIL.ImageOps import exif_transpose
from tqdm.auto import tqdm

from transformers import CLIPTokenizer, T5TokenizerFast, PretrainedConfig

from diffusers import AutoencoderKL, SD3Transformer2DModel, StableDiffusion3Pipeline
from diffusers.training_utils import free_memory

from lycoris import create_lycoris, LycorisNetwork
import safetensors.torch as sft


# ------------------------------
# Dataset: image + caption pairs
# ------------------------------

class RenderExpoCaptionDataset(Dataset):
    """
    Caption-aware dataset:

    - Scans instance_data_root for image files (.jpg/.jpeg/.png/.webp)
    - For each image `name.ext`, looks for `name.txt`
    - If found and non-empty: uses that caption
    - Otherwise: falls back to global --instance_prompt
    - Optional caption_dropout: sometimes use instance_prompt for generalization
    """

    def __init__(
        self,
        instance_data_root: str,
        default_prompt: str,
        resolution: int = 1024,
        center_crop: bool = False,
        caption_dropout: float = 0.10,
    ):
        self.default_prompt = default_prompt
        self.center_crop = center_crop
        self.resolution = resolution
        self.caption_dropout = caption_dropout

        root = Path(instance_data_root)
        if not root.exists():
            raise ValueError(f"Instance data root does not exist: {instance_data_root}")

        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        self.image_paths = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]

        if len(self.image_paths) == 0:
            raise ValueError(f"No valid images found in: {instance_data_root}")

        # Sort for determinism
        self.image_paths = sorted(self.image_paths, key=lambda p: p.name.lower())

        self.resize = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)
        self.random_crop = transforms.RandomCrop(resolution)
        self.center_crop_tf = transforms.CenterCrop(resolution)
        self.hflip = transforms.RandomHorizontalFlip(p=1.0)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5], [0.5])

    def __len__(self):
        return len(self.image_paths)

    def _load_caption(self, image_path: Path) -> str:
        txt_path = image_path.with_suffix(".txt")
        if txt_path.exists():
            try:
                text = txt_path.read_text(encoding="utf-8").strip()
                if text:
                    return text
            except Exception:
                pass
        return self.default_prompt

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        image = Image.open(path)
        image = exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = self.resize(image)

        # Random flip 50%
        if random.random() < 0.5:
            image = self.hflip(image)

        if self.center_crop:
            image = self.center_crop_tf(image)
        else:
            y1, x1, h, w = self.random_crop.get_params(image, (self.resolution, self.resolution))
            image = crop(image, y1, x1, h, w)

        image = self.to_tensor(image)
        image = self.normalize(image)

        caption = self._load_caption(path)
        if self.caption_dropout > 0.0 and random.random() < self.caption_dropout:
            caption = self.default_prompt

        return {"pixel_values": image, "prompt": caption}


# ------------------------------
# Text encoding helpers for SD3
# ------------------------------

def import_text_encoder_class(model_path: str, subfolder: str):
    cfg = PretrainedConfig.from_pretrained(model_path, subfolder=subfolder)
    arch = cfg.architectures[0]
    if arch == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    elif arch == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"Unsupported text encoder architecture: {arch}")


def encode_prompt_sd3(
    prompts: List[str],
    text_encoder_one,
    text_encoder_two,
    text_encoder_three,
    tokenizer_one,
    tokenizer_two,
    tokenizer_three,
    max_sequence_length: int = 77,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float16,
):
    # CLIP encoders
    clip_tokenizers = [tokenizer_one, tokenizer_two]
    clip_encoders = [text_encoder_one, text_encoder_two]
    clip_embeds_list = []
    clip_pooled_list = []

    for tok, enc in zip(clip_tokenizers, clip_encoders):
        text_inputs = tok(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids  # CPU

        with torch.no_grad():
            outputs = enc(input_ids, output_hidden_states=True)

        pooled = outputs[0]
        hidden = outputs.hidden_states[-2]

        hidden = hidden.to(dtype=torch.float32)
        pooled = pooled.to(dtype=torch.float32)

        clip_embeds_list.append(hidden)
        clip_pooled_list.append(pooled)

    clip_prompt_embeds = torch.cat(clip_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_list, dim=-1)

    # T5 encoder
    t5_inputs = tokenizer_three(
        prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    t5_ids = t5_inputs.input_ids
    with torch.no_grad():
        t5_out = text_encoder_three(t5_ids)[0]

    t5_out = t5_out.to(dtype=torch.float32)

    if clip_prompt_embeds.shape[-1] < t5_out.shape[-1]:
        pad_channels = t5_out.shape[-1] - clip_prompt_embeds.shape[-1]
        clip_prompt_embeds = torch.nn.functional.pad(clip_prompt_embeds, (0, pad_channels))

    prompt_embeds = torch.cat([clip_prompt_embeds, t5_out], dim=-2)

    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)
    return prompt_embeds, pooled_prompt_embeds


# ------------------------------
# LR schedule (3-phase decay)
# ------------------------------

def three_phase_lr(step: int, total_steps: int, lr_start: float, lr_mid: float, lr_final: float) -> float:
    if total_steps <= 1:
        return lr_final
    half = total_steps // 2
    if step <= half:
        t = step / max(1, half)
        return lr_start + t * (lr_mid - lr_start)
    else:
        t = (step - half) / max(1, total_steps - half)
        return lr_mid + t * (lr_final - lr_mid)


# ------------------------------
# Sampling helpers
# ------------------------------

def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def generate_reference_samples(
    base_model_path: str,
    transformer_with_lyco: SD3Transformer2DModel,
    out_dir: str,
    step: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    Saves a few “proof images” every N steps.

    Reliability > speed:
    - We build a temporary SD3 pipeline
    - Swap in our transformer (already has LyCORIS attached)
    - Generate fixed prompts with fixed seed
    - Save PNGs
    """

    os.makedirs(out_dir, exist_ok=True)

    prompts = [
        # Exterior geometry stress test
        "Ultra-sharp architectural photograph, rigid rectilinear geometry, straight verticals and horizontals, crisp balcony edges, clean glass boundaries, no smudginess, no blur, photographic spatial clarity at all depths",
        # Interior geometry/material boundary stress test
        "Ultra-sharp interior architectural photograph, straight ceiling lines, clean wall intersections, sharp cabinetry edges, discrete material boundaries, no texture smear, no softened edges, realistic depth clarity",
    ]
    negative = "smudgy, blurry, softened edges, melted geometry, warped lines, painterly, CGI look, over-smoothed, haze, bloom"

    _log(f"[SAMPLES] Building temporary pipeline for step {step}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(base_model_path, torch_dtype=dtype)

    # Swap transformer (LyCORIS lives inside this transformer)
    pipe.transformer = transformer_with_lyco

    pipe = pipe.to(device)

    g = torch.Generator(device=device).manual_seed(seed + step)

    for i, p in enumerate(prompts):
        out_path = os.path.join(out_dir, f"step_{step:06d}_{i:02d}.png")
        _log(f"[SAMPLES] Generating {out_path}")
        img = pipe(
            prompt=p,
            negative_prompt=negative,
            num_inference_steps=28,
            guidance_scale=4.5,
            generator=g,
        ).images[0]
        img.save(out_path)

    # Cleanup
    del pipe
    free_memory()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _log(f"[SAMPLES] Done for step {step}.")


# ------------------------------
# Args
# ------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        "RENDEREXPO SD3.5 LyCORIS trainer (GEOMETRY+MATERIALS, caption-aware, rank 16, saves+samples every 500)"
    )

    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument("--instance_data_dir", type=str, required=True)
    p.add_argument("--instance_prompt", type=str, required=True)

    p.add_argument("--output_dir", type=str, required=True, help="Where to save LyCORIS weights (.safetensors)")
    p.add_argument("--samples_dir", type=str, required=True, help="Where to save reference sample images")

    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--train_batch_size", type=int, default=1)

    # For 342 images batch=1:
    # 6000 steps ~ 17.5 epochs, good for “behavior correction” without heavy overfit.
    p.add_argument("--max_train_steps", type=int, default=6000)

    # Slightly safer LR for geometry/material training (less chance of “noise weights”)
    p.add_argument("--learning_rate", type=float, default=6e-5)
    p.add_argument("--lr_mid", type=float, default=2.5e-5)
    p.add_argument("--lr_final", type=float, default=1.0e-5)

    # Rank 16 (you requested)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=16.0)

    # locon tends to preserve crisp structure well
    p.add_argument("--lycoris_algo", type=str, default="locon", choices=["locon", "loha"])

    # Keep some dropout so this doesn’t “lock” to captions too hard
    p.add_argument("--caption_dropout", type=float, default=0.10)

    # Must save every 500
    p.add_argument("--save_steps", type=int, default=500)

    # Must also save sample images every 500
    p.add_argument("--sample_steps", type=int, default=500)

    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ------------------------------
# Main
# ------------------------------

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.samples_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")

    # Transformer FP16 on GPU
    weight_dtype = torch.float16

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("===============================================================")
    print("RENDEREXPO — GEOMETRY + MATERIALS LyCORIS Trainer (Rank 16)")
    print("===============================================================")
    print(f"Base model path:        {args.pretrained_model_name_or_path}")
    print(f"Instance data dir:      {args.instance_data_dir}")
    print(f"Output dir:             {args.output_dir}")
    print(f"Samples dir:            {args.samples_dir}")
    print(f"Resolution:             {args.resolution}")
    print(f"Batch size:             {args.train_batch_size}")
    print(f"Max steps:              {args.max_train_steps}")
    print(f"LR schedule:            {args.learning_rate} -> {args.lr_mid} -> {args.lr_final}")
    print(f"LyCORIS algo:           {args.lycoris_algo}")
    print(f"LyCORIS rank (dim):     {args.rank}")
    print(f"LyCORIS alpha:          {args.alpha}")
    print(f"Caption dropout:        {args.caption_dropout}")
    print(f"Save steps:             {args.save_steps}")
    print(f"Sample steps:           {args.sample_steps}")
    print(f"Seed:                   {args.seed}")
    print(f"GPU dtype:              {weight_dtype}")
    print("===============================================================")

    # Load model components
    _log("Loading SD3.5 components...")

    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to(device)
    vae.requires_grad_(False)

    transformer: SD3Transformer2DModel = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    ).to(device)
    transformer.enable_gradient_checkpointing()
    transformer.requires_grad_(True)

    tokenizer_one: CLIPTokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_two: CLIPTokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    tokenizer_three: T5TokenizerFast = T5TokenizerFast.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_3")

    te1_cls = import_text_encoder_class(args.pretrained_model_name_or_path, "text_encoder")
    te2_cls = import_text_encoder_class(args.pretrained_model_name_or_path, "text_encoder_2")
    te3_cls = import_text_encoder_class(args.pretrained_model_name_or_path, "text_encoder_3")

    text_encoder_one = te1_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=torch.float32
    ).to(cpu_device)
    text_encoder_two = te2_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", torch_dtype=torch.float32
    ).to(cpu_device)
    text_encoder_three = te3_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", torch_dtype=torch.float32
    ).to(cpu_device)

    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)

    # Attach LyCORIS
    _log("Attaching LyCORIS adapter to SD3 transformer...")

    # IMPORTANT: we target attention modules like before (stable behavior)
    LycorisNetwork.apply_preset({"target_name": [".*attn.*"]})

    lyco_net = create_lycoris(
        transformer,
        multiplier=1.0,
        linear_dim=args.rank,
        linear_alpha=args.alpha,
        algo=args.lycoris_algo,
    )
    lyco_net.apply_to()

    lyco_params = list(lyco_net.parameters())
    _log(f"LyCORIS trainable params: {sum(p.numel() for p in lyco_params):,}")

    optimizer = torch.optim.AdamW(
        lyco_params,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        eps=1e-8,
    )

    # Dataset
    train_dataset = RenderExpoCaptionDataset(
        instance_data_root=args.instance_data_dir,
        default_prompt=args.instance_prompt,
        resolution=args.resolution,
        center_crop=False,
        caption_dropout=args.caption_dropout,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0)

    # Training loop
    total_steps = args.max_train_steps
    global_step = 0

    transformer.train()
    noise_scale = 0.25  # keeps it “sharp” like your previous runs

    _log(f"Starting training for {total_steps} steps...")

    progress_bar = tqdm(range(total_steps), desc="Training")

    # Save an initial sample at step 0 (so you have a baseline reference)
    try:
        generate_reference_samples(
            base_model_path=args.pretrained_model_name_or_path,
            transformer_with_lyco=transformer,
            out_dir=args.samples_dir,
            step=0,
            seed=args.seed,
            device=device,
            dtype=weight_dtype,
        )
    except Exception as e:
        _log(f"[SAMPLES] Initial sample failed (not fatal): {e}")

    while global_step < total_steps:
        for batch in train_dataloader:
            if global_step >= total_steps:
                break

            pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
            prompts = batch["prompt"]

            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = encode_prompt_sd3(
                    prompts,
                    text_encoder_one,
                    text_encoder_two,
                    text_encoder_three,
                    tokenizer_one,
                    tokenizer_two,
                    tokenizer_three,
                    max_sequence_length=77,
                    device=device,
                    dtype=weight_dtype,
                )

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()

            if hasattr(vae.config, "scaling_factor"):
                latents = latents * vae.config.scaling_factor

            latents = torch.clamp(latents, -10.0, 10.0).to(device=device, dtype=weight_dtype)

            noise = torch.randn_like(latents, dtype=weight_dtype)
            noise = torch.clamp(noise, -10.0, 10.0)

            timesteps = torch.zeros(latents.shape[0], device=device, dtype=torch.long)

            noisy_latents = latents + noise_scale * noise
            noisy_latents = torch.clamp(noisy_latents, -10.0, 10.0)

            lr = three_phase_lr(global_step, total_steps, args.learning_rate, args.lr_mid, args.lr_final)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=True, dtype=weight_dtype):
                model_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]

                loss = F.mse_loss(model_pred.to(dtype=torch.float32), noise.to(dtype=torch.float32))

            if not torch.isfinite(loss):
                _log(f"⚠️ Non-finite loss at step {global_step}: {loss.item()} (skipping)")
                global_step += 1
                progress_bar.update(1)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(lyco_params, 1.0)
            optimizer.step()

            global_step += 1
            progress_bar.set_postfix({"loss": float(loss.item()), "lr": lr})
            progress_bar.update(1)

            # Checkpoint save every N steps
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                ckpt_path = os.path.join(args.output_dir, f"pytorch_lycoris_step_{global_step:06d}.safetensors")
                sft.save_file(lyco_net.state_dict(), ckpt_path)
                _log(f"[Checkpoint] Saved: {ckpt_path}")

            # Sample save every N steps
            if args.sample_steps > 0 and global_step % args.sample_steps == 0:
                try:
                    generate_reference_samples(
                        base_model_path=args.pretrained_model_name_or_path,
                        transformer_with_lyco=transformer,
                        out_dir=args.samples_dir,
                        step=global_step,
                        seed=args.seed,
                        device=device,
                        dtype=weight_dtype,
                    )
                except Exception as e:
                    _log(f"[SAMPLES] Step {global_step} sample failed (not fatal): {e}")

            if global_step >= total_steps:
                break

    progress_bar.close()
    _log("Training finished. Saving final weights...")

    transformer.eval()

    final_path = os.path.join(args.output_dir, "pytorch_lycoris_final.safetensors")
    sft.save_file(lyco_net.state_dict(), final_path)
    _log(f"✅ Final LyCORIS saved: {final_path}")

    # Cleanup
    del vae, text_encoder_one, text_encoder_two, text_encoder_three
    free_memory()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
