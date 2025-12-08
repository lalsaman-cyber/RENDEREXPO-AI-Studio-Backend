#!/usr/bin/env python
# RENDEREXPO SD3.5-Large LyCORIS trainer (PRO v2, caption-aware, fp32, 896px, rank 8)
#
# - Base model: Stable Diffusion 3.5 Large (diffusers format)
# - True LyCORIS adapters (LoCon / LoHa) using the `lycoris-lora` library
# - 225 fully captioned Corona-quality RENDEREXPO images
# - Resolution: 896x896 (square, random-crop, random-flip)
# - Rank: 8, Alpha: 8  (per your spec)
# - LR schedule: 1e-4  →  5e-5  →  2e-5 over 6000 steps
# - Caption-aware:
#       * Uses per-image .txt caption if present
#       * If missing/empty, falls back to --instance_prompt
#       * Optional caption_dropout mixes in the global style prompt
#
# NOTE:
#   This script ONLY handles the training logic.
#   tmux + nohup will be handled in the shell when we actually start training.

import argparse
import os
import random
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

from diffusers import AutoencoderKL, SD3Transformer2DModel
from diffusers.training_utils import free_memory

import safetensors.torch as sft

# 🔸 True LyCORIS (LoCon / LoHa)
from lycoris import create_lycoris, LycorisNetwork


# --------------------------------------------------
# Dataset: image + caption pairs (225 Corona renders)
# --------------------------------------------------


class RenderExpoCaptionDataset(Dataset):
    """
    Caption-aware dataset:

    - Scans instance_data_root for image files (.jpg/.jpeg/.png/.webp)
    - For each image `name.ext`, looks for a text file `name.txt` in the same folder
    - If found and non-empty: uses that text as the prompt
    - Otherwise: falls back to the global --instance_prompt
    - Optional caption_dropout: with probability p, we ignore the caption and use
      the global instance_prompt instead (helps the LoRA/LyCORIS generalize).
    """

    def __init__(
        self,
        instance_data_root: str,
        default_prompt: str,
        resolution: int = 896,
        center_crop: bool = False,
        caption_dropout: float = 0.15,
    ):
        self.default_prompt = default_prompt
        self.center_crop = center_crop
        self.resolution = resolution
        self.caption_dropout = caption_dropout

        root = Path(instance_data_root)
        if not root.exists():
            raise ValueError(f"Instance data root does not exist: {instance_data_root}")

        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        self.image_paths = [
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in valid_exts
        ]

        if len(self.image_paths) == 0:
            raise ValueError(f"No valid images found in: {instance_data_root}")

        # Basic transforms (896px)
        self.resize = transforms.Resize(
            resolution, interpolation=transforms.InterpolationMode.BILINEAR
        )
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
        # Fallback
        return self.default_prompt

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        # --- Image ---
        image = Image.open(path)
        image = exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = self.resize(image)

        # Random flip 50% of the time
        if random.random() < 0.5:
            image = self.hflip(image)

        if self.center_crop:
            image = self.center_crop_tf(image)
        else:
            y1, x1, h, w = self.random_crop.get_params(
                image, (self.resolution, self.resolution)
            )
            image = crop(image, y1, x1, h, w)

        image = self.to_tensor(image)
        image = self.normalize(image)

        # --- Caption ---
        caption = self._load_caption(path)

        # Caption dropout: sometimes replace caption with default_prompt
        if self.caption_dropout > 0.0 and random.random() < self.caption_dropout:
            caption = self.default_prompt

        return {
            "pixel_values": image,
            "prompt": caption,
        }


# --------------------------------------------------
# Text encoding helpers for SD3
# --------------------------------------------------


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
):
    """
    Encodes a batch of prompts for SD3:

    - 2x CLIP text encoders
    - 1x T5 encoder

    All encoders are assumed to be on CPU in this script.
    Returns (prompt_embeds, pooled_prompt_embeds) on `device`, float32.
    """

    # --- CLIP encoders (two) ---
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
            outputs = enc(
                input_ids,
                output_hidden_states=True,
            )

        pooled = outputs[0]  # last layer CLS
        hidden = outputs.hidden_states[-2]  # penultimate hidden

        hidden = hidden.to(dtype=torch.float32)
        pooled = pooled.to(dtype=torch.float32)

        clip_embeds_list.append(hidden)
        clip_pooled_list.append(pooled)

    clip_prompt_embeds = torch.cat(clip_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_list, dim=-1)

    # --- T5 encoder ---
    t5_inputs = tokenizer_three(
        prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    t5_ids = t5_inputs.input_ids  # CPU
    with torch.no_grad():
        t5_out = text_encoder_three(t5_ids)[0]  # [batch, seq, hidden]

    t5_out = t5_out.to(dtype=torch.float32)

    # Pad CLIP channels to match T5 hidden size along channel dimension, then concat along sequence
    if clip_prompt_embeds.shape[-1] < t5_out.shape[-1]:
        pad_channels = t5_out.shape[-1] - clip_prompt_embeds.shape[-1]
        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, pad_channels)
        )

    prompt_embeds = torch.cat([clip_prompt_embeds, t5_out], dim=-2)

    # Move to requested device (GPU) at the very end
    prompt_embeds = prompt_embeds.to(device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device)

    return prompt_embeds, pooled_prompt_embeds


# --------------------------------------------------
# LR schedule: 1e-4 → 5e-5 → 2e-5 across training
# --------------------------------------------------


def three_phase_lr(step: int, total_steps: int, lr_start: float, lr_mid: float, lr_final: float) -> float:
    """
    Simple 3-phase schedule:

    - First 40% of steps: lr_start
    - Next 30% of steps:  lr_mid
    - Last 30% of steps:  lr_final
    """
    if total_steps <= 0:
        return lr_final

    ratio = step / float(total_steps)

    if ratio < 0.4:
        return lr_start
    elif ratio < 0.7:
        return lr_mid
    else:
        return lr_final


# --------------------------------------------------
# Argument parsing
# --------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        "RENDEREXPO SD3.5 LoRA/LyCORIS trainer PRO v2 (caption-aware, fp32, 896px, rank 8)"
    )

    p.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to SD3.5-Large diffusers model",
    )
    p.add_argument(
        "--instance_data_dir",
        type=str,
        required=True,
        help="Folder with training images + .txt captions.",
    )
    p.add_argument(
        "--instance_prompt",
        type=str,
        required=True,
        help='Fallback prompt if an image has no caption, e.g. "architectural building visualization".',
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs/lora/sd35_renderexpo_pro_v2",
        help="Where to save LyCORIS weights",
    )
    p.add_argument(
        "--resolution",
        type=int,
        default=896,
        help="Training resolution (PRO v2 default = 896)",
    )
    p.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size per step",
    )
    p.add_argument(
        "--max_train_steps",
        type=int,
        default=6000,
        help="Total training steps (PRO v2 default = 6000)",
    )
    p.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Base learning rate (start LR, PRO v2 = 1e-4)",
    )
    p.add_argument(
        "--lr_mid",
        type=float,
        default=5e-5,
        help="Mid-phase learning rate target (PRO v2 = 5e-5).",
    )
    p.add_argument(
        "--lr_final",
        type=float,
        default=2e-5,
        help="Final-phase learning rate target (PRO v2 = 2e-5).",
    )
    p.add_argument(
        "--rank",
        type=int,
        default=8,
        help="LyCORIS rank (PRO v2 default = 8).",
    )
    p.add_argument(
        "--alpha",
        type=int,
        default=8,
        help="LyCORIS alpha (PRO v2 default = 8).",
    )
    p.add_argument(
        "--lycoris_algo",
        type=str,
        default="locon",
        choices=["locon", "loha"],
        help='LyCORIS algorithm to use: "locon" or "loha".',
    )
    p.add_argument(
        "--caption_dropout",
        type=float,
        default=0.15,
        help="Probability to replace caption with default prompt (0.0 to disable).",
    )
    p.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save a checkpoint every N steps (0 to disable periodic saves).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return p.parse_args()


# --------------------------------------------------
# Main training logic
# --------------------------------------------------


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")

    # Everything in VAE + transformer uses float32 for stability
    weight_dtype = torch.float32

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------------------------
    # CONFIG SUMMARY
    # ---------------------------------
    print("==========================================")
    print("RENDEREXPO PRO v2 – SD3.5 LyCORIS Trainer")
    print("==========================================")
    print(f"Base model path:        {args.pretrained_model_name_or_path}")
    print(f"Instance data dir:      {args.instance_data_dir}")
    print(f"Output dir:             {args.output_dir}")
    print(f"Resolution:             {args.resolution}")
    print(f"Batch size:             {args.train_batch_size}")
    print(f"Max steps:              {args.max_train_steps}")
    print(f"LR schedule:            {args.learning_rate} → {args.lr_mid} → {args.lr_final}")
    print(f"LyCORIS algo:           {args.lycoris_algo}")
    print(f"LyCORIS rank (dim):     {args.rank}")
    print(f"LyCORIS alpha:          {args.alpha}")
    print(f"Caption dropout:        {args.caption_dropout}")
    print(f"Seed:                   {args.seed}")
    print("==========================================")

    # ---------------------------------
    # Load SD3.5 components
    # ---------------------------------
    print("Loading SD3.5-Large components...")
    print(f"Using default instance prompt: {args.instance_prompt}")

    # VAE on GPU
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=weight_dtype,
    ).to(device)

    # Transformer on GPU
    transformer: SD3Transformer2DModel = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    ).to(device)

    # Tokenizers (CPU)
    tokenizer_one: CLIPTokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    tokenizer_two: CLIPTokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
    )
    tokenizer_three: T5TokenizerFast = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
    )

    # Text encoders (CPU ONLY)
    te1_cls = import_text_encoder_class(args.pretrained_model_name_or_path, "text_encoder")
    te2_cls = import_text_encoder_class(args.pretrained_model_name_or_path, "text_encoder_2")
    te3_cls = import_text_encoder_class(args.pretrained_model_name_or_path, "text_encoder_3")

    text_encoder_one = te1_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype
    ).to(cpu_device)
    text_encoder_two = te2_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", torch_dtype=weight_dtype
    ).to(cpu_device)
    text_encoder_three = te3_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", torch_dtype=weight_dtype
    ).to(cpu_device)

    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)

    # ---------------------------------
    # Add TRUE LyCORIS adapter (LoCon / LoHa)
    # ---------------------------------
    print("Attaching LyCORIS adapter to SD3.5 transformer...")

    # Target preset: here we follow the standard example and adapt attention modules.
    # LyCon itself is implemented inside LyCORIS; we just pick algo="locon" or "loha".
    LycorisNetwork.apply_preset(
        {
            "target_name": [".*attn.*"],  # focus on attention modules; LyCon extends into convs internally
        }
    )

    lyco_net = create_lycoris(
        transformer,
        1.0,  # multiplier (weight at inference time, we keep 1.0 here)
        linear_dim=args.rank,
        linear_alpha=float(args.alpha),
        algo=args.lycoris_algo,
    )
    lyco_net.apply_to()

    # Collect trainable parameters from LyCORIS wrapper
    trainable_params = list(lyco_net.parameters())
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable LyCORIS parameters found – something went wrong with create_lycoris().")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        eps=1e-8,
    )

    # ---------------------------------
    # Dataset
    # ---------------------------------
    train_dataset = RenderExpoCaptionDataset(
        instance_data_root=args.instance_data_dir,
        default_prompt=args.instance_prompt,
        resolution=args.resolution,
        center_crop=False,
        caption_dropout=args.caption_dropout,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
    )

    # ---------------------------------
    # Training loop
    # ---------------------------------
    total_steps = args.max_train_steps
    global_step = 0

    print(f"Starting PRO v2 training for {total_steps} steps (LyCORIS {args.lycoris_algo}, 896px)...")
    progress_bar = tqdm(range(total_steps), desc="Training steps")

    transformer.train()

    # noise_scale controls how hard the denoising task is; small is more stable
    noise_scale = 0.2

    # Optional: record last loss for quick sanity check
    last_loss_value = None

    while global_step < total_steps:
        for batch in train_dataloader:
            if global_step >= total_steps:
                break

            pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
            prompts = batch["prompt"]  # list of strings

            # Encode prompts using CPU text encoders, then move embeds to GPU
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
                )

            # Encode images to latents with VAE (fp32)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()

            # Scaling for SD
            if hasattr(vae.config, "scaling_factor"):
                latents = latents * vae.config.scaling_factor

            latents = torch.clamp(latents, -10.0, 10.0)

            # Sample noise
            noise = torch.randn_like(latents)
            noise = torch.clamp(noise, -10.0, 10.0)

            # Simple fixed timestep (0) noise setup
            timesteps = torch.zeros(
                latents.shape[0],
                device=device,
                dtype=torch.long,
            )

            noisy_latents = latents + noise_scale * noise
            noisy_latents = torch.clamp(noisy_latents, -10.0, 10.0)

            # Set LR from three-phase scheduler
            lr = three_phase_lr(
                global_step,
                total_steps,
                lr_start=args.learning_rate,
                lr_mid=args.lr_mid,
                lr_final=args.lr_final,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            model_pred = transformer(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]

            # Check for NaNs in model_pred before loss
            if not torch.isfinite(model_pred).all():
                print(f"⚠️ model_pred has non-finite values at step {global_step}, skipping.")
                global_step += 1
                progress_bar.update(1)
                continue

            # MSE loss in float32
            loss = F.mse_loss(model_pred, noise)

            if not torch.isfinite(loss):
                print(
                    f"⚠️ Non-finite loss detected at step {global_step}: {loss.item()}, skipping this step."
                )
                global_step += 1
                progress_bar.update(1)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            last_loss_value = float(loss.item())
            global_step += 1
            progress_bar.set_postfix({"loss": last_loss_value, "lr": lr})
            progress_bar.update(1)

            # Periodic checkpoint saving
            if args.save_steps > 0 and (global_step % args.save_steps == 0):
                ckpt_path = os.path.join(
                    args.output_dir,
                    f"pytorch_lora_lycoris_step_{global_step:06d}.safetensors",
                )
                print(f"Saving intermediate LyCORIS weights at step {global_step} -> {ckpt_path}")
                lyco_state = lyco_net.state_dict()
                sft.save_file(lyco_state, ckpt_path)

            if global_step >= total_steps:
                break

    progress_bar.close()
    print("Training finished, saving FINAL LyCORIS weights...")

    # ---------------------------------
    # Save final LyCORIS weights
    # ---------------------------------
    transformer.eval()
    lyco_state = lyco_net.state_dict()

    # Free big stuff before saving
    del vae, text_encoder_one, text_encoder_two, text_encoder_three
    free_memory()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs(args.output_dir, exist_ok=True)
    final_path = os.path.join(args.output_dir, "pytorch_lora_lycoris_pro_v2_final.safetensors")
    sft.save_file(lyco_state, final_path)

    print("==========================================")
    print("✅ RENDEREXPO PRO v2 LyCORIS training DONE")
    print(f"Final LyCORIS file: {final_path}")
    if last_loss_value is not None:
        print(f"Final loss (last step): {last_loss_value:.6f}")
    print("==========================================")


if __name__ == "__main__":
    main()
