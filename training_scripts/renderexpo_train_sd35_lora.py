#!/usr/bin/env python
# RENDEREXPO SD3.5-Large LoRA trainer (stable version)
#
# - Trains LoRA on Stable Diffusion 3.5 Large (diffusers format)
# - Uses a local image folder (instance_data_dir)
# - Ignores .txt caption files; uses a single instance_prompt for all images
# - Uses normalized sigmas + full float32 math to avoid NaNs
#
# Designed for your setup:
#   --pretrained_model_name_or_path /workspace-data/models/sd35-large
#   --instance_data_dir /workspace-data/lora_datasets/lora_test_dataset

import argparse
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import crop
from PIL import Image
from PIL.ImageOps import exif_transpose
from tqdm.auto import tqdm

from transformers import CLIPTokenizer, T5TokenizerFast, PretrainedConfig

from diffusers import (
    StableDiffusion3Pipeline,
    FlowMatchEulerDiscreteScheduler,
    AutoencoderKL,
    SD3Transformer2DModel,
)
from diffusers.training_utils import free_memory
from peft import LoraConfig, get_peft_model_state_dict


# ------------------------------
# Dataset: images only, ignore .txt
# ------------------------------


class RenderExpoImageDataset(Dataset):
    """
    Simple dataset:
    - Loads images from instance_data_root
    - Ignores .txt files
    - Outputs pixel_values tensor and a fixed instance_prompt string
    """

    def __init__(
        self,
        instance_data_root: str,
        instance_prompt: str,
        resolution: int = 512,
        center_crop: bool = False,
    ):
        self.instance_prompt = instance_prompt
        self.center_crop = center_crop
        self.resolution = resolution

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

        # Basic transforms
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

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path)
        image = exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = self.resize(image)
        # Random flip 50% of the time
        if random.random() < 0.5:
            image = self.hflip(image)

        if self.center_crop:
            # Center crop
            image = self.center_crop_tf(image)
        else:
            # Random crop
            y1, x1, h, w = self.random_crop.get_params(
                image, (self.resolution, self.resolution)
            )
            image = crop(image, y1, x1, h, w)

        image = self.to_tensor(image)
        image = self.normalize(image)

        return {
            "pixel_values": image,
            "instance_prompt": self.instance_prompt,
        }


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
    prompt: str,
    text_encoder_one,
    text_encoder_two,
    text_encoder_three,
    tokenizer_one,
    tokenizer_two,
    tokenizer_three,
    max_sequence_length: int = 77,
    device: torch.device = torch.device("cuda"),
):
    """
    Encodes a single prompt for SD3:
    - 2x CLIP text encoders
    - 1x T5 encoder
    Returns prompt_embeds, pooled_prompt_embeds
    """
    prompt_list = [prompt]

    # --- CLIP encoders (two) ---
    clip_tokenizers = [tokenizer_one, tokenizer_two]
    clip_encoders = [text_encoder_one, text_encoder_two]
    clip_embeds_list = []
    clip_pooled_list = []

    for tok, enc in zip(clip_tokenizers, clip_encoders):
        text_inputs = tok(
            prompt_list,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)

        outputs = enc(
            input_ids,
            output_hidden_states=True,
        )
        pooled = outputs[0]  # last layer CLS
        hidden = outputs.hidden_states[-2]  # penultimate hidden

        hidden = hidden.to(dtype=enc.dtype, device=device)
        pooled = pooled.to(dtype=enc.dtype, device=device)

        # [batch, seq, hidden]
        clip_embeds_list.append(hidden)
        clip_pooled_list.append(pooled)

    clip_prompt_embeds = torch.cat(clip_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_list, dim=-1)

    # --- T5 encoder ---
    t5_inputs = tokenizer_three(
        prompt_list,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    t5_ids = t5_inputs.input_ids.to(device)
    t5_out = text_encoder_three(t5_ids)[0]  # [batch, seq, hidden]
    t5_out = t5_out.to(dtype=text_encoder_three.dtype, device=device)

    # Pad CLIP channels to match T5 hidden size along channel dimension, then concat along sequence
    if clip_prompt_embeds.shape[-1] < t5_out.shape[-1]:
        pad_channels = t5_out.shape[-1] - clip_prompt_embeds.shape[-1]
        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, pad_channels)
        )

    prompt_embeds = torch.cat([clip_prompt_embeds, t5_out], dim=-2)

    return prompt_embeds, pooled_prompt_embeds


# ------------------------------
# Training loop
# ------------------------------


def parse_args():
    p = argparse.ArgumentParser("RENDEREXPO SD3.5 LoRA trainer")

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
        help="Folder with training images (jpg/png/webp). .txt will be ignored.",
    )
    p.add_argument(
        "--instance_prompt",
        type=str,
        required=True,
        help='Prompt that describes your instance, e.g. "renderexpo building lora style".',
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="sd35_lora_output",
        help="Where to save LoRA weights",
    )
    p.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Training resolution",
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
        default=800,
        help="Total training steps",
    )
    p.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,  # safer default
    )
    p.add_argument(
        "--rank",
        type=int,
        default=8,
        help="LoRA rank",
    )
    p.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Kept for compatibility; training math is forced to float32 for stability.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # For stability: force full float32 math during training
    weight_dtype = torch.float32

    # ---------------------------------
    # Load base SD3.5 pipeline & pieces
    # ---------------------------------
    print("Loading SD3.5-Large pipeline...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # Grab components
    vae: AutoencoderKL = pipe.vae
    transformer: SD3Transformer2DModel = pipe.transformer
    tokenizer_one: CLIPTokenizer = pipe.tokenizer
    tokenizer_two: CLIPTokenizer = pipe.tokenizer_2
    tokenizer_three: T5TokenizerFast = pipe.tokenizer_3

    # Load text encoders in trainable form (not frozen copies)
    te1_cls = import_text_encoder_class(args.pretrained_model_name_or_path, "text_encoder")
    te2_cls = import_text_encoder_class(args.pretrained_model_name_or_path, "text_encoder_2")
    te3_cls = import_text_encoder_class(args.pretrained_model_name_or_path, "text_encoder_3")

    text_encoder_one = te1_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    ).to(device, dtype=weight_dtype)
    text_encoder_two = te2_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2"
    ).to(device, dtype=weight_dtype)
    text_encoder_three = te3_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3"
    ).to(device, dtype=weight_dtype)

    # Freeze everything except LoRA layers
    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)

    # ---------------------------
    # Add LoRA to transformer
    # ---------------------------
    target_modules = [
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "attn.to_k",
        "attn.to_out.0",
        "attn.to_q",
        "attn.to_v",
    ]

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=0.0,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(lora_config)

    # Collect trainable LoRA params
    lora_params = [p for p in transformer.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        lora_params,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        eps=1e-8,
    )

    # ---------------------------
    # Scheduler & dataset
    # ---------------------------
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    schedule_sigmas = noise_scheduler.sigmas.to(device, dtype=torch.float32)

    # Normalize sigmas into [0, 1] to avoid huge scales
    max_sigma = schedule_sigmas.max().clamp(min=1e-6)
    norm_sigmas = schedule_sigmas / max_sigma  # in [0, 1]

    train_dataset = RenderExpoImageDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        resolution=args.resolution,
        center_crop=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
    )

    # ---------------------------------
    # Pre-encode the fixed instance prompt
    # ---------------------------------
    print("Encoding instance prompt...")
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt_sd3(
            args.instance_prompt,
            text_encoder_one,
            text_encoder_two,
            text_encoder_three,
            tokenizer_one,
            tokenizer_two,
            tokenizer_three,
            max_sequence_length=77,
            device=device,
        )

    prompt_embeds = prompt_embeds.to(device, dtype=weight_dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device, dtype=weight_dtype)

    # ---------------------------------
    # Training loop
    # ---------------------------------
    total_steps = args.max_train_steps
    global_step = 0

    print(f"Starting training for {total_steps} steps...")
    progress_bar = tqdm(range(total_steps), desc="Training steps")

    vae.to(device, dtype=weight_dtype)
    transformer.to(device, dtype=weight_dtype)

    num_t = schedule_timesteps.shape[0]
    # Avoid extreme timesteps (use 90% of range)
    max_idx_for_sampling = max(1, int(0.9 * num_t))

    while global_step < total_steps:
        for batch in train_dataloader:
            if global_step >= total_steps:
                break

            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)

            # Encode images to latents
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()

            # Standard latent scaling for SD models
            latents = latents * vae.config.scaling_factor
            latents = latents.to(dtype=weight_dtype)

            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            idx = torch.randint(
                0,
                max_idx_for_sampling,
                (bsz,),
                device=device,
                dtype=torch.long,
            )
            timesteps = schedule_timesteps[idx]

            # Normalized sigma in [0, 1]
            sigma = norm_sigmas[idx].view(bsz, 1, 1, 1)

            # Simple noise mixing: x_t = x_0 + sigma * noise
            noisy_latents = latents + sigma * noise

            transformer.train()
            optimizer.zero_grad()

            model_pred = transformer(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds.repeat(bsz, 1, 1),
                pooled_projections=pooled_prompt_embeds.repeat(bsz, 1),
                return_dict=False,
            )[0]

            # Target is the true noise
            loss = F.mse_loss(model_pred.float(), noise.float())

            if not torch.isfinite(loss):
                print(f"⚠️ Non-finite loss detected at step {global_step}: {loss.item()}, skipping this step.")
                # Still count the step so we don't loop forever
                global_step += 1
                progress_bar.update(1)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_((lora_params), 1.0)
            optimizer.step()

            global_step += 1
            progress_bar.set_postfix({"loss": float(loss.item())})
            progress_bar.update(1)

            if global_step >= total_steps:
                break

    progress_bar.close()
    print("Training finished, saving LoRA weights...")

    # ---------------------------------
    # Save LoRA weights (PEFT → diffusers format)
    # ---------------------------------
    transformer.eval()
    lora_state = get_peft_model_state_dict(transformer)

    StableDiffusion3Pipeline.save_lora_weights(
        save_directory=args.output_dir,
        transformer_lora_layers=lora_state,
        text_encoder_lora_layers=None,
        text_encoder_2_lora_layers=None,
    )

    print(f"LoRA weights saved to: {args.output_dir}")
    del pipe
    free_memory()


if __name__ == "__main__":
    main()
