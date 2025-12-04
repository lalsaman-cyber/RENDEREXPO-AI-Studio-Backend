#!/usr/bin/env python
# RENDEREXPO SD3.5-Large LoRA trainer with image+caption pairs
#
# GOAL
# - Train a LoRA on Stable Diffusion 3.5 Large (diffusers format)
# - Use *both* images AND their per-image captions (225 images + 225 .txt files)
# - No special RENDEREXPO token needed in the user prompt.
#   In production we simply always load this LoRA into the pipeline → RENDEREXPO style by default.
#
# ASSUMPTIONS
# - For every image like `xxx.jpg` in instance_data_dir, there is a `xxx.txt` with the caption.
# - Images: .jpg/.jpeg/.png/.webp
# - Captions: UTF-8 text, one prompt per file.
#
# DESIGN
# - VAE + Transformer run in float32 on GPU for stability.
# - Text encoders stay on CPU (to avoid GPU OOM).
# - Training objective is a stable, simplified noise-prediction MSE, same as the working v2 script,
#   but now with *per-image* prompt embeddings instead of one fixed global prompt.

import argparse
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
    AutoencoderKL,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from diffusers.training_utils import free_memory
from peft import LoraConfig, get_peft_model_state_dict


# ------------------------------
# Dataset: image + caption pairs
# ------------------------------


class RenderExpoImageCaptionDataset(Dataset):
    """
    Dataset that:
      - Loads images from instance_data_root
      - Matches each image with a .txt caption file having the same stem
      - Falls back to a global instance_prompt if caption file is missing

    Example:
      my_image_001.jpg  <-- image
      my_image_001.txt  <-- caption (same stem)
    """

    def __init__(
        self,
        instance_data_root: str,
        default_prompt: str,
        resolution: int = 512,
        center_crop: bool = False,
    ):
        self.default_prompt = default_prompt
        self.center_crop = center_crop
        self.resolution = resolution

        root = Path(instance_data_root)
        if not root.exists():
            raise ValueError(f"Instance data root does not exist: {instance_data_root}")

        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        self.samples = []

        for p in sorted(root.iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower() not in valid_exts:
                continue

            txt_path = p.with_suffix(".txt")
            if txt_path.exists():
                try:
                    caption = txt_path.read_text(encoding="utf-8").strip()
                except Exception:
                    caption = default_prompt
            else:
                caption = default_prompt

            if not caption:
                caption = default_prompt

            self.samples.append((p, caption))

        if len(self.samples) == 0:
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
        return len(self.samples)

    def __getitem__(self, idx):
        path, caption = self.samples[idx]
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

        return {
            "pixel_values": image,
            "caption": caption,
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


@torch.no_grad()
def encode_prompt_sd3_batch(
    prompts,
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
    Vectorized version of encode_prompt_sd3:
      - prompts: list[str] or a 1D batch of strings
      - text encoders live on CPU
      - returns prompt_embeds, pooled_prompt_embeds on `device` in float32

    SD3 uses:
      - 2 x CLIP text encoders
      - 1 x T5 encoder
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    batch_size = len(prompts)

    # ---------- CLIP encoders (two) ----------
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
        input_ids = text_inputs.input_ids  # [B, seq], stays on CPU

        outputs = enc(
            input_ids,
            output_hidden_states=True,
        )

        # For CLIPTextModelWithProjection:
        #  - outputs[0] is usually last_hidden_state or pooled embedding;
        #  - we keep the pattern of the earlier working script.
        pooled = outputs[0]  # [B, hidden] CLS-like pooled
        hidden = outputs.hidden_states[-2]  # [B, seq, hidden] penultimate hidden

        hidden = hidden.to(dtype=torch.float32)
        pooled = pooled.to(dtype=torch.float32)

        clip_embeds_list.append(hidden)
        clip_pooled_list.append(pooled)

    # concat CLIP embeddings along channel dimension
    clip_prompt_embeds = torch.cat(clip_embeds_list, dim=-1)  # [B, seq, C_clip]
    pooled_prompt_embeds = torch.cat(clip_pooled_list, dim=-1)  # [B, C_clip]

    # ---------- T5 encoder ----------
    t5_inputs = tokenizer_three(
        prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    t5_ids = t5_inputs.input_ids  # [B, seq], CPU
    t5_out = text_encoder_three(t5_ids)[0]  # [B, seq, C_t5]
    t5_out = t5_out.to(dtype=torch.float32)

    # Pad CLIP channels to match T5 hidden size along channel dimension, then concat along sequence
    if clip_prompt_embeds.shape[-1] < t5_out.shape[-1]:
        pad_channels = t5_out.shape[-1] - clip_prompt_embeds.shape[-1]
        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, pad_channels)
        )

    prompt_embeds = torch.cat([clip_prompt_embeds, t5_out], dim=-2)  # [B, seq_total, C]

    # Move to requested device (GPU) at the very end
    prompt_embeds = prompt_embeds.to(device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device)

    return prompt_embeds, pooled_prompt_embeds


# ------------------------------
# Training loop
# ------------------------------


def parse_args():
    p = argparse.ArgumentParser("RENDEREXPO SD3.5 LoRA trainer (image+caption pairs)")

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
        help="Folder with training images (jpg/png/webp) + matching .txt captions.",
    )
    p.add_argument(
        "--instance_prompt",
        type=str,
        required=True,
        help='Fallback prompt if a caption .txt is missing, e.g. "renderexpo building style".',
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="sd35_lora_captions_output",
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
        default=1e-5,
        help="Learning rate for LoRA params",
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
        help="Ignored for the core training (we use float32 in VAE/Transformer).",
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
    cpu_device = torch.device("cpu")

    # Everything in VAE + transformer uses float32
    weight_dtype = torch.float32

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------------------------
    # Load SD3.5 components individually
    # ---------------------------------
    print("Loading SD3.5-Large components...")

    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=weight_dtype,
    ).to(device)

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

    # Text encoders (CPU ONLY to avoid OOM)
    te1_cls = import_text_encoder_class(args.pretrained_model_name_or_path, "text_encoder")
    te2_cls = import_text_encoder_class(args.pretrained_model_name_or_path, "text_encoder_2")
    te3_cls = import_text_encoder_class(args.pretrained_model_name_or_path, "text_encoder_3")

    text_encoder_one = te1_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
    ).to(cpu_device)
    text_encoder_two = te2_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        torch_dtype=weight_dtype,
    ).to(cpu_device)
    text_encoder_three = te3_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_3",
        torch_dtype=weight_dtype,
    ).to(cpu_device)

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
    # Dataset & DataLoader
    # ---------------------------
    train_dataset = RenderExpoImageCaptionDataset(
        instance_data_root=args.instance_data_dir,
        default_prompt=args.instance_prompt,
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
    # Training loop (simple noise, fp32)
    # ---------------------------------
    total_steps = args.max_train_steps
    global_step = 0

    print(f"Starting training for {total_steps} steps (image+caption objective, fp32)...")
    progress_bar = tqdm(range(total_steps), desc="Training steps")

    transformer.train()

    noise_scale = 0.2  # small, safe noise

    while global_step < total_steps:
        for batch in train_dataloader:
            if global_step >= total_steps:
                break

            pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
            captions = batch["caption"]  # list of strings of length B

            # Encode images to latents in float32 for stability
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()

            if hasattr(vae.config, "scaling_factor"):
                latents = latents * vae.config.scaling_factor

            latents = torch.clamp(latents, -10.0, 10.0)

            # Sample noise
            noise = torch.randn_like(latents)
            noise = torch.clamp(noise, -10.0, 10.0)

            # Fake timestep (0) and simple noise mixing
            timesteps = torch.zeros(
                latents.shape[0],
                device=device,
                dtype=torch.long,
            )

            noisy_latents = latents + noise_scale * noise
            noisy_latents = torch.clamp(noisy_latents, -10.0, 10.0)

            # Encode captions → prompt embeddings (on CPU, then move to GPU)
            prompt_embeds, pooled_prompt_embeds = encode_prompt_sd3_batch(
                captions,
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

            optimizer.zero_grad(set_to_none=True)

            model_pred = transformer(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]

            if not torch.isfinite(model_pred).all():
                print(f"⚠️ model_pred has non-finite values at step {global_step}, skipping.")
                global_step += 1
                progress_bar.update(1)
                continue

            loss = F.mse_loss(model_pred, noise)

            if not torch.isfinite(loss):
                print(
                    f"⚠️ Non-finite loss detected at step {global_step}: {loss.item()}, "
                    "skipping this step."
                )
                global_step += 1
                progress_bar.update(1)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
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

    # Free GPU before building pipeline for saving
    del vae, text_encoder_one, text_encoder_two, text_encoder_three
    free_memory()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
    )

    pipe.save_lora_weights(
        save_directory=args.output_dir,
        transformer_lora_layers=lora_state,
        text_encoder_lora_layers=None,
        text_encoder_2_lora_layers=None,
    )

    print(f"LoRA weights saved to: {args.output_dir}")

    free_memory()


if __name__ == "__main__":
    main()
