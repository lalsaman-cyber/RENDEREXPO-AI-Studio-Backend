#!/usr/bin/env python
# RENDEREXPO SD3.5-Large LyCORIS trainer (PRO v2.1 RESUME, caption-aware, 1024px, rank 16)
#
# - Same as PRO v2.1
# - Adds SAFE resume from LyCORIS checkpoint
# - Adds gradient clipping control
# - Prevents NaN cascades seen after ~6k steps
#
# ✅ THIS FILE IS MEANT TO BE RUN DIRECTLY

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

from lycoris import create_lycoris, LycorisNetwork
import safetensors.torch as sft


# ------------------------------
# Dataset
# ------------------------------

class RenderExpoCaptionDataset(Dataset):
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
        self.image_paths = [
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in valid_exts
        ]

        if len(self.image_paths) == 0:
            raise ValueError("No training images found.")

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
        txt = image_path.with_suffix(".txt")
        if txt.exists():
            try:
                t = txt.read_text(encoding="utf-8").strip()
                if t:
                    return t
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

        if random.random() < 0.5:
            image = self.hflip(image)

        if self.center_crop:
            image = self.center_crop_tf(image)
        else:
            y1, x1, h, w = self.random_crop.get_params(
                image, (self.resolution, self.resolution)
            )
            image = crop(image, y1, x1, h, w)

        image = self.normalize(self.to_tensor(image))

        caption = self._load_caption(path)
        if self.caption_dropout > 0 and random.random() < self.caption_dropout:
            caption = self.default_prompt

        return {"pixel_values": image, "prompt": caption}


# ------------------------------
# Text encoding helpers
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
        raise ValueError(f"Unsupported encoder: {arch}")


def encode_prompt_sd3(
    prompts: List[str],
    te1, te2, te3,
    tok1, tok2, tok3,
    device, dtype
):
    clip_embeds = []
    pooled = []

    for tok, enc in [(tok1, te1), (tok2, te2)]:
        inputs = tok(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = enc(inputs.input_ids, output_hidden_states=True)
        clip_embeds.append(out.hidden_states[-2].float())
        pooled.append(out[0].float())

    clip_embeds = torch.cat(clip_embeds, dim=-1)
    pooled = torch.cat(pooled, dim=-1)

    t5_inputs = tok3(
        prompts,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        t5_out = te3(t5_inputs.input_ids)[0].float()

    if clip_embeds.shape[-1] < t5_out.shape[-1]:
        pad = t5_out.shape[-1] - clip_embeds.shape[-1]
        clip_embeds = F.pad(clip_embeds, (0, pad))

    embeds = torch.cat([clip_embeds, t5_out], dim=-2)
    return embeds.to(device, dtype), pooled.to(device, dtype)


# ------------------------------
# LR schedule
# ------------------------------

def three_phase_lr(step, total, a, b, c):
    half = total // 2
    if step <= half:
        return a + (b - a) * (step / max(1, half))
    return b + (c - b) * ((step - half) / max(1, total - half))


# ------------------------------
# Args
# ------------------------------

def parse_args():
    p = argparse.ArgumentParser("RENDEREXPO SD3.5 LyCORIS PRO v2.1 RESUME")

    p.add_argument("--pretrained_model_name_or_path", required=True)
    p.add_argument("--instance_data_dir", required=True)
    p.add_argument("--instance_prompt", required=True)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--max_train_steps", type=int, default=15000)

    p.add_argument("--learning_rate", type=float, default=8e-6)
    p.add_argument("--lr_mid", type=float, default=1.2e-5)
    p.add_argument("--lr_final", type=float, default=8e-6)

    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--lycoris_algo", default="locon")
    p.add_argument("--caption_dropout", type=float, default=0.10)

    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)

    # ✅ RESUME
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--resume_step", type=int, default=0)
    p.add_argument("--max_grad_norm", type=float, default=0.5)

    return p.parse_args()


# ------------------------------
# Main
# ------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda")
    cpu = torch.device("cpu")
    dtype = torch.float16

    print("=== RENDEREXPO PRO v2.1 RESUME TRAINER ===")
    print(f"Resume from: {args.resume_from}")
    print(f"Resume step: {args.resume_step}")

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to(device)

    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=dtype,
    ).to(device)

    transformer.enable_gradient_checkpointing()

    tok1 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    tok3 = T5TokenizerFast.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_3")

    te1 = import_text_encoder_class(args.pretrained_model_name_or_path, "text_encoder") \
        .from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").to(cpu)
    te2 = import_text_encoder_class(args.pretrained_model_name_or_path, "text_encoder_2") \
        .from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2").to(cpu)
    te3 = import_text_encoder_class(args.pretrained_model_name_or_path, "text_encoder_3") \
        .from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_3").to(cpu)

    LycorisNetwork.apply_preset({"target_name": [".*attn.*"]})

    lyco = create_lycoris(
        transformer,
        linear_dim=args.rank,
        linear_alpha=args.alpha,
        algo=args.lycoris_algo,
    )
    lyco.apply_to()

    if args.resume_from:
        print(f"[Resume] Loading {args.resume_from}")
        sd = sft.load_file(args.resume_from)
        lyco.load_state_dict(sd, strict=False)
missing, unexpected = lyco.load_state_dict(sd, strict=False)
print(f"[Resume] missing={len(missing)} unexpected={len(unexpected)}")


    optimizer = torch.optim.AdamW(
        lyco.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
        eps=1e-8,
    )

    dataset = RenderExpoCaptionDataset(
        args.instance_data_dir,
        args.instance_prompt,
        args.resolution,
        caption_dropout=args.caption_dropout,
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    total = args.max_train_steps
    step = args.resume_step

    bar = tqdm(range(step, total), initial=step, total=total, desc="Training steps")
    noise_scale = 0.25

    transformer.train()

    while step < total:
        for batch in loader:
            if step >= total:
                break

            imgs = batch["pixel_values"].to(device, torch.float32)
            prompts = batch["prompt"]

            with torch.no_grad():
                embeds, pooled = encode_prompt_sd3(
                    prompts, te1, te2, te3, tok1, tok2, tok3, device, dtype
                )
                latents = vae.encode(imgs).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents = latents.clamp(-10, 10).to(device, dtype)

            noise = torch.randn_like(latents).clamp(-10, 10)
            noisy = (latents + noise * noise_scale).clamp(-10, 10)
            timesteps = torch.zeros(latents.shape[0], device=device, dtype=torch.long)

            lr = three_phase_lr(step, total, args.learning_rate, args.lr_mid, args.lr_final)
            optimizer.param_groups[0]["lr"] = lr
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(True):
                pred = transformer(
                    noisy,
                    timestep=timesteps,
                    encoder_hidden_states=embeds,
                    pooled_projections=pooled,
                )[0]
                loss = F.mse_loss(pred.float(), noise.float())

            if not torch.isfinite(loss):
                print(f"⚠️ NaN at step {step}, skipping")
                step += 1
                bar.update(1)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(lyco.parameters(), args.max_grad_norm)
            optimizer.step()

            step += 1
            bar.set_postfix(loss=float(loss), lr=lr)
            bar.update(1)

            if step % args.save_steps == 0:
                path = f"{args.output_dir}/pytorch_lycoris_step_{step:06d}.safetensors"
                sft.save_file(lyco.state_dict(), path)
                print(f"[Checkpoint] Saved {path}")

    bar.close()

    final = f"{args.output_dir}/pytorch_lycoris_final.safetensors"
    sft.save_file(lyco.state_dict(), final)
    print(f"✅ Final saved: {final}")


if __name__ == "__main__":
    main()
