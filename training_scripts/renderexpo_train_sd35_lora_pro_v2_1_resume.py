cd /workspace-data/RENDEREXPO-AI-Studio-Backend || exit 1

cat > training_scripts/renderexpo_train_sd35_lora_pro_v2_1_resume.py <<'PY'
#!/usr/bin/env python
# RENDEREXPO SD3.5-Large LyCORIS trainer (PRO v2.1 RESUME, caption-aware, 1024px, rank 16)
#
# - Same training loop as PRO v2.1
# - Adds resume_from (.safetensors) + resume_step (global_step start)
# - Adds configurable grad clipping (max_grad_norm) to reduce fp16 NaN cascades
#
# NOTE: This script is meant to run directly.

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
# Dataset: image + caption pairs
# ------------------------------

class RenderExpoCaptionDataset(Dataset):
    """
    - Scans instance_data_root for image files (.jpg/.jpeg/.png/.webp)
    - For each image `name.ext`, looks for `name.txt` in the same folder
    - If caption exists + non-empty: use it
    - Else: fallback to global --instance_prompt
    - Optional caption_dropout: with probability p, ignore caption and use fallback prompt
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
        self.image_paths = [
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in valid_exts
        ]
        if len(self.image_paths) == 0:
            raise ValueError(f"No valid images found in: {instance_data_root}")

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
                text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
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
    if arch == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
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
    # 2x CLIP
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

    # T5
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
        t5_out = text_encoder_three(t5_ids)[0]

    t5_out = t5_out.to(dtype=torch.float32)

    # Pad CLIP channels to match T5 hidden size (channels), then concat along sequence
    if clip_prompt_embeds.shape[-1] < t5_out.shape[-1]:
        pad_channels = t5_out.shape[-1] - clip_prompt_embeds.shape[-1]
        clip_prompt_embeds = F.pad(clip_prompt_embeds, (0, pad_channels))

    prompt_embeds = torch.cat([clip_prompt_embeds, t5_out], dim=-2)

    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)
    return prompt_embeds, pooled_prompt_embeds


# ------------------------------
# LR schedule (3-phase)
# ------------------------------

def three_phase_lr(step: int, total_steps: int, lr_start: float, lr_mid: float, lr_final: float) -> float:
    if total_steps <= 1:
        return lr_final
    half = total_steps // 2
    if step <= half:
        t = step / max(1, half)
        return lr_start + t * (lr_mid - lr_start)
    t = (step - half) / max(1, total_steps - half)
    return lr_mid + t * (lr_final - lr_mid)


# ------------------------------
# Args
# ------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        "RENDEREXPO SD3.5 LyCORIS trainer PRO v2.1 RESUME (caption-aware, fp16, 1024px, rank 16)"
    )

    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument("--instance_data_dir", type=str, required=True)
    p.add_argument("--instance_prompt", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--max_train_steps", type=int, default=15000)

    # IMPORTANT: defaults kept safe/stable for resume
    p.add_argument("--learning_rate", type=float, default=8e-6)
    p.add_argument("--lr_mid", type=float, default=1.2e-5)
    p.add_argument("--lr_final", type=float, default=8e-6)

    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--lycoris_algo", type=str, default="locon", choices=["locon", "loha"])
    p.add_argument("--caption_dropout", type=float, default=0.10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)

    # Resume + stability
    p.add_argument("--resume_from", type=str, default=None, help="Path to LyCORIS .safetensors checkpoint to resume from.")
    p.add_argument("--resume_step", type=int, default=0, help="Step number of the resume checkpoint (global_step starts here).")
    p.add_argument("--max_grad_norm", type=float, default=0.5, help="Gradient clipping norm (helps prevent fp16 NaNs).")

    return p.parse_args()


# ------------------------------
# Main
# ------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")
    weight_dtype = torch.float16

    print("==============================================")
    print("RENDEREXPO PRO v2.1 – SD3.5 LyCORIS RESUME")
    print("==============================================")
    print(f"Base model path:        {args.pretrained_model_name_or_path}")
    print(f"Instance data dir:      {args.instance_data_dir}")
    print(f"Output dir:             {args.output_dir}")
    print(f"Resolution:             {args.resolution}")
    print(f"Batch size:             {args.train_batch_size}")
    print(f"Max steps:              {args.max_train_steps}")
    print(f"LR schedule:            {args.learning_rate} -> {args.lr_mid} -> {args.lr_final}")
    print(f"LyCORIS algo:           {args.lycoris_algo}")
    print(f"LyCORIS rank (dim):     {args.rank}")
    print(f"LyCORIS alpha:          {args.alpha}")
    print(f"Caption dropout:        {args.caption_dropout}")
    print(f"Resume from:            {args.resume_from}")
    print(f"Resume step:            {args.resume_step}")
    print(f"Max grad norm:          {args.max_grad_norm}")
    print(f"weight_dtype (GPU):     {weight_dtype}")
    print("==============================================")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading SD3.5-Large components...")
    print(f"Using default instance prompt: {args.instance_prompt}")

    # VAE on GPU float32 for stability
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to(device)

    # Transformer on GPU fp16
    transformer: SD3Transformer2DModel = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    ).to(device)

    transformer.enable_gradient_checkpointing()

    # Tokenizers CPU
    tokenizer_one: CLIPTokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    tokenizer_two: CLIPTokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2"
    )
    tokenizer_three: T5TokenizerFast = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_3"
    )

    # Text encoders CPU ONLY
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

    vae.requires_grad_(False)
    transformer.requires_grad_(True)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)

    print("Attaching LyCORIS adapter to SD3.5 transformer...")

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
    print(f"LyCORIS parameters: {sum(p.numel() for p in lyco_params):,}")

    # Resume weights if provided
    if args.resume_from:
        print(f"[Resume] Loading LyCORIS weights from: {args.resume_from}")
        sd = sft.load_file(args.resume_from)
        missing, unexpected = lyco_net.load_state_dict(sd, strict=False)
        print(f"[Resume] Loaded. missing={len(missing)} unexpected={len(unexpected)}")

    optimizer = torch.optim.AdamW(
        lyco_params,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        eps=1e-8,
    )

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

    total_steps = args.max_train_steps
    global_step = int(args.resume_step)

    print(f"Starting PRO v2.1 RESUME training from step {global_step} to {total_steps}...")
    progress_bar = tqdm(range(global_step, total_steps), desc="Training steps", initial=global_step, total=total_steps)

    transformer.train()

    noise_scale = 0.25

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

            lr = three_phase_lr(
                global_step,
                total_steps,
                args.learning_rate,
                args.lr_mid,
                args.lr_final,
            )
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

                loss = F.mse_loss(
                    model_pred.to(dtype=torch.float32),
                    noise.to(dtype=torch.float32),
                )

            if not torch.isfinite(loss):
                print(f"⚠️ Non-finite loss detected at step {global_step}: {loss.item()}, skipping this step.")
                global_step += 1
                progress_bar.update(1)
                continue

            loss.backward()

            if args.max_grad_norm and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(lyco_params, args.max_grad_norm)

            optimizer.step()

            global_step += 1
            progress_bar.set_postfix({"loss": float(loss.item()), "lr": lr})
            progress_bar.update(1)

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                ckpt_path = os.path.join(
                    args.output_dir, f"pytorch_lycoris_step_{global_step:06d}.safetensors"
                )
                state = lyco_net.state_dict()
                sft.save_file(state, ckpt_path)
                print(f"[Checkpoint] Saved LyCORIS weights at step {global_step} -> {ckpt_path}")

            if global_step >= total_steps:
                break

    progress_bar.close()
    print("Training finished, saving final LyCORIS weights...")

    transformer.eval()

    del vae, text_encoder_one, text_encoder_two, text_encoder_three
    free_memory()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_path = os.path.join(args.output_dir, "pytorch_lycoris_final.safetensors")
    os.makedirs(args.output_dir, exist_ok=True)
    sft.save_file(lyco_net.state_dict(), final_path)
    print(f"✅ LyCORIS weights saved to: {final_path}")


if __name__ == "__main__":
    main()
PY
