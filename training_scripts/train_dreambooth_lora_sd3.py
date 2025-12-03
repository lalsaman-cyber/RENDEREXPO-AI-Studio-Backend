#!/usr/bin/env python3

# -------------------------------------------------------------
# RENDEREXPO PATCH:
# The original script requires diffusers 0.36.0.dev0 — too strict.
# We relax that requirement so it works with diffusers 0.35.2+.
# -------------------------------------------------------------

from diffusers.utils import is_xformers_available
# NOTE: We intentionally disable the strict minimum version check.
# from diffusers.utils import check_min_version
# check_min_version("0.36.0.dev0")   # ← removed on purpose

import argparse
import math
import os
from pathlib import Path
from typing import Optional, List

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

class DreamBoothDataset(Dataset):
    """
    RENDEREXPO PATCH:
    - Only load real image files from instance_data_root
    - Ignore .txt caption files (those are for LoRA text, not images)
    """

    def __init__(
        self,
        instance_data_root: str,
        instance_prompt: str,
        class_data_root: Optional[str] = None,
        class_prompt: Optional[str] = None,
        size: int = 512,
        center_crop: bool = False,
        tokenizer=None,
        class_tokenizer=None,
        use_augmentation: bool = False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.instance_prompt = instance_prompt
        self.use_augmentation = use_augmentation
        self.tokenizer = tokenizer
        self.class_tokenizer = class_tokenizer

        instance_root = Path(instance_data_root)

        if not instance_root.exists():
            raise ValueError(f"Instance data root does not exist: {instance_data_root}")

        # ------------------------------------------------------------
        # RENDEREXPO PATCH — IGNORE .txt FILES, LOAD IMAGES ONLY
        # ------------------------------------------------------------
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        instance_image_paths: List[Path] = [
            p for p in instance_root.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        ]

        if len(instance_image_paths) == 0:
            raise ValueError(
                f"No valid images found in dataset folder: {instance_data_root}"
            )

        # Open and normalize to RGB
        self.instance_images: List[Image.Image] = [
            Image.open(p).convert("RGB") for p in instance_image_paths
        ]

        # Optional: class images support (left mostly as original)
        self.class_data_root = class_data_root
        self.class_prompt = class_prompt
        self.class_images: Optional[List[Image.Image]] = None

        if class_data_root is not None:
            class_root = Path(class_data_root)
            class_exts = {".jpg", ".jpeg", ".png", ".webp"}
            class_image_paths: List[Path] = [
                p for p in class_root.iterdir()
                if p.is_file() and p.suffix.lower() in class_exts
            ]
            if len(class_image_paths) == 0:
                raise ValueError(
                    f"No valid class images found in folder: {class_data_root}"
                )
            self.class_images = [Image.open(p).convert("RGB") for p in class_image_paths]

        # Basic torchvision-style transforms
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.size) if self.center_crop else transforms.Resize(self.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        if self.class_images is not None:
            return max(len(self.instance_images), len(self.class_images))
        return len(self.instance_images)

    def __getitem__(self, index: int):
        # Instance image (repeat if index exceeds list)
        img = self.instance_images[index % len(self.instance_images)]
        img = self.image_transforms(img)

        example = {
            "instance_images": img,
            "instance_prompt": self.instance_prompt,
        }

        if self.class_images is not None and self.class_prompt is not None:
            class_img = self.class_images[index % len(self.class_images)]
            class_img = self.image_transforms(class_img)
            example["class_images"] = class_img
            example["class_prompt"] = self.class_prompt

        return example
