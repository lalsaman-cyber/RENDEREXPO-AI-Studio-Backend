"""
training_scripts/train_lora.py

Skeleton script for training LoRAs on top of SD3.5.

IMPORTANT:
- This is ONLY a placeholder.
- It does NOT run real training.
- It is here so the project structure is ready.

Later, on a GPU machine (RunPod), this script will:
- Parse a YAML config from config/training/lora_*.yaml
- Load SD3.5 base model
- Apply LoRA adapters
- Run the actual training loop
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="RENDEREXPO LoRA training (skeleton).")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training/lora_interiors.yaml",
        help="Path to the LoRA training config file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)

    print("========================================")
    print(" RENDEREXPO AI STUDIO - LoRA Trainer")
    print(" SKELETON ONLY - NO TRAINING RUNS HERE")
    print("========================================")
    print(f"- Expected config: {config_path}")
    print()
    print("Next phases (future):")
    print(" 1) Load YAML config (learning rate, steps, dataset paths).")
    print(" 2) Initialize SD3.5 base model with diffusers / torch.")
    print(" 3) Attach LoRA adapters to the model.")
    print(" 4) Build training dataloader from your private datasets.")
    print(" 5) Run training loop on GPU (RunPod).")
    print()
    print("For now, this script only documents the flow and keeps structure ready.")


if __name__ == "__main__":
    main()
