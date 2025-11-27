import os
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_REPO = "stabilityai/stable-diffusion-3.5-large"
TARGET_DIR = Path("/workspace/models/sd35-large")

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable is not set. Make sure it is configured in the pod environment.")

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_REPO} to {TARGET_DIR} ...")
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=str(TARGET_DIR),
        local_dir_use_symlinks=False,
        token=token,
        ignore_patterns=["*.md", "LICENSE*"]
    )
    print("Download complete.")

if __name__ == "__main__":
    main()
