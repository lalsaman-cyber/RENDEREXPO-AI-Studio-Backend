# ============================================
# RENDEREXPO AI STUDIO - GPU Backend (SD3.5)
# Dockerfile for RunPod / CUDA GPU environment
# ============================================

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# -----------------------------
# System packages
# -----------------------------
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    git wget curl vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# -----------------------------
# Copy project files
# -----------------------------
COPY ./app ./app
COPY ./runtime ./runtime
COPY ./config ./config
COPY ./docs ./docs
COPY ./Licenses ./Licenses
COPY ./requirements.txt ./requirements.txt
COPY ./start.sh ./start.sh

# Models & outputs will be mounted as volumes later:
#   /workspace/models
#   /workspace/outputs

# Make start.sh executable
RUN chmod +x /workspace/start.sh

# -----------------------------
# Python & core deps
# -----------------------------
RUN python3 -m pip install --upgrade pip

# 1) Install GPU Torch first (CUDA wheels)
RUN python3 -m pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0

# 2) Install the main libraries (FastAPI, diffusers, etc.)
#    We install the diffusion stack explicitly so dependencies are pulled,
#    then install the rest of requirements without dependencies to avoid
#    re-resolving Torch.
RUN python3 -m pip install \
    fastapi==0.109.2 \
    uvicorn[standard]==0.29.0 \
    starlette==0.36.3 \
    pydantic==2.12.4 \
    typing-extensions==4.14.1 \
    tqdm==4.67.1 \
    python-dotenv==1.0.1 \
    diffusers==0.27.2 \
    transformers==4.39.3 \
    accelerate==0.28.0 \
    safetensors==0.4.2 \
    huggingface_hub==0.23.0 \
    Pillow==10.2.0 \
    opencv-python==4.9.0.80 \
    einops==0.7.0 \
    timm==0.9.16

# (Optional) install any extra libs listed in requirements.txt, but
# we disable dependency resolution here to avoid messing with Torch.
RUN python3 -m pip install --no-deps -r requirements.txt || true

# -----------------------------
# Environment variables
# -----------------------------
# SD3.5 model directory inside the container
ENV SD35_MODEL_DIR=/workspace/models/sd35-large
ENV SD_DEVICE=cuda

# HF_TOKEN will be injected in RunPod environment (never hard-coded)
# ENV HF_TOKEN=...

# Outputs directory (will be mounted to a volume in RunPod)
ENV OUTPUTS_ROOT=/workspace/outputs

# -----------------------------
# Networking
# -----------------------------
EXPOSE 8000

# -----------------------------
# Entrypoint
# -----------------------------
CMD ["/workspace/start.sh"]
