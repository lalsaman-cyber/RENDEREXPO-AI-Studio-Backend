# ============================================
# RENDEREXPO AI STUDIO - GPU Backend (SD3.5)
# Dockerfile for RunPod / CUDA GPU environment
# UPDATED FOR:
# - SD3.5 Large
# - dual ControlNet sketch route
# - Canny + Depth preprocessing
# ============================================

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# -----------------------------
# System packages
# -----------------------------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    vim \
    libgl1 \
    libglib2.0-0 \
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

# Models & outputs are expected as mounted volumes:
#   /workspace/models
#   /workspace/outputs

RUN chmod +x /workspace/start.sh

# -----------------------------
# Python bootstrap
# -----------------------------
RUN python3 -m pip install --upgrade pip setuptools wheel

# -----------------------------
# Install GPU Torch first
# IMPORTANT:
# Keep Torch pinned explicitly from CUDA 12.1 wheels.
# Then install the rest from requirements.txt.
# -----------------------------
RUN python3 -m pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.2.2 \
    torchvision==0.17.2 \
    torchaudio==2.2.2

# -----------------------------
# Install backend requirements
# IMPORTANT:
# Do NOT use --no-deps here.
# We want the upgraded diffusers / transformers / controlnet stack
# to resolve correctly for the new sketch route.
# -----------------------------
RUN python3 -m pip install -r requirements.txt

# -----------------------------
# Environment variables
# -----------------------------
# Base SD3.5 model
ENV SD35_MODEL_PATH=/workspace/models/sd35-large

# Official SD3.5 ControlNet model folders
ENV SD35_CONTROLNET_CANNY_PATH=/workspace/models/sd35-controlnet-canny
ENV SD35_CONTROLNET_DEPTH_PATH=/workspace/models/sd35-controlnet-depth

# Depth preprocessor model
ENV SD35_DEPTH_MODEL_ID=Intel/dpt-hybrid-midas

# Runtime controls
ENV SD35_DEVICE=cuda
ENV RUN_REAL_SD35=1
ENV SD35_RUNTIME_MODE=lazy
ENV SD35_ENABLE_CPU_OFFLOAD=0

# Outputs directory
ENV OUTPUTS_ROOT=/workspace/outputs

# HF token is injected at runtime in RunPod if needed
# ENV HF_TOKEN=...

# -----------------------------
# Networking
# -----------------------------
# Main API port
EXPOSE 8000

# -----------------------------
# Entrypoint
# -----------------------------
CMD ["/workspace/start.sh"]