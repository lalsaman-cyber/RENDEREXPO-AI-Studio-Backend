#!/usr/bin/env bash
set -e

cd /workspace/RENDEREXPO-AI-Studio-Backend
source .venv/bin/activate

export SD35_MODEL_PATH=/workspace/models/sd35-large

# GPU worker – SD3.5 real mode
export SD35_RUNTIME_MODE=real
export RUN_REAL_SD35=1
uvicorn app.gpu_entry:app --host 0.0.0.0 --port 8011 &

# CPU planner – foreground
uvicorn app.main:app --host 0.0.0.0 --port 8000
