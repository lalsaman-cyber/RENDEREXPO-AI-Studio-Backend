#!/usr/bin/env bash
set -e

# -------------------------------------------------------------------
# RENDEREXPO AI STUDIO – START SCRIPT
# -------------------------------------------------------------------

# 1) Go to the backend code folder baked into the image
cd /workspace/RENDEREXPO-AI-Studio-Backend

# 2) Activate the virtual environment
source .venv/bin/activate

# 3) SD3.5 model path
#    Default points to the mounted RunPod volume.
#    Can be overridden via environment variable if needed.
export SD35_MODEL_PATH="${SD35_MODEL_PATH:-/workspace-data/models/sd35-large}"

# 4) Runtime mode flags (override-safe)
export SD35_RUNTIME_MODE="${SD35_RUNTIME_MODE:-real}"
export RUN_REAL_SD35="${RUN_REAL_SD35:-1}"

# -------------------------------------------------------------------
# Start services
# -------------------------------------------------------------------

# 5) Start GPU worker (SD3.5) on port 8011 (background)
uvicorn app.gpu_entry:app --host 0.0.0.0 --port 8011 &

# 6) Start CPU planner on port 8000 (foreground)
uvicorn app.main:app --host 0.0.0.0 --port 8000
