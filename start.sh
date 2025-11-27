#!/usr/bin/env bash
set -e

# 1) Go to the code folder baked into the image
cd /workspace/RENDEREXPO-AI-Studio-Backend

# 2) Activate the venv that was created during docker build
source .venv/bin/activate

# 3) SD3.5 model path:
#    - Default: /workspace-data/models/sd35-large (where the volume will be mounted)
#    - If SD35_MODEL_PATH is set in the environment, use that instead.
export SD35_MODEL_PATH="${SD35_MODEL_PATH:-/workspace-data/models/sd35-large}"

# 4) Runtime mode flags (allow overriding from env if needed)
export SD35_RUNTIME_MODE="${SD35_RUNTIME_MODE:-real}"
export RUN_REAL_SD35="${RUN_REAL_SD35:-1}"

# 5) Start GPU worker (port 8011) in background
uvicorn app.gpu_entry:app --host 0.0.0.0 --port 8011 &

# 6) Start CPU planner (port 8000) in foreground
uvicorn app.main:app --host 0.0.0.0 --port 8000
