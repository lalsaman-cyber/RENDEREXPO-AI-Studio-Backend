#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------------
# RENDEREXPO AI STUDIO – START SCRIPT
# -------------------------------------------------------------------
# Canonical service mapping:
#   Planner    -> 8012
#   GPU worker -> 8002
#   Jupyter    -> 8888 (started separately if needed)
#
# Canonical project root:
#   /workspace-data/RENDEREXPO-AI-Studio-Backend
#
# IMPORTANT:
# - NEVER boot from /workspace/... old image copy
# - Boot LIGHT by default
# - Do NOT eagerly preload SD3.5 unless explicitly enabled
# - Sketch mode now uses SD3.5 dual ControlNet:
#       sketch -> canny + depth -> SD3.5 -> output
# -------------------------------------------------------------------

PROJECT_ROOT="/workspace-data/RENDEREXPO-AI-Studio-Backend"
LOG_DIR="${PROJECT_ROOT}/logs"

# 1) Hard-stop if canonical project root is missing
if [ ! -d "${PROJECT_ROOT}" ]; then
  echo "FATAL: canonical project root not found: ${PROJECT_ROOT}" >&2
  exit 1
fi

# 2) Enter canonical repo only
cd "${PROJECT_ROOT}" || exit 1

# 3) Hard-stop if venv is missing
if [ ! -f ".venv/bin/activate" ]; then
  echo "FATAL: virtualenv activate script not found at ${PROJECT_ROOT}/.venv/bin/activate" >&2
  exit 1
fi

# 4) Activate venv
source .venv/bin/activate

# 5) Canonical Python path
export PYTHONPATH="${PROJECT_ROOT}"

# -------------------------------------------------------------------
# Canonical model/env paths
# -------------------------------------------------------------------

# Base SD3.5 model
export SD35_MODEL_PATH="${SD35_MODEL_PATH:-/workspace-data/models/sd35-large}"

# Official SD3.5 ControlNet model folders
export SD35_CONTROLNET_CANNY_PATH="${SD35_CONTROLNET_CANNY_PATH:-/workspace-data/models/sd35-controlnet-canny}"
export SD35_CONTROLNET_DEPTH_PATH="${SD35_CONTROLNET_DEPTH_PATH:-/workspace-data/models/sd35-controlnet-depth}"

# Depth preprocessor model id
export SD35_DEPTH_MODEL_ID="${SD35_DEPTH_MODEL_ID:-Intel/dpt-hybrid-midas}"

# Device/runtime controls
export SD35_DEVICE="${SD35_DEVICE:-cuda}"
export SD35_RUNTIME_MODE="${SD35_RUNTIME_MODE:-lazy}"
export RUN_REAL_SD35="${RUN_REAL_SD35:-1}"
export PRELOAD_SD35_ON_STARTUP="${PRELOAD_SD35_ON_STARTUP:-0}"
export SD35_ENABLE_CPU_OFFLOAD="${SD35_ENABLE_CPU_OFFLOAD:-0}"

# Outputs and planner -> GPU dispatch
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-/workspace-data/outputs}"
export RENDEREXPO_OUTPUTS_MOUNT="${RENDEREXPO_OUTPUTS_MOUNT:-/outputs}"
export RENDEREXPO_GPU_DISPATCH_URL="${RENDEREXPO_GPU_DISPATCH_URL:-http://127.0.0.1:8002/api/gpu/dispatch}"
export RENDEREXPO_GPU_TIMEOUT_SECONDS="${RENDEREXPO_GPU_TIMEOUT_SECONDS:-180}"
export RENDEREXPO_GPU_POLL_SECONDS="${RENDEREXPO_GPU_POLL_SECONDS:-0.5}"

# Worker label
export GPU_WORKER_NAME="${GPU_WORKER_NAME:-runpod-gpu-worker}"

# Optional Hugging Face token passthrough
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"

# 6) Optional HMAC secret load
if [ -z "${RENDEREXPO_HMAC_SECRET:-}" ] && [ -f "secrets/hmac_secret.txt" ]; then
  export RENDEREXPO_HMAC_SECRET="$(cat secrets/hmac_secret.txt)"
fi

# 7) Logs folder
mkdir -p "${LOG_DIR}"

# 8) Make sure outputs folder exists
mkdir -p "${OUTPUTS_ROOT}"

# 9) Safety cleanup: stop old listeners on canonical ports only
PID8002="$(ss -ltnp 2>/dev/null | awk '/:8002/ {print $NF}' | sed -E 's/.*pid=([0-9]+).*/\1/' | head -n1 || true)"
PID8012="$(ss -ltnp 2>/dev/null | awk '/:8012/ {print $NF}' | sed -E 's/.*pid=([0-9]+).*/\1/' | head -n1 || true)"

if [ -n "${PID8002:-}" ]; then
  kill -TERM "${PID8002}" 2>/dev/null || true
fi
if [ -n "${PID8012:-}" ]; then
  kill -TERM "${PID8012}" 2>/dev/null || true
fi

sleep 2

# -------------------------------------------------------------------
# Optional sanity logging
# -------------------------------------------------------------------
echo "=================================================="
echo "RENDEREXPO AI STUDIO startup"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "SD35_MODEL_PATH=${SD35_MODEL_PATH}"
echo "SD35_CONTROLNET_CANNY_PATH=${SD35_CONTROLNET_CANNY_PATH}"
echo "SD35_CONTROLNET_DEPTH_PATH=${SD35_CONTROLNET_DEPTH_PATH}"
echo "SD35_DEPTH_MODEL_ID=${SD35_DEPTH_MODEL_ID}"
echo "SD35_DEVICE=${SD35_DEVICE}"
echo "SD35_RUNTIME_MODE=${SD35_RUNTIME_MODE}"
echo "RUN_REAL_SD35=${RUN_REAL_SD35}"
echo "PRELOAD_SD35_ON_STARTUP=${PRELOAD_SD35_ON_STARTUP}"
echo "SD35_ENABLE_CPU_OFFLOAD=${SD35_ENABLE_CPU_OFFLOAD}"
echo "OUTPUTS_ROOT=${OUTPUTS_ROOT}"
echo "RENDEREXPO_GPU_DISPATCH_URL=${RENDEREXPO_GPU_DISPATCH_URL}"
echo "=================================================="

# -------------------------------------------------------------------
# Optional preload
# IMPORTANT:
# - Off by default
# - Only runs if explicitly enabled
# -------------------------------------------------------------------
if [ "${PRELOAD_SD35_ON_STARTUP}" = "1" ]; then
  echo "Preloading SD3.5 runtime because PRELOAD_SD35_ON_STARTUP=1"
  python3 - <<'PY'
import os
from runtime.sd35_runtime import SD35Runtime

device = os.getenv("SD35_DEVICE", "cuda")
run_real = os.getenv("RUN_REAL_SD35", "1").strip().lower() in ("1", "true", "yes", "on")

if not run_real:
    raise RuntimeError("PRELOAD_SD35_ON_STARTUP=1 but RUN_REAL_SD35 is disabled.")

rt = SD35Runtime(mode="real", device=device)
rt.load()
print("SD35 preload completed. loaded =", rt.is_loaded)
PY
fi

# -------------------------------------------------------------------
# Start services
# -------------------------------------------------------------------

# 10) Start GPU worker on 8002 (background)
uvicorn app.gpu_entry:app \
  --host 0.0.0.0 \
  --port 8002 \
  --log-level info \
  > "${LOG_DIR}/gpu_8002.log" 2>&1 &

GPU_PID=$!
echo "GPU worker started on :8002 (pid=${GPU_PID})"

# 11) Wait briefly and fail early if GPU worker crashes immediately
sleep 3
if ! kill -0 "${GPU_PID}" 2>/dev/null; then
  echo "FATAL: GPU worker failed to stay up. Check ${LOG_DIR}/gpu_8002.log" >&2
  exit 1
fi

# 12) Start planner on 8012 (foreground)
echo "Planner starting on :8012"
exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8012 \
  --log-level info \
  > "${LOG_DIR}/planner_8012.log" 2>&1