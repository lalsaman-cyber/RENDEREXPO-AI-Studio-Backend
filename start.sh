cat > /workspace-data/RENDEREXPO-AI-Studio-Backend/start.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------------
# RENDEREXPO AI STUDIO – START SCRIPT
# -------------------------------------------------------------------
# Canonical service mapping:
#   Planner    -> 8012
#   GPU worker -> 8002
#   ComfyUI    -> 8188
#   Jupyter    -> 8888 (started separately if needed)
#
# Canonical project root:
#   /workspace-data/RENDEREXPO-AI-Studio-Backend
# -------------------------------------------------------------------

PROJECT_ROOT="/workspace-data/RENDEREXPO-AI-Studio-Backend"
LOG_DIR="${PROJECT_ROOT}/logs"
COMFY_ROOT="/workspace-data/ComfyUI_app"
COMFY_LOG_DIR="${COMFY_ROOT}/logs"
COMFY_DB_URL="sqlite:////workspace-data/ComfyUI_app/runtime_db/comfyui.db"

COMFY_PID=""
GPU_PID=""
PLANNER_PID=""

cleanup() {
  echo "Shutdown requested. Stopping services..."
  for pid in "${PLANNER_PID:-}" "${GPU_PID:-}" "${COMFY_PID:-}"; do
    if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  wait || true
}
trap cleanup EXIT INT TERM

# 1) Hard-stop if canonical project root is missing
if [ ! -d "${PROJECT_ROOT}" ]; then
  echo "FATAL: canonical project root not found: ${PROJECT_ROOT}" >&2
  exit 1
fi

# 2) Enter canonical repo only
cd "${PROJECT_ROOT}" || exit 1

# 3) Hard-stop if backend venv is missing
if [ ! -f ".venv/bin/activate" ]; then
  echo "FATAL: virtualenv activate script not found at ${PROJECT_ROOT}/.venv/bin/activate" >&2
  exit 1
fi

# 4) Activate backend venv
source .venv/bin/activate

# 5) Canonical Python path for backend
export PYTHONPATH="${PROJECT_ROOT}"

# -------------------------------------------------------------------
# Canonical model/env paths
# -------------------------------------------------------------------

export SD35_MODEL_PATH="${SD35_MODEL_PATH:-/workspace-data/models/sd35-large}"
export SD35_CONTROLNET_CANNY_PATH="${SD35_CONTROLNET_CANNY_PATH:-/workspace-data/models/sd35-controlnet-canny}"
export SD35_CONTROLNET_DEPTH_PATH="${SD35_CONTROLNET_DEPTH_PATH:-/workspace-data/models/sd35-controlnet-depth}"
export SD35_DEPTH_MODEL_ID="${SD35_DEPTH_MODEL_ID:-Intel/dpt-hybrid-midas}"

export SD35_DEVICE="${SD35_DEVICE:-cuda}"
export SD35_RUNTIME_MODE="${SD35_RUNTIME_MODE:-lazy}"
export RUN_REAL_SD35="${RUN_REAL_SD35:-1}"
export PRELOAD_SD35_ON_STARTUP="${PRELOAD_SD35_ON_STARTUP:-0}"
export SD35_ENABLE_CPU_OFFLOAD="${SD35_ENABLE_CPU_OFFLOAD:-0}"

export OUTPUTS_ROOT="${OUTPUTS_ROOT:-/workspace-data/outputs}"
export RENDEREXPO_OUTPUTS_MOUNT="${RENDEREXPO_OUTPUTS_MOUNT:-/outputs}"
export RENDEREXPO_GPU_DISPATCH_URL="${RENDEREXPO_GPU_DISPATCH_URL:-http://127.0.0.1:8002/api/gpu/dispatch}"
export RENDEREXPO_GPU_TIMEOUT_SECONDS="${RENDEREXPO_GPU_TIMEOUT_SECONDS:-180}"
export RENDEREXPO_GPU_POLL_SECONDS="${RENDEREXPO_GPU_POLL_SECONDS:-0.5}"

export GPU_WORKER_NAME="${GPU_WORKER_NAME:-runpod-gpu-worker}"
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"

# 6) Optional HMAC secret load
if [ -z "${RENDEREXPO_HMAC_SECRET:-}" ] && [ -f "secrets/hmac_secret.txt" ]; then
  export RENDEREXPO_HMAC_SECRET="$(cat secrets/hmac_secret.txt)"
fi

# 7) Logs / outputs
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUTS_ROOT}"
mkdir -p "${COMFY_LOG_DIR}"
mkdir -p "${COMFY_ROOT}/runtime_db"

# 8) Safety cleanup: stop old listeners on canonical ports
for PORT in 8002 8012 8188; do
  PID="$(ss -ltnp 2>/dev/null | awk "/:${PORT}/ {print \$NF}" | sed -E 's/.*pid=([0-9]+).*/\1/' | head -n1 || true)"
  if [ -n "${PID:-}" ]; then
    kill -TERM "${PID}" 2>/dev/null || true
  fi
done
sleep 3

# -------------------------------------------------------------------
# Optional sanity logging
# -------------------------------------------------------------------
echo "=================================================="
echo "RENDEREXPO AI STUDIO startup"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "COMFY_ROOT=${COMFY_ROOT}"
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
echo "COMFY_DB_URL=${COMFY_DB_URL}"
echo "=================================================="

# -------------------------------------------------------------------
# Optional preload
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
# Start ComfyUI on 8188 FIRST, then wait until healthy
# -------------------------------------------------------------------
(
  cd "${COMFY_ROOT}" || exit 1
  source .venv/bin/activate
  exec python -u main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --database-url "${COMFY_DB_URL}" \
    --verbose DEBUG
) > "${COMFY_LOG_DIR}/comfyui_8188.log" 2>&1 &

COMFY_PID=$!
echo "ComfyUI started on :8188 (pid=${COMFY_PID})"

for i in $(seq 1 180); do
  if curl -sf http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
    echo "ComfyUI healthy on :8188"
    break
  fi
  if ! kill -0 "${COMFY_PID}" 2>/dev/null; then
    echo "FATAL: ComfyUI died during startup. Check ${COMFY_LOG_DIR}/comfyui_8188.log" >&2
    tail -n 200 "${COMFY_LOG_DIR}/comfyui_8188.log" >&2 || true
    exit 1
  fi
  sleep 1
done

if ! curl -sf http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
  echo "FATAL: ComfyUI did not become healthy on :8188 in time." >&2
  tail -n 200 "${COMFY_LOG_DIR}/comfyui_8188.log" >&2 || true
  exit 1
fi

# -------------------------------------------------------------------
# Start GPU worker on 8002
# -------------------------------------------------------------------
uvicorn app.gpu_entry:app \
  --host 0.0.0.0 \
  --port 8002 \
  --log-level info \
  > "${LOG_DIR}/gpu_8002.log" 2>&1 &

GPU_PID=$!
echo "GPU worker started on :8002 (pid=${GPU_PID})"

for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8002/api/gpu/health >/dev/null 2>&1; then
    echo "GPU worker healthy on :8002"
    break
  fi
  if ! kill -0 "${GPU_PID}" 2>/dev/null; then
    echo "FATAL: GPU worker died during startup. Check ${LOG_DIR}/gpu_8002.log" >&2
    tail -n 200 "${LOG_DIR}/gpu_8002.log" >&2 || true
    exit 1
  fi
  sleep 1
done

if ! curl -sf http://127.0.0.1:8002/api/gpu/health >/dev/null 2>&1; then
  echo "FATAL: GPU worker did not become healthy on :8002 in time." >&2
  tail -n 200 "${LOG_DIR}/gpu_8002.log" >&2 || true
  exit 1
fi

# -------------------------------------------------------------------
# Start planner on 8012
# -------------------------------------------------------------------
uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8012 \
  --log-level info \
  > "${LOG_DIR}/planner_8012.log" 2>&1 &

PLANNER_PID=$!
echo "Planner started on :8012 (pid=${PLANNER_PID})"

for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8012/api/health >/dev/null 2>&1; then
    echo "Planner healthy on :8012"
    break
  fi
  if ! kill -0 "${PLANNER_PID}" 2>/dev/null; then
    echo "FATAL: Planner died during startup. Check ${LOG_DIR}/planner_8012.log" >&2
    tail -n 200 "${LOG_DIR}/planner_8012.log" >&2 || true
    exit 1
  fi
  sleep 1
done

if ! curl -sf http://127.0.0.1:8012/api/health >/dev/null 2>&1; then
  echo "FATAL: Planner did not become healthy on :8012 in time." >&2
  tail -n 200 "${LOG_DIR}/planner_8012.log" >&2 || true
  exit 1
fi

echo "All services are healthy: 8188 / 8002 / 8012"

# Keep container alive and fail if any critical service exits
wait -n "${COMFY_PID}" "${GPU_PID}" "${PLANNER_PID}"
echo "A critical service exited. Shutting down the others." >&2
exit 1
EOF