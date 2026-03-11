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

# 5) Canonical Python path + model path
export PYTHONPATH="${PROJECT_ROOT}"
export SD35_MODEL_PATH="${SD35_MODEL_PATH:-/workspace-data/models/sd35-large}"

# 6) Boot LIGHT by default
export SD35_RUNTIME_MODE="${SD35_RUNTIME_MODE:-lazy}"
export RUN_REAL_SD35="${RUN_REAL_SD35:-0}"
export PRELOAD_SD35_ON_STARTUP="${PRELOAD_SD35_ON_STARTUP:-0}"

# 7) Optional HMAC secret load
if [ -z "${RENDEREXPO_HMAC_SECRET:-}" ] && [ -f "secrets/hmac_secret.txt" ]; then
  export RENDEREXPO_HMAC_SECRET="$(cat secrets/hmac_secret.txt)"
fi

# 8) Logs folder
mkdir -p "${LOG_DIR}"

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

# 11) Start planner on 8012 (foreground)
echo "Planner starting on :8012"
exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8012 \
  --log-level info \
  > "${LOG_DIR}/planner_8012.log" 2>&1