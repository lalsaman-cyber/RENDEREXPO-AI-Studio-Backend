#!/usr/bin/env bash

# --- HARD-LOCK: load & export HMAC secret for planner ---
if [ -z "${RENDEREXPO_HMAC_SECRET:-}" ]; then
  if [ -f "secrets/hmac_secret.txt" ]; then
    export RENDEREXPO_HMAC_SECRET="$(cat secrets/hmac_secret.txt)"
  fi
fi
if [ -z "${RENDEREXPO_HMAC_SECRET:-}" ]; then
  echo "FATAL: RENDEREXPO_HMAC_SECRET is not set and secrets/hmac_secret.txt not found." >&2
  exit 1
fi
# sanity: length check (>=32)
if [ "${#RENDEREXPO_HMAC_SECRET}" -lt 32 ]; then
  echo "FATAL: RENDEREXPO_HMAC_SECRET too short (${#RENDEREXPO_HMAC_SECRET})." >&2
  exit 1
fi
# --- END HMAC block ---

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

# kill existing 8002
PID8002=$(ss -ltnp 2>/dev/null | awk '/:8002/ {print $NF}' | sed -E 's/.*pid=([0-9]+).*/\1/' | head -n1 || true)
[ -n "${PID8002:-}" ] && kill -TERM "$PID8002" 2>/dev/null || true
sleep 2

export RENDEREXPO_HMAC_SECRET="$(cat secrets/hmac_secret.txt)"

set -a
source secrets/api.env
set +a

nohup env \
  RENDEREXPO_HMAC_SECRET="$RENDEREXPO_HMAC_SECRET" \
  GPU_BASE_URL="$GPU_BASE_URL" \
  ./.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8002 --log-level info \
  > logs/api_8002.log 2>&1 &

API_PID=$!
echo "API_PID=$API_PID"

for i in $(seq 1 60); do
  if curl -sS --max-time 1 http://127.0.0.1:8002/api/health >/dev/null 2>&1; then
    echo "API_OK"
    exit 0
  fi
  sleep 1
done

echo "API_NOT_READY"
tail -n 200 logs/api_8002.log || true
exit 1
