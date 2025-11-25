#!/usr/bin/env bash
set -e

echo "==============================================="
echo " Starting RENDEREXPO AI STUDIO GPU API (SD3.5)"
echo "==============================================="

# Optional: print basic torch / CUDA info
python3 - << 'EOF'
try:
    import torch
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
except Exception as e:
    print("Torch/CUDA check failed:", e)
EOF

# Start FastAPI app (GPU-only entrypoint)
# This will use app/gpu_entry.py and SD35Runtime inside runtime/sd35_runtime.py
uvicorn app.gpu_entry:app --host 0.0.0.0 --port 8000
