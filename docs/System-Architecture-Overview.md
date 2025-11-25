\# RENDEREXPO AI STUDIO — System Architecture Overview



\_Last updated: 2025-11-22\_



This document gives a \*\*big-picture map\*\* of how RENDEREXPO AI STUDIO is built:



\- What parts exist now

\- What parts will exist later

\- How SD3.5 and RENDEREXPO-ULTRA fit in

\- What runs on your laptop vs. what runs on GPU (RunPod)

\- What future “wow” features plug in where



It works together with:



\- `docs/SDXL\_to\_SD35\_Migration\_Plan.md`

\- `docs/Model-Licensing-Summary.md`



---



\## 1. High-Level Goals



RENDEREXPO AI STUDIO is:



\- A \*\*backend + web UI\*\* for architects and developers.

\- Built on \*\*Stable Diffusion 3.5 Large (SD3.5)\*\* and your future finetune \*\*RENDEREXPO-ULTRA\*\*.

\- Focused on:

&nbsp; - Architectural, interior, and real-estate visualization.

&nbsp; - Fast iteration (like Nano Banana / Krea style).

&nbsp; - Advanced tools:

&nbsp;   - Floor plans

&nbsp;   - VR scenes

&nbsp;   - Product insertion

&nbsp;   - Moodboards

&nbsp;   - Real-time sketching



The system is split into:



1\. \*\*Local Dev (Laptop)\*\* → Light FastAPI app, no heavy AI.

2\. \*\*GPU Runtime (RunPod)\*\* → SD3.5, LoRAs, ControlNet, ESRGAN, MiDaS.

3\. \*\*Future Wix UI\*\* → Frontend for all tools (later phase).



---



\## 2. Environments: Local vs GPU



\### 2.1 Local Environment (Your Laptop)



Purpose:



\- Write code.

\- Organize configs and docs.

\- Check that SD3.5 files exist.

\- Design endpoints (without actually running AI).



Key rules:



\- ❌ Do NOT load SD3.5 on CPU.

\- ❌ Do NOT run heavy inference.

\- ✅ Only run health + file-check endpoints.



Main pieces:



\- `app/main.py`

&nbsp; - `GET /` — hello message.

&nbsp; - `GET /api/health` — simple health check.

&nbsp; - `GET /api/sd35/files` — list SD3.5 directory contents.



\- `app/sd35\_loader.py`

&nbsp; - Reads `config/model\_paths.yaml`.

&nbsp; - Confirms `models/sd35-large` exists and is complete.



\- `docs/` + `Licenses/`

&nbsp; - All your legal and design docs live here.



\### 2.2 GPU Environment (RunPod)



Purpose:



\- Load SD3.5 / RENDEREXPO-ULTRA on CUDA.

\- Run all heavy endpoints.

\- Serve images and VR content to the Wix frontend.



Key rules:



\- ✅ All SD3.5, ControlNet, ESRGAN, MiDaS, VR, etc. run \*\*only here\*\*.

\- ✅ GPU pods are only started when code is ready (to control cost).

\- ❌ No experimenting directly on your laptop CPU.



Main pieces (planned):



\- `app/gpu\_entry.py`

&nbsp; - FastAPI app for GPU-only endpoints.

&nbsp; - Uses `SD35Runtime` to do actual work.



\- `runtime/sd35\_runtime.py`

&nbsp; - Loads SD3.5 / RENDEREXPO-ULTRA.

&nbsp; - Handles text2img, img2img, and more.

&nbsp; - Knows where to save outputs.



\- `runtime/controlnet\_runtime.py`

\- `runtime/midas\_runtime.py`

\- `runtime/esrgan\_runtime.py`

\- `runtime/pipeline\_manager.py`



\- `models/`

&nbsp; - `sd35-large/` (base model)

&nbsp; - `renderexpo-ultra/` (future trained model, LoRAs, refiners)



---



\## 3. Backend Folder Overview



Target structure (simplified):



```text

Backend/

│

├── app/

│   ├── main.py           # Local dev FastAPI

│   ├── gpu\_entry.py      # GPU FastAPI (RunPod)

│   ├── routers/          # Endpoints grouped by feature (future)

│

├── runtime/

│   ├── sd35\_runtime.py

│   ├── controlnet\_runtime.py

│   ├── midas\_runtime.py

│   ├── esrgan\_runtime.py

│   └── pipeline\_manager.py

│

├── modules/

│   ├── sd35\_text2img/

│   ├── sd35\_img2img/

│   ├── sd35\_pipelines/

│   ├── controlnet\_depth/

│   ├── midas\_depth/

│   ├── floorplan/

│   ├── materials/

│   ├── mesh\_from\_image/

│   ├── cad\_from\_image/

│   ├── video\_from\_image/

│   ├── vr\_reconstruct/

│   ├── moodboard/

│   ├── sketch\_realtime/

│   ├── presets/

│   └── architecture\_scene/

│

├── models/

│   ├── sd35-large/

│   └── renderexpo-ultra/        # future

│

├── config/

│   └── model\_paths.yaml

│

├── docs/

│   ├── Model-Licensing-Summary.md

│   ├── SDXL\_to\_SD35\_Migration\_Plan.md

│   └── System-Architecture-Overview.md

│

├── Licenses/

│

└── outputs/



