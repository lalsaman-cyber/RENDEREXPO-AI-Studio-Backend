# SDXL → SD3.5 Migration Plan (RENDEREXPO AI STUDIO)

_Last updated: 2025-11-22_

This document tracks:

- All features that existed (or were planned) in the SDXL phase  
- How they are re-implemented or adapted for **Stable Diffusion 3.5 Large (SD3.5)**  
- All new SD3.5-only features (VR, product insertion, moodboards, sketch engine, floorplans, etc.)  
- The plan for **RENDEREXPO-ULTRA** (our trained SD3.5 model + LoRAs + refiners)

Nothing from the SDXL era (mesh, CAD, floor plans, surface editing, video, etc.) is allowed to be forgotten.  
This file is the **source of truth** for the migration and future features.

> IMPORTANT: **SDXL is now LEGACY ONLY.**  
> - No more SDXL inference in production.  
> - All future work is **SD3.5-only**.

---

## 1. Current SD3.5 Backend State

- ✅ SD3.5 Large downloaded to:  
  `models/sd35-large`

- ✅ HF token configured via `HF_TOKEN` env var

- ✅ Licenses stored under `/Licenses/`

- ✅ `docs/Model-Licensing-Summary.md` created

- ✅ `config/model_paths.yaml` with:
  - `sd35_large_dir: "models/sd35-large"`

- ✅ `app/sd35_loader.py`  
  - Verifies SD3.5 files exist and are readable.

- ✅ `app/sd35_runtime.py`  
  - Runtime skeleton; **NO heavy model load on CPU yet**.
  - Designed to be used on GPU (RunPod) only.

- ✅ `app/main.py` with routes (local dev only):
  - `GET /` → health message
  - `GET /api/health` → backend alive
  - `GET /api/sd35/files` → lists SD3.5 files for sanity checks

- ✅ FastAPI + Uvicorn are running locally at:  
  `http://127.0.0.1:8000`

**Important rules:**

- Local dev (laptop) **must not** load SD3.5 into CPU RAM or run heavy inference.  
- All heavy tasks (SD3.5, ControlNet, ESRGAN, MiDaS, VR baking, etc.) are for **RunPod GPU** only.

---

## 2. SDXL-Era Modules to Carry Over

These modules existed for SDXL and MUST be carried into the SD3.5 world by adapting or reusing their ideas:

From previous SDXL phase:

- `__init__`
- `architecture_scene`
- `cad_from_image`
- `controlnet_depth.py`
- `floorplan`
- `materials`
- `mesh_from_image`
- `midas_depth`
- `pipeline_manager`
- `presets`
- `sd3_img2img`
- `sd3_text2img`
- `sd35_pipelines`
- `sdxl_img2img`
- `sdxl_pipelines`
- `sdxl_text2img`
- `selective_edit`
- `space_capture_stub`
- `upscale_esrgan`
- `video_from_image`

> SDXL files are **legacy**.  
> We **do not** use SDXL pipelines in production anymore, but we keep the files and ideas as reference.

### 2.1 Mapping SDXL → SD3.5

Conceptually:

- **SDXL-specific pipes:**
  - `sdxl_text2img` / `sdxl_img2img` / `sdxl_pipelines`  
  → **Replaced by** SD3.5 versions under `sd35_*` modules and `sd35_runtime`.

- **Shared high-level tools:**
  - `controlnet_depth.py`, `midas_depth`, `upscale_esrgan`, `video_from_image`,  
    `mesh_from_image`, `cad_from_image`, `floorplan`, `architecture_scene`,  
    `materials`, `presets`, `selective_edit`, `space_capture_stub`  
  → **Kept and adapted** to SD3.5’s architecture + new runtime.

We are **not** throwing away any concepts.  
We only change what lives underneath (SDXL → SD3.5).

---

## 3. Core SD3.5 Endpoints (Phase 1)

These are the **first** SD3.5 features to wire into the API before advanced CAD/mesh/floor-plan/VR features.

### 3.1 Text-to-Image (Primary Endpoint)

Planned endpoint:

- `POST /api/sd35/render`

Request JSON (draft):

- `prompt` (string, required)
- `negative_prompt` (string, optional)
- `width` / `height` (optional; default safe values, e.g. 1024×1024)
- `seed` (optional; `null` for random)
- `num_inference_steps` (optional; default ~20–30)
- `guidance_scale` (optional; default ~5–7)
- Future:
  - `style_preset` (link to RENDEREXPO style presets)
  - `material_preset`
  - `lighting_preset`

Response JSON (draft):

- `job_id`
- `image_path` or `image_url`  
  (e.g. `outputs/2025-11-22/<JOBID>/final.png`)
- `meta`:
  - `prompt`
  - `seed`
  - `model: "sd3.5-large"` (or `"renderexpo-ultra"`)
  - `inference_time_ms` (optional)

### 3.2 Image-to-Image

Planned endpoint:

- `POST /api/sd35/render-from-image`

Features:

- Input: uploaded sketch / clay / b&w / photo / base render
- Optional:
  - `strength` (how strongly SD3.5 overrides the image)
  - ControlNet (see Section 4)

Request (multipart):

- `image` file
- `prompt`, `negative_prompt`
- `strength`
- `control_type` (optional)
- `control_strength` (optional)

Response:

- Same pattern as `/api/sd35/render` (job_id, image_path, meta).

---

## 4. ControlNet Integration (Phase 1.5)

ControlNet models to support (SD15-based or SD3.x-compatible variants):

- `lllyasviel/control_v11p_sd15_canny`
- `lllyasviel/control_v11f1p_sd15_depth`
- `lllyasviel/control_v11p_sd15_lineart`
- (Later) layout / sketch models if needed

### 4.1 Control Types

In request JSON:

- `control_type`:
  - `"canny"`
  - `"depth"`
  - `"lineart"`
  - `null` (no control)

- `control_strength` (0.0–1.5 typical)

Possible future fields:

- `control_guidance_start`
- `control_guidance_end`

### 4.2 Use Cases

- Floor plans → lineart/canny → SD3.5 render
- Sketch → Canny → detailed interior
- Line-art product → final product render
- Depth → geometry-preserving transforms
- Layout → consistent room arrangements

### 4.3 Licensing Reminder

ControlNet uses **CreativeML OpenRAIL-M**, which requires:

- We keep use-based restrictions in our Terms of Use.
- We avoid prohibited outputs (harm, hate, minors, etc.).
- We **propagate** restrictions downstream to users.

We already store license texts in:

- `Licenses/ControlNet-Canny-LICENSE.txt`
- `Licenses/ControlNet-Depth-LICENSE.txt`
- `Licenses/ControlNet-Lineart-LICENSE.txt`

---

## 5. Upscaling & Depth (Phase 2 Core)

### 5.1 Upscaling: Real-ESRGAN

Endpoint (planned):

- `POST /api/upscale`

Request:

- `image` file (or existing image path)
- `scale` (2, 4; maybe 8 with tiling in future)
- `mode`:
  - `"realesrgan"` (default)
  - `"latent_sd35"` (future, if we add latent upscaling)

Response:

- `upscaled_image_path`
- `original_image_path`
- `scale`

### 5.2 Depth Maps: MiDaS

Endpoint (planned):

- `POST /api/depth-map`

Request:

- `image` file or existing image path

Response:

- `depth_map_path` (grayscale PNG)
- Optional: normalized 16-bit map (later)

Use cases:

- Parallax / video-from-image
- Mesh-from-image
- CAD-from-image assistance
- Geometry-aware edits
- VR reconstruction

---

## 6. Job / File Management & Safety

### 6.1 File Scheme

Standard output structure (draft):

- `outputs/{YYYY-MM-DD}/{job_id}/`
  - `input.png`       (optional)
  - `final.png`       (main output)
  - `depth.png`       (if generated)
  - `upscaled.png`    (if requested)
  - `meta.json`       (prompt, settings, etc.)
  - `video.mp4`       (if a video job)
  - `control_input.png` (if ControlNet conditioning used)

### 6.2 Basic Job Info

`meta.json` fields:

- `job_id`
- `created_at`
- `type`: `"text2img" | "img2img" | "upscale" | "depth" | "video" | "vr" | ...`
- `model_name`: `"sd3.5-large"` or `"renderexpo-ultra"`
- `control_type`
- `seed`
- `width`, `height`
- `steps`
- `guidance_scale`
- `style_preset`
- `material_preset`
- `lighting_preset`

Future (optional) endpoint:

- `GET /api/jobs/{job_id}` → returns `meta.json`.

### 6.3 Safety & AUP

Before any SD3.5 generation:

- Run prompt through a basic safety check:
  - Block content that violates:
    - Stability AI AUP
    - OpenRAIL-M restricted uses
- Reject or warn if:
  - Sexual content with minors
  - Non-consensual explicit imagery
  - Harassment / hate / terrorism
  - Graphic violence / gore
  - Predictive policing, biometric classification, etc.

We will later add more advanced checks, but this first layer is mandatory.

---

## 7. Local vs GPU Behavior

### 7.1 Local (Your Laptop)

Local is only for:

- `/`
- `/api/health`
- `/api/sd35/files`
- Simple configs and docs
- Possibly listing presets and non-heavy info

**Do NOT** on local:

- Load SD3.5 into CPU RAM
- Run SD3.5 inference
- Run ControlNet, ESRGAN, MiDaS, or VR reconstruction

### 7.2 GPU Runtime (RunPod)

A GPU entrypoint (e.g. `app/gpu_entry.py`) on RunPod will:

- Create `SD35Runtime(device="cuda")`
- Call `load()` on startup to load:
  - SD3.5 transformer
  - VAE
  - Text encoders
  - ControlNet (later)
  - ESRGAN (later)
  - MiDaS (later)

- Expose heavy endpoints:
  - `/api/sd35/render`
  - `/api/sd35/render-from-image`
  - `/api/upscale`
  - `/api/depth-map`
  - `/api/video-from-image`
  - `/api/vr/reconstruct`
  - `/api/insert-object`
  - `/api/moodboard/generate`
  - `/api/sketch/realtime`
  - `/api/floorplan/*`

---

## 8. Advanced Architecture & CAD Features (Planned, NOT Dropped)

These are higher-level features important for RENDEREXPO that build on top of SD3.5, ControlNet, MiDaS, ESRGAN, and our own LoRAs/refiners.

### 8.1 Floor Plans → 3D Views

Modules:

- `floorplan/`
- `architecture_scene/`

Planned capabilities:

- Upload floor plans (raster or vector).
- Detect:
  - Walls
  - Openings
  - Doors
  - Windows
  - Room layout & types (kitchen, bath, living, bedroom, etc.).
- Allow drawing floor plans directly on the platform:
  - Wall thickness
  - Room labels
  - Doors and windows
- Define **virtual cameras** inside the floor plan:
  - Camera position (x, y in plan)
  - Orientation (angle)
  - Approximate FOV / focal length
- Generate interior views from those cameras using:
  - SD3.5 text2img/img2img
  - ControlNet lineart / canny / depth
- Store per-camera:
  - `camera_id`
  - `room_type`
  - `render_paths`
  so Wix can show “click a room” → “see that room rendered”.

### 8.2 Mesh Functions (OBJ, DXF, etc.)

Module:

- `mesh_from_image/` (and future CAD/mesh modules)

Planned:

- Upload 3D files:
  - OBJ, FBX, GLB, DXF (exact formats TBD).
- Convert geometry to:
  - Depth maps
  - Normal maps
  - Clay / contour passes
- Feed those into SD3.5:
  - For photoreal renders that respect original geometry.
- Export mapping:
  - Keep consistent camera + mapping for re-renders and alternative materials.

### 8.3 Surface Editing / Local Editing

Modules:

- `selective_edit/`
- `materials/`

Planned:

- Select specific surfaces:
  - Floors, walls, ceilings, facades, furniture
- Methods:
  - Mask-based selection
  - Possible CAD-driven selection later
- Apply changes only to that region:
  - Material replacement
  - Color changes
  - Texture changes
  - Lighting tweaks
- Use SD3.5 with:
  - img2img + masks
  - (Later) ControlNet normal/segmentation

### 8.4 CAD From Image (Reverse)

Module:

- `cad_from_image/`

Planned:

- From image → interpret architectural lines:
  - Edges
  - Vanishing lines
  - Wall layout
- Output:
  - DXF / SVG
  - Simple JSON layout
- This is a **research** feature for later phases, not Phase 1.

### 8.5 Materials & Presets System

Modules:

- `materials/`
- `presets/`

Planned:

- Material presets:
  - `white_plaster_modern`
  - `polished_concrete_floor`
  - `warm_oak_veneer`
  - `brushed_brass`
  - `clear_glass`
  - `smoked_glass`
  - etc.
- Style presets:
  - `Scandinavian_minimal`
  - `Brutalist`
  - `Soft_luxury`
  - `Industrial_loft`
  - `Mediterranean`
  - `Farmhouse_modern`
- Lighting presets:
  - `daylight_soft`
  - `evening_warm`
  - `studio_neutral`
- Wire presets into core endpoints:
  - `/api/sd35/render`
  - `/api/sd35/render-from-image`
  using:
  - `style_preset`
  - `material_preset`
  - `lighting_preset`

Wix UI will show user-friendly names; backend uses stable IDs.

### 8.6 Video From Image (Parallax / Camera)

Module:

- `video_from_image/`

Planned:

- Use MiDaS depth maps → 2.5D parallax animations.
- Camera motions:
  - Orbit
  - Push-in / pull-out
  - Left-right slides
- Endpoint:
  - `POST /api/video-from-image`
- Inputs:
  - `image` (existing render)
  - Optional `depth_map` (or auto-compute)
  - `motion_preset` (`"orbit"`, `"slide"`, `"zoom"`)
  - Duration, FPS
- Output:
  - MP4
  - Or sequence of PNG frames

---

## 9. Directory Structure (Planned Target)

After migration stabilizes, target structure:

```text
Backend/
│
├── app/
│   ├── main.py
│   ├── gpu_entry.py
│   ├── routers/
│   │   ├── text2img.py
│   │   ├── img2img.py
│   │   ├── upscale.py
│   │   ├── controlnet.py
│   │   ├── depth.py
│   │   ├── floorplan.py
│   │   ├── materials.py
│   │   ├── mesh.py
│   │   ├── cad.py
│   │   ├── video.py
│   │   ├── vr.py
│   │   ├── moodboard.py
│   │   └── sketch.py
│
├── runtime/
│   ├── sd35_runtime.py
│   ├── controlnet_runtime.py
│   ├── midas_runtime.py
│   ├── esrgan_runtime.py
│   └── pipeline_manager.py
│
├── modules/
│   ├── sd35_text2img/
│   ├── sd35_img2img/
│   ├── sd35_pipelines/
│   ├── controlnet_depth/
│   ├── midas_depth/
│   ├── floorplan/
│   ├── materials/
│   ├── mesh_from_image/
│   ├── cad_from_image/
│   ├── video_from_image/
│   ├── vr_reconstruct/
│   ├── moodboard/
│   ├── sketch_realtime/
│   ├── presets/
│   └── architecture_scene/
│
├── models/
│   ├── sd35-large/
│   └── renderexpo-ultra/        # future trained SD3.5 finetune + LoRAs
│
├── config/
│   └── model_paths.yaml
│
└── docs/
    ├── Model-Licensing-Summary.md
    ├── SDXL_to_SD35_Migration_Plan.md
    └── System-Architecture-Overview.md
