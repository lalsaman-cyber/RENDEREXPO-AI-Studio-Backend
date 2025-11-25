# RENDEREXPO AI STUDIO – Licensing & Model Usage Notes

_Last updated: 2025-11-22_

This document records how RENDEREXPO AI STUDIO uses third-party AI models and licenses, with a focus on **Stable Diffusion 3.5 (SD3.5)** as the core rendering engine.

It is intended to support:

- Legal review  
- IP filing  
- Future audits (licensing, compliance, revenue thresholds)  

Primary development provenance document for this system (full ChatGPT engineering log):

- `/mnt/data/RENDEREXPO AI STUDO - CHATGPT Chat.docx`


---

## 1. Business Context

- Product: **RENDEREXPO AI STUDIO**  
- Domain: Architectural visualization, real estate, interior/exterior design.  
- Core engine: **Stable Diffusion 3.5 Large** (text-to-image + image-to-image).  
- Revenue status: **Intended to operate under < USD $1M annual revenue** for the foreseeable future.  

- Ownership: RENDEREXPO (LAS GROUP LLC, DBA RENDEREXPO) owns:  
  - All custom code in this repository.  
  - All custom datasets (the ~1000+ renderings created by the founder).  
  - All custom LoRAs and finetuned SD3.5 derivatives created specifically for RENDEREXPO.

Third-party model providers (Stability AI, Hugging Face, etc.) retain ownership of the underlying base models and their respective IP.


---

## 2. Stable Diffusion 3.5 – License & Model

### 2.1 Model Source

- Provider: **Stability AI**  
- Primary model: **Stable Diffusion 3.5 Large**  
- Hugging Face model repo:  
  - `stabilityai/stable-diffusion-3.5-large`  
- License file location (in this project):  
  - `Licenses/SD3.5-LICENSE.txt` (copied from the model’s LICENSE content on Hugging Face at the time of first download)

The SD3.5 model is provided under the **Stability AI Community License**.


### 2.2 Stability AI Community License

Stability AI provides a **Community License** that covers, among other things, the **Stable Diffusion 3.5 Suite**, and is intended for individuals, researchers, developers, and small businesses with **less than USD $1M annual revenue**.

Key points (high-level summary, not legal advice):

- The Community License allows:
  - Research and non-commercial use.  
  - Many commercial uses **as long as yearly revenue is < USD $1M** (for you + your affiliates, in aggregate).  
- If annual revenue (regardless of whether it is directly or indirectly related to the Stability AI Materials) reaches or exceeds **USD $1M/year**, an **Enterprise license** must be obtained from Stability AI and the Community License rights terminate.

RENDEREXPO AI STUDIO is designed and operated under the Community License conditions:

- Target revenue: < USD $1M/year (at least initially).  
- RENDEREXPO agrees to Stability’s Community License terms as part of operating SD3.5 for commercial visualization work.

**Action recorded:**

- Stability Community License accepted by: **Louai Alsaman / LAS GROUP LLC, DBA RENDEREXPO**  
- Date accepted: **2025-11-22**  
- URL at time of acceptance: `https://stability.ai/license`  


### 2.3 Model-Specific License (SD3.5)

In addition to the general Community License, each SD3.5 model on Hugging Face includes a specific LICENSE file (for example: `LICENSE.md` in `stabilityai/stable-diffusion-3.5-large`).

This repository stores a copy of that model-specific license here:

- `Licenses/SD3.5-LICENSE.txt`  
  (Created by copying the full text of the LICENSE from the Hugging Face SD3.5 Large repo at the time of download.)

**Action recorded:**

- SD3.5 Large model license accepted by HF user: **`lalsaman`**  
- Date accepted: **2025-11-22**  
- Hugging Face URL at time of acceptance:  
  - `https://huggingface.co/stabilityai/stable-diffusion-3.5-large`  


### 2.4 Mandatory Attribution & Notices (Stability AI)

Under the Stability AI Community License, when distributing or making available products or services that use the Stability AI Materials (including SD3.5 and derivatives), RENDEREXPO must:

1. **Provide a copy of the Stability AI Community License** to any third party receiving the Stability AI Materials or Derivative Works.  
2. **Retain the following attribution notice** within a “Notice” text file distributed as part of such copies:

   > This Stability AI Model is licensed under the Stability AI Community License, Copyright © Stability AI Ltd. All Rights Reserved.

3. **Prominently display “Powered by Stability AI”** on at least one of:  
   - The RENDEREXPO AI STUDIO website  
   - The RENDEREXPO AI STUDIO user interface  
   - A blog post, about page, or product documentation describing the AI features

RENDEREXPO will:

- Keep a `Licenses/NOTICE.txt` (or `NOTICE.md`) file containing the required Stability AI notice.  
- Ensure that the Wix-based UI for RENDEREXPO AI STUDIO displays **“Powered by Stability AI”** in an appropriate legal / footer / about / credits section.


---

## 3. Hugging Face Access Token

To download and use SD3.5 weights from Hugging Face, RENDEREXPO uses a **personal access token**:

- Provider: **Hugging Face**  
- Type: **“Read”** token (sufficient for pulling model weights)  
- Purpose: Authentication for downloading SD3.5 and related models (ControlNets, etc.)  
- Storage:
  - As an environment variable: `HF_TOKEN`  
  - Optional HF cache directory (example):  
    - `HF_HOME=/workspace/huggingface` (on RunPod)  
    - Or an equivalent directory locally, e.g. under the project root.

**Action recorded:**

- HF account used: **`lalsaman`**  
- Token name: **(to be documented here once created, e.g. `renderexpo-sd35-runtime`)**  
- Token scope: **Read** (model download only; no write).  
- Token created on: **(date to be filled once token is created)**  
- Token handling:
  - The token is **NOT** committed to git or stored in the repository.  
  - It is injected via environment variables / secrets on:
    - The local development machine  
    - The RunPod GPU instance (and any future deployment environment)


---

## 4. Other Models (ControlNet, Upscaler, Depth, etc.)

RENDEREXPO AI STUDIO uses additional models alongside SD3.5, each under their own licenses. The following are examples and may evolve over time:

1. **ControlNet models** (for edges, depth, line art, etc.)  
   - Typical sources (may be updated over time):
     - `lllyasviel/control_v11p_sd15_canny`  
     - `lllyasviel/control_v11f1p_sd15_depth`  
     - `lllyasviel/control_v11p_sd15_lineart`  
   - Each model’s license is stored under `Licenses/ControlNet-LICENSE.txt` or a subfolder such as `Licenses/ControlNet/`.  
   - Used only in ways allowed by their respective licenses.

2. **Upscaler: Real-ESRGAN**  
   - Source: `xinntao/Real-ESRGAN`  
   - License stored as: `Licenses/Real-ESRGAN-LICENSE.txt`  
   - Used to upscale SD3.5 outputs (e.g., to 2K/4K renders) as part of the RENDEREXPO pipeline.

3. **Depth Estimation: MiDaS**  
   - Source: `intel-isl/MiDaS`  
   - License stored as: `Licenses/MiDaS-LICENSE.txt`  
   - Used for geometry/depth extraction, camera effects, and future mesh/CAD tools.

**Note:** For each model, the exact license text is copied into the corresponding file under `Licenses/`, and this document should be updated whenever:

- A new model is added.  
- A model is replaced or removed.  
- The license terms change in a material way.  


---

## 5. Custom Training (LoRA + Finetunes)

RENDEREXPO plans to train the following custom SD3.5 derivatives (examples):

- **LoRAs**:
  - `RENDEREXPO_Exteriors`  
  - `RENDEREXPO_Interiors`  
  - `RENDEREXPO_Aerials`  

- **Full Finetuned Model**:
  - `RENDEREXPO_SD3.5_Ultra` (working name)

Training data:

- ~1000+ high-quality renders created by RENDEREXPO over ~2 years.  
- Data is:
  - Owned by RENDEREXPO (LAS GROUP LLC, DBA RENDEREXPO).  
  - Curated and captioned for architecture (exteriors, interiors, aerials).  
  - High-resolution, high-quality only.

Under the Stability Community License and the specific SD3.5 model license:

- Creating LoRAs and derivative finetuned models for **commercial use under USD $1M/year revenue** is permitted, subject to all terms in those licenses.  
- RENDEREXPO is responsible for ensuring that:
  - Training data is lawfully obtained and owned (which it is, as RENDEREXPO’s own renderings).  
  - Outputs are used in compliance with Stability AI’s Acceptable Use Policy.  

RENDEREXPO will:

- Keep training code and configs in this repository (or a companion private repo).  
- Store outputs (LoRAs, finetuned checkpoints) under RENDEREXPO’s namespace.  
- Document each derivative’s:
  - Training dataset  
  - Training date  
  - Hyperparameters  
  - Hardware used (e.g., NVIDIA RTX 6000 / A6000 on RunPod)  


---

## 6. Future Nano Banana (Google) Integration

In the future, RENDEREXPO AI STUDIO may integrate:

- **Google “Nano Banana” API** as an optional **premium mode**.

Key constraints for this integration:

- SD3.5 remains the **core engine** and primary IP of RENDEREXPO AI STUDIO.  
- Nano Banana is used **only as an external service** under its own API terms.  
- RENDEREXPO will:
  - Negotiate separate terms with Google (if/when applicable).  
  - Keep Nano Banana usage clearly separated in code and UI.  
  - Ensure any API keys/secrets are never committed to this repo.

This document will be updated once:

- Formal terms with Google are in place.  
- The Nano Banana integration is implemented.  


---

## 7. Wix UI Legal & Attribution Requirements (High-Level)

The RENDEREXPO AI STUDIO Wix-based frontend must:

1. Display **“Powered by Stability AI”** clearly in:
   - The app footer, OR  
   - The “About / Legal / Credits” section, OR  
   - A dedicated “AI Technology” section.

2. Provide a link or text pointing to:
   - Stability AI Community License page, and/or  
   - The local /legal or /licenses page describing usage of SD3.5.

3. Provide a **Privacy / Terms** page for RENDEREXPO that:
   - Clarifies that AI-generated images are produced using models licensed from Stability AI and hosted via RENDEREXPO’s infrastructure.  
   - States that RENDEREXPO owns the rendered outputs delivered to clients (subject to any contract).

A separate `docs/wix_ui_legal_requirements.md` file can expand on exact copy, layout, and placement in the Wix UI.


---

## 8. Summary & Commitments

- RENDEREXPO AI STUDIO is built around **Stable Diffusion 3.5** and related models under the **Stability AI Community License**, operating under the **< USD $1M annual revenue** condition.  
- All third-party licenses are stored under the `Licenses/` directory and respected as written.  
- RENDEREXPO (LAS GROUP LLC, DBA RENDEREXPO) owns:
  - Its code.  
  - Its custom datasets.  
  - Its custom LoRAs and finetuned models.  

- External APIs (e.g., Google Nano Banana) and external providers (Stability AI, Hugging Face, etc.) retain their own IP and license rights.

This document should be updated whenever:

- Revenue crosses a major threshold (e.g., approaching or exceeding USD $1M/year).  
- A new external model is added.  
- Major licensing terms change.  
- A new commercial agreement (e.g., with Google) is signed.  
