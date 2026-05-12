# THIRD-PARTY MODEL LICENSES AND ATTRIBUTION

## Purpose

This file documents the third-party AI model licenses, attribution requirements, commercial-use notes, and operational usage boundaries for RENDEREXPO AI STUDIO.

This file is intended for internal compliance tracking and launch preparation. It is not legal advice. RENDEREXPO should consult legal counsel before public launch if contractual legal certainty, indemnification, or enterprise-risk protection is required.

---

## Current compliance position

RENDEREXPO AI STUDIO uses and/or may use selected third-party AI models for architectural visualization, sketch-to-render generation, material transfer, moodboard workflows, inpainting, object insertion, and interior staging.

The projectâ€™s compliance principles are:

- Use official model repositories only.
- Avoid random mirrors, unknown checkpoint reuploads, and unverified model sources.
- Preserve required license notices and attribution.
- Do not imply endorsement by model authors or model providers.
- Keep model usage separated by service lane.
- Track which model is used for which RENDEREXPO AI STUDIO service.
- Use all models only for lawful architectural, visualization, design, and real-estate workflows.
- Do not use third-party models for prohibited or high-risk uses disallowed by their licenses or provider terms.
- If RENDEREXPO later requires legal indemnification, use a paid enterprise model/vendor agreement rather than relying only on open-source/public model licenses.

---

# 1) TheMistoAI / MistoLine

## Model

- Name: `TheMistoAI/MistoLine`
- Common description: MistoLine-SDXL-ControlNet
- Use in RENDEREXPO AI STUDIO: Sketch-to-render structural line/sketch conditioning

## License

- License family: OpenRAIL++ / CreativeML Open RAIL++-M style licensing, as published on the model repository.

## Commercial use

Commercial use is permitted provided the attribution, non-endorsement, and license conditions are followed.

## Required attribution

RENDEREXPO must include visible attribution for commercial use of MistoLine.

Approved attribution text:

> This product uses the MistoLine-SDXL-ControlNet developed by TheMisto.ai.  
> TheMisto.ai does not endorse or sponsor this product.

## Recommended public website wording

For the RENDEREXPO AI STUDIO website, use:

> Certain sketch-to-render workflows use MistoLine-SDXL-ControlNet developed by TheMisto.ai. TheMisto.ai does not endorse, sponsor, or promote RENDEREXPO AI STUDIO.

## Attribution placement

The attribution should appear in at least one visible place, such as:

- Website legal page
- AI Model Attribution / Third-Party Notices page
- Product About / Credits page
- Product documentation
- README or model credits section

Recommended website placement:

- Footer link: `AI Model Attribution & Third-Party Notices`
- Or legal/footer group near:
  - Privacy Policy
  - Terms of Use
  - AI Usage Policy

## Non-endorsement rule

RENDEREXPO must not state or imply that TheMisto.ai endorses, sponsors, approves, promotes, or is affiliated with RENDEREXPO AI STUDIO beyond the required model attribution itself.

## Usage boundary

MistoLine will be used only for architectural sketch-to-render and sketch-conditioned rendering workflows inside RENDEREXPO AI STUDIO.

## Operational notes

- Continue using the locked MistoLine checkpoint unless intentionally changed.
- Keep MistoLine isolated to the SDXL sketch/line-control service lane.
- Do not use MistoLine in the SD3.5 Large text-to-image, moodboard, or material-transfer lanes unless a new license/runtime review is performed.
- Do not remove public attribution before launch.

---

# 2) TheMistoAI / Anyline or ComfyUI-Anyline

## Component

- Name: Anyline / ComfyUI-Anyline
- Use in RENDEREXPO AI STUDIO: Line preprocessing / sketch preprocessing for MistoLine sketch workflows

## License

- License: MIT License, according to the repository listing.

## Commercial use

MIT generally allows commercial use, modification, distribution, sublicensing, and sale, provided the copyright and permission notices are preserved.

## Recommended documentation note

> Some sketch preprocessing workflows may use Anyline / ComfyUI-Anyline by TheMistoAI.

## Website attribution

A separate public website attribution is not currently required beyond general third-party notices, but it may be included on the same AI Model Attribution / Third-Party Notices page.

## Operational notes

- Preserve the MIT license notice if the component is redistributed.
- Keep Anyline tied to sketch preprocessing and MistoLine workflows.
- Do not imply endorsement by TheMistoAI.

---

# 3) Stability AI / SDXL Base 1.0

## Model

- Name: `stabilityai/stable-diffusion-xl-base-1.0`
- Use in RENDEREXPO AI STUDIO: Base image generation model paired with MistoLine for sketch-to-render workflows

## License

- License: CreativeML Open RAIL++-M

## Commercial use

Use is allowed subject to the model license terms and restricted-use conditions.

## Usage boundary

SDXL Base 1.0 will be used only as the dedicated sketch-generation engine paired with MistoLine and related sketch-conditioning workflows.

SDXL Base 1.0 should remain separate from:

- SD3.5 Large text-to-image
- SD3.5 img2img
- Moodboard workflows
- Moodboard Material Transfer
- Interior staging / object insertion, unless specifically reviewed

## Operational notes

- Keep the SDXL + MistoLine sketch lane isolated.
- Do not mix SDXL/MistoLine runtime behavior into SD3.5 services without explicit review.
- Preserve license notices and restricted-use compliance.

---

# 4) Stability AI / Stable Diffusion 3.5 Large

## Model

- Name: `stabilityai/stable-diffusion-3.5-large`
- Use in RENDEREXPO AI STUDIO:
  - Main text-to-image generation
  - SD3.5 img2img
  - Moodboard-to-space generation
  - Moodboard Material Transfer
  - Selected SD3.5-based visualization workflows

## License

- License: Stability AI Community License, as published by Stability AI for Stable Diffusion 3.5 Large.

## Commercial use

The Stability AI Community License permits commercial use for organizations or individuals under the revenue threshold stated by Stability AI, currently understood by RENDEREXPO as under USD $1,000,000 annual revenue.

If RENDEREXPO exceeds the applicable revenue threshold or usage category, RENDEREXPO must obtain the appropriate enterprise/commercial license from Stability AI or replace the model with a model whose license supports the new business condition.

## Usage boundary

SD3.5 Large is the primary RENDEREXPO AI STUDIO generation model for general text-to-image and SD3.5-specific workflows.

Current/approved service lanes include:

- Text-to-image
- Img2img
- Moodboard-to-space
- Moodboard Material Transfer
- Other SD3.5 workflows only when explicitly reviewed

## Operational notes

- SD3.5 Large should remain the default engine/version for RENDEREXPO AI STUDIO unless intentionally changed.
- Keep SD3.5 workflows separate from SDXL/MistoLine sketch workflows.
- Track annual revenue threshold compliance.
- Do not use SD3.5 Large beyond the licenseâ€™s restricted-use conditions.
- If business revenue or product scope changes, re-review the license before further commercial rollout.

---

# 5) PowerPaint / PowerPaint-v2-1

## Model / project

- Code repository: `open-mmlab/PowerPaint`
- Model repository: `JunhaoZhuang/PowerPaint-v2-1`
- Use in RENDEREXPO AI STUDIO: Candidate / approved for Interior Staging, Object Insertion, Object Removal, Inpainting, and Outpainting workflows, subject to isolated implementation and successful quality testing.

## Code license

- PowerPaint code license: MIT License

## Model license

- PowerPaint-v2-1 model license listed on Hugging Face: Apache License 2.0

## Commercial-use decision

PowerPaint / PowerPaint-v2-1 is approved for commercial use in RENDEREXPO AI STUDIO based on the public MIT code license and Apache-2.0 model license, provided required notices are preserved and the model is used within applicable law, platform policies, and license conditions.

This means RENDEREXPO may use PowerPaint / PowerPaint-v2-1 to support a website-based or client-facing paid service, including interior staging, object insertion, inpainting, and related visualization workflows.

## Important legal caveat

This approval does not mean the model is indemnified or lawsuit-proof. Open-source model licenses generally grant permission to use the software/model under the stated terms, but they do not provide enterprise indemnification.

If RENDEREXPO requires contractual indemnity, a paid enterprise model/vendor agreement would be required.

## License meaning for RENDEREXPO

MIT generally allows commercial use, modification, distribution, sublicensing, and sale, provided copyright and permission notices are preserved.

Apache-2.0 generally allows commercial use, reproduction, modification, distribution, sublicensing, public display/performance, and creation of derivative works, provided the license conditions are followed, including preserving notices and marking modified files where applicable.

## Website attribution

No specific prominent website attribution requirement was identified beyond preserving required license notices. However, RENDEREXPO should include PowerPaint on the same AI Model Attribution / Third-Party Notices page for transparency and compliance tracking.

Recommended public website wording:

> Certain object insertion, inpainting, and interior staging workflows may use PowerPaint / PowerPaint-v2-1, an open-source image inpainting model. PowerPaint and its authors are not affiliated with, endorsed by, or responsible for RENDEREXPO AI STUDIO.

## Operational notes

- Use the official Hugging Face repository only.
- Do not use random mirrors, CivitAI reuploads, or unverified checkpoint files.
- Keep PowerPaint isolated from the existing SD3.5, MistoLine, moodboard, and sketch runtimes until validated.
- Add downloaded model snapshot/license files to internal license tracking where possible.
- Do not imply endorsement by PowerPaint, OpenMMLab, JunhaoZhuang, or model authors.
- Do not modify core production services until PowerPaint quality is tested in isolation.
- Do not integrate into the FastAPI/GPU worker lane until a standalone POD test is successful.

---

# 6) RENDEREXPO Moodboard System

## Services

The RENDEREXPO AI STUDIO moodboard backend currently includes:

1. Space to Moodboard
2. Moodboard to Space
3. Moodboard Material Transfer

## Current locked baseline

### Space to Moodboard

- Status: Locked
- Version: Moodboard V8
- Commit: `f640818`
- Layout version: `moodboard_v8_premium_physical_flatlay`

### Moodboard Material Transfer

- Status: Locked
- Default strength: `0.55`
- Commit: `f21a505`

### Recommended transfer modes

- Subtle Transfer: `0.45`
- Balanced Transfer / Default: `0.55`
- Strong Transfer: `0.65`
- Creative Reinterpretation: `0.75`

## Licensing note

Moodboard workflows currently rely on existing approved internal RENDEREXPO backend generation/analysis lanes and SD3.5 Large where applicable.

Any third-party model used inside the moodboard workflow must be separately tracked in this document before public launch.

## Operational notes

- Do not retune the locked Moodboard V8 baseline unless intentionally starting a new controlled version.
- Do not change the Moodboard Material Transfer default strength without a new validation run.
- When moving to Wix, use the locked V8 and 0.55 material-transfer baseline as the integration foundation.

---

# 7) Internal architecture boundary

RENDEREXPO AI STUDIO separates generation paths by service lane.

## A. Text-to-image

- Engine: SD3.5 Large
- Purpose: Main text-to-image generation

## B. Img2img

- Engine: SD3.5 Large
- Purpose: Controlled refinement, render improvement, and selected transformation workflows

## C. Sketch-to-render

- Engine: SDXL Base 1.0 + MistoLine
- Purpose: Convert architectural sketches / line drawings into photorealistic renders

## D. Sketch-to-redesign

- Engine: SDXL Base 1.0 + MistoLine-based redesign workflow
- Purpose: Reinterpret sketch faÃ§ade/material/style while preserving the sketch-driven base structure within the intended redesign rules

## E. Moodboard workflows

- Engine: SD3.5 Large and internal analysis/asset extraction workflows where applicable
- Purpose:
  - Generate material moodboards from spaces
  - Generate new spaces from moodboards
  - Apply moodboard direction to existing renders

## F. Interior staging / object insertion

- Candidate engine: PowerPaint / PowerPaint-v2-1
- Purpose:
  - Object insertion
  - Interior staging
  - Inpainting
  - Object removal
  - Shape-guided object generation

This service must remain isolated until tested and approved.

---

# 8) Restricted-use compliance rule

RENDEREXPO will not use third-party models for prohibited, unlawful, or high-risk uses disallowed by their governing licenses, model cards, or provider terms.

RENDEREXPO will use these models only for lawful architectural visualization, interior design, design communication, real-estate visualization, digital construction, and related creative/professional workflows.

Prohibited or restricted uses include, but are not limited to:

- Illegal activity
- Exploitation or harm involving minors
- Privacy infringement
- Defamation, harassment, or impersonation
- Fraudulent or deceptive content
- Harmful automated decision-making
- Discriminatory uses
- Medical, legal, financial, immigration, or law-enforcement decision systems
- Any other use prohibited by the applicable model license or provider terms

---

# 9) Website attribution and public notices

Before launching RENDEREXPO AI STUDIO publicly, create a website footer/legal page titled:

> AI Model Attribution & Third-Party Notices

Recommended public website wording:

> RENDEREXPO AI STUDIO uses selected open-source and commercially licensed AI model components to support architectural rendering, sketch-to-render, material transfer, inpainting, and visualization workflows.
>
> Certain sketch-to-render workflows use MistoLine-SDXL-ControlNet developed by TheMisto.ai. TheMisto.ai does not endorse, sponsor, or promote RENDEREXPO AI STUDIO.
>
> Certain object insertion, inpainting, and interior staging workflows may use PowerPaint / PowerPaint-v2-1, an open-source image inpainting model. PowerPaint and its authors are not affiliated with, endorsed by, or responsible for RENDEREXPO AI STUDIO.
>
> All outputs remain subject to RENDEREXPO AI STUDIOâ€™s terms of use, user input rights, and applicable law.

## Footer placement recommendation

In the website footer, add a link:

> AI Model Attribution & Third-Party Notices

Place it near:

- Privacy Policy
- Terms of Use
- AI Usage Policy
- Contact

---

# 10) Operational compliance checklist

Before launch or public rollout, confirm:

- [ ] MistoLine attribution is added to the website or product credits.
- [ ] Non-endorsement language is included for TheMisto.ai.
- [ ] PowerPaint is listed in the Third-Party Notices page if used.
- [ ] Non-endorsement language is included for PowerPaint / model authors.
- [ ] This license note file is stored in the repo.
- [ ] Sketch mode is implemented as a separate SDXL + MistoLine pipeline.
- [ ] SD3.5 Large remains separate from sketch mode.
- [ ] Moodboard V8 baseline is preserved unless intentionally versioned.
- [ ] Moodboard Material Transfer default remains locked at `0.55`.
- [ ] PowerPaint is tested in isolation before FastAPI/GPU integration.
- [ ] Team members do not remove or alter required attribution text.
- [ ] Model usage remains limited to lawful architectural visualization, design, and real-estate workflows.
- [ ] Annual revenue threshold and licensing requirements are re-reviewed before scaling beyond the current commercial-use assumptions.
- [ ] Any newly added model is documented in this file before public use.

---

# 11) Internal decision notes

## MistoLine decision

RENDEREXPO selected MistoLine because it is a dedicated line/sketch-oriented control model more suitable for sketch-to-render than the prior SD3.5 Large raw Canny-only approach.

This decision was made to improve architectural sketch fidelity while preserving a separate and stable SD3.5 Large text-to-image pipeline.

## Moodboard decision

RENDEREXPO locked Moodboard V8 as the baseline because it produced the strongest physical flat-lay material-board output, with readable material naming and a matched materials schedule.

Moodboard Material Transfer default strength was locked at `0.55` because it provided the best balance between applying the moodboard direction and preserving the original render.

## PowerPaint decision

RENDEREXPO is evaluating PowerPaint / PowerPaint-v2-1 because SD3.5 Large img2img alone is not specialized enough for high-quality object insertion or furniture/interior staging.

PowerPaint is considered the preferred candidate for the Interior Staging / Object Insertion service because it is designed for inpainting, object insertion, object removal, and shape-guided insertion, and its public license signals are commercially permissive.

PowerPaint must be tested in isolation before integration into the production GPU worker.

---

# 12) Records to update over time

When a third-party model is downloaded or integrated, add:

- Model name
- Repository URL
- License name
- Download date
- Snapshot / commit hash if available
- Local path
- Service lane using the model
- Commercial-use decision
- Attribution requirement
- Any restricted-use notes
- Whether the model is integrated, candidate-only, deprecated, or removed

