\# RENDEREXPO ULTRA – SD3.5 Training Plan



\_Last updated: 2025-11-23\_



This document is the \*\*source of truth\*\* for how we will train and manage:



\- \*\*RENDEREXPO ULTRA\*\* (full SD3.5 finetune)

\- \*\*RENDEREXPO LoRAs\*\*:

&nbsp; - Interiors

&nbsp; - Exteriors

&nbsp; - Aerials

&nbsp; - Furniture / materials

&nbsp; - Room types (kitchens, living rooms, bedrooms)

&nbsp; - Architectural lighting

\- Custom \*\*Refiners\*\* (low-denoise, material detail, lighting, geometry, high-res)



No actual training happens yet. This is just planning + config structure.



---



\## 1. Base Model \& Legal Scope



\- \*\*Base model\*\*: Stable Diffusion \*\*3.5 Large\*\*

\- \*\*Location\*\*: `models/sd35-large`

\- \*\*Usage\*\*: Within Stability AI’s \*\*Stable Diffusion 3.5\*\* license and AUP

\- \*\*RENDEREXPO ULTRA\*\*:

&nbsp; - A finetuned SD3.5 model

&nbsp; - Used only within RENDEREXPO AI STUDIO

&nbsp; - Under \*\*$1M/year\*\* revenue (for now), staying inside allowed scope



We will:



\- Keep \*\*all training configs\*\* in `config/training/`

\- Keep \*\*all training logs \& outputs\*\* in `training\_runs/` (planned)

\- Never ship any training code that violates:

&nbsp; - Stability AI AUP

&nbsp; - OpenAI / platform safety rules

&nbsp; - Open-source model licenses



---



\## 2. Training Folder Structure (Planned)



Target layout:



```text

Backend/

│

├── config/

│   ├── model\_paths.yaml

│   └── training/

│       ├── base\_sd35.yaml

│       ├── lora\_interiors.yaml

│       ├── lora\_exteriors.yaml

│       ├── lora\_aerials.yaml

│       ├── lora\_furniture\_materials.yaml

│       ├── lora\_room\_types.yaml

│       ├── lora\_lighting.yaml

│       ├── refiner\_detail.yaml

│       ├── refiner\_lighting.yaml

│       ├── refiner\_geometry.yaml

│       └── refiner\_highres.yaml

│

├── training\_scripts/   (future – skeleton only for now)

│   ├── train\_lora.py

│   ├── train\_refiner.py

│   └── train\_full\_finetune.py

│

└── training\_runs/      (future – created when actual training starts)

&nbsp;   ├── loras/

&nbsp;   ├── refiners/

&nbsp;   └── full\_models/



