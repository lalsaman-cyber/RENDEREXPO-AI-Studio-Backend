\# H1A — Hugging Face Legal Compliance  

RENDEREXPO AI STUDIO  

Last Updated: 11/22/2025



This document records how RENDEREXPO AI STUDIO complies with all Hugging Face legal requirements, repository-level model licenses, usage restrictions, and acceptable use guidelines.  

It is intended for:

\- IP filings  

\- Legal \& licensing audits  

\- Internal compliance verification  

\- Regulatory review  

\- Partner negotiations (e.g., Google Nano Banana)  



---



\## 1. Hugging Face Account Information



\*\*Hugging Face Username:\*\* lalsaman  

\*\*Email:\*\* Linked to LAS GROUP LLC / RENDEREXPO  

\*\*Account Owner:\*\* Louai Alsaman  

\*\*Organization:\*\* LAS GROUP LLC, DBA RENDEREXPO  

\*\*Purpose:\*\* Commercial usage of architectural rendering models under < $1M annual revenue.



This account is authorized to:

\- Access models

\- Accept model licenses  

\- Obtain read tokens  

\- Download permitted weights for SD3.5 and related models  



No private or write-access tokens are stored in the codebase or repository.



---



\## 2. Hugging Face Access Token Compliance



\*\*Token Type:\*\* READ-ONLY  

\*\*Token Purpose:\*\*  

\- Pulling model weights  

\- Authenticating SD3.5 + ControlNet downloads  

\- Using diffusers pipelines  

\- Accessing Real-ESRGAN, MiDaS, and other public models  



\*\*Token Storage Rules:\*\*  

\- NEVER committed to Git  

\- Stored only as secure environment variable `HF\_TOKEN`  

\- Encrypted in RunPod secret storage (when GPU pod is launched)  

\- NEVER embedded in any file, config, or code distributed to users  



---



\## 3. Hugging Face Model Licenses



Hugging Face does \*\*NOT\*\* provide its own universal AUP.  

Instead, each model carries its own license.  

RENDEREXPO complies with:



\### 3.1 — Stability AI Models (SD3.5 and SDXL)  

License: \*\*Stability AI Community License\*\*  

Stored locally in:

```

/docs/licenses/StabilityAI-Community-License.txt

```



\### 3.2 — ControlNet Models  

Taken from the official `lllyasviel` repositories.  

Licenses stored at:

```

/docs/licenses/ControlNet-LICENSE.txt

```



\### 3.3 — Real-ESRGAN  

License stored at:

```

/docs/licenses/Real-ESRGAN-LICENSE.txt

```



\### 3.4 — MiDaS  

License stored at:

```

/docs/licenses/MiDaS-LICENSE.txt

```



\### 3.5 — Any Future Models  

For every new HF model added:  

\- The LICENSE file must be copied here  

\- Logged with date + purpose  

\- Reviewed for commercial usage rights  



All models used by RENDEREXPO must:

\- Allow commercial usage under < $1M revenue  

\- Allow derivative works (needed for LoRA + finetune)  

\- Not restrict 3D, textile, CAD, or architecture use cases  



---



\## 4. Hugging Face Acceptable Use Summary



Hugging Face policies prohibit:



\- Illegal activities or harmful conduct  

\- Harassment, hate, abuse, or discrimination  

\- Malware, hacking, or security compromise  

\- Misinformation intended to harm others  

\- Unauthorized collection of sensitive personal data  

\- Unauthorized regulated professional advice (medical/legal/financial)  



RENDEREXPO AI STUDIO does \*\*not\*\* engage in any restricted activity.  

It is strictly a \*\*visual rendering system\*\* for architecture and real estate.



---



\## 5. Use of Hugging Face Models in RENDEREXPO AI STUDIO



\### 5.1 SD3.5 Large  

Used for:

\- txt2img  

\- img2img  

\- ControlNet-guided architectural rendering  

\- LoRA training \& fine-tuning  

\- Interior, exterior, aerial, and real estate visualization  



\### 5.2 ControlNet  

Used for:

\- Depth guidance  

\- Edge alignment  

\- CAD/floorplan extraction  

\- Clay/line-art conditioning  



\### 5.3 Real-ESRGAN  

Used for:

\- 2K/4K/8K architectural upscaling  

\- Texture sharpening  

\- Detail preservation  



\### 5.4 MiDaS  

Used for:

\- Camera depth maps  

\- 3D reconstruction helpers  

\- Scene geometry understanding  



No model is used outside its licensed scope.



---



\## 6. Commercial Use Declaration



RENDEREXPO AI STUDIO falls under:



\*\*Commercial Use < $1M Revenue\*\*, allowed by:

\- Stability AI Community License  

\- Hugging Face model-level agreements  



If RENDEREXPO exceeds $1M annual revenue,  

we will upgrade immediately to Stability AI Enterprise licensing.



---



\## 7. Public Attribution Requirements



As required by Stability + Hugging Face licenses:



\- “\*\*Powered by Stability AI\*\*” must be displayed  

\- Model license files must be redistributed when models are redistributed  

\- A “Notice” file must accompany derivative works  

\- Hugging Face must receive attribution where their infrastructure is used  



The required public attribution text is stored in section \*\*7 of H1B\*\*.



When we build the WIX pages, these exact sentences will be placed automatically in:

\- Footer

\- Terms

\- AI Studio interface

\- Legal / About page



---



\## 8. Storage \& Security



\- All HF model weights stored in `/models/` are permitted  

\- Weights are downloaded only after license acceptance  

\- No model requiring “non-commercial” terms is allowed  

\- No private model or gated model is redistributed  



---



\## 9. Conclusion



RENDEREXPO AI STUDIO is in full compliance with all Hugging Face licensing expectations, AUPs, and Stability AI model requirements.  

This file should be updated whenever:

\- A new model is added  

\- A license changes  

\- A revenue threshold changes  

\- IP filing milestones occur  





