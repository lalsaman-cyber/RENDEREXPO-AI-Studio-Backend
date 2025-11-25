\# RENDEREXPO AI STUDIO — Backend



\## How to run (local)



1\. Open a terminal in this folder:

&nbsp;  cd RENDEREXPO-AI-Studio/Backend



2\. Activate the virtual environment:

&nbsp;  .venv\\Scripts\\activate.bat



3\. Start the server:

&nbsp;  uvicorn main:app --reload --port 8000



4\. Open the docs in your browser:

&nbsp;  http://127.0.0.1:8000/docs



\## Current endpoints (stubs)



\- GET /health

\- POST /v1/txt2img

\- POST /v1/img2img



> All endpoints are stubs for now — no real models loaded yet, zero GPU usage.



