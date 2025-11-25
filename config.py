from pathlib import Path
from dataclasses import dataclass

# -------------------------------------------------------------------
# Base paths
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Outputs: where generated images (txt2img, img2img, upscale, depth) go
OUTPUTS_DIR = BASE_DIR / "outputs"

# Uploads: where /v1/upload saves user-uploaded files
UPLOADS_DIR = BASE_DIR / "uploads"

# Logs: for any log files
LOGS_DIR = BASE_DIR / "logs"

# Make sure these directories exist.
for directory in (OUTPUTS_DIR, UPLOADS_DIR, LOGS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# App name / version
# -------------------------------------------------------------------

APP_NAME = "RENDEREXPO AI STUDIO Backend"
APP_VERSION = "0.1.0"


# -------------------------------------------------------------------
# App configuration dataclass
# -------------------------------------------------------------------

@dataclass
class AppConfig:
    # General
    app_name: str = APP_NAME
    app_version: str = APP_VERSION

    # TXT2IMG defaults
    txt2img_default_cfg_scale: float = 7.0
    txt2img_default_inference_steps: int = 30
    txt2img_default_width: int = 1024
    txt2img_default_height: int = 1024

    # IMG2IMG defaults
    img2img_default_strength: float = 0.8
    img2img_default_cfg_scale: float = 7.0
    img2img_default_inference_steps: int = 30
    img2img_default_width: int = 1024
    img2img_default_height: int = 1024


# Global config instance used by pipelines
app_config = AppConfig()

__all__ = [
    "BASE_DIR",
    "OUTPUTS_DIR",
    "UPLOADS_DIR",
    "LOGS_DIR",
    "APP_NAME",
    "APP_VERSION",
    "AppConfig",
    "app_config",
]
