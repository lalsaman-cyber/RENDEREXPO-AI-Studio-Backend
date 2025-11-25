from pathlib import Path
from datetime import datetime
from typing import Union, Optional
import shutil

try:
    from logs.log_utils import get_logger  # type: ignore
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_logger(name: str):
        return logging.getLogger(name)

from config import OUTPUTS_DIR, UPLOADS_DIR, LOGS_DIR  # type: ignore

logger = get_logger(__name__)


# -------------------------------------------------------------------
# Filename / path helpers
# -------------------------------------------------------------------

def generate_output_filename(prefix: str = "output", ext: str = "png") -> str:
    """
    Generate a unique filename using UTC timestamp and a prefix.

    Example:
      prefix="txt2img", ext="png"
      -> "txt2img_20251115T123456789012.png"
    """
    # Normalize extension to not have a leading dot.
    ext = ext.lstrip(".")
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    filename = f"{prefix}_{ts}.{ext}"
    logger.info("Generated output filename: %s", filename)
    return filename


def validate_image_file(path: Union[str, Path]) -> Path:
    """
    Validate that the given path points to an existing image-like file.

    This is a lightweight check:
      - file exists
      - extension is one of a basic set of known image types
    """
    p = Path(path)

    if not p.is_file():
        raise FileNotFoundError(f"Input image file does not exist: {p}")

    allowed_exts = {".png", ".jpg", ".jpeg", ".webp"}
    if p.suffix.lower() not in allowed_exts:
        raise ValueError(
            f"Unsupported image extension '{p.suffix}' for file {p}. "
            f"Allowed: {sorted(allowed_exts)}"
        )

    logger.info("Validated image file: %s", p)
    return p


# -------------------------------------------------------------------
# Upload handling
# -------------------------------------------------------------------

def save_upload_file(
    upload_file,
    prefix: str = "upload",
) -> Path:
    """
    Save an uploaded file (FastAPI UploadFile or file-like) into UPLOADS_DIR
    with a safe, unique name, and return the final Path.

    We only care about:
      - extension from original filename (default .png if missing)
      - writing binary content to disk
    """
    # Get extension from original filename, default to .png
    original_name = getattr(upload_file, "filename", "") or ""
    ext = Path(original_name).suffix or ".png"
    ext = ext.lstrip(".")

    filename = generate_output_filename(prefix=prefix, ext=ext)
    dest_path = UPLOADS_DIR / filename

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    # upload_file.file is a SpooledTemporaryFile when using FastAPI UploadFile
    source = getattr(upload_file, "file", upload_file)

    with dest_path.open("wb") as buffer:
        shutil.copyfileobj(source, buffer)

    logger.info("Saved uploaded file to: %s", dest_path)
    return dest_path


def resolve_image_path(
    file_id: Optional[str] = None,
    input_image_path: Optional[str] = None,
) -> Path:
    """
    Resolve an image path given either:
      - a file_id previously returned by /v1/upload (stored under UPLOADS_DIR)
      - a direct input_image_path (absolute or relative)

    Returns a validated Path.
    """
    if file_id:
        # Interpret as a file we saved under UPLOADS_DIR
        candidate = UPLOADS_DIR / file_id
        logger.info("Resolving image by file_id: %s -> %s", file_id, candidate)
        return validate_image_file(candidate)

    if input_image_path:
        p = Path(input_image_path)
        if not p.is_absolute():
            # If it's a relative path, assume it's under OUTPUTS_DIR or UPLOADS_DIR
            # Here we try OUTPUTS_DIR first, then UPLOADS_DIR.
            candidate_output = OUTPUTS_DIR / p.name
            candidate_upload = UPLOADS_DIR / p.name

            if candidate_output.is_file():
                logger.info("Resolved relative image path under OUTPUTS_DIR: %s", candidate_output)
                return validate_image_file(candidate_output)
            if candidate_upload.is_file():
                logger.info("Resolved relative image path under UPLOADS_DIR: %s", candidate_upload)
                return validate_image_file(candidate_upload)

        logger.info("Resolving image by direct path: %s", p)
        return validate_image_file(p)

    raise ValueError("Either file_id or input_image_path must be provided.")


# -------------------------------------------------------------------
# Logs dir (backwards compatibility)
# -------------------------------------------------------------------

def get_logs_dir() -> Path:
    """
    Legacy helper used by the old log_utils.py near main.py.

    Returns the logs directory defined in config.py.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return LOGS_DIR


__all__ = [
    "generate_output_filename",
    "validate_image_file",
    "save_upload_file",
    "resolve_image_path",
    "get_logs_dir",
    "OUTPUTS_DIR",
    "UPLOADS_DIR",
    "LOGS_DIR",
]
