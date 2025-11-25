"""
pipelines/video_from_image.py

CPU-friendly stub for "video from image" for RENDEREXPO AI STUDIO.

What this does (today, on CPU only):
- Takes a single input image.
- Uses ffmpeg to turn it into a short MP4 clip.
- The clip is essentially a looped still image, with a simple timing.
- We *pretend* to have camera motions ("orbit", "push_in", "pan"),
  but for now they all just create the same simple video.

Later, on GPU:
- We can replace this with a real image-to-video / parallax model.
- The API contract stays the same, so the frontend doesn’t have to change.
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, Any

from config import OUTPUTS_DIR
from file_utils import validate_image_file, generate_output_filename

logger = logging.getLogger(__name__)


def _run_ffmpeg_still_to_video(
    input_image: Path,
    output_video: Path,
    duration_seconds: float,
) -> Dict[str, Any]:
    """
    Use ffmpeg to create a simple MP4 video from a single still image.

    Command pattern (roughly):
      ffmpeg -y -loop 1 -i input.png -t 3 -r 24 -c:v libx264 -pix_fmt yuv420p out.mp4

    -loop 1     : loop the single frame as a "video"
    -t <seconds>: duration
    -r 24       : frame rate
    -c:v libx264 -pix_fmt yuv420p : standard MP4 encoding
    """
    cmd = [
        "ffmpeg",
        "-y",  # overwrite
        "-loop",
        "1",
        "-i",
        str(input_image),
        "-t",
        str(duration_seconds),
        "-r",
        "24",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_video),
    ]

    logger.info("Running ffmpeg command: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        success = proc.returncode == 0
        if success:
            logger.info("ffmpeg video creation succeeded: %s", output_video)
        else:
            logger.error(
                "ffmpeg video creation failed (code=%s). stderr=%s",
                proc.returncode,
                proc.stderr.decode("utf-8", errors="ignore"),
            )
        return {
            "success": success,
            "returncode": proc.returncode,
            "stdout": proc.stdout.decode("utf-8", errors="ignore"),
            "stderr": proc.stderr.decode("utf-8", errors="ignore"),
            "command": " ".join(cmd),
        }
    except FileNotFoundError:
        # ffmpeg not installed or not in PATH
        logger.exception("ffmpeg not found. Video stub will fall back.")
        return {
            "success": False,
            "returncode": None,
            "stdout": "",
            "stderr": "ffmpeg executable not found",
            "command": " ".join(cmd),
        }
    except Exception:
        logger.exception("Unexpected error while running ffmpeg.")
        return {
            "success": False,
            "returncode": None,
            "stdout": "",
            "stderr": "unexpected exception running ffmpeg",
            "command": " ".join(cmd),
        }


def generate_video_from_image(
    input_image_path: str,
    duration_seconds: float = 3.0,
    camera_motion: str = "orbit",
) -> Dict[str, Any]:
    """
    Public API used by main.py /v1/video/from-image.

    Parameters
    ----------
    input_image_path : str
        Path to the input image (/app/uploads/... or /app/outputs/...).
    duration_seconds : float
        Duration of the output video in seconds.
    camera_motion : str
        One of: 'orbit', 'push_in', 'pan' (for now only used for metadata).

    Returns
    -------
    dict
        {
          "engine": "ffmpeg_stub",
          "input_image": "...",
          "output_video_path": "...",
          "duration_seconds": 3.0,
          "camera_motion": "orbit",
          "debug": {...}
        }
    """
    image_path = Path(input_image_path)

    # Validate that the image exists and is a supported format.
    validate_image_file(image_path)

    # Clamp duration to something reasonable on CPU
    try:
        duration = float(duration_seconds)
    except (TypeError, ValueError):
        duration = 3.0

    duration = max(0.5, min(duration, 10.0))  # 0.5s - 10s

    # Generate an output filename under OUTPUTS_DIR
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    video_filename = generate_output_filename(prefix="video_from_image", ext="mp4")
    output_video_path = OUTPUTS_DIR / video_filename

    # Try ffmpeg
    ffmpeg_debug = _run_ffmpeg_still_to_video(
        input_image=image_path,
        output_video=output_video_path,
        duration_seconds=duration,
    )

    if not ffmpeg_debug["success"]:
        # Fallback: create an empty placeholder file so the path exists.
        # (This is just a safety net; in normal Docker setup ffmpeg should exist.)
        logger.warning(
            "ffmpeg failed or not found; creating placeholder video file at %s",
            output_video_path,
        )
        try:
            output_video_path.write_bytes(b"")
        except Exception:
            logger.exception("Failed to create placeholder video file.")

    return {
        "engine": "ffmpeg_stub",
        "input_image": str(image_path),
        "output_video_path": str(output_video_path),
        "duration_seconds": float(duration),
        "camera_motion": camera_motion,
        "debug": ffmpeg_debug,
    }


__all__ = ["generate_video_from_image"]
