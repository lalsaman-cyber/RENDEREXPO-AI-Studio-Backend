# app/gpu/video/from_image.py
"""
RENDEREXPO AI STUDIO - GPU Video From Image (REAL)

JOB TYPE:
- "video_from_image" (dispatched by app/routers/video_from_image.py)

GOAL:
- Take a single input image (image.png)
- Produce a highly realistic short cinematic video:
    video_from_image.mp4

REAL POLICY:
- This file NEVER pretends success.
- If required tools/models are missing, it raises actionable errors.

BEST-OF-THE-BEST (production-ready, upgradeable):
We keep this module COMMAND-DRIVEN so you can plug in the BEST pipeline available
without changing the API.

You must provide at least ONE of:

(1) Direct MP4 generator (fastest integration):
- VIDEO_FROM_IMAGE_MP4_CMD="python /workspace/pipelines/from_image.py --in {IN} --out {OUT} --fps {FPS} --seconds {SECONDS} --motion {MOTION} --seed {SEED}"
  Requirement: must create {OUT} (mp4)

OR

(2) Frame generator + ffmpeg assembler (best control):
- VIDEO_FROM_IMAGE_FRAMES_CMD="python /workspace/pipelines/from_image_frames.py --in {IN} --outdir {FRAMES} --fps {FPS} --seconds {SECONDS} --motion {MOTION} --seed {SEED}"
  Requirement: must create frames in {FRAMES} named:
      frame_0001.png ... frame_NNNN.png
  Then we assemble mp4 using ffmpeg.

Optional premium refinement hook:
- VIDEO_FROM_IMAGE_REFINE_CMD="python /workspace/pipelines/refine_frames.py --indir {FRAMES} --outdir {FRAMES_REF}"
  Requirement: outputs refined frames with same naming pattern.
  If present, we prefer refined frames for encoding.

ENV VARIABLES (GPU POD):
- VIDEO_FFMPEG_BIN="ffmpeg" (recommended)

INPUTS (job_folder):
- image.png (required)
- meta.json

OUTPUTS (job_folder):
- video_from_image.mp4
- frames/ (optional)
- frames_refined/ (optional)
- viewer/index.html (basic player shell)
"""

from __future__ import annotations

import os
import json
import re
import shlex
import shutil
import subprocess
from typing import Any, Dict, List, Optional


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

_FRAME_RE = re.compile(r"^frame_(\d{4})\.png$", re.IGNORECASE)


def _meta_path(job_folder: str) -> str:
    return os.path.join(job_folder, "meta.json")


def _read_meta(job_folder: str) -> Dict[str, Any]:
    p = _meta_path(job_folder)
    if not os.path.isfile(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_meta(job_folder: str, meta: Dict[str, Any]) -> None:
    with open(_meta_path(job_folder), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _run(cmd: List[str], cwd: Optional[str] = None) -> None:
    p = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if p.returncode != 0:
        out = (p.stdout or "")[-2500:]
        err = (p.stderr or "")[-2500:]
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{out}\n\nSTDERR:\n{err}"
        )


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _require_file(path: str, label: str) -> None:
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing required file: {label} ({path})")


def _cmd_split(cmd: str) -> List[str]:
    # Important: supports quoted args and paths with spaces
    return shlex.split(cmd, posix=True)


def _list_frames(frames_dir: str) -> List[str]:
    if not os.path.isdir(frames_dir):
        return []
    frames = []
    for f in os.listdir(frames_dir):
        if _FRAME_RE.match(f):
            frames.append(f)
    frames.sort(key=lambda x: int(_FRAME_RE.match(x).group(1)))  # type: ignore[union-attr]
    return frames


def _validate_frame_sequence(frames_dir: str, min_frames: int) -> int:
    frames = _list_frames(frames_dir)
    if len(frames) < min_frames:
        raise RuntimeError(
            f"Frames generator did not produce enough frames in {frames_dir}.\n"
            f"Found {len(frames)} frames, need at least {min_frames}.\n"
            "Expected naming: frame_0001.png ... frame_NNNN.png"
        )

    # Premium strictness: verify first ~300 frames are sequential (no gaps)
    check_n = min(len(frames), 300)
    expected = 1
    for i in range(check_n):
        idx = int(_FRAME_RE.match(frames[i]).group(1))  # type: ignore[union-attr]
        if idx != expected:
            raise RuntimeError(
                f"Frame sequence gap or misnaming in {frames_dir}.\n"
                f"Expected frame_{expected:04d}.png but found {frames[i]}.\n"
                "Fix your VIDEO_FROM_IMAGE_FRAMES_CMD output naming."
            )
        expected += 1

    return len(frames)


def _write_basic_video_viewer(job_folder: str, mp4_rel: str = "video_from_image.mp4") -> str:
    viewer_dir = os.path.join(job_folder, "viewer")
    _ensure_dir(viewer_dir)
    index_path = os.path.join(viewer_dir, "index.html")

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>RENDEREXPO AI STUDIO — Video From Image</title>
  <style>
    html,body{{margin:0;height:100%;background:#000;color:#fff;font-family:system-ui,Segoe UI,Roboto,Arial;}}
    .wrap{{display:flex;flex-direction:column;height:100%;}}
    header{{padding:10px 12px;background:#111;border-bottom:1px solid #222;}}
    main{{flex:1;display:flex;align-items:center;justify-content:center;}}
    video{{max-width:96vw;max-height:86vh;border-radius:14px;box-shadow:0 0 0 1px rgba(255,255,255,.08);}}
    .note{{padding:10px 12px;font-size:13px;opacity:.85}}
    code{{color:#fff}}
  </style>
</head>
<body>
  <div class="wrap">
    <header><b>RENDEREXPO AI STUDIO</b> — Video From Image</header>
    <div class="note">Playing: <code>../{mp4_rel}</code></div>
    <main>
      <video controls autoplay loop playsinline>
        <source src="../{mp4_rel}" type="video/mp4" />
      </video>
    </main>
  </div>
</body>
</html>
"""
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html)

    return "viewer/index.html"


def _assemble_mp4_from_frames(job_folder: str, frames_dir: str, fps: int, out_mp4: str) -> None:
    ffmpeg = os.getenv("VIDEO_FFMPEG_BIN", "ffmpeg")
    if not _which(ffmpeg):
        raise RuntimeError("ffmpeg not found. Install ffmpeg or set VIDEO_FFMPEG_BIN to its path.")

    pattern = os.path.join(frames_dir, "frame_%04d.png")
    _run(
        [
            ffmpeg, "-y",
            "-framerate", str(int(fps)),
            "-i", pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "16",
            "-preset", "slow",
            out_mp4,
        ],
        cwd=job_folder,
    )


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def run_video_from_image(job_folder: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute REAL video-from-image pipeline.

    Returns a result dict that dispatcher stores into meta["dispatch"]["result"].
    """
    meta = _read_meta(job_folder) or meta

    img_path = os.path.join(job_folder, "image.png")
    _require_file(img_path, "image.png")

    vr = meta.get("video_runtime") if isinstance(meta.get("video_runtime"), dict) else {}
    fps = int(vr.get("fps", 24))
    seconds = float(vr.get("duration_seconds", 4.0))
    motion = str(vr.get("motion", "cinematic")).strip() or "cinematic"
    seed = int(vr.get("seed", 0)) if str(vr.get("seed", "0")).strip() else 0

    if fps < 8 or fps > 60:
        raise RuntimeError("video_runtime.fps must be between 8 and 60 for production stability.")
    if seconds < 1.0 or seconds > 12.0:
        raise RuntimeError("video_runtime.duration_seconds must be between 1.0 and 12.0.")

    out_mp4_rel = "video_from_image.mp4"
    out_mp4_abs = os.path.join(job_folder, out_mp4_rel)

    frames_dir = os.path.join(job_folder, "frames")
    frames_ref_dir = os.path.join(job_folder, "frames_refined")

    # Meta start
    meta["status"] = "running"
    meta.setdefault("outputs", {})
    meta["outputs"]["video_from_image"] = out_mp4_rel
    meta["outputs"]["viewer_dir"] = "viewer/"
    meta["outputs"]["frames_dir"] = "frames/"
    meta.setdefault("video_from_image", {})
    meta["video_from_image"]["inputs"] = {"image": "image.png"}
    meta["video_from_image"]["runtime"] = {
        "fps": fps,
        "duration_seconds": seconds,
        "motion": motion,
        "seed": seed,
    }
    _write_meta(job_folder, meta)

    mp4_cmd_tpl = os.getenv("VIDEO_FROM_IMAGE_MP4_CMD", "").strip()
    frames_cmd_tpl = os.getenv("VIDEO_FROM_IMAGE_FRAMES_CMD", "").strip()
    refine_cmd_tpl = os.getenv("VIDEO_FROM_IMAGE_REFINE_CMD", "").strip()

    if not mp4_cmd_tpl and not frames_cmd_tpl:
        raise RuntimeError(
            "No GPU pipeline configured for video-from-image.\n\n"
            "Set ONE of:\n"
            "  - VIDEO_FROM_IMAGE_MP4_CMD (must output mp4)\n"
            "  - VIDEO_FROM_IMAGE_FRAMES_CMD (must output frames/)\n\n"
            "Example:\n"
            "  VIDEO_FROM_IMAGE_FRAMES_CMD='python /workspace/pipelines/from_image_frames.py "
            "--in {IN} --outdir {FRAMES} --fps {FPS} --seconds {SECONDS} --motion {MOTION} --seed {SEED}'"
        )

    # (1) Direct MP4 generator
    if mp4_cmd_tpl:
        cmd_str = mp4_cmd_tpl.format(
            IN=img_path,
            OUT=out_mp4_abs,
            FPS=fps,
            SECONDS=seconds,
            MOTION=motion,
            SEED=seed,
            JOB=job_folder,
        )
        _run(_cmd_split(cmd_str), cwd=job_folder)

        if not os.path.isfile(out_mp4_abs) or os.path.getsize(out_mp4_abs) < 50_000:
            raise RuntimeError("MP4 generator finished but video_from_image.mp4 is missing or too small.")

        viewer = _write_basic_video_viewer(job_folder, mp4_rel=out_mp4_rel)

        meta = _read_meta(job_folder) or meta
        meta["status"] = "completed"
        meta["outputs"]["viewer_dir"] = "viewer/"
        meta["outputs"]["video_from_image"] = out_mp4_rel
        meta["outputs"]["preview_video"] = out_mp4_rel
        meta["video_from_image"]["method"] = "mp4_cmd"
        meta["video_from_image"]["viewer"] = viewer
        _write_meta(job_folder, meta)

        return {
            "mode": "video_from_image",
            "method": "mp4_cmd",
            "video": out_mp4_rel,
            "viewer": viewer,
            "fps": fps,
            "seconds": seconds,
            "motion": motion,
            "seed": seed,
        }

    # (2) Frames generator + optional refine + ffmpeg
    _ensure_dir(frames_dir)

    cmd_str = frames_cmd_tpl.format(
        IN=img_path,
        FRAMES=frames_dir,
        FPS=fps,
        SECONDS=seconds,
        MOTION=motion,
        SEED=seed,
        JOB=job_folder,
    )
    _run(_cmd_split(cmd_str), cwd=job_folder)

    min_frames = max(12, int(fps))
    n = _validate_frame_sequence(frames_dir, min_frames=min_frames)

    frames_for_encode = frames_dir

    if refine_cmd_tpl:
        _ensure_dir(frames_ref_dir)
        rcmd_str = refine_cmd_tpl.format(
            FRAMES=frames_dir,
            FRAMES_REF=frames_ref_dir,
            FPS=fps,
            SECONDS=seconds,
            MOTION=motion,
            SEED=seed,
            JOB=job_folder,
        )
        _run(_cmd_split(rcmd_str), cwd=job_folder)

        n_ref = len(_list_frames(frames_ref_dir))
        if n_ref >= n:
            _validate_frame_sequence(frames_ref_dir, min_frames=min_frames)
            frames_for_encode = frames_ref_dir

    _assemble_mp4_from_frames(job_folder, frames_for_encode, fps=fps, out_mp4=out_mp4_abs)

    if not os.path.isfile(out_mp4_abs) or os.path.getsize(out_mp4_abs) < 50_000:
        raise RuntimeError("ffmpeg finished but video_from_image.mp4 is missing or too small.")

    viewer = _write_basic_video_viewer(job_folder, mp4_rel=out_mp4_rel)

    meta = _read_meta(job_folder) or meta
    meta["status"] = "completed"
    meta["outputs"]["viewer_dir"] = "viewer/"
    meta["outputs"]["video_from_image"] = out_mp4_rel
    meta["outputs"]["preview_video"] = out_mp4_rel
    meta["video_from_image"]["method"] = "frames_cmd+ffmpeg" + ("+refine" if refine_cmd_tpl else "")
    meta["video_from_image"]["viewer"] = viewer
    meta["video_from_image"]["frames_used_for_encode"] = "frames_refined/" if frames_for_encode == frames_ref_dir else "frames/"
    meta["video_from_image"]["frames_count"] = len(_list_frames(frames_for_encode))
    _write_meta(job_folder, meta)

    return {
        "mode": "video_from_image",
        "method": meta["video_from_image"]["method"],
        "video": out_mp4_rel,
        "viewer": viewer,
        "fps": fps,
        "seconds": seconds,
        "motion": motion,
        "seed": seed,
        "frames_dir": "frames/",
        "frames_used_for_encode": meta["video_from_image"]["frames_used_for_encode"],
        "frames_count": meta["video_from_image"]["frames_count"],
    }
