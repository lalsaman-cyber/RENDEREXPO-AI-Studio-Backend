# app/gpu/video/between_frames.py
"""
RENDEREXPO AI STUDIO - GPU Video Between Frames (REAL)

JOB TYPE:
- "video_between_frames" (dispatched by app/routers/video_between_frames.py)

GOAL:
- Take first.png + last.png (two keyframes)
- Produce a highly realistic bridge video:
    outputs/video_between.mp4

REAL POLICY:
- This file NEVER pretends success.
- If required tools/models are missing, it raises actionable errors.

BEST-OF-THE-BEST (production-ready structure):
A) Premium bridge (recommended)
   1) Generate intermediate frames (your best method)
   2) Optional refinement stage
   3) Assemble mp4 via ffmpeg (libx264)

B) Direct MP4 generator (fastest integration)
   - One command produces mp4 directly (still REAL)

ENV VARIABLES (GPU POD):
- VIDEO_FFMPEG_BIN="ffmpeg" (optional but recommended)

You must provide at least ONE of these:

(1) Direct MP4 generator:
- VIDEO_BETWEEN_MP4_CMD="python /workspace/pipelines/between_frames.py --first {FIRST} --last {LAST} --out {OUT} --fps {FPS} --seconds {SECONDS}"

(2) Frame generator + ffmpeg assembler:
- VIDEO_BETWEEN_FRAMES_CMD="python /workspace/pipelines/between_frames_frames.py --first {FIRST} --last {LAST} --outdir {FRAMES} --fps {FPS} --seconds {SECONDS}"
  Must produce: frame_0001.png ... frame_NNNN.png

Optional refinement hook:
- VIDEO_BETWEEN_REFINE_CMD="python /workspace/pipelines/refine_frames.py --indir {FRAMES} --outdir {FRAMES_REF}"

INPUTS (job_folder):
- first.png + last.png   (primary)
  OR
- frame_start.png + frame_end.png (accepted fallback)
- meta.json

OUTPUTS (job_folder):
- video_between.mp4
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
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

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


def _pick_input_pair(job_folder: str) -> Tuple[str, str, str]:
    """
    Accept either:
      - first.png + last.png  (preferred)
      - frame_start.png + frame_end.png (fallback)

    Returns: (first_path, last_path, naming_mode)
    """
    first_a = os.path.join(job_folder, "first.png")
    last_a = os.path.join(job_folder, "last.png")
    if os.path.isfile(first_a) and os.path.isfile(last_a):
        return first_a, last_a, "first_last"

    first_b = os.path.join(job_folder, "frame_start.png")
    last_b = os.path.join(job_folder, "frame_end.png")
    if os.path.isfile(first_b) and os.path.isfile(last_b):
        return first_b, last_b, "frame_start_end"

    # fail with clear info
    raise RuntimeError(
        "Missing input keyframes.\n"
        "Provide either:\n"
        "  - first.png AND last.png\n"
        "OR\n"
        "  - frame_start.png AND frame_end.png"
    )


_FRAME_RE = re.compile(r"^frame_(\d{4})\.png$", re.IGNORECASE)


def _list_frames(frames_dir: str) -> List[str]:
    if not os.path.isdir(frames_dir):
        return []
    frames = []
    for f in os.listdir(frames_dir):
        m = _FRAME_RE.match(f)
        if m:
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

    # optional strict sequential check (no gaps) if you want premium stability:
    # only check first ~300 to avoid huge overhead
    check_n = min(len(frames), 300)
    expected = 1
    for i in range(check_n):
        idx = int(_FRAME_RE.match(frames[i]).group(1))  # type: ignore[union-attr]
        if idx != expected:
            raise RuntimeError(
                f"Frame sequence gap or misnaming in {frames_dir}.\n"
                f"Expected frame_{expected:04d}.png but found {frames[i]}.\n"
                "Fix your VIDEO_BETWEEN_FRAMES_CMD output naming."
            )
        expected += 1

    return len(frames)


def _write_basic_video_viewer(job_folder: str, mp4_rel: str = "video_between.mp4") -> str:
    viewer_dir = os.path.join(job_folder, "viewer")
    _ensure_dir(viewer_dir)
    index_path = os.path.join(viewer_dir, "index.html")

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>RENDEREXPO AI STUDIO — Video Between Frames</title>
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
    <header><b>RENDEREXPO AI STUDIO</b> — Video Between Frames</header>
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
            "-crf", "16",        # high quality
            "-preset", "slow",   # better quality/compression
            out_mp4,
        ],
        cwd=job_folder,
    )


def _cmd_split(cmd: str) -> List[str]:
    """
    Use shlex.split so quoted args work (paths with spaces, etc).
    """
    return shlex.split(cmd, posix=True)


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def run_video_between_frames(job_folder: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    meta = _read_meta(job_folder) or meta

    first_path, last_path, naming_mode = _pick_input_pair(job_folder)

    # Runtime knobs
    vr = meta.get("video_runtime") if isinstance(meta.get("video_runtime"), dict) else {}
    fps = int(vr.get("fps", 24))
    seconds = float(vr.get("duration_seconds", 4.0))

    if fps < 8 or fps > 60:
        raise RuntimeError("video_runtime.fps must be between 8 and 60 for production stability.")
    if seconds < 1.0 or seconds > 12.0:
        raise RuntimeError("video_runtime.duration_seconds must be between 1.0 and 12.0.")

    # Output targets
    out_mp4_rel = "video_between.mp4"
    out_mp4_abs = os.path.join(job_folder, out_mp4_rel)

    frames_dir = os.path.join(job_folder, "frames")
    frames_ref_dir = os.path.join(job_folder, "frames_refined")

    # Meta start
    meta["status"] = "running"
    meta.setdefault("outputs", {})
    meta["outputs"]["video_between"] = out_mp4_rel
    meta["outputs"]["viewer_dir"] = "viewer/"
    meta["outputs"]["frames_dir"] = "frames/"
    meta.setdefault("video_between", {})
    meta["video_between"]["inputs"] = {
        "first": os.path.basename(first_path),
        "last": os.path.basename(last_path),
        "naming_mode": naming_mode,
    }
    meta["video_between"]["runtime"] = {"fps": fps, "duration_seconds": seconds}
    _write_meta(job_folder, meta)

    # -------------------------------------------------------------------------
    # Choose execution mode (REAL)
    # -------------------------------------------------------------------------

    mp4_cmd_tpl = os.getenv("VIDEO_BETWEEN_MP4_CMD", "").strip()
    frames_cmd_tpl = os.getenv("VIDEO_BETWEEN_FRAMES_CMD", "").strip()
    refine_cmd_tpl = os.getenv("VIDEO_BETWEEN_REFINE_CMD", "").strip()

    if not mp4_cmd_tpl and not frames_cmd_tpl:
        raise RuntimeError(
            "No GPU pipeline configured for between-frames video.\n\n"
            "Set ONE of:\n"
            "  - VIDEO_BETWEEN_MP4_CMD (must output mp4)\n"
            "  - VIDEO_BETWEEN_FRAMES_CMD (must output frames/)\n\n"
            "Example:\n"
            "  VIDEO_BETWEEN_FRAMES_CMD='python /workspace/pipelines/between_frames_frames.py "
            "--first {FIRST} --last {LAST} --outdir {FRAMES} --fps {FPS} --seconds {SECONDS}'"
        )

    # (1) Direct MP4 generator
    if mp4_cmd_tpl:
        cmd_str = mp4_cmd_tpl.format(
            FIRST=first_path,
            LAST=last_path,
            OUT=out_mp4_abs,
            FPS=fps,
            SECONDS=seconds,
            JOB=job_folder,
        )
        _run(_cmd_split(cmd_str), cwd=job_folder)

        if not os.path.isfile(out_mp4_abs) or os.path.getsize(out_mp4_abs) < 50_000:
            raise RuntimeError("MP4 generator finished but video_between.mp4 is missing or too small.")

        viewer = _write_basic_video_viewer(job_folder, mp4_rel=out_mp4_rel)

        meta = _read_meta(job_folder) or meta
        meta["status"] = "completed"
        meta["outputs"]["viewer_dir"] = "viewer/"
        meta["outputs"]["video_between"] = out_mp4_rel
        meta["outputs"]["preview_video"] = out_mp4_rel
        meta["video_between"]["method"] = "mp4_cmd"
        meta["video_between"]["viewer"] = viewer
        _write_meta(job_folder, meta)

        return {
            "mode": "between_frames",
            "method": "mp4_cmd",
            "video": out_mp4_rel,
            "viewer": viewer,
            "fps": fps,
            "seconds": seconds,
            "naming_mode": naming_mode,
        }

    # (2) Frames generator + optional refine + ffmpeg
    _ensure_dir(frames_dir)

    cmd_str = frames_cmd_tpl.format(
        FIRST=first_path,
        LAST=last_path,
        FRAMES=frames_dir,
        FPS=fps,
        SECONDS=seconds,
        JOB=job_folder,
    )
    _run(_cmd_split(cmd_str), cwd=job_folder)

    # Require at least ~1 second of content, but also minimum 12 frames for stability
    min_frames = max(12, int(fps))
    n = _validate_frame_sequence(frames_dir, min_frames=min_frames)

    frames_for_encode = frames_dir

    # Optional refinement stage (premium realism / temporal consistency)
    if refine_cmd_tpl:
        _ensure_dir(frames_ref_dir)
        rcmd_str = refine_cmd_tpl.format(
            FRAMES=frames_dir,
            FRAMES_REF=frames_ref_dir,
            FPS=fps,
            SECONDS=seconds,
            JOB=job_folder,
        )
        _run(_cmd_split(rcmd_str), cwd=job_folder)

        # Only use refined frames if they are at least as complete
        n_ref = len(_list_frames(frames_ref_dir))
        if n_ref >= n:
            # If refine produced files but with gaps, fail (REAL)
            _validate_frame_sequence(frames_ref_dir, min_frames=min_frames)
            frames_for_encode = frames_ref_dir

    # Encode MP4
    _assemble_mp4_from_frames(job_folder, frames_for_encode, fps=fps, out_mp4=out_mp4_abs)

    if not os.path.isfile(out_mp4_abs) or os.path.getsize(out_mp4_abs) < 50_000:
        raise RuntimeError("ffmpeg finished but video_between.mp4 is missing or too small.")

    viewer = _write_basic_video_viewer(job_folder, mp4_rel=out_mp4_rel)

    meta = _read_meta(job_folder) or meta
    meta["status"] = "completed"
    meta["outputs"]["viewer_dir"] = "viewer/"
    meta["outputs"]["video_between"] = out_mp4_rel
    meta["outputs"]["preview_video"] = out_mp4_rel
    meta["video_between"]["method"] = "frames_cmd+ffmpeg" + ("+refine" if refine_cmd_tpl else "")
    meta["video_between"]["viewer"] = viewer
    meta["video_between"]["frames_used_for_encode"] = "frames_refined/" if frames_for_encode == frames_ref_dir else "frames/"
    meta["video_between"]["frames_count"] = len(_list_frames(frames_for_encode))
    _write_meta(job_folder, meta)

    return {
        "mode": "between_frames",
        "method": meta["video_between"]["method"],
        "video": out_mp4_rel,
        "viewer": viewer,
        "fps": fps,
        "seconds": seconds,
        "frames_dir": "frames/",
        "frames_used_for_encode": meta["video_between"]["frames_used_for_encode"],
        "frames_count": meta["video_between"]["frames_count"],
        "naming_mode": naming_mode,
    }
