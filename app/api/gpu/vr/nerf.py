# app/gpu/vr/nerf.py
"""
RENDEREXPO AI STUDIO - VR NeRF (REAL)

Highest photorealism VR pipeline.

Pipeline:
1. COLMAP SfM (camera poses)
2. NeRF training (Instant-NGP / Nerfstudio / custom)
3. Viewer export
4. Optional preview video

Expected inputs:
- view_001.png/jpg/jpeg, view_002.png/jpg/jpeg, ... (3+)

Expected outputs:
- nerf_model/           (trained model + configs)
- viewer/index.html     (viewer shell)
- preview.mp4           (optional)

Env vars (GPU POD):
- NERF_COLMAP_BIN="colmap"
- NERF_TRAIN_CMD="ns-train nerfacto --data {DATA} --output-dir {OUT}"
  OR
  NERF_TRAIN_CMD="python train.py --data {DATA} --out {OUT}"
- NERF_FFMPEG_BIN="ffmpeg"
"""

from __future__ import annotations

import os
import json
import shutil
import subprocess
from typing import Dict, Any, List, Optional


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
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"STDOUT:\n{(p.stdout or '')[-2000:]}\n"
            f"STDERR:\n{(p.stderr or '')[-2000:]}"
        )


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _list_views(job_folder: str) -> List[str]:
    """
    Accept PNG/JPG/JPEG inputs. Planner currently saves as PNG, but we keep this robust.
    """
    return sorted(
        f for f in os.listdir(job_folder)
        if f.lower().startswith("view_") and f.lower().endswith((".png", ".jpg", ".jpeg"))
    )


def _write_viewer(job_folder: str) -> str:
    viewer_dir = os.path.join(job_folder, "viewer")
    _ensure_dir(viewer_dir)
    index_path = os.path.join(viewer_dir, "index.html")

    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>RENDEREXPO VR Viewer (NeRF)</title>
  <style>
    html,body{margin:0;height:100%;background:#000;color:#fff;font-family:system-ui;}
    header{padding:10px;background:#111;border-bottom:1px solid #222;}
    main{padding:24px;color:#bbb;}
    code{color:#fff}
  </style>
</head>
<body>
  <header><strong>RENDEREXPO AI STUDIO</strong> — NeRF Viewer</header>
  <main>
    <p>NeRF model generated successfully.</p>
    <p>Model directory: <code>../nerf_model/</code></p>
    <p>Plug this into your preferred NeRF Web viewer (Nerfstudio / Instant-NGP / custom).</p>
  </main>
</body>
</html>
"""
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html)

    return "viewer/index.html"


def _maybe_make_preview(job_folder: str, enabled: bool) -> Optional[str]:
    if not enabled:
        return None

    ffmpeg = os.getenv("NERF_FFMPEG_BIN", "ffmpeg")
    if not _which(ffmpeg):
        return None

    frames_dir = os.path.join(job_folder, "frames")
    if not os.path.isdir(frames_dir):
        return None

    out = os.path.join(job_folder, "preview.mp4")
    _run([
        ffmpeg, "-y",
        "-framerate", "30",
        "-i", os.path.join(frames_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        out,
    ])
    return "preview.mp4"


def _colmap_feature_extractor(colmap: str, db: str, images_dir: str) -> List[str]:
    """
    Consistency improvement for typical same-device multi-view captures:
    - ImageReader.single_camera=1 reduces intrinsics drift and pose instability.
    """
    return [
        colmap, "feature_extractor",
        "--database_path", db,
        "--image_path", images_dir,
        "--ImageReader.single_camera", "1",
    ]


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def run_nerf(job_folder: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    meta = _read_meta(job_folder) or meta
    views = _list_views(job_folder)
    if len(views) < 3:
        raise RuntimeError("NeRF requires at least 3 input images (view_*.png/jpg).")

    # Setup dirs
    work_dir = os.path.join(job_folder, "nerf_work")
    images_dir = os.path.join(work_dir, "images")
    sparse_dir = os.path.join(work_dir, "sparse")
    nerf_out = os.path.join(job_folder, "nerf_model")
    _ensure_dir(images_dir)
    _ensure_dir(sparse_dir)
    _ensure_dir(nerf_out)

    # Copy images (idempotent)
    for v in views:
        src = os.path.join(job_folder, v)
        dst = os.path.join(images_dir, v)
        if not os.path.isfile(dst):
            shutil.copy2(src, dst)

    # Tools
    colmap = os.getenv("NERF_COLMAP_BIN", "colmap")
    if not _which(colmap):
        raise RuntimeError("COLMAP not found. Install COLMAP on the GPU worker (or set NERF_COLMAP_BIN).")

    train_cmd_tpl = os.getenv("NERF_TRAIN_CMD")
    if not train_cmd_tpl:
        raise RuntimeError(
            "NERF_TRAIN_CMD not set. "
            "Example: NERF_TRAIN_CMD='ns-train nerfacto --data {DATA} --output-dir {OUT}'"
        )

    # Update meta
    meta["status"] = "running"
    meta["vr_mode"] = "nerf"
    meta.setdefault("outputs", {})
    meta["outputs"]["nerf_dir"] = "nerf_model/"
    _write_meta(job_folder, meta)

    db = os.path.join(work_dir, "database.db")

    # COLMAP SfM
    _run(_colmap_feature_extractor(colmap, db, images_dir))
    _run([colmap, "exhaustive_matcher", "--database_path", db])
    _run([colmap, "mapper", "--database_path", db, "--image_path", images_dir, "--output_path", sparse_dir])

    # NeRF train
    # We keep simple .split() (matches your current style); if you later need quoted args,
    # we can switch to shlex.split across all runners consistently.
    train_cmd = train_cmd_tpl.format(DATA=work_dir, OUT=nerf_out).split()
    _run(train_cmd, cwd=job_folder)

    if not os.listdir(nerf_out):
        raise RuntimeError("NeRF training finished but nerf_model/ is empty.")

    viewer = _write_viewer(job_folder)
    preview = _maybe_make_preview(job_folder, enabled=bool(meta.get("vr_runtime", {}).get("preview_video", True)))

    meta = _read_meta(job_folder) or meta
    meta["status"] = "completed"
    meta.setdefault("outputs", {})
    meta["outputs"]["viewer_dir"] = "viewer/"
    meta["outputs"]["preview_video"] = preview
    _write_meta(job_folder, meta)

    return {
        "mode": "nerf",
        "artifact_dir": "nerf_model/",
        "viewer": viewer,
        "preview_video": preview,
        "views_used": len(views),
    }
