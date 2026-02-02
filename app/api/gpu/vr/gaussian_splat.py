# app/gpu/vr/gaussian_splat.py
"""
RENDEREXPO AI STUDIO - VR Gaussian Splat (REAL, Strict)

This module executes a REAL Gaussian Splat VR pipeline.

Pipeline (REAL):
1) COLMAP SfM (camera poses)
2) 3DGS training (your chosen repo/tool via GS_TRAIN_CMD)
3) Export scene.splat (your chosen exporter via GS_EXPORT_CMD)
4) Viewer export (real Web viewer that loads scene.splat)
5) Optional preview video (if frames/ exist + ffmpeg exists)

Expected inputs (job_folder):
- view_001.png, view_002.png, ... (3+)

Expected outputs (job_folder):
- scene.splat
- viewer/index.html
- preview.mp4 (optional)

CRITICAL:
- This file does NOT "fake success".
- If required tools/commands are not installed, it raises actionable errors.

ENV (GPU POD):
- GS_COLMAP_BIN="colmap"
- GS_TRAIN_CMD="python /workspace/3dgs/train.py --data {DATA} --out {OUTDIR}"
- GS_EXPORT_CMD="python /workspace/3dgs/export_splat.py --model {OUTDIR} --out {OUTSPLAT}"

Optional:
- GS_FFMPEG_BIN="ffmpeg"

Notes:
- We keep this command-driven so you can swap repos freely (Inria 3DGS, gsplat, nerfstudio splat, etc.).
- You will set the exact repo later on the POD (after push/pull), then define GS_TRAIN_CMD + GS_EXPORT_CMD accordingly.
"""

from __future__ import annotations

import os
import json
import shutil
import subprocess
from typing import Any, Dict, List, Optional


# -----------------------------------------------------------------------------
# Meta helpers
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


def _safe_set(meta: Dict[str, Any], path: str, value: Any) -> None:
    """
    Set nested dict value using dotted path: "dispatch.result"
    """
    keys = path.split(".")
    cur: Dict[str, Any] = meta
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]  # type: ignore[assignment]
    cur[keys[-1]] = value


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _list_views(job_folder: str) -> List[str]:
    return sorted(
        f for f in os.listdir(job_folder)
        if f.lower().startswith("view_") and f.lower().endswith(".png")
    )


def _run(cmd: List[str], cwd: Optional[str] = None) -> None:
    """
    Run a subprocess and fail loudly with truncated logs.
    """
    p = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if p.returncode != 0:
        out = (p.stdout or "")[-3000:]
        err = (p.stderr or "")[-3000:]
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n"
            f"STDOUT (tail):\n{out}\n\nSTDERR (tail):\n{err}"
        )


def _split_cmd(cmd_str: str) -> List[str]:
    """
    Minimal, predictable splitting.
    IMPORTANT: keep your GS_*_CMD values simple (no fancy shell pipes).
    If you need quoting/complex shell, wrap with: bash -lc "...."
    """
    return cmd_str.strip().split()


def _require_env(name: str) -> str:
    v = (os.getenv(name) or "").strip()
    if not v:
        raise RuntimeError(
            f"{name} is not set. This pipeline is REAL and requires it.\n"
            f"Set {name} on the GPU POD worker.\n"
            f"Example:\n"
            f'  {name}="python /workspace/3dgs/train.py --data {{DATA}} --out {{OUTDIR}}"\n'
        )
    return v


# -----------------------------------------------------------------------------
# Viewer + preview
# -----------------------------------------------------------------------------

def _write_splat_viewer(job_folder: str) -> str:
    """
    Writes a REAL viewer page that loads ../scene.splat using gaussian-splats-3d.
    This gives an actual browser viewing experience once the .splat is present.
    """
    viewer_dir = os.path.join(job_folder, "viewer")
    _ensure_dir(viewer_dir)
    index_path = os.path.join(viewer_dir, "index.html")

    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>RENDEREXPO AI STUDIO — Gaussian Splat Viewer</title>
  <style>
    html,body{margin:0;height:100%;background:#000;overflow:hidden;}
    #hud{
      position:absolute;top:12px;left:12px;z-index:10;
      font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;
      color:#fff;background:rgba(0,0,0,.45);
      padding:10px 12px;border-radius:12px;max-width:560px;
      backdrop-filter: blur(6px);
    }
    #hud .small{margin-top:6px;font-size:13px;opacity:.88;line-height:1.35;}
    a{color:#9fd3ff}
  </style>
</head>
<body>
  <div id="hud">
    <div><b>RENDEREXPO AI STUDIO</b> — Gaussian Splat Viewer</div>
    <div class="small">
      Loading: <code>../scene.splat</code><br/>
      If it fails, confirm <code>scene.splat</code> exists next to this viewer.<br/>
      Controls: mouse orbit / zoom (trackpad supported).
    </div>
  </div>

  <script type="module">
    import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.161.0/build/three.module.js";
    import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.161.0/examples/jsm/controls/OrbitControls.js";
    // gaussian-splats-3d ESM build via jsdelivr
    import { GaussianSplats3D } from "https://cdn.jsdelivr.net/npm/gaussian-splats-3d@0.4.7/dist/gaussian-splats-3d.module.js";

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.01, 2000);
    camera.position.set(0.7, 0.4, 1.2);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    document.body.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // Optional lights (splats are usually emissive-ish, but keep subtle)
    const hemi = new THREE.HemisphereLight(0xffffff, 0x111111, 0.35);
    scene.add(hemi);

    const splats = new GaussianSplats3D.GaussianSplattingMesh({
      // Conservative defaults; you can tune later after we see quality
      // You can also expose these via meta in the future.
      // maxSplatCount: 2000000,
    });

    scene.add(splats);

    async function load() {
      try {
        await splats.load("../scene.splat");
      } catch (e) {
        console.error(e);
        const hud = document.getElementById("hud");
        hud.innerHTML += '<div class="small" style="margin-top:8px;color:#ffb3b3;">Failed to load scene.splat.</div>';
      }
    }
    load();

    window.addEventListener("resize", () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });

    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();
  </script>
</body>
</html>
"""
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html)

    return "viewer/index.html"


def _maybe_make_preview(job_folder: str, enabled: bool) -> Optional[str]:
    """
    If frames exist in job_folder/frames/frame_%04d.png and ffmpeg exists,
    render preview.mp4. Otherwise skip (no fake output).
    """
    if not enabled:
        return None

    ffmpeg = (os.getenv("GS_FFMPEG_BIN") or "ffmpeg").strip()
    if not _which(ffmpeg):
        return None

    frames_dir = os.path.join(job_folder, "frames")
    if not os.path.isdir(frames_dir):
        return None

    out_mp4 = os.path.join(job_folder, "preview.mp4")
    cmd = [
        ffmpeg, "-y",
        "-framerate", "30",
        "-i", os.path.join(frames_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        out_mp4,
    ]
    _run(cmd)
    return "preview.mp4"


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def run_gaussian_splat(job_folder: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute REAL Gaussian Splat pipeline.
    Returns a result dict persisted by dispatcher into meta["dispatch"]["result"].
    """
    meta = _read_meta(job_folder) or meta
    views = _list_views(job_folder)
    if len(views) < 3:
        raise RuntimeError("Gaussian Splat requires at least 3 input views: view_*.png")

    vr_runtime = meta.get("vr_runtime") if isinstance(meta.get("vr_runtime"), dict) else {}
    preview_video = bool((vr_runtime or {}).get("preview_video", True))

    # --- Work dirs
    work_dir = os.path.join(job_folder, "gs_work")
    images_dir = os.path.join(work_dir, "images")
    sparse_dir = os.path.join(work_dir, "sparse")
    _ensure_dir(images_dir)
    _ensure_dir(sparse_dir)

    # Copy views into work/images
    for v in views:
        src = os.path.join(job_folder, v)
        dst = os.path.join(images_dir, v)
        if not os.path.isfile(dst):
            shutil.copy2(src, dst)

    # --- Tools / Commands
    colmap = (os.getenv("GS_COLMAP_BIN") or "colmap").strip()
    if not _which(colmap):
        raise RuntimeError(
            "COLMAP not found on GPU worker. Install it on the POD or set GS_COLMAP_BIN to its path."
        )

    train_cmd_tpl = _require_env("GS_TRAIN_CMD")
    export_cmd_tpl = _require_env("GS_EXPORT_CMD")

    # Update meta status early
    meta["status"] = "running"
    meta["vr_mode"] = "gaussian_splat"
    meta.setdefault("outputs", {})
    meta["outputs"]["viewer_dir"] = "viewer/"
    meta["outputs"]["gaussian_splat"] = "scene.splat"
    _write_meta(job_folder, meta)

    # --- 1) COLMAP SfM
    db_path = os.path.join(work_dir, "database.db")

    _run([
        colmap, "feature_extractor",
        "--database_path", db_path,
        "--image_path", images_dir,
        "--ImageReader.single_camera", "1",
    ], cwd=work_dir)

    _run([
        colmap, "exhaustive_matcher",
        "--database_path", db_path,
    ], cwd=work_dir)

    _run([
        colmap, "mapper",
        "--database_path", db_path,
        "--image_path", images_dir,
        "--output_path", sparse_dir,
    ], cwd=work_dir)

    # --- 2) Train splats (repo-specific; YOU define the command)
    # Contract: training writes into OUTDIR (folder).
    out_dir = os.path.join(job_folder, "gs_model")
    _ensure_dir(out_dir)

    train_cmd = train_cmd_tpl.format(DATA=work_dir, OUTDIR=out_dir)
    train_tokens = _split_cmd(train_cmd)
    _run(train_tokens, cwd=job_folder)

    # Sanity: OUTDIR should not be empty
    if not os.listdir(out_dir):
        raise RuntimeError("GS_TRAIN_CMD finished but gs_model/ is empty. Check your trainer command.")

    # --- 3) Export scene.splat (repo-specific)
    out_splat = os.path.join(job_folder, "scene.splat")
    export_cmd = export_cmd_tpl.format(OUTDIR=out_dir, OUTSPLAT=out_splat)
    export_tokens = _split_cmd(export_cmd)
    _run(export_tokens, cwd=job_folder)

    if not os.path.isfile(out_splat) or os.path.getsize(out_splat) < 50_000:
        raise RuntimeError(
            "Export finished but scene.splat is missing or too small. "
            "Check GS_EXPORT_CMD; it must write a valid .splat file."
        )

    # --- 4) Viewer
    viewer_index = _write_splat_viewer(job_folder)

    # --- 5) Optional preview video
    preview_out = _maybe_make_preview(job_folder, enabled=preview_video)

    # Finalize meta
    meta = _read_meta(job_folder) or meta
    meta["status"] = "completed"
    meta.setdefault("outputs", {})
    meta["outputs"]["viewer_dir"] = "viewer/"
    meta["outputs"]["gaussian_splat"] = "scene.splat"
    meta["outputs"]["preview_video"] = preview_out
    _write_meta(job_folder, meta)

    return {
        "mode": "gaussian_splat",
        "artifact": "scene.splat",
        "viewer": viewer_index,
        "preview_video": preview_out,
        "views_used": len(views),
        "work_dir": "gs_work/",
        "model_dir": "gs_model/",
        "colmap_used": colmap,
        "train_cmd_used": train_cmd_tpl,
        "export_cmd_used": export_cmd_tpl,
    }
