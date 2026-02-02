# app/gpu/vr/mesh.py
"""
RENDEREXPO AI STUDIO - VR Mesh (REAL)

Editable geometry export pipeline (GLB), good for:
- downstream editing (Blender, Unreal, Twinmotion)
- CAD-ish workflows
- "real model" delivery beyond walkthrough

Pipeline (REAL, strict):
1) COLMAP SfM (camera poses)
2) Depth per view (your chosen backend via env command)
3) Fusion + meshing (your chosen backend via env command) -> scene_mesh.glb
4) Viewer export (Three.js GLB viewer)
5) Optional preview video (if your fusion stage renders frames OR you provide frames/)

IMPORTANT:
- This file DOES NOT pretend to run if tools are missing.
- It will fail loudly with actionable errors.

Expected inputs (job_folder):
- view_001.png/jpg/jpeg, view_002.png/jpg/jpeg, ... (3+)

Expected outputs (job_folder):
- scene_mesh.glb
- viewer/index.html
- preview.mp4 (optional)

ENV (GPU POD):
- MESH_COLMAP_BIN="colmap"

Depth stage (choose one command template you implement on the POD):
- MESH_DEPTH_CMD="python app/gpu/vr/tools/midas_depth.py --in {IN} --out {OUT}"
  (must write a depth file per input view)

Fusion/Mesh stage (choose one command template you implement on the POD):
- MESH_FUSE_CMD="python app/gpu/vr/tools/fuse_to_mesh.py --work {WORK} --out {OUT}"
  (must output a GLB at {OUT})

Optional:
- MESH_FFMPEG_BIN="ffmpeg"
- If you want preview.mp4:
  create frames in {JOB}/frames/frame_0001.png ... then ffmpeg will assemble.

Notes:
- This is deliberately "command-driven" so you can swap implementations:
  Open3D TSDF, nerfstudio mesh export, custom photogrammetry mesher, etc.
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
    Accept PNG/JPG/JPEG inputs. The planner saves as PNG, but users sometimes
    upload JPGs and some upstream code might preserve extensions.
    """
    return sorted(
        f for f in os.listdir(job_folder)
        if f.lower().startswith("view_") and f.lower().endswith((".png", ".jpg", ".jpeg"))
    )


def _write_glb_viewer(job_folder: str, glb_rel: str = "scene_mesh.glb") -> str:
    """
    Writes a minimal Three.js viewer that loads ../scene_mesh.glb.
    This is a REAL viewer artifact you can serve from /outputs/.../viewer/index.html
    """
    viewer_dir = os.path.join(job_folder, "viewer")
    _ensure_dir(viewer_dir)
    index_path = os.path.join(viewer_dir, "index.html")

    # Using CDN imports (simple, works in most environments).
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>RENDEREXPO VR Viewer (Mesh)</title>
  <style>
    html, body {{ margin:0; height:100%; overflow:hidden; background:#000; }}
    #hud {{
      position: absolute; top: 10px; left: 10px;
      color: #fff; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial;
      background: rgba(0,0,0,0.45); padding: 10px 12px; border-radius: 10px;
      max-width: 520px;
    }}
    a {{ color: #9fd3ff; }}
    code {{ color:#fff; }}
  </style>
</head>
<body>
  <div id="hud">
    <div><b>RENDEREXPO AI STUDIO</b> — Mesh Viewer</div>
    <div style="margin-top:6px;font-size:13px;opacity:.9;">
      Mouse to orbit. If it doesn’t load, confirm:
      <code>{glb_rel}</code> exists next to this viewer.
    </div>
  </div>

  <script type="module">
    import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.161.0/build/three.module.js';
    import {{ OrbitControls }} from 'https://cdn.jsdelivr.net/npm/three@0.161.0/examples/jsm/controls/OrbitControls.js';
    import {{ GLTFLoader }} from 'https://cdn.jsdelivr.net/npm/three@0.161.0/examples/jsm/loaders/GLTFLoader.js';

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.01, 2000);
    camera.position.set(1.8, 1.2, 2.8);

    const renderer = new THREE.WebGLRenderer({{ antialias:true }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    document.body.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    const hemi = new THREE.HemisphereLight(0xffffff, 0x222222, 1.0);
    scene.add(hemi);

    const dir = new THREE.DirectionalLight(0xffffff, 1.1);
    dir.position.set(4, 6, 3);
    scene.add(dir);

    const loader = new GLTFLoader();
    loader.load(
      '../{glb_rel}',
      (gltf) => {{
        const obj = gltf.scene;
        scene.add(obj);

        // Auto frame
        const box = new THREE.Box3().setFromObject(obj);
        const size = box.getSize(new THREE.Vector3()).length();
        const center = box.getCenter(new THREE.Vector3());
        controls.target.copy(center);

        camera.near = size / 1000;
        camera.far = size * 10;
        camera.updateProjectionMatrix();

        camera.position.copy(center);
        camera.position.x += size * 0.35;
        camera.position.y += size * 0.18;
        camera.position.z += size * 0.55;
      }},
      undefined,
      (err) => {{
        console.error(err);
        document.getElementById('hud').innerHTML += '<div style="margin-top:8px;color:#ffb3b3;">Failed to load GLB.</div>';
      }}
    );

    window.addEventListener('resize', () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }}
    animate();
  </script>
</body>
</html>
"""
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html)

    return "viewer/index.html"


def _maybe_make_preview(job_folder: str, enabled: bool) -> Optional[str]:
    if not enabled:
        return None

    ffmpeg = os.getenv("MESH_FFMPEG_BIN", "ffmpeg")
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
    Consistency improvement: single_camera=1 for typical phone/DSLR same-intrinsics
    multi-view captures. This reduces pose instability.
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

def run_mesh(job_folder: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    meta = _read_meta(job_folder) or meta
    views = _list_views(job_folder)
    if len(views) < 3:
        raise RuntimeError("Mesh reconstruction requires at least 3 input images (view_*.png/jpg).")

    # Dirs
    work_dir = os.path.join(job_folder, "mesh_work")
    images_dir = os.path.join(work_dir, "images")
    sparse_dir = os.path.join(work_dir, "sparse")
    depth_dir = os.path.join(work_dir, "depth")
    _ensure_dir(images_dir)
    _ensure_dir(sparse_dir)
    _ensure_dir(depth_dir)

    # Copy inputs for consistent tooling
    for v in views:
        shutil.copy2(os.path.join(job_folder, v), os.path.join(images_dir, v))

    # Tools
    colmap = os.getenv("MESH_COLMAP_BIN", "colmap")
    if not _which(colmap):
        raise RuntimeError("COLMAP not found. Install COLMAP on the GPU worker (or set MESH_COLMAP_BIN).")

    depth_cmd_tpl = os.getenv("MESH_DEPTH_CMD")
    if not depth_cmd_tpl:
        raise RuntimeError(
            "MESH_DEPTH_CMD not set. "
            "Example: MESH_DEPTH_CMD='python app/gpu/vr/tools/midas_depth.py --in {IN} --out {OUT}'"
        )

    fuse_cmd_tpl = os.getenv("MESH_FUSE_CMD")
    if not fuse_cmd_tpl:
        raise RuntimeError(
            "MESH_FUSE_CMD not set. "
            "Example: MESH_FUSE_CMD='python app/gpu/vr/tools/fuse_to_mesh.py --work {WORK} --out {OUT}'"
        )

    # Meta begin
    meta["status"] = "running"
    meta["vr_mode"] = "mesh"
    meta.setdefault("outputs", {})
    meta["outputs"]["mesh_glb"] = "scene_mesh.glb"
    _write_meta(job_folder, meta)

    # COLMAP SfM
    db = os.path.join(work_dir, "database.db")
    _run(_colmap_feature_extractor(colmap, db, images_dir))
    _run([colmap, "exhaustive_matcher", "--database_path", db])
    _run([colmap, "mapper", "--database_path", db, "--image_path", images_dir, "--output_path", sparse_dir])

    # Depth per view (always output PNG depth maps for consistency)
    for v in views:
        in_path = os.path.join(images_dir, v)
        base, _ext = os.path.splitext(v)
        out_path = os.path.join(depth_dir, f"{base}_depth.png")

        cmd = depth_cmd_tpl.format(IN=in_path, OUT=out_path).split()
        _run(cmd, cwd=job_folder)

        if not os.path.isfile(out_path):
            raise RuntimeError(f"Depth stage did not produce expected file: {out_path}")

    # Fusion / Meshing -> GLB
    glb_out = os.path.join(job_folder, "scene_mesh.glb")
    cmd = fuse_cmd_tpl.format(WORK=work_dir, OUT=glb_out).split()
    _run(cmd, cwd=job_folder)

    if not os.path.isfile(glb_out) or os.path.getsize(glb_out) < 10_000:
        raise RuntimeError("Mesh stage failed: scene_mesh.glb missing or too small.")

    viewer = _write_glb_viewer(job_folder, glb_rel="scene_mesh.glb")
    preview = _maybe_make_preview(job_folder, enabled=bool(meta.get("vr_runtime", {}).get("preview_video", True)))

    meta = _read_meta(job_folder) or meta
    meta["status"] = "completed"
    meta.setdefault("outputs", {})
    meta["outputs"]["viewer_dir"] = "viewer/"
    meta["outputs"]["preview_video"] = preview
    _write_meta(job_folder, meta)

    return {
        "mode": "mesh",
        "artifact": "scene_mesh.glb",
        "viewer": viewer,
        "preview_video": preview,
        "views_used": len(views),
    }
