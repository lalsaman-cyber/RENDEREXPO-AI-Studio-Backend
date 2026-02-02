# app/gpu/mesh/from_image.py
"""
RENDEREXPO AI STUDIO - Mesh From Image (REAL)

Runs inside GPU worker. Produces REAL geometry.

Inputs (job_folder):
- image.png

Outputs (job_folder):
- depth.png (debug)
- preview.png (debug shaded render)
- mesh.obj (required)
- mesh.glb (optional; if trimesh GLB export works)
- meta.json updated

Dependencies (GPU env):
- torch
- opencv-python
- numpy

Optional (recommended):
- trimesh (for GLB export + decimation)

Notes:
- This is REAL: if it cannot write mesh.obj, it raises an error.
- MiDaS is loaded via torch.hub; if you want offline behavior, pre-cache MiDaS.
"""

from __future__ import annotations

import os
import json
import time
import traceback
from typing import Any, Dict, Tuple, Optional

import numpy as np

try:
    import cv2  # type: ignore
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("Missing dependency: opencv-python") from exc

try:
    import torch  # type: ignore
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("Missing dependency: torch (required for depth model).") from exc


# ----------------------------
# Meta helpers
# ----------------------------

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


def _safe_set(meta: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur: Dict[str, Any] = meta
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]  # type: ignore[assignment]
    cur[keys[-1]] = value


def _now_epoch() -> int:
    return int(time.time())


# ----------------------------
# IO
# ----------------------------

def _image_path(job_folder: str) -> str:
    p = os.path.join(job_folder, "image.png")
    if not os.path.isfile(p):
        raise RuntimeError("mesh_from_image requires image.png in job_folder.")
    return p


# ----------------------------
# Depth model (MiDaS via torch.hub)
# ----------------------------

def _load_midas(device: str):
    """
    REAL depth inference using MiDaS.
    NOTE: torch.hub may download if not cached.
    """
    model_type = os.getenv("MESH_DEPTH_MODEL", "DPT_Hybrid").strip()

    # torch.hub downloads are REAL. For strict offline:
    # - pre-cache once
    # - set TORCH_HOME to stable cache folder
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type in ("DPT_Large", "DPT_Hybrid"):
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    model.to(device)
    return model, transform, model_type


@torch.inference_mode()
def _predict_depth_bgr(img_bgr: np.ndarray, device: str) -> Tuple[np.ndarray, str]:
    model, transform, model_type = _load_midas(device)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    prediction = model(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth = prediction.detach().float().cpu().numpy()

    # Normalize depth to 0..1 (robust)
    d_min = float(np.percentile(depth, 1.0))
    d_max = float(np.percentile(depth, 99.0))
    depth01 = np.clip((depth - d_min) / max(d_max - d_min, 1e-6), 0.0, 1.0)

    return depth01, model_type


# ----------------------------
# Mesh reconstruction (grid mesh)
# ----------------------------

def _detail_to_downscale(detail_level: str) -> int:
    # downscale factor (lower = more detail)
    if detail_level == "high":
        return 1
    if detail_level == "low":
        return 4
    return 2  # medium


def _build_grid_mesh(depth01: np.ndarray, detail_level: str, max_depth_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct a triangulated grid mesh from depth map (REAL).
    Returns:
      verts: (N,3) float32
      faces: (M,3) int32
    """
    h, w = depth01.shape
    ds = _detail_to_downscale(detail_level)

    if ds > 1:
        depth01_ds = cv2.resize(depth01, (max(2, w // ds), max(2, h // ds)), interpolation=cv2.INTER_AREA)
    else:
        depth01_ds = depth01

    h2, w2 = depth01_ds.shape

    # Convert depth 0..1 into meters (relative)
    z = (1.0 - depth01_ds).astype(np.float32) * float(max_depth_m)

    xs = np.linspace(0.0, 1.0, w2, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, h2, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    # Center XY around 0 for nicer mesh
    x = (xx - 0.5).astype(np.float32)
    y = (0.5 - yy).astype(np.float32)  # flip Y

    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)

    faces: list[tuple[int, int, int]] = []

    def vid(r: int, c: int) -> int:
        return r * w2 + c

    for r in range(h2 - 1):
        for c in range(w2 - 1):
            v0 = vid(r, c)
            v1 = vid(r, c + 1)
            v2 = vid(r + 1, c)
            v3 = vid(r + 1, c + 1)
            faces.append((v0, v2, v1))
            faces.append((v1, v2, v3))

    faces_np = np.array(faces, dtype=np.int32)
    return verts, faces_np


def _compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute per-vertex normals (REAL) for nicer previews + GLB export.
    """
    v = verts.astype(np.float32)
    f = faces.astype(np.int32)

    normals = np.zeros_like(v, dtype=np.float32)

    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)  # face normals (unnormalized)

    # Accumulate to vertices
    for i in range(3):
        normals[f[:, i]] += fn

    # Normalize
    nrm = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    normals = normals / nrm
    return normals.astype(np.float32)


def _write_obj(obj_path: str, verts: np.ndarray, faces: np.ndarray, normals: Optional[np.ndarray] = None) -> None:
    """
    Write OBJ with vertices + optional normals.
    """
    with open(obj_path, "w", encoding="utf-8") as f:
        f.write("# RENDEREXPO AI STUDIO - mesh_from_image (REAL)\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        if normals is not None and len(normals) == len(verts):
            for n in normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

        # Faces (OBJ 1-indexed)
        if normals is not None and len(normals) == len(verts):
            for tri in faces:
                a, b, c = tri.tolist()
                f.write(f"f {a+1}//{a+1} {b+1}//{b+1} {c+1}//{c+1}\n")
        else:
            for tri in faces:
                a, b, c = tri.tolist()
                f.write(f"f {a+1} {b+1} {c+1}\n")


def _try_decimate(verts: np.ndarray, faces: np.ndarray, target_faces: int) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    REAL decimation (best-effort) using trimesh if available.
    If not available, returns original geometry with a note.
    """
    if target_faces <= 0:
        return verts, faces, "target_faces<=0; no decimation."

    if faces.shape[0] <= target_faces:
        return verts, faces, "No decimation needed (already <= target_faces)."

    try:
        import trimesh  # type: ignore
    except Exception:
        return verts, faces, "trimesh not installed; skipping decimation."

    try:
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        if hasattr(tm, "simplify_quadratic_decimation"):
            tm2 = tm.simplify_quadratic_decimation(int(target_faces))
            v2 = np.asarray(tm2.vertices, dtype=np.float32)
            f2 = np.asarray(tm2.faces, dtype=np.int32)
            if v2.size > 0 and f2.size > 0:
                return v2, f2, "Decimated via trimesh.simplify_quadratic_decimation."
            return verts, faces, "Decimation attempted but produced empty mesh; kept original."
        return verts, faces, "trimesh installed but simplify_quadratic_decimation not available; kept original."
    except Exception as exc:  # noqa: BLE001
        return verts, faces, f"Decimation failed; kept original. Error: {exc}"


def _try_export_glb(glb_path: str, verts: np.ndarray, faces: np.ndarray, normals: Optional[np.ndarray]) -> Tuple[bool, str]:
    """
    REAL GLB export (best-effort). If trimesh missing or export fails, we do NOT error.
    """
    try:
        import trimesh  # type: ignore
    except Exception:
        return False, "trimesh not installed; skipping GLB export."

    try:
        tm = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)
        tm.export(glb_path)
        if os.path.isfile(glb_path):
            return True, "GLB exported via trimesh."
        return False, "trimesh export did not write file."
    except Exception as exc:  # noqa: BLE001
        return False, f"GLB export failed: {exc}"


def _write_depth_png(depth01: np.ndarray, path: str) -> None:
    d = (depth01 * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(path, d)


def _write_preview_png(depth01: np.ndarray, path: str) -> None:
    # simple shaded preview from depth
    d = depth01.astype(np.float32)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    n = np.dstack([-gx, -gy, np.ones_like(d, dtype=np.float32)])
    norm = np.linalg.norm(n, axis=2, keepdims=True) + 1e-6
    n = n / norm
    l = np.array([0.4, 0.3, 0.85], dtype=np.float32)
    l = l / (np.linalg.norm(l) + 1e-6)
    shade = (n @ l).clip(0, 1)
    img = (shade * 255.0).astype(np.uint8)
    cv2.imwrite(path, img)


# ----------------------------
# Public runner
# ----------------------------

def run_mesh_from_image(job_folder: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point called by GPU dispatcher.
    Must never fake outputs. If it can't write OBJ, it errors.
    """
    meta_disk = _read_meta(job_folder)
    if isinstance(meta_disk, dict) and meta_disk:
        meta = meta_disk

    try:
        _safe_set(meta, "mesh_runtime.started_at_epoch", _now_epoch())
        _write_meta(job_folder, meta)

        img_path = _image_path(job_folder)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("cv2.imread failed for image.png")

        mesh_rt = meta.get("mesh_runtime") or {}

        detail = str(mesh_rt.get("detail_level") or "medium").strip().lower()
        if detail not in ("low", "medium", "high"):
            detail = "medium"

        target_faces = int(mesh_rt.get("target_faces") or 250000)
        max_depth_m = float(mesh_rt.get("max_depth_m") or 40.0)
        seed = mesh_rt.get("seed", None)

        # Seed (best-effort determinism)
        if seed is not None:
            try:
                s = int(seed)
                np.random.seed(s)
                torch.manual_seed(s)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(s)
                _safe_set(meta, "mesh_runtime._seed_applied", s)
            except Exception:
                _safe_set(meta, "mesh_runtime._seed_applied", None)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _safe_set(meta, "mesh_runtime.device", device)

        depth01, model_type = _predict_depth_bgr(img, device=device)
        _safe_set(meta, "mesh_runtime.depth_model", model_type)

        outputs = meta.get("outputs") or {}
        depth_name = outputs.get("depth") or "depth.png"
        preview_name = outputs.get("preview") or "preview.png"
        obj_name = outputs.get("obj") or "mesh.obj"
        glb_name = outputs.get("glb") or "mesh.glb"

        depth_path = os.path.join(job_folder, depth_name)
        preview_path = os.path.join(job_folder, preview_name)
        obj_path = os.path.join(job_folder, obj_name)
        glb_path = os.path.join(job_folder, glb_name)

        _write_depth_png(depth01, depth_path)
        _write_preview_png(depth01, preview_path)

        # Build geometry
        verts, faces = _build_grid_mesh(depth01, detail_level=detail, max_depth_m=max_depth_m)
        _safe_set(meta, "mesh_runtime.grid_faces", int(faces.shape[0]))
        _safe_set(meta, "mesh_runtime.grid_verts", int(verts.shape[0]))

        # Decimate (best-effort REAL)
        verts2, faces2, dec_msg = _try_decimate(verts, faces, target_faces=target_faces)
        _safe_set(meta, "mesh_runtime.decimation_message", dec_msg)
        _safe_set(meta, "mesh_runtime.final_faces", int(faces2.shape[0]))
        _safe_set(meta, "mesh_runtime.final_verts", int(verts2.shape[0]))

        # Normals
        normals = _compute_vertex_normals(verts2, faces2)

        # OBJ (required)
        _write_obj(obj_path, verts2, faces2, normals=normals)
        if not os.path.isfile(obj_path):
            raise RuntimeError("OBJ write failed (mesh.obj not found after write).")

        # GLB (optional but real)
        glb_ok, glb_msg = _try_export_glb(glb_path, verts2, faces2, normals=normals)
        _safe_set(meta, "mesh_runtime.glb_written", bool(glb_ok and os.path.isfile(glb_path)))
        _safe_set(meta, "mesh_runtime.glb_message", glb_msg)

        _safe_set(meta, "mesh_runtime.completed_at_epoch", _now_epoch())
        _safe_set(meta, "mesh_runtime.obj_written", True)
        _safe_set(meta, "mesh_runtime.preview_written", os.path.isfile(preview_path))
        _safe_set(meta, "mesh_runtime.depth_written", os.path.isfile(depth_path))

        _write_meta(job_folder, meta)

        return {
            "status": "ok",
            "obj": obj_path,
            "glb": glb_path if (glb_ok and os.path.isfile(glb_path)) else None,
            "preview": preview_path if os.path.isfile(preview_path) else None,
            "depth": depth_path if os.path.isfile(depth_path) else None,
            "depth_model": model_type,
            "detail_level": detail,
            "faces": int(faces2.shape[0]),
            "verts": int(verts2.shape[0]),
            "decimation": dec_msg,
            "glb_message": glb_msg,
        }

    except Exception as exc:
        _safe_set(meta, "mesh_runtime.completed_at_epoch", _now_epoch())
        _safe_set(
            meta,
            "mesh_runtime.error",
            {"detail": str(exc), "trace": traceback.format_exc()},
        )
        _write_meta(job_folder, meta)
        raise
