"""
RENDEREXPO AI STUDIO
Option A — Reference-Guided Furniture / Product Staging

Production direction:
- Isolated background removal happens outside this file.
- This module receives a room image + transparent reference cutout.
- It performs deterministic placement, scale, color/light matching,
  shadow, reflection, and compositing.
- SDXL/IP-Adapter generation is intentionally OFF by default because
  prior tests showed it can delete or distort the inserted object.

This file is safe to add because it is not wired into dispatch/router/main yet.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


@dataclass
class ReferenceStagingControls:
    mode: str = "add_object"

    relative_x: float = 0.375
    relative_y: float = 0.730
    relative_width: float = 0.105

    # Furniture/product staging should be placed by floor-contact anchor,
    # not by top-left image-box placement.
    use_floor_anchor: bool = True
    anchor_x: Optional[float] = None
    anchor_y: Optional[float] = None

    rotation_degrees: float = 0.0
    horizontal_scale: float = 1.0
    vertical_scale: float = 1.0
    perspective_compress_y: float = 0.96

    alpha_threshold: int = 10
    edge_feather_radius: float = 0.40

    saturation_factor: float = 0.76
    brightness_factor: float = 0.84
    contrast_factor: float = 0.90
    use_local_light_sampling: bool = True
    local_light_strength: float = 0.35

    contact_shadow_opacity: int = 92
    soft_shadow_opacity: int = 40
    contact_shadow_blur: int = 8
    soft_shadow_blur: int = 22

    reflection_enabled: bool = True
    reflection_opacity_top: float = 0.26
    reflection_blur: float = 2.8

    generative_polish_enabled: bool = False


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _validate_controls(c: ReferenceStagingControls) -> ReferenceStagingControls:
    c.mode = c.mode or "add_object"

    if c.mode not in {"add_object"}:
        raise ValueError(f"Unsupported Option A mode for this module: {c.mode!r}")

    c.relative_x = _clamp(c.relative_x, 0.0, 1.0)
    c.relative_y = _clamp(c.relative_y, 0.0, 1.0)
    c.relative_width = _clamp(c.relative_width, 0.02, 0.60)

    if c.anchor_x is not None:
        c.anchor_x = _clamp(c.anchor_x, 0.0, 1.0)
    if c.anchor_y is not None:
        c.anchor_y = _clamp(c.anchor_y, 0.0, 1.0)

    c.horizontal_scale = _clamp(c.horizontal_scale, 0.50, 1.80)
    c.vertical_scale = _clamp(c.vertical_scale, 0.50, 1.80)
    c.perspective_compress_y = _clamp(c.perspective_compress_y, 0.65, 1.10)

    c.edge_feather_radius = _clamp(c.edge_feather_radius, 0.0, 5.0)

    c.saturation_factor = _clamp(c.saturation_factor, 0.25, 1.50)
    c.brightness_factor = _clamp(c.brightness_factor, 0.25, 1.50)
    c.contrast_factor = _clamp(c.contrast_factor, 0.25, 1.50)
    c.local_light_strength = _clamp(c.local_light_strength, 0.0, 1.0)

    c.contact_shadow_opacity = int(_clamp(c.contact_shadow_opacity, 0, 180))
    c.soft_shadow_opacity = int(_clamp(c.soft_shadow_opacity, 0, 140))

    c.reflection_opacity_top = _clamp(c.reflection_opacity_top, 0.0, 0.65)

    return c


def _load_rgba(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")
    return Image.open(path).convert("RGBA")


def trim_alpha(img: Image.Image, threshold: int = 10) -> Image.Image:
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3]

    ys, xs = np.where(alpha > threshold)

    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("Cutout appears empty after alpha trim.")

    return rgba.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def soften_alpha(img: Image.Image, radius: float) -> Image.Image:
    rgba = img.convert("RGBA")

    if radius <= 0:
        return rgba

    alpha = rgba.getchannel("A").filter(ImageFilter.GaussianBlur(radius=radius))
    rgba.putalpha(alpha)

    return rgba


def _rgb_mean(img: Image.Image) -> Tuple[float, float, float]:
    rgb = np.array(img.convert("RGB")).astype(np.float32)
    return tuple(np.mean(rgb.reshape(-1, 3), axis=0).tolist())


def _sample_scene_patch(
    scene: Image.Image,
    paste_x: int,
    paste_y: int,
    target_w: int,
    target_h: int,
) -> Image.Image:
    sw, sh = scene.size

    x1 = max(0, paste_x - int(target_w * 0.30))
    y1 = max(0, paste_y + int(target_h * 0.35))
    x2 = min(sw, paste_x + int(target_w * 1.30))
    y2 = min(sh, paste_y + int(target_h * 1.15))

    if x2 <= x1 or y2 <= y1:
        return scene.convert("RGB")

    return scene.crop((x1, y1, x2, y2)).convert("RGB")


def apply_room_light_match(
    cutout: Image.Image,
    controls: ReferenceStagingControls,
    scene: Optional[Image.Image] = None,
    paste_x: Optional[int] = None,
    paste_y: Optional[int] = None,
    target_w: Optional[int] = None,
    target_h: Optional[int] = None,
) -> Image.Image:
    rgba = cutout.convert("RGBA")
    alpha = rgba.getchannel("A")
    rgb = rgba.convert("RGB")

    rgb = ImageEnhance.Color(rgb).enhance(controls.saturation_factor)
    rgb = ImageEnhance.Brightness(rgb).enhance(controls.brightness_factor)
    rgb = ImageEnhance.Contrast(rgb).enhance(controls.contrast_factor)

    if (
        controls.use_local_light_sampling
        and scene is not None
        and paste_x is not None
        and paste_y is not None
        and target_w is not None
        and target_h is not None
    ):
        try:
            patch = _sample_scene_patch(scene, paste_x, paste_y, target_w, target_h)

            obj_mean = np.array(_rgb_mean(rgb), dtype=np.float32)
            room_mean = np.array(_rgb_mean(patch), dtype=np.float32)

            ratio = np.clip(room_mean / np.maximum(obj_mean, 1.0), 0.70, 1.18)
            ratio = 1.0 + (ratio - 1.0) * controls.local_light_strength

            arr = np.array(rgb).astype(np.float32)
            arr *= ratio.reshape(1, 1, 3)
            arr = np.clip(arr, 0, 255).astype(np.uint8)

            rgb = Image.fromarray(arr, mode="RGB")

        except Exception:
            pass

    out = rgb.convert("RGBA")
    out.putalpha(alpha)

    return out


def resize_cutout_for_scene(
    cutout: Image.Image,
    scene_size: Tuple[int, int],
    controls: ReferenceStagingControls,
) -> Tuple[Image.Image, int, int, float]:
    sw, _ = scene_size
    cw, ch = cutout.size

    target_w = max(1, int(sw * controls.relative_width))
    scale = target_w / float(cw)

    target_h = max(1, int(ch * scale * controls.perspective_compress_y))
    target_w_adjusted = max(1, int(target_w * controls.horizontal_scale))
    target_h_adjusted = max(1, int(target_h * controls.vertical_scale))

    resized = cutout.resize(
        (target_w_adjusted, target_h_adjusted),
        Image.Resampling.LANCZOS,
    )

    return resized, target_w_adjusted, target_h_adjusted, scale


def create_contact_shadow(
    scene_size: Tuple[int, int],
    paste_x: int,
    paste_y: int,
    target_w: int,
    target_h: int,
    controls: ReferenceStagingControls,
) -> Image.Image:
    sw, sh = scene_size

    tight = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    td = ImageDraw.Draw(tight)

    td.ellipse(
        (
            paste_x + int(target_w * 0.10),
            paste_y + int(target_h * 0.88),
            paste_x + int(target_w * 0.90),
            paste_y + int(target_h * 1.065),
        ),
        fill=(0, 0, 0, controls.contact_shadow_opacity),
    )

    tight = tight.filter(ImageFilter.GaussianBlur(radius=controls.contact_shadow_blur))

    soft = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    sd = ImageDraw.Draw(soft)

    sd.ellipse(
        (
            paste_x - int(target_w * 0.24),
            paste_y + int(target_h * 0.80),
            paste_x + int(target_w * 1.24),
            paste_y + int(target_h * 1.18),
        ),
        fill=(0, 0, 0, controls.soft_shadow_opacity),
    )

    soft = soft.filter(ImageFilter.GaussianBlur(radius=controls.soft_shadow_blur))

    return Image.alpha_composite(soft, tight)


def create_floor_reflection(
    scene_size: Tuple[int, int],
    cutout: Image.Image,
    paste_x: int,
    paste_y: int,
    target_h: int,
    controls: ReferenceStagingControls,
) -> Image.Image:
    sw, sh = scene_size
    reflection = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))

    if not controls.reflection_enabled or controls.reflection_opacity_top <= 0:
        return reflection

    ref_obj = cutout.copy().transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    arr = np.array(ref_obj)

    alpha = arr[:, :, 3].astype(np.float32)
    h = alpha.shape[0]

    fade = np.linspace(controls.reflection_opacity_top, 0.0, h).reshape(h, 1)
    alpha = alpha * fade

    arr[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)

    ref_obj = Image.fromarray(arr, mode="RGBA").filter(
        ImageFilter.GaussianBlur(radius=controls.reflection_blur)
    )

    reflection_y = paste_y + target_h - int(target_h * 0.015)
    reflection.alpha_composite(ref_obj, (paste_x, reflection_y))

    return reflection


def create_grounding_glaze(
    scene_size: Tuple[int, int],
    paste_x: int,
    paste_y: int,
    target_w: int,
    target_h: int,
) -> Image.Image:
    sw, sh = scene_size

    glaze = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glaze)

    gd.rectangle(
        (
            paste_x + int(target_w * 0.05),
            paste_y + int(target_h * 0.20),
            paste_x + int(target_w * 0.95),
            paste_y + int(target_h * 0.29),
        ),
        fill=(0, 0, 0, 14),
    )

    gd.rectangle(
        (
            paste_x + int(target_w * 0.20),
            paste_y + int(target_h * 0.65),
            paste_x + int(target_w * 0.80),
            paste_y + int(target_h * 0.98),
        ),
        fill=(0, 0, 0, 8),
    )

    return glaze.filter(ImageFilter.GaussianBlur(radius=2.0))


def stage_reference_object(
    scene_path: Path,
    cutout_path: Path,
    job_dir: Path,
    controls: Optional[ReferenceStagingControls] = None,
) -> Dict[str, Any]:
    controls = _validate_controls(controls or ReferenceStagingControls())

    job_dir.mkdir(parents=True, exist_ok=True)

    scene_out = job_dir / "scene_input.png"
    cutout_out = job_dir / "reference_cutout.png"
    placed_out = job_dir / "placed_cutout.png"
    shadow_out = job_dir / "shadow_layer.png"
    reflection_out = job_dir / "reflection_layer.png"
    output_out = job_dir / "output.png"
    meta_out = job_dir / "meta.json"

    scene = _load_rgba(scene_path)
    sw, sh = scene.size
    scene.save(scene_out)

    raw_cutout = _load_rgba(cutout_path)
    cutout_out.write_bytes(cutout_path.read_bytes())

    cutout = trim_alpha(raw_cutout, controls.alpha_threshold)
    cutout = soften_alpha(cutout, controls.edge_feather_radius)

    if controls.rotation_degrees:
        cutout = cutout.rotate(
            controls.rotation_degrees,
            expand=True,
            resample=Image.Resampling.BICUBIC,
        )

    cutout, target_w, target_h, scale = resize_cutout_for_scene(
        cutout,
        scene.size,
        controls,
    )

    if controls.use_floor_anchor:
        anchor_x_rel = controls.anchor_x if controls.anchor_x is not None else controls.relative_x
        anchor_y_rel = controls.anchor_y if controls.anchor_y is not None else controls.relative_y

        anchor_x_px = int(sw * anchor_x_rel)
        anchor_y_px = int(sh * anchor_y_rel)

        # Floor-anchor placement:
        # bottom-center of the object lands on the anchor point.
        paste_x = int(anchor_x_px - target_w / 2)
        paste_y = int(anchor_y_px - target_h)
    else:
        anchor_x_px = None
        anchor_y_px = None

        # Legacy placement:
        # relative_x / relative_y represent top-left placement.
        paste_x = int(sw * controls.relative_x)
        paste_y = int(sh * controls.relative_y)

    cutout = apply_room_light_match(
        cutout,
        controls,
        scene=scene,
        paste_x=paste_x,
        paste_y=paste_y,
        target_w=target_w,
        target_h=target_h,
    )

    placed_canvas = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    placed_canvas.alpha_composite(cutout, (paste_x, paste_y))
    placed_canvas.save(placed_out)

    shadow = create_contact_shadow(scene.size, paste_x, paste_y, target_w, target_h, controls)
    shadow.save(shadow_out)

    reflection = create_floor_reflection(scene.size, cutout, paste_x, paste_y, target_h, controls)
    reflection.save(reflection_out)

    composite = scene.copy()
    composite = Image.alpha_composite(composite, shadow)
    composite = Image.alpha_composite(composite, reflection)
    composite.alpha_composite(cutout, (paste_x, paste_y))
    composite = Image.alpha_composite(
        composite,
        create_grounding_glaze(scene.size, paste_x, paste_y, target_w, target_h),
    )

    final = composite.convert("RGB")
    final.save(output_out)

    meta: Dict[str, Any] = {
        "status": "completed",
        "service": "option_a_reference_guided_furniture_product_staging",
        "mode": "placement_engine_add_object",
        "engine": "rembg_cutout_plus_deterministic_placement_compositing",
        "commercial_use_note": (
            "This engine uses deterministic compositing with a pre-isolated cutout. "
            "No SDXL/IP-Adapter generation is used in the final output by default."
        ),
        "safety_note": (
            "This module does not install packages and does not modify existing services. "
            "It is inactive until explicitly called by a test, dispatch, or router."
        ),
        "scene_image": str(scene_path),
        "cutout_image": str(cutout_path),
        "outputs": {
            "scene_input": "scene_input.png",
            "reference_cutout": "reference_cutout.png",
            "placed_cutout": "placed_cutout.png",
            "shadow_layer": "shadow_layer.png",
            "reflection_layer": "reflection_layer.png",
            "output": "output.png",
        },
        "controls": asdict(controls),
        "placement_pixels": {
            "target_w": target_w,
            "target_h": target_h,
            "paste_x": paste_x,
            "paste_y": paste_y,
            "anchor_x_px": anchor_x_px,
            "anchor_y_px": anchor_y_px,
            "scale_factor": scale,
        },
        "processing": {
            "sdxl_inpaint": False,
            "ip_adapter": False,
            "generative_polish_enabled": False,
        },
    }

    meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "status": "completed",
        "job_dir": str(job_dir),
        "output": str(output_out),
        "meta": str(meta_out),
        "artifacts": meta["outputs"],
    }


def run_direct_test() -> Dict[str, Any]:
    repo = Path("/workspace-data/RENDEREXPO-AI-Studio-Backend")

    scene = repo / "outputs/2026-05-13/6db982b99d8a4624abebd8201f4bb179/output.png"
    cutout = repo / "outputs/2026-05-17/option_a_rembg_cutout_001/rembg_cutout.png"

    today = datetime.now().strftime("%Y-%m-%d")
    job_dir = repo / "outputs" / today / "option_a_reference_staging_module_test_001"

    controls = ReferenceStagingControls(
        mode="add_object",
        relative_x=0.405,
        relative_y=0.792,
        relative_width=0.118,
        use_floor_anchor=True,
        anchor_x=0.405,
        anchor_y=0.792,
        perspective_compress_y=0.90,
        saturation_factor=0.72,
        brightness_factor=0.80,
        contrast_factor=0.88,
        use_local_light_sampling=True,
        local_light_strength=0.48,
        contact_shadow_opacity=112,
        soft_shadow_opacity=50,
        reflection_enabled=True,
        reflection_opacity_top=0.18,
    )

    return stage_reference_object(scene, cutout, job_dir, controls)


if __name__ == "__main__":
    result = run_direct_test()
    print(json.dumps(result, indent=2))