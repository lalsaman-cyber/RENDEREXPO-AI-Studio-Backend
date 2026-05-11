# app/gpu/moodboard.py
"""
RENDEREXPO AI STUDIO - Moodboard GPU Runners

PURPOSE
-------
This file adds an isolated Moodboard GPU lane without touching working services.

PROTECTED EXISTING SERVICES
---------------------------
Do NOT modify these from this file:
- SD3.5 text2img
- SD3.5 img2img
- Upscale
- Sketch to Render
- Sketch to Redesign
- Video
- CAD
- Mesh
- VR

MOODBOARD FLOWS
---------------
1) space_to_moodboard
   Input:
      - space.png
   Output:
      - moodboard_grid.png
      - palette.json
      - extracted_assets.json

2) sd35_moodboard_to_space
   Input:
      - moodboard_000.png, moodboard_001.png, ...
      - optional floorplan.png
   Output:
      - moodboard_grid.png
      - palette.json
      - extracted_assets.json
      - output.png

3) sd35_apply_moodboard_to_render
   Input:
      - input.png
      - moodboard_job_id / moodboard reference metadata
   Output:
      - output.png

IMPORTANT
---------
This first moodboard implementation is intentionally safe:
- It creates real files.
- It updates meta through returned results.
- It does not alter the SD3.5 runtime.
- It does not require IP-Adapter yet.
- It prepares the structure for future Mattoboard-like reference conditioning.

Later, the advanced version can add:
- SD3.5 IP-Adapter
- SD3.5 ControlNet
- richer material/product extraction
without disturbing the other services.
"""

from __future__ import annotations

import colorsys
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

OUTPUTS_ROOT = "outputs"

MOODBOARD_GRID_NAME = "moodboard_grid.png"
PALETTE_JSON_NAME = "palette.json"
EXTRACTED_ASSETS_JSON_NAME = "extracted_assets.json"
OUTPUT_IMAGE_NAME = "output.png"
MATERIAL_SAMPLES_DIR_NAME = "material_samples"


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _job_folder_from_payload(payload: Dict[str, Any]) -> str:
    job_folder = payload.get("job_folder")
    if not job_folder or not isinstance(job_folder, str):
        raise RuntimeError("payload.job_folder is required for moodboard jobs.")

    if not os.path.isabs(job_folder):
        job_folder = os.path.join("/workspace-data/RENDEREXPO-AI-Studio-Backend", job_folder)

    if not os.path.isdir(job_folder):
        raise RuntimeError(f"Moodboard job folder does not exist: {job_folder}")

    return job_folder


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data if isinstance(data, dict) else {}


def _write_json(path: str, data: Dict[str, Any]) -> str:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return path


def _find_first_existing(job_folder: str, names: List[str]) -> Optional[str]:
    for name in names:
        p = os.path.join(job_folder, name)
        if os.path.isfile(p):
            return p
    return None


def _list_moodboard_images(job_folder: str) -> List[str]:
    paths: List[str] = []

    for name in sorted(os.listdir(job_folder)):
        lower = name.lower()
        if lower.startswith("moodboard_") and lower.endswith((".png", ".jpg", ".jpeg")):
            p = os.path.join(job_folder, name)
            if os.path.isfile(p):
                paths.append(p)

    return paths


def _safe_copy(src: str, dst: str) -> str:
    if not os.path.isfile(src):
        raise RuntimeError(f"Source file does not exist: {src}")
    _ensure_dir(os.path.dirname(dst))
    shutil.copyfile(src, dst)
    return dst


def _find_job_folder_by_job_id(job_id: str) -> Optional[str]:
    job_id = (job_id or "").strip()
    if not job_id:
        return None

    outputs_dir = OUTPUTS_ROOT
    if not os.path.isdir(outputs_dir):
        return None

    for date_dir in sorted(os.listdir(outputs_dir), reverse=True):
        date_path = os.path.join(outputs_dir, date_dir)
        if not os.path.isdir(date_path):
            continue

        candidate = os.path.join(date_path, job_id)
        if os.path.isdir(candidate):
            return candidate

    return None


# ---------------------------------------------------------------------
# PIL helpers
# ---------------------------------------------------------------------

def _require_pil():
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageStat  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"PIL/Pillow is required for moodboard processing: {exc}") from exc

    return Image, ImageDraw, ImageFont, ImageStat


def _open_rgb(path: str):
    Image, _, _, _ = _require_pil()
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to open image: {path} :: {exc}") from exc


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _relative_luminance(rgb: Tuple[int, int, int]) -> float:
    r, g, b = [v / 255.0 for v in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _warmth_score(rgb: Tuple[int, int, int]) -> float:
    r, g, b = rgb
    return float(r - b) / 255.0


def _hex_to_rgb(hex_value: str) -> Tuple[int, int, int]:
    hx = str(hex_value or "").strip().lstrip("#")
    if len(hx) != 6:
        return (204, 204, 204)
    try:
        return tuple(int(hx[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except Exception:
        return (204, 204, 204)


def _color_saturation(rgb: Tuple[int, int, int]) -> float:
    r, g, b = [v / 255.0 for v in rgb]
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx <= 0:
        return 0.0
    return (mx - mn) / mx


def _is_board_background_color(rgb: Tuple[int, int, int]) -> bool:
    lum = _relative_luminance(rgb)
    sat = _color_saturation(rgb)
    if lum >= 0.91 and sat <= 0.08:
        return True
    if min(rgb) >= 235 and max(rgb) >= 242:
        return True
    return False


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _rgb_to_hsv_tuple(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    r, g, b = [max(0.0, min(1.0, v / 255.0)) for v in rgb]
    return colorsys.rgb_to_hsv(r, g, b)


def _hue_name(rgb: Tuple[int, int, int]) -> str:
    h, s, v = _rgb_to_hsv_tuple(rgb)
    if v <= 0.08:
        return "black"
    if s <= 0.10 and v >= 0.88:
        return "white"
    if s <= 0.12:
        return "gray"

    deg = h * 360.0
    if deg < 18 or deg >= 345:
        return "red"
    if deg < 42:
        return "orange"
    if deg < 70:
        return "yellow"
    if deg < 165:
        return "green"
    if deg < 205:
        return "cyan"
    if deg < 255:
        return "blue"
    if deg < 315:
        return "purple"
    return "magenta"


def _box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return float(inter) / float(a_area + b_area - inter)



def _iter_subboxes(width: int, height: int, scales: List[Tuple[float, float]]) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    x_positions = [0.0, 0.12, 0.24, 0.38]
    y_positions = [0.0, 0.12, 0.24, 0.38]

    for sw, sh in scales:
        box_w = max(56, min(width, int(width * sw)))
        box_h = max(56, min(height, int(height * sh)))
        if box_w >= width:
            box_w = width
        if box_h >= height:
            box_h = height

        for xn in x_positions:
            for yn in y_positions:
                x1 = int((width - box_w) * xn)
                y1 = int((height - box_h) * yn)
                x1 = max(0, min(width - box_w, x1))
                y1 = max(0, min(height - box_h, y1))
                x2 = x1 + box_w
                y2 = y1 + box_h
                if x2 - x1 >= 48 and y2 - y1 >= 48:
                    boxes.append((x1, y1, x2, y2))

    boxes.append((0, 0, width, height))

    deduped: List[Tuple[int, int, int, int]] = []
    seen = set()
    for box in boxes:
        if box not in seen:
            seen.add(box)
            deduped.append(box)
    return deduped



def _score_texture_patch(metrics: Dict[str, Any], family: str) -> float:
    family_score = _score_material_family(metrics, family)
    texture = float(metrics.get("texture_score", 0.0) or 0.0)
    detail = float(metrics.get("detail_score", 0.0) or 0.0)
    lum = float(metrics.get("luminance", 0.5) or 0.5)
    sat = float(metrics.get("saturation", 0.0) or 0.0)
    hue_name = str(metrics.get("hue_name") or "")
    aspect = float(metrics.get("aspect_ratio", 1.0) or 1.0)
    norm = metrics.get("norm_box") or {}
    x1n = float(norm.get("x1n", 0.0) or 0.0)
    y1n = float(norm.get("y1n", 0.0) or 0.0)
    x2n = float(norm.get("x2n", 1.0) or 1.0)
    y2n = float(norm.get("y2n", 1.0) or 1.0)
    cx = (x1n + x2n) / 2.0
    cy = (y1n + y2n) / 2.0
    center_bias = 1.0 - min(1.0, abs(cx - 0.5) + abs(cy - 0.5))

    score = family_score * 0.58 + center_bias * 0.08

    if family == "stone":
        score += texture * 0.22 + detail * 0.10
        score += 0.08 * (1.0 if sat <= 0.30 else 0.0)
        score += 0.06 * (1.0 if 0.60 <= aspect <= 1.80 else 0.0)
    elif family == "wood":
        score += texture * 0.18 + detail * 0.10
        score += 0.08 * (1.0 if hue_name in {"orange", "yellow", "red", "brown"} else 0.0)
        score += 0.06 * (1.0 if 0.55 <= aspect <= 1.90 else 0.0)
    elif family == "glass":
        score += 0.12 * (1.0 if sat <= 0.30 else 0.0)
        score += 0.10 * (1.0 if 0.25 <= lum <= 0.82 else 0.0)
        score += 0.10 * (1.0 - min(1.0, abs(detail - 0.30) / 0.50))
    elif family == "water":
        score += 0.18 * (1.0 if hue_name in {"cyan", "blue"} else 0.0)
        score += texture * 0.10 + detail * 0.08 + sat * 0.10
        score += 0.06 * (1.0 if cy >= 0.42 else 0.0)
    elif family == "paving":
        score += texture * 0.18 + detail * 0.10
        score += 0.08 * (1.0 if sat <= 0.24 else 0.0)
        score += 0.06 * (1.0 if cy >= 0.45 else 0.0)
    elif family == "metal":
        score += texture * 0.12 + detail * 0.16
        score += 0.08 * (1.0 if sat <= 0.22 else 0.0)
        score += 0.06 * (1.0 if lum <= 0.35 else 0.0)
    elif family == "planting":
        score += texture * 0.12 + detail * 0.10 + sat * 0.12
        score += 0.08 * (1.0 if hue_name == "green" else 0.0)
    elif family == "fabric":
        score += texture * 0.10 + detail * 0.10
        score += 0.08 * (1.0 if 0.42 <= lum <= 0.88 else 0.0)
        score += 0.06 * (1.0 if sat <= 0.24 else 0.0)

    return round(_clamp(score, 0.0, 1.0), 4)



def _refine_material_patch(
    crop: Any,
    *,
    family: str,
    source_file: str,
) -> Tuple[Any, Dict[str, Any], Dict[str, int]]:
    src_w, src_h = crop.size
    if src_w < 72 or src_h < 72:
        metrics = _compute_crop_metrics(
            crop,
            box=(0, 0, src_w, src_h),
            source_file=source_file,
            norm_box={"x1n": 0.0, "y1n": 0.0, "x2n": 1.0, "y2n": 1.0},
        )
        return crop.copy(), metrics, {"x1": 0, "y1": 0, "x2": src_w, "y2": src_h}

    if family in {"stone", "wood", "paving", "metal"}:
        scales = [(0.52, 0.52), (0.60, 0.56), (0.68, 0.62)]
    elif family in {"glass", "water"}:
        scales = [(0.60, 0.48), (0.72, 0.54), (0.84, 0.58)]
    else:
        scales = [(0.56, 0.56), (0.68, 0.60), (0.80, 0.64)]

    best_crop = crop.copy()
    best_metrics = _compute_crop_metrics(
        crop,
        box=(0, 0, src_w, src_h),
        source_file=source_file,
        norm_box={"x1n": 0.0, "y1n": 0.0, "x2n": 1.0, "y2n": 1.0},
    )
    best_score = _score_texture_patch(best_metrics, family)
    best_box = {"x1": 0, "y1": 0, "x2": src_w, "y2": src_h}

    for x1, y1, x2, y2 in _iter_subboxes(src_w, src_h, scales):
        sub = crop.crop((x1, y1, x2, y2))
        metrics = _compute_crop_metrics(
            sub,
            box=(x1, y1, x2, y2),
            source_file=source_file,
            norm_box={
                "x1n": round(x1 / float(src_w), 4),
                "y1n": round(y1 / float(src_h), 4),
                "x2n": round(x2 / float(src_w), 4),
                "y2n": round(y2 / float(src_h), 4),
            },
        )
        avg_rgb_dict = metrics.get("average_color", {}).get("rgb", {})
        avg_rgb = (
            int(avg_rgb_dict.get("r", 204)),
            int(avg_rgb_dict.get("g", 204)),
            int(avg_rgb_dict.get("b", 204)),
        )
        if _is_board_background_color(avg_rgb):
            continue
        if float(metrics.get("sky_score", 0.0) or 0.0) >= 0.72:
            continue

        score = _score_texture_patch(metrics, family)
        if score > best_score:
            best_score = score
            best_crop = sub.copy()
            best_metrics = metrics
            best_box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    return best_crop, best_metrics, best_box


def _compute_crop_metrics(img: Any, *, box: Tuple[int, int, int, int], source_file: str, norm_box: Dict[str, float]) -> Dict[str, Any]:
    _, _, _, ImageStat = _require_pil()
    stat = ImageStat.Stat(img)

    avg = tuple(int(v) for v in stat.mean[:3])
    std = stat.stddev[:3] if getattr(stat, "stddev", None) else [0.0, 0.0, 0.0]
    std_avg = float(sum(float(v) for v in std[:3])) / 3.0

    lum = _relative_luminance(avg)
    warm = _warmth_score(avg)
    sat = _color_saturation(avg)
    h, _, _ = _rgb_to_hsv_tuple(avg)
    hue_deg = h * 360.0
    hue_name = _hue_name(avg)

    r, g, b = avg
    channel_total = max(1.0, float(r + g + b))
    blue_ratio = b / channel_total
    green_ratio = g / channel_total
    red_ratio = r / channel_total

    crop_w, crop_h = img.size
    aspect = float(crop_w) / float(max(1, crop_h))
    area = crop_w * crop_h

    sky_score = 0.0
    if norm_box["y1n"] <= 0.35:
        if hue_name in {"blue", "cyan"} and lum >= 0.55 and sat <= 0.35:
            sky_score += 0.65
        if blue_ratio >= 0.36 and green_ratio >= 0.30 and red_ratio <= 0.30:
            sky_score += 0.25
    if lum >= 0.76 and sat <= 0.18:
        sky_score += 0.10

    texture_score = _clamp((std_avg - 10.0) / 55.0)
    detail_score = _clamp((std_avg - 8.0) / 42.0)

    return {
        "source_file": source_file,
        "crop_box": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]},
        "norm_box": norm_box,
        "width": crop_w,
        "height": crop_h,
        "area": area,
        "aspect_ratio": round(aspect, 4),
        "average_color": {
            "hex": _rgb_to_hex(avg),
            "rgb": {"r": avg[0], "g": avg[1], "b": avg[2]},
        },
        "luminance": round(lum, 4),
        "warmth": round(warm, 4),
        "saturation": round(sat, 4),
        "hue_degrees": round(hue_deg, 2),
        "hue_name": hue_name,
        "texture_score": round(texture_score, 4),
        "detail_score": round(detail_score, 4),
        "sky_score": round(_clamp(sky_score), 4),
        "rgb_stddev": [round(float(v), 3) for v in std[:3]],
        "channel_ratios": {
            "red": round(red_ratio, 4),
            "green": round(green_ratio, 4),
            "blue": round(blue_ratio, 4),
        },
    }


def _score_material_family(metrics: Dict[str, Any], family: str) -> float:
    lum = float(metrics.get("luminance", 0.5) or 0.5)
    warm = float(metrics.get("warmth", 0.0) or 0.0)
    sat = float(metrics.get("saturation", 0.0) or 0.0)
    texture = float(metrics.get("texture_score", 0.0) or 0.0)
    detail = float(metrics.get("detail_score", 0.0) or 0.0)
    sky = float(metrics.get("sky_score", 0.0) or 0.0)
    hue_name = str(metrics.get("hue_name") or "")
    norm = metrics.get("norm_box") or {}
    x1n = float(norm.get("x1n", 0.0) or 0.0)
    y1n = float(norm.get("y1n", 0.0) or 0.0)
    x2n = float(norm.get("x2n", 1.0) or 1.0)
    y2n = float(norm.get("y2n", 1.0) or 1.0)
    cx = (x1n + x2n) / 2.0
    cy = (y1n + y2n) / 2.0
    aspect = float(metrics.get("aspect_ratio", 1.0) or 1.0)

    score = 0.02

    if family == "water":
        score += 0.34 * (1.0 if hue_name in {"cyan", "blue"} else 0.0)
        score += 0.18 * sat
        score += 0.16 * detail
        score += 0.14 * texture
        score += 0.18 * _clamp((cy - 0.50) / 0.50)
        score += 0.06 * (1.0 if lum >= 0.22 and lum <= 0.72 else 0.0)
        score -= 0.35 * sky
    elif family == "wood":
        score += 0.30 * _clamp((warm + 0.05) / 0.50)
        score += 0.18 * texture
        score += 0.14 * detail
        score += 0.10 * sat
        score += 0.16 * (1.0 if hue_name in {"orange", "yellow", "red"} else 0.0)
        score += 0.10 * _clamp((0.55 - cy) / 0.55)
        score -= 0.18 * sky
    elif family == "stone":
        score += 0.22 * texture
        score += 0.14 * detail
        score += 0.12 * (1.0 if 0.22 <= lum <= 0.72 else 0.0)
        score += 0.10 * (1.0 if sat <= 0.30 else 0.0)
        score += 0.10 * (1.0 if hue_name in {"gray", "orange", "yellow"} else 0.0)
        score += 0.08 * (1.0 if cx <= 0.28 or cx >= 0.72 else 0.0)
        score -= 0.12 * sky
    elif family == "glass":
        score += 0.24 * (1.0 if hue_name in {"cyan", "blue", "gray"} else 0.0)
        score += 0.18 * _clamp((lum - 0.35) / 0.50)
        score += 0.12 * (1.0 if sat <= 0.35 else 0.0)
        score += 0.10 * detail
        score += 0.12 * (1.0 if 0.75 <= aspect <= 2.8 else 0.0)
        score += 0.10 * (1.0 if 0.18 <= cy <= 0.72 else 0.0)
        score -= 0.16 * sky
    elif family == "paving":
        score += 0.24 * _clamp((cy - 0.45) / 0.55)
        score += 0.16 * (1.0 if 0.45 <= lum <= 0.85 else 0.0)
        score += 0.16 * (1.0 if sat <= 0.28 else 0.0)
        score += 0.14 * texture
        score += 0.10 * detail
        score += 0.08 * (1.0 if hue_name in {"gray", "yellow", "orange"} else 0.0)
        score -= 0.18 * sky
    elif family == "metal":
        score += 0.24 * (1.0 if lum <= 0.28 else 0.0)
        score += 0.18 * (1.0 if sat <= 0.24 else 0.0)
        score += 0.14 * detail
        score += 0.12 * texture
        score += 0.08 * (1.0 if hue_name in {"gray", "blue", "black"} else 0.0)
        score += 0.08 * (1.0 if 0.15 <= cy <= 0.80 else 0.0)
        score -= 0.10 * sky
    elif family == "planting":
        score += 0.34 * (1.0 if hue_name == "green" else 0.0)
        score += 0.18 * sat
        score += 0.12 * texture
        score += 0.10 * detail
        score += 0.08 * (1.0 if cx <= 0.25 or cx >= 0.75 else 0.0)
        score -= 0.12 * sky
    elif family == "fabric":
        score += 0.22 * (1.0 if 0.50 <= lum <= 0.88 else 0.0)
        score += 0.16 * (1.0 if sat <= 0.22 else 0.0)
        score += 0.12 * detail
        score += 0.12 * texture
        score += 0.10 * _clamp((cy - 0.40) / 0.60)
        score -= 0.10 * sky

    if texture < 0.10 and family in {"stone", "wood", "paving", "metal"}:
        score -= 0.15
    if sky > 0.45:
        score -= 0.30

    return round(_clamp(score, 0.0, 1.0), 4)


def _material_labels() -> Dict[str, str]:
    return {
        "stone": "Stone / masonry",
        "wood": "Wood soffit / slats",
        "glass": "Glass / glazing",
        "water": "Pool water",
        "paving": "Light stone paving",
        "metal": "Dark metal frame",
        "planting": "Landscape planting",
        "fabric": "Outdoor fabric",
    }


def _pick_best_family(scores: Dict[str, float]) -> Tuple[str, float]:
    if not scores:
        return "stone", 0.0
    fam = max(scores.items(), key=lambda kv: kv[1])[0]
    return fam, float(scores.get(fam, 0.0))


def _fallback_family_from_metrics(metrics: Dict[str, Any]) -> str:
    hue_name = str(metrics.get("hue_name") or "")
    lum = float(metrics.get("luminance", 0.5) or 0.5)
    warm = float(metrics.get("warmth", 0.0) or 0.0)
    sat = float(metrics.get("saturation", 0.0) or 0.0)
    norm = metrics.get("norm_box") or {}
    cy = (float(norm.get("y1n", 0.0) or 0.0) + float(norm.get("y2n", 1.0) or 1.0)) / 2.0

    if hue_name in {"cyan", "blue"} and sat >= 0.18 and cy >= 0.45:
        return "water"
    if hue_name == "green":
        return "planting"
    if warm >= 0.12 and sat >= 0.12:
        return "wood"
    if lum <= 0.28 and sat <= 0.24:
        return "metal"
    if cy >= 0.55 and lum >= 0.45:
        return "paving"
    if sat <= 0.18 and lum >= 0.58:
        return "fabric"
    return "stone"



def _safe_rgb_from_sample(sample: Dict[str, Any], fallback: Tuple[int, int, int] = (150, 140, 128)) -> Tuple[int, int, int]:
    avg = sample.get("average_color") or {}
    rgb_obj = avg.get("rgb") if isinstance(avg, dict) else None
    if isinstance(rgb_obj, dict):
        try:
            return (
                max(0, min(255, int(rgb_obj.get("r", fallback[0])))),
                max(0, min(255, int(rgb_obj.get("g", fallback[1])))),
                max(0, min(255, int(rgb_obj.get("b", fallback[2])))),
            )
        except Exception:
            return fallback
    return fallback


def _mix_rgb(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    t = _clamp(t)
    return (
        int(a[0] * (1.0 - t) + b[0] * t),
        int(a[1] * (1.0 - t) + b[1] * t),
        int(a[2] * (1.0 - t) + b[2] * t),
    )


def _adjust_rgb(rgb: Tuple[int, int, int], amount: float) -> Tuple[int, int, int]:
    if amount >= 0:
        return _mix_rgb(rgb, (255, 255, 255), amount)
    return _mix_rgb(rgb, (0, 0, 0), abs(amount))


def _draw_soft_noise(draw: Any, width: int, height: int, base: Tuple[int, int, int], *, step: int = 18) -> None:
    for y in range(0, height, step):
        for x in range(0, width, step):
            n = ((x * 37 + y * 19 + width * 11 + height * 7) % 31) / 30.0
            color = _adjust_rgb(base, (n - 0.5) * 0.18)
            draw.rectangle((x, y, min(width, x + step), min(height, y + step)), fill=color)


def _generate_material_swatch(
    *,
    sample: Dict[str, Any],
    output_path: str,
    width: int = 520,
    height: int = 340,
) -> str:
    """
    Generate clean material-style swatches for the visible moodboard.

    PIL-only: no SD3.5, no ComfyUI, no external material library, no changes to
    pinned generation settings. The detected crop remains recorded in JSON as
    reference_crop_file, while the visible tile becomes a clean material card.
    """
    Image, ImageDraw, _, _ = _require_pil()

    family = str(sample.get("material_family") or "material").lower()
    base = _safe_rgb_from_sample(sample)
    canvas = Image.new("RGB", (width, height), _adjust_rgb(base, 0.10))
    draw = ImageDraw.Draw(canvas)

    if family == "stone":
        mortar = _adjust_rgb(base, 0.28)
        _draw_soft_noise(draw, width, height, _adjust_rgb(base, 0.02), step=22)
        block_h = max(34, height // 7)
        y = 0
        row = 0
        while y < height:
            offset = 0 if row % 2 == 0 else -(width // 8)
            block_w = max(70, width // 4)
            x = offset
            while x < width:
                n = ((x * 13 + y * 17 + row * 23) % 17) / 16.0
                color = _adjust_rgb(base, (n - 0.5) * 0.20)
                draw.rectangle((x, y, x + block_w, y + block_h), fill=color, outline=mortar, width=3)
                for k in range(2):
                    lx = x + 10 + ((k * 41 + row * 23) % max(16, block_w - 20))
                    ly = y + 8 + ((k * 17 + row * 11) % max(12, block_h - 16))
                    draw.line((lx, ly, min(x + block_w - 8, lx + block_w // 3), ly + ((k % 2) * 8 - 4)), fill=_adjust_rgb(color, -0.18), width=1)
                x += block_w
            y += block_h
            row += 1

    elif family == "wood":
        warm_base = _mix_rgb(base, (150, 82, 36), 0.30)
        draw.rectangle((0, 0, width, height), fill=_adjust_rgb(warm_base, 0.03))
        slat_w = max(38, width // 10)
        for x in range(0, width + slat_w, slat_w):
            n = ((x * 29 + 17) % 23) / 22.0
            color = _adjust_rgb(warm_base, (n - 0.5) * 0.22)
            draw.rectangle((x, 0, x + slat_w - 4, height), fill=color)
            draw.line((x + slat_w - 3, 0, x + slat_w - 3, height), fill=_adjust_rgb(warm_base, -0.24), width=2)
            for yy in range(14, height, 22):
                wave = int(7 * math.sin((yy + x) / 38.0))
                draw.line((x + 6, yy, min(width, x + slat_w - 12), yy + wave), fill=_adjust_rgb(color, -0.16), width=1)
                draw.line((x + 8, yy + 7, min(width, x + slat_w - 16), yy + 7 + wave), fill=_adjust_rgb(color, 0.14), width=1)

    elif family == "glass":
        top = _mix_rgb(base, (198, 235, 245), 0.55)
        bottom = _mix_rgb(base, (75, 105, 120), 0.35)
        for y in range(height):
            t = y / float(max(1, height - 1))
            draw.line((0, y, width, y), fill=_mix_rgb(top, bottom, t))
        for x in range(-width // 2, width, 76):
            draw.line((x, height, x + width // 2, 0), fill=_adjust_rgb(top, 0.25), width=4)
            draw.line((x + 18, height, x + width // 2 + 18, 0), fill=_adjust_rgb(bottom, -0.18), width=1)
        draw.rectangle((0, 0, width - 1, height - 1), outline=_adjust_rgb(base, -0.18), width=3)

    elif family == "water":
        aqua = _mix_rgb(base, (30, 185, 190), 0.45)
        deep = _mix_rgb(base, (8, 55, 75), 0.45)
        for y in range(height):
            t = y / float(max(1, height - 1))
            draw.line((0, y, width, y), fill=_mix_rgb(_adjust_rgb(aqua, 0.25), deep, t))
        for y in range(22, height, 28):
            amp = 7 + (y % 17)
            points = []
            for x in range(0, width + 8, 8):
                yy = y + int(math.sin(x / 28.0 + y / 23.0) * amp)
                points.append((x, yy))
            draw.line(points, fill=_adjust_rgb(aqua, 0.42), width=2)
            draw.line([(x, yy + 5) for x, yy in points], fill=_adjust_rgb(deep, -0.12), width=1)

    elif family == "paving":
        stone = _mix_rgb(base, (205, 190, 165), 0.36)
        draw.rectangle((0, 0, width, height), fill=_adjust_rgb(stone, 0.10))
        tile_w = max(86, width // 5)
        tile_h = max(54, height // 5)
        grout = _adjust_rgb(stone, -0.12)
        for y in range(-tile_h, height + tile_h, tile_h):
            offset = 0 if (y // tile_h) % 2 == 0 else -tile_w // 2
            for x in range(offset, width + tile_w, tile_w):
                n = ((x * 17 + y * 23) % 29) / 28.0
                color = _adjust_rgb(stone, (n - 0.5) * 0.16)
                draw.rectangle((x, y, x + tile_w, y + tile_h), fill=color, outline=grout, width=3)
                draw.line((x + 10, y + tile_h // 2, x + tile_w - 12, y + tile_h // 2 + ((x + y) % 7 - 3)), fill=_adjust_rgb(color, -0.10), width=1)

    elif family == "metal":
        metal = _mix_rgb(base, (36, 36, 34), 0.55)
        for y in range(height):
            t = y / float(max(1, height - 1))
            draw.line((0, y, width, y), fill=_mix_rgb(_adjust_rgb(metal, 0.22), _adjust_rgb(metal, -0.18), t))
        for y in range(0, height, 5):
            n = ((y * 31) % 19) / 18.0
            draw.line((0, y, width, y), fill=_adjust_rgb(metal, (n - 0.5) * 0.18), width=1)
        for x in range(0, width, 80):
            draw.line((x, 0, x + width // 4, height), fill=_adjust_rgb(metal, 0.18), width=2)

    elif family == "planting":
        green = _mix_rgb(base, (45, 105, 55), 0.50)
        draw.rectangle((0, 0, width, height), fill=_adjust_rgb(green, -0.05))
        for i in range(180):
            x = (i * 47 + 19) % width
            y = (i * 29 + 31) % height
            leaf_w = 8 + (i % 18)
            leaf_h = 14 + (i % 26)
            n = ((i * 13) % 31) / 30.0
            color = _adjust_rgb(green, (n - 0.5) * 0.38)
            draw.ellipse((x, y, x + leaf_w, y + leaf_h), fill=color, outline=_adjust_rgb(color, -0.12))
            if i % 3 == 0:
                draw.line((x + leaf_w // 2, y + 2, x + leaf_w // 2, y + leaf_h - 2), fill=_adjust_rgb(color, 0.16), width=1)

    elif family == "fabric":
        textile = _mix_rgb(base, (178, 170, 155), 0.35)
        draw.rectangle((0, 0, width, height), fill=textile)
        for x in range(0, width, 8):
            draw.line((x, 0, x, height), fill=_adjust_rgb(textile, 0.12 if (x // 8) % 2 == 0 else -0.10), width=2)
        for y in range(0, height, 8):
            draw.line((0, y, width, y), fill=_adjust_rgb(textile, -0.08 if (y // 8) % 2 == 0 else 0.10), width=2)

    else:
        _draw_soft_noise(draw, width, height, base, step=18)

    _ensure_dir(os.path.dirname(output_path))
    canvas.save(output_path)
    return output_path

def _extract_palette_from_image(
    path: str,
    max_colors: int = 8,
    *,
    filter_board_background: bool = True,
) -> List[Dict[str, Any]]:
    Image, _, _, ImageStat = _require_pil()
    img = _open_rgb(path)

    small = img.copy()
    small.thumbnail((260, 260))

    pal_img = small.convert("P", palette=Image.ADAPTIVE, colors=max(max_colors * 3, 12))
    palette_raw = pal_img.getpalette() or []
    counts = pal_img.getcolors(maxcolors=260 * 260) or []

    items: List[Tuple[int, Tuple[int, int, int]]] = []
    for count, idx in counts:
        base = int(idx) * 3
        if base + 2 >= len(palette_raw):
            continue
        rgb = (int(palette_raw[base]), int(palette_raw[base + 1]), int(palette_raw[base + 2]))
        if filter_board_background and _is_board_background_color(rgb):
            continue
        items.append((int(count), rgb))

    if not items:
        for count, idx in counts:
            base = int(idx) * 3
            if base + 2 >= len(palette_raw):
                continue
            rgb = (int(palette_raw[base]), int(palette_raw[base + 1]), int(palette_raw[base + 2]))
            items.append((int(count), rgb))

    items.sort(key=lambda x: x[0], reverse=True)
    total = sum(c for c, _ in items) or 1

    result: List[Dict[str, Any]] = []
    for count, rgb in items[:max_colors]:
        result.append(
            {
                "hex": _rgb_to_hex(rgb),
                "rgb": {"r": rgb[0], "g": rgb[1], "b": rgb[2]},
                "percentage": round(float(count) / float(total), 4),
                "luminance": round(_relative_luminance(rgb), 4),
                "warmth": round(_warmth_score(rgb), 4),
                "saturation": round(_color_saturation(rgb), 4),
            }
        )

    if not result:
        stat = ImageStat.Stat(img)
        avg = tuple(int(v) for v in stat.mean[:3])
        result.append(
            {
                "hex": _rgb_to_hex(avg),  # type: ignore[arg-type]
                "rgb": {"r": avg[0], "g": avg[1], "b": avg[2]},
                "percentage": 1.0,
                "luminance": round(_relative_luminance(avg), 4),  # type: ignore[arg-type]
                "warmth": round(_warmth_score(avg), 4),  # type: ignore[arg-type]
                "saturation": round(_color_saturation(avg), 4),  # type: ignore[arg-type]
            }
        )

    return result


def _extract_basic_image_features(path: str) -> Dict[str, Any]:
    _, _, _, ImageStat = _require_pil()
    img = _open_rgb(path)
    stat = ImageStat.Stat(img)

    avg = tuple(int(v) for v in stat.mean[:3])
    lum = _relative_luminance(avg)  # type: ignore[arg-type]
    warm = _warmth_score(avg)  # type: ignore[arg-type]

    if lum < 0.28:
        brightness_label = "dark"
    elif lum < 0.58:
        brightness_label = "medium"
    else:
        brightness_label = "bright"

    if warm > 0.12:
        temperature_label = "warm"
    elif warm < -0.12:
        temperature_label = "cool"
    else:
        temperature_label = "neutral"

    width, height = img.size

    return {
        "file": os.path.basename(path),
        "width": width,
        "height": height,
        "average_color": {
            "hex": _rgb_to_hex(avg),  # type: ignore[arg-type]
            "rgb": {"r": avg[0], "g": avg[1], "b": avg[2]},
        },
        "brightness": brightness_label,
        "temperature": temperature_label,
        "luminance": round(lum, 4),
        "warmth": round(warm, 4),
    }


def _merge_palettes(image_paths: List[str], max_colors: int = 10) -> List[Dict[str, Any]]:
    all_colors: List[Dict[str, Any]] = []

    for path in image_paths:
        try:
            colors = _extract_palette_from_image(path, max_colors=8, filter_board_background=True)
            for c in colors:
                c = dict(c)
                c["source_file"] = os.path.basename(path)
                all_colors.append(c)
        except Exception:
            continue

    by_hex: Dict[str, Dict[str, Any]] = {}
    for c in all_colors:
        hx = str(c.get("hex") or "").upper()
        if not hx:
            continue
        if hx not in by_hex:
            by_hex[hx] = c
        else:
            old_pct = float(by_hex[hx].get("percentage", 0.0))
            new_pct = float(c.get("percentage", 0.0))
            if new_pct > old_pct:
                by_hex[hx] = c

    merged = list(by_hex.values())
    merged.sort(key=lambda x: float(x.get("percentage", 0.0)), reverse=True)

    selected = merged[:max_colors]
    total = sum(float(c.get("percentage", 0.0)) for c in selected) or 1.0
    for c in selected:
        c["percentage_normalized"] = round(float(c.get("percentage", 0.0)) / total, 4)

    return selected


def _material_label_from_region(region_name: str, feature: Dict[str, Any]) -> str:
    """
    Backward-compatible helper retained for safety.
    V3 extraction uses semantic family labels instead of region-only labels.
    """
    scores = feature.get("family_scores") or {}
    family = str(feature.get("material_family") or "").strip().lower()
    if not family and isinstance(scores, dict) and scores:
        family, _ = _pick_best_family({str(k): float(v or 0.0) for k, v in scores.items()})
    if not family:
        family = _fallback_family_from_metrics(feature)
    return _material_labels().get(family, "Material sample")


def _extract_material_crops(
    *,
    job_folder: str,
    image_paths: List[str],
    max_samples: int = 8,
) -> List[Dict[str, Any]]:
    sample_dir = os.path.join(job_folder, MATERIAL_SAMPLES_DIR_NAME)
    _ensure_dir(sample_dir)

    families_order = ["stone", "wood", "glass", "water", "paving", "metal", "planting", "fabric"]
    label_map = _material_labels()
    candidates: List[Dict[str, Any]] = []

    for img_idx, path in enumerate(image_paths):
        try:
            img = _open_rgb(path)
        except Exception:
            continue

        w, h = img.size
        if w < 32 or h < 32:
            continue

        base_sizes = [
            (0.16, 0.16),
            (0.18, 0.14),
            (0.20, 0.16),
            (0.22, 0.18),
            (0.24, 0.18),
            (0.18, 0.22),
        ]

        grid_points_x = [0.04, 0.10, 0.18, 0.28, 0.40, 0.52, 0.64, 0.76]
        grid_points_y = [0.05, 0.12, 0.22, 0.34, 0.48, 0.62, 0.74]

        for bw, bh in base_sizes:
            box_w = max(84, int(bw * w))
            box_h = max(84, int(bh * h))
            if box_w >= w or box_h >= h:
                continue

            for x_norm in grid_points_x:
                for y_norm in grid_points_y:
                    x1 = int(x_norm * w)
                    y1 = int(y_norm * h)
                    if x1 + box_w > w:
                        x1 = max(0, w - box_w)
                    if y1 + box_h > h:
                        y1 = max(0, h - box_h)
                    x2 = min(w, x1 + box_w)
                    y2 = min(h, y1 + box_h)
                    if x2 - x1 < 72 or y2 - y1 < 72:
                        continue

                    crop = img.crop((x1, y1, x2, y2))
                    norm_box = {
                        "x1n": round(x1 / float(w), 4),
                        "y1n": round(y1 / float(h), 4),
                        "x2n": round(x2 / float(w), 4),
                        "y2n": round(y2 / float(h), 4),
                    }
                    metrics = _compute_crop_metrics(crop, box=(x1, y1, x2, y2), source_file=os.path.basename(path), norm_box=norm_box)

                    avg_rgb_dict = metrics.get("average_color", {}).get("rgb", {})
                    avg_rgb = (
                        int(avg_rgb_dict.get("r", 204)),
                        int(avg_rgb_dict.get("g", 204)),
                        int(avg_rgb_dict.get("b", 204)),
                    )
                    if _is_board_background_color(avg_rgb):
                        continue
                    if float(metrics.get("sky_score", 0.0) or 0.0) >= 0.78:
                        continue
                    if float(metrics.get("detail_score", 0.0) or 0.0) <= 0.06:
                        continue

                    family_scores: Dict[str, float] = {}
                    for family in families_order:
                        family_scores[family] = _score_material_family(metrics, family)

                    best_family, confidence = _pick_best_family(family_scores)
                    if confidence < 0.20:
                        best_family = _fallback_family_from_metrics(metrics)
                        confidence = max(confidence, _score_material_family(metrics, best_family))

                    preview_score = round(
                        float(metrics.get("texture_score", 0.0) or 0.0) * 0.42
                        + float(metrics.get("detail_score", 0.0) or 0.0) * 0.28
                        + confidence * 0.30,
                        4,
                    )

                    metrics["family_scores"] = family_scores
                    metrics["material_family"] = best_family
                    metrics["material_label"] = label_map.get(best_family, "Material sample")
                    metrics["confidence"] = round(confidence, 4)
                    metrics["preview_score"] = preview_score
                    metrics["_crop_image"] = crop.copy()
                    candidates.append(metrics)

    candidates.sort(
        key=lambda item: (
            float(item.get("confidence", 0.0) or 0.0),
            float(item.get("texture_score", 0.0) or 0.0),
            float(item.get("detail_score", 0.0) or 0.0),
        ),
        reverse=True,
    )

    selected: List[Dict[str, Any]] = []
    selected_boxes: List[Tuple[int, int, int, int]] = []
    used_family: set[str] = set()

    for family in families_order:
        family_candidates = [c for c in candidates if str(c.get("material_family") or "") == family]
        family_candidates.sort(
            key=lambda item: (
                float(item.get("confidence", 0.0) or 0.0),
                float(item.get("preview_score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        for cand in family_candidates:
            box = cand.get("crop_box") or {}
            tup = (int(box.get("x1", 0)), int(box.get("y1", 0)), int(box.get("x2", 1)), int(box.get("y2", 1)))
            if any(_box_iou(tup, prev) > 0.55 for prev in selected_boxes):
                continue
            selected.append(cand)
            selected_boxes.append(tup)
            used_family.add(family)
            if len(selected) >= max_samples:
                break
            break
        if len(selected) >= max_samples:
            break

    if len(selected) < max_samples:
        for cand in candidates:
            family = str(cand.get("material_family") or "")
            if family in used_family and len(selected) >= min(5, max_samples):
                continue
            box = cand.get("crop_box") or {}
            tup = (int(box.get("x1", 0)), int(box.get("y1", 0)), int(box.get("x2", 1)), int(box.get("y2", 1)))
            if any(_box_iou(tup, prev) > 0.65 for prev in selected_boxes):
                continue
            selected.append(cand)
            selected_boxes.append(tup)
            used_family.add(family)
            if len(selected) >= max_samples:
                break

    output_samples: List[Dict[str, Any]] = []
    for idx, sample in enumerate(selected[:max_samples], start=1):
        crop = sample.pop("_crop_image", None)
        if crop is None:
            continue
        family = str(sample.get("material_family") or f"sample_{idx}")
        refined_crop, refined_metrics, refined_box = _refine_material_patch(
            crop,
            family=family,
            source_file=str(sample.get("source_file") or f"sample_{idx}.png"),
        )

        reference_name = f"reference_{idx:02d}_{family}.png"
        reference_path = os.path.join(sample_dir, reference_name)
        refined_crop.save(reference_path)

        refined_avg = refined_metrics.get("average_color") or sample.get("average_color")
        refined_lum = float(refined_metrics.get("luminance", sample.get("luminance", 0.5)) or 0.5)
        refined_warm = float(refined_metrics.get("warmth", sample.get("warmth", 0.0)) or 0.0)
        refined_sat = float(refined_metrics.get("saturation", sample.get("saturation", 0.0)) or 0.0)

        sample_name = f"material_{idx:02d}_{family}.png"
        sample_path = os.path.join(sample_dir, sample_name)
        swatch_sample = dict(sample)
        swatch_sample["average_color"] = refined_avg
        swatch_sample["luminance"] = round(refined_lum, 4)
        swatch_sample["warmth"] = round(refined_warm, 4)
        swatch_sample["saturation"] = round(refined_sat, 4)
        swatch_sample["hue_name"] = refined_metrics.get("hue_name", sample.get("hue_name"))
        swatch_sample["material_family"] = family
        _generate_material_swatch(sample=swatch_sample, output_path=sample_path, width=520, height=340)

        output_samples.append(
            {
                "file": os.path.relpath(sample_path, job_folder),
                "source_file": sample.get("source_file"),
                "reference_crop_file": os.path.relpath(reference_path, job_folder),
                "material_family": family,
                "label": sample.get("material_label"),
                "confidence": sample.get("confidence"),
                "texture_score": refined_metrics.get("texture_score", sample.get("texture_score")),
                "detail_score": refined_metrics.get("detail_score", sample.get("detail_score")),
                "hue_name": refined_metrics.get("hue_name", sample.get("hue_name")),
                "average_color": refined_avg,
                "brightness": "bright" if refined_lum >= 0.58 else ("dark" if refined_lum <= 0.28 else "medium"),
                "temperature": "warm" if refined_warm > 0.12 else ("cool" if refined_warm < -0.12 else "neutral"),
                "luminance": round(refined_lum, 4),
                "warmth": round(refined_warm, 4),
                "saturation": round(refined_sat, 4),
                "crop_box": sample.get("crop_box"),
                "norm_box": sample.get("norm_box"),
                "sample_crop_box": refined_box,
                "sample_norm_box": refined_metrics.get("norm_box"),
                "family_scores": sample.get("family_scores"),
                "sample_role": "generated_material_swatch",
            }
        )

    return output_samples


def _fit_crop_to_box(img: Any, box_w: int, box_h: int) -> Any:
    Image, _, _, _ = _require_pil()
    src_w, src_h = img.size
    if src_w <= 0 or src_h <= 0:
        return Image.new("RGB", (box_w, box_h), (230, 228, 222))

    scale = max(box_w / float(src_w), box_h / float(src_h))
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = img.resize((new_w, new_h))
    left = max(0, (new_w - box_w) // 2)
    top = max(0, (new_h - box_h) // 2)
    return resized.crop((left, top, left + box_w, top + box_h))


def _draw_label(draw: Any, xy: Tuple[int, int], text: str, font: Any, fill: Tuple[int, int, int] = (45, 45, 45)) -> None:
    try:
        draw.text(xy, text, fill=fill, font=font)
    except Exception:
        draw.text(xy, str(text), fill=fill)




def _material_display_label(sample: Dict[str, Any]) -> str:
    family = str(sample.get("material_family") or "material").lower()
    raw_label = str(sample.get("label") or "").strip().lower()

    if family == "stone":
        return "STONE / MASONRY"
    if family == "wood":
        return "WOOD SOFFIT / SLATTED CEILING"
    if family == "glass":
        return "AQUA-TINTED GLAZING"
    if family == "water":
        return "POOL TILE / CERAMIC FINISH"
    if family == "paving":
        return "LIGHT STONE PAVING"
    if family == "metal":
        return "DARK METAL FRAME"
    if family == "planting":
        return "GROUNDCOVER PLANTING"
    if family == "fabric":
        return "OUTDOOR TEXTILE / UPHOLSTERY"

    if raw_label:
        return raw_label.upper()
    return "MATERIAL SAMPLE"


def _material_category(sample: Dict[str, Any]) -> str:
    family = str(sample.get("material_family") or "material").lower()
    return {
        "stone": "Masonry",
        "wood": "Wood / Ceiling",
        "glass": "Glazing",
        "water": "Pool Finish",
        "paving": "Paving",
        "metal": "Metalwork",
        "planting": "Softscape",
        "fabric": "Textile",
    }.get(family, "Material")


def _material_finish_note(sample: Dict[str, Any]) -> str:
    family = str(sample.get("material_family") or "material").lower()
    temperature = str(sample.get("temperature") or "neutral").lower()
    brightness = str(sample.get("brightness") or "medium").lower()
    hue_name = str(sample.get("hue_name") or "tone").lower()

    hue_note = {
        "gray": "neutral gray",
        "black": "dark",
        "white": "light",
        "cyan": "aqua",
        "blue": "blue-gray",
        "brown": "warm brown",
        "green": "soft green",
    }.get(hue_name, hue_name)

    if family == "stone":
        return "neutral gray stacked stone texture"
    if family == "wood":
        return "warm linear wood grain finish"
    if family == "glass":
        return "cool reflective blue-gray glass finish"
    if family == "water":
        return "aqua reflective ceramic pool tile"
    if family == "paving":
        return "light exterior stone paving finish"
    if family == "metal":
        return "dark bronze / black metal frame finish"
    if family == "planting":
        return "soft green planted edge / groundcover"
    if family == "fabric":
        return "light outdoor textile finish"

    return f"{brightness} {temperature} material finish"


def _material_schedule_name(sample: Dict[str, Any]) -> str:
    family = str(sample.get("material_family") or "material").lower()
    return {
        "stone": "Stacked stone wall finish",
        "wood": "Linear wood soffit / slat finish",
        "glass": "Aqua-tinted reflective glazing",
        "water": "Aqua ceramic pool tile finish",
        "paving": "Light stone pool-deck paving",
        "metal": "Dark metal frame finish",
        "planting": "Low green groundcover planting",
        "fabric": "Outdoor lounge textile",
    }.get(family, _material_display_label(sample).title())


def _material_brand_placeholder(sample: Dict[str, Any]) -> str:
    return "To be specified"


def _make_moodboard_grid(
    *,
    image_paths: List[str],
    palette: List[Dict[str, Any]],
    output_path: str,
    title: str = "RENDEREXPO Moodboard",
    max_tiles: int = 12,
    material_samples: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Moodboard V8 — premium physical flat-lay material board with matched materials schedule.

    This function is presentation-only:
    - no change to extraction
    - no change to generated material swatches
    - no change to JSON structure
    - no change to SD3.5 / routes / dispatch / strength settings
    """
    Image, ImageDraw, ImageFont, _ = _require_pil()

    if not image_paths and not palette:
        raise RuntimeError("Cannot create moodboard grid without images or palette.")

    material_samples = material_samples or []

    canvas_w = 1800
    canvas_h = 1350
    margin = 54
    bg = (244, 242, 236)
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)
    draw = ImageDraw.Draw(canvas)

    ink = (34, 34, 31)
    muted = (105, 101, 94)
    soft = (120, 116, 108)
    line = (205, 201, 192)
    card = (251, 250, 246)
    shadow = (215, 211, 202)
    warm_card = (248, 246, 239)

    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 44)
        subtitle_font = ImageFont.truetype("DejaVuSans.ttf", 21)
        section_font = ImageFont.truetype("DejaVuSans.ttf", 18)
        label_font = ImageFont.truetype("DejaVuSans.ttf", 16)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 14)
        tiny_font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        section_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
        tiny_font = ImageFont.load_default()

    def _text_size(txt: str, font: Any) -> Tuple[int, int]:
        try:
            box = draw.textbbox((0, 0), str(txt), font=font)
            return (box[2] - box[0], box[3] - box[1])
        except Exception:
            return (len(str(txt)) * 7, 14)

    def _sample_abs(sample: Dict[str, Any]) -> Optional[str]:
        rel = str(sample.get("file") or "").strip()
        if not rel:
            return None
        abs_path = rel if os.path.isabs(rel) else os.path.join(os.path.dirname(output_path), rel)
        return abs_path if os.path.isfile(abs_path) else None

    def _draw_shadow_rect(box: Tuple[int, int, int, int], radius: int = 18) -> None:
        x1, y1, x2, y2 = box
        # V8 premium layered shadow: physical depth without CAD/grid styling.
        draw.rounded_rectangle((x1 + 12, y1 + 14, x2 + 12, y2 + 14), radius=radius, fill=(229, 226, 218))
        draw.rounded_rectangle((x1 + 7, y1 + 9, x2 + 7, y2 + 9), radius=radius, fill=(216, 212, 202))
        draw.rounded_rectangle((x1 + 3, y1 + 4, x2 + 3, y2 + 4), radius=radius, fill=(239, 237, 230))
        draw.rounded_rectangle(box, radius=radius, fill=card, outline=line, width=2)

    def _draw_tag(x: int, y: int, title_text: str, note_text: str, anchor: str = "left") -> None:
        title_text = str(title_text or "Material").upper()
        note_text = str(note_text or "material finish cue")
        tw, th = _text_size(title_text, label_font)
        nw, nh = _text_size(note_text, tiny_font)
        w = min(310, max(tw, nw) + 28)
        h = 58

        if anchor == "right":
            x = x - w

        draw.rounded_rectangle((x, y, x + w, y + h), radius=14, fill=(255, 254, 249), outline=line, width=1)
        _draw_label(draw, (x + 14, y + 10), title_text, label_font, ink)
        _draw_label(draw, (x + 14, y + 34), note_text[:52], tiny_font, muted)

    def _draw_material_piece(
        *,
        sample: Dict[str, Any],
        box: Tuple[int, int, int, int],
        label_xy: Tuple[int, int],
        label_anchor: str = "left",
        radius: int = 18,
        rotate_hint: Optional[str] = None,
    ) -> None:
        x1, y1, x2, y2 = box
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        abs_path = _sample_abs(sample)
        family = str(sample.get("material_family") or "material").lower()
        label = _material_display_label(sample)
        finish = _material_finish_note(sample)

        # V8 physical sample shadow: soft, layered, premium.
        draw.rounded_rectangle((x1 + 14, y1 + 16, x2 + 14, y2 + 16), radius=radius, fill=(228, 224, 214))
        draw.rounded_rectangle((x1 + 8, y1 + 10, x2 + 8, y2 + 10), radius=radius, fill=(207, 202, 192))
        draw.rounded_rectangle((x1 + 3, y1 + 4, x2 + 3, y2 + 4), radius=radius, fill=(238, 235, 228))

        if abs_path:
            try:
                img = _open_rgb(abs_path)
                fitted = _fit_crop_to_box(img, w, h)
                # PIL rounded-mask paste for a physical sample feel.
                mask = Image.new("L", (w, h), 0)
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.rounded_rectangle((0, 0, w, h), radius=radius, fill=255)
                canvas.paste(fitted, (x1, y1), mask)
            except Exception:
                draw.rounded_rectangle((x1, y1, x2, y2), radius=radius, fill=(226, 223, 217))
        else:
            draw.rounded_rectangle((x1, y1, x2, y2), radius=radius, fill=(226, 223, 217))

        draw.rounded_rectangle((x1, y1, x2, y2), radius=radius, outline=(170, 166, 156), width=1)

        # Tiny material pin / sample mark.
        draw.ellipse((x1 + 16, y1 + 16, x1 + 28, y1 + 28), fill=(255, 254, 248), outline=line)

        _draw_tag(label_xy[0], label_xy[1], label, finish, anchor=label_anchor)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    display_title = "RENDEREXPO Curated Material Moodboard"
    if "reference" in title.lower():
        display_title = "RENDEREXPO Moodboard Reference"
    elif "space" in title.lower():
        display_title = "RENDEREXPO Curated Material Moodboard"

    _draw_label(draw, (margin, 30), display_title, title_font, ink)
    _draw_label(
        draw,
        (margin, 86),
        "Premium physical material board, palette, atmosphere, and matched materials schedule",
        subtitle_font,
        muted,
    )

    # ------------------------------------------------------------------
    # Resolve materials
    # ------------------------------------------------------------------
    resolved_samples: List[Dict[str, Any]] = []
    for sample in material_samples[:8]:
        if not isinstance(sample, dict):
            continue
        if _sample_abs(sample):
            resolved_samples.append(dict(sample))

    # Fallback placeholders if no swatches are available.
    fallback_labels = [
        ("STONE / MASONRY", "stone"),
        ("WOOD SOFFIT / SLATTED CEILING", "wood"),
        ("AQUA-TINTED GLAZING", "glass"),
        ("POOL TILE / CERAMIC FINISH", "water"),
        ("LIGHT STONE PAVING", "paving"),
        ("DARK METAL FRAME", "metal"),
        ("GROUNDCOVER PLANTING", "planting"),
    ]
    while len(resolved_samples) < 7:
        label, family = fallback_labels[len(resolved_samples) % len(fallback_labels)]
        resolved_samples.append(
            {
                "label": label,
                "material_family": family,
                "brightness": "medium",
                "temperature": "neutral",
                "hue_name": "neutral",
            }
        )

    # ------------------------------------------------------------------
    # Main physical sample canvas
    # ------------------------------------------------------------------
    board_x = margin
    board_y = 135
    board_w = 1145
    board_h = 760
    _draw_shadow_rect((board_x, board_y, board_x + board_w, board_y + board_h), radius=24)

    _draw_label(draw, (board_x + 28, board_y + 24), "PHYSICAL MATERIAL BOARD", section_font, ink)
    _draw_label(draw, (board_x + 28, board_y + 50), "flat-lay material direction curated from the source render", small_font, muted)

    # Physical-style overlapping samples.
    s = resolved_samples

    families = {str(sample.get("material_family") or "").lower() for sample in resolved_samples}
    if {"fabric", "textile"}.intersection(families):
        context_note = "Interior / FF&E material palette"
    elif {"water", "paving", "planting"}.intersection(families):
        context_note = "Exterior architectural material palette"
    else:
        context_note = "Architectural material palette"

    _draw_label(draw, (board_x + board_w - 350, board_y + 28), context_note, small_font, muted)

    # V8 tabletop surface must be drawn BEFORE the physical samples.
    draw.rounded_rectangle(
        (board_x + 36, board_y + 86, board_x + board_w - 36, board_y + board_h - 34),
        radius=20,
        fill=(252, 251, 247),
        outline=(238, 235, 228),
        width=1,
    )

    # Re-draw board header after tabletop surface.
    _draw_label(draw, (board_x + 28, board_y + 24), "PHYSICAL MATERIAL BOARD", section_font, ink)
    _draw_label(draw, (board_x + 28, board_y + 50), "flat-lay material direction curated from the source render", small_font, muted)
    _draw_label(draw, (board_x + board_w - 350, board_y + 28), context_note, small_font, muted)

    # Large stone / masonry tile
    _draw_material_piece(
        sample=s[0],
        box=(board_x + 58, board_y + 135, board_x + 405, board_y + 365),
        label_xy=(board_x + 72, board_y + 382),
        radius=18,
    )

    # Wood plank / slat sample
    _draw_material_piece(
        sample=s[1],
        box=(board_x + 420, board_y + 105, board_x + 585, board_y + 510),
        label_xy=(board_x + 400, board_y + 528),
        radius=16,
    )

    # Glass translucent chip
    _draw_material_piece(
        sample=s[2],
        box=(board_x + 705, board_y + 112, board_x + 1058, board_y + 255),
        label_xy=(board_x + 735, board_y + 272),
        radius=18,
    )

    # Water tone strip
    _draw_material_piece(
        sample=s[3],
        box=(board_x + 690, board_y + 365, board_x + 1068, board_y + 520),
        label_xy=(board_x + 725, board_y + 538),
        radius=22,
    )

    # Paving slab
    _draw_material_piece(
        sample=s[4],
        box=(board_x + 70, board_y + 520, board_x + 392, board_y + 680),
        label_xy=(board_x + 88, board_y + 694),
        radius=18,
    )

    # Dark metal strip
    _draw_material_piece(
        sample=s[5],
        box=(board_x + 445, board_y + 615, board_x + 840, board_y + 678),
        label_xy=(board_x + 495, board_y + 696),
        radius=14,
    )

    # Planting / organic swatch
    _draw_material_piece(
        sample=s[6],
        box=(board_x + 915, board_y + 565, board_x + 1055, board_y + 705),
        label_xy=(board_x + 850, board_y + 722),
        label_anchor="left",
        radius=18,
    )

    # Small physical-sample decorative circles / paint pucks.
    puck_colors = []
    for c in palette[:4]:
        hx = str(c.get("hex") or "#CCCCCC")
        puck_colors.append(_hex_to_rgb(hx))
    while len(puck_colors) < 4:
        puck_colors.append((210, 205, 195))

    puck_y = board_y + 82
    for i, rgb in enumerate(puck_colors[:4]):
        px = board_x + 48 + i * 42
        draw.ellipse((px, puck_y, px + 30, puck_y + 30), fill=rgb, outline=(150, 146, 138), width=1)

    # ------------------------------------------------------------------
    # Source reference card
    # ------------------------------------------------------------------
    ref_x = board_x + board_w + 30
    ref_y = board_y
    ref_w = canvas_w - margin - ref_x
    ref_h = 455

    _draw_shadow_rect((ref_x, ref_y, ref_x + ref_w, ref_y + ref_h), radius=22)
    _draw_label(draw, (ref_x + 24, ref_y + 22), "SOURCE RENDER", section_font, ink)
    _draw_label(draw, (ref_x + 24, ref_y + 48), "visual reference analyzed by RENDEREXPO", small_font, muted)

    hero_x = ref_x + 24
    hero_y = ref_y + 84
    hero_w = ref_w - 48
    hero_h = 295

    if image_paths:
        try:
            hero = _open_rgb(image_paths[0])
            hero_img = _fit_crop_to_box(hero, hero_w, hero_h)
            canvas.paste(hero_img, (hero_x, hero_y))
            draw.rectangle((hero_x, hero_y, hero_x + hero_w, hero_y + hero_h), outline=line, width=2)
        except Exception:
            draw.rectangle((hero_x, hero_y, hero_x + hero_w, hero_y + hero_h), fill=(225, 222, 216), outline=line)
            _draw_label(draw, (hero_x + 18, hero_y + 18), os.path.basename(image_paths[0]), small_font, muted)

    _draw_label(draw, (hero_x, hero_y + hero_h + 18), "HERO REFERENCE", label_font, ink)

    # ------------------------------------------------------------------
    # Atmosphere notes card
    # ------------------------------------------------------------------
    notes_y = ref_y + ref_h + 28
    notes_h = 277
    _draw_shadow_rect((ref_x, notes_y, ref_x + ref_w, notes_y + notes_h), radius=22)

    _draw_label(draw, (ref_x + 24, notes_y + 24), "MATERIAL DIRECTION", section_font, ink)

    direction_lines = []
    for sample in resolved_samples[:6]:
        lbl = _material_display_label(sample)
        note = _material_finish_note(sample)
        direction_lines.append(f"{lbl}: {note}")

    y = notes_y + 62
    for line_text in direction_lines[:6]:
        _draw_label(draw, (ref_x + 24, y), "• " + line_text[:66], small_font, muted)
        y += 31

    # ------------------------------------------------------------------
    # Clean color palette strip
    # ------------------------------------------------------------------
    palette_y = 930
    palette_h = 88
    _draw_shadow_rect((margin, palette_y, canvas_w - margin, palette_y + palette_h), radius=22)

    _draw_label(draw, (margin + 26, palette_y + 18), "COLOR PALETTE", section_font, ink)

    swatch_count = max(1, min(len(palette), 8))
    swatch_gap = 12
    pal_x = margin + 220
    swatch_y = palette_y + 20
    swatch_w = int((canvas_w - margin - pal_x - swatch_gap * (swatch_count - 1)) / swatch_count)
    swatch_h = 48

    for idx, color in enumerate(palette[:swatch_count]):
        hx = str(color.get("hex") or "#CCCCCC").upper()
        rgb = _hex_to_rgb(hx)
        x = pal_x + idx * (swatch_w + swatch_gap)

        draw.rounded_rectangle(
            (x, swatch_y, x + swatch_w, swatch_y + swatch_h),
            radius=12,
            fill=rgb,
            outline=(150, 148, 142),
            width=1,
        )
        text_fill = (20, 20, 20) if _relative_luminance(rgb) > 0.55 else (245, 245, 245)
        _draw_label(draw, (x + 12, swatch_y + 10), hx, tiny_font, text_fill)

        pct = color.get("percentage_normalized", color.get("percentage"))
        if pct is not None:
            try:
                pct_text = f"{float(pct) * 100:.0f}%"
                _draw_label(draw, (x + 12, swatch_y + 28), pct_text, tiny_font, text_fill)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Matched materials / material schedule
    # ------------------------------------------------------------------
    schedule_y = 1045
    schedule_h = 250
    _draw_shadow_rect((margin, schedule_y, canvas_w - margin, schedule_y + schedule_h), radius=22)

    _draw_label(draw, (margin + 26, schedule_y + 18), "MATCHED MATERIALS", section_font, ink)
    _draw_label(draw, (margin + 26, schedule_y + 43), "Material Sheet — suggested systems extracted from the source render", tiny_font, muted)

    table_x = margin + 26
    table_y = schedule_y + 82
    row_h = 30

    col_img = table_x
    col_cat = table_x + 58
    col_name = table_x + 245
    col_finish = table_x + 660
    col_brand = table_x + 1180

    # Header row
    header_y = table_y - 30
    draw.rounded_rectangle((table_x - 10, header_y - 8, canvas_w - margin - 24, header_y + 18), radius=8, fill=(242, 240, 234))
    _draw_label(draw, (col_img, header_y), "IMAGE", tiny_font, soft)
    _draw_label(draw, (col_cat, header_y), "CATEGORY", tiny_font, soft)
    _draw_label(draw, (col_name, header_y), "MATERIAL / PRODUCT NAME", tiny_font, soft)
    _draw_label(draw, (col_finish, header_y), "FINISH DIRECTION", tiny_font, soft)
    _draw_label(draw, (col_brand, header_y), "SOURCE / BRAND", tiny_font, soft)

    schedule_samples = resolved_samples[:6]
    for idx, sample in enumerate(schedule_samples):
        y = table_y + idx * row_h

        if idx % 2 == 0:
            draw.rounded_rectangle((table_x - 10, y - 5, canvas_w - margin - 24, y + row_h - 5), radius=8, fill=(247, 246, 241))

        abs_path = _sample_abs(sample)
        if abs_path:
            try:
                thumb = _fit_crop_to_box(_open_rgb(abs_path), 38, 24)
                canvas.paste(thumb, (col_img, y - 1))
                draw.rectangle((col_img, y - 1, col_img + 38, y + 23), outline=line, width=1)
            except Exception:
                draw.rectangle((col_img, y - 1, col_img + 38, y + 23), fill=(220, 217, 210), outline=line)
        else:
            draw.rectangle((col_img, y - 1, col_img + 38, y + 23), fill=(220, 217, 210), outline=line)

        _draw_label(draw, (col_cat, y + 3), _material_category(sample)[:28], tiny_font, ink)
        _draw_label(draw, (col_name, y + 3), _material_schedule_name(sample)[:50], tiny_font, ink)
        _draw_label(draw, (col_finish, y + 3), _material_finish_note(sample)[:58], tiny_font, muted)
        _draw_label(draw, (col_brand, y + 3), _material_brand_placeholder(sample), tiny_font, muted)

    _ensure_dir(os.path.dirname(output_path))
    canvas.save(output_path)
    return output_path


def _build_style_summary(palette: List[Dict[str, Any]], assets: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not palette:
        return {
            "palette_temperature": "neutral",
            "brightness": "medium",
            "style_summary": "neutral architectural moodboard",
        }

    warmth_values = [float(c.get("warmth", 0.0)) for c in palette]
    lum_values = [float(c.get("luminance", 0.5)) for c in palette]

    avg_warmth = sum(warmth_values) / max(1, len(warmth_values))
    avg_lum = sum(lum_values) / max(1, len(lum_values))

    if avg_warmth > 0.10:
        temp = "warm"
    elif avg_warmth < -0.10:
        temp = "cool"
    else:
        temp = "neutral"

    if avg_lum < 0.32:
        brightness = "dark"
    elif avg_lum > 0.62:
        brightness = "bright"
    else:
        brightness = "medium"

    summary = f"{brightness} {temp} architectural moodboard"

    return {
        "palette_temperature": temp,
        "brightness": brightness,
        "average_warmth": round(avg_warmth, 4),
        "average_luminance": round(avg_lum, 4),
        "style_summary": summary,
    }


# ---------------------------------------------------------------------
# Analysis builders
# ---------------------------------------------------------------------

def _analyze_images(
    image_paths: List[str],
    *,
    job_folder: Optional[str] = None,
    material_samples: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if not image_paths:
        raise RuntimeError("Moodboard analysis requires at least one image.")

    palette = _merge_palettes(image_paths, max_colors=10)

    assets: List[Dict[str, Any]] = []
    for path in image_paths:
        try:
            assets.append(_extract_basic_image_features(path))
        except Exception as exc:
            assets.append({"file": os.path.basename(path), "error": str(exc)})

    summary = _build_style_summary(palette, assets)
    material_samples = material_samples or []

    return {
        "palette": palette,
        "assets": assets,
        "material_samples": material_samples,
        "summary": summary,
        "analyzed_files": [os.path.basename(p) for p in image_paths],
    }


def _write_analysis_outputs(job_folder: str, image_paths: List[str], title: str) -> Dict[str, str]:
    material_samples = _extract_material_crops(job_folder=job_folder, image_paths=image_paths, max_samples=8)
    analysis = _analyze_images(image_paths, job_folder=job_folder, material_samples=material_samples)

    palette_path = os.path.join(job_folder, PALETTE_JSON_NAME)
    assets_path = os.path.join(job_folder, EXTRACTED_ASSETS_JSON_NAME)
    grid_path = os.path.join(job_folder, MOODBOARD_GRID_NAME)

    _write_json(
        palette_path,
        {
            "created_at": _utc_now(),
            "palette": analysis["palette"],
            "summary": analysis["summary"],
            "palette_policy": {
                "filter_board_background": True,
                "reason": "Avoid white/off-white moodboard canvas dominating architectural material palette.",
            },
        },
    )

    _write_json(
        assets_path,
        {
            "created_at": _utc_now(),
            "assets": analysis["assets"],
            "material_samples": analysis["material_samples"],
            "summary": analysis["summary"],
            "analyzed_files": analysis["analyzed_files"],
            "layout_version": "moodboard_v8_premium_physical_flatlay",
        },
    )

    _make_moodboard_grid(
        image_paths=image_paths,
        palette=analysis["palette"],
        output_path=grid_path,
        title=title,
        max_tiles=12,
        material_samples=analysis["material_samples"],
    )

    return {
        "moodboard_grid_png": grid_path,
        "palette_json": palette_path,
        "extracted_assets_json": assets_path,
    }


def _load_moodboard_context_from_folder(folder: str) -> Dict[str, Any]:
    palette_path = os.path.join(folder, PALETTE_JSON_NAME)
    assets_path = os.path.join(folder, EXTRACTED_ASSETS_JSON_NAME)
    meta_path = os.path.join(folder, "meta.json")

    return {
        "palette": _read_json(palette_path),
        "assets": _read_json(assets_path),
        "meta": _read_json(meta_path),
    }


def _normalize_material_phrase(label: str) -> str:
    label = (label or "").strip().lower()

    mapping = {
        "stone / masonry": "gray stone masonry walls",
        "stone": "gray stone masonry walls",
        "wood soffit / slats": "warm wood slat ceiling",
        "wood": "warm wood slat ceiling",
        "glass / glazing": "glass glazing",
        "glass": "glass glazing",
        "pool water": "aqua pool water reflections",
        "water": "aqua pool water reflections",
        "light stone paving": "light stone paving",
        "paving": "light stone paving",
        "dark metal frame": "dark metal frames",
        "metal": "dark metal frames",
        "landscape planting": "soft indoor planting",
        "planting": "soft indoor planting",
        "outdoor fabric": "light lounge fabric",
        "fabric": "light lounge fabric",
    }

    if label in mapping:
        return mapping[label]

    label = label.replace("/", " ").replace("_", " ")
    label = " ".join(label.split())
    return label[:48]


def _compact_client_prompt(prompt: str, max_words: int = 22) -> str:
    prompt = (prompt or "").strip()
    if not prompt:
        return ""

    replacements = {
        "generate a": "",
        "generate an": "",
        "inspired by this moodboard": "",
        "using the same": "",
        "with the same": "",
        "and refined resort atmosphere": "resort atmosphere",
    }

    cleaned = prompt
    lower = cleaned.lower()
    for old, new in replacements.items():
        if old in lower:
            cleaned = cleaned.replace(old, new)
            cleaned = cleaned.replace(old.capitalize(), new)
            lower = cleaned.lower()

    words = [w.strip(" ,.") for w in cleaned.split() if w.strip(" ,.")]
    return " ".join(words[:max_words])


def _style_text_from_analysis(job_folder: str, fallback_images: List[str]) -> str:
    """
    Build a compact material direction string for SD3.5.

    IMPORTANT:
    - Keep this short.
    - Do not include hex colors here.
    - Put material intent before general style words.
    - The SD3.5/CLIP side can truncate long prompts, so this function must prioritize.
    """
    palette_path = os.path.join(job_folder, PALETTE_JSON_NAME)
    assets_path = os.path.join(job_folder, EXTRACTED_ASSETS_JSON_NAME)

    palette_data = _read_json(palette_path)
    assets_data = _read_json(assets_path)

    if not palette_data or not assets_data:
        try:
            _write_analysis_outputs(
                job_folder=job_folder,
                image_paths=fallback_images,
                title="RENDEREXPO Moodboard Reference",
            )
            palette_data = _read_json(palette_path)
            assets_data = _read_json(assets_path)
        except Exception:
            pass

    material_phrases: List[str] = []
    for sample in (assets_data.get("material_samples") or [])[:7]:
        if not isinstance(sample, dict):
            continue

        label = str(sample.get("label") or "").strip()
        family = str(sample.get("material_family") or "").strip()

        phrase = _normalize_material_phrase(label or family)
        if phrase and phrase.lower() not in [x.lower() for x in material_phrases]:
            material_phrases.append(phrase)

    if not material_phrases:
        material_phrases = [
            "warm wood slat ceiling",
            "gray stone masonry walls",
            "glass glazing",
            "light stone paving",
            "dark metal frames",
        ]

    # Keep only the strongest, most visual material cues.
    material_text = ", ".join(material_phrases[:6])

    summary = {}
    if isinstance(palette_data.get("summary"), dict):
        summary.update(palette_data.get("summary") or {})
    if isinstance(assets_data.get("summary"), dict):
        summary.update(assets_data.get("summary") or {})

    temp = str(summary.get("palette_temperature") or "").strip().lower()
    brightness = str(summary.get("brightness") or "").strip().lower()

    atmosphere_parts: List[str] = []
    if temp in {"warm", "cool", "neutral"}:
        atmosphere_parts.append(f"{temp} palette")
    if brightness in {"dark", "medium", "bright"}:
        atmosphere_parts.append(f"{brightness} mood")

    atmosphere_parts.append("luxury resort spa atmosphere")
    atmosphere_parts.append("soft evening lighting")

    return f"{material_text}, {', '.join(atmosphere_parts)}"


def _build_moodboard_to_space_prompt(job_folder: str, meta: Dict[str, Any], moodboard_images: List[str]) -> str:
    user_prompt = _compact_client_prompt(str(meta.get("prompt") or "").strip(), max_words=22)
    style_text = _style_text_from_analysis(job_folder, moodboard_images)

    # Put the highest-value visual instructions first, before CLIP truncation can remove them.
    base = (
        "luxury interior pool lounge, "
        f"{style_text}, "
        "photorealistic architectural visualization, realistic furniture, refined spatial depth"
    )

    if user_prompt:
        return f"{base}, {user_prompt}"

    return base


def _build_apply_moodboard_prompt(
    *,
    job_folder: str,
    meta: Dict[str, Any],
    moodboard_folder: Optional[str],
) -> str:
    user_prompt = str(meta.get("prompt") or "").strip()

    moodboard_images: List[str] = []
    if moodboard_folder and os.path.isdir(moodboard_folder):
        moodboard_images = _list_moodboard_images(moodboard_folder)
        if not moodboard_images:
            grid = os.path.join(moodboard_folder, MOODBOARD_GRID_NAME)
            if os.path.isfile(grid):
                moodboard_images = [grid]

    style_text = ""
    if moodboard_folder and os.path.isdir(moodboard_folder):
        style_text = _style_text_from_analysis(moodboard_folder, moodboard_images or [])

    if not style_text:
        style_text = "cohesive moodboard-inspired material palette, refined design direction"

    base = (
        "apply the moodboard design direction to the existing render, preserve the original room/building layout, "
        "preserve camera angle and main composition, update materials, colors, finishes, furniture styling, "
        "lighting mood, and decorative atmosphere with photorealistic architectural visualization quality"
    )

    if user_prompt:
        return f"{base}, {style_text}, client direction: {user_prompt}"

    return f"{base}, {style_text}"


# ---------------------------------------------------------------------
# Public runners
# ---------------------------------------------------------------------

def run_space_to_moodboard(job: Any, payload: Dict[str, Any]) -> Dict[str, str]:
    """
    Analyze an uploaded space image and create a real moodboard output package.

    Expected files in job_folder:
      - space.png

    Writes:
      - moodboard_grid.png
      - palette.json
      - extracted_assets.json
    """
    job_folder = _job_folder_from_payload(payload)

    space_image = _find_first_existing(job_folder, ["space.png", "input.png", "image.png"])
    if not space_image:
        raise RuntimeError("space_to_moodboard requires space.png, input.png, or image.png inside job_folder.")

    outputs = _write_analysis_outputs(
        job_folder=job_folder,
        image_paths=[space_image],
        title="RENDEREXPO Space Moodboard",
    )

    return {
        "moodboard_grid_png": outputs["moodboard_grid_png"],
        "palette_json": outputs["palette_json"],
        "extracted_assets_json": outputs["extracted_assets_json"],
        "mode": "space_to_moodboard",
    }


def run_sd35_moodboard_to_space(job: Any, payload: Dict[str, Any]) -> Dict[str, str]:
    """
    Generate a space inspired by moodboard images using the existing SD3.5 text2img lane.

    This first safe implementation:
    - analyzes moodboard images
    - creates moodboard_grid.png / palette.json / extracted_assets.json
    - builds an SD3.5 prompt from the moodboard analysis + client prompt
    - calls the existing SD3.5 text2img runner

    It does NOT yet use IP-Adapter.
    That can be added later without changing this external contract.
    """
    job_folder = _job_folder_from_payload(payload)
    meta = dict(payload)

    moodboard_images = _list_moodboard_images(job_folder)
    if not moodboard_images:
        raise RuntimeError("sd35_moodboard_to_space requires moodboard_000.png, moodboard_001.png, etc.")

    analysis_outputs = _write_analysis_outputs(
        job_folder=job_folder,
        image_paths=moodboard_images,
        title="RENDEREXPO Moodboard Reference",
    )

    prompt = _build_moodboard_to_space_prompt(job_folder, meta, moodboard_images)

    exec_payload = dict(meta)
    exec_payload["job_folder"] = job_folder
    exec_payload["prompt"] = prompt
    exec_payload.setdefault(
        "negative_prompt",
        (
            "low quality, blurry, distorted architecture, messy layout, unrealistic materials, "
            "bad furniture, warped perspective, cartoon, painterly, noisy, watermark, logo, text"
        ),
    )
    exec_payload["type"] = "text2img"
    exec_payload["pipeline_key"] = "sd35::moodboard_to_space"
    exec_payload["moodboard_analysis"] = {
        "palette_json": PALETTE_JSON_NAME,
        "extracted_assets_json": EXTRACTED_ASSETS_JSON_NAME,
        "moodboard_grid": MOODBOARD_GRID_NAME,
        "conditioning_mode": "text_prompt_from_moodboard_analysis",
        "ip_adapter_enabled": False,
    }

    from app.gpu.sd35 import run_sd35_txt2img

    output_path = run_sd35_txt2img(
        job=job,
        payload=exec_payload,
    )

    if not os.path.isfile(output_path):
        raise RuntimeError(f"SD3.5 moodboard_to_space output was not created: {output_path}")

    return {
        "output_png": output_path,
        "moodboard_grid_png": analysis_outputs["moodboard_grid_png"],
        "palette_json": analysis_outputs["palette_json"],
        "extracted_assets_json": analysis_outputs["extracted_assets_json"],
        "mode": "sd35_moodboard_to_space",
        "conditioning_mode": "text_prompt_from_moodboard_analysis",
    }


def run_sd35_apply_moodboard_to_render(job: Any, payload: Dict[str, Any]) -> Dict[str, str]:
    """
    Apply a moodboard direction to an existing render using the existing SD3.5 img2img lane.

    Expected files in job_folder:
      - input.png

    Expected meta:
      - moodboard_job_id OR moodboard_folder_abs optional

    This first safe implementation:
    - reads palette/material context if available
    - builds an SD3.5 img2img prompt
    - calls the existing SD3.5 img2img runner

    It does NOT yet use IP-Adapter.
    That can be added later without changing this external contract.
    """
    job_folder = _job_folder_from_payload(payload)
    meta = dict(payload)

    input_image = _find_first_existing(job_folder, ["input.png", "image.png", "target.png", "render.png"])
    if not input_image:
        raise RuntimeError("sd35_apply_moodboard_to_render requires input.png, image.png, target.png, or render.png.")

    moodboard_folder: Optional[str] = None

    explicit_folder = meta.get("moodboard_folder_abs")
    if isinstance(explicit_folder, str) and explicit_folder.strip() and os.path.isdir(explicit_folder.strip()):
        moodboard_folder = explicit_folder.strip()

    if not moodboard_folder:
        mb_job_id = str(meta.get("moodboard_job_id") or "").strip()
        if mb_job_id:
            moodboard_folder = _find_job_folder_by_job_id(mb_job_id)

    prompt = _build_apply_moodboard_prompt(
        job_folder=job_folder,
        meta=meta,
        moodboard_folder=moodboard_folder,
    )

    exec_payload = dict(meta)
    exec_payload["job_folder"] = job_folder
    exec_payload["input_image"] = input_image
    exec_payload["prompt"] = prompt
    exec_payload.setdefault(
        "negative_prompt",
        (
            "low quality, blurry, distorted structure, changed layout, broken room geometry, "
            "warped windows, messy architecture, unrealistic materials, cartoon, painterly, watermark, logo, text"
        ),
    )
    exec_payload["type"] = "img2img"
    exec_payload["pipeline_key"] = "sd35::apply_moodboard_to_render"
    exec_payload.setdefault("strength", float(meta.get("strength", 0.45)))
    exec_payload.setdefault("preserve_input_aspect_ratio", True)
    exec_payload.setdefault("explicit_dimensions", False)
    exec_payload["moodboard_analysis"] = {
        "moodboard_folder": moodboard_folder,
        "conditioning_mode": "text_prompt_from_moodboard_analysis",
        "ip_adapter_enabled": False,
    }

    from app.gpu.sd35 import run_sd35_img2img

    output_path = run_sd35_img2img(
        job=job,
        payload=exec_payload,
    )

    if not os.path.isfile(output_path):
        raise RuntimeError(f"SD3.5 apply_moodboard_to_render output was not created: {output_path}")

    return {
        "output_png": output_path,
        "input_image": input_image,
        "moodboard_folder": moodboard_folder or "",
        "mode": "sd35_apply_moodboard_to_render",
        "conditioning_mode": "text_prompt_from_moodboard_analysis",
    }