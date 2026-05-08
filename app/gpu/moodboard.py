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
        sample_name = f"material_{idx:02d}_{family}.png"
        sample_path = os.path.join(sample_dir, sample_name)
        refined_crop.save(sample_path)

        refined_avg = refined_metrics.get("average_color") or sample.get("average_color")
        refined_lum = float(refined_metrics.get("luminance", sample.get("luminance", 0.5)) or 0.5)
        refined_warm = float(refined_metrics.get("warmth", sample.get("warmth", 0.0)) or 0.0)
        refined_sat = float(refined_metrics.get("saturation", sample.get("saturation", 0.0)) or 0.0)

        output_samples.append(
            {
                "file": os.path.relpath(sample_path, job_folder),
                "source_file": sample.get("source_file"),
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
                "sample_role": "texture_patch",
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




def _material_finish_note(sample: Dict[str, Any]) -> str:
    family = str(sample.get("material_family") or "material").lower()
    temperature = str(sample.get("temperature") or "neutral").lower()
    brightness = str(sample.get("brightness") or "medium").lower()
    hue_name = str(sample.get("hue_name") or "tone").lower()

    hue_note = {
        "gray": "neutral",
        "black": "dark",
        "white": "light",
        "cyan": "aqua",
    }.get(hue_name, hue_name)

    if family == "stone":
        if hue_note in {"neutral", "dark", "light"}:
            return f"{hue_note} stone texture"
        return f"{temperature} {hue_note} stone texture"
    if family == "wood":
        return f"{temperature} wood grain sample"
    if family == "glass":
        return f"{brightness} {hue_note} glazing cue"
    if family == "water":
        return f"{brightness} aqua reflective surface"
    if family == "paving":
        return f"{brightness} paving texture sample"
    if family == "metal":
        return f"{brightness} metal finish cue"
    if family == "planting":
        return f"{temperature} organic foliage texture"
    if family == "fabric":
        return f"{brightness} textile finish cue"
    return "material texture sample"


def _make_moodboard_grid(
    *,
    image_paths: List[str],
    palette: List[Dict[str, Any]],
    output_path: str,
    title: str = "RENDEREXPO Moodboard",
    max_tiles: int = 12,
    material_samples: Optional[List[Dict[str, Any]]] = None,
) -> str:
    Image, ImageDraw, ImageFont, _ = _require_pil()

    if not image_paths and not palette:
        raise RuntimeError("Cannot create moodboard grid without images or palette.")

    material_samples = material_samples or []

    canvas_w = 1800
    canvas_h = 1200
    margin = 52
    gap = 22

    bg = (244, 242, 236)
    ink = (35, 35, 33)
    muted = (104, 101, 95)
    line = (206, 202, 194)
    card = (251, 250, 246)

    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)
    draw = ImageDraw.Draw(canvas)

    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 44)
        subtitle_font = ImageFont.truetype("DejaVuSans.ttf", 22)
        label_font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    _draw_label(draw, (margin, 28), title, title_font, ink)
    _draw_label(draw, (margin, 84), "Material palette, extracted textures, curated surface direction", subtitle_font, muted)

    hero_x = margin
    hero_y = 132
    hero_w = 1030
    hero_h = 700

    draw.rounded_rectangle(
        (hero_x - 12, hero_y - 12, hero_x + hero_w + 12, hero_y + hero_h + 54),
        radius=18,
        fill=card,
        outline=line,
        width=2,
    )

    if image_paths:
        try:
            hero = _open_rgb(image_paths[0])
            hero_img = _fit_crop_to_box(hero, hero_w, hero_h)
            canvas.paste(hero_img, (hero_x, hero_y))
        except Exception:
            draw.rectangle((hero_x, hero_y, hero_x + hero_w, hero_y + hero_h), fill=(225, 222, 216))
            _draw_label(draw, (hero_x + 20, hero_y + 20), os.path.basename(image_paths[0]), label_font, muted)

    _draw_label(draw, (hero_x, hero_y + hero_h + 18), "HERO REFERENCE", label_font, ink)

    stack_x = hero_x + hero_w + 46
    stack_y = hero_y
    stack_w = canvas_w - margin - stack_x
    card_h = 150

    resolved_samples: List[Dict[str, Any]] = []
    for sample in material_samples[:5]:
        rel = str(sample.get("file") or "")
        if not rel:
            continue
        abs_path = rel if os.path.isabs(rel) else os.path.join(os.path.dirname(output_path), rel)
        if os.path.isfile(abs_path):
            s = dict(sample)
            s["_abs_path"] = abs_path
            resolved_samples.append(s)

    for idx in range(6):
        y = stack_y + idx * (card_h + gap)
        draw.rounded_rectangle(
            (stack_x, y, stack_x + stack_w, y + card_h),
            radius=16,
            fill=card,
            outline=line,
            width=2,
        )

        label = "Material sample"
        sample_img = None

        if idx < len(resolved_samples):
            sample = resolved_samples[idx]
            label = str(sample.get("label") or sample.get("region") or label)
            try:
                sample_img = _open_rgb(str(sample["_abs_path"]))
            except Exception:
                sample_img = None
        elif image_paths:
            try:
                ref = _open_rgb(image_paths[0])
                w, h = ref.size
                fx1 = int((0.08 + 0.11 * idx) * w) % max(1, w - 1)
                fy1 = int((0.12 + 0.13 * idx) * h) % max(1, h - 1)
                fx2 = min(w, fx1 + max(80, int(0.36 * w)))
                fy2 = min(h, fy1 + max(80, int(0.24 * h)))
                sample_img = ref.crop((fx1, fy1, fx2, fy2))
                label = ["Surface", "Stone / wall", "Wood / ceiling", "Glass / light", "Water / accent"][idx]
            except Exception:
                sample_img = None

        img_x = stack_x + 16
        img_y = y + 16
        img_w = 190
        img_h = card_h - 32

        if sample_img is not None:
            fitted = _fit_crop_to_box(sample_img, img_w, img_h)
            canvas.paste(fitted, (img_x, img_y))
        else:
            draw.rounded_rectangle((img_x, img_y, img_x + img_w, img_y + img_h), radius=12, fill=(226, 223, 217))

        _draw_label(draw, (stack_x + 230, y + 38), label.upper(), label_font, ink)
        family_text = str((sample.get("material_family") if idx < len(resolved_samples) else "sample") if idx < len(resolved_samples) else "material cue").replace("_", " ").title() if idx < len(resolved_samples) else "material cue"
        _draw_label(draw, (stack_x + 230, y + 68), family_text, small_font, muted)
        if idx < len(resolved_samples):
            finish_note = _material_finish_note(resolved_samples[idx])
            _draw_label(draw, (stack_x + 230, y + 96), finish_note, small_font, muted)
        else:
            _draw_label(draw, (stack_x + 230, y + 96), "RENDEREXPO material direction", small_font, muted)

    lower_y = hero_y + hero_h + 92
    draw.rounded_rectangle(
        (margin, lower_y, canvas_w - margin, canvas_h - margin),
        radius=18,
        fill=card,
        outline=line,
        width=2,
    )

    _draw_label(draw, (margin + 24, lower_y + 22), "COLOR + ATMOSPHERE", label_font, ink)

    pal_x = margin + 24
    pal_y = lower_y + 64
    pal_w = canvas_w - margin * 2 - 48
    swatch_count = max(1, min(len(palette), 8))
    swatch_gap = 16
    swatch_w = int((pal_w - swatch_gap * (swatch_count - 1)) / swatch_count)
    swatch_h = 104

    for idx, color in enumerate(palette[:swatch_count]):
        hx = str(color.get("hex") or "#CCCCCC").upper()
        x = pal_x + idx * (swatch_w + swatch_gap)
        y = pal_y

        rgb = _hex_to_rgb(hx)
        draw.rounded_rectangle((x, y, x + swatch_w, y + swatch_h), radius=14, fill=rgb, outline=(150, 148, 142), width=1)

        text_fill = (20, 20, 20) if _relative_luminance(rgb) > 0.55 else (245, 245, 245)
        _draw_label(draw, (x + 14, y + 30), hx, label_font, text_fill)

        pct = color.get("percentage_normalized", color.get("percentage"))
        if pct is not None:
            try:
                pct_text = f"{float(pct) * 100:.0f}%"
                _draw_label(draw, (x + 14, y + 62), pct_text, small_font, text_fill)
            except Exception:
                pass

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
            "layout_version": "moodboard_v4_texture_samples",
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


def _style_text_from_analysis(job_folder: str, fallback_images: List[str]) -> str:
    palette_path = os.path.join(job_folder, PALETTE_JSON_NAME)
    assets_path = os.path.join(job_folder, EXTRACTED_ASSETS_JSON_NAME)

    palette_data = _read_json(palette_path)
    assets_data = _read_json(assets_path)

    if not palette_data or not assets_data:
        try:
            _write_analysis_outputs(job_folder=job_folder, image_paths=fallback_images, title="RENDEREXPO Moodboard Reference")
            palette_data = _read_json(palette_path)
            assets_data = _read_json(assets_path)
        except Exception:
            pass

    colors = []
    for c in (palette_data.get("palette") or [])[:5]:
        hx = c.get("hex")
        if hx:
            colors.append(str(hx))

    material_labels: List[str] = []
    for sample in (assets_data.get("material_samples") or [])[:6]:
        if isinstance(sample, dict):
            label = str(sample.get("label") or "").strip()
            if label and label.lower() not in [x.lower() for x in material_labels]:
                material_labels.append(label)

    summary = {}
    if isinstance(palette_data.get("summary"), dict):
        summary.update(palette_data.get("summary") or {})
    if isinstance(assets_data.get("summary"), dict):
        summary.update(assets_data.get("summary") or {})

    temp = str(summary.get("palette_temperature") or "neutral")
    brightness = str(summary.get("brightness") or "medium")
    color_text = ", ".join(colors) if colors else "cohesive architectural palette"
    material_text = ", ".join(material_labels) if material_labels else "wood, stone, glass, fabric, metal accents"

    return (
        f"{brightness} {temp} palette, colors {color_text}, "
        f"materials {material_text}, refined architectural interior direction"
    )


def _build_moodboard_to_space_prompt(job_folder: str, meta: Dict[str, Any], moodboard_images: List[str]) -> str:
    user_prompt = str(meta.get("prompt") or "").strip()
    style_text = _style_text_from_analysis(job_folder, moodboard_images)

    base = (
        "photorealistic luxury architectural interior, cohesive moodboard-driven materials, "
        "realistic furniture, warm ambient lighting, refined spatial depth"
    )

    if user_prompt:
        return f"{base}, {style_text}, {user_prompt}"

    return f"{base}, {style_text}"


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