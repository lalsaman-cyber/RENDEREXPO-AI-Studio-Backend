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


def _extract_palette_from_image(path: str, max_colors: int = 8) -> List[Dict[str, Any]]:
    Image, _, _, ImageStat = _require_pil()
    img = _open_rgb(path)

    # Small image for stable color extraction.
    small = img.copy()
    small.thumbnail((240, 240))

    # Adaptive quantization gives us a simple dominant palette without extra dependencies.
    pal_img = small.convert("P", palette=Image.ADAPTIVE, colors=max_colors)
    palette_raw = pal_img.getpalette() or []
    counts = pal_img.getcolors(maxcolors=240 * 240) or []

    items: List[Tuple[int, Tuple[int, int, int]]] = []

    for count, idx in counts:
        base = int(idx) * 3
        if base + 2 >= len(palette_raw):
            continue
        rgb = (
            int(palette_raw[base]),
            int(palette_raw[base + 1]),
            int(palette_raw[base + 2]),
        )
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
            }
        )

    # Fallback average if quantization returns nothing.
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
            colors = _extract_palette_from_image(path, max_colors=6)
            for c in colors:
                c = dict(c)
                c["source_file"] = os.path.basename(path)
                all_colors.append(c)
        except Exception:
            continue

    # Simple de-duplication by hex, keeping highest percentage.
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

    return merged[:max_colors]


def _make_moodboard_grid(
    *,
    image_paths: List[str],
    palette: List[Dict[str, Any]],
    output_path: str,
    title: str = "RENDEREXPO Moodboard",
    max_tiles: int = 12,
) -> str:
    Image, ImageDraw, ImageFont, _ = _require_pil()

    if not image_paths and not palette:
        raise RuntimeError("Cannot create moodboard grid without images or palette.")

    canvas_w = 1600
    canvas_h = 1100
    margin = 48
    gap = 22
    title_h = 84
    palette_h = 150

    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 244, 241))
    draw = ImageDraw.Draw(canvas)

    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 42)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        title_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    draw.text((margin, 28), title, fill=(35, 35, 35), font=title_font)

    # Image tile area.
    tile_area_top = margin + title_h
    tile_area_bottom = canvas_h - margin - palette_h
    tile_area_h = tile_area_bottom - tile_area_top
    tile_area_w = canvas_w - margin * 2

    selected_images = image_paths[:max_tiles]

    if selected_images:
        n = len(selected_images)
        cols = min(4, max(2, int(math.ceil(math.sqrt(n)))))
        rows = int(math.ceil(n / cols))

        tile_w = int((tile_area_w - gap * (cols - 1)) / cols)
        tile_h = int((tile_area_h - gap * (rows - 1)) / rows)

        for idx, path in enumerate(selected_images):
            row = idx // cols
            col = idx % cols
            x = margin + col * (tile_w + gap)
            y = tile_area_top + row * (tile_h + gap)

            try:
                img = _open_rgb(path)
                img.thumbnail((tile_w, tile_h))
                bg = Image.new("RGB", (tile_w, tile_h), (232, 230, 225))
                ox = (tile_w - img.width) // 2
                oy = (tile_h - img.height) // 2
                bg.paste(img, (ox, oy))
                canvas.paste(bg, (x, y))
                draw.rectangle((x, y, x + tile_w, y + tile_h), outline=(210, 208, 202), width=2)
            except Exception:
                draw.rectangle((x, y, x + tile_w, y + tile_h), fill=(225, 220, 214), outline=(210, 208, 202), width=2)
                draw.text((x + 12, y + 12), os.path.basename(path), fill=(80, 80, 80), font=small_font)

    # Palette strip.
    pal_top = canvas_h - margin - palette_h + 26
    swatch_count = max(1, min(len(palette), 10))
    swatch_w = int((canvas_w - margin * 2 - gap * (swatch_count - 1)) / swatch_count)
    swatch_h = 82

    for idx, color in enumerate(palette[:swatch_count]):
        hx = str(color.get("hex") or "#CCCCCC").upper()
        x = margin + idx * (swatch_w + gap)
        y = pal_top

        try:
            rgb = tuple(int(hx.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        except Exception:
            rgb = (204, 204, 204)

        draw.rectangle((x, y, x + swatch_w, y + swatch_h), fill=rgb, outline=(160, 160, 160), width=1)

        text_fill = (20, 20, 20) if _relative_luminance(rgb) > 0.55 else (245, 245, 245)
        draw.text((x + 12, y + 26), hx, fill=text_fill, font=small_font)

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

def _analyze_images(image_paths: List[str]) -> Dict[str, Any]:
    if not image_paths:
        raise RuntimeError("Moodboard analysis requires at least one image.")

    palette = _merge_palettes(image_paths, max_colors=10)

    assets: List[Dict[str, Any]] = []
    for path in image_paths:
        try:
            assets.append(_extract_basic_image_features(path))
        except Exception as exc:
            assets.append(
                {
                    "file": os.path.basename(path),
                    "error": str(exc),
                }
            )

    summary = _build_style_summary(palette, assets)

    return {
        "palette": palette,
        "assets": assets,
        "summary": summary,
        "analyzed_files": [os.path.basename(p) for p in image_paths],
    }


def _write_analysis_outputs(job_folder: str, image_paths: List[str], title: str) -> Dict[str, str]:
    analysis = _analyze_images(image_paths)

    palette_path = os.path.join(job_folder, PALETTE_JSON_NAME)
    assets_path = os.path.join(job_folder, EXTRACTED_ASSETS_JSON_NAME)
    grid_path = os.path.join(job_folder, MOODBOARD_GRID_NAME)

    _write_json(
        palette_path,
        {
            "created_at": _utc_now(),
            "palette": analysis["palette"],
            "summary": analysis["summary"],
        },
    )

    _write_json(
        assets_path,
        {
            "created_at": _utc_now(),
            "assets": analysis["assets"],
            "summary": analysis["summary"],
            "analyzed_files": analysis["analyzed_files"],
        },
    )

    _make_moodboard_grid(
        image_paths=image_paths,
        palette=analysis["palette"],
        output_path=grid_path,
        title=title,
        max_tiles=12,
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
            _write_analysis_outputs(
                job_folder=job_folder,
                image_paths=fallback_images,
                title="RENDEREXPO Moodboard Reference",
            )
            palette_data = _read_json(palette_path)
            assets_data = _read_json(assets_path)
        except Exception:
            pass

    colors = []
    for c in (palette_data.get("palette") or [])[:6]:
        hx = c.get("hex")
        if hx:
            colors.append(str(hx))

    summary = {}
    if isinstance(palette_data.get("summary"), dict):
        summary.update(palette_data.get("summary") or {})
    if isinstance(assets_data.get("summary"), dict):
        summary.update(assets_data.get("summary") or {})

    style_summary = str(summary.get("style_summary") or "refined architectural moodboard")
    temp = str(summary.get("palette_temperature") or "neutral")
    brightness = str(summary.get("brightness") or "medium")

    color_text = ", ".join(colors) if colors else "cohesive interior design palette"

    return (
        f"{style_summary}, {brightness} {temp} palette, "
        f"dominant colors {color_text}, coordinated materials, refined design direction"
    )


def _build_moodboard_to_space_prompt(job_folder: str, meta: Dict[str, Any], moodboard_images: List[str]) -> str:
    user_prompt = str(meta.get("prompt") or "").strip()
    style_text = _style_text_from_analysis(job_folder, moodboard_images)

    base = (
        "photorealistic architectural interior visualization inspired by the uploaded moodboard, "
        "cohesive material palette, realistic furniture and finishes, refined lighting, "
        "high-end residential or hospitality design, realistic spatial depth, professional design presentation"
    )

    if user_prompt:
        return f"{base}, {style_text}, client direction: {user_prompt}"

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