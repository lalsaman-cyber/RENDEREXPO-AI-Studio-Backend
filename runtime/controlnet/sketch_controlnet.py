from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

from runtime.preprocess.sketch_canny import build_canny_image
from runtime.preprocess.sketch_depth import build_depth_image


@dataclass
class SketchControlNetConfig:
    enabled: bool = True
    save_preprocessed_images: bool = True
    default_canny_scale: float = 1.0
    default_depth_scale: float = 0.85
    default_output_name: str = "output.png"
    default_canny_name: str = "canny.png"
    default_depth_name: str = "depth.png"
    default_sketch_name: str = "sketch.png"


def _require_pil() -> None:
    if Image is None:
        raise RuntimeError("Pillow is required for sketch ControlNet execution.")


def _filter_supported_call_kwargs(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Only pass kwargs supported by the installed pipeline signature.
    """
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return dict(kwargs)

        allowed = set(params.keys())
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return filtered
    except Exception:
        return dict(kwargs)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _control_list(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    controlnet_cfg = meta.get("controlnet")
    if not isinstance(controlnet_cfg, dict):
        return []
    controls = controlnet_cfg.get("controls")
    if not isinstance(controls, list):
        return []
    return [c for c in controls if isinstance(c, dict)]


def _find_control(meta: Dict[str, Any], control_type: str) -> Optional[Dict[str, Any]]:
    needle = str(control_type).strip().lower()
    for c in _control_list(meta):
        if str(c.get("control_type", "")).strip().lower() == needle:
            return c
    return None


def _resolve_control_scale(meta: Dict[str, Any], control_type: str, default: float) -> float:
    control = _find_control(meta, control_type)
    if not isinstance(control, dict):
        return float(default)
    return _safe_float(control.get("conditioning_scale", default), default)


def _resolve_output_name(meta: Dict[str, Any], key: str, fallback: str) -> str:
    outputs = meta.get("outputs")
    if isinstance(outputs, dict):
        raw = outputs.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return fallback


def _save_image(image: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)


def _resolve_negative_prompt(meta: Dict[str, Any]) -> Optional[str]:
    raw = meta.get("negative_prompt")
    if raw is None:
        return None
    text = str(raw).strip()
    return text if text else None


def _resolve_generator(torch_module: Any, device: str, meta: Dict[str, Any]) -> Any:
    seed = meta.get("seed")
    if seed is None:
        return None
    try:
        return torch_module.Generator(device=device).manual_seed(int(seed))
    except Exception:
        return None


def _extract_cleanup_config(control: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(control, dict):
        return None
    prep = control.get("preprocessor")
    if not isinstance(prep, dict):
        return None

    cleanup = prep.get("cleanup")
    if isinstance(cleanup, dict):
        return cleanup
    return None


def _extract_canny_config(control: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(control, dict):
        return out

    prep = control.get("preprocessor")
    if not isinstance(prep, dict):
        return out

    if "low_threshold" in prep:
        out["low_threshold"] = prep.get("low_threshold")
    if "high_threshold" in prep:
        out["high_threshold"] = prep.get("high_threshold")
    if "invert_if_dark_background" in prep:
        out["invert_output"] = prep.get("invert_if_dark_background")
    if "line_boost" in prep:
        out["line_boost"] = prep.get("line_boost")
    if "line_boost_kernel" in prep:
        out["line_boost_kernel"] = prep.get("line_boost_kernel")

    cleanup_cfg = _extract_cleanup_config(control)
    if isinstance(cleanup_cfg, dict):
        out["cleanup_enabled"] = bool(cleanup_cfg.get("enabled", True))

    return out


def _extract_depth_config(control: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(control, dict):
        return out

    prep = control.get("preprocessor")
    if not isinstance(prep, dict):
        return out

    if "normalize_to_png" in prep:
        out["normalize_output"] = bool(prep.get("normalize_to_png", True))
    if "invert_output" in prep:
        out["invert_output"] = bool(prep.get("invert_output", False))
    if "blur_radius" in prep:
        out["blur_radius"] = prep.get("blur_radius")

    cleanup_cfg = _extract_cleanup_config(control)
    if isinstance(cleanup_cfg, dict):
        out["cleanup_enabled"] = bool(cleanup_cfg.get("enabled", True))

    return out


def _optional_polish_enabled(meta: Dict[str, Any]) -> bool:
    op = meta.get("optional_polish")
    return isinstance(op, dict) and bool(op.get("enabled", False))


def _resolve_final_delivery_size(
    working_size: Tuple[int, int],
    original_size: Tuple[int, int],
    explicit_dimensions: bool,
) -> Tuple[int, int]:
    if explicit_dimensions:
        return int(working_size[0]), int(working_size[1])
    return int(original_size[0]), int(original_size[1])


def _emit_stage(stage_callback: Any, stage: str, **extra: Any) -> None:
    if not callable(stage_callback):
        return
    payload: Dict[str, Any] = {"sketch_stage": stage}
    payload.update(extra)
    try:
        stage_callback(stage, payload)
    except Exception:
        pass



def run_sketch_controlnet_generation(
    *,
    pipe: Any,
    torch_module: Any,
    device: str,
    job_folder: str,
    meta: Dict[str, Any],
    sketch_image: Any,
    depth_image_processor: Any,
    depth_model: Any,
    target_size: Tuple[int, int],
    explicit_dimensions: bool,
    detail_pass_fn: Any,
    upscale_if_enabled_fn: Any,
    resize_exact_fn: Any,
    enforce_output_size_fn: Any,
    stage_callback: Any = None,
) -> Dict[str, Any]:
    """
    Execute the sketch -> ControlNet -> output pipeline.

    Production rule for now:
    - Canny is the primary structural control.
    - Depth is optional and disabled unless explicitly enabled in meta.
    """
    _require_pil()

    cfg = SketchControlNetConfig()

    if not cfg.enabled:
        raise RuntimeError("SketchControlNetConfig.enabled is false.")

    if not isinstance(sketch_image, Image.Image):
        raise RuntimeError("run_sketch_controlnet_generation expects a PIL sketch_image.")

    if pipe is None:
        raise RuntimeError("Sketch ControlNet pipeline is missing.")
    if torch_module is None:
        raise RuntimeError("Torch module is required.")

    os.makedirs(job_folder, exist_ok=True)

    original_size = (int(sketch_image.size[0]), int(sketch_image.size[1]))
    target_w, target_h = int(target_size[0]), int(target_size[1])

    canny_control = _find_control(meta, "canny")
    depth_control = _find_control(meta, "depth")

    canny_cfg = _extract_canny_config(canny_control)
    depth_cfg = _extract_depth_config(depth_control)

    use_depth_control = bool(meta.get("use_depth_control", False))
    meta["use_depth_control"] = use_depth_control

    _emit_stage(stage_callback, "building_canny", target_width=target_w, target_height=target_h)
    canny_image = build_canny_image(
        sketch_image,
        config=canny_cfg,
        target_size=(target_w, target_h),
    )

    depth_image = None
    if use_depth_control:
        if depth_image_processor is None or depth_model is None:
            raise RuntimeError("Depth control was enabled but depth preprocessor objects are missing.")
        _emit_stage(stage_callback, "building_depth")
        depth_image = build_depth_image(
            sketch_image,
            image_processor=depth_image_processor,
            depth_model=depth_model,
            device=device,
            torch_module=torch_module,
            config=depth_cfg,
            target_size=(target_w, target_h),
        )

    sketch_name = _resolve_output_name(meta, "sketch_image", cfg.default_sketch_name)
    canny_name = _resolve_output_name(meta, "canny_image", cfg.default_canny_name)
    depth_name = _resolve_output_name(meta, "depth_image", cfg.default_depth_name)
    output_name = _resolve_output_name(meta, "final_image", cfg.default_output_name)

    prepared_sketch = sketch_image.convert("RGB")
    if prepared_sketch.size != (target_w, target_h):
        prepared_sketch = prepared_sketch.resize((target_w, target_h), Image.LANCZOS)

    if cfg.save_preprocessed_images:
        _save_image(prepared_sketch, os.path.join(job_folder, sketch_name))
        _save_image(canny_image, os.path.join(job_folder, canny_name))
        if depth_image is not None:
            _save_image(depth_image, os.path.join(job_folder, depth_name))

    prompt = str(meta.get("prompt") or "").strip()
    if not prompt:
        raise RuntimeError("Sketch ControlNet requires a non-empty prompt.")

    negative_prompt = _resolve_negative_prompt(meta)
    num_steps = int(meta.get("num_inference_steps", 46))
    guidance_scale = float(meta.get("guidance_scale", 5.6))

    canny_scale = _resolve_control_scale(meta, "canny", cfg.default_canny_scale)
    depth_scale = _resolve_control_scale(meta, "depth", cfg.default_depth_scale)

    generator = _resolve_generator(torch_module, device, meta)

    control_images = [canny_image]
    control_scales = [canny_scale]
    if use_depth_control and depth_image is not None:
        control_images.append(depth_image)
        control_scales.append(depth_scale)

    candidate_kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "control_image": control_images,
        "controlnet_conditioning_scale": control_scales,
        "num_inference_steps": num_steps,
        "guidance_scale": guidance_scale,
        "width": target_w,
        "height": target_h,
    }
    if generator is not None:
        candidate_kwargs["generator"] = generator

    kwargs = _filter_supported_call_kwargs(pipe.__call__, candidate_kwargs)

    _emit_stage(stage_callback, "running_inference", control_count=len(control_images))
    images = pipe(**kwargs).images
    if not images:
        raise RuntimeError("Sketch ControlNet pipeline returned no images.")

    image = images[0]
    raw_output_size = getattr(image, "size", None)
    if not raw_output_size or len(raw_output_size) != 2:
        raise RuntimeError("Sketch ControlNet pipeline returned an image without a readable size.")

    meta["pipeline_output_size_before_enforce"] = {
        "width": int(raw_output_size[0]),
        "height": int(raw_output_size[1]),
    }

    if tuple(raw_output_size) != (target_w, target_h):
        raw_path = os.path.join(job_folder, "output_pipeline_raw.png")
        try:
            image.save(raw_path)
            meta["output_pipeline_raw_image"] = "output_pipeline_raw.png"
        except Exception:
            pass

    image, enforced, method, before_sz, after_sz = enforce_output_size_fn(
        image,
        (target_w, target_h),
    )

    meta["output_size_enforced"] = bool(enforced)
    meta["output_size_enforcement_method"] = method
    meta["pipeline_output_size_after_enforce"] = {
        "width": int(after_sz[0]),
        "height": int(after_sz[1]),
    }

    if getattr(image, "size", None) != (target_w, target_h):
        raise RuntimeError(
            f"Sketch ControlNet output size enforcement failed. Expected {(target_w, target_h)}, "
            f"got {getattr(image, 'size', None)}"
        )

    _emit_stage(stage_callback, "detail_pass")
    image = detail_pass_fn(image, meta)

    final_delivery_size = _resolve_final_delivery_size(
        (target_w, target_h),
        original_size,
        explicit_dimensions,
    )

    image = resize_exact_fn(image, final_delivery_size)

    if getattr(image, "size", None) != final_delivery_size:
        raise RuntimeError(
            f"Final sketch delivery size failed. Expected {final_delivery_size}, got {getattr(image, 'size', None)}"
        )

    base_out_name = "output_base.png"
    base_out_path = os.path.join(job_folder, base_out_name)
    try:
        image.save(base_out_path)
    except Exception:
        pass

    output_path = os.path.join(job_folder, output_name)
    image.save(output_path)

    final_up2x_name: Optional[str] = None
    if _optional_polish_enabled(meta):
        _emit_stage(stage_callback, "optional_polish")
        upscaled = upscale_if_enabled_fn(image, meta)
        if getattr(upscaled, "size", None) != getattr(image, "size", None):
            final_up2x_name = "final_up2x.png"
            upscaled.save(os.path.join(job_folder, final_up2x_name))

    meta["status"] = "completed"
    meta["mode"] = "real-sd35-sketch-controlnet"
    meta["output_image"] = output_name
    meta["output_base_image"] = base_out_name
    meta["canny_image"] = canny_name
    meta["sketch_image"] = sketch_name
    meta["sketch_input_size"] = {"width": original_size[0], "height": original_size[1]}
    meta["sketch_working_size"] = {"width": target_w, "height": target_h}
    meta["sketch_explicit_dimensions"] = bool(explicit_dimensions)
    meta["final_delivery_size"] = {
        "width": int(final_delivery_size[0]),
        "height": int(final_delivery_size[1]),
    }
    meta["final_delivery_resize_applied"] = final_delivery_size != (target_w, target_h)

    if use_depth_control and depth_image is not None:
        meta["depth_image"] = depth_name
    else:
        meta.pop("depth_image", None)

    if final_up2x_name:
        meta["final_up2x_image"] = final_up2x_name

    _emit_stage(stage_callback, "completed", output_image=output_name)
    return meta
