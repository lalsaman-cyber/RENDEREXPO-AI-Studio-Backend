from __future__ import annotations

from typing import Dict, Optional

from app.config.sketch_runtime import get_anyline_mistoline_config
from app.services.comfy_anyline_mistoline import AnylineMistolineSketchService


def run_anyline_mistoline_sketch(
    input_image_path: str,
    output_dir: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict:
    config = get_anyline_mistoline_config()
    service = AnylineMistolineSketchService(config=config)

    return service.run(
        input_image_path=input_image_path,
        output_dir=output_dir,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
    )