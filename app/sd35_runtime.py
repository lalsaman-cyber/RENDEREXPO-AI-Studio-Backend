"""
SD3.5 runtime skeleton.

This is the SD3.5 equivalent of the old SDXL "engine".
Right now it is JUST a placeholder interface – no heavy model loading yet.

Later, on RunPod GPU, we will:
- import the correct SD3.5 pipeline class from the official docs
- load weights from models/sd35-large
- implement the real generate() function.
"""

from pathlib import Path
from typing import Optional


BASE_DIR = Path(__file__).resolve().parent.parent
SD35_DIR = BASE_DIR / "models" / "sd35-large"


class SD35Runtime:
    def __init__(
        self,
        model_dir: Optional[Path] = None,
        device: str = "cuda",
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir else SD35_DIR
        self.device = device

        # We DO NOT load the model here yet.
        # Heavy loading will be implemented on the GPU pod only.
        self._loaded = False

    def load(self) -> None:
        """
        Placeholder: in the RunPod GPU environment, this will:

        - import the official SD3.5 Large pipeline class
        - load weights from self.model_dir
        - move to self.device ("cuda")
        - set self._loaded = True
        """
        # TODO: Fill in once we implement GPU runtime.
        raise NotImplementedError("SD3.5 runtime load() not implemented yet.")

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Placeholder: run a single SD3.5 text-to-image generation.

        This will only be implemented for GPU (RunPod) use.
        Local calls should not invoke this.
        """
        raise NotImplementedError("SD3.5 runtime generate() not implemented yet.")
