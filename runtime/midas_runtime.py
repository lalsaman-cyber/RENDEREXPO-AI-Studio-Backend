# runtime/midas_runtime.py
"""
runtime/midas_runtime.py

Skeleton runtime wrapper around a MiDaS-style depth model.

IMPORTANT:
- On your laptop (local dev), this MUST NOT load any weights
  or run any heavy inference.
- This is just a placeholder that will be used LATER in the GPU runtime
  (RunPod) to actually estimate depth maps.

Depth maps will power:
- VR world reconstruction
- Parallax video-from-image
- Product insertion (geometry awareness)
- Mesh / CAD helpers
"""

from dataclasses import dataclass


class MidasNotLoadedError(RuntimeError):
    """Raised when someone tries to use MiDaS in skeleton mode."""


@dataclass
class MidasRuntime:
    """
    Very thin wrapper around a future MiDaS model.

    Fields:
        device: "cpu" | "cuda" | etc.
        model_name: identifier string for documentation.
        loaded: whether the real model weights are loaded (GPU runtime).
    """

    device: str = "cpu"
    model_name: str = "midas-large"
    loaded: bool = False

    def load(self) -> None:
        """
        Placeholder for future depth-model loading on GPU.

        On local dev, we *do not* load anything.
        In the real GPU runtime, we will:
        - load MiDaS weights using torch / timm
        - move the model to self.device ("cuda")
        """
        # Intentionally do NOT load anything here for local dev.
        self.loaded = False

    def estimate_depth(self, input_path: str, output_path: str) -> None:
        """
        Placeholder depth estimation.

        On GPU, this will:
        - read the input image from input_path
        - run MiDaS depth estimation
        - write a depth map image to output_path

        On local CPU app, this MUST NOT be used; we raise.
        """
        raise MidasNotLoadedError(
            "MidasRuntime.estimate_depth() called in skeleton mode. "
            "Real implementation will run only in the GPU runtime."
        )
