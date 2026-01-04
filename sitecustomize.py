"""
Compatibility shim for Basicsr/Real-ESRGAN on modern torchvision.

Basicsr expects: torchvision.transforms.functional_tensor.rgb_to_grayscale
But that module path no longer exists in torchvision 0.23+.
Python auto-imports sitecustomize.py if present on sys.path, so we inject
a small module into sys.modules to satisfy the legacy import.
"""

import sys
import types

try:
    import torchvision.transforms.functional as F
except Exception:
    F = None

mod_name = "torchvision.transforms.functional_tensor"

if mod_name not in sys.modules:
    m = types.ModuleType(mod_name)

    def rgb_to_grayscale(img, num_output_channels=1):
        # Prefer torchvision's current implementation if available
        if F is not None and hasattr(F, "rgb_to_grayscale"):
            return F.rgb_to_grayscale(img, num_output_channels=num_output_channels)
        raise ImportError("torchvision.transforms.functional.rgb_to_grayscale not available")

    m.rgb_to_grayscale = rgb_to_grayscale
    sys.modules[mod_name] = m
