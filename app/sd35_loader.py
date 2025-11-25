"""
app/sd35_loader.py

Very small, SAFE helper for SD3.5 model path and file checks.

IMPORTANT:
- This file does NOT load SD3.5 into RAM.
- It does NOT run any AI inference.
- It ONLY:
    * reads config/model_paths.yaml
    * finds sd35_large_dir
    * lists the files in that folder
"""

import os
from typing import List, Dict, Any


class SD35ConfigError(Exception):
    """Raised when there is a problem with SD3.5 config or model folder."""
    pass


def _read_sd35_model_dir_from_config() -> str:
    """
    Very simple parser for config/model_paths.yaml to find sd35_large_dir.

    Expected line in config/model_paths.yaml:

        sd35_large_dir: "models/sd35-large"

    We:
    - look for a line starting with 'sd35_large_dir'
    - split on ':'
    - strip quotes and spaces
    """
    config_path = os.path.join("config", "model_paths.yaml")

    if not os.path.isfile(config_path):
        raise SD35ConfigError(f"Config file not found: {config_path}")

    sd35_dir: str | None = None

    with open(config_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            if line.startswith("sd35_large_dir"):
                # Example: sd35_large_dir: "models/sd35-large"
                parts = line.split(":", 1)
                if len(parts) != 2:
                    continue
                value = parts[1].strip()

                # Remove optional quotes
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                if value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                sd35_dir = value
                break

    if not sd35_dir:
        raise SD35ConfigError(
            "Could not find 'sd35_large_dir' in config/model_paths.yaml."
        )

    return sd35_dir


def _list_directory_contents(path: str, max_items: int = 200) -> Dict[str, Any]:
    """
    Return a simple listing of the given directory.

    - If the directory does not exist, raise an error.
    - Only goes ONE level deep (top-level files & folders).
    """
    if not os.path.isdir(path):
        raise SD35ConfigError(f"Directory not found: {path}")

    items: List[Dict[str, Any]] = []

    # List only the immediate contents (no deep recursion)
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        item_type = "dir" if os.path.isdir(full_path) else "file"
        items.append({"name": name, "type": item_type})

        if len(items) >= max_items:
            break

    return {
        "path": path,
        "items": items,
        "count": len(items),
        "truncated": len(items) >= max_items,
    }


def verify_sd35_files() -> Dict[str, Any]:
    """
    Top-level function used by FastAPI.

    It:
    - reads config/model_paths.yaml
    - finds sd35_large_dir
    - lists its contents

    It does NOT:
    - load any SD3.5 model into memory
    - run any GPU / CPU-heavy inference
    """
    sd35_dir = _read_sd35_model_dir_from_config()
    listing = _list_directory_contents(sd35_dir)

    return {
        "sd35_large_dir": sd35_dir,
        "contents": listing,
    }
