"""
app/core/lora_registry.py

Simple registry for:
- RENDEREXPO LoRA profiles (interiors, exteriors, aerials, etc.)
- RENDEREXPO refiner profiles (ultra_detail, lighting_fix, etc.)

Reads from: config/lora_profiles.json

This does NOT load any models. It is only used to:
- validate profile names
- inspect what a profile contains
- add them into job meta / planning
"""

import json
import os
from functools import lru_cache
from typing import Any, Dict, Optional


CONFIG_PATH = os.path.join("config", "lora_profiles.json")


class LoraConfigError(RuntimeError):
    """Raised when there is an issue loading or reading the LoRA config."""
    pass


@lru_cache(maxsize=1)
def _load_config() -> Dict[str, Any]:
    """
    Load and cache the LoRA/refiner configuration from JSON.

    We use a tiny JSON file instead of YAML to avoid extra dependencies.
    """
    if not os.path.isfile(CONFIG_PATH):
        raise LoraConfigError(f"LoRA config file not found: {CONFIG_PATH}")

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise LoraConfigError(f"Failed to parse {CONFIG_PATH}: {e}") from e

    # Basic structure checks
    if "lora_profiles" not in data:
        data["lora_profiles"] = {}
    if "refiner_profiles" not in data:
        data["refiner_profiles"] = {}

    return data


def list_lora_profiles() -> Dict[str, Dict[str, Any]]:
    """Return all LoRA profiles as a dict: {profile_name: {...}}."""
    cfg = _load_config()
    return cfg.get("lora_profiles", {})


def list_refiner_profiles() -> Dict[str, Dict[str, Any]]:
    """Return all refiner profiles as a dict: {profile_name: {...}}."""
    cfg = _load_config()
    return cfg.get("refiner_profiles", {})


def get_lora_profile(name: str) -> Optional[Dict[str, Any]]:
    """Return a single LoRA profile by name, or None if not found."""
    name = (name or "").strip()
    if not name:
        return None

    profiles = list_lora_profiles()
    return profiles.get(name)


def get_refiner_profile(name: str) -> Optional[Dict[str, Any]]:
    """Return a single refiner profile by name, or None if not found."""
    name = (name or "").strip()
    if not name:
        return None

    profiles = list_refiner_profiles()
    return profiles.get(name)
