# app/security/__init__.py
"""
RENDEREXPO AI STUDIO - Security Package

Central export for all security-related logic.

Currently includes:
- HMAC authentication helpers (Option A, LOCKED)

This package is intentionally small, strict, and dependency-free.
All API security MUST route through this layer.
"""

from .hmac import (
    SIG_HEADER,
    TS_HEADER,
    NONCE_HEADER,
    TS_WINDOW_SECONDS,
    NONCE_TTL_SECONDS,
    now_epoch,
    compute_signature,
    constant_time_equal,
    purge_expired_nonces,
    register_nonce,
)

__all__ = [
    "SIG_HEADER",
    "TS_HEADER",
    "NONCE_HEADER",
    "TS_WINDOW_SECONDS",
    "NONCE_TTL_SECONDS",
    "now_epoch",
    "compute_signature",
    "constant_time_equal",
    "purge_expired_nonces",
    "register_nonce",
]
