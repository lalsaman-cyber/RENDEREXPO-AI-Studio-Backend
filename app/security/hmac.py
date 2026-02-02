# app/security/hmac.py
"""
RENDEREXPO AI STUDIO - HMAC Security (Option A, LOCKED)

This module implements STRICT HMAC authentication used by:
- app/main.py middleware
- any internal service-to-service calls (planner -> GPU)

POLICY (DO NOT WEAKEN):
- Sign RAW request body bytes
- Prevent replay attacks (nonce + TTL)
- Enforce tight timestamp window
- Constant-time signature comparison
- ZERO external dependencies

HEADERS (REQUIRED):
- X-RENDEREXPO-SIGNATURE : hex HMAC-SHA256
- X-RENDEREXPO-TIMESTAMP : unix epoch seconds (int)
- X-RENDEREXPO-NONCE     : random unique string

MESSAGE TO SIGN (BYTES):
    f"{timestamp}\\n{nonce}\\n".encode("utf-8") + raw_body_bytes

ENV:
- RENDEREXPO_HMAC_SECRET (REQUIRED, 64+ chars recommended)
"""

from __future__ import annotations

import os
import time
import hmac
import hashlib
from typing import Dict


# ---------------------------------------------------------------------------
# Constants (LOCKED)
# ---------------------------------------------------------------------------

SIG_HEADER = "X-RENDEREXPO-SIGNATURE"
TS_HEADER = "X-RENDEREXPO-TIMESTAMP"
NONCE_HEADER = "X-RENDEREXPO-NONCE"

TS_WINDOW_SECONDS = 30
NONCE_TTL_SECONDS = 90

HMAC_SECRET_ENV = "RENDEREXPO_HMAC_SECRET"


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def now_epoch() -> int:
    return int(time.time())


# ---------------------------------------------------------------------------
# Signature helpers
# ---------------------------------------------------------------------------

def compute_signature(secret: str, timestamp: str, nonce: str, body: bytes) -> str:
    """
    Compute hex HMAC-SHA256 signature over:
        f"{timestamp}\\n{nonce}\\n".encode("utf-8") + body
    """
    prefix = f"{timestamp}\n{nonce}\n".encode("utf-8")
    msg = prefix + (body or b"")
    return hmac.new(
        secret.encode("utf-8"),
        msg,
        hashlib.sha256,
    ).hexdigest()


def constant_time_equal(a: str, b: str) -> bool:
    """
    Constant-time comparison to prevent timing attacks.
    """
    return hmac.compare_digest(a or "", b or "")


# ---------------------------------------------------------------------------
# Nonce replay protection
# ---------------------------------------------------------------------------

# nonce -> expires_at_epoch
_NONCE_CACHE: Dict[str, int] = {}


def purge_expired_nonces() -> None:
    now = now_epoch()
    expired = [n for n, exp in _NONCE_CACHE.items() if exp <= now]
    for n in expired:
        _NONCE_CACHE.pop(n, None)


def register_nonce(nonce: str) -> None:
    """
    Register a nonce after successful verification.
    """
    _NONCE_CACHE[nonce] = now_epoch() + NONCE_TTL_SECONDS


def nonce_seen(nonce: str) -> bool:
    return nonce in _NONCE_CACHE


# ---------------------------------------------------------------------------
# Secret access
# ---------------------------------------------------------------------------

def get_hmac_secret() -> str:
    secret = os.getenv(HMAC_SECRET_ENV, "").strip()
    if not secret or len(secret) < 32:
        raise RuntimeError(
            f"Missing or weak {HMAC_SECRET_ENV}. "
            "Set a strong secret (64+ chars) before running."
        )
    return secret
