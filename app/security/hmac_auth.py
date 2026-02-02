# app/security/hmac_auth.py
"""
RENDEREXPO AI STUDIO - HMAC Auth (Option A - Recommended)

Goal:
- One shared HMAC scheme used across services (Planner <-> GPU Worker)
- Works for BOTH:
  (1) Server-side verification (FastAPI middleware / dependency)
  (2) Client-side signing (Planner calling GPU dispatch)

LOCKED POLICY:
- Sign RAW request body bytes
- Headers:
    X-RENDEREXPO-SIGNATURE: hex digest (HMAC-SHA256)
    X-RENDEREXPO-TIMESTAMP: unix epoch seconds (int)
    X-RENDEREXPO-NONCE: random string (unique per request)
- Message to sign (bytes):
    f"{timestamp}\\n{nonce}\\n".encode("utf-8") + raw_body_bytes
- Timestamp window: ±30 seconds
- Nonce replay protection: in-memory cache (short TTL)

Environment:
- RENDEREXPO_HMAC_SECRET (required)

Notes:
- Nonce cache is per-process (good enough for single worker / single instance).
  If you scale horizontally, use Redis or a shared store for replay protection.
"""

from __future__ import annotations

import os
import time
import hmac
import hashlib
import secrets
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

HMAC_SECRET_ENV = "RENDEREXPO_HMAC_SECRET"

SIG_HEADER = "X-RENDEREXPO-SIGNATURE"
TS_HEADER = "X-RENDEREXPO-TIMESTAMP"
NONCE_HEADER = "X-RENDEREXPO-NONCE"

TS_WINDOW_SECONDS = 30
NONCE_TTL_SECONDS = 90

# -----------------------------------------------------------------------------
# Nonce cache (in-memory)
# -----------------------------------------------------------------------------

@dataclass
class NonceCache:
    """
    In-memory nonce cache for replay protection.
    Maps nonce -> expires_at_epoch
    """
    store: Dict[str, int]

    def purge(self, now: Optional[int] = None) -> None:
        n = int(now if now is not None else time.time())
        expired = [k for k, exp in self.store.items() if exp <= n]
        for k in expired:
            self.store.pop(k, None)

    def seen(self, nonce: str) -> bool:
        return nonce in self.store

    def mark(self, nonce: str, now: Optional[int] = None, ttl: int = NONCE_TTL_SECONDS) -> None:
        n = int(now if now is not None else time.time())
        self.store[nonce] = n + int(ttl)


# Single-process nonce cache (imported module global)
NONCE_CACHE = NonceCache(store={})

# -----------------------------------------------------------------------------
# Core crypto
# -----------------------------------------------------------------------------

def _now_epoch() -> int:
    return int(time.time())


def get_secret() -> str:
    """
    Load and validate secret from env.
    """
    secret = (os.getenv(HMAC_SECRET_ENV) or "").strip()
    if not secret or len(secret) < 32:
        raise RuntimeError(
            f"Missing or too-short {HMAC_SECRET_ENV}. "
            f"Set a strong secret (64+ chars) before running the API."
        )
    return secret


def compute_signature(secret: str, timestamp: str, nonce: str, body: bytes) -> str:
    """
    Compute hex HMAC-SHA256 over:
        f"{timestamp}\\n{nonce}\\n".encode("utf-8") + body
    """
    prefix = f"{timestamp}\n{nonce}\n".encode("utf-8")
    msg = prefix + (body or b"")
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def constant_time_equal(a: str, b: str) -> bool:
    return hmac.compare_digest((a or "").lower(), (b or "").lower())


# -----------------------------------------------------------------------------
# Client-side signing (Planner -> GPU)
# -----------------------------------------------------------------------------

def make_nonce(nbytes: int = 16) -> str:
    """
    Random nonce for each request.
    """
    return secrets.token_hex(max(8, int(nbytes)))


def sign_request_headers(body: bytes, secret: Optional[str] = None) -> Dict[str, str]:
    """
    Create headers for an outgoing request that must pass HMAC verification.
    """
    s = secret or get_secret()
    ts = str(_now_epoch())
    nonce = make_nonce()
    sig = compute_signature(secret=s, timestamp=ts, nonce=nonce, body=body or b"")
    return {
        SIG_HEADER: sig,
        TS_HEADER: ts,
        NONCE_HEADER: nonce,
    }


# -----------------------------------------------------------------------------
# Server-side verification (FastAPI middleware helper)
# -----------------------------------------------------------------------------

def verify_hmac_headers(
    provided_sig: str,
    ts: str,
    nonce: str,
    body: bytes,
    secret: Optional[str] = None,
    now_epoch: Optional[int] = None,
    ts_window_seconds: int = TS_WINDOW_SECONDS,
    nonce_ttl_seconds: int = NONCE_TTL_SECONDS,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Verify HMAC + timestamp + nonce replay protection.

    Returns: (ok, error_code, detail)
    Error codes (minimal/structured):
      - missing_auth
      - timestamp_invalid
      - timestamp_expired
      - nonce_reused
      - invalid_signature
    """
    if not provided_sig or not ts or not nonce:
        return False, "missing_auth", f"Required headers: {SIG_HEADER}, {TS_HEADER}, {NONCE_HEADER}"

    try:
        ts_int = int(ts)
    except Exception:
        return False, "timestamp_invalid", "Timestamp must be unix epoch seconds (int)."

    now = int(now_epoch if now_epoch is not None else _now_epoch())
    if abs(now - ts_int) > int(ts_window_seconds):
        return False, "timestamp_expired", f"Timestamp outside ±{int(ts_window_seconds)}s window."

    # Nonce replay
    NONCE_CACHE.purge(now=now)
    if NONCE_CACHE.seen(nonce):
        return False, "nonce_reused", "Nonce has already been used within the allowed window."

    s = secret or get_secret()
    expected = compute_signature(secret=s, timestamp=ts, nonce=nonce, body=body or b"")
    if not constant_time_equal(provided_sig, expected):
        return False, "invalid_signature", "Signature verification failed."

    # Mark nonce used
    NONCE_CACHE.mark(nonce, now=now, ttl=int(nonce_ttl_seconds))
    return True, None, None


async def verify_request_or_401(
    request: Request,
    open_paths: Optional[set[str]] = None,
) -> Optional[JSONResponse]:
    """
    Middleware-style helper:
    - returns JSONResponse (401) if failed
    - returns None if ok (caller continues request)

    open_paths: paths that bypass auth (e.g., {"/api/health"})
    """
    if open_paths and request.url.path in open_paths:
        return None

    provided_sig = request.headers.get(SIG_HEADER, "")
    ts = request.headers.get(TS_HEADER, "")
    nonce = request.headers.get(NONCE_HEADER, "")

    body = await request.body()

    ok, err, detail = verify_hmac_headers(
        provided_sig=provided_sig,
        ts=ts,
        nonce=nonce,
        body=body,
    )

    if ok:
        return None

    return JSONResponse(
        status_code=401,
        content={"error": err, "detail": detail},
    )
