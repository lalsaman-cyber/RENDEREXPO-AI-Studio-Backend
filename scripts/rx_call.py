import sys
import os
import time
import uuid
import hmac
import hashlib
import requests

"""
RENDEREXPO HMAC signing (LOCKED):

Headers:
  X-RENDEREXPO-SIGNATURE
  X-RENDEREXPO-TIMESTAMP
  X-RENDEREXPO-NONCE

Message bytes:
  f"{timestamp}\\n{nonce}\\n".encode("utf-8") + raw_body_bytes

This matches the signing rule you locked in your docs.
"""

def sign(secret: str, timestamp: str, nonce: str, body_bytes: bytes) -> str:
    msg = (f"{timestamp}\n{nonce}\n").encode("utf-8") + (body_bytes or b"")
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()

def send(method: str, url: str, body_bytes: bytes) -> int:
    secret = os.environ.get("RENDEREXPO_HMAC_SECRET", "")
    if not secret:
        print("ERROR: RENDEREXPO_HMAC_SECRET is not set in this shell.", file=sys.stderr)
        return 2

    ts = str(int(time.time()))
    nonce = uuid.uuid4().hex

    headers = {
        "X-RENDEREXPO-TIMESTAMP": ts,
        "X-RENDEREXPO-NONCE": nonce,
        "X-RENDEREXPO-SIGNATURE": sign(secret, ts, nonce, body_bytes),
    }

    if body_bytes:
        headers["Content-Type"] = "application/json"

    r = requests.request(method.upper(), url, data=body_bytes if body_bytes else None, headers=headers, timeout=120)
    print("STATUS", r.status_code)
    print(r.text[:4000])
    return 0

def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python scripts/rx_call.py <METHOD> <URL> [JSON_BODY_STRING]", file=sys.stderr)
        return 2

    method = sys.argv[1].upper()
    url = sys.argv[2]
    body_str = sys.argv[3] if len(sys.argv) >= 4 else ""
    body_bytes = body_str.encode("utf-8") if body_str else b""
    return send(method, url, body_bytes)

if __name__ == "__main__":
    raise SystemExit(main())
