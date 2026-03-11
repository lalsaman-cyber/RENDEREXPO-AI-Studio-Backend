import os, sys, json, time, uuid, hmac, hashlib, requests

BASE = os.getenv("PLANNER_BASE", "").rstrip("/")
SECRET = os.environ["RENDEREXPO_HMAC_SECRET"]

def sign(ts: str, nonce: str, body: bytes) -> str:
    msg = (f"{ts}\n{nonce}\n").encode("utf-8") + (body or b"")
    return hmac.new(SECRET.encode("utf-8"), msg, hashlib.sha256).hexdigest()

if len(sys.argv) < 3:
    print("Usage: python signed_post_pc.py /api/path '{\"json\":true}'")
    sys.exit(2)

path = sys.argv[1]
payload = json.loads(sys.argv[2])

body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
ts = str(int(time.time()))
nonce = uuid.uuid4().hex
sig = sign(ts, nonce, body)

headers = {
    "Content-Type":"application/json",
    "X-RENDEREXPO-TIMESTAMP": ts,
    "X-RENDEREXPO-NONCE": nonce,
    "X-RENDEREXPO-SIGNATURE": sig,
}

r = requests.post(BASE + path, data=body, headers=headers, timeout=900)
print("STATUS", r.status_code)
print(r.text)
