import os, sys, json, time, uuid, hmac, hashlib, requests

BASE = os.environ["PLANNER_BASE"].rstrip("/")
SECRET = os.environ["RENDEREXPO_HMAC_SECRET"]

def sign(ts: str, nonce: str, body: bytes) -> str:
    msg = (f"{ts}\n{nonce}\n").encode("utf-8") + (body or b"")
    return hmac.new(SECRET.encode("utf-8"), msg, hashlib.sha256).hexdigest()

if len(sys.argv) < 3:
    print("Usage: python signed_form_pc.py <image_path> <category> [shot] [seed]")
    sys.exit(2)

image_path = sys.argv[1]
category = sys.argv[2]
shot = sys.argv[3] if len(sys.argv) > 3 else "wide"
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 601

ts = str(int(time.time()))
nonce = uuid.uuid4().hex
sig = sign(ts, nonce, b"")

headers = {
    "X-RENDEREXPO-TIMESTAMP": ts,
    "X-RENDEREXPO-NONCE": nonce,
    "X-RENDEREXPO-SIGNATURE": sig,
}

prompt = (
    "Apply realistic exterior materials to this design: stone veneer base, warm stucco walls, "
    "wood accents, charcoal standing-seam metal roof, black aluminum window frames with realistic reflections, "
    "natural daylight, clean shadows, photoreal architectural visualization, no CGI look"
)

negative = "cartoon, illustration, anime, watercolor, lowres, blurry, warped geometry"

files = {
    "file": open(image_path, "rb"),
}

data = {
    "prompt": prompt,
    "negative_prompt": negative,
    "category": category,
    "shot": shot,
    "seed": str(seed),
    "upscale_2x": "false",
}

r = requests.post(BASE + "/api/sd35/render-form", files=files, data=data, headers=headers, timeout=900)
print("STATUS", r.status_code)
print(r.text)
