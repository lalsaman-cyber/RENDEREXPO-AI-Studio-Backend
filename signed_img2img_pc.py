import os, json, time, uuid, hmac, hashlib, base64, sys, requests

BASE = os.environ["PLANNER_BASE"].rstrip("/")
SECRET = os.environ["RENDEREXPO_HMAC_SECRET"]

def sign(ts: str, nonce: str, body: bytes) -> str:
    msg = (f"{ts}\n{nonce}\n").encode("utf-8") + (body or b"")
    return hmac.new(SECRET.encode("utf-8"), msg, hashlib.sha256).hexdigest()

if len(sys.argv) < 4:
    print('Usage: python signed_img2img_pc.py "C:\\path\\image.jpg" <category> <shot>')
    sys.exit(2)

image_path = sys.argv[1]
category = sys.argv[2]
shot = sys.argv[3] if len(sys.argv) > 3 else "wide"
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 601

with open(image_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("ascii")

payload = {
  "image": img_b64,   # IMPORTANT: key name is "image"
  "prompt": "Apply realistic exterior materials to this design: stone veneer base, warm stucco walls, wood accents, charcoal standing-seam metal roof, black aluminum window frames with realistic reflections, natural daylight, clean shadows, photoreal architectural visualization, no CGI look",
  "negative_prompt": "cartoon, illustration, anime, watercolor, lowres, blurry, warped geometry",
  "category": category,
  "shot": shot,
  "seed": seed,
  "override": {"lycoris_multiplier": 0.05, "geo_multiplier": 0.01},
  "upscale_2x": False
}

body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
ts = str(int(time.time()))
nonce = uuid.uuid4().hex
sig = sign(ts, nonce, body)

headers = {
  "Content-Type": "application/json",
  "X-RENDEREXPO-TIMESTAMP": ts,
  "X-RENDEREXPO-NONCE": nonce,
  "X-RENDEREXPO-SIGNATURE": sig
}

r = requests.post(BASE + "/api/sd35/render-from-image", data=body, headers=headers, timeout=900)
print("STATUS", r.status_code)
print(r.text)
